from filelock import FileLock
from datasets import IronMarch
import torch
import geomloss
import os
from ray import tune
from torch.utils.data.dataloader import DataLoader

from models import VAE, InterpolatedLinearLayers

BERT_embedding_size = 768
epsilon = 1e-20

class KeepBest(tune.Trainable):
    def __init__(self, pytorch_trainable, metric = 'loss', *args, **kwargs):
        super(KeepBest, self).__init__(*args, **kwargs)
        self.trainable = pytorch_trainable
        self.best = torch.tensor(float('inf'))
        self.metric = metric

    def setup(self, config):
        return self.trainable.setup(config)

    def step(self):
        metrics = self.trainable.step()
        if metrics[self.metric] < self.best:
            metrics.update({
                'should_checkpoint' : True
            })
        return metrics

    def save_checkpoint(self, checkpoint_dir):
        return self.trainable.save_checkpoint(checkpoint_dir)

    def load_checkpoint(self, checkpoint):
        return self.trainable.load_checkpoint(checkpoint)

class VAEBERT(tune.Trainable):
    def setup(self, config):
        self.config = config
        self.device = self._get_cuda_device(
            config = self.config
        )
        self.model = self._get_model(
            config = self.config,
            device = self.device
        )
        self.optimizer = self._get_optimizer(
            config = self.config,
            parameters = self.model.parameters()
        )
        self.train_data, self.validation_data, class_proportions = self._get_datasets(
            self.config
        )
        
        self.reconstruction_loss_fn = torch.nn.MSELoss()
        
        self.distribution_loss_fn = geomloss.SamplesLoss(
            loss = config['losses']['distribution']['type'], 
            p = config['losses']['distribution']['p'], 
            blur = config['losses']['distribution']['blur'])
        self.distribution_sampler = torch.distributions.dirichlet.Dirichlet(torch.tensor([config['losses']['distribution']['alpha'] for i in range(config['model']['latent_dims'])]))
        
        class_weights = (1 / class_proportions) * config['losses']['class']['bias']
        self.class_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = class_weights.to(self.device))

    def step(self):
        self._train(
            config = self.config
        )
        metrics = self._validate(
            config = self.config
        )

        return metrics

    def _train(self, config):
        for batch in self.train_data:
            self.optimizer.zero_grad()

            loss = self._single_batch(batch, config)['loss']

            loss.backward()
            self.optimizer.step()

    def _validate(self, config):
        loss_accum = 0
        class_accum = 0
        dist_accum = 0
        recon_accum = 0
        
        tp_accum = 0
        fp_accum = 0
        tn_accum = 0
        fn_accum = 0

        with torch.no_grad():
            for batch_num, batch in enumerate(self.validation_data):
                batch_output = self._single_batch(batch, config, noise = False)

                loss_accum += batch_output['loss'].detach().item()
                class_accum += batch_output['class'].detach().item()
                dist_accum += batch_output['distribution'].detach().item()
                recon_accum += batch_output['reconstruction'].detach().item()

                binary_predictions = torch.ge(batch_output['feature_predictions'], config['losses']['class']['threshold'])
                features = batch['features'].to(self.device).float()
                tp_accum += ((binary_predictions == 1.0) & (features == 1.0)).detach().sum().item()
                fp_accum += ((binary_predictions == 1.0) & (features == 0.0)).detach().sum().item()
                tn_accum += ((binary_predictions == 0.0) & (features == 0.0)).detach().sum().item()
                fn_accum += ((binary_predictions == 0.0) & (features == 1.0)).detach().sum().item()
        
        return {
            'loss' : loss_accum / (batch_num + 1),
            'class_loss' : class_accum / (batch_num + 1),
            'dist_loss' : dist_accum / (batch_num + 1),
            'recon_loss' : recon_accum / (batch_num + 1),
            'precision' : tp_accum / (tp_accum + fp_accum + epsilon),
            'recall' : tp_accum / (tp_accum + fn_accum + epsilon),
            'accuracy' : (tp_accum + tn_accum) / (tp_accum + fp_accum + tn_accum + fn_accum + epsilon),
            'specificity' : tn_accum / (tn_accum + fp_accum + epsilon),
            'f1_score' : tp_accum / (tp_accum + (0.5 * (fp_accum + fn_accum)) + epsilon)
        }


    def _single_batch(self, batch, config, noise = True):
        features = batch['features'].to(self.device).float()
        posts = batch['posts'].to(self.device).float()

        if noise:
            noise = torch.normal(
                mean = 0.0, 
                std = config['latent_space']['noise']['std'], 
                size = posts.shape).to(self.device)
        else:
            noise = torch.zeros(
                size = posts.shape
            )
        augmented_posts = posts + noise

        mean, logvar = self.model.encode(augmented_posts)
        eps = self.model.reparameterize(mean, logvar)
        logits = self.model.decoder(eps)
        feature_predictions = self.model.feature_head(eps)

        sampled_dirichlet = self.distribution_sampler.sample([config['training']['batch_size']]).to(self.device)
        distribution_loss = self.distribution_loss_fn(eps, sampled_dirichlet) 

        reconstruction_loss = self.reconstruction_loss_fn(posts, logits)

        class_loss = self.class_loss_fn(feature_predictions, features)

        loss = class_loss * config['losses']['class']['weight'] + distribution_loss * config['losses']['distribution']['weight'] + reconstruction_loss * config['losses']['reconstruction']['weight']

        return {
            'loss' : loss,
            'class' : class_loss,
            'distribution' : distribution_loss,
            'reconstruction' : reconstruction_loss,
            'feature_predictions' : feature_predictions
        }

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), checkpoint_path)

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        model_state, optimizer_state = torch.load(checkpoint_path)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

    def _get_cuda_device(self, config):
        device = config['device']
        if device == 'cuda:0':
            if not torch.cuda.is_available():
                device = 'cpu'
                print('Could not find GPU!')
                print(f'Using {device}', flush = True)
        return device

    def _get_datasets(self, config):
        with FileLock(config['dataset']['directory'] + '.lock'):
            dataset = IronMarch(
                dataroot = os.path.join(config['dataset']['root_directory'], config['dataset']['directory']),
                preprocessing_function = None,
                side_information = None,
                use_context = config['dataset']['context'],
                cache = True,
                cache_location = os.path.join(config['dataset']['root_directory'], 'datasets\caches')
            )

            train_sampler, val_sampler = dataset.split_validation(validation_split = 0.1)
            train_loader = DataLoader(dataset, batch_size = config['training']['batch_size'], sampler = train_sampler)
            val_loader = DataLoader(dataset, batch_size = config['training']['batch_size'], sampler = val_sampler)

        class_proportions = dataset.get_class_proportions()

        return train_loader, val_loader, class_proportions

    def _get_model(self, config, device):
        if config['dataset']['context']:
            input_dim = BERT_embedding_size * 2
        else:
            input_dim = BERT_embedding_size
        encoder = InterpolatedLinearLayers(
            input_size = input_dim, 
            output_size = config['model']['latent_dims'] * 2, 
            num_layers = config['model']['encoder']['depth'], 
            bias = config['model']['encoder']['bias']
        )
        decoder = InterpolatedLinearLayers(
            config['model']['latent_dims'], 
            input_dim, 
            num_layers = config['model']['decoder']['depth'], 
            bias = config['model']['decoder']['bias']
        )
        feature_head = torch.nn.Linear(config['model']['latent_dims'], 7)
        model = VAE(
            encoder = encoder,
            decoder = decoder,
            feature_head = feature_head,
            use_softmax = config['model']['softmax']
        )
        model.to(device)
        return model

    def _get_optimizer(self, config, parameters):
        optimizer = torch.optim.Adam(
            parameters, 
            lr = config['adam']['learning_rate'], 
            betas = (config['adam']['betas']['zero'], config['adam']['betas']['one'])
        )
        return optimizer