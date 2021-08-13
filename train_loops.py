from ray.tune import schedulers
from ray.tune import suggest
from ray.tune.progress_reporter import CLIReporter
import torch
from torch.utils.data import DataLoader

from datasets import IronMarch, BertPreprocess
from models import VAE, InterpolatedLinearLayers
from side_information import SideLoader

from torch.optim import Adam

import os

import geomloss

from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler, ASHAScheduler
from ray.tune.suggest.optuna import OptunaSearch

from training.search_space import config

from filelock import FileLock

bert_embedding_size = 768
epsilon = 1e-20
os.environ['TUNE_MAX_LEN_IDENTIFIER'] = '50'
#os.environ["TUNE_PLACEMENT_GROUP_AUTO_DISABLED"] = "1"

def get_model(config, device):
    if config['dataset']['context']:
        input_dim = bert_embedding_size * 2
    else:
        input_dim = bert_embedding_size
    encoder = InterpolatedLinearLayers(
        input_dim, 
        config['model']['latent_dims'] * 2, 
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

def train_sinkhorn_vae(config, checkpoint_dir = None):
    device = config['device']
    if device == 'cuda:0':
        if not torch.cuda.is_available():
            device = 'cpu'
            print('Could not find GPU!')
            print(f'Using {device}', flush = True)
    
    train_loader, val_loader, class_proportions = get_datasets(config, device)
    model = get_model(config, device)

    optimizer = Adam(
        model.parameters(), 
        lr = config['adam']['learning_rate'], 
        betas = (config['adam']['betas']['zero'], config['adam']['betas']['one'])
    )

    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    sinkhorn_loss_fn = geomloss.SamplesLoss(
        loss = config['losses']['distribution']['type'], 
        p = config['losses']['distribution']['p'], 
        blur = config['losses']['distribution']['blur'])
    reconstruction_loss_fn = torch.nn.MSELoss()
    class_weights = (1 / class_proportions) * config['losses']['class']['bias']
    class_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = class_weights.to(device))

    dirichlet_distribution = torch.distributions.dirichlet.Dirichlet(torch.tensor([config['losses']['distribution']['alpha'] for i in range(config['model']['latent_dims'])]))

    for epoch_num in range(config['training']['max_epochs']):
        for batch_num, batch in enumerate(train_loader):
            features = batch['features'].to(device).float()
            posts = batch['posts'].to(device).float()

            noise = torch.normal(
                mean = 0.0, 
                std = config['latent_space']['noise']['std'], 
                size = posts.shape).to(device)
            augmented_posts = posts + noise

            optimizer.zero_grad()

            mean, logvar = model.encode(augmented_posts)
            eps = model.reparameterize(mean, logvar)
            logits = model.decoder(eps)
            feature_predictions = model.feature_head(eps)

            sampled_dirichlet = dirichlet_distribution.sample([config['training']['batch_size']]).to(device)
            distribution_loss = sinkhorn_loss_fn(eps, sampled_dirichlet) 

            reconstruction_loss = reconstruction_loss_fn(posts, logits)

            class_loss = class_loss_fn(feature_predictions, features)

            loss = class_loss * config['losses']['class']['weight'] + distribution_loss * config['losses']['distribution']['weight'] + reconstruction_loss * config['losses']['reconstruction']['weight']

            loss.backward()
            optimizer.step()

        loss_accum = 0
        class_accum = 0
        dist_accum = 0
        recon_accum = 0
        
        tp_accum = 0
        fp_accum = 0
        tn_accum = 0
        fn_accum = 0

        with torch.no_grad():
            for batch_num, batch in enumerate(val_loader):
                features = batch['features'].to(device).float()
                posts = batch['posts'].to(device).float()

                mean, logvar = model.encode(posts)
                eps = model.reparameterize(mean, logvar)
                logits = model.decoder(eps)
                feature_predictions = model.feature_head(eps)

                sampled_dirichlet = dirichlet_distribution.sample([config['training']['batch_size']]).to(device)
                distribution_loss = sinkhorn_loss_fn(eps, sampled_dirichlet) 

                reconstruction_loss = reconstruction_loss_fn(posts, logits)

                class_loss = class_loss_fn(feature_predictions, features)

                loss = class_loss * config['losses']['class']['weight'] + distribution_loss * config['losses']['distribution']['weight'] + reconstruction_loss * config['losses']['reconstruction']['weight']

                loss_accum += loss.detach().item()
                class_accum += class_loss.detach().item()
                dist_accum += distribution_loss.detach().item()
                recon_accum += reconstruction_loss.detach().item()

                binary_predictions = torch.ge(feature_predictions, config['losses']['class']['threshold'])
                tp_accum += ((binary_predictions == 1.0) & (features == 1.0)).detach().sum().item()
                fp_accum += ((binary_predictions == 1.0) & (features == 0.0)).detach().sum().item()
                tn_accum += ((binary_predictions == 0.0) & (features == 0.0)).detach().sum().item()
                fn_accum += ((binary_predictions == 0.0) & (features == 1.0)).detach().sum().item()

        with tune.checkpoint_dir(epoch_num) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, 'checkpoint')
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        
        yield {
            'loss' : loss_accum / (batch_num + 1),
            'class_loss' : class_accum / (batch_num + 1),
            'dist_loss' : dist_accum / (batch_num + 1),
            'recon_loss' : recon_accum / (batch_num + 1),
            'precision' : tp_accum / (tp_accum + fp_accum + epsilon),
            'recall' : tp_accum / (tp_accum + fn_accum + epsilon),
            'accuracy' : (tp_accum + tn_accum) / (tp_accum + fp_accum + tn_accum + fn_accum + epsilon),
            'specificity' : tn_accum / (tn_accum + fp_accum + epsilon),
            'f1_score' : tp_accum / (tp_accum + (0.5 * (fp_accum + fn_accum)) + epsilon),
            'epoch' : epoch_num
        }

def get_datasets(config, device):
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
    
def run_optuna_tune():
    config['dataset']['root_directory'] = os.getcwd()
    algorithm = OptunaSearch()
    algorithm = ConcurrencyLimiter(
        algorithm,
        max_concurrent = 6
    )
    scheduler = ASHAScheduler(
        max_t = 100,
        grace_period = 5,
        reduction_factor = 2
    )
    reporter = CLIReporter(
        metric_columns = ['loss', 'precision', 'recall', 'accuracy', 'f1_score', 'epoch']
    )
    analysis = tune.run(
        train_sinkhorn_vae,
        metric = 'loss',
        mode = 'min',
        resources_per_trial = {
            'cpu' : 4,
            'gpu' : 0
        },
        search_alg = algorithm,
        progress_reporter = reporter,
        scheduler = scheduler,
        num_samples = 100,
        config = config,
        local_dir = 'results'
    )

if __name__ == "__main__":
    run_optuna_tune()
