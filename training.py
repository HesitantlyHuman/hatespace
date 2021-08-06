import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from datasets import IronMarch, BertPreprocess
from models import VAE
from side_information import SideLoader

from torch.optim import Adam

from tqdm import tqdm

import geomloss

import side_information

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print('Using {}'.format(device))

    #Metadata
    model_version = '1.2.0'

    #Hyperparameters
    epochs = 50
    batch_size = 32
    latent_dim_size = 16
    use_context = True
    #Distribution
    sinkhorn_weight = 1
    concentration_alpha = 1.0
    #Reconstruction
    gaussian_std = 0.1
    reconstruction_weight = 10
    softmax = True
    #Class
    class_weight = 0.05
    use_features = True
    feature_threshold = 0.5
    bias_weight_strength = 1.5

    #Constants
    bert_embedding_size = 768
    epsilon = 1e-20
    
    print('Gathering side information...')
    side_information_file_paths = [
        'side_information\hate_words\processed_side_information.csv'
    ]
    side_information_loader = SideLoader(side_information_file_paths)

    print('Loading BERT and pre-embedding-...')
    bert_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'roberta-base')
    bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'roberta-base')
    preprocessing_fn = BertPreprocess(bert = bert, tokenizer = bert_tokenizer, device = device)
    dataset = IronMarch(
        dataroot = 'iron_march_201911',
        preprocessing_function = preprocessing_fn,
        side_information = side_information_loader,
        use_context = use_context,
        load_from_cache = True,
        cache_location = 'datasets\caches\cache.pickle'
    )
    del bert, bert_tokenizer

    print('Splitting dataset...')
    train_sampler, val_sampler = dataset.split_validation(validation_split = 0.1)
    train_loader = DataLoader(dataset, batch_size = batch_size, sampler = train_sampler)
    val_loader = DataLoader(dataset, batch_size = batch_size, sampler = val_sampler)

    print('Creating VAE model...')
    model = VAE(latent_dim = latent_dim_size, input_dim = bert_embedding_size * 2, feature_dim = 7, use_softmax = softmax)
    model.to(device)

    print('Initializing optimizer...')
    params = model.parameters()
    optimizer = Adam(params, lr = 0.0001, betas = (0.95, 0.9999))
    
    print('Calculating class biases...')
    weight_tensor = torch.tensor([0 for i in range(7)])
    for item in dataset:
        weight_tensor += item['features']
    weight_tensor = 1 / (weight_tensor / len(dataset)) * bias_weight_strength
    weight_tensor = weight_tensor.to(device)
    print(weight_tensor)

    print('Creating loss functions...')
    #Loss Functions
    sinkhorn_loss_fn = geomloss.SamplesLoss(loss = 'sinkhorn', p = 2, blur = 0.05)
    reconstruction_loss_fn = torch.nn.MSELoss()
    class_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = weight_tensor)

    #Distribution sampler
    dirichlet_distribution = torch.distributions.dirichlet.Dirichlet(torch.tensor([concentration_alpha for i in range(latent_dim_size)]))

    #Training loop
    for epoch in range(epochs):

        loss_accum = 0
        class_accum = 0
        dist_accum = 0
        recon_accum = 0

        tp_accum = 0
        fp_accum = 0
        tn_accum = 0
        fn_accum = 0

        print(f'---Training {epoch}---')
        progress_bar = tqdm(train_loader, position = 0, leave = True)
        for batch_num, batch in enumerate(progress_bar):
            features = batch['features'].to(device).float()
            posts = batch['posts'].to(device).float()

            augmented_posts = posts + torch.normal(mean = 0.0, std = gaussian_std, size = posts.shape).to(device)

            optimizer.zero_grad()

            mean, logvar = model.encode(augmented_posts)
            eps = model.reparameterize(mean, logvar)
            logits = model.decoder(eps)
            feature_predictions = model.feature_head(eps)

            sampled_dirichlet = dirichlet_distribution.sample([batch_size]).to(device)
            distribution_loss = sinkhorn_loss_fn(eps, sampled_dirichlet) 

            reconstruction_loss = reconstruction_loss_fn(posts, logits)

            class_loss = class_loss_fn(feature_predictions, features)

            loss = class_loss * class_weight + distribution_loss * sinkhorn_weight + reconstruction_loss * reconstruction_weight

            loss.backward()
            optimizer.step()

            loss_accum += loss.detach().item()
            class_accum += class_loss.detach().item()
            dist_accum += distribution_loss.detach().item()
            recon_accum += reconstruction_loss.detach().item()
            
            binary_predictions = torch.ge(feature_predictions, feature_threshold)
            tp_accum += ((binary_predictions == 1.0) & (features == 1.0)).detach().sum().item()
            fp_accum += ((binary_predictions == 1.0) & (features == 0.0)).detach().sum().item()
            tn_accum += ((binary_predictions == 0.0) & (features == 0.0)).detach().sum().item()
            fn_accum += ((binary_predictions == 0.0) & (features == 1.0)).detach().sum().item()

            accuracy = (tp_accum + tn_accum) / (tp_accum + fp_accum + tn_accum + fn_accum + epsilon)
            precision = tp_accum / (tp_accum + fp_accum + epsilon)
            recall = tp_accum / (tp_accum + fn_accum + epsilon)

            progress_bar.set_postfix(
                {
                    'Loss' : '{:4.3f}'.format(loss_accum / (batch_num + 1)),
                    'Class' : '{:4.3f}'.format(class_accum / (batch_num + 1)),
                    'Dist' : '{:4.3f}'.format(dist_accum / (batch_num + 1)),
                    'Recon' : '{:4.3f}'.format(recon_accum / (batch_num + 1)),
                    'Acc' : '{:4.3f}'.format(accuracy),
                    'Recall' : '{:4.3f}'.format(recall),
                    'F1' : '{:4.3f}'.format((2 * precision * recall) / (precision + recall + epsilon))
                }
            )

        loss_accum = 0
        class_accum = 0
        dist_accum = 0
        recon_accum = 0
        
        tp_accum = 0
        fp_accum = 0
        tn_accum = 0
        fn_accum = 0

        print(f'---Validation {epoch}---')
        progress_bar = tqdm(val_loader, position = 0, leave = True)
        with torch.no_grad():
            for batch_num, batch in enumerate(progress_bar):
                features = batch['features'].to(device).float()
                posts = batch['posts'].to(device).float()

                mean, logvar = model.encode(posts)
                eps = model.reparameterize(mean, logvar)
                logits = model.decoder(eps)
                feature_predictions = model.feature_head(eps)

                sampled_dirichlet = dirichlet_distribution.sample([batch_size]).to(device)
                distribution_loss = sinkhorn_loss_fn(eps, sampled_dirichlet) 

                reconstruction_loss = reconstruction_loss_fn(posts, logits)

                class_loss = class_loss_fn(feature_predictions, features)

                loss = class_loss * class_weight + distribution_loss * sinkhorn_weight + reconstruction_loss * reconstruction_weight

                loss_accum += loss.detach().item()
                class_accum += class_loss.detach().item()
                dist_accum += distribution_loss.detach().item()
                recon_accum += reconstruction_loss.detach().item()

                binary_predictions = torch.ge(feature_predictions, feature_threshold)
                tp_accum += ((binary_predictions == 1.0) & (features == 1.0)).detach().sum().item()
                fp_accum += ((binary_predictions == 1.0) & (features == 0.0)).detach().sum().item()
                tn_accum += ((binary_predictions == 0.0) & (features == 0.0)).detach().sum().item()
                fn_accum += ((binary_predictions == 0.0) & (features == 1.0)).detach().sum().item()

                accuracy = (tp_accum + tn_accum) / (tp_accum + fp_accum + tn_accum + fn_accum + epsilon)
                precision = tp_accum / (tp_accum + fp_accum + epsilon)
                recall = tp_accum / (tp_accum + fn_accum + epsilon)

                progress_bar.set_postfix(
                    {
                        'Loss' : '{:4.3f}'.format(loss_accum / (batch_num + 1)),
                        'Class' : '{:4.3f}'.format(class_accum / (batch_num + 1)),
                        'Dist' : '{:4.3f}'.format(dist_accum / (batch_num + 1)),
                        'Recon' : '{:4.3f}'.format(recon_accum / (batch_num + 1)),
                        'Acc' : '{:4.3f}'.format(accuracy),
                        'Recall' : '{:4.3f}'.format(recall),
                        'F1' : '{:4.3f}'.format((2 * precision * recall) / (precision + recall + epsilon))
                    }
                )

        torch.save(model, f'saved_models\\vae\\sinkhorn\\alpha_{concentration_alpha}_latent_{latent_dim_size}_softmax_{softmax}_version_{model_version}_features_{use_features}')