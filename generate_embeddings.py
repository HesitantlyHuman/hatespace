import numpy as np
import pandas as pd

import sys

import os

import torch
from torch.utils.data import DataLoader

from models import VAE, InterpolatedLinearLayers

from datasets import IronMarch

from tqdm import tqdm

bert_embedding_size = 768

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

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print('Using {}'.format(device))

    #Metadata
    model_version = '1.0.0'
    model_path = sys.argv[1]
    embeddings_save_location = 'saved_embeddings'
    embeddings_save_path = os.path.join(embeddings_save_location, *model_path.split('\\')[1:-1])

    #Hyperparameters
    batch_size = 1024

    config = {
        'dataset' : {
            'context' : False
        },
        'model' : {
            'decoder' : {
                'bias' : 0.12467699766661923,
                'depth' : 7
            },
            'encoder' : {
                'bias' : 7.043664750687762,
                'depth': 4
            },
            'latent_dims' : 16,
            'softmax' : True
        }
    }

    print('Loading dataset from cache...')
    dataset = IronMarch(
            dataroot = 'iron_march_201911',
            preprocessing_function = None,
            side_information = None,
            use_context = config['dataset']['context'],
            cache = True,
            cache_location = 'datasets\caches'
        )
    loader = DataLoader(dataset, batch_size = batch_size)

    print('Loading VAE model...')
    model = get_model(config, device)
    state_dict, _ = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()

    embeddings = []

    progress_bar = tqdm(loader, position = 0, leave = True)
    with torch.no_grad():
        for batch_num, batch in enumerate(progress_bar):
            posts = batch['posts'].to(device).float()
            ids = batch['ids']
            
            mean, logvar = model.encode(posts)
            eps = model.reparameterize(mean, logvar)

            eps = eps.to('cpu').detach().numpy()

            for id, embedding in zip(ids, eps):
                embeddings.append(
                    {
                        'id' : id,
                        'embedding' : list(embedding)
                    }
                )

    if not os.path.exists(embeddings_save_path):
        os.makedirs(embeddings_save_path)

    embeddings_dataframe = pd.DataFrame(embeddings)
    embeddings_dataframe.to_csv(embeddings_save_path + '\embeddings.csv', index = False)