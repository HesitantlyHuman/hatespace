import numpy as np
import pandas as pd

import sys

import os

import torch
from torch.utils.data import DataLoader

from datasets import IronMarch

from tqdm import tqdm


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
    embeddings_save_file = os.path.join(embeddings_save_location, *model_path.split('\\')[1:]) + '.csv'

    #Hyperparameters
    batch_size = 32

    print('Loading dataset from cache...')
    dataset = IronMarch(
        dataroot = 'iron_march_201911\core_message_posts.csv',
        preprocessing_function = None,
        side_information = None,
        use_context = True,
        load_from_cache = True,
        cache_location = 'datasets\caches\cache.pickle')
    loader = DataLoader(dataset, batch_size = batch_size)

    print('Loading VAE model...')
    model = torch.load(model_path)
    model.eval().to(device)

    embeddings = []

    progress_bar = tqdm(loader, position = 0, leave = True)
    with torch.no_grad():
        for batch_num, batch in enumerate(progress_bar):
            posts = batch['posts'].to(device).float()
            ids = batch['ids'].numpy()
            
            mean, logvar = model.encode(posts)
            eps = model.reparameterize(mean, logvar)

            eps = eps.to('cpu').detach().numpy()

            for id, embedding in zip(ids, eps):
                embeddings.append(
                    {
                        'id' : id,
                        'embedding' : embedding
                    }
                )

    embeddings_dataframe = pd.DataFrame(embeddings)
    embeddings_dataframe.to_csv(embeddings_save_file, index = False)