from typing import Callable, List
import torch
import pandas as pd
from torch import tensor
from torch.utils import data
from utils import chunk_list
import numpy as np
import csv
import html2text

from torch.utils.data import Dataset
import os

class IronMarch(Dataset):
    FILE_NAMES = {
        'direct_messages' : 'core_message_posts.csv',
        'forum_posts' : 'core_search_index.csv'
    }
    def __init__(self, dataroot) -> None:
        super().__init__()
        self.data = []
        self.dataroot = dataroot
        self._load_data()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _load_data(self):
        dm_file_path = os.path.join(self.dataroot, self.FILE_NAMES['direct_messages'])
        dm_csv = pd.read_csv(dm_file_path)[['msg_id', 'msg_post']].rename({'msg_id' : 'id', 'msg_post' : 'data'}, axis = 'columns')
        dm_csv['id'] = dm_csv['id'].map(lambda id : 'direct_message-' + str(id))
        dm_csv['label'] = 'direct_message'
        dm_csv['data'] = dm_csv['data'].map(lambda data: html2text.html2text(str(data)))

        posts_file_path = os.path.join(self.dataroot, self.FILE_NAMES['forum_posts'])
        posts_csv = pd.read_csv(posts_file_path)[['index_id', 'index_content']].rename({'index_id' : 'id', 'index_content' : 'data'}, axis = 'columns')
        posts_csv['id'] = posts_csv['id'].map(lambda id : 'forum_post-' + str(id))
        posts_csv['label'] = 'forum_post'
        posts_csv['data'] = posts_csv['data'].map(lambda data: html2text.html2text(str(data)))

        self.data = dm_csv.to_dict(orient = 'records') + posts_csv.to_dict(orient = 'records')

# Assumes that the dataset is using the {'id', 'data', 'label'} format
class BertPreprocess():
    def __init__(
        self,
        dataset: Dataset,
        bert: torch.nn.Module,
        tokenizer: Callable,
        device = 'cpu',
        batch_size = 1024,
        **kwargs
    ):
        self.bert = bert.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = dataset
        self.batch_size = batch_size
        self.data = self._preprocess(self.dataset)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _preprocess(self, dataset: Dataset) -> List[dict]:
        unprocessed_data = []
        for item in dataset:
            unprocessed_data.append(item['data'])
        for chunk in chunk_list(unprocessed_data, self.batch_size):
            tokens = self._tokenize(chunk)
            bert_embeddings = self.bert(**tokens)
        
    def _tokenize(self, strings_to_tokenize):
        tokens = self.tokenizer(
            strings_to_tokenize,
            padding = 'max_length',
            max_length = self.config['max_length'],
            return_tensors = 'pt',
            verbose = False
        )
        return {key : value[0][len(value[0]) - self.config['max_length']:] for key, value in tokens.items()}

if __name__ == '__main__':
    dataset = IronMarch('data/iron-march')
    print(dataset[0])