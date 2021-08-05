import torch
import pandas as pd
from torch import tensor
from torch.utils import data
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from ast import literal_eval

import os

import pickle

class BertPreprocess():
    def __init__(self, bert, tokenizer, device = 'cpu', **kwargs):
        self.bert = bert.to(device)
        self.tokenizer = tokenizer

        self.device = device

        self.config = {
            'max_length' : 512
        }
        self.config.update(kwargs)
    
    def __call__(self, *args):
        if len(args) == 0:
            raise ValueError('BertPreprocess requires at least one argument for __call__')
        elif len(args) == 1:
            tokens_dict = self._tokenize_single(args[0])
            tokens_dict = {key: torch.unsqueeze(value.to(self.device), dim = 0) for key, value in tokens_dict.items()}
            return torch.squeeze(torch.mean(self.bert(**tokens_dict)['last_hidden_state'], dim = 1).detach().to('cpu'))
        elif len(args) == 2:
            tokens_dict = self._tokenize_context_pair(args[0], args[1])
            tokens_dict = {key: torch.unsqueeze(value.to(self.device), dim = 0) for key, value in tokens_dict.items()}
            return torch.squeeze(torch.mean(self.bert(**tokens_dict)['last_hidden_state'], dim = 1).detach().to('cpu'))
        else:
            raise ValueError(f'{len(args)} is an invalid number of arguments for BertPreprocess __call__')
        
    def _tokenize_single(self, string_to_tokenize):
        tokens = self.tokenizer(
            string_to_tokenize,
            padding = 'max_length',
            max_length = self.config['max_length'],
            return_tensors = 'pt',
            verbose = False
        )

        return {key : value[0][len(value[0]) - self.config['max_length']:] for key, value in tokens.items()}

    def _tokenize_context_pair(self, context_string, main_string):
        tokens = self.tokenizer(
            context_string,
            main_string,
            padding = 'max_length',
            max_length = self.config['max_length'],
            return_tensors = 'pt',
            verbose = False
        )

        return {key : value[0][len(value[0]) - self.config['max_length']:] for key, value in tokens.items()}

class IronMarch(Dataset):
    def __init__(self, dataroot, preprocessing_function, **kwargs):
        super(IronMarch, self).__init__()

        self.preprocessing_function = preprocessing_function
        self.dataroot = dataroot
        self.data = []

        self.config = {
            'use_context' : True,
            'min_poster_posts' : None,
            'msg_topic_ids' : None,
            'msg_poster_ids' : None,
            'msg_ids' : None,
            'side_information' : None,
            'load_from_cache' : False,
            'cache_location' : None
        }

        self.config.update(kwargs)
        if not self.config['load_from_cache']:
            self._preprocess_data()
            if not self.config['cache_location'] is None:
                self.save(self.config['cache_location'])
        else:
            self._load_preprocessed_data_from_file(self.config['cache_location'])

    def _preprocess_data(self):
        direct_messages = self._load_posts_csv_file_as_dictionary('core_message_posts.csv', 'msg_topic_id')
        forum_posts = self._load_posts_csv_file_as_dictionary('core_search_index.csv', 'index_date_created')

        self._format_and_append_data_from_dictionary(direct_messages, 'direct_messages')
        self._format_and_append_data_from_dictionary(forum_posts, 'forum_posts')

    def _format_and_append_data_from_dictionary(self, dictionary, dictionary_name):
        previous_message = None
        for item in dictionary:
            if not previous_message is None:
                if previous_message.get('msg_topic_id', None) != item.get('msg_topic_id', None):
                    previous_message = None
            item_id = self._get_data_item_id(item, dictionary_name)
            self.data.append(self._format_example(previous_message, item, item_id))
            previous_message = item

    def _load_posts_csv_file_as_dictionary(self, relative_file_path, column_to_sort):
        data = pd.read_csv(os.path.join(self.dataroot, relative_file_path))
        data = data.sort_values(by = [column_to_sort], ascending = [True])
        return data.to_dict(orient = 'records')

    def _get_data_item_id(self, item, dictionary_name):
        id = item.get('msg_id', None)
        if id is None:
            id = item['index_id']
        return dictionary_name + '_' + str(id)

    def _format_example(self, context_data_item, main_data_item, id):
        posts = self._preprocess_example(context_data_item, main_data_item)
        features = self._get_example_features(id)
        
        return {
            'ids' : id,
            'posts' : posts,
            'features' : features
        }

    def _preprocess_example(self, context_data_item, main_data_item):
        main_content = str(self._get_post_content(main_data_item))

        if self.config['use_context']:
            if context_data_item is None:
                processed_context = self.preprocessing_function('')
            else:
                context_content = str(self._get_post_content(context_data_item))
                processed_context = self.preprocessing_function(context_content)
            return torch.cat([
                processed_context,
                self.preprocessing_function(main_content)
            ])
        else:
            return self.preprocessing_function(main_content)

    def _get_post_content(self, post):
        content = post.get('msg_post', None)
        if content is None:
            content = post['index_content']
        return content

    def _get_example_features(self, msg_id):
        if self.config['side_information'] is None:
            return []
        else:
            try:
                features = self.config['side_information'][msg_id]
                return features
            except KeyError:
                raise KeyError(f'msg_post_key {msg_id} was not found in the provided feature set')

    def _load_preprocessed_data_from_file(self, location):
        with open(location, 'rb') as pickle_file:
            self.data = pickle.load(pickle_file)

    def split_validation(self, validation_split = 0.1, shuffle = True, random_seed = None):
        idx = list(range(len(self)))
        split = int(np.floor(validation_split * len(self)))

        if not random_seed is None:
            np.random.seed(random_seed)
        if shuffle:
            np.random.shuffle(idx)

        train_idx, val_idx = idx[split:], idx[:split]

        return SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx)

    def save(self, location):
        with open(location, 'wb') as pickle_file:
            pickle.dump(self.data, pickle_file)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)