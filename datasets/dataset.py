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

        #Use a hash function to cache the dataset
        self.config = {
            'use_context' : True,
            'side_information' : None,
            'cache' : False,
            'cache_location' : 'datasets/caches'
        }
        self.config.update(kwargs)

        if not self.config['cache']:
            self._preprocess_data()
        else:
            cache_name = ''.join([str(key).replace(' ', '') + str(value).replace(' ', '') if not key == 'cache_location' and not key == 'side_information' else '' for key, value in self.config.items()]) + '.pickle'
            self.cache_location = os.path.join(self.config['cache_location'], cache_name)

            try:
                self._load_from_cache(self.cache_location)
            except FileNotFoundError:
                self._preprocess_data()
                if not os.path.exists(self.config['cache_location']):
                    os.mkdir(self.config['cache_location'])
                self.save(self.cache_location)

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
        try:
            features = []
            for loader in self.config['side_information']:
                features.append(loader[msg_id])
            return features
        except KeyError:
            raise KeyError(f'msg_post_key {msg_id} was not found in the provided feature set')

    def _load_from_cache(self, location):
        with open(location, 'rb') as pickle_file:
            self.data = pickle.load(pickle_file)

    def get_class_proportions(self):
        class_totals = torch.zeros(size = (len(self.data[0]['features']),))
        for item in self.data:
            class_totals += item['features']
        return class_totals / len(self)

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