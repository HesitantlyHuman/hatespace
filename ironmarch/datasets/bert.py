from typing import List, Callable
from ironmarch.datasets.utils import chunk_list
from ironmarch.dataset.ironmarch import IronmarchDataset
import torch
import emoji
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

# TODO implement the mentions and urls tokenization as described on https://huggingface.co/vinai/bertweet-base
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
            #Take the average or the cls token (make this configurable)
        
    def _tokenize(self, strings_to_tokenize):
        tokens = self.tokenizer(
            strings_to_tokenize,
            padding = 'max_length',
            max_length = self.config['max_length'],
            return_tensors = 'pt',
            verbose = False
        )
        return {key : value[0][len(value[0]) - self.config['max_length']:] for key, value in tokens.items()}

    def _tokenize_emojis(dirty_string: str) -> str:
        return emoji.demojize(dirty_string)

    def _tokenize_mentions(dirty_string: str) -> str:
        return dirty_string

    def _tokenize_urls(dirty_string: str) -> str:
        return dirty_string

if __name__ == '__main__':
    dataset = IronMarch('data/iron-march')
    bertweet = AutoModel.from_pretrained('vinai/bertweet-base')
    tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', use_fast = False)
    preprocessed_dataset = BertPreprocess(
        dataset = dataset,
        bert = bertweet,
        tokenizer = tokenizer,
        device = 'cpu'
    )
    print(preprocessed_dataset[0].shape)