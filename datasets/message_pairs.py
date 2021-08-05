import torch
import pandas as pd

from torch.utils.data import Dataset

class IronMarchMessagePairs(Dataset):
    def __init__(self, dataroot,
                probability_of_relation = 0.5,
                tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased'),
                encoder = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased'),
                maximum_sequence_length = 512):

        self.messages_dataframe = pd.read_csv(dataroot)
        self.probability_of_relation = probability_of_relation
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.maximum_sequence_length = maximum_sequence_length

    def __getitem__(self, index):
        #Read from the encoded csv
        selected_message = self.messages_dataframe.loc[index]
        pair_message = None
        are_related = 0

        if torch.rand(()) > self.probability_of_relation:
            pair_message_post = self._get_random_related_message(selected_message['msg_id'], selected_message['msg_topic_id'])
            are_related = 1

        if pair_message is None:
            pair_message_post = self._get_random_unrelated_message(selected_message['msg_id'])

        encodings = torch.cat([
            self.encode_string(selected_message['msg_post']),
            self.encode_string(pair_message_post)
        ])

        return {
            'messages' : encodings,
            'related' : are_related
        }

    def _convert_token_dict_to_tensor(dictionary):
        return torch.tensor([dictionary['input_ids'], dictionary['token_type_ids'], dictionary['attention_mask']])

    def _get_random_unrelated_message(self, msg_topic_id):
        candidate_messages = self.messages_dataframe[self.messages_dataframe['msg_topic_id'] != msg_topic_id]
        return candidate_messages.sample().to_dict(orient = 'list')['msg_post'][0]

    def _get_random_related_message(self, index, msg_topic_id):
        #All messages with msg_topic_id not including the message of given index
        candidate_messages = self.messages_dataframe[(self.messages_dataframe['msg_topic_id'] == msg_topic_id) & (self.messages_dataframe['msg_id'] != index)]
        if len(candidate_messages) > 0:
            return candidate_messages.sample().to_dict(orient = 'list')['msg_post'][0]
        else:
            return None

    def __len__(self):
        return len(self.messages_dataframe)

    def encode_string(self, string):
        tokens = self.get_padded_tensor_tokens(string)
        with torch.no_grad():
            return self.encoder(**tokens)['last_hidden_state']

    def get_padded_tensor_tokens(self, string):
        return {key: IronMarchMessagePairs.get_padded_tensor(value, self.maximum_sequence_length) for key, value in self.tokenizer(string).items()}

    def get_padded_tensor(input_value, desired_length):
        tensor = torch.tensor(input_value)
        padding = desired_length - tensor.shape[-1]
        return torch.nn.functional.pad(torch.unsqueeze(tensor, dim = 0), pad = (0, padding))

    def create_encoded_csv(csv_path, new_csv_path):
        pass