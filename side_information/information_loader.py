import pandas as pd
import numpy as np
from ast import literal_eval

class SideLoader():
    def __init__(self, data_paths):
        assert type(data_paths) is list, f'SideLoader expected a list of data_paths, but received {type(data_paths)}'
        self.data = self._load_data(data_paths)

    def _load_data(self, data_paths):
        data = {}

        for path in data_paths:
            new_dataframe = pd.read_csv(path)
            for idx, data_item in new_dataframe.iterrows():
                data_id = data_item['msg_id']
                data_value = np.array(literal_eval(data_item['data'])).astype('float')
                try:
                    data[data_id] = np.concatenate(data[data_id], data_value)
                except KeyError:
                    data[data_id] = data_value

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]