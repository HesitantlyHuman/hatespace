from typing import Callable, List
import pickle

#Add some tqdm action up in here
class SideLoader():
    '''
        Loads and saves side information by data item id.
    '''
    def __init__(self, data: dict, cache_path: str = 'side_information/data/side_information.pickle') -> None:
        self.data = data
        self.keys = list(data.keys())
        self.names = []
        self.side_information = {key: [] for key in self.data.keys()}
        
        self.cache_path = cache_path
        self.cache = self._load_cache(self.cache_path)

    def _load_cache(self, path) -> dict:
        try:
            with open(path, 'rb') as cache_file:
                cache = pickle.load(cache_file)
            assert(isinstance(cache, dict))
            return cache
        except:
            return {}

    def _write_cache(self, path) -> None:
        with open(path, 'wb') as cache_file:
            pickle.dump(self.cache, cache_file, protocol = pickle.HIGHEST_PROTOCOL)

    def add(self, name: str, function: Callable):
        if name in self.names:
            raise KeyError(f"{name} already exists")
        if name not in self.cache.keys():
            self.cache[name] = {}
        for key in self.keys:
            new_side_info = function(
                {
                    'id' : key,
                    'data' : self.data[key]
                }
            )
            side_info = self.side_information[key]
            if isinstance(new_side_info, list):
                side_info = side_info + new_side_info
            else:
                side_info.append(new_side_info)
            self.side_information[key] = side_info
            self.cache[name][key] = new_side_info
        self.names.append(name)
        self._write_cache(self.cache_path)

    def load(self, name: str):
        if self.cache is None:
            raise AttributeError('No cache found')
        if name in self.cache.keys():
            self.add(name, lambda data_dict : self.cache[name][data_dict['id']])
        else:
            raise AttributeError(f'{name} not found in cache')

    def load_all(self):
        if self.cache is None:
            raise AttributeError('No cache found')
        for name in self.cache.keys():
            try:
                self.load(name)
            except KeyError:
                pass

    def list(self):
        return self.names

    def cached(self):
        return list(self.cache.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.side_information[key]

class KeyLoader(SideLoader):
    def add(self, name: str, function: Callable):
        function = lambda data_dict: function(data_dict['id'])
        super().add(name, function)

class ModelLoader(SideLoader):
    def add(self, name: str, function: Callable):
        function = lambda data_dict: function(data_dict['data'])
        super().add(name, function)

def test_function(data_dict):
    return data_dict['data'] + ['some more']

if __name__ == '__main__':
    # Normal data loader
    data = {
        'data_item-1' : ['some data', 0]
    }
    side_information = SideLoader(data)
    side_information.add('yet-another-function', test_function)
    side_information.add('other-function', test_function)
    del(side_information)
    side_information = SideLoader(data)
    side_information.add('test-function', test_function)
    side_information.load('other-function')
    assert(side_information.list() == ['test-function', 'other-function'])
    assert(set(side_information.cached()) == set(['test-function', 'other-function', 'yet-another-function']))
    data_item = {
        'id' : 'data_item-1',
        'data' : ['some data', 0]
    }
    assert(set(side_information['data_item-1']) == set(test_function(data_item) + test_function(data_item)))
    side_information.load_all()
    assert(set(side_information['data_item-1']) == set(test_function(data_item) + test_function(data_item) + test_function(data_item)))

    # Loader by data content