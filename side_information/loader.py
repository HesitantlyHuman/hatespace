from typing import Callable, List
import random

#Add some tqdm action up in here
class SideLoader():
    '''
        Loads and saves side information by data item id.
    '''
    def __init__(self, data: dict, save_path: str = 'side_information/data/side_information.csv') -> None:
        self.data = data
        self.keys = list(data.keys())
        self.names = []
        self.side_information = {key: [] for key in self.data.keys()}
        
        if self._cache_exists(save_path):
            self.cache = self._load_cache(save_path)
        else:
            self.cache = None

    def _cache_exists(self, path) -> bool:
        pass
    
    #Should be separable by name
    def _load_cache(self, path) -> dict:
        pass

    def add(self, name: str, function: Callable):
        if name in self.names:
            if self.verify(name, function):
                return
            else:
                raise KeyError(f"Information with the name {name} already exists in Side Loader")
        #Check if we can find the function in the cache
        for key in self.data.keys():
            new_side_info = function(self.data[key])
            side_info = self.side_information[key]
            if isinstance(new_side_info, list):
                side_info = side_info + new_side_info
            else:
                side_info.append(new_side_info)
            self.side_information[key] = side_info

    def load(self, name: str):
        

    def verify(self, name: str, function: Callable, n_items_to_verify: int = 10) -> bool:
        assert(self.cache is not None)
        verification_set = random.sample(self.keys, n_items_to_verify)
        for key in verification_set:
            if self.cache[key] != function(self.data[key]):
                return False
        return True

    def load_all(self, n_items_to_verify: int = 10):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.side_information[key]

def test_function(id: str):
    return 0

if __name__ == '__main__':
    side_information = SideLoader()
    side_information.add('yet-another-function', test_function)
    side_information.add('other-function', test_function)
    del(side_information)
    side_information = SideLoader()
    side_information.add('test-function', test_function)
    side_information.load('other-function')
    assert(side_information.list() == ['test-function', 'other-function'])
    assert(side_information.cached() == ['test-function', 'other-function', 'yet-another-function'])
    assert(side_information[0] == [test_function[0], test_function[0]])
    side_information.load_all()
    assert(side_information[0] == [test_function[0], test_function[0], test_function[0]])