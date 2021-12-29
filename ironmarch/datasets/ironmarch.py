from torch.utils.data import Dataset
import pandas as pd
import html2text
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

if __name__ == '__main__':
    dataset = IronMarch('data/iron-march')
    print(dataset[0])