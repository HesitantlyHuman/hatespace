from typing import List, Union, Any
import os
import csv
import sys
from html2text import html2text
from hatespace.datasets.base import Dataset, DataItem

csv.field_size_limit(sys.maxsize)


class IronMarch(Dataset):
    """Dataset of forum posts originating from the far-right internet forum IronMarch.

    TRIGGER WARNING:
    This dataset contains html snippets of both public and private conversations between
    users of the facist IronMarch forum. The content as a whole tends to be racist,
    sexist and almost every other flavor of vile. None of the content has been censored.

    The text within the dataset comes from two main sources. Private messages between
    users, and posts made on the public forum. To check if a given message is was a
    forum post or a direct message, simply check the item id. The item ids are prefixed
    accordingly.
    """

    FILE_NAMES = {
        "direct_messages": "core_message_posts.csv",
        "forum_posts": "core_search_index.csv",
    }

    def __init__(
        self,
        root: str = None,
        download: bool = None,
        tasks: List[dict] = [],
        side_information: Union[List[dict], dict] = None,
    ) -> None:
        print("Loading IronMarch dataset...")
        super().__init__(root, download, tasks)
        if side_information is not None:
            self.add_side_information(side_information=side_information)
        self.format_all()

    def prepare_data(self, directory: str) -> List["DataItem"]:
        dm_file_path = os.path.join(directory, self.FILE_NAMES["direct_messages"])
        direct_message_items = self.read_csv(
            dm_file_path, "msg_id", "msg_post", "direct_message"
        )
        posts_file_path = os.path.join(directory, self.FILE_NAMES["forum_posts"])
        post_items = self.read_csv(
            posts_file_path, "index_id", "index_content", "forum_post"
        )

        return direct_message_items + post_items

    def read_csv(
        self, path: str, id_column: str, data_column: str, id_prefix: str
    ) -> List[DataItem]:
        data_items = []
        with open(path) as csv_file:
            reader = csv.reader(csv_file)
            headers = next(reader)
            id_index, data_index = headers.index(id_column), headers.index(data_column)

            for item in reader:
                data_items.append(
                    DataItem(
                        data=str(item[data_index]),
                        id=f"{id_prefix}-" + str(item[id_index]),
                        target=None,
                    )
                )
        return data_items

    def format_all(self) -> None:
        print("Formatting posts...")
        try:
            from tqdm import tqdm
            p_bar = tqdm(total=len(self))

            def format_post_with_progress(post: str) -> str:
                p_bar.update(1)
                return self.format_post(post)
            self.map(format_post_with_progress)
        except ModuleNotFoundError:
            print('Please install tqdm if you wish to have a progress bar.')
            self.map(self.format_post)

    def format_post(self, post: str) -> str:
        return html2text(post)
        # TODO experiment with various strategies for things like links (Somthing like domain only)

    def download(self, directory: str) -> None:
        raise AttributeError(
            "Cannot download the IronMarch dataset!\nIf you do not have the data, you may request it by contacting tannersims@generallyintelligent.me."
        )

    def add_side_information(self, side_information: Union[List[dict], dict]) -> None:
        raise NotImplementedError


if __name__ == "__main__":
    dataset = IronMarch("iron_march_201911")
    print(dataset.summary())
