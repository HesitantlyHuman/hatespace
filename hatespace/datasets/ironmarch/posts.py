from typing import List, Union, Any
import os
import csv
import sys
from hatespace.datasets.base import Dataset, DataItem
from hatespace.datasets.ironmarch.clean import format_post

csv.field_size_limit(sys.maxsize)


class IronMarch(Dataset):
    """Dataset of forum posts originating from the far-right internet forum IronMarch.

    TRIGGER WARNING:\n
    This dataset contains html snippets of both public and private conversations between
    users of the facist IronMarch forum. The content taken as a whole tends to be racist,
    sexist and hateful. None of the content has been censored.

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
        verbose: bool = True,
    ) -> None:
        self.verbose = verbose
        if self.verbose:
            print("Loading IronMarch dataset...")
        super().__init__(root, download, tasks)
        if side_information is not None:
            self.add_side_information(side_information=side_information)

    def prepare_data(self, directory: str) -> List["DataItem"]:
        dm_file_path = os.path.join(directory, self.FILE_NAMES["direct_messages"])
        direct_message_items = self.read_csv(
            dm_file_path, "msg_id", "msg_post", "direct_message"
        )
        posts_file_path = os.path.join(directory, self.FILE_NAMES["forum_posts"])
        post_items = self.read_csv(
            posts_file_path, "index_id", "index_content", "forum_post"
        )

        return post_items + direct_message_items

    def read_csv(
        self, path: str, id_column: str, data_column: str, id_prefix: str
    ) -> List[DataItem]:
        data_items = []
        with open(path) as csv_file:
            reader = list(csv.reader(csv_file))
            headers = reader.pop(0)
            id_index, data_index = headers.index(id_column), headers.index(data_column)

            if self.verbose:
                try:
                    from tqdm import tqdm

                    progress_bar = tqdm(
                        total=len(reader),
                        position=0,
                        leave=True,
                        desc=f"Formatting {path}...",
                    )
                    empty_posts = 0

                except ModuleNotFoundError:
                    print("Please install tqdm if you wish to have a progress bar.")

            for item in reader:
                try:
                    post_data = format_post(str(item[data_index]))
                except ValueError:
                    if self.verbose:
                        empty_posts += 1
                    continue
                data_items.append(
                    DataItem(
                        data=post_data,
                        id=f"{id_prefix}-" + str(item[id_index]),
                        target=None,
                    )
                )

                if self.verbose:
                    progress_bar.update(1)
                    progress_bar.set_postfix({"Empty Posts": empty_posts})

        return data_items

    def download(self, directory: str) -> None:
        raise AttributeError(
            "Cannot download the IronMarch dataset!\nIf you do not have the data, you may request it by contacting tannersims@hesitantlyhuman.com"
        )

    def add_side_information(self, side_information: Union[List[dict], dict]) -> None:
        raise NotImplementedError


if __name__ == "__main__":
    dataset = IronMarch("data/iron_march")
    dataset.summary()
    print(dataset[32])
