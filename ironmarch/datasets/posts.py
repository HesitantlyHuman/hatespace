from typing import List, Union
import os
import pandas as pd
from ironmarch.datasets.base import Dataset, DataItem


class IronMarch(Dataset):
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
        super().__init__(root, download, tasks)
        if side_information is not None:
            self.add_side_information(side_information=side_information)

    def prepare_data(self, directory: str) -> List["DataItem"]:
        dm_file_path = os.path.join(directory, self.FILE_NAMES["direct_messages"])
        dm_csv = pd.read_csv(dm_file_path)[["msg_id", "msg_post"]].rename(
            {"msg_id": "id", "msg_post": "data"}, axis="columns"
        )
        dm_csv["id"] = dm_csv["id"].map(lambda id: "direct_message-" + str(id))

        posts_file_path = os.path.join(directory, self.FILE_NAMES["forum_posts"])
        posts_csv = pd.read_csv(posts_file_path)[["index_id", "index_content"]].rename(
            {"index_id": "id", "index_content": "data"}, axis="columns"
        )
        posts_csv["id"] = posts_csv["id"].map(lambda id: "forum_post-" + str(id))

        data = dm_csv.to_dict(orient="records") + posts_csv.to_dict(orient="records")
        return [
            DataItem(str(data_item["data"]), id=data_item["id"]) for data_item in data
        ]

    def download(self, directory: str) -> None:
        raise AttributeError(
            "Cannot download the IronMarch dataset!\nIf you do not have the data, you may request it by contacting tannersims@generallyintelligent.me."
        )

    def add_side_information(self, side_information: Union[List[dict], dict]) -> None:
        raise NotImplementedError


if __name__ == "__main__":
    dataset = IronMarch("iron_march_201911")
    print(dataset[0])
