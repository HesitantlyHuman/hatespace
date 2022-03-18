from typing import List
import os
from hatespace.datasets.base import Dataset, DataItem


class SideInformation(Dataset):
    def prepare_data(self, directory: str) -> List["DataItem"]:
        raise NotImplementedError

    def download(self, directory: str) -> None:
        raise NotImplementedError
