"""Base class for ironmarch datasets.

Implements the basics of the functional API for ironmarch datasets and
dataset-like objects. (For example DatasetView)
"""

from typing import Callable, Sequence, Tuple, List, Union
import os
import random
from torch.utils.data import Dataset

import hatespace.datasets.base.dataset as ironmarch_datasets
from hatespace.datasets.base.utils import get_nested_data_types
from hatespace.datasets.base.display_utils import (
    format_docstring,
    format_examples_tabular,
    format_statistics,
)


class FunctionalDataset(Dataset):
    """Abstract class for the functional dataset API.

    Implements shared behavior for Dataset, DatasetView, and
    ConcatDatasetView. This includes things like slicable indexing,
    and formatted display functionality.
    """

    def __init__(self) -> None:
        self.data_transforms = []
        self.target_transforms = []
        self.has_data_transforms = False
        self.has_target_transforms = False

    def __len__(self):
        """Returns the dataset length"""
        raise NotImplementedError

    def __getitem__(self, index: Union[int, slice, tuple, list]):
        """Indexes dataset

        If the index specified is an integer, the dataset will call its index method,
        pack the result into a dictionary, and return it. However, if the index is
        instead a slice or a list of integers, the dataset will return a view of itself
        with the appropriate indices.

        Args:
            index Union[int, slice, tuple, list]: Specifies the portion of the dataset
                to select.

        Returns:
            Union[dict, DatasetView]: Either a single data item, containing an
                `"id"`, `"data"` and a `"target"`, or a :class:`DatasetView`
                of the desired indices.
        """
        if isinstance(index, int):
            id, data, target = self.index(index)
            return {"id": id, "data": data, "target": target}
        elif isinstance(index, slice):
            data_indices = list(range(len(self))[index])
            return ironmarch_datasets.DatasetView(self, data_indices, sorted=False)
        elif isinstance(index, (tuple, list)):
            return ironmarch_datasets.DatasetView(self, index)

    def __iter__(self):
        """Iterates over dataset

        To avoid the additional overhead of checking types with :meth:`__getitem__`
        we call the dataset's :meth:`index` method for each index. The data is packed
        into a dictionary, so that the outward facing API is consistent.

        Yields:
            dict: A dictionary containing an `"id"`, `"data"` and a `"target"`
        """
        for index in range(len(self)):
            id, data, target = self.index(index)
            yield {"id": id, "data": data, "target": target}

    def index(self, index: int) -> tuple:
        """Gets a data item at the index"""
        raise NotImplementedError

    def split(
        self, validation_proportion: float = 0.1, seed: int = None
    ) -> Tuple["ironmarch_datasets.DatasetView", "ironmarch_datasets.DatasetView"]:
        """Splits the dataset into a train set and a validation set.

        Will return two dataset views of the dataset. Each view is randomly sampled
        from the dataset, and they do not contain any of the same elements.

        Args:
            validation_proportion (float): The proportion of the total dataset size
                to contain in the validation set.
            seed (:obj:`int`, optional): An optional seed for the randomization.

        Returns:
            Tuple[DatasetView, DatasetView]: A tuple containing,
                respectively, the train set and the validation set.
        """
        dataset_length = len(self)
        indices = list(range(dataset_length))
        random.Random(seed).shuffle(indices)
        n_test = int(dataset_length * validation_proportion)
        return (
            ironmarch_datasets.DatasetView(self, indices[n_test:], sorted=False),
            ironmarch_datasets.DatasetView(self, indices[:n_test], sorted=False),
        )

    def map(
        self,
        function: Union[Callable, List[Callable]],
        targets: bool = False,
        batch_size: int = None,
    ) -> Union["ironmarch_datasets.Dataset", "ironmarch_datasets.DatasetView"]:
        """Maps a function over the dataset"""
        raise NotImplementedError

    def where(
        self,
        filter_function: Callable,
        targets: bool = False,
    ) -> "ironmarch_datasets.DatasetView":
        """Selects a dataset from a conditional function

        Creates a new view from the dataset of every item for which the conditional
        statement is `True`. Does not modify the original dataset.

        Args:
            filter_function (Callable): A conditional function which returns `True`
                for items you would like to have in the resulting, filtered, set.
            targets (:obj:`bool`, optional): A flag indicating whether the
                filter_function is applied to the dataset targets instead of the data

        Returns:
            DatasetView: A view of the dataset which contains only dataset
                items for which the filter_function was `True`
        """
        if targets:
            conditional_attr = "target"
        else:
            conditional_attr = "data"

        filtered_indices = []
        for index, item in enumerate(self):
            if filter_function(item[conditional_attr]):
                filtered_indices.append(index)
        return ironmarch_datasets.DatasetView(self, filtered_indices)

    # TODO if we wanted transform to be truly functional, we could just return
    # a new view, but that may be a costly abstraction
    def transform(
        self, function: Union[Callable, List[Callable]], targets: bool = False
    ) -> Union["ironmarch_datasets.Dataset", "ironmarch_datasets.DatasetView"]:
        """Applies a transform to the dataset

        Adds a new transform to the dataset, after all current transforms. The new
        transform will be run each time an item is selected from this dataset.
        (Useful for random augmentation, for example).

        :meth:`transform` returns self to better support a functional API. Keep in
        mind that it is not truly functional, and that the dataset is modified in
        place for space, storage and speed concerns.

        Args:
            function (Union[Callable, List[Callable]]): The transform function or
                list of functions you would like to add.
            targets (:obj:`bool`, optional): Wether this transform should apply
                to the dataset targets.

        Returns:
            Union[Dataset, DatasetView]: Returns `self`.
        """
        if not isinstance(function, (tuple, list)):
            function = [function]
        if targets:
            self.target_transforms += function
            self.has_target_transforms = True
        else:
            self.data_transforms += function
            self.has_data_transforms = True
        return self

    def __add__(
        self, dataset: "FunctionalDataset"
    ) -> "ironmarch_datasets.ConcatDatasetView":
        """Adds two datasets together

        Creates a new ConcatDatasetView given `self` and another
        Dataset or similar.

        Args:
            dataset (FunctionalDataset): The dataset to add to `self`.

        Returns:
            ConcatDatasetView: A new, concatenated dataset.

        Raises:
            AttributeError: If `dataset` is not a :class:`FunctionalDataset`.
        """
        if not isinstance(dataset, FunctionalDataset):
            raise AttributeError(f"Cannot add a {type(dataset)} to a dataset")

        return ironmarch_datasets.ConcatDatasetView(self, dataset)

    def tasks(self):
        """Returns dataset target tasks"""
        raise NotImplementedError

    def stats(self):
        """Gets common statistics for dataset.

        Calculates a set of common and useful statistics for the dataset. May attempt
        to decide dynamically which statistics to include, depending on the data and
        target types.

        Returns:
            dict: A dictionary of the collected statistics.
        """
        data_example = self[0]["data"]
        target_example = self[0]["target"]
        tasks = self.tasks()
        if tasks is not None and len(tasks) == 1:
            tasks = tasks[0]
        return {
            "length": len(self),
            "data_types": get_nested_data_types(data_example),
            "target_types": get_nested_data_types(target_example),
            "tasks": tasks,
        }

    def examples(self, num_examples: int = 5) -> List[dict]:
        """Returns multiple examples from the dataset

        Gets a list of examples from the dataset, up to the specified number or the
        size of the dataset, if the requested amount is too large.

        Args:
            num_examples (int): The number of examples to get.

        Returns:
            List[dict]: A list of examples from the dataset.
        """
        num_examples = min(len(self), num_examples)
        return [self[i] for i in range(num_examples)]

    def summary(self, output_width: int = None):
        """Print a formatted summary of the dataset

        Collects the dataset docstring, statistics and some examples, then prints them
        in a formatted summary to the console.

        Args:
            output_width (:obj:`int`, optional): Optional control for the width of the
                formatted output. Defaults to 150 or the console width, whichever is
                smaller.
        """
        terminal_size = os.get_terminal_size()
        if output_width is None:
            description_width = min(150, terminal_size.columns)
        else:
            description_width = output_width

        print(f"{type(self).__name__}:")
        print(format_docstring(type(self).__doc__, description_width, indent=True))

        print("\nStats:")
        print(format_statistics(self.stats(), description_width, indent=True))

        print("\nExamples:")
        print(format_examples_tabular(self.examples(), description_width, indent=True))
