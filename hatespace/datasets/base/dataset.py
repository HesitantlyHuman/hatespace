"""Base dataset classes for ironmarch

Houses Dataset and the associated classes DatasetView and
ConcatDatasetView
"""

from typing import (
    Callable,
    Hashable,
    Mapping,
    Sequence,
    Tuple,
    List,
    Union,
    Any,
    Iterator,
)
import logging
import os

from h11 import Data
from hatespace import __location__ as ROOTFLOW_LOCATION
from hatespace.datasets.base.functional import FunctionalDataset
from hatespace.datasets.base.utils import (
    batch_enumerate,
    map_functions,
    get_unique,
    infer_task_from_targets,
)


class Dataset(FunctionalDataset):
    """Abstract class for a ironmarch dataset.

    Implments boilerplate downloading, loading and indexing functionality. Extends
    :class:`FunctionalDataset`, and provides all of the same functional API for
    interacting with the dataset. Except in special cases, this functionality does not
    need to be implemented if you are extending Dataset, and will be available
    by default.

    Supported Functionality:
        slicing: Can be sliced to generate a new dataset, supports lists and slices.
        `map` and `transform`: Supports transforms and mapping functions over data.
        filtering and `where`: New datasets can be created using conditional functions.
        addition: New datasets may be created using the addition operator, which will
            concatenate two datasets together.
        statistics and summaries: Dynamically calculate statistics on the dataset, and
            display useful summaries about its contents.

    Only one function is necessary to extend a Dataset. That is the method
    :meth:`prepare_data`. If you wish to dynamically download the dataset, then the
    :meth:`download` method should also be implemented. For any additional steps which
    your dataset needs to perform, you may also implement the :meth:`setup` method.
    """

    def __init__(
        self, root: str = None, download: bool = None, tasks: List[dict] = []
    ) -> None:
        """Creates an instance of a ironmarch dataset.

        Attempts to load the dataset from root using the :meth:`prepare_data` method.
        Should this fail to find the given file, it will attempt to download the data
        using the :meth:`download` method, after which it will try once again to load
        the data.

        If download is `True` the data will always be downloaded, even if it is already
        present. Alternatively if it is `False`, then no download is allowed,
        regardless. In the case of the default, `None`, the data will be downloaded
        only if :meth:`prepare_data` fails.

        If a root is not provided, the dataset will default to using the
        the following path:
            <path to ironmarch installation>/datasets/data/<dataset class name>/data

        If the dataset is succesfully loaded, it will then attempt to infer the task
        type, given the data targets, if tasks are not provided.

        Args:
            root (:obj:`str`, optional): Where the data is or should be stored.
            download (:obj:`bool`, optional): Whether the dataset should download the
                data.
            tasks: (:type:`List[bool]`, optional): Dataset task names, types and shapes.
        """
        super().__init__()
        self.DEFAULT_DIRECTORY = os.path.join(
            ROOTFLOW_LOCATION, "datasets/data", type(self).__name__, "data"
        )
        if root is None:
            logging.info(
                f"{type(self).__name__} root is not set, using the default data root of {self.DEFAULT_DIRECTORY}"
            )
            root = self.DEFAULT_DIRECTORY

        if download is None:
            try:
                self.data = self.prepare_data(root)
            except FileNotFoundError:
                logging.warning(f"Data could not be loaded from {root}.")
                download = True

        if download is True:
            logging.info(f"Downloading {type(self).__name__} data to location {root}.")
            if not os.path.exists(root):
                os.makedirs(root)
            self.download(root)
            self.data = self.prepare_data(root)
        elif download is False:
            self.data = self.prepare_data(root)
        logging.info(f"Loaded {type(self).__name__} from {root}.")

        self.setup()
        logging.info(f"Setup {type(self).__name__}.")

        if tasks is not None and len(tasks) == 0:
            tasks = self._infer_tasks()
            logging.info(f"Tasks not specified, setting automatically")
        self._tasks = tasks

    def prepare_data(self, directory: str) -> List["DataItem"]:
        """Prepares data for a ironmarch dataset.

        Loads the data from a directory path and returns a list of
        :class:`DataItem`s, one for each dataset example in dataset.

        Args:
            directory (str): The directory where we should look for our data.

        Returns:
            List[DataItem]: The loaded data items.
        """
        raise NotImplementedError

    def download(self, directory: str) -> None:
        """Downloads the data for the dataset to a specified directory.

        Args:
            directory (str): Directory to download the data to.
        """
        raise NotImplementedError

    def setup(self):
        """Performs additional setup steps for the dataset"""
        pass

    def tasks(self):
        """Returns a list of dataset tasks

        Returns a list containing each task for the dataset. The tasks are formatted
        as a dictionary with the following fields:
            {
                "name" : <task name> (str),
                "type" : <task type> (str),
                "shape" : <task shape> (tuple)
            }

        Returns:
            List[dict]: The list of tasks associated with the dataset.
        """
        return self._tasks

    def _infer_tasks(self):
        """Splits targets and infers task information"""
        example_targets = self.index(0)[2]
        if example_targets is None:
            return None
        if isinstance(example_targets, Mapping):
            tasks = []

            def multitask_generator(task_name):
                for item in self:
                    yield item["target"][task_name]

            for task_name in example_targets.keys():
                generator = multitask_generator(task_name)
                task_type, task_shape = infer_task_from_targets(generator)
                tasks.append(
                    {"name": task_name, "type": task_type, "shape": task_shape}
                )
            return tasks
        else:

            def single_task_generator():
                for item in self:
                    yield item["target"]

            task_type, task_shape = infer_task_from_targets(single_task_generator())
            return [{"name": "task", "type": task_type, "shape": task_shape}]

    def __len__(self) -> int:
        """Gets the length of the dataset."""
        return len(self.data)

    def index(self, index: int) -> tuple:
        """Gets a single data example

        Retrieves a single data example at the given index. Since :meth:`index` is used
        internally, it does not pack the result into a dict, instead returning a tuple.
        (This is prefered for performance)

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple of three items, respectively, the id of the data item, the
                data content of the item, and the target of the data item.
        """
        data_item = self.data[index]
        id, data, target = data_item.id, data_item.data, data_item.target
        if id is None:
            id = f"{type(self).__name__}-{index}"
        if self.has_data_transforms:
            data = map_functions(data, self.data_transforms)
        if self.has_target_transforms:
            target = map_functions(target, self.target_transforms)
        return (id, data, target)

    def set_index(self, index: int, value: Any) -> None:
        if len(value) == 3:
            id, data, target = value
        elif len(value) == 2:
            id = None
            data, target = value
        else:
            raise AttributeError(
                f"Tried to set dataset entry to value {value}, which is could not be unpacked into 2 or 3 elements!"
            )
        self.data[index] = DataItem(id=id, data=data, target=target)


# TODO Add custom getattr for the dataset views so that if there is a custom
# attribute on a dataset, a view of that dataset will have the same attribute
class DatasetView(FunctionalDataset):
    """Noncopy subset of a dataset.

    A dataset view is a low cost abstraction which allows for interacting with a
    subset of the dataset without duplication of data. Like :class:`Dataset`
    the view extends :class:`FunctionalDataset`, and provides all of the same
    functional API. (i.e. You can map, transform, take slices, etc)
    """

    def __init__(
        self,
        dataset: FunctionalDataset,
        view_indices: List[int],
        sorted: bool = True,
    ) -> None:
        """Creates an new view of a dataset.

        Args:
            dataset (FunctionalDataset): The dataset which we are taking a view of.
            view_indices (List[int]): Indices corresponding to which data items from
                the dataset we would like to include in the view.
            sorted (:obj:`bool`, optional): Wether to sort the indices so that the
                view maintains ordering when iterating.
        """
        super().__init__()
        self.dataset = dataset
        unique_indices = get_unique(view_indices, ordered=sorted)
        self.data_indices = unique_indices

    def tasks(self) -> List[dict]:
        """Returns a list of dataset tasks.

        Returns a list containing each task for the dataset. The tasks are formatted
        as a dictionary with the following fields:
            {
                "name" : <task name> (str),
                "type" : <task type> (str),
                "shape" : <task shape> (tuple)
            }

        Returns:
            List[dict]: The list of tasks associated with the dataset.
        """
        return self.dataset.tasks()

    def __len__(self):
        """Returns the length of the view"""
        return len(self.data_indices)

    def index(self, index):
        """Gets a single data example.

        Retrieves a single data example at the given index from the underlying dataset.
        This may be a :class:`Dataset` or another :class:`DatasetView`,
        potentially even a :class:`ConcatDatasetView`.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple of three items, respectively, the id of the data item, the
                data content of the item, and the target of the data item.
        """
        id, data, target = self.dataset.index(self.data_indices[index])
        if self.has_data_transforms:
            data = map_functions(data, self.data_transforms)
        if self.has_target_transforms:
            target = map_functions(target, self.target_transforms)
        return (id, data, target)

    def set_index(self, index: int, value: Any) -> None:
        self.dataset.set_index(self.data_indices[index], value)

    def __getattr__(self, attribute_name):
        try:
            return super().__getattr__(attribute_name)
        except AttributeError:
            return self.dataset.__getattr__(attribute_name)


class ConcatDatasetView(FunctionalDataset):
    """Noncopy concatenation of two datasets.

    A concat dataset view is a low cost abstraction which allows for interacting with a
    concatenation of two datasets without duplication of data. Like
    :class:`Dataset` the view extends :class:`FunctionalDataset`, and provides
    all of the same functional API. (i.e. You can map, transform, take slices, etc)
    """

    def __init__(
        self,
        datatset_one: FunctionalDataset,
        dataset_two: FunctionalDataset,
    ):
        """Creates an new concatenated view of two datasets.

        Combines the two given datasets to form a concatenated dataset. Indexing with
        i < len(`dataset_one`) will access the first dataset and indexing
        i >= len(`dataset_one`) will access the second dataset. Creating the combination
        will fail if either of the datasets are not an instance of
        :class:`FunctionalDataset` or if the datasets each have the same task with
        different shapes or types.

        Args:
            dataset_one (FunctionalDataset): The first component of our new dataset.
            dataset_two (FunctionalDataset): The second component of our new dataset.
        """
        assert isinstance(datatset_one, FunctionalDataset) and isinstance(
            dataset_two, FunctionalDataset
        ), f"Cannot concatenate {type(datatset_one)} and {type(dataset_two)}!"
        super().__init__()
        self.dataset_one = datatset_one
        self.dataset_two = dataset_two
        self.transition_point = len(datatset_one)

        self._tasks = ConcatDatasetView._combine_tasks(
            datatset_one.tasks(), dataset_two.tasks()
        )

    def tasks(self):
        """Returns a list of dataset tasks for the two datasets.

        Returns a list containing each unique task for the datasets. The tasks are
        formatted as a dictionary with the following fields:
            {
                "name" : <task name> (str),
                "type" : <task type> (str),
                "shape" : <task shape> (tuple)
            }

        Returns:
            List[dict]: The list of tasks associated with the datasets.
        """
        return self._tasks

    # TODO This function is doing a bit too much maybe should be refactored
    # and some of the functionality moved into utils
    def _combine_tasks(task_list_one, task_list_two):
        """Returns the unique tasks from two lists of tasks"""
        tasks = []

        task_names_one = {task["name"] for task in task_list_one}
        task_names_two = {task["name"] for task in task_list_two}
        combined_names = task_names_one | task_names_two

        for task_name in combined_names:
            if task_name in task_names_one and task_name in task_names_two:
                overlapping_task_one = [
                    task for task in task_list_one if task["name"] == task_name
                ][0]
                overlapping_task_two = [
                    task for task in task_list_one if task["name"] == task_name
                ][0]

                task_type_one = overlapping_task_one["type"]
                task_type_two = overlapping_task_two["type"]
                if not task_type_one == task_type_two:
                    raise ValueError(
                        f"Found two tasks with name {task_name} but types {task_type_one} and {task_type_two}"
                    )

                shape_one = overlapping_task_one["shape"]
                shape_two = overlapping_task_two["shape"]
                if not shape_one == shape_two:
                    raise ValueError(
                        f"Found two tasks with name {task_name} and type {task_type_one} but shapes {shape_one} and {shape_two}"
                    )

                tasks.append(overlapping_task_one)
            elif task_name in task_names_one:
                task = [task for task in task_list_one if task["name"] == task_name][0]
                tasks.append(task)
            elif task_name in task_names_two:
                task = [task for task in task_list_two if task["name"] == task_name][0]
                tasks.append(task)
        return tasks

    def __len__(self):
        """Returns the total length of the concatenated datasets."""
        return len(self.dataset_one) + len(self.dataset_two)

    def index(self, index):
        """Gets a single data example.

        Retrieves a single data example at the given index from the underlying datasets.
        These may be a :class:`Dataset` or another :class:`DatasetView`,
        potentially even a :class:`ConcatDatasetView`.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple of three items, respectively, the id of the data item, the
                data content of the item, and the target of the data item.
        """
        if index < self.transition_point:
            selected_dataset = self.dataset_one
        else:
            selected_dataset = self.dataset_two
            index -= self.transition_point
        id, data, target = selected_dataset.index(index)
        if self.has_data_transforms:
            data = map_functions(data, self.data_transforms)
        if self.has_target_transforms:
            target = map_functions(target, self.target_transforms)
        return (id, data, target)

    def set_index(self, index: int, value: Any) -> None:
        if index < self.transition_point:
            selected_dataset = self.dataset_one
        else:
            selected_dataset = self.dataset_two
            index -= self.transition_point
        selected_dataset.set_index(index, value)


class DataItem:
    """A single data example for ironmarch datasets.

    A container class for data in ironmarch datasets, intended to provide a rigid API
    on which the :class:`FunctionalDataset`s can depened. Behaviorally, it is similar
    to a named tuple, since the only available slots are `id`, `data` and `target`.

    Attributes:
        id (:obj:`Hashable`, optional): A unique id for the dataset example.
        data (Any): The data of the dataset example.
        target (:obj:`Any`, optional): The task target(s) for the dataset example.
    """

    __slots__ = ("id", "data", "target")

    # TODO We may want to unpack lists with only a single item for mappings and nested lists as well
    def __init__(self, data: Any, id: Hashable = None, target: Any = None) -> None:
        """Creates a new data item.

        Args:
            id (:obj:`Hashable`, optional): A unique id for the dataset example.
            data (Any): The data of the dataset example.
            target (:obj:`Any`, optional): The task target(s) for the dataset example
        """
        self.data = data
        self.id = id

        if isinstance(target, Sequence) and not isinstance(target, str):
            target_length = len(target)
            if target_length == 0:
                target = None
            elif target_length == 1:
                target = target[0]
        self.target = target

    def __getitem__(self, index: int):
        if index == 0:
            return self.id
        elif index == 1:
            return self.data
        elif index == 2:
            return self.target
        else:
            raise ValueError(f"Invalid index {index} for DataItem")

    def __iter__(self) -> Iterator[Tuple[Hashable, Any, Any]]:
        """Returns an iterator to support tuple unpacking

        For example:
            >>> data_item = DataItem([1, 2, 3], id = 'item', target = 0)
            >>> id, data, target = data_item
        """
        return iter((self.id, self.data, self.target))
