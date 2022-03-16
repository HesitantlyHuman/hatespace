"""Functionality utilities for rootflow datasets.

Houses simple utility functions key to the behavior of rootflow datasets.
"""

from typing import Any, Callable, Iterable, Mapping, Sequence, Union, Tuple, List
import torch
from torch.utils.data.dataloader import default_collate


def default_collate_without_key(
    unprocessed_batch: List[dict],
    key_to_remove: str,
) -> Union[torch.Tensor, dict, tuple, list]:
    """Collates batches removing the target.

    Args:
        unprocessed_batch (list): A list of data elements returned from a :class:`Dataset`
        key_to_remove (str): The key to remove from the batch.

    Returns:
        Union[torch.Tensor, dict, tuple, list]: A batch of data formatted according to
            the type of the data elements in the batch. (i.e. a batch of dicts will return
            a dict)
    """
    unprocessed_batch = [
        {key: value for key, value in batch_item.items() if not key == key_to_remove}
        for batch_item in unprocessed_batch
    ]
    return default_collate(unprocessed_batch)


def batch(iterable: Iterable, batch_size: int = 1) -> list:
    """Batches an iterable.

    Yeilds chunks from an iterable at a specified size. Note that the last batch will
    have size of `len(iterable) % batch_size` instead of `batch_size`.

    Args:
        iterable (Iterable): The iterable to be batched
        batch_size (:obj:`int`, optional): The size of each batch, except the last.

    Yeilds:
        list: A list of the iterable items in each batch.
    """
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        upper = min(ndx + batch_size, length)
        yield iterable[ndx:upper]


def batch_enumerate(iterable: Iterable, batch_size: int = 1) -> Tuple[slice, list]:
    """Enumerates in batches.

    Enumerates an iterable in consistent length batches, yielding the slice and batch
    for each. Note that the last batch will have size of `len(iterable) % batch_size`
    instead of `batch_size`.

    Args:
        iterable (Iterable): Some iterable which we would like to split into batches.
        batch_size (:obj:`int`, optional): The size of each batch, except the last.

    Yields:
        Tuple[slice, list]: A tuple containing, respectively, the slice corresponding
            to the batch's location in the iterable and a list containing the batch.
    """
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        upper = min(ndx + batch_size, length)
        yield (slice(ndx, upper), iterable[ndx:upper])


# TODO Using the term composition instead of map might be better and more mathematically accurate.
def map_functions(obj: object, function_list: Iterable[Callable]) -> Any:
    """Maps multiple functions on an object.

    Returns the composition of multiple functions on an object. The functions are
    called in the order given.

    Args:
        obj (object): The object which you would like to map multiple functions on.
        function_list (Iterable[Callable]): An ordered collection of functions to map.

    Returns:
        Any: The result of the function_list mapped on obj.
    """
    value = obj
    for function in function_list:
        value = function(value)
    return value


def get_unique(input_iterator: Iterable, ordered: bool = True) -> list:
    """Returns unique elements.

    Returns only the unique elements of a given iterable, optionally in the order
    which they were given. If the order is maintained, elements will appear in the
    order of their first appearance.

    Args:
        input_iterator (Iterable): the iterator which you would like to reduce to only
            its unique elements.
        ordered (bool): A flag indicating wether the element order should be preserved.

    Returns:
        list: The unique items from input_iterator.
    """
    if ordered:
        unique = list(set(input_iterator))
        unique.sort()
        return unique
    else:
        # TODO A better algorithm should be used. This is worst case O(n**2) in time,
        # and not great for memory either. Will cause problem for excessively large
        # iterator inputs.
        seen = set()
        seen_add = seen.add
        return [item for item in input_iterator if not (item in seen or seen_add(item))]


def get_nested_data_types(object: Any) -> Union[dict, list, type]:
    """Returns the types of potentially nested structures.

    Collects the types of Sequence and Mapping structures into a :obj:`list` or
    :obj:`dict` respectively. The function will do so recursively. (For example,
    potentially returning a list of dicts of types). When the object in question
    is a Sequence, order is maintained. When the object in question is a Mapping
    the keys are maintained.

    Args:
        object (Any): The object for which you would like its type(s).

    Returns:
        Union[dict, list, type]: Either a type, dict, or list, depending on the
            type of object.
    """
    if (
        isinstance(object, Sequence)
        and not isinstance(object, str)
        and not isinstance(object, torch.Tensor)
    ):
        return list(set([get_nested_data_types(element) for element in object]))
    elif isinstance(object, Mapping):
        return {key: get_nested_data_types(value) for key, value in object.items()}
    else:
        return type(object)


def infer_task_from_targets(target_list: list) -> Tuple[str, tuple]:
    """Infers the type and shape of a task.

    Infers the supervised task type and shape given a list of task targets.
    Currently supported tasks are: `"classification"`, `"binary"`, `"multitarget"`,
    and `"regression"`. (`"multitarget"` is a multitarget binary classification task).
    If the targets are not of the types and shapes expected for any of the above
    mentioned task types, then the function will instead return None for the type.

    Args:
        target_list (list): A list of targets for a particular supervised task.

    Returns:
        Tuple[str, tuple]: A tuple containing, respectively, the string corresponding
            to the type of task and a tuple which describes the shape of the target,
            given the infered task.
    """
    if not target_list:
        return None

    first_target = next(target_list)
    if isinstance(first_target, Sequence) and not isinstance(first_target, str):
        first_target_element = next(first_target_element)
        if isinstance(first_target_element, (int, torch.LongTensor)):
            # This needs to be adjusted to work with >1D tensors
            max_element = max([max(target) for target in target_list])
            if max_element > 1:
                return ("binary", max_element)
            max_list_sum = max([sum(target) for target in target_list])
            if max_list_sum > 1:
                return ("binary", len(first_target))
            else:
                return ("classification", len(first_target))
        elif isinstance(first_target_element, (bool, torch.BoolTensor)):
            raise NotImplementedError
        elif isinstance(first_target_element, float):
            return ("regression", len(first_target))
        elif isinstance(first_target_element, torch.FloatTensor):
            return ("regression", (len(first_target), *first_target_element.shape))
    elif isinstance(first_target, (int, torch.LongTensor)):
        max_class_val = max(target_list)
        if max_class_val > 1:
            return ("classification", max_class_val)
        else:
            return ("binary", max_class_val)
    elif isinstance(first_target, (bool, torch.BoolTensor)):
        return ("binary", 2)
    elif isinstance(first_target, float):
        return ("regression", 1)
    elif isinstance(first_target, torch.FloatTensor):
        return ("regression", first_target.shape)
    else:
        return (None, None)
