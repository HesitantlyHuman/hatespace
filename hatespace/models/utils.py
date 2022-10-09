from typing import Any, List, Dict
import torch
import numpy as np


def get_sequence_bookends(sequence, length_start, length_end):
    start_component = sequence[:length_start]
    end_component = sequence[-length_end:]
    if isinstance(sequence, torch.Tensor):
        return torch.cat((start_component, end_component), dim=0)
    else:
        return np.concatenate((start_component, end_component), axis=0)


def listify_tokens(tokens: Dict[Any, list]) -> List[dict]:
    keys = tokens.keys()
    return [
        {key: value for key, value in zip(keys, value_tuple)}
        for value_tuple in zip(*tokens.values())
    ]
