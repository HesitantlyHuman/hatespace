from typing import Any, List, Dict
import torch
import numpy as np


def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError(
            "Make sure to set the decoder_start_token_id attribute of the model's configuration."
        )
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError(
            "Make sure to set the pad_token_id attribute of the model's configuration."
        )
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


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
