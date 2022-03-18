from typing import Union, List, Dict
import torch
import numpy as np
from transformers import AutoTokenizer
from hatespace.models.nlp.utils import get_sequence_bookends, listify_tokens


class Tokenizer:
    def __init__(
        self, model_name_or_path: str, max_token_length: int, mode: str = "split"
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer = lambda tokenizer_input: tokenizer(
            tokenizer_input,
            padding="max_length",
            truncation=False,
            max_length=max_token_length,
        )
        if mode == "split":
            start_length = max_token_length // 2
            end_length = max_token_length - start_length
            # TODO: Fix this, it is unnecessarily slow
            self._tokenize_function = lambda tokenization_input: {
                key: torch.Tensor(
                    np.array(
                        [
                            get_sequence_bookends(x, start_length, end_length)
                            for x in value
                        ]
                    )
                )
                for key, value in self.tokenizer(tokenization_input).items()
            }
        elif mode == "first":
            self._tokenize_function = lambda tokenization_input: {
                key: torch.Tensor(value[:, :max_token_length])
                for key, value in self.tokenizer(tokenization_input).items()
            }
        elif mode == "last":
            self._tokenize_function = lambda tokenization_input: {
                key: torch.Tensor(value[:, -max_token_length:])
                for key, value in self.tokenizer(tokenization_input).items()
            }
        else:
            raise NotImplementedError(f"Mode {mode} is not implemented for Tokenizer!")

    def __call__(self, input_strings: Union[List[str], str]) -> Dict[str, torch.Tensor]:
        tokenized = self._tokenize_function(input_strings)
        return listify_tokens(tokenized)
