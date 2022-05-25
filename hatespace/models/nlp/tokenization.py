from re import I
from typing import Callable, Sequence, Union, List, Dict
import torch
import numpy as np
from transformers import AutoTokenizer
from hatespace.models.nlp.utils import get_sequence_bookends, listify_tokens


class Tokenizer:
    """Tokenizer wrapper class for transformer tokenizers.

    Easy managment of tokenizers from the transformers library, with additional
    functionality to support split tokenization of sequences which are too long.
    """

    def __init__(
        self, model_name_or_path: str, max_token_length: int, mode: str = "split"
    ) -> None:
        """Create a new tokenizer for a specific transformer model

        Creates a new tokenizer to tokenize based of the scheme of a particular
        transformer. Can be set to three different modes: `"split"`, `"first"`, and
        `"last"`. The mode dictates the tokenizer behaviour when the input sequence is
        longer than the `max_token_length`. In the case of `"split"`, half of the max
        length is selected from the front of the sequence, and half of the max length
        is selected from the back.

        Args:
            model_name_or_path (:obj:`str`): Name of the transformer architecture or path to presaved tokenizer.
            max_token_length (:obj:`int`): The longest allowed token sequence.
            mode (:obj:`str`): Strategy for clipping long sequences.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._tokenization_function = Tokenizer._create_tokenization_function(
            self.tokenizer, max_token_length=max_token_length, mode=mode
        )

    def __call__(self, input_strings: Union[List[str], str]) -> Dict[str, torch.Tensor]:
        if isinstance(input_strings, str):
            input_strings = [input_strings]
            return self._tokenization_function(input_strings)
        else:
            tokenized = self._tokenization_function(input_strings)
            return listify_tokens(tokenized)

    def decode(
        self,
        tokens: Union[int, List[int], torch.Tensor],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        """Decodes a token sequence into a readable string

        Args:
            tokens (`Sequence[int]`): A list of input tokens.
            skip_special_tokens (:obj:`bool`, optional): Whether or not to remove special tokens in the decoding.

        Returns:
            `str`: The decoded sequence.
        """
        return self.tokenizer.decode(
            tokens, skip_special_tokens=skip_special_tokens, **kwargs
        )

    def batch_decode(
        self,
        tokens: Union[List[int], List[List[int]], torch.Tensor],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> List[str]:
        """Decodes a batch of token sequences into readable strings

        Args:
            tokens (:type:`Sequence[Sequence[int]]`): A list of token sequences.
            skip_special_tokens (:obj:`bool`, optional): Whether or not to remove special tokens in the decoding.

        Returns:
            `List[str]`: A list of strings, each corresponding to a set of input tokens.
        """
        return self.tokenizer.batch_decode(
            tokens, skip_special_tokens=skip_special_tokens, **kwargs
        )

    def _create_tokenization_function(
        tokenizer: Callable[[Union[torch.Tensor, Sequence]], dict],
        max_token_length: int,
        mode: str = "split",
    ) -> Callable[[Union[torch.Tensor, Sequence]], dict]:
        tokenize_max = lambda tokenizer_input: tokenizer(
            tokenizer_input,
            padding="max_length",
            truncation=False,
            max_length=max_token_length,
        )
        if mode == "split":
            start_length = max_token_length // 2
            end_length = max_token_length - start_length
            # TODO: Fix this, it is unnecessarily slow
            return lambda tokenization_input: {
                key: torch.Tensor(
                    np.array(
                        [
                            get_sequence_bookends(x, start_length, end_length)
                            for x in value
                        ]
                    )
                ).long()
                for key, value in tokenize_max(tokenization_input).items()
            }
        elif mode == "first":
            return lambda tokenization_input: {
                key: torch.Tensor(value[:, :max_token_length])
                for key, value in tokenize_max(tokenization_input).items()
            }
        elif mode == "last":
            return lambda tokenization_input: {
                key: torch.Tensor(value[:, -max_token_length:])
                for key, value in tokenize_max(tokenization_input).items()
            }
        else:
            raise NotImplementedError(f"Mode {mode} is not implemented for Tokenizer!")
