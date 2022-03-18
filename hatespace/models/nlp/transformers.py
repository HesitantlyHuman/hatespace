from typing import Union, Tuple

from torch.nn import Module
from transformers import AutoModel
from hatespace.models.archetypal import LinearArchetypal
from hatespace.models.base import Embedder

from transformers import logging

logging.set_verbosity_error()


class ArchetypalTransformer(Module):
    def __init__(
        self,
        transformer_name_or_path: Union[str, Tuple[str]],
        num_archetypes: int,
        latent_head: Module = None,
    ) -> None:
        transformers = TransformerEmbedder(transformer_name_or_path)
        archetypal = LinearArchetypal(
            transformers.encoder.config.hidden_size, num_archetypes=num_archetypes
        )
        super().__init__(outer_embedder=transformers, inner_embedder=archetypal)

    def forward(self, x):
        raise NotImplementedError


class TransformerEmbedder(Embedder):
    def __init__(self, model_name_or_path: Union[str, tuple]) -> None:
        if isinstance(model_name_or_path, (tuple, list)):
            encoder_type, decoder_type = model_name_or_path
        else:
            encoder_type = model_name_or_path
            decoder_type = model_name_or_path
        encoder = AutoModel.from_pretrained(encoder_type)
        decoder = AutoModel.from_pretrained(decoder_type)
        super().__init__(encoder=encoder, decoder=decoder)
        raise NotImplementedError
        # TODO The decoder transformer must still be converted into a decoder using the config

    def forward(self, x):
        raise NotImplementedError
