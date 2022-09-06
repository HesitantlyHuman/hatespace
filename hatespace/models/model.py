from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.modeling_utils import ModelOutput

from typing import Optional, Tuple, Union
import torch
from torch.nn import Module
from hatespace.models.linear import Embedder
from transformers import EncoderDecoderModel, AutoTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from itertools import chain

from transformers import logging

logging.set_verbosity_error()


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


# encoder_config = RobertaConfig.from_pretrained("roberta-base")
# encoder = RobertaModel(config=encoder_config)

# decoder_config = RobertaConfig(
#     vocab_size=encoder_config.vocab_size,
#     max_position_embeddings=encoder_config.max_position_embeddings,  # this should be some large value
#     num_attention_heads=8,
#     num_hidden_layers=8,
#     hidden_size=512,
#     add_cross_attention=True,
#     type_vocab_size=1,
#     is_decoder=True,
# )  # Very Important

# decoder = RobertaForCausalLM(config=decoder_config)

# TODO Make this easy to switch between inner embedder and no inner_embedder (or other inner_embedders for that matter)
# TODO This guy needs a better name
class TransformerArchetypal(EncoderDecoderModel):
    def __init__(
        self,
        model_name_or_path: Union[str, Tuple[str]],
        inner_embedder: Embedder = None,
    ) -> None:
        if isinstance(model_name_or_path, (tuple, list)):
            encoder_type, decoder_type = model_name_or_path
        else:
            encoder_type = model_name_or_path
            decoder_type = model_name_or_path
        encoder_decoder = EncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_type, decoder_type
        )

        super().__init__(
            config=encoder_decoder.config,
            encoder=encoder_decoder.encoder,
            decoder=encoder_decoder.decoder,
        )
        del encoder_decoder

        self.train()
        self.gradient_checkpointing_disable()

        self.inner_embedder = inner_embedder
        self.vocab_size = self.decoder.config.vocab_size

        # TODO Find a better solution to this. Possibly pass in the tokenizer
        t = AutoTokenizer.from_pretrained(decoder_type)
        self.config.decoder_start_token_id = t.cls_token_id
        self.config.pad_token_id = t.pad_token_id
        self.config.vocab_size = self.config.decoder.vocab_size
        self.config.bos_token_id = t.cls_token_id

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = (
            return_dict
            if return_dict is not None
            else self.encoder.config.use_return_dict
        )

        kwargs_encoder = {
            argument: value
            for argument, value in kwargs.items()
            if not argument.startswith("decoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )

        if self.inner_embedder is not None:
            predicted_encoder_hidden_states, embeddings = self.inner_embedder(
                encoder_outputs[0]
            )
        else:
            predicted_encoder_hidden_states, embeddings = encoder_outputs[0], None

        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=predicted_encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        return ArchetypalTransformerModelOutput(
            logits=decoder_outputs.logits,
            embeddings=embeddings,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def generate_from_sequence(
        self, inputs: torch.Tensor, *args, **kwargs
    ) -> torch.LongTensor:
        if len(inputs.shape) <= 1:
            inputs = torch.unsqueeze(inputs, dim=0)
        return self.generate(input_ids=inputs, *args, **kwargs)

    def generate_from_embeddings(
        self, embeddings: torch.Tensor, *args, **kwargs
    ) -> torch.LongTensor:
        intermediate_encodings = self.inner_embedder.decoder(embeddings)
        intermediate_encodings = torch.reshape(
            intermediate_encodings, (embeddings.shape[0], 512, 768)
        )
        intermediate_encodings = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=intermediate_encodings
        )
        return self.generate(
            inputs=None, encoder_outputs=intermediate_encodings, *args, **kwargs
        )


@dataclass
class ArchetypalTransformerModelOutput(ModelOutput):
    # TODO update docstring
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    embeddings: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
