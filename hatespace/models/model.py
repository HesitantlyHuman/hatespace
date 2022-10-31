from typing import Optional, Tuple, Union
import torch
from torch.nn import Module
from transformers import EncoderDecoderModel, AutoTokenizer
from hatespace.models.outputs import ArchetypalTransformerModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from hatespace.models.utils import shift_tokens_right

from transformers import logging

logging.set_verbosity_error()

# TODO Make this easy to switch between inner embedder and no inner_embedder (or other inner_embedders for that matter)
# TODO This guy needs a better name
class TransformerArchetypal(EncoderDecoderModel):
    def __init__(
        self,
        encoder_decoder: EncoderDecoderModel,
        inner_embedder: Module,
        tokenizer: AutoTokenizer,
    ) -> None:
        super().__init__(
            config=encoder_decoder.config,
            encoder=encoder_decoder.encoder,
            decoder=encoder_decoder.decoder,
        )
        del encoder_decoder

        self.train()
        # self.gradient_checkpointing_disable()
        self.gradient_checkpointing_enable()

        self.inner_embedder = inner_embedder
        self.vocab_size = self.decoder.config.vocab_size

        self.config.decoder_start_token_id = tokenizer.cls_token_id
        self.config.pad_token_id = tokenizer.pad_token_id
        self.config.vocab_size = self.config.decoder.vocab_size
        self.config.bos_token_id = tokenizer.cls_token_id

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Tuple[str, str]],
        inner_embedder: Module = None,
        tokenizer: AutoTokenizer = None,
    ) -> "TransformerArchetypal":
        if isinstance(model_name_or_path, (tuple, list)):
            encoder_type, decoder_type = model_name_or_path
        else:
            encoder_type = model_name_or_path
            decoder_type = model_name_or_path
        encoder_decoder = EncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_type, decoder_type
        )
        if inner_embedder is None:
            inner_embedder = ArchetypalHead(512, 769, 12)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(encoder_type)
        return cls(encoder_decoder, inner_embedder, tokenizer)

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

        predicted_encoder_hidden_states, embeddings = self.inner_embedder(
            encoder_outputs.last_hidden_state
        )

        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # When training the values of these are as follows:
        #   decoder_input_ids: <class 'torch.Tensor'> with shape (batch_size, 512)
        #   decoder_attention_mask: <class 'torch.Tensor'> with shape (batch_size, 512, 768)
        #   predicted_encoder_hidden_states: <class 'torch.Tensor'> with shape (batch_size, 512)
        #   attention_mask: <class 'torch.Tensor'> with shape (batch_size, 512)
        #   decoder_inputs_embeds: <class 'NoneType'>
        #   output_attentions: <class 'NoneType'>
        #   output_hidden_states: <class 'NoneType'>
        #   use_cache: <class 'NoneType'>
        #   past_key_values: True

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


# TODO Add batch normalization
# TODO remember to turn off bias on layers just before batch norm


class ArchetypalHead(Module):
    def __init__(
        self, max_token_length: int, token_dimensions: int, num_archetypes: int
    ) -> None:
        super().__init__()
        self.num_archetypes = num_archetypes
        self.max_token_length = max_token_length
        self.token_dimensions = token_dimensions
        self.input_size = max_token_length * token_dimensions
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.num_archetypes),
            torch.nn.Softmax(dim=1),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.num_archetypes, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, self.input_size),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        input_shape = x.shape
        x = torch.flatten(x, start_dim=1)
        x = torch.nn.functional.pad(x, (0, 393216 - x.shape[1]))
        embedding = self.encoder(x)
        output = torch.reshape(
            self.decoder(embedding),
            (input_shape[0], self.max_token_length, self.token_dimensions),
        )[:, : input_shape[1], :]
        return (output, embedding)
