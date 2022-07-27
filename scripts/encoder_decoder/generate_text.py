import torch
from hatespace.models.tokenizer import Tokenizer
from transformers import EncoderDecoderModel

from transformers import logging

logging.set_verbosity_error()

tokenizer = Tokenizer("roberta-base", 512, return_as_list=False)
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "roberta-base", "roberta-base"
)
state_dict = torch.load("checkpoints/encoder_decoder/lower_lr_rate/best_model.pth")
# state_dict = state_dict["model"]
state_dict = {key.split("module.")[1]: value for key, value in state_dict.items()}
model.load_state_dict(state_dict)

test_strings = [
    "here is a test string",
    "here is another test string",
    "hello there big guy",
    "why is the model behaving in this manner?",
    "it seems like the model is clipping out the first couple of tokens for some reason, "
    "not sure why this is happening only on the generate function?? Needs to be investigated.",
]

test_string_tokens = tokenizer(test_strings)
input_ids = test_string_tokens["input_ids"]
attention_mask = test_string_tokens["attention_mask"]

outputs = model(
    input_ids=input_ids,
    decoder_input_ids=input_ids,
    attention_mask=attention_mask,
    decoder_attention_mask=attention_mask,
)
max_likelihood_tokens = torch.argmax(outputs.logits, dim=2)
decoded_tokens = tokenizer.batch_decode(max_likelihood_tokens)
print(decoded_tokens)

outputs = model.generate(input_ids=input_ids)
decoded_tokens = tokenizer.batch_decode(outputs)
print(decoded_tokens)
