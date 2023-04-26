import torch
from hatespace.models.tokenizer import Tokenizer
from transformers import EncoderDecoderModel

from transformers import logging

logging.set_verbosity_error()

# TODO this functionality should be combined with the train and test functions

model_path = "checkpoints/encoder_decoder/fine-raven/best_model.pt"
base_model_name = "roberta-base"

tokenizer = Tokenizer(base_model_name, 512)
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    base_model_name, base_model_name
)
state_dict = torch.load(model_path)
state_dict = {key.split("module.")[1]: value for key, value in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()
model.to("cuda")

test_strings = [
    "here is a test string",
    "here is another test string",
    "hello there big guy",
    "why is the model behaving in this manner?",
    "it seems like the model is clipping out the first couple of tokens for some reason, "
    "not sure why this is happening only on the generate function?? Needs to be investigated.",
]

test_string_tokens = tokenizer(test_strings)
input_ids = test_string_tokens["input_ids"].to("cuda")
attention_mask = test_string_tokens["attention_mask"].to("cuda")

outputs = model(
    input_ids=input_ids,
    decoder_input_ids=input_ids,
    attention_mask=attention_mask,
    decoder_attention_mask=attention_mask,
)
max_likelihood_tokens = torch.argmax(outputs.logits, dim=-1).cpu()
decoded_tokens = tokenizer.batch_decode(max_likelihood_tokens)
print(decoded_tokens)
