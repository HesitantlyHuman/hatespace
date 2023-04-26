from transformers import EncoderDecoderModel, AutoTokenizer

# We are using huggingface to download and cache the roberta models during the build
# process of our docker image. This way, they are baked in.

model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "roberta-base", "roberta-base"
)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
