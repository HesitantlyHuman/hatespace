from transformers import AutoTokenizer

import os

# Allow for parallelism and disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Tokenizer:
    """Simple huggingface tokenizer wrapper"""

    def __init__(self, model_name, max_length: int = 512):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, text):
        return self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)
