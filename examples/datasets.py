from ironmarch.datasets import IronMarch
from ironmarch.models.nlp import Tokenizer

dataset = IronMarch("iron_march_201911")
dataset.summary()

tokenizer = Tokenizer("roberta-base", 512)
dataset = dataset.map(tokenizer, batch_size=256)
dataset.summary()
