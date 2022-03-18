from hatespace.datasets import IronMarch
from hatespace.models.nlp import Tokenizer, ArchetypalTransformer
from hatespace.training import HatespaceTrainer
from transformers.training_args import TrainingArguments

# TODO: Add a cli, so that running the code is even easier

dataset = IronMarch("iron_march_201911")
dataset.summary()

tokenizer = Tokenizer("roberta-base", 512)
dataset = dataset.map(tokenizer, batch_size=256)
train, test = dataset.split(validation_proportion=0.1)

model = ArchetypalTransformer("roberta-base", num_archetypes=12)

training_args = TrainingArguments("./results")
trainer = HatespaceTrainer(
    model=model, args=training_args, train_dataset=train, eval_dataset=test
)
trainer.train()
