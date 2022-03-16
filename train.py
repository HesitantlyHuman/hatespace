from ironmarch.datasets import IronMarch
from ironmarch.models.nlp import Tokenizer, ArchetypalTransformer
from ironmarch.training import IronmarchTrainer
from transformers.training_args import TrainingArguments

# TODO: Add a cli, so that running the code is even easier

dataset = IronMarch("iron_march_201911")
dataset.summary()

tokenizer = Tokenizer("roberta-base", 512)
dataset = dataset.map(tokenizer, batch_size=256)
train, test = dataset.split(validation_proportion=0.1)

model = ArchetypalTransformer("roberta-base", num_archetypes=12)

training_args = TrainingArguments("./results")
trainer = IronmarchTrainer(
    model=model, args=training_args, train_dataset=train, eval_dataset=test
)
trainer.train()
