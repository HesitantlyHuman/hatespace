import torch
from tqdm import tqdm
from hatespace.datasets import IronMarch
from hatespace.datasets.base import DataLoader
from hatespace.models.nlp import Tokenizer, TransformerEmbedder
from hatespace.models.archetypal import TransformerArchetypal, LinearArchetypal

# TODO: Add a cli, so that running the code is even easier

if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
print(f"Using {DEVICE}...")

print("Loading dataset...")
dataset = IronMarch("iron_march_201911")
dataset.summary()

print("Tokenizing dataset...")
tokenizer = Tokenizer("roberta-base", 512)
dataset = dataset.map(tokenizer, batch_size=256)
train, test = dataset.split(validation_proportion=0.1)

test_loader = DataLoader(train, batch_size=8)

print("Loading transformer models...")
transformer_model = TransformerEmbedder("roberta-base")
transformer_model.to(DEVICE)
for batch in test_loader:
    with torch.no_grad():
        data = {key: tensor.to(DEVICE) for key, tensor in batch["data"].items()}
        sequence_logits, hidden_state = transformer_model(**data)
    break

inner_embedder = LinearArchetypal(512 * 768, 12)
full_model = TransformerArchetypal(
    transformers=transformer_model, inner_embedder=inner_embedder
)
full_model.to(DEVICE)

print("Run inference test...")
test_length = 500
for batch_num, batch in enumerate(tqdm(test_loader, total=test_length)):
    with torch.no_grad():
        data = {key: tensor.to(DEVICE) for key, tensor in batch["data"].items()}
        sequence_logits, embedding = full_model(**data)
    if batch_num >= test_length:
        break
