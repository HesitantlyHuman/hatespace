from hatecomp import HatecompClassifier

# Load in the iron march dataset
from hatespace.datasets import IronMarch

dataset = IronMarch()

hatecomp_models = [
    "MLMA",
    "Vicomtech",
    "TwitterSexism"
]

side_information = {}
    
for model_name in hatecomp_models:
    model = HatecompClassifier.from_hatecomp_pretrained(model_name)
    for post in dataset:
        post_id = post["id"]
        post_features = side_information.get(post_id, [])
        predictions = model(post["text"])