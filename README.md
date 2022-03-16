# Ironmarch Sinkhorn Archetypal Analysis
Novel Archetypal Analysis NLP on the ironmarch SQL dataset.

# Overview
This repository is an implementation of a novel method for deep archetypal analysis, with the aim of creating highly interpretable latent spaces. We apply these tools to the analysis of the white supremicist forum iron march, with the aim of quantitative descriptions of sentiment among the forum over time. We use both Google's BERT model, and our embedding VAE to generate a descriptive latent space of this forum, where subsections of these users can be easily localized.

## Dataset
The ironmarch dataset is a datadump originating from the SQL databases of a fascist internet forum by the same name. It contains all posts and direct messages which were made on the site while it was operational from 2011 until 2017.

## Methods
Our work is further development of the method proposed by Keller et al. in the paper titled [Deep Archetypal Analysis](https://arxiv.org/abs/1901.10799). There are two distinct differences to our approach chosen to promote greater interpretability of the resulting latent space.

### Sinkhorn vs Archetypal Loss
The DeepAA framework uses archetypal loss to promote spread towards the vertices of their embedding simplex. Without this term, the model will not map to these locations of the latent space, and interpreting the resulting archetypes becomes impossible. Instead of introducing this additional term, we utilize the sinkhorn distance, as desribed in the paper [Sinkhorn Distances: Lightspeed Computation of Optimal Transport](https://papers.nips.cc/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf) (Cuturi). This distance optimizes toward a uniform distribution on the simplex, fulfilling the role of the archetypal loss, as well as preventing posterior collapse in the encoder network. This allows us to remove KL Divergence, and simplify the hyperparameter space.

### Side Information
Secondly, we enforce strictly linear relationships between the latent space and any side information which the model is training with. This allows to identify vectors in the latent space along which particular features are organized, and the linear requirements ensure that we can easily extrapolate across the space during our analysis. Providing these relationships allows us to contextualize the embeddings in a way that is meaningful to humans, and not just the decoder network.

# Running the Code
## Installation
To install simply clone the repo, then run
```
pip install .
```
from the repo root. The `ironmarch` package will then be installed in the currently active environment.

## Training
To train a model, simply run the `train.py` script. (Currently still under construction. A cli will also be included later)

# Analyzing the Latent Space
This project is currently under development, and no embeddings are available at this time. Feel free to clone the repo and run the code to generate your own.
