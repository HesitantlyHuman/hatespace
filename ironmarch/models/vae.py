import torch

from ironmarch.models.base import Embedder


class VAE(Embedder):
    def __init__(self, encoder, decoder, use_softmax=False):
        self.softmax = torch.nn.Softmax(dim=1)
        self.use_softmax = use_softmax
        super().__init__(encoder=encoder, decoder=decoder)

    def forward(self, x):
        mean, log_variance = self.encode(x)
        z = self.reparameterize(mean, log_variance)
        logits = self.decoder(z)
        return logits

    def encode(self, x):
        z = self.encoder(x)
        mean, var = torch.split(z, z.shape[1] // 2, dim=1)
        return mean, var

    def reparameterize(self, mean, logvar):
        eps = torch.normal(torch.zeros(mean.shape), 1).to(mean.device)
        eps = eps * torch.exp(logvar * 0.5) + mean
        if self.use_softmax:
            eps = self.softmax(eps)
        return eps
