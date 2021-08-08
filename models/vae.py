import torch
from torch.utils import data
from torch.utils.data import DataLoader

class VAE(torch.nn.Module):
    def __init__(self, latent_dim, input_dim, feature_dim, use_softmax = False):
        super(VAE, self).__init__()

        self.encoder = VAE_Encoder(latent_dim,  input_dim)
        self.decoder = VAE_Decoder(latent_dim, input_dim)
        self.feature_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, feature_dim)
        )

        self.softmax = torch.nn.Softmax(dim = 1)
        self.use_softmax = use_softmax

    def forward(self, x):
        mean, log_variance = self.encode(x)
        z = self.reparameterize(mean, log_variance)
        logits = self.decoder(z)
        return logits

    def encode(self, x):
        z = self.encoder(x)
        mean, var = torch.split(z, z.shape[1] // 2, dim = 1)
        return mean, var

    def reparameterize(self, mean, logvar):
        eps = torch.normal(torch.zeros(mean.shape), 1).to(mean.device)
        eps = eps * torch.exp(logvar * .5) + mean
        if self.use_softmax:
            eps = self.softmax(eps)
        return eps

class VAE_Encoder(torch.nn.Module):
    def __init__(self, latent_dim, input_size):
        super(VAE_Encoder, self).__init__()

        self.latent_dim = latent_dim

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1536, 1459),
            torch.nn.ReLU(),
            torch.nn.Linear(1459, 1351),
            torch.nn.ReLU(),
            torch.nn.Linear(1351, 1206),
            torch.nn.ReLU(),
            torch.nn.Linear(1206, 1023),
            torch.nn.ReLU(),
            torch.nn.Linear(1023, 810),
            torch.nn.ReLU(),
            torch.nn.Linear(810, 588),
            torch.nn.ReLU(),
            torch.nn.Linear(588, 386),
            torch.nn.ReLU(),
            torch.nn.Linear(386, 229),
            torch.nn.ReLU(),
            torch.nn.Linear(229, 125),
            torch.nn.ReLU(),
            torch.nn.Linear(125, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.layers(x)

class VAE_Decoder(torch.nn.Module):
    def __init__(self, latent_dim, output_size):
        super(VAE_Decoder, self).__init__()

        self.latent_dim = latent_dim

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(16, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 39),
            torch.nn.ReLU(),
            torch.nn.Linear(39, 76),
            torch.nn.ReLU(),
            torch.nn.Linear(76, 141),
            torch.nn.ReLU(),
            torch.nn.Linear(141, 245),
            torch.nn.ReLU(),
            torch.nn.Linear(245, 394),
            torch.nn.ReLU(),
            torch.nn.Linear(394, 579),
            torch.nn.ReLU(),
            torch.nn.Linear(579, 777),
            torch.nn.ReLU(),
            torch.nn.Linear(777, 964),
            torch.nn.ReLU(),
            torch.nn.Linear(964, 1124),
            torch.nn.ReLU(),
            torch.nn.Linear(1124, 1251),
            torch.nn.ReLU(),
            torch.nn.Linear(1251, 1347),
            torch.nn.ReLU(),
            torch.nn.Linear(1347, 1416),
            torch.nn.ReLU(),
            torch.nn.Linear(1416, 1464),
            torch.nn.ReLU(),
            torch.nn.Linear(1464, 1497),
            torch.nn.ReLU(),
            torch.nn.Linear(1497, 1520),
            torch.nn.ReLU(),
            torch.nn.Linear(1520, 1536)
        )

    def forward(self, x):
        return self.layers(x)

def get_list_of_layers(input_size, output_size, num_layers):
    step_size = 1 / num_layers
    size_list = []

    for i in range(num_layers):
        size_list.append()