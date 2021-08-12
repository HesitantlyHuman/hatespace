import torch
from torch.utils import data
from torch.utils.data import DataLoader
from models.utils import interpolate_from_fractionals, squircle_interpolation

class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder, feature_head = None, use_softmax = False):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.feature_head = feature_head

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

class InterpolatedLinearLayers(torch.nn.Module):
    def __init__(self, input_size, output_size, num_layers = 10, bias = 1.0):
        super(InterpolatedLinearLayers, self).__init__()

        fractions = squircle_interpolation(num_layers, power = bias)
        layer_sizes = interpolate_from_fractionals(input_size, output_size, fraction_list = fractions)
        layers = []
        for i in range(1, len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(layer_sizes[-2], layer_sizes[-1]))
            
        self.layers = torch.nn.Sequential(
           *layers
        )

    def forward(self, x):
        return self.layers(x)