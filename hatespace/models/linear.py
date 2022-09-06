import torch
from torch.nn import Module


class Embedder(Module):
    def __init__(self, encoder: Module, decoder: Module) -> None:
        super().__init__()
        assert isinstance(encoder, Module), "Embedder encoder must be a torch Module!"
        assert isinstance(decoder, Module), "Embedder decoder must be a torch Module!"
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, *args, **kwargs):
        encoder_ouputs = self.encoder(*args, **kwargs)
        return (self.decoder(encoder_ouputs), encoder_ouputs)

    def freeze(self):
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze(self):
        for parameter in self.parameters():
            parameter.requires_grad = True


class LinearArchetypal(Embedder):
    def __init__(self, input_dimensions, num_archetypes) -> None:
        encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dimensions, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_archetypes),
            torch.nn.Softmax(dim=1),
        )
        decoder = torch.nn.Sequential(
            torch.nn.Linear(num_archetypes, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, input_dimensions),
            torch.nn.ReLU(),
        )
        super().__init__(encoder=encoder, decoder=decoder)

    def forward(self, x):
        input_shape = x.shape
        x = torch.flatten(x, start_dim=1)
        embedding = self.encoder(x)
        output = torch.reshape(self.decoder(embedding), input_shape)
        return (output, embedding)
