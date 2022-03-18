from torch.nn import Module


class Embedder(Module):
    def __init__(self, encoder: Module, decoder: Module) -> None:
        super().__init__()
        assert isinstance(encoder, Module), "Embedder encoder must be a torch Module!"
        assert isinstance(decoder, Module), "Embedder decoder must be a torch Module!"
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        embedding = self.encoder(x)
        return (self.decoder(x), embedding)
