import torch

#Takes a tensor on the unit simplex, and deforms it normally within a given variance on the simplex
class SimplexDeform(torch.nn.Module):
    def __init__(self, n, radius = 0.1):
        self.radius = radius
        self.n = n
        self.std = torch.pow(radius / torch.pow(n, 0.5), 0.5)

    def call(self, x):
        d = torch.normal(mean = 0, std = self.std)
        d = d - torch.sum(d)
        return x + d