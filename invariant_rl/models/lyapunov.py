
import torch, torch.nn as nn

class SigmoidGeometryAwareW(nn.Module):
    def __init__(self, dims, region):
        super().__init__()
        layers = []
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.Sigmoid()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.model = nn.Sequential(*layers)
        self.region = region

    def forward(self, x):
        x2 = (x**2).sum(dim=-1, keepdim=True)
        a = 1 - torch.sqrt((x[:,0]/self.region.a)**2 + (x[:,1]/self.region.b)**2).unsqueeze(-1)
        a = torch.clamp(a, 0, 1)
        return a * x2 * torch.nn.functional.softplus(self.model(x)) + (1-a)

    def forward_with_grad(self, x):
        x = x.detach().requires_grad_(True)
        y = self.forward(x)
        g = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        return y.squeeze(-1), g
