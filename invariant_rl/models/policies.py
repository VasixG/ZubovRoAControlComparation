
import torch.nn as nn
class MLPPolicy(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers=[]
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.Tanh()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
