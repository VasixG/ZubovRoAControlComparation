
import torch

class InvertedPendulum:
    def __init__(self):
        self.nx, self.nu = 2, 1
        self.g, self.l, self.m, self.b = 9.81, 1.0, 1.0, 0.1

    def f_torch(self, x, u):
        th, om, tq = x[:, 0], x[:, 1], u[:, 0]
        th_dot = om
        om_dot = (
            (self.g / self.l) * torch.sin(th)
            + (1 / (self.m * self.l**2)) * tq
            - (self.b / (self.m * self.l**2)) * om
        )
        return torch.stack([th_dot, om_dot], dim=1)
