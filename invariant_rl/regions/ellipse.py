import torch
from .base import Region2D


class EllipseRegion(Region2D):
    def __init__(self, a=1.0, b=0.5, tol=1e-3):
        self.a, self.b, self.tol = a, b, tol

    def inside(self, x):
        return (x[..., 0] / self.a) ** 2 + (x[..., 1] / self.b) ** 2 < 1.0

    def sample_inside(self, N, device=None):
        lo = torch.tensor([-self.a, -self.b], device=device)
        hi = torch.tensor([self.a, self.b], device=device)
        pts = []
        while len(pts) < N:
            c = lo + (hi - lo) * torch.rand((N, 2), device=device)
            m = self.inside(c)
            if m:
                pts += [c[m]]
        return torch.cat(pts)[:N]

    def sample_on_boundary(self, N, device=None):
        th = 2 * torch.pi * torch.rand((N,), device=device)
        return torch.stack([self.a * torch.cos(th), self.b * torch.sin(th)], dim=1)
