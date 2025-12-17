
import torch
from invariant_rl.losses.zubov import pde_loss

class ModelBasedTrainer:
    def __init__(self, system, region, W, policy, opt, device):
        self.system, self.region = system, region
        self.W, self.policy, self.opt = W, policy, opt
        self.device = device

    def step(self, batch):
        xs = self.region.sample_inside(batch, device=self.device)
        loss = pde_loss(self.W, self.policy, self.system, xs)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()
