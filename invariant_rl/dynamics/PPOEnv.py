import torch
import numpy as np
from invariant_rl.losses.zubov import rk4_step


class PendulumPPOEnv:
    def __init__(self, system, region, dt=0.01, max_steps=500, device="cpu"):
        self.system = system
        self.region = region
        self.dt = dt
        self.max_steps = max_steps
        self.device = device
        self.reset()

    def reset(self):
        self.x = self.region.sample_inside(1, device=self.device)
        self.t = 0
        return self.x.clone()

    def step(self, action):

        torque = torch.where(
            action == 0,
            torch.tensor([-1.0], device=self.device),
            torch.tensor([+1.0], device=self.device),
        ).unsqueeze(0)

        x = self.x
        x_next = rk4_step(self.system.f_torch, x, torque, self.dt)

        self.x = x_next
        self.t += 1

        theta, omega = x[0, 0], x[0, 1]
        r_stab = -(theta**2 + 0.1 * omega**2)
        r_ctrl = -0.01 * torque.pow(2).sum()
        inside = self.region.inside(x_next).all()
        r_A = 1.0 if inside else -1.0

        reward = r_stab + r_ctrl + r_A
        done = (not inside) or (self.t >= self.max_steps)

        return x_next.clone(), reward.item(), done
