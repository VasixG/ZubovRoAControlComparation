import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path

from invariant_rl.dynamics.pendulum import InvertedPendulum
from invariant_rl.regions.ellipse import EllipseRegion
from invariant_rl.models.DQNModel import DQN
from invariant_rl.losses.zubov import rk4_step


checkpoint_path = "runs/dqn/dqn_policy.pt"
gif_path = "runs/dqn/animation.gif"

dt = 0.01
T = 800
device = "cuda" if torch.cuda.is_available() else "cpu"

system = InvertedPendulum()
region = EllipseRegion(0.3, 0.5)

q = DQN().to(device)
q.load_state_dict(torch.load(checkpoint_path, map_location=device))
q.eval()

theta0 = 0.1
omega0 = 0.0

x = torch.tensor(
    [[theta0, omega0]],
    dtype=torch.float32,
    device=device,
)

frames = []

for t in range(T):
    theta = x[0, 0].item()

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, np.sin(theta)], [0, np.cos(theta)], "o-", lw=3)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"DQN | t = {t}")

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.renderer.buffer_rgba())
    frames.append(frame)
    plt.close(fig)

    with torch.no_grad():
        qvals = q(x)
        action = qvals.argmax(dim=-1, keepdim=True)

        u = torch.where(
            action == 0,
            torch.tensor([[-1.0]], device=device),
            torch.tensor([[+1.0]], device=device),
        )

        x = rk4_step(system.f_torch, x, u, dt)

    if not region.inside(x).all():
        print(f"Exited region A at t = {t}")
        break


Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
imageio.mimsave(gif_path, frames, fps=30)

print(f"DQN animation saved to {gif_path}")
