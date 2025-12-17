import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path

from invariant_rl.dynamics.pendulum import InvertedPendulum
from invariant_rl.regions.ellipse import EllipseRegion
from invariant_rl.models.PPOModel import PPOModel
from invariant_rl.losses.zubov import rk4_step


checkpoint_path = "runs/ppo/ppo_policy.pt"
gif_path = "runs/ppo/animation.gif"

dt = 0.01
T = 800
device = "cuda" if torch.cuda.is_available() else "cpu"


system = InvertedPendulum()
region = EllipseRegion(0.3, 0.5)

model = PPOModel().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()


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
    ax.set_title(f"PPO | t = {t}")

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.renderer.buffer_rgba())
    frames.append(frame)

    plt.close(fig)

    with torch.no_grad():
        probs = model.policy(x).probs
        action = probs.argmax(dim=-1, keepdim=True)

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

print(f"PPO animation saved to {gif_path}")
