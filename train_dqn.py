import torch
import torch.nn.functional as F
import time
import random

from invariant_rl.dynamics.pendulum import InvertedPendulum
from invariant_rl.regions.ellipse import EllipseRegion
from invariant_rl.dynamics.PPOEnv import PendulumPPOEnv

from invariant_rl.models.DQNModel import DQN
from invariant_rl.utils.replay_buffer import ReplayBuffer
from invariant_rl.metrics.invariance import evaluate_invariance_policy
from invariant_rl.utils.logger import MetricLogger


device = "cuda" if torch.cuda.is_available() else "cpu"

gamma = 0.99
batch_size = 128
buffer_capacity = 100_000
lr = 1e-3

epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 50_000

target_update_every = 1000
max_steps_per_episode = 500
epochs = 3000
val_every = 25


system = InvertedPendulum()
region = EllipseRegion(0.3, 0.5)

env = PendulumPPOEnv(system, region, device=device)

q = DQN().to(device)
q_target = DQN().to(device)
q_target.load_state_dict(q.state_dict())

optimizer = torch.optim.Adam(q.parameters(), lr=lr)
buffer = ReplayBuffer(buffer_capacity)

logger = MetricLogger("runs/dqn")

train_time = 0.0
global_step = 0


for ep in range(epochs):
    loss = None
    t0 = time.time()
    x = env.reset()

    for t in range(max_steps_per_episode):
        global_step += 1

        eps = (
            epsilon_end
            + (epsilon_start - epsilon_end)
            * torch.exp(torch.tensor(-global_step / epsilon_decay)).item()
        )

        if random.random() < eps:
            action = torch.randint(0, 2, (1, 1), device=device)
        else:
            with torch.no_grad():
                qvals = q(x)
                action = qvals.argmax(dim=-1, keepdim=True)

        x_next, r, done_env = env.step(action)

        buffer.push(
            x.detach(),
            action.detach(),
            torch.tensor([[r]], device=device),
            x_next.detach(),
            torch.tensor([[float(done_env)]], device=device),
        )

        x = x_next

        if len(buffer) >= batch_size:
            s, a, r, s_next, done = buffer.sample(batch_size)

            q_sa = q(s).gather(1, a)
            with torch.no_grad():
                q_next = q_target(s_next).max(dim=1, keepdim=True)[0]
                target = r + gamma * (1 - done) * q_next

            loss = F.mse_loss(q_sa, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
            optimizer.step()

        if global_step % target_update_every == 0:
            q_target.load_state_dict(q.state_dict())

        if done_env:
            break

    train_time += time.time() - t0

    if ep % val_every == 0 or ep == epochs - 1:

        def greedy_policy(x):
            with torch.no_grad():
                a = q(x).argmax(dim=-1, keepdim=True)
            return torch.where(
                a == 0,
                torch.tensor([[-1.0]], device=x.device),
                torch.tensor([[+1.0]], device=x.device),
            )

        metrics = evaluate_invariance_policy(
            system=system,
            policy=greedy_policy,
            region=region,
            device=device,
            N=128,
            max_steps=500,
            dt=0.01,
        )

        logger.log(
            epoch=ep,
            train_time=train_time,
            loss=float(loss.item()) if loss is not None else 0.0,
            metrics=metrics,
        )

        loss_str = f"{loss.item():.3f}" if loss is not None else "n/a"

        print(
            f"[DQN ep {ep:04d}] "
            f"loss={loss_str} | "
            f"MTIA={metrics['MTIA']:.1f} | "
            f"P(inv)={metrics['P_inv']:.2f} | "
            f"Energy={metrics['Energy']:.2f} | "
            f"train_time={train_time:.1f}s"
        )


torch.save(q.state_dict(), "runs/dqn/dqn_policy.pt")
print("Model saved to runs/dqn/dqn_policy.pt")
