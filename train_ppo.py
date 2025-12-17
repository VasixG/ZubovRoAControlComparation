import torch
import torch.nn.functional as F
import time

from invariant_rl.dynamics.pendulum import InvertedPendulum
from invariant_rl.regions.ellipse import EllipseRegion

from invariant_rl.dynamics.PPOEnv import PendulumPPOEnv
from invariant_rl.models.PPOModel import PPOModel
from invariant_rl.metrics.invariance import evaluate_invariance_policy
from invariant_rl.utils.logger import MetricLogger


def collect_trajectories(env, model, horizon, gamma, lam):
    states, actions, rewards, dones, logps, values = [], [], [], [], [], []

    x = env.reset()

    for _ in range(horizon):
        with torch.no_grad():
            dist = model.policy(x)

            if torch.rand(1).item() < 0.1:
                action = torch.randint(0, 2, (1,), device=x.device)
            else:
                action = dist.probs.argmax(dim=-1)

            logp = dist.log_prob(action)
            value = model.value(x).squeeze(-1)

        x_next, r, done = env.step(action)

        states.append(x)
        actions.append(action)
        rewards.append(torch.tensor([r], device=x.device))
        dones.append(torch.tensor([float(done)], device=x.device))
        logps.append(logp)
        values.append(value)

        x = x_next
        if done:
            x = env.reset()

    with torch.no_grad():
        values.append(model.value(x).squeeze(-1))

    adv, ret = [], []
    gae = 0.0

    for t in reversed(range(horizon)):
        delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        adv.insert(0, gae)
        ret.insert(0, gae + values[t])

    return (
        torch.cat(states),
        torch.cat(actions),
        torch.cat(logps),
        torch.cat(ret),
        torch.cat(adv),
    )


def ppo_update(
    model,
    optimizer,
    states,
    actions,
    old_logps,
    returns,
    advantages,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
):
    dist = model.policy(states)
    logps = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    ratio = torch.exp(logps - old_logps)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()

    value_loss = F.mse_loss(
        model.value(states).squeeze(-1),
        returns,
    )

    loss = actor_loss + value_coef * value_loss - entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


device = "cuda" if torch.cuda.is_available() else "cpu"

system = InvertedPendulum()
region = EllipseRegion(0.3, 0.5)

env = PendulumPPOEnv(system, region, device=device)
model = PPOModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

logger = MetricLogger("runs/ppo")

train_time = 0.0

gamma = 0.99
lam = 0.95
horizon = 256
epochs = 2000
val_every = 25

ppo_epochs = 5
minibatch_size = 64


for ep in range(epochs):

    t0 = time.time()

    states, actions, old_logps, returns, adv = collect_trajectories(env, model, horizon, gamma, lam)

    states = states.detach()
    actions = actions.detach()
    old_logps = old_logps.detach()
    returns = returns.detach()
    adv = adv.detach()

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N = states.shape[0]
    idx = torch.randperm(N)

    loss = 0.0
    for _ in range(ppo_epochs):
        for start in range(0, N, minibatch_size):
            mb = idx[start : start + minibatch_size]

            loss = ppo_update(
                model,
                optimizer,
                states[mb],
                actions[mb],
                old_logps[mb],
                returns[mb],
                adv[mb],
            )

    train_time += time.time() - t0

    if ep % val_every == 0 or ep == epochs - 1:

        def greedy_policy(x):
            probs = model.policy(x).probs
            action = probs.argmax(dim=-1, keepdim=True)
            return torch.where(
                action == 0,
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
            loss=float(loss),
            metrics=metrics,
        )

        print(
            f"[PPO ep {ep:04d}] "
            f"loss={loss:.3f} | "
            f"MTIA={metrics['MTIA']:.1f} | "
            f"P(inv)={metrics['P_inv']:.2f} | "
            f"Energy={metrics['Energy']:.2f} | "
            f"train_time={train_time:.1f}s"
        )

print("\nPPO training finished.")
torch.save(model.state_dict(), "runs/ppo/ppo_policy.pt")
print("Model saved to runs/ppo/ppo_policy.pt")
