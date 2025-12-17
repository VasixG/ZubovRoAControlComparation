import numpy as np
import torch
from invariant_rl.losses.zubov import rk4_step


@torch.no_grad()
def evaluate_invariance_model_based(
    system,
    policy,
    region,
    device,
    N=200,
    max_steps=500,
    dt=0.01,
):
    times = []
    energies = []

    for _ in range(N):
        x0 = region.sample_inside(1, device=device)

        t = 0
        energy = 0.0
        x = x0

        for _ in range(max_steps):
            if x.numel() == 0 or not region.inside(x).all():
                break

            u = policy(x)
            energy += (u**2).sum().item()
            x = rk4_step(system.f_torch, x, u, dt)
            t += 1

        times.append(t)
        energies.append(energy)

    times = np.asarray(times)
    energies = np.asarray(energies)

    return {
        "MTIA": float(times.mean()),
        "P_inv": float((times >= max_steps - 1).mean()),
        "Energy": float(energies.mean()),
    }


@torch.no_grad()
def evaluate_invariance_policy(
    system,
    policy,
    region,
    device,
    N=200,
    max_steps=500,
    dt=0.01,
):
    times = []
    energies = []

    for _ in range(N):

        x = region.sample_inside(1, device=device)

        t = 0
        energy = 0.0

        for _ in range(max_steps):
            if x.numel() == 0 or not region.inside(x).all():
                break

            u = policy(x)
            energy += (u**2).sum().item()
            x = rk4_step(system.f_torch, x, u, dt)
            t += 1

        times.append(t)
        energies.append(energy)

    times = np.asarray(times)
    energies = np.asarray(energies)

    return {
        "MTIA": float(times.mean()),
        "P_inv": float((times >= max_steps - 1).mean()),
        "Energy": float(energies.mean()),
    }
