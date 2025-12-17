import time
import torch
import os


from invariant_rl.dynamics.pendulum import InvertedPendulum
from invariant_rl.regions.ellipse import EllipseRegion
from invariant_rl.models.lyapunov import SigmoidGeometryAwareW
from invariant_rl.models.policies import MLPPolicy
from invariant_rl.trainers.model_based_trainer import ModelBasedTrainer

from invariant_rl.metrics.invariance import evaluate_invariance_model_based
from invariant_rl.utils.logger import MetricLogger


device = "cuda" if torch.cuda.is_available() else "cpu"

system = InvertedPendulum()
region = EllipseRegion(0.3, 0.5)

W = SigmoidGeometryAwareW([2, 64, 64, 1], region).to(device)
policy = MLPPolicy([2, 64, 64, 1]).to(device)

optimizer = torch.optim.Adam(
    list(W.parameters()) + list(policy.parameters()),
    lr=1e-3,
)

trainer = ModelBasedTrainer(
    system=system,
    region=region,
    W=W,
    policy=policy,
    opt=optimizer,
    device=device,
)

logger = MetricLogger("runs/model_based")

epochs = 500
batch_size = 256
val_every = 25

train_time = 0.0


for ep in range(epochs):

    t0 = time.time()
    loss = trainer.step(batch_size)
    train_time += time.time() - t0

    if ep % val_every == 0 or ep == epochs - 1:
        metrics = evaluate_invariance_model_based(
            system=system,
            policy=policy,
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
            f"[ep {ep:04d}] "
            f"loss={loss:.3e} | "
            f"MTIA={metrics['MTIA']:.1f} | "
            f"P(inv)={metrics['P_inv']:.2f} | "
            f"Energy={metrics['Energy']:.2f} | "
            f"train_time={train_time:.1f}s"
        )

save_dir = "runs/model_based/checkpoints"
os.makedirs(save_dir, exist_ok=True)

torch.save(
    {
        "policy": policy.state_dict(),
        "W": W.state_dict(),
    },
    os.path.join(save_dir, "final.pt"),
)
