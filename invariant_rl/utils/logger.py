import csv
from pathlib import Path


class MetricLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "metrics.csv"

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_time_sec",
                    "loss",
                    "MTIA",
                    "P_inv",
                    "Energy",
                ]
            )

    def log(self, epoch, train_time, loss, metrics):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    train_time,
                    loss,
                    metrics["MTIA"],
                    metrics["P_inv"],
                    metrics["Energy"],
                ]
            )
