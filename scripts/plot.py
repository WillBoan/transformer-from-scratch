import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .utils import get_run_dir, get_latest_run_dir


class TrainingDashboardPlotter:
    """
    A class to generate a comprehensive training dashboard from a run's metrics.
    """

    def __init__(self, run_id: str | None = None) -> None:
        if run_id:
            self.run_id = run_id
            self.run_dir = get_run_dir(run_id)
        else:
            self.run_dir, self.run_id = get_latest_run_dir()

        self.metrics_file = os.path.join(self.run_dir, "metrics.jsonl")
        self.output_file = os.path.join(self.run_dir, "training_dashboard.png")

        if not os.path.isdir(self.run_dir):
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")
        if not os.path.exists(self.metrics_file):
            raise FileNotFoundError(f"Metrics file not found: {self.metrics_file}")

    def _prepare_data(self) -> pd.DataFrame:
        """Loads and derives all necessary metrics."""
        df = pd.read_json(self.metrics_file, lines=True)

        # Derived Metrics
        df["bpc"] = df["val_loss"] / np.log(2)
        df["overfit_gap"] = df["val_loss"] - df["train_loss"]

        # Cumulative Throughput (Tokens per Second)
        # time_ms is cumulative time since training started
        df["cumulative_time_sec"] = df["time_ms"] / 1000.0
        df["tokens_per_sec"] = df["tokens_processed"] / df["cumulative_time_sec"]

        return df

    def plot(self) -> None:
        """Generates and saves the 2x3 dashboard plot."""
        print(f"Reading metrics from: {self.metrics_file}")
        df = self._prepare_data()
        x = df["tokens_processed"]

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Training Dashboard | Run: {self.run_id}", fontsize=16, weight="bold")

        # --- Panel 1: Loss Curve ---
        ax = axes[0, 0]
        ax.plot(x, df["train_loss"], label="Train Loss", alpha=0.8)
        ax.plot(x, df["val_loss"], label="Val Loss", linestyle="--", linewidth=2)
        ax.set_title("Cross-Entropy Loss")
        ax.set_ylabel("Loss (nats)")
        ax.legend()

        # --- Panel 2: Bits Per Character (BPC) ---
        ax = axes[0, 1]
        ax.plot(x, df["bpc"], color="purple", linewidth=2)
        ax.set_title("Validation Bits Per Character (BPC)")
        ax.set_ylabel("BPC")

        # --- Panel 3: Overfitting Gap ---
        ax = axes[0, 2]
        ax.plot(x, df["overfit_gap"], color="red", alpha=0.8)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title("Overfitting Gap (Val - Train)")
        ax.set_ylabel("Loss Difference")

        # --- Panel 4: Gradient Norm ---
        ax = axes[1, 0]
        ax.plot(x, df["avg_grad_norm"], color="teal", alpha=0.8, label="Average Grad Norm")
        ax.plot(x, df["grad_norm"], color="blue", alpha=0.5, label="Current Grad Norm")
        ax.set_title("Gradient Norm")
        ax.set_ylabel("L2 Norm")
        ax.legend()

        # --- Panel 5: Update-to-Weight Ratio ---
        ax = axes[1, 1]
        ax.plot(x, df["update_to_weight_ratio"], color="orange", alpha=0.8)
        ax.axhline(1e-3, color="black", linestyle="--", label="Target (~1e-3)")
        ax.set_yscale("log")
        ax.set_title("Update-to-Weight Ratio (lm_head)")
        ax.set_ylabel("Ratio (Log Scale)")
        ax.legend()

        # --- Panel 6: Throughput ---
        ax = axes[1, 2]
        ax.plot(x, df["tokens_per_sec"], color="green", alpha=0.8)
        ax.set_title("Cumulative Throughput")
        ax.set_ylabel("Tokens / Second")

        # --- Global Formatting ---
        for ax in axes.flat:
            ax.set_xlabel("Tokens Processed")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            # Format X-axis to show 'K' or 'M' for readability
            ax.xaxis.set_major_formatter(mticker.EngFormatter())

        plt.tight_layout(rect=(0, 0, 1, 0.96))  # Adjust layout to fit suptitle
        plt.savefig(self.output_file, dpi=300, bbox_inches="tight")
        print(f"Dashboard successfully saved to: {self.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a training dashboard for a run.")
    parser.add_argument(
        "run_id",
        nargs="?",
        default=None,
        help="The ID of the run. If not provided, the latest run will be used.",
    )
    args = parser.parse_args()

    try:
        plotter = TrainingDashboardPlotter(run_id=args.run_id)
        plotter.plot()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
