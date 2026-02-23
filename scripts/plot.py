import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .utils import get_run_dir, get_latest_run_dir


class LossCurvePlotter:
    """
    A class to plot training and validation loss curves from a training run.
    """

    def __init__(self, run_id: str | None = None) -> None:
        """
        Initializes the plotter for a specific run.

        Args:
            run_id (str): The ID of the training run
                (e.g., '2026-02-22_15-30-00_shakespeare_v1').

        Raises:
            FileNotFoundError: If the run directory or metrics file does not exist.
        """
        if run_id:
            self.run_id = run_id
            self.run_dir = get_run_dir(run_id)
        else:
            self.run_dir, self.run_id = get_latest_run_dir()

        self.metrics_file = os.path.join(self.run_dir, "metrics.jsonl")
        self.output_file = os.path.join(self.run_dir, "loss_curve.png")

        if not os.path.isdir(self.run_dir):
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")
        if not os.path.exists(self.metrics_file):
            raise FileNotFoundError(f"Metrics file not found: {self.metrics_file}")

    def plot(self) -> None:
        """
        Generates and saves the loss curve plot.
        """
        print(f"Reading metrics from: {self.metrics_file}")
        df = pd.read_json(self.metrics_file, lines=True)

        plt.style.use("seaborn-v0_8-whitegrid")
        _fig, ax = plt.subplots(figsize=(12, 7))

        ax.plot(df["iter_num"], df["train_loss"], label="Train Loss", alpha=0.9)
        ax.plot(
            df["iter_num"],
            df["val_loss"],
            label="Validation Loss",
            linestyle="--",
            linewidth=2,
        )

        # Formatting
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(
            f"Training & Validation Loss\nRun: {self.run_id}",
            fontsize=14,
            weight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Use a formatter to avoid scientific notation on y-axis
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.tick_params(axis="both", which="major", labelsize=10)

        plt.savefig(self.output_file, dpi=300, bbox_inches="tight")
        print(f"Plot successfully saved to: {self.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot loss curves for a specific training run."
    )
    parser.add_argument(
        "run_id",
        nargs="?",
        default=None,
        help=(
            "The ID of the run (e.g., '2026-02-20_22-11-06_shakespeare_v1'). "
            "If not provided, the latest run will be used."
        ),
    )
    args = parser.parse_args()

    try:
        plotter = LossCurvePlotter(run_id=args.run_id)
        plotter.plot()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)
