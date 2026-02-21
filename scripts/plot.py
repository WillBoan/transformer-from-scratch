# pyright: reportUnknownMemberType=false
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Define the parent directory for checkpoints relative to this script's location
# scripts/../checkpoints -> checkpoints/
CHECKPOINT_PARENT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "checkpoints")
)


def find_latest_run_dir(parent_dir: str) -> str | None:
    """Finds the most recent run directory based on directory name."""
    dirs = [
        os.path.join(parent_dir, d)
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)


def plot_loss_curve(run_dir: str, output_file: str) -> None:
    """
    Plots the training and validation loss curves from a metrics file.

    Args:
        run_dir (str): The directory of the training run.
        output_file (str): The path to save the output plot image.
    """
    metrics_file = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Metrics file not found in {run_dir}")

    df = pd.read_json(metrics_file, lines=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    _fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["iter_num"], df["train_loss"], label="Train Loss")
    ax.plot(df["iter_num"], df["val_loss"], label="Validation Loss", linestyle="--")

    # Formatting
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training and Validation Loss Curves", fontsize=14, weight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Use a formatter to avoid scientific notation on y-axis if numbers are small
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot loss curves from a training run."
    )
    parser.add_argument(
        "run_id",
        nargs="?",
        default=None,
        help=(
            "The ID of the run to plot (e.g., '2026-02-20_22-11-06_shakespeare_v1'). "
            "If not provided, the latest run will be used."
        ),
    )
    parser.add_argument(
        "--out",
        default="examples/loss_curve.png",
        help="Path to save the output plot image.",
    )
    args = parser.parse_args()

    if args.run_id:
        run_directory = os.path.join(CHECKPOINT_PARENT_DIR, args.run_id)
    else:
        print("No run_id provided, attempting to find the latest run...")
        run_directory = find_latest_run_dir(CHECKPOINT_PARENT_DIR)
        if not run_directory:
            print("Error: No training runs found in 'checkpoints/'.")
            exit(1)
        print(f"Found latest run: {os.path.basename(run_directory)}")

    try:
        plot_loss_curve(run_directory, args.out)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
