# pyright: reportUnknownMemberType=false

import pandas as pd
import matplotlib.pyplot as plt
import config as cfg


def plot_loss_curve(log_file: str, output_file: str) -> None:
    df = pd.read_json(log_file, lines=True)
    plt.plot(df["iter_num"], df["train_loss"], label="Train Loss")
    plt.plot(df["iter_num"], df["val_loss"], label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(output_file)


if __name__ == "__main__":
    plot_loss_curve(cfg.TRAINING_LOG_FILE, "examples/loss_curve.png")
