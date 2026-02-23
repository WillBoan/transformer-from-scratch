from typing import Any, Literal
import os

# import sys
import json
import argparse
import torch

# Add the src directory to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config as current_cfg
from model import Transformer
from tokenizer import CharTokenizer
from checkpoint_manager import CheckpointManager
from .utils import get_run_dir, get_latest_run_dir


class TextGenerator:
    """
    A class to handle loading a trained Transformer model and generating text.
    Ensures that the model is instantiated with the exact configuration and
    vocabulary artifact it was trained with.
    """

    run_id: str
    run_dir: str
    device: torch.device
    run_cfg: dict[str, Any]
    tokenizer: CharTokenizer
    model: Transformer

    def __init__(
        self,
        run_id: str | None = None,
        checkpoint_type: Literal["best", "latest"] = "best",
    ) -> None:
        if run_id:
            self.run_id = run_id
            self.run_dir = get_run_dir(run_id)
        else:
            self.run_dir, self.run_id = get_latest_run_dir()

        self.device = torch.device(current_cfg.DEVICE)

        # 1. Load the historical config used during training
        config_path = os.path.join(self.run_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.run_cfg = json.load(f)

        # 2. Load the trained vocabulary artifact
        vocab_path = os.path.join(self.run_dir, "vocab.json")
        self.tokenizer = CharTokenizer()
        print(f"Loading tokenizer vocabulary from {vocab_path}...")
        self.tokenizer.load(vocab_path)

        # 3. Instantiate the model using historical config values
        print("Instantiating model from historical config...")
        self.model = Transformer(
            vocab_size=self.tokenizer.vocab_size,
            n_layer=self.run_cfg["N_LAYER"],
            n_head=self.run_cfg["N_HEAD"],
            block_size=self.run_cfg["BLOCK_SIZE"],
            n_embd=self.run_cfg["N_EMBD"],
            dropout=0.0,  # Force dropout to 0 for inference
        )

        # 4. Load the checkpoint weights
        cm = CheckpointManager(self.run_dir)
        state = cm.load(checkpoint_type)
        if state is None:
            raise FileNotFoundError(
                f"Could not load '{checkpoint_type}' checkpoint from {self.run_dir}"
            )

        self.model.load_state_dict(state.model_state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully to {self.device}.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> str:
        """Generates text continuing from the given prompt."""
        # Encode prompt
        idx_list = self.tokenizer.encode(prompt)
        idx = torch.tensor([idx_list], dtype=torch.long, device=self.device)

        # Generate
        with torch.no_grad():
            # Note: Requires the temperature parameter update in model.py
            out_idx = self.model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        # Decode
        generated_text = self.tokenizer.decode(out_idx[0].tolist())
        return generated_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from a trained model.")
    parser.add_argument(
        "run_id",
        nargs="?",
        default=None,
        help=(
            "The ID of the run (e.g., '2026-02-20_22-11-06_shakespeare_v1'). "
            "If not provided, the latest run will be used."
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="O Romeo, Romeo! wherefore art thou Romeo?",
        help="Starting text prompt",
    )
    parser.add_argument(
        "--tokens", type=int, default=200, help="Number of tokens to generate"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="best",
        choices=["best", "latest"],
        help="Which checkpoint to load",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Higher values increase diversity, lower values make it more predictable.",
    )

    args = parser.parse_args()

    try:
        generator = TextGenerator(run_id=args.run_id, checkpoint_type=args.ckpt)

        print("\n--- Prompt ---")
        print(args.prompt)
        print("\n--- Generated Text ---")

        output = generator.generate(
            prompt=args.prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
        )
        print(output)
        print("\n----------------------")

    except Exception as e:
        print(f"Error: {e}")
