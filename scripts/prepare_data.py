import os
import requests
import numpy as np
from src.utils.tokenizer import CharTokenizer


# --- Configuration ---
DATASET_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DATA_DIR = "data/tinyshakespeare"
RAW_DATA_PATH = os.path.join(DATA_DIR, "input.txt")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")
TRAIN_PATH = os.path.join(DATA_DIR, "train.bin")
VAL_PATH = os.path.join(DATA_DIR, "val.bin")
SPLIT_RATIO = 0.9


def prepare_data():
    """
    Downloads the Tiny Shakespeare dataset, creates a vocabulary,
    tokenizes the data, and saves it into training and validation binary files.
    This script should be run once before any training.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        # --- 1. Download Data ---
        if not os.path.exists(RAW_DATA_PATH):
            print(f"Downloading dataset to {RAW_DATA_PATH}...")
            with open(RAW_DATA_PATH, "w", encoding="utf-8") as f:
                f.write(requests.get(DATASET_URL).text)

        with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Dataset loaded. Length: {len(text)} characters.")

        # --- 2. Build and Save Vocabulary ---
        tokenizer = CharTokenizer()
        tokenizer.fit(text)
        tokenizer.save(VOCAB_PATH)
        print(
            f"Vocabulary built and saved to {VOCAB_PATH}. "
            f"Vocab size: {tokenizer.vocab_size}"
        )

        # --- 3. Tokenize and Split Data ---
        data = np.array(tokenizer.encode(text), dtype=np.uint16)
        n = int(SPLIT_RATIO * len(data))
        train_data = data[:n]
        val_data = data[n:]

        # --- 4. Save as Binary Files ---
        train_data.tofile(TRAIN_PATH)
        val_data.tofile(VAL_PATH)
        print("Data tokenized and split.")
        print(f"Training data saved to {TRAIN_PATH} ({len(train_data)} tokens)")
        print(f"Validation data saved to {VAL_PATH} ({len(val_data)} tokens)")
        print("\nData preparation complete.")
        print("You can now run the training script.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        # Clean up partial file if it exists
        if os.path.exists(RAW_DATA_PATH):
            os.remove(RAW_DATA_PATH)
            print(f"Removed partially downloaded file: {RAW_DATA_PATH}")
        return
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        return


if __name__ == "__main__":
    prepare_data()
