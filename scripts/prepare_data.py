import os
import requests


# URL for the Tiny Shakespeare dataset
DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# Path to save the dataset
SAVE_DIR = "data/tinyshakespeare"
FILE_PATH = os.path.join(SAVE_DIR, "input.txt")


def prepare_data():
    """
    Downloads the Tiny Shakespeare dataset, with robust error handling.
    """
    print("Preparing data...")

    # Create the directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Check if the file already exists
    if os.path.exists(FILE_PATH):
        print(f"Dataset already exists at {FILE_PATH}")
        return

    # Download the dataset
    try:
        print(f"Downloading dataset from {DATASET_URL}...")
        # Add a timeout to the request
        response = requests.get(DATASET_URL, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Save the dataset to the file
        with open(FILE_PATH, "w", encoding="utf-8") as f:
            f.write(response.text)

        print(f"Dataset saved to {FILE_PATH}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        # Clean up partial file if it exists
        if os.path.exists(FILE_PATH):
            os.remove(FILE_PATH)
            print(f"Removed partially downloaded file: {FILE_PATH}")
        return

    print("Data preparation complete.")

if __name__ == "__main__":
    prepare_data()