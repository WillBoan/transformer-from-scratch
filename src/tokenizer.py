import json
import os


class TextTokenizer:
    """
    A base text tokenizer class.

    Provides common functionality for encoding and decoding text.

    Attributes:
        _stoi (dict[str, int]): Mapping from characters to integers (for encoding).
        _itos (dict[int, str]): Mapping from integers to characters (for decoding).
        vocab_size (int): The number of unique characters in the vocabulary.
    """

    _stoi: dict[str, int]  # Mapping from characters to integers, for encoding
    _itos: dict[int, str]  # Mapping from integers to characters, for decoding
    vocab_size: int

    def __init__(self) -> None:
        self._stoi: dict[str, int] = {}
        self._itos: dict[int, str] = {}
        self.vocab_size: int = 0

    def fit(self, text: str) -> None:
        """
        Builds the vocabulary from the given text.

        NOTE: Must be implemented by subclasses.
        """
        raise NotImplementedError("fit() method must be implemented by subclasses")

    def encode(self, s: str) -> list[int]:
        """Encode a string into a list of integers."""
        if not self._stoi:
            raise ValueError(
                "Tokenizer vocabulary is empty. Call fit() or load() first."
            )
        return [self._stoi[c] for c in s]

    def decode(self, int_list: list[int]) -> str:
        """Decode a list of integers into a string."""
        if not self._itos:
            raise ValueError(
                "Tokenizer vocabulary is empty. Call fit() or load() first."
            )
        return "".join([self._itos[i] for i in int_list])

    def save(self, path: str) -> None:
        """Saves the vocabulary to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self._stoi, "itos": self._itos}, f, indent=2)

    def load(self, path: str) -> None:
        """Loads the vocabulary from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._stoi = data["stoi"]
        # JSON keys are always strings, so we must cast the itos keys back to integers
        self._itos = {int(k): v for k, v in data["itos"].items()}
        self.vocab_size = len(self._itos)


class CharTokenizer(TextTokenizer):
    """A character-level tokenizer. Inherits from TextTokenizer."""

    def fit(self, text: str) -> None:
        """Builds the vocabulary from the given text."""
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self._stoi = {ch: i for i, ch in enumerate(chars)}
        self._itos = {i: ch for i, ch in enumerate(chars)}
