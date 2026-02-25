import hydra

from src.config import TransformerConfig
from src.trainers.base_trainer import Trainer


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(cfg: TransformerConfig) -> None:
    """
    Main entrypoint for training the Transformer model.

    Hydra decorates this function to handle configuration management.

    It will:
    1. Load the YAML configuration specified in `config_path` and `config_name`.
    2. Validate and instantiate the `TransformerConfig` dataclass with the values.
    3. Pass the populated `cfg` object to this function.
    """
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
