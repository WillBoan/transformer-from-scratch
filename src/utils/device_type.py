"""
A simple data type for representing the device type to use for training,
with support for "auto" detection.
"""

from typing import Literal
import torch


type DeviceTypeConfig = Literal["auto", "cuda", "mps", "cpu"]
type DeviceType = Literal["cuda", "mps", "cpu"]


def get_device_type(
    device_type_config: DeviceTypeConfig,
) -> DeviceType:
    """
    Determines the device type to use based on the input string and system capabilities.

    Args:
        device_type_config: A string specifying the desired device type.
            Can be "auto", "cuda", "mps", or "cpu".

    Returns:
        A string indicating the device type to use: "cuda", "mps", or "cpu".
    """
    if device_type_config == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    elif device_type_config in ["cuda", "mps", "cpu"]:
        return device_type_config
    else:
        raise ValueError(
            f"Invalid device string: {device_type_config}. "
            f"Must be 'auto', 'cuda', 'mps', or 'cpu'."
        )
