
import torch
import numpy as np

import yaml




# --------------------------------
# Basic Functions
# --------------------------------

def set_random(seed: int) -> None:
    """
    Reset seed for Numpy and PyTorch

    Args:
        param seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_config(yaml_path) -> dict:
    """
    Load configuration parameters from a YAML file.

    Args:
        yaml_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# --------------------------------
# Image manupulation & Augmentation
# --------------------------------

def aggregate_subcarriers(spectrogram, num_sc_per_rb) -> np.ndarray:
    """
    Aggregate spectrogram subcarriers into resource blocks.

    Args:
        spectrogram (np.ndarray):
            Input spectrogram with shape (num_subcarriers, num_time_steps).

        num_sc_per_rb (int):
            Number of subcarriers per resource block.

    Returns:
        np.ndarray:
            Aggregated spectrogram.
    """
    num_rows, num_cols = spectrogram.shape
    num_aggregated_rows = num_rows // num_sc_per_rb
    aggregated_spectrogram = np.zeros((num_aggregated_rows, num_cols))

    for i in range(num_aggregated_rows):
        start_row = i * num_sc_per_rb
        end_row = start_row + num_sc_per_rb
        aggregated_spectrogram[i, :] = 10*np.log10(np.sum(10**(spectrogram[start_row:end_row, :] / 10), axis=0))

    return aggregated_spectrogram


def fold_image_horizontal(img: torch.Tensor, shift: int) -> torch.Tensor:
    """
    Circularly shift image to the right by `shift` pixels.

    Args:
        img: Tensor of shape (C, H, W)
        shift: Number of pixels to shift right

    Returns:
        Folded image tensor
    """
    return torch.roll(img, shifts=shift, dims=-1)


def flip_x_axis(img: torch.Tensor) -> torch.Tensor:
    """
    Flip image along the x-axis.

    Args:
        img: Tensor of shape (C, H, W)

    Returns:
        Flipped image tensor
    """
    return torch.flip(img, dims=[-1])