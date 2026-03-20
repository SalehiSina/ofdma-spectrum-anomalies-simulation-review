"""This script converts custom dataset samples into images for further processing.

In addition, it generates a labels file containing the jammer type, the number of
authorized transmitters, and the SNR for each sample. Also, a metadata file
containing the minimum and maximum values of the spectrograms is generated to enable
rescaling of the spectrograms back to their original values.
"""

__docformat__ = "numpy"

import os
import sys

import argparse
import compress_pickle as cpkl
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# import own modules
_repo_name = "ofdma-spectrum-anomalies-simulation"
_module_path = __file__[: __file__.find(_repo_name) + len(_repo_name)]
sys.path.append(os.path.abspath(_module_path))

from src.utils.ofdm_utils import get_total_allocated_resources
from src.utils.data_utils import (
    get_datapath,
    get_spectrogram_img_filenamename,
    get_resource_alloc_img_filenamename,
)

# initialize the datasets path
_datapath = get_datapath(_repo_name)


def find_min_max_values(filenames):
    """Finding the minimum and maximum values of the spectrograms
    across all samples in the dataset (PT and DT) to ensure consistent
    scaling across all images.

    Parameters
    ----------
    filenames : list
        List of filenames containing the samples.

    Returns
    -------
    min_val : float
        Minimum value found in the spectrograms.
    max_val : float
        Maximum value found in the spectrograms.
    """

    p_bar = tqdm(total=len(filenames), desc="Finding min and max values")

    min_val = np.inf
    max_val = -np.inf
    for filename in filenames:
        with open(filename, "rb") as f:
            samples = cpkl.load(f)
        for sample in samples:
            for su_idx in sample.spectrograms:
                min_val = min(min_val, np.min(sample.spectrograms[su_idx]))
                max_val = max(max_val, np.max(sample.spectrograms[su_idx]))
        del samples  # free memory
        p_bar.update(1)

    p_bar.close()

    print(f"Min value: {min_val}, Max value: {max_val}")
    return min_val, max_val


def samples_to_imgs_and_labels(
    data_path, dataset_nr, filename, total_sample_idx, labels, min_val, max_val
):
    """Creates images for easier processing of the dataset from the custom data format.

    Parameters
    -----
    data_path : str
        Path to the datasets folder.
    dataset_nr : int
        Dataset number.
    filename : str
        Filename of the samples in custom format (to process).
    total_sample_idx : int
        Index of the current sample.
    labels : dict
        Dictionary containing the labels (jammer type and number
        of authorized transmitters).
    min_val : float
        Minimum value found in the spectrograms (for consistent scaling).
    max_val : float
        Maximum value found in the spectrograms (for consistent scaling).

    Returns
    -------
    total_sample_idx : int
        Updated index of the current sample.
    labels : dict
        Updated dictionary containing the labels (jammer type,
        jammer power, number of authorized transmitters, SNR).
        Only updated for PT type samples! For DT type samples, the
        it is returned as is.
    """

    filename = os.path.join(
        data_path,
        f"{dataset_nr}",
        "custom",
        filename,
    )

    with open(filename, "rb") as f:
        samples = cpkl.load(f)

    target_path = os.path.join(data_path, f"{dataset_nr}", "images")

    for sample in samples:
        # get label (jammer type)
        if len(sample.jammers) > 0:
            labels["jammer_type"].append(sample.jammers[0].type)
            labels["jammer_power"].append(sample.jammers[0].transmit_power)
            labels["jammer_location"].append(sample.jammers[0].location)
        else:
            labels["jammer_type"].append("no jammer")
            labels["jammer_power"].append(np.nan)
            labels["jammer_location"].append(np.nan)
        labels["num_legitimate_transmitters"].append(len(sample.transmitters))
        labels["snr"].append(sample.snr)

        # get total allocated resources (for all authorized TX) and save as image
        total_allocated_resources = get_total_allocated_resources(sample, 12, 14)

        resource_img = Image.fromarray(total_allocated_resources)
        resource_img.save(
            os.path.join(
                target_path,
                get_resource_alloc_img_filenamename(total_sample_idx),
            )
        )

        # save the spectrograms as image
        for su_idx in sample.spectrograms:
            su_spectrogram = sample.spectrograms[su_idx]
            # conversion to the expected range and type for PNG output
            su_spectrogram = (
                (su_spectrogram - min_val) / (max_val - min_val) * 255
            ).astype(np.uint8)
            spectrogram_img = Image.fromarray(su_spectrogram)
            spectrogram_img.save(
                os.path.join(
                    target_path,
                    get_spectrogram_img_filenamename(total_sample_idx, su_idx),
                )
            )

        total_sample_idx += 1

    return total_sample_idx, labels


def parse_arguments():
    """Parse command line arguments.

    Returns
    -------
    args : Namespace
        Parsed command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Generate a dataset for a given scene and configuration."
    )

    parser.add_argument(
        "-d",
        "--dataset-number",
        default=0,
        type=int,
        required=False,
        help="Dataset number to process.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_arguments()

    dataset_nr = int(args.dataset_number)
    print(f"Processing dataset number: {dataset_nr}")

    total_sample_idx = 0
    labels = {
        "jammer_type": [],
        "jammer_power": [],
        "jammer_location": [],
        "num_authorized_transmitters": [],
        "snr": [],
    }

    custom_files_dir = os.path.join(_datapath, f"{dataset_nr}", "custom")
    filenames = glob(
        os.path.join(
            custom_files_dir,
            f"samples-*.gz",
        )
    )

    target_path = os.path.join(_datapath, f"{dataset_nr}", "images")

    # create folder for images or clear files the folder
    if os.path.exists(target_path):
        for file in os.listdir(target_path):
            os.remove(os.path.join(target_path, file))
    else:
        os.makedirs(target_path)

    min_val, max_val = find_min_max_values(filenames)

    # save the minimum and maximum values, so that original values can be restored
    pd.DataFrame({"min_val": [min_val], "max_val": [max_val]}).to_csv(
        os.path.join(_datapath, f"{dataset_nr}", "spectrogram_min_max.csv"),
        index=False,
    )

    for file_idx, filename in enumerate(tqdm(filenames)):

        total_sample_idx, labels = samples_to_imgs_and_labels(
            _datapath,
            dataset_nr,
            filename,
            total_sample_idx,
            labels,
            min_val,
            max_val,
        )

    pd.DataFrame(labels).to_csv(
        os.path.join(_datapath, f"{dataset_nr}", "labels.csv"),
        index=False,
    )
