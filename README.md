# OFDMA Spectrum Anomaly Simulation

## Introduction

This repository contains code for simulating spectrum anomalies in OFDMA systems and generate spectrograms to work on the problem of spectrum anomaly detection. The simulation is based on ray tracing, which is executed in a scene created with Blender and the Mitsuba Add-on. The generated spectrograms can be used to train machine learning models for anomaly detection.

## Scene Generation

The key of this simulation is a scene, in which ray tracing is executed to obtain channel frequency responses (CFRs) between transmitters and sensing units. The corresponding scene is created with a Python script in Blender, which creates the scene and exports it in a format that can be loaded by sionna.

### Requirements

* Blender 4.2.19 LTS
* Mitsuba Add-on for Blender 0.4.0 (follow the installation instructions [here](https://github.com/mitsuba-renderer/mitsuba-blender))

### Usage

Open Blender and go to the Scripting workspace. Open the `blender-python/create_scenario_with_obstacles.py`. It creates a scene according to the specifications in `blender-python\conf\scene_attributes.yaml`. The scene is exported to the `scenes` directory in the repository root. The exported scene can then be loaded by sionna for ray tracing.

## Data Generation

### Requirements

The simulation workflow has been tested with Python 3.12 and Sionna 1.2.1.

### Create custom intermediate data

Entry path for the simulation is the script `src/dataset_generation.py`, which executes the ray tracing and creates custom intermediate data. Simulation parameters can be configured in the file `src\conf\dataset_generation.yaml`. The generated data is stored in the directory that is specified in the file `datapath.txt` which is located in the repository root. In this directory, a subdirectory with the specified dataset number is created, in which the custom intermediate data is stored in a subdirectory named `custom`.

### Create image data and labels

To further utilize the data, the script `src/create_image_data_and_labels.py` can be executed, which creates spectrogram images and labels from the custom intermediate data. The generated spectrograms are stored in the same directory as the custom intermediate data in the subdirectory `images`. Two types of images are generated:
* Spectrograms per sensing unit (SU): Those are normalized spectrograms, and the required values are stored in the file `spectrogram_min_max.csv` which is in the dataset folder. The filename format is `spectrogram-{sample_idx}-{su_idx}.png`.
* Resource allocation images: Those are binary images that show the resource allocation of the transmitters. The filename format is `alloc_res-{sample_idx}.png`.

In addition, in the same directory, a file named `labels.csv` is created, which contains the following labels for each sample:
* `jammer_type`: Type of the jammer (if there is no jammer, the value is "no jammer")
* `jammer_power`: Transmit power of the jammer in dBm (if there is no jammer, the value is NaN)
* `jammer_location`: Location of the jammer (if there is no jammer, the value is NaN)
* `num_legitimate_transmitters`: Number of legitimate transmitters in the scene
* `snr`: Signal-to-noise ratio of the received signal in dB

## Documentation

Generate the documentation from the root of the repository with the following command:

```bash
pydoctor --config pydoctor.ini
```