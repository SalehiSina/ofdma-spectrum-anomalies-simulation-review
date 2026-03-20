# OFDMA Spectrum Anomaly Simulation

## Introduction

## Scene Generation

The key of this simulation is a scene, in which ray tracing is executed to obtain channel frequency responses (CFRs) between transmitters and sensing units. The corresponding scene is created with a Python script in Blender, which creates the scene and exports it in a format that can be loaded by sionna.

### Requirements

* Blender 4.2.19 LTS
* Mitsuba Add-on for Blender 0.4.0 (follow the installation instructions [here](https://github.com/mitsuba-renderer/mitsuba-blender))

### Usage

Open Blender and go to the Scripting workspace. Open the `blender-python/create_scenario_with_obstacles.py`. It creates a scene according to the specifications in `blender-python\conf\scene_attributes.yaml`. The scene is exported to the `scenes` directory in the repository root. The exported scene can then be loaded by sionna for ray tracing.

## Documentation

```bash
pydoctor --project-name="ofdma-spectrum-anomalies-simulation" --make-html src/
```