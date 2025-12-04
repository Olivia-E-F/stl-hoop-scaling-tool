# STL Hoop Scaling Tool

This script scales only the hoop portion of a wall-mount STL while keeping the mounting base and connection geometry unchanged.

## Overview

The tool loads a binary STL, identifies the hoop region using simple geometric filters, computes the hoop center in the Y–Z plane, and applies a uniform scaling factor to that region only. This enables resizing the hoop for different ball diameters while preserving alignment with the original mount.

During execution, the script prints:
* Original vs. target hoop radius
* Observed scale factor
* Approximate real-world dimensions (CAD units → inches)
* Hoop center coordinates
* Number of vertices scaled
* Final output filename
* Original STL Attribution

### This script modifies the following file (not included in this repository):
"Sports Ball Wall Mount for Football / Soccer / Basketball"
https://www.printables.com/model/708264-sports-ball-wall-mount-for-football-soccer-basketb
All rights belong to the original creator.

## Usage

### Activate the virtual environment:
```python
source .venv/bin/activate
```


### Run the scaling script:
```python
python scale_hoop.py
```


A new scaled STL file will be created in the project directory.
The filename includes the target hoop size and applied scale factor.

## Configuration

Edit the parameters at the bottom of scale_hoop.py:
* input_path — input STL filename
* output_path — output STL filename
* target_radius_in or scale_factor — depending on your workflow

### versioning settings (optional, explained below)
Automatic Versioning (Optional)
If enabled, the script will automatically append a version suffix to the output file.
For example:

```python
hoop_3.00in_scale0.288_v001.stl
hoop_3.00in_scale0.288_v002.stl
```


#### How it works

The script scans the project directory for existing files matching the pattern.
It finds the highest version number.
It increments it for the next output.

#### Enabling or disabling versioning
At the bottom of scale_hoop.py, set:
```python
use_versioning = True  # or False
```
If disabled, the output filename is written exactly as provided in output_path.

### Dependencies

Listed in requirements.txt:
numpy

## Project Structure
```
stl-hoop-scaling-tool
├── scale_hoop.py
├── requirements.txt
├── .gitignore
└── (generated .stl files)
```