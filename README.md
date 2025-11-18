# STL Hoop Scaling Tool

This script scales down only the hoop part of the wall mount STL. The mounting base and connection points stay the same.

## Overview

This tool reads a binary STL file, identifies the hoop region using simple geometric filters, computes the hoop’s center in the Y–Z plane, and applies a uniform scaling factor to that region only. This allows the hoop to be resized for different ball diameters while preserving alignment with the existing mount.

## Original STL Attribution

This script modifies the following model (not included in this repository):

"Sports Ball Wall Mount for Football / Soccer / Basketball"
https://www.printables.com/model/708264-sports-ball-wall-mount-for-football-soccer-basketb

All rights belong to the original creator.

## Usage

Activate the virtual environment:

source .venv/bin/activate

Run the script:

.venv/bin/python scale_hoop.py


A new scaled STL file will be created in the project directory.

## Configuration

Edit the parameters at the bottom of scale_hoop.py:

input_path — input STL filename

output_path — output STL filename

scale_factor — e.g., 0.6 for a 40% reduction

## Dependencies

Defined in requirements.txt:

numpy

## Project Structure

stl-hoop-scaling-tool

├── scale_hoop.

├── requirements.txt

├── .gitignore
