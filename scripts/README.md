# Scripts Directory

This directory contains various scripts for metaplan generation, sampling, and evaluation.

## Main Files

### `gen_metaplan.py`
Script for generating metaplans. Supports asynchronous batch processing for efficiency.

### `construct_metaplan_pairs.py`
Constructs preference pairs based on metaplan execution success rates. This script analyzes the execution results of different metaplans and selects the best and worst performing metaplans as preference pairs.

### `split_metaplans_for_sample.py`
Splits generated metaplan samples into multiple files for parallel evaluation.

### `mc_sample.sh`
Main script for metaplan quality evaluation. It coordinates the entire workflow:
1. Sampling multiple metaplans
2. Launching task completion agents
3. Evaluating metaplan quality
4. Constructing preference pairs

### `template.py`
Contains prompt templates for ALFWorld and SciWorld environments. These templates guide large language models to generate structured metaplans.

## Usage

A typical workflow is:
1. Use `gen_metaplan.py` to generate multiple metaplans
2. Use `split_metaplans_for_sample.py` to split metaplans into multiple files
4. Use `construct_metaplan_pairs.py` to construct preference pairs

For detailed parameters and options, please refer to the help documentation of each script.
