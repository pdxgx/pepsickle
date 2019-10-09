#!usr/bin/env python3
"""
merge_datasets.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script takes multiple .csv files of extracted epitopes/cleavage sites
from different sources and compiles them into a single csv for
further filtering and featurization downstream.

Input:
- .csv files from the ./data_processing/pre-merged_data directory

Output:
- a single merged .csv with consistent column headers
"""
