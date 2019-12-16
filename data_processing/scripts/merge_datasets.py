#!usr/bin/env python3
"""
merge_datasets.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script takes multiple .csv files of extracted epitopes/cleavage sites
from different sources and compiles them into a single csv for
further filtering and featurization downstream.

Input:
- .csv files from the ./data_processing/un-merged_data directory

Output:
- a single merged .csv with consistent column headers
"""
import os
import re
import pandas as pd

# Later replace file_dir with -i argparse
file_dir = "/Users/weeder/PycharmProjects/proteasome/data_processing/un-merged_data/"
# get lists of positive and negative files to load in
pos_data_files = os.listdir(file_dir + "positives/")
neg_data_files = os.listdir(file_dir + "negatives/")

# compile all files into one list
data_files = [file_dir + "positives/" + pf for pf in pos_data_files]
for nf in neg_data_files:
    data_files.append(file_dir + "negatives/" + nf)

# annotate the type of example (pos or neg)
source_type = ["pos"] * len(pos_data_files) + ['neg'] * len(neg_data_files)

# strip file paths to retrieve base names
source_names = [re.sub("\..*","", re.sub(".*/", "", f)) for f in data_files]

# initialize with first data set
merged_df_raw = pd.read_csv(data_files[0], low_memory=False)
# add row with source database
merged_df_raw['data_source'] = source_names[0]
merged_df_raw['example_type'] = source_type[0]

# loop through and repeat for remaining files
for i in range(1, len(data_files)):
    # load in single file and add necessary columns
    tmp_df = pd.read_csv(data_files[i], low_memory=False)
    tmp_df['data_source'] = source_names[i]
    tmp_df['example_type'] = source_type[i]
    # append onto initiated df
    merged_df_raw = merged_df_raw.append(tmp_df, sort=False)
    del tmp_df


# next, retrieve parent protein if missing ... may need to be function
# depending on which IRI is present

# use code to check/repair sequence indices
