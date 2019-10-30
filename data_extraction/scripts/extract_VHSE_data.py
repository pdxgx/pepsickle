#!/usr/bin/env Python3
"""
doi: 10.1371/journal.pone.0074506
"""
import pandas as pd
import numpy as np

indir = "/Users/weeder/PycharmProjects/proteasome/data_extraction/" \
        "raw_data/VHSE_data/"

s5_df = pd.read_excel(indir + "Dataset_s5.xlsx")
# s7 relevant for digesion map data?

