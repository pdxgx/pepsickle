#!/usr/bin/env Python3
"""

"""
import pandas as pd
import numpy as np

indir = "/Users/weeder/PycharmProjects/proteasome/data_extraction/" \
        "raw_data/VHSE_data/"

s1_df = pd.read_excel(indir + "Dataset_s1.xlsx")
s3_df = pd.read_excel(indir + "Dataset_s3.xls")
s5_df = pd.read_excel(indir + "Dataset_s5.xlsx")
# s7 relevant for digesion map data?


s1_cleaned_df = pd.DataFrame()
for e in range(len(s1_df)):
    entry = s1_df.iloc[e]

    if entry['Epitope'] is not np.nan:
        entry_n = entry['Number of results']
        tmp_df = s1_df.iloc[e:(e+entry_n)]

