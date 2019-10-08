#! usr/bin/env python3
"""
positive_event_featurization.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script takes a pandas data frame with epitopes, origin peptide seq's,
start positions, and stop positions and returns the feature window around
C-terminal cleavage sites of interest.
"""
import sys
import pandas as pd
import numpy as np
import re
sys.path.insert(1, './data_processing/scripts')  # think this is not the best way to import
import sequence_featurization_tools as sf


epitope_df = pd.read_csv("new/path/to/converged_filtered.csv",
                         low_memory=False)

pos_cleavage_windows = []
for i in range(len(epitope_df)):
    print(i)
    row = epitope_df.iloc[i]
    window = sf.get_peptide_window(row)
    pos_cleavage_windows.append(window)

positive_feature_matrix = \
    sf.generate_sparse_feature_matrix(pos_cleavage_windows)

np.save("converged_positive_feature_matrix_csr.npy", positive_feature_matrix)
