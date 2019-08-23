#!/usr/bin/env Python3
"""markov_matrix_generation.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script allows the user to obtain a pandas Dataframe containing the
information of a hmm file (hmmer-3.2.1) which used all of the positive window
data for the NeoChop model to be fitted on

This script requires that `pandas`be installed within the Python
environment you are running this script in.

Inputs:
    The location of the hmm file (created using hmmer-3.2.1) created with the
    window sequences to be used to train the model (str)

Outputs:
    The pandas Dataframe containing the hmm information saved as a csv file
"""

expand = 8

import pandas as pd

amino_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
              'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

hmm_dict = {
    'A': [], 'C': [], 'D': [], 'E': [], 'F': [], 'G': [], 'H': [], 'I': [],
    'K': [], 'L': [], 'M': [], 'N': [], 'P': [], 'Q': [], 'R': [], 'S': [],
    'T': [], 'V': [], 'W': [], 'Y': [],
}

with open("output.hmm") as f:
    count = 0
    while count <= 2*expand + 1:
        buffer = f.readline().split()
        if buffer and buffer[0].isdigit():
            for i in range(1, 21):
                hmm_dict[amino_list[i - 1]].append(float(buffer[i]))
            count += 1
        if count == 2*expand + 1:
            buffer = f.readline().split()
            for i in range(0, 20):
                hmm_dict[amino_list[i]].append(float(buffer[i]))
            count += 1

df = pd.DataFrame(hmm_dict)

print(df)
df.to_csv("markov_matrix.csv", index=False)