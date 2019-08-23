#!/usr/bin/env Python3
"""positive_generation.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script allows the user to obtain a 3D numpy array containing the positive
data to be inputted into the NeoChop machine learning model. It only accepts
comma separated value files (.csv) at the moment.

This script requires that `pandas` and `numpy` be installed within the Python
environment you are running this script in.

Inputs:
    The location a hmm converted into a csv file made with the window sequences
    to be used to train the model; see markov_matrix_generation.py (str)

    The location of the csv file containing at least the following column
    headers containing the following information:
        ["Description"]: the sequence of the epitope/proteasome product (str)
        ["Protein Sequence"]: the entire sequence of the corresponding protein;
                              the exact sequence of the epitope/product must
                              be in the protein sequence (str)

Outputs:
    The numpy 3D array containing the positive data set for the model saved as
    a np file
"""

expand = 8

import pandas as pd
import numpy as np

# X below denotes an incomplete window
_sparse_encoding = {
    'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'C': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'D': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'E': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'F': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'G': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'H': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'I': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'K': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'N': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

_blOsum50 = {
    'A': [ 5, -2, -1, -2, -1, -1, -1,  0, -2, -1, -2, -1, -1, -3, -1,  1,  0, -3, -2,  0],
    'C': [-1, -4, -2, -4, 13, -3, -3, -3, -3, -2, -2, -3, -2, -2, -4, -1, -1, -5, -3, -1],
    'D': [-2, -2,  2,  8, -4,  0,  2, -1, -1, -4, -4, -1, -4, -5, -1,  0, -1, -5, -3, -4],
    'E': [-1,  0,  0,  2, -3,  2,  6, -3,  0, -4, -3,  1, -2, -3, -1, -1, -1, -3, -2, -3],
    'F': [-3, -3, -4, -5, -2, -4, -3, -4, -1,  0,  1, -4,  0,  8, -4, -3, -2,  1,  4, -1],
    'G': [ 0, -3,  0, -1, -3, -2, -3,  8, -2, -4, -4, -2, -3, -4, -2,  0, -2, -3, -3, -4],
    'H': [-2,  0,  1, -1, -3,  1,  0, -2, 10, -4, -3,  0, -1, -1, -2, -1, -2, -3,  2, -4],
    'I': [-1, -4, -3, -4, -2, -3, -4, -4, -4,  5,  2, -3,  2,  0, -3, -3, -1, -3, -1,  4],
    'K': [-1,  3,  0, -1, -3,  2,  1, -2,  0, -3, -3,  6, -2, -4, -1,  0, -1, -3, -2, -3],
    'L': [-2, -3, -4, -4, -2, -2, -3, -4, -3,  2,  5, -3,  3,  1, -4, -3, -1, -2, -1,  1],
    'M': [-1, -2, -2, -4, -2,  0, -2, -3, -1,  2,  3, -2,  7,  0, -3, -2, -1, -1,  0,  1],
    'N': [-1, -1,  7,  2, -2,  0,  0,  0,  1, -3, -4,  0, -2, -4, -2,  1,  0, -4, -2, -3],
    'P': [-1, -3, -2, -1, -4, -1, -1, -2, -2, -3, -4, -1, -3, -4, 10, -1, -1, -4, -3, -3],
    'Q': [-1,  1,  0,  0, -3,  7,  2, -2,  1, -3, -2,  2,  0, -4, -1,  0, -1, -1, -1, -3],
    'R': [-2,  7, -1, -2, -4,  1,  0, -3,  0, -4, -3,  3, -2, -3, -3, -1, -1, -3, -1, -3],
    'S': [ 1, -1,  1,  0, -1,  0, -1,  0, -1, -3, -3,  0, -2, -3, -1,  5,  2, -4, -2, -2],
    'T': [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  2,  5, -3, -2,  0],
    'V': [ 0, -3, -3, -4, -1, -3, -3, -4, -4,  4,  1, -3,  1, -1, -3, -2,  0, -3, -1,  5],
    'W': [-3, -3, -4, -5, -5, -1, -3, -3, -3, -3, -2, -3, -1,  1, -4, -4, -3, 15,  2, -3],
    'Y': [-2, -1, -2, -3, -3, -1, -2, -3,  2, -1, -1, -2,  0,  4, -3, -2, -2,  2,  8, -1],
    'X': [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5]
}

hmm_df = pd.read_csv("csv/markov_matrix.csv")

df = pd.read_csv("csv/syfpeithi_ncbi.csv")
df.drop(df[["X" in x for x in df["Protein Sequence"]]].index, inplace=True)
df.drop(df[["U" in x for x in df["Protein Sequence"]]].index, inplace=True)


def get_window(x):
    """Obtains the window of interest

       The window contains the cleavage site at the center and expands upstream
       and downstream exactly window_size (default = 8) amino acids.
       When necessary, it adds "X" to the ends of the window when the window is
       incomplete

       Arguments:
            x (pd.Series): directory of the dataframe
       Returns:
           str: the window sequence
    """
    window = ""
    position = x["Protein Sequence"].find(x["Description"]) \
        + len(x["Description"])
    if position < expand:
        for i in range(expand + 1 - position):
            window = window + "X"
        return window + x["Protein Sequence"][:position + expand]

    if position > len(x["Protein Sequence"]) - expand:
        for i in range(expand - (len(x["Protein Sequence"]) - position)):
            window = window + "X"
        return x["Protein Sequence"][position - expand - 1:
                                     len(x["Protein Sequence"])] + window
    else:
        return x["Protein Sequence"][position - expand - 1: position
                                                            + expand]


df["Window"] = df.apply(get_window, axis=1)

df = df[[len(x) == 2 * expand + 1 for x in df["Window"]]]


def get_markov(x, pos):
    """Obtains the markov matrix value

       The hidden markov matrix value is (log(pi,j)/log(qi):
       p is the probability of finding the amino acid, i, at position j in the
       window data given
       q the probability of finding the amino acid i in the SwissProt database
       Both these values are calculated using the hmm file inputted
       (see markov_matrix_generation.py)

       Arguments:
            x (pd.Series): directory of the dataframe
            pos (int): position in the window
       Returns:
           float: weighted float value calculated
    """
    if x[pos] == "X":
        return 0
    else:
        return np.log(np.exp(-hmm_df[x[pos]][pos]))\
               / np.log(np.exp(-hmm_df[x[pos]][2 * expand + 1]))


for i in range(2 * expand + 1):
    df[i] = df["Window"].apply(lambda x: _sparse_encoding[x[i]])
    df[i + 2*expand + 1] = df["Window"].apply(lambda x: _blOsum50[x[i]])
    df[i + 2*(2*expand + 1)] = df["Window"].apply(lambda x: get_markov(x, i))


def get_arrays(x):
    """Obtains the 2D numpy array for a window using the information from the
       Dataframe created for it
       Arguments:
           x (pd.Series): directory of the dataframe
       Returns:
           numpy.array: the 2D numpy array
       """
    return pd.DataFrame({y: x[y] + x[y + 2*expand + 1] + [x[y + 2*(2*expand + 1)]]
                         for y in range(2*expand + 1)}).to_numpy()


np.save("syfpeithi_ncbi_positives_X_.npy",
        np.array(df.apply(get_arrays, axis=1).to_list()))
