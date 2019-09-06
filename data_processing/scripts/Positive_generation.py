#!/usr/bin/env Python3
"""positive_generation.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script allows the user to obtain a 3D and 2D numpy array containing the
positive raw_data to be inputted into the machine learning model. It only accepts
comma separated value files (.csv) at the moment.

This script requires that `pandas` and `numpy` be installed within the Python
environment you are running this script in.

Inputs:
    The location of the csv file containing at least the following column
    headers containing the following information:
        ["Description"]: the sequence of the epitope/proteasome product (str)
        ["Protein Sequence"]: the entire sequence of the corresponding protein;
                              the exact sequence of the epitope/product must
                              be in the protein sequence (str)

Outputs:
    The numpy 3D and 2D array containing the positive raw_data sets for the model
"""

import pandas as pd
import numpy as np
import re

expand = 10

# X below denotes an incomplete window
_features = {
    'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6,     29.5,  -0.495],
    'C': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.07,  51.6,  0.081],
    'D': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.77,  44.2,  9.573],
    'E': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.22,  70.6,  3.173],
    'F': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 5.48,  135.2, -0.37],
    'G': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.97,  0,     0.386],
    'H': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 7.59,  96.3,  2.029],
    'I': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6.02,  108.5, -0.528],
    'K': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.74,  98,    2.101],
    'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.98,  108.6, -0.342],
    'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.74,  104.9, -0.324],
    'N': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.41,  58.8,  2.354],
    'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6.3,   54.1,  -0.322],
    'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 5.65,  81.5,  2.176],
    'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 10.76, 110.5, 4.383],
    'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 5.68,  29.9,  0.936],
    'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 5.6,   56.8,  0.853],
    'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 5.96,  80.5,  -0.308],
    'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 5.89,  164,   -0.27],
    'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 5.66,  137,   1.677],
    'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     -1,    -1000]
}

df = pd.read_csv("csv/converged_filtered.csv", low_memory=False)
df.drop(df[["X" in x for x in df["Protein Sequence"]]].index, inplace=True)
df.drop(df[["U" in x for x in df["Protein Sequence"]]].index, inplace=True)


def get_position(x):
    """Obtains the position of the C-terminal of the epitope/digest product
       Arguments:
           x (int): directory of the dataframe
       Returns:
           int: the position of the cleavage site
    """
    position = x["Protein Sequence"].find(x["Description"]) \
               + len(x["Description"]) - 1

    if str(x["Starting Position"]) != "nan" and \
            x["Protein Sequence"][int(x["Starting Position"]) - 1:
            int(x["Starting Position"]) - 1
            + len(x["Description"])] == x["Description"]:
        position = int(x["Starting Position"]) - 1 + len(x["Description"]) - 1

    elif str(x["Starting Position"]) != "nan" and \
            x["Protein Sequence"].count(x["Description"]) > 1:
        potential_positions = [y + len(x["Description"])
                               for y in [m.start() for m in re.finditer
            ("(?=" + x["Description"] + ")", x["Protein Sequence"])]]

        position = min(potential_positions, key=lambda y:
        abs(y - int(x["Starting Position"]) - len(x["Description"]) + 1))
        print("Position " + str(position) + " taken from "
              + str(potential_positions) + "\n(Actual Position: " + str(
            int(x["Starting Position"]) - 1 + len(x["Description"])) + ")")
    return position


def get_window(x):
    """Obtains the window of interest (with the cleavage site at the
       center) from the information given; when necessary, adds "X" to the
       ends of the window if cleavage site is at the beginning/end of protein
       (meaning the window is incomplete)
       Arguments:
           x (int): directory of the dataframe
       Returns:
           str: the sequence representing the window
    """
    window = ""
    incomplete = False
    position = get_position(x)
    if position < expand:
        incomplete = True
        for i in range(expand - position):
            window += "X"
        window += x["Protein Sequence"][:position + expand + 1]
    if position >= len(x["Protein Sequence"]) - expand:
        incomplete = True
        if len(window) == 0:
            window += x["Protein Sequence"][position - expand:
                                            len(x["Protein Sequence"]) + 1]
        for i in range(expand - (len(x["Protein Sequence"])
                                 - position - 1)):
            window += "X"
    if incomplete is False:
        window = x["Protein Sequence"][position - expand: position
                                                          + expand
                                                          + 1]
    return window

#
df["Window"] = df.apply(get_window, axis=1)
#
df = df[[len(x) == expand * 2 + 1 for x in df["Window"]]]

for i in range(expand * 2 + 1):
    df[i] = df["Window"].apply(lambda x: _features[x[i]])

np_positives = np.array(df.apply(lambda x: pd.DataFrame(
    {y: x[y] for y in range(expand * 2 + 1)}).to_numpy(),
                                 axis=1).to_list())
np.save("converged_filtered_positives_raw", np_positives)

np_positives = np.load("converged_filtered_positives_raw.npy")

n_samples, nx, ny = np_positives.shape
np.save("converged_filtered_positives_2d",
        np_positives.reshape((n_samples, nx*ny)))
