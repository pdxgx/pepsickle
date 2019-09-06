#!/usr/bin/env Python3
"""Negative_generation_grouped_exclusion.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script allows the user to obtain a 3D numpy array containing the negative
raw_data to be inputted into the machine learning model. It only accepts
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
    The numpy 3D array containing the negative raw_data set for the model saved as
    a np file
"""

import pandas as pd
import numpy as np
import re
import time
import multiprocessing as mp

max_epitope_length = 20
conservative_range = 3
np_expand = 10
unprocessed_epitope_length = 24
align_expand = unprocessed_epitope_length - 3 - conservative_range + np_expand

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

df = pd.read_csv("csv/IEDB_filtered.csv", low_memory=False)
# df = pd.read_csv("csv/alignment_validation.csv", low_memory=False)
# df = pd.read_csv("csv/exclusion_validation.csv", low_memory=False)

df.drop(df[["X" in x for x in df["Protein Sequence"]]].index, inplace=True)
df.drop(df[["U" in x for x in df["Protein Sequence"]]].index, inplace=True)
df.drop(df[[len(x) > max_epitope_length for x in df["Description"]]].index,
        inplace=True)
df.reset_index(drop=True, inplace=True)


def get_position(x):
    """Obtains the position of the C-terminal of the epitope/digest product
       Arguments:
           x (int): directory of the dataframe
       Returns:
           int: the position of the cleavage site
    """
    position = x["Protein Sequence"].find(x["Description"]) \
               + len(x["Description"]) - 1

    # Check if epitope is found using the starting position given
    if str(x["Starting Position"]) != "nan" and \
            x["Protein Sequence"][int(x["Starting Position"]) - 1:
            int(x["Starting Position"]) - 1
            + len(x["Description"])] == x["Description"]:
        position = int(x["Starting Position"]) - 1 + len(x["Description"]) - 1

    # If multiple epitopes are found, selects the one closest to where it is
    # expected (at the starting position + len(epitope))
    elif str(x["Starting Position"]) != "nan" and \
            x["Protein Sequence"].count(x["Description"]) > 1:
        potential_positions = [y + len(x["Description"])
                               for y in [m.start() for m in re.finditer
            ("(?=" + x["Description"] + ")", x["Protein Sequence"])]]

        position = min(potential_positions, key=lambda y:
        abs(y - int(x["Starting Position"]) - len(x["Description"]) + 1))
        # print("Position " + str(position) + " taken from "
        #       + str(potential_positions) + "\n(Reported Position: " + str(
        #     int(x["Starting Position"]) - 1 + len(x["Description"])) + ")")
        # print("Index: " + str(x.name) + "\t" + "Description: "
        #       + x["Description"] + "\n")

    # If only one epitope in protein sequence or no starting position given,
    # returns the first epitope found
    return position


def get_alignement_sequence(x):
    """Obtains the sequence to be aligned from the information given; when
       necessary, adds "X" to the ends of the sequence if incomplete
       Arguments:
           x (int): directory of the dataframe
       Returns:
           str: the sequence representing the window
    """
    alignment_seq = ""
    incomplete = False
    position = get_position(x)

    # Checks if X's need to be added at the beginning of the alignment and adds
    if position < unprocessed_epitope_length + np_expand:
        incomplete = True
        for i in range(unprocessed_epitope_length + np_expand - position - 1):
            alignment_seq += "X"
        alignment_seq += x["Protein Sequence"][:position + align_expand + 1]

    # Checks if X's need to be added at the end of the alignment and if so adds
    if position >= len(x["Protein Sequence"]) - align_expand - 1:
        incomplete = True
        # Checks if beginning of alignment already generated and if not adds
        if len(alignment_seq) == 0:
            alignment_seq += x["Protein Sequence"][position
                                                   - unprocessed_epitope_length
                                                   - np_expand + 1:
                                                   len(x["Protein Sequence"])
                                                   + 1]
        for i in range(align_expand
                       - len(x["Protein Sequence"][position:])
                       + 1):
            alignment_seq += "X"

    # If no X's need to be added generates the entire alignment
    if incomplete is False:
        alignment_seq = x["Protein Sequence"][position
                                              - unprocessed_epitope_length
                                              - np_expand
                                              + 1: position + align_expand + 1]
    return alignment_seq


df["Alignment"] = df.apply(get_alignement_sequence, axis=1)
df["Range"] = df.apply(lambda x: list(
    range(x["Alignment"].find(x["Description"]) + conservative_range - 1,
          x["Alignment"].find(x["Description"]) + len(x["Description"])
          - conservative_range)), axis=1)
df["Positive Window"] = df.apply(
    lambda x: x["Alignment"][-(align_expand - np_expand + 2 * np_expand + 1):
                             -(align_expand - np_expand)], axis=1)

df["Exclusion Window"] = df.apply(
    lambda x: x["Alignment"][:x["Alignment"].find(x["Description"])
                              + np_expand], axis=1)

# Convert the df into smaller lists
alignment_list = df["Alignment"].to_list()
range_list = df["Range"].to_list()
pos_window_list = df["Positive Window"].to_list()
exc_window_list = df["Exclusion Window"].to_list()
negative_df = pd.DataFrame()
digestion_list = pd.read_csv("csv/digestion.csv")["Window"].to_list()

""" Debugging Processes: """
# # Alignment
# for i in range(len(alignment_list)):
#     print("Description:       " + description_list[i]
#           + "\nAlignment:         " + alignment_list[i] + "\t"
#           + str(len(alignment_list[i])) + "\n"
#           + "Desired Alignment: " + df["Desired Alignment"][i]
#           + "\t" + str(alignment_list[i] == df["Desired Alignment"][i])
#           + "\n")

# # Exclusion
# exclusions = []
# for i in range(len(alignment_list)):
#     buffer = []
#     print(str(i)
#           + "\nAlignment:   " + alignment_list[i])
#     if pos_window_list[i] in alignment_list[0]:
#         exclusions += [alignment_list[0].find(pos_window_list[i])
#                        + np_expand]
#         buffer += [alignment_list[0].find(pos_window_list[i])
#                   + np_expand]
#
#     if exc_window_list[i] in alignment_list[0]:
#         pos = alignment_list[0].find(exc_window_list[i]) + np_expand
#         exclusions += list(range(
#             pos, pos + len(exc_window_list[i]) - 2*np_expand))
#         buffer += list(range(
#             pos, pos + len(exc_window_list[i]) - 2*np_expand))
#     exclusions.sort()
#     buffer.sort()
#     print("             " + str(buffer))
#
# print([list(set(range_list[0]) ^ (set(range_list[0]) & set(exclusions))),
#        alignment_list[0]])


def parallel_exclusion_generation(x):
    """Calls get_exclusions and allows the program to run parallel processing
       Arguments:
           x (int): the directory of the alignment list
       Returns:
           int: the list of positions of exclusion sites
    """

    exclusions = []
    for i in range(len(pos_window_list)):
        if pos_window_list[i] in alignment_list[x]:
            exclusions += [alignment_list[x].find(pos_window_list[i])
                           + np_expand]

        if exc_window_list[i] in alignment_list[x]:
            pos = alignment_list[x].find(exc_window_list[i]) + np_expand
            exclusions += list(range(
                pos, pos + len(exc_window_list[i]) - 2 * np_expand))

    for j in range(len(digestion_list)):
        if digestion_list[j] in alignment_list[x]:
            exclusions += [alignment_list[x].find(digestion_list[j])
                           + np_expand]

    return [list(set(range_list[x]) ^ (set(range_list[x]) & set(exclusions))),
            alignment_list[x]]


start_time = time.time()
if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    negative_np = np.array(pool.map(parallel_exclusion_generation,
                                    [i for i in range(len(alignment_list))]))
    pool.close()

for row in negative_np:
    if len(row[0]) > 0:
        negative_df = negative_df.append(pd.DataFrame({"Position": row[0],
                                                       "Alignment": row[1]}))

negative_df.to_csv("negatives_.csv", index=False)

print("--- %s minutes ---" % ((time.time() - start_time) / 60))


df = pd.read_csv("negatives_.csv")
df["Window"] = df.apply(lambda x: x["Alignment"][x["Position"]
                                                 - np_expand:
                                                 x["Position"]
                                                 + np_expand + 1], axis=1)

df = df[[len(x) == np_expand * 2 + 1 for x in df["Window"]]]

for i in range(np_expand * 2 + 1):
    df[i] = df["Window"].apply(lambda x: _features[x[i]])

np_negatives = np.array(df.apply(lambda x: pd.DataFrame(
    {y: x[y] for y in range(np_expand * 2 + 1)}).to_numpy(),
                                 axis=1).to_list())
np.save("npy/converged_filtered_negatives_raw", np_negatives)

n_samples, nx, ny = np_negatives.shape
np.save("npy/converged_filtered_negatives_2d",
        np_negatives.reshape((n_samples, nx*ny)))
