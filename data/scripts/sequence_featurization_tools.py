#!/usr/bin/env python3
"""
sequence_featurization_tools.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script contains an amino acid feature matrix as well as functions for
extracting, processing, encoding and formatting amino acid sequences into
feature arrays for downstream analysis.
"""

import numpy as np
from scipy.sparse import csr_matrix

# aa matrix, cols 1:20 are sparse encodings of aa identity, 21:25 are:
# Aromatic (0/1)
# Hydroxyl (0/1)
# Polarity: Isoelectric point (pI = pH at isoelectric point),
#            obtained from: https://www.sigmaaldrich.com/life-science/
#            metabolomics/learning-center/amino-acid-reference-chart.html#4
#            (D.R. Lide, Handbook of Chemistry and Physics, 72nd Edition,
#            CRC Press, Boca Raton, FL, 1991)
# Molecular Volume: Partial molar volume of AA sidechain @ 37 degrees C, 
#                    obtained from:
#                    https://doi.org/10.1016/S0301-4622(99)00104-0
# Hydrophobicity: Hydrophobicity Scale using the contact angle of a water
#                  nanodroplet (cos θ),
#                  obtained from: https://doi.org/10.1073/pnas.1616138113
# Conformational Entropy: Conformational entropy of AA's not interacting in
#                          secondary structures.
#                          Data Pulled from:
#                          https://doi:10.1371/journal.pone.0132356

# B, J, & Z represent ambiguous aa's that could be one of two similar residues
# * = absence of amino acid: pI = 7.5 (normal cytoplasmic pH),
#      Molecular volume = 0, Hydrophobicity = 1.689157 (extrapolated cos
#      θ for water), Entropy = 0
# X = any amino acid: Aromatic = 0.5, Hydroxl = 0.5, Hydrophobicity = median,
#     all other values are averages
_features = {
    'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,      6.0, 56.15265,   -0.495,  -2.4],
    'C': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,     5.07, 69.61701,    0.081,  -4.7],
    'D': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,     2.77, 70.04515,    9.573,  -4.5],
    'E': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,     3.22, 86.35615,    3.173,  -5.2],
    'F': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   1,   0,     5.48,  119.722,   -0.370,  -4.9],
    'G': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,     5.97, 37.80307,    0.386,  -1.9],
    'H': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   1,   0,     7.59, 97.94236,    2.029,  -4.4],
    'I': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,     6.02, 103.6644,   -0.528,  -6.6],
    'K': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,     9.74, 102.7783,    2.101,  -7.5],
    'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,     5.98, 102.7545,   -0.342,  -6.3],
    'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,     5.74,  103.928,   -0.324,  -6.1],
    'N': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,     5.41, 76.56687,    2.354,  -4.7],
    'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,   0,   0,      6.3, 71.24858,   -0.322,  -0.8],
    'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,   0,   0,     5.65, 88.62562,    2.176,  -5.5],
    'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,   0,   0,    10.76, 110.5867,    4.383,  -6.9],
    'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,   0,   1,     5.68, 55.89516,    0.936,  -4.6],
    'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,   0,   1,      5.6,  72.0909,    0.853,  -5.1],
    'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0,   0,     5.96, 86.28358,   -0.308,  -4.6],
    'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,   1,   0,     5.89, 137.5186,    -0.27,  -4.8],
    'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,   1,   1,     5.66, 121.5862,    1.677,  -5.4],
    '*': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,      7.5,      0.0, 1.689157,   0.0],
    'B': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,     4.09, 73.30601,    5.964,  -4.6],
    'Z': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,   0,   0,     4.44, 87.49089,    2.675, -5.35],
    'J': [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,      6.0, 103.2094,   -0.426, -6.45],
    'U': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0,     5.07, 69.61701,    0.081,  -4.7],
    'X': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 6.008095, 88.55829,   0.6195, -4.845]
}


# define function that returns np array of feature vectors
def featurize_sequence(seq):
    """
    takes an input aa sequence of any length and returns a 2D numpy array
    of feature values with rows as positions and columns as feature values
    :param seq: a string of amino acid symbols of any length
    :return feature_matrix:
    """
    feature_matrix = np.array([_features[aa] for aa in seq], dtype=float)
    return feature_matrix


def generate_feature_array(seq_list):
    """
    generates a 3D array of of 2D feature matrices for a list of sequences
    :param seq_list: list of sequences to featurize (of the same length)
    :return feature_array: 3D numpy array of 2D feature matrices for each seq
    """
    feature_array = np.array([featurize_sequence(s) for s in seq_list])
    return feature_array


def create_sequence_regex(epitope_sequence):
    """
    creates a regular expression from a possibly ambiguous AA sequence
    :param epitope_sequence: epitope sequence of any length (string)
    :return: epitope sequence in re form (string)
    """
    # ensure that sequence is a string
    epitope_sequence = str(epitope_sequence)
    # if no ambiguous characters are present, return the original seq.
    if ("B" not in epitope_sequence and "J" not in epitope_sequence and "Z"
            not in epitope_sequence and "X" not in epitope_sequence):
        return epitope_sequence
    # if any ambiguous characters are present, replace them with OR statements
    else:
        epitope_sequence = epitope_sequence.replace("B", "[B|D|N]")
        epitope_sequence = epitope_sequence.replace("J", "[J|I|L]")
        epitope_sequence = epitope_sequence.replace("Z", "[Z|E|Q]")
        epitope_sequence = epitope_sequence.replace("X", "[A-Z]")
        return epitope_sequence


def get_peptide_window(sequence, starting_position, ending_position, upstream=10, 
                       downstream=10, c_terminal=True):
    """
    returns the window of AA's around the C-term of an epitope, given defined
    upstream and downstream window sizes and a row from a pandas df with
    ending position and full origin sequence of the epitope.
    :param pd_entry: pandas entry with (at min.) ending_position and sequence
    :param sequence: protein sequence from which to extract peptide window
    :param starting_position: N terminal starting position for epitope
                              (1 based, inclusive)
    :param ending_position: C terminal ending position for epitope (1 based,
                             exclusive)
    :param upstream: number of upstream AA's to return
    :param downstream: number of downstream AA's to return
    :param c_terminal: whether c or n terminal cleavage site is to be used for
    window midpoint
    :return: full window of AA's including the cleavage site
    """
    # set cleavage site index based on function flag
    if c_terminal:
        cleave_index = int(ending_position) - 1
    if not c_terminal:
        cleave_index = int(starting_position) - 1

    if cleave_index < 0 or (cleave_index - 1) >= len(sequence):
        return None

    # if upstream window does not hit boundary
    if (cleave_index - upstream) >= 0:
        upstream_seq = sequence[(cleave_index - upstream):cleave_index]
    # if peptide boundary is within upstream window
    if cleave_index - upstream < 0:
        # retrieve relevant sequence
        tmp_seq = sequence[0:cleave_index]
        # add empty AA's prior to seq start
        upstream_seq = abs(cleave_index - upstream) * "*" + tmp_seq
    # repeat above with downstream window
    if (cleave_index + 1 + downstream) < len(sequence):
        downstream_seq = sequence[(cleave_index + 1):(cleave_index +
                                                      downstream + 1)]
    if (cleave_index + 1 + downstream) >= len(sequence):
        # handles issue where cleavage site was end of protein and
        # cleave_index + 1 was beyond sequence bounds
        if cleave_index == (len(sequence) - 1):
            downstream_seq = downstream * "*"
        else:
            tmp_seq = sequence[(cleave_index + 1):len(sequence)]
            downstream_seq = tmp_seq + (downstream + 1 + cleave_index -
                                        len(sequence)) * "*"
    # return up/down stream windows + cleavage site
    return upstream_seq + sequence[cleave_index] + downstream_seq
