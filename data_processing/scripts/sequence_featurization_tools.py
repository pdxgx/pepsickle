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
# Polarity (PI)
# Molecular Volume (1024 Vtrib)
# Hydrophobicity (cos θ)

# B, J, & Z represent ambiguous aa's that could be one of two similar residues
# U is still in progress and needs to be edited further
_features = {
    'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   6.0,   29.5, -0.495],
    'C': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  5.07,   51.6,  0.081],
    'D': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  2.77,   44.2,  9.573],
    'E': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  3.22,   70.6,  3.173],
    'F': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  5.48,  135.2, -0.370],
    'G': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  5.97,    0.0,  0.386],
    'H': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  7.59,   96.3,  2.029],
    'I': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  6.02,  108.5, -0.528],
    'K': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  9.74,   98.0,  2.101],
    'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  5.98,  108.6, -0.342],
    'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  5.74,  104.9, -0.324],
    'N': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  5.41,   58.8,  2.354],
    'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,   6.3,   54.1, -0.322],
    'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,  5.65,   81.5,  2.176],
    'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 10.76,  110.5,  4.383],
    'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,  5.68,   29.9,  0.936],
    'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,   5.6,   56.8,  0.853],
    'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,  5.96,   80.5, -0.308],
    'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,  5.89,  164.0,  -0.27],
    'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,  5.66,  137.0,  1.677],
    '*': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   np.nan, 0.0, np.nan],
    'B': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  4.09,   51.5,  5.964],
    'Z': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,  4.44,   76.1,  2.675],
    'J': [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   6.0,  108.6,  -.426],
    'U': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x, x,  x,   x,  x],
    'X': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, np.nan, np.nan,   med,med, med]
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


def generate_sparse_feature_matrix(seq_list):
    """
    generates 2D array of (sequence, features) in condensed sparse row format
    from a given sequence list
    :param seq_list: list of amino acid strings of the same length to featurize
    :return feature_matrix: 2D array of features for each sequence given
    """
    feature_matrix = csr_matrix(
        [featurize_sequence(s).flatten() for s in seq_list]
    )
    return feature_matrix


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


# NOTE!!! may need to edit to use value other than X (x only for unknown but
# present amino acids
def get_peptide_window(pd_entry, upstream=10, downstream=10, c_terminal=True):
    """
    returns the window of AA's around the C-term of an epitope, given defined
    upstream and downstream window sizes and a row from a pandas df with
    ending position and full origin sequence of the epitope.
    :param pd_entry: pandas entry with (at min.) ending_position and sequence
    :param upstream: number of upstream AA's to return
    :param downstream: number of downstream AA's to return
    :param c_terminal: whether c or n terminal cleavage site is to be used for
    window midpoint
    :return: full window of AA's including the cleavage site
    """
    # set cleavage site index based on function flag
    if c_terminal:
        cleave_index = int(pd_entry['ending_position']) - 1
    if not c_terminal:
        cleave_index = int(pd_entry['starting_position']) - 1

    # if upstream window does not hit boundary
    if (cleave_index - upstream) >= 0:
        upstream_seq = pd_entry['sequence'][
                       (cleave_index - upstream):cleave_index]
    # if peptide boundary is within upstream window
    if cleave_index - upstream < 0:
        # retrieve relevant sequence
        tmp_seq = pd_entry['sequence'][0:cleave_index]
        # add empty AA's prior to seq start
        upstream_seq = abs(cleave_index - upstream) * "X" + tmp_seq
    # repeat above with downstream window
    if (cleave_index + 1 + downstream) < len(pd_entry['sequence']):
        downstream_seq = pd_entry['sequence'][
                         (cleave_index + 1):(cleave_index + downstream + 1)]
    if (cleave_index + 1 + downstream) >= len(pd_entry['sequence']):
        # handles issue where cleavage site was end of protein and
        # cleave_index + 1 was beyond sequence bounds
        if cleave_index == (len(pd_entry['sequence']) - 1):
            downstream_seq = downstream * "X"
        else:
            tmp_seq = pd_entry['sequence'][
                      (cleave_index + 1):len(pd_entry['sequence'])]
            downstream_seq = tmp_seq + (downstream + 1 + cleave_index -
                                        len(pd_entry['sequence'])) * "X"
    # return up/down stream windows + cleavage site
    return upstream_seq + pd_entry['sequence'][cleave_index] + downstream_seq
