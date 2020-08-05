#!/usr/bin/env python3
"""
model_functions.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script contains functions for wrapping generated proteasomal cleavage
prediction models and handling fasta protein inputs for easy model
implementation.
"""

import os
import pepsickle.sequence_featurization_tools as sft
import pandas as pd
from Bio import SeqIO
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

# sets path to stored model weights
_model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class epitopeFullNet(nn.Module):
    """
    Epitope trained proteasomal prediction model using encoded amino acid
    sequences and physical properties. This network uses a 1D convolutional
    mask over the physical property values, followed by 3 fully connected
    internal layers with dropout and batch normalization at each step.
    """
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(p=0.2)
        self.conv = nn.Conv1d(4, 4, 3, groups=4)
        self.input = nn.Linear(304, 136)
        self.bn1 = nn.BatchNorm1d(136)
        self.fc1 = nn.Linear(136, 68)
        self.bn2 = nn.BatchNorm1d(68)
        self.fc2 = nn.Linear(68, 34)
        self.bn3 = nn.BatchNorm1d(34)
        self.out = nn.Linear(34, 2)

    def forward(self, x):
        # perform convolution prior to flattening
        x_seq = x[:, :, :20]
        x_conv = x[:, :, 22:].transpose(1, 2)
        x_conv = self.conv(x_conv)

        # make sure input tensor is flattened
        x_seq = x_seq.reshape(x_seq.shape[0], -1)
        x_conv = x_conv.reshape(x_conv.shape[0], -1)
        x = torch.cat((x_seq, x_conv), 1)

        # pass through  network architecture
        x = self.drop(F.relu(self.bn1(self.input(x))))
        x = self.drop(F.relu(self.bn2(self.fc1(x))))
        x = self.drop(F.relu(self.bn3(self.fc2(x))))
        x = F.log_softmax(self.out(x), dim=1)

        return x


class digestionFullNet(nn.Module):
    """
    in-vitro digestion data trained proteasomal prediction model using encoded
    amino acid sequences and physical properties. This network uses a 1D
    convolutional mask over the physical property values, followed by 3 fully
    connected internal layers with dropout and batch normalization at each step.
    """
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(p=0.2)
        self.conv = nn.Conv1d(4, 4, 3, groups=4)
        self.input = nn.Linear(306, 136)
        self.bn1 = nn.BatchNorm1d(136)
        self.fc1 = nn.Linear(136, 68)
        self.bn2 = nn.BatchNorm1d(68)
        self.fc2 = nn.Linear(68, 34)
        self.bn3 = nn.BatchNorm1d(34)
        self.out = nn.Linear(34, 2)

    def forward(self, x, c_prot, i_prot):
        # make sure input tensor is flattened
        x_seq = x[:, :, :20]
        x_conv = x[:, :, 22:].transpose(1, 2)
        x_conv = self.conv(x_conv)

        # make sure input tensor is flattened
        x_seq = x_seq.reshape(x_seq.shape[0], -1)
        x_conv = x_conv.reshape(x_conv.shape[0], -1)
        x = torch.cat((x_seq, x_conv), 1)

        # add on proteasome type one-hot encoding
        x = torch.cat((x, c_prot.reshape(c_prot.shape[0], -1)), 1)
        x = torch.cat((x, i_prot.reshape(i_prot.shape[0], -1)), 1)

        # pass through the network architecture
        x = self.drop(F.relu(self.bn1(self.input(x))))
        x = self.drop(F.relu(self.bn2(self.fc1(x))))
        x = self.drop(F.relu(self.bn3(self.fc2(x))))
        x = F.log_softmax(self.out(x), dim=1)

        return x


def initialize_epitope_model(human_only=True):
    """
    initializes an epitope based cleavage prediction model
    :param human_only: if true, model weights trained using human data only and
     non-human mammal data will be excluded
    :return: functional pytorch model for predicting proteasomal cleavage
    """
    _model_path = os.path.join(_model_dir,
                              "pepsickle",
                              "trained_model_dict.pickle")
    _model_dict = pickle.load(open(_model_path, 'rb'))

    # set proper model file
    if human_only:
        mod_state = _model_dict['human_only_epitope_full_mod']
    else:
        mod_state = _model_dict['all_mammal_epitope_full_mod']

    # initialize model
    mod = epitopeFullNet()
    mod.load_state_dict(mod_state)
    mod.eval()
    return mod


def initialize_digestion_model(human_only=False):
    """
    initializes an in-vitro digestion based cleavage prediction model
    :param human_only: if true, model weights trained using human data only and
    non-human mammal data will be excluded
    :return: functional pytorch model for predicting proteasomal cleavage
    """
    _model_path = os.path.join(_model_dir,
                              "pepsickle",
                              "trained_model_dict.pickle")
    _model_dict = pickle.load(open(_model_path, 'rb'))

    # set proper model file
    if human_only:
        mod_state = _model_dict['human_only_cleavage_map_full_mod']
    else:
        mod_state = _model_dict['all_mammal_cleavage_map_full_mod']

    # initialize model
    mod = digestionFullNet()
    mod.load_state_dict(mod_state)
    mod.eval()
    return mod


def predict_epitope_mod(model, features):
    """
    Model wrapper that takes an epitope based model and feature array and
    returns a vector of cleavage prediction probabilities
    :param model: epitope based cleavage prediction model
    :param features: array of features from generate_feature_array()
    :return: vector of cleavage probabilities
    """
    features = torch.from_numpy(features)
    with torch.no_grad():
        p_cleavage = torch.exp(
            model(features.type(torch.FloatTensor))[:, 1]
        )

    output_p = [float(x) for x in p_cleavage]
    return output_p


def predict_digestion_mod(model, features, proteasome_type="C"):
    """
    Model wrapper that takes an in-vitro digestion based model and feature
    array and returns a vector of cleavage prediction probabilities
    :param model: digestion based cleavage prediction model
    :param features: array of features from generate_feature_array()
    :param proteasome_type: takes "C" to base predictions on the constitutive
    pepsickle or "I" to base predictions on the immunoproteasome
    :return: vector of cleavage probabilities
    """
    # assert features.shape[2] == 24
    features = torch.from_numpy(features)
    mod1 = model

    if proteasome_type == "C":
        c_prot = torch.tensor([1] * features.shape[0]).type(torch.FloatTensor)
        i_prot = torch.tensor([0] * features.shape[0]).type(torch.FloatTensor)
    elif proteasome_type == "I":
        c_prot = torch.tensor([0] * features.shape[0]).type(torch.FloatTensor)
        i_prot = torch.tensor([1] * features.shape[0]).type(torch.FloatTensor)
    else:
        return ValueError("Proteasome type was not recognized")

    with torch.no_grad():
        log_p1 = mod1(features.type(torch.FloatTensor),
                      c_prot, i_prot)[:, 1]
        p_cleavage = torch.exp(log_p1)

    output_p = [float(x) for x in p_cleavage]
    return output_p


def create_windows_from_protein(protein_seq, **kwargs):
    """
    wrapper for get_peptide_window(). takes in a protein sequence and returns
    a vector of k-merized windows.
    :param protein_seq: protein sequence
    :return: vector of protein windows
    """
    # NOTE: last AA not made into window since c-terminal would be cleavage pos
    protein_windows = []
    for pos in range(len(protein_seq)-1):
        start_pos = pos + 1
        end_pos = pos + 2
        tmp_window = sft.get_peptide_window(protein_seq,
                                            starting_position=start_pos,
                                            ending_position=end_pos,
                                            **kwargs)
        protein_windows.append(tmp_window)

    return protein_windows


def predict_protein_cleavage_locations(protein_seq, model, protein_id=None,
                                       mod_type="epitope",
                                       proteasome_type="C",
                                       threshold=.5):
    """
    general wrapper that accepts full protein information and returns a pandas
    data frame with cleavage site probabilities and predictions
    :param protein_id: protein identifier
    :param protein_seq: full protein sequence
    :param model: functional pytorch model to be used for predictions
    :param mod_type: whether model is using "epitope" or "digestion"
    :param proteasome_type: if digestion, the type of pepsickle to use for
    predictions (C or I)
    :param threshold: threshold used to call cleavage vs. non-cleavage
    :return: summary table for each position in the peptide
    """
    # TODO: get desired window size from model expected size, maybe leave as comment
    protein_windows = create_windows_from_protein(protein_seq)
    window_features = sft.generate_feature_array(protein_windows)

    if mod_type == "epitope":
        preds = predict_epitope_mod(model, window_features)
    if mod_type == "digestion":
        preds = predict_digestion_mod(model, window_features,
                                      proteasome_type=proteasome_type)

    # By definition, last position can never be a cleavage site
    preds[-1] = 0
    positions = range(1, len(preds)+1)
    cleave = [p > threshold for p in preds]

    out_df = pd.DataFrame(zip(positions, preds, cleave),
                          columns=["pos", "p_cleavage", "cleaved"])
    out_df['prot_id'] = protein_id
    return out_df


def process_fasta(fasta_file, cleavage_model, verbose=False, **kwargs):
    """
    handles fasta file path and returns pandas df with cleavage prediction
    results
    :param fasta_file: path to the fasta file that needs processed
    :param cleavage_model: active model or model initialization to be used
    :param verbose: flag to print out progress when list of proteins is given
    :param kwargs: parameters to be passed to the cleavage prediction model
    :return: pandas dataframe with cleavage predictions
    """
    protein_list = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    cleavage_preds = []
    end = len(protein_list)

    for i, protein_id in enumerate(protein_list):
        if i % 100 == 0 and verbose:
            print("completed:", i, "of", end)
        tmp_out = predict_protein_cleavage_locations(protein_id=protein_id,
                                                     protein_seq=protein_list[
                                                         protein_id],
                                                     model=cleavage_model,
                                                     **kwargs)
        cleavage_preds.append(tmp_out)
    out_df = pd.concat(cleavage_preds)
    return out_df
