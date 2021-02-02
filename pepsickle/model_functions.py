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
from Bio import SeqIO
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib


# sets path to stored model weights
_model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DigestionSeqNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.in_nodes = 262 # for normal 13aa window
        self.in_nodes = 7 * 20 + 2
        self.drop = nn.Dropout(p=0.25)
        self.input = nn.Linear(self.in_nodes, 136)
        self.bn1 = nn.BatchNorm1d(136)
        self.fc1 = nn.Linear(136, 68)
        self.bn2 = nn.BatchNorm1d(68)
        self.fc2 = nn.Linear(68, 34)
        self.bn3 = nn.BatchNorm1d(34)
        self.out = nn.Linear(34, 2)

    def forward(self, x, c_prot, i_prot):
        # make sure input tensor is flattened

        x = x.reshape(x.shape[0], -1)
        x = torch.cat((x, c_prot.reshape(c_prot.shape[0], -1)), 1)
        x = torch.cat((x, i_prot.reshape(i_prot.shape[0], -1)), 1)

        x = self.drop(F.relu(self.bn1(self.input(x))))
        x = self.drop(F.relu(self.bn2(self.fc1(x))))
        x = self.drop(F.relu(self.bn3(self.fc2(x))))
        x = F.log_softmax(self.out(x), dim=1)

        return x


class DigestionMotifNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.in_nodes = 46
        self.in_nodes = (7 - 2) * 4 + 2
        self.drop = nn.Dropout(p=.25)
        self.conv = nn.Conv1d(4, 4, 3, groups=4)
        # self.fc1 = nn.Linear(78, 38)
        self.fc1 = nn.Linear(self.in_nodes, 38)
        self.bn1 = nn.BatchNorm1d(38)
        self.fc2 = nn.Linear(38, 20)
        self.bn2 = nn.BatchNorm1d(20)
        self.out = nn.Linear(20, 2)

    def forward(self, x, c_prot, i_prot):
        # perform convolution prior to flattening
        x = x.transpose(1, 2)
        x = self.conv(x)

        # make sure input tensor is flattened
        x = x.reshape(x.shape[0], -1)
        x = torch.cat((x, c_prot.reshape(c_prot.shape[0], -1)), 1)
        x = torch.cat((x, i_prot.reshape(i_prot.shape[0], -1)), 1)

        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        x = F.log_softmax(self.out(x), dim=1)

        return x


class EpitopeSeqNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_nodes = 17 * 20
        self.drop = nn.Dropout(p=0.2)
        self.input = nn.Linear(self.in_nodes, 136)
        self.bn1 = nn.BatchNorm1d(136)
        self.fc1 = nn.Linear(136, 68)
        self.bn2 = nn.BatchNorm1d(68)
        self.fc2 = nn.Linear(68, 34)
        self.bn3 = nn.BatchNorm1d(34)
        self.out = nn.Linear(34, 2)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.reshape(x.shape[0], -1)

        x = self.drop(F.relu(self.bn1(self.input(x))))
        x = self.drop(F.relu(self.bn2(self.fc1(x))))
        x = self.drop(F.relu(self.bn3(self.fc2(x))))
        x = F.log_softmax(self.out(x), dim=1)

        return x


class EpitopeMotifNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_nodes = (17 - 2) * 4
        self.drop = nn.Dropout(p=0.2)
        self.conv = nn.Conv1d(4, 4, 3, groups=4)
        self.fc1 = nn.Linear(self.in_nodes, 38)
        self.bn1 = nn.BatchNorm1d(38)
        self.fc2 = nn.Linear(38, 20)
        self.bn2 = nn.BatchNorm1d(20)
        self.out = nn.Linear(20, 2)

    def forward(self, x):
        # perform convolution prior to flattening
        x = x.transpose(1, 2)
        x = self.conv(x)

        # make sure input tensor is flattened
        x = x.reshape(x.shape[0], -1)

        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        x = F.log_softmax(self.out(x), dim=1)

        return x


def initialize_epitope_model(human_only=False):
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
        seq_mod_state = _model_dict['human_epitope_sequence_mod']
        motif_mod_state = _model_dict['human_epitope_motif_mod']
    else:
        seq_mod_state = _model_dict['all_mammal_epitope_sequence_mod']
        motif_mod_state = _model_dict['all_mammal_epitope_motif_mod']

    # initialize models
    seq_mod = EpitopeSeqNet()
    motif_mod = EpitopeMotifNet()
    seq_mod.load_state_dict(seq_mod_state)
    motif_mod.load_state_dict(motif_mod_state)
    seq_mod.eval()
    motif_mod.eval()
    return [seq_mod, motif_mod]


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
        seq_mod_state = _model_dict['human_20S_digestion_sequence_mod']
        motif_mod_state = _model_dict['human_20S_digestion_motif_mod.']
    else:
        seq_mod_state = _model_dict['all_mammal_20S_digestion_sequence_mod']
        motif_mod_state = _model_dict['all_mammal_20S_digestion_motif_mod']

    # initialize models
    seq_mod = DigestionSeqNet()
    motif_mod = DigestionMotifNet()
    seq_mod.load_state_dict(seq_mod_state)
    motif_mod.load_state_dict(motif_mod_state)
    seq_mod.eval()
    motif_mod.eval()
    return [seq_mod, motif_mod]


def initialize_digestion_rf_model(human_only=False):
    # TODO: add in human only/non-human only options
    _model_path = os.path.join(_model_dir,
                               "pepsickle",
                               "model.joblib")
    model = joblib.load(_model_path)
    return model


def predict_epitope_mod(model, features):
    """
    Model wrapper that takes an epitope based model and feature array and
    returns a vector of cleavage prediction probabilities
    :param model: epitope based cleavage prediction models (list)
    :param features: array of features from generate_feature_array()
    :return: vector of cleavage probabilities
    """
    features = torch.from_numpy(features)
    with torch.no_grad():
        p_cleavage1 = torch.exp(
            model[0](features.type(torch.FloatTensor)[:, :, :20])[:, 1]
        )
        p_cleavage2 = torch.exp(
            model
            [1](features.type(torch.FloatTensor)[:, :, 22:])[:, 1]
        )
        p_cleavage_avg = (p_cleavage1 + p_cleavage2) / 2

    output_p = [float(x) for x in p_cleavage_avg]
    return output_p


def predict_digestion_mod(model, features, proteasome_type="C"):
    """
    Model wrapper that takes an in-vitro digestion based model and feature
    array and returns a vector of cleavage prediction probabilities
    :param model: digestion based cleavage prediction model (list)
    :param features: array of features from generate_feature_array()
    :param proteasome_type: takes "C" to base predictions on the constitutive
    pepsickle or "I" to base predictions on the immunoproteasome
    :return: vector of cleavage probabilities
    """
    # assert features.shape[2] == 24
    features = torch.from_numpy(features)

    if proteasome_type == "C":
        c_prot = torch.tensor([1] * features.shape[0]).type(torch.FloatTensor)
        i_prot = torch.tensor([0] * features.shape[0]).type(torch.FloatTensor)
    elif proteasome_type == "I":
        c_prot = torch.tensor([0] * features.shape[0]).type(torch.FloatTensor)
        i_prot = torch.tensor([1] * features.shape[0]).type(torch.FloatTensor)
    else:
        return ValueError("Proteasome type was not recognized")

    with torch.no_grad():
        p1 = torch.exp(
            model[0](features.type(torch.FloatTensor)[:, :, :20], c_prot,
                      i_prot)[:, 1]
        )
        p2 = torch.exp(
            model[1](features.type(torch.FloatTensor)[:, :, 22:], c_prot,
                      i_prot)[:, 1]
        )
        p_cleavage = (p1 + p2) / 2

    output_p = [float(x) for x in p_cleavage]
    return output_p


def predict_digestion_rf_mod(model, features, proteasome_type="C",
                             shift_p=False):
    # set c/i identity for each entry
    if proteasome_type == "C":
        c_prot = np.array([1] * features.shape[0])
        i_prot = np.array([0] * features.shape[0])
    elif proteasome_type == "I":
        c_prot = np.array([0] * features.shape[0])
        i_prot = np.array([1] * features.shape[0])
    x = features[:, :, 22:].reshape(features.shape[0], -1)
    x = np.concatenate((x, c_prot.reshape(c_prot.shape[0], -1)), 1)
    x = np.concatenate((x, i_prot.reshape(i_prot.shape[0], -1)), 1)

    # shift based on class imbalance:
    if shift_p:
        shift = (0.5 - 0.361)  # training set overall imbalance
        """
        if proteasome_type == "C":
            shift = (0.5 - 0.371)  # proteasome specific class imbalance
        else:
            shift = (0.5 - .343)  # proteasome specific class imbalance
        """
    else:
        shift = 0
    p = model.predict_proba(x)[:, 1]
    probs = [min((float(x) + shift), 1) for x in p]
    return probs


def create_windows_from_protein(protein_seq, **kwargs):
    """
    wrapper for get_peptide_window(). takes in a protein sequence and returns
    a vector of k-merized windows.
    :param protein_seq: protein sequence
    :return: vector of protein windows
    """
    # NOTE: last AA not made into window since c-terminal would be cleavage pos
    protein_windows = []
    for pos in range(len(protein_seq)):
        start_pos = pos + 1
        end_pos = pos + 1
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

    if mod_type == "epitope":
        upstream = 8
        downstream = 8
        protein_windows = create_windows_from_protein(protein_seq,
                                                      upstream=upstream,
                                                      downstream=downstream)
        window_features = sft.generate_feature_array(protein_windows)
        preds = predict_epitope_mod(model, window_features)

    elif mod_type == "in-vitro-2":
        upstream = 3
        downstream = 3
        protein_windows = create_windows_from_protein(protein_seq,
                                                      upstream=upstream,
                                                      downstream=downstream)
        window_features = sft.generate_feature_array(protein_windows)
        preds = predict_digestion_mod(model, window_features,
                                      proteasome_type=proteasome_type)
    elif mod_type == "in-vitro":
        upstream = 3
        downstream = 3
        protein_windows = create_windows_from_protein(protein_seq,
                                                      upstream=upstream,
                                                      downstream=downstream)
        window_features = sft.generate_feature_array(protein_windows,
                                                     normalize=True)
        preds = predict_digestion_rf_mod(model, window_features,
                                         proteasome_type=proteasome_type)

    # By definition, last position can never be a cleavage site
    preds[-1] = 0
    out_preds = [round(p, 4) for p in preds]
    positions = range(1, len(preds)+1)
    cleave = [p > threshold for p in preds]
    prot_list = [protein_id] * len(positions)
    out_zip = zip(positions, out_preds, cleave, prot_list)
    out = [i for i in out_zip]
    return out


def format_protein_cleavage_locations(protein_preds):
    out_lines = []
    for item in protein_preds:
        line = "{}\t{}\t{}\t{}".format(item[0], item[1], item[2], item[3])
        out_lines.append(line)
    return out_lines


def process_fasta(fasta_file, cleavage_model, verbose=False,  **kwargs):
    """
    handles fasta file path and returns pandas df with cleavage prediction
    results
    :param fasta_file: path to the fasta file that needs processed
    :param cleavage_model: active model or model initialization to be used
    :param verbose: flag to print out progress when list of proteins is given
    :param out_file_location: output location where results are written.
    :param kwargs: parameters to be passed to the cleavage prediction model
    :return: pandas dataframe with cleavage predictions
    """
    protein_list = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    end = len(protein_list)
    master_lines = ["positions \t cleav_prob \t cleaved \t protein_id"]
    for i, protein_id in enumerate(protein_list):
        if i % 100 == 0 and verbose:
            print("completed:", i, "of", end)
        tmp_out = predict_protein_cleavage_locations(
            protein_id=protein_id, protein_seq=protein_list[protein_id],
            model=cleavage_model, **kwargs)

        for line in format_protein_cleavage_locations(tmp_out):
            master_lines.append(line)

    return master_lines
