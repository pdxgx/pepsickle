#!/usr/bin/env python3
"""
digestion_map_based_ensemble_net.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script contains trains neural networks based on both the sequence identity
and physical property motifs of cleavage and non-cleavage examples from
digestion map data. Exports trained model wieghts.
"""
from sequence_featurization_tools import *
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import torch.nn.functional as F
import pandas as pd

# visualization tools
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torchvision import models
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz


def attribute_features(model, algo, input, label, **kwargs):
    model.zero_grad
    tensor_attributions = algo.attribute(input, label, **kwargs)
    return tensor_attributions

# set CPU or GPU
dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU


# load in data and set output directory
# indir = "D:/Hobbies/Coding/proteasome_networks/data/"
indir = "/Users/weeder/PycharmProjects/pepsickle/data/generated_training_sets/"
file = "/cleavage_windows_dig_only_all_C.pickle"
out_dir = "/models/model_weights"
test_holdout_p = .1
n_epoch = 42

# set seed for consistency
torch.manual_seed(123)


# define network structures
class SeqNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(p=0.2)
        self.input = nn.Linear(260, 136)
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


class MotifNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(p=.2)
        self.conv = nn.Conv1d(4, 4, 3, groups=4)
        # self.fc1 = nn.Linear(78, 38)
        self.fc1 = nn.Linear(44, 38)
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


# initialize networks
sequence_model = SeqNet()
motif_model = MotifNet()

# convert to cuda scripts if on GPU
if dtype is torch.cuda.FloatTensor:
    sequence_model = sequence_model.cuda()
    motif_model = motif_model.cuda()

# load in data from pickled dictionary
handle = open(indir + file, "rb")
data = pickle.load(handle)

# create list of cleavage windows
positive_dict = data['pepsickle']['positives']
pos_windows = []
for key in positive_dict.keys():
    if key not in pos_windows:
        pos_windows.append(key)

# create list of non-cleavage windows
negative_dict = data['pepsickle']['negatives']
neg_windows = []
for key in negative_dict.keys():
    if key not in neg_windows:
        neg_windows.append(key)


# generate feature set for cleavage windows
pos_feature_matrix = torch.from_numpy(generate_feature_array(pos_windows))
# append pepsickle type indicators

# generate feature set for non-cleavage windows
neg_feature_matrix = torch.from_numpy(generate_feature_array(neg_windows))


# set number of positive and negative examples based on test proportion
pos_train_k = torch.tensor(round((1-test_holdout_p) * len(pos_feature_matrix)))
neg_train_k = torch.tensor(round((1-test_holdout_p) * len(pos_feature_matrix)))

# permute and split data
pos_perm = torch.randperm(torch.tensor(len(pos_feature_matrix)))
pos_train = [pos_feature_matrix[i] for i in pos_perm[:pos_train_k]]
pos_test = [pos_feature_matrix[i] for i in pos_perm[pos_train_k:]]

neg_perm = torch.randperm(torch.tensor(len(neg_feature_matrix)))
neg_train = [neg_feature_matrix[i] for i in neg_perm[:neg_train_k]]
# force balanced testing set
neg_test = [neg_feature_matrix[i] for i in neg_perm[neg_train_k:(torch.tensor(
    len(pos_test)) + neg_train_k)]]


# pair training data with labels
pos_train_labeled = []
for i in range(len(pos_train)):
    pos_train_labeled.append([pos_train[i], torch.tensor(1)])
neg_train_labeled = []
for i in range(len(neg_train)):
    neg_train_labeled.append([neg_train[i], torch.tensor(0)])

train_data = pos_train_labeled + neg_train_labeled
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=64, shuffle=True)

# pair test data with labels
pos_test_labeled = []
for i in range(len(pos_test)):
    pos_test_labeled.append([pos_test[i], torch.tensor(1)])
neg_test_labeled = []
for i in range(len(neg_test)):
    neg_test_labeled.append([neg_test[i], torch.tensor(0)])

test_data = pos_test_labeled + neg_test_labeled
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=64, shuffle=True)

# establish training parameters
# inverse weighting for class imbalance in training set
seq_criterion = nn.NLLLoss(
    weight=torch.tensor([1, len(neg_train)/len(pos_train)]).type(dtype))
seq_optimizer = optim.Adam(sequence_model.parameters(), lr=.001)

motif_criterion = nn.NLLLoss(
    weight=torch.tensor([1, len(neg_train)/len(pos_train)]).type(dtype))
motif_optimizer = optim.Adam(motif_model.parameters(), lr=.001)

# train
prev_seq_auc = 0
prev_motif_auc = 0
for epoch in range(n_epoch):
    # reset running loss for each epoch
    seq_running_loss = 0
    motif_running_loss = 0
    # load data, convert if needed
    for dat, labels in train_loader:
        # convert to proper data type
        matrix_dat = dat.type(dtype)
        if dtype == torch.cuda.FloatTensor:
            labels = labels.cuda()

        # reset gradients
        seq_optimizer.zero_grad()
        motif_optimizer.zero_grad()

        # generate model predictions
        seq_est = sequence_model(matrix_dat[:, :, :20])  # one hot encoded sequences
        # physical properties (not side chains)
        motif_est = motif_model(matrix_dat[:, :, 22:])

        # calculate loss
        seq_loss = seq_criterion(seq_est, labels)
        motif_loss = motif_criterion(motif_est, labels)

        # back prop loss and step
        seq_loss.backward()
        seq_optimizer.step()
        motif_loss.backward()
        motif_optimizer.step()

        seq_running_loss += seq_loss.item()
        motif_running_loss += motif_loss.item()
    else:
        # output progress
        print(f'Epoch: {epoch + 1}')
        print(f'Sequence Model Running Loss: {seq_running_loss}')
        print(f'Motif Model Running Loss: {motif_running_loss}')

        # test with no grad to speed up
        with torch.no_grad():
            # set to eval mode
            sequence_model.eval()
            motif_model.eval()
            dat, labels = next(iter(test_loader))

            # convert to proper data type
            matrix_dat = dat.type(dtype)
            # convert labels if on GPU
            if dtype == torch.cuda.FloatTensor:
                labels = labels.cuda()

            # get est probability of cleavage event
            exp_seq_est = torch.exp(sequence_model(
                matrix_dat[:, :, :20]))[:, 1].cpu()  # one hot encoded sequences
            exp_motif_est = torch.exp(motif_model(
                matrix_dat[:, :, 22:]))[:, 1].cpu()  # not including side chains
            # take simple average
            consensus_est = (exp_seq_est + exp_motif_est) / 2

            # calculate AUC for each
            seq_auc = metrics.roc_auc_score(labels, exp_seq_est)
            motif_auc = metrics.roc_auc_score(labels, exp_motif_est)
            consensus_auc = metrics.roc_auc_score(labels, consensus_est)

            # store if most performant model so far
            if seq_auc > prev_seq_auc:
                seq_state = sequence_model.state_dict()
                prev_seq_auc = seq_auc

            if motif_auc > prev_motif_auc:
                motif_state = motif_model.state_dict()
                prev_motif_auc = motif_auc

            # print out performance
            print("Test Set Results:")
            print(f'Sequence Model AUC: {seq_auc}')
            print(f'Motif Model AUC: {motif_auc}')
            print(f'Consensus Model AUC: {consensus_auc}')
            print("\n")

    # return to training mode
    sequence_model.train()
    motif_model.train()

# re-set model states to best performing model
sequence_model.load_state_dict(seq_state)
sequence_model.eval()
motif_model.load_state_dict(motif_state)
motif_model.eval()

# save model states to file
torch.save(seq_state, out_dir + "/all_mammal_cleavage_map_sequence_mod.pt")
torch.save(motif_state, out_dir + "/all_mammal_cleavage_map_motif_mod.pt")

## identify feature improtance
dat, labels = next(iter(test_loader))
saliency = Saliency(motif_model)
grads = saliency.attribute(dat[:, :, 22:].type(dtype), target=labels)
mean_grads = torch.mean(grads, dim=0)


pos_grads = [grads[i] for i in range(len(labels)) if labels[i] == 1]
pos_mean_tensor = torch.mean(torch.stack(pos_grads), dim=0)

neg_grads = [grads[i] for i in range(len(labels)) if labels[i] == 0]
neg_mean_tensor = torch.mean(torch.stack(neg_grads), dim=0)

scaled_grads =torch.mul(dat[:,:,22:], grads)
scaled_mean_grads = torch.mean(scaled_grads, dim=0)

out_df = pd.DataFrame(scaled_mean_grads.numpy(), columns=["pI", "mol_vol", "hydrophobicity", "bond_entropy"])
out_df['pos'] = range(-6, 6+1)
out_df.to_csv("/Users/weeder/PycharmProjects/pepsickle/models/results/mean_grads.csv", index=False)
