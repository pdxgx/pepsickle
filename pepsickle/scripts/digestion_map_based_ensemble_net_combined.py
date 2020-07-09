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
    tensor_attributions = algo.attribure(input, label, **kwargs)
    return tensor_attributions

# set CPU or GPU
dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU


# load in data and set output directory
# indir = "D:/Hobbies/Coding/proteasome_networks/data/"
indir = "/Users/weeder/PycharmProjects/proteasome/data/generated_training_sets/"
file = "/cleavage_windows_all_mammal_13aa.pickle"
out_dir = "/pepsickle/model_weights"
test_holdout_p = .1
n_epoch = 42

# set seed for consistency
torch.manual_seed(123)


# define network structures
class SeqNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(p=0.2)
        self.input = nn.Linear(262, 136)
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


class MotifNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(p=.2)
        self.conv = nn.Conv1d(4, 4, 3, groups=4)
        # self.fc1 = nn.Linear(78, 38)
        self.fc1 = nn.Linear(46, 38)
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


class FullNet(nn.Module):
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

        x = torch.cat((x, c_prot.reshape(c_prot.shape[0], -1)), 1)
        x = torch.cat((x, i_prot.reshape(i_prot.shape[0], -1)), 1)

        x = self.drop(F.relu(self.bn1(self.input(x))))
        x = self.drop(F.relu(self.bn2(self.fc1(x))))
        x = self.drop(F.relu(self.bn3(self.fc2(x))))
        x = F.log_softmax(self.out(x), dim=1)

        return x


# initialize networks
model = FullNet()

# convert to cuda scripts if on GPU
if dtype is torch.cuda.FloatTensor:
    model = model.cuda()

# load in data from pickled dictionary
handle = open(indir + file, "rb")
data = pickle.load(handle)

# create list of cleavage windows
positive_dict = data['proteasome']['positives']
pos_windows = []
for key in positive_dict.keys():
    if key not in pos_windows:
        pos_windows.append(key)

# create list of non-cleavage windows
negative_dict = data['proteasome']['negatives']
neg_windows = []
for key in negative_dict.keys():
    if key not in neg_windows:
        neg_windows.append(key)

# generate lists of proteasome type for each positive example
pos_constitutive_proteasome = []
pos_immuno_proteasome = []
pos_type_list = []
for key in pos_windows:
    proteasome_type = []
    # by default neither proteasome
    c_flag = False
    i_flag = False
    # create list of all associated data entries
    tmp_entry = list(positive_dict[key])

    for entry in tmp_entry:
        # create list of all unique proteasome types recorded for given window
        p_type = entry[1]
        if p_type not in proteasome_type:
            proteasome_type.append(p_type)
    # swap flags were relevant
    if "C" in proteasome_type:
        c_flag = True
        pos_type_list.append("C")
    if "I" in proteasome_type:
        i_flag = True
        pos_type_list.append("I")
    if "M" in proteasome_type:
        c_flag = True
        i_flag = True
        pos_type_list.append("M")

    # based on flags, append binary indicator
    if c_flag:
        pos_constitutive_proteasome.append(1)
    if not c_flag:
        pos_constitutive_proteasome.append(0)

    if i_flag:
        pos_immuno_proteasome.append(1)
    if not i_flag:
        pos_immuno_proteasome.append(0)


# repeat above for non-cleavage windows
neg_constitutive_proteasome = []
neg_immuno_proteasome = []
neg_type_list = []
for key in neg_windows:
    proteasome_type = []
    c_flag = False
    i_flag = False
    tmp_entry = list(negative_dict[key])

    for entry in tmp_entry:
        p_type = entry[1]
        if p_type not in proteasome_type:
            proteasome_type.append(p_type)

    if "C" in proteasome_type:
        c_flag = True
        neg_type_list.append("C")
    if "I" in proteasome_type:
        i_flag = True
        neg_type_list.append("I")
    if "M" in proteasome_type:
        c_flag = True
        i_flag = True
        neg_type_list.append("M")

    if c_flag:
        neg_constitutive_proteasome.append(1)
    if not c_flag:
        neg_constitutive_proteasome.append(0)

    if i_flag:
        neg_immuno_proteasome.append(1)
    if not i_flag:
        neg_immuno_proteasome.append(0)

# generate feature set for cleavage windows
pos_feature_matrix = torch.from_numpy(generate_feature_array(pos_windows))
# append proteasome type indicators
pos_feature_set = [f_set for f_set in zip(pos_feature_matrix,
                                          pos_constitutive_proteasome,
                                          pos_immuno_proteasome)]

# generate feature set for non-cleavage windows
neg_feature_matrix = torch.from_numpy(generate_feature_array(neg_windows))
neg_feature_set = [f_set for f_set in zip(neg_feature_matrix,
                                          neg_constitutive_proteasome,
                                          neg_immuno_proteasome)]

# set number of positive and negative examples based on test proportion
pos_train_k = torch.tensor(round((1-test_holdout_p) * len(pos_feature_set)))
neg_train_k = torch.tensor(round((1-test_holdout_p) * len(pos_feature_set)))

# permute and split data
pos_perm = torch.randperm(torch.tensor(len(pos_feature_set)))
pos_train = [pos_feature_set[i] for i in pos_perm[:pos_train_k]]
pos_test = [pos_feature_set[i] for i in pos_perm[pos_train_k:]]

neg_perm = torch.randperm(torch.tensor(len(neg_feature_set)))
neg_train = [neg_feature_set[i] for i in neg_perm[:neg_train_k]]
# force balanced testing set
neg_test = [neg_feature_set[i] for i in neg_perm[neg_train_k:(torch.tensor(
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
mod_criterion = nn.NLLLoss(
    weight=torch.tensor([1, len(neg_train)/len(pos_train)]).type(dtype))
mod_optimizer = optim.Adam(model.parameters(), lr=.001)

# train
prev_auc = 0
for epoch in range(n_epoch):
    # reset running loss for each epoch
    mod_running_loss = 0
    # load data, convert if needed
    for dat, labels in train_loader:
        # convert to proper data type
        matrix_dat = dat[0].type(dtype)
        c_proteasome_dat = dat[1].clone().detach().type(dtype)
        i_proteasome_dat = dat[2].clone().detach().type(dtype)
        if dtype == torch.cuda.FloatTensor:
            labels = labels.cuda()

        # reset gradients
        mod_optimizer.zero_grad()

        # generate model predictions
        mod_est = model(matrix_dat,
                        c_proteasome_dat,
                        i_proteasome_dat)  # one hot encoded sequences

        # calculate loss
        mod_loss = mod_criterion(mod_est, labels)

        # back prop loss and step
        mod_loss.backward()
        mod_optimizer.step()

        mod_running_loss += mod_loss.item()

    else:
        # output progress
        print(f'Epoch: {epoch + 1}')
        print(f'Model Running Loss: {mod_running_loss}')

        # test with no grad to speed up
        with torch.no_grad():
            # set to eval mode
            model.eval()
            dat, labels = next(iter(test_loader))

            # convert to proper data type
            matrix_dat = dat[0].type(dtype)
            # extract proteasome type parameters
            c_proteasome_dat = dat[1].clone().detach().type(dtype)
            i_proteasome_dat = dat[2].clone().detach().type(dtype)
            # convert labels if on GPU
            if dtype == torch.cuda.FloatTensor:
                labels = labels.cuda()

            # get est probability of cleavage event
            exp_mod_est = torch.exp(model(
                matrix_dat, c_proteasome_dat,
                i_proteasome_dat))[:, 1].cpu()  # one hot encoded sequences

            # calculate AUC for each
            mod_auc = metrics.roc_auc_score(labels, exp_mod_est)

            # store if most performant model so far
            if mod_auc > prev_auc:
                mod_state = model.state_dict()
                prev_auc = mod_auc

            # print out performance
            print("Test Set Results:")
            print(f'Model AUC: {mod_auc}')
            print("\n")

    # return to training mode
    model.train()

# save model states to file
torch.save(mod_state, out_dir + "/all_mammal_cleavage_map_full_mod.pt")

## identify feature importance

# define also
"""
saliency = Saliency(motif_model)
grads = saliency.attribute((dat[0][:, :, 22:], dat[1].float(), dat[2].float()), target=labels)
grads = np.transpose(grads.squeeze().cpu)
"""

