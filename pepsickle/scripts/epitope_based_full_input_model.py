#!/usr/bin/env python3
"""
epitope_based_ensemble_net.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script contains trains neural networks based on both the sequence identity
and physical property motifs of cleavage and non-cleavage examples from epitope
databases. Exports trained model wieghts.
"""
from sequence_featurization_tools import *
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# set CPU or GPU
dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# prep data
# indir = "D:/Hobbies/Coding/proteasome_networks/data/"
in_dir = "/Users/weeder/PycharmProjects/proteasome/data/generated_training_sets"
out_dir = "/pepsickle/model_weights"
file = "/cleavage_windows_human_only_13aa.pickle"
test_holdout_p = .2  # proportion of data held out for testing set
n_epoch = 42

# set seed for consistency
torch.manual_seed(123)


class FullNet(nn.Module):
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

        x = self.drop(F.relu(self.bn1(self.input(x))))
        x = self.drop(F.relu(self.bn2(self.fc1(x))))
        x = self.drop(F.relu(self.bn3(self.fc2(x))))
        x = F.log_softmax(self.out(x), dim=1)

        return x





# initialize networks
model = FullNet()

# convert models to cuda if on GPU
if dtype is torch.cuda.FloatTensor:
    model = model.cuda()

#  open pickled dictionary and load in data
handle = open(in_dir + file, "rb")
data = pickle.load(handle)
# subset to epitope data for this model
data = data['epitope']

# create list of cleavage windows
pos_windows = []
for key in data['positives'].keys():
    pos_windows.append(key)

# generate features
pos_feature_matrix = torch.from_numpy(generate_feature_array(pos_windows))

# create list of non cleavage windows
neg_windows = []
neg_digestion_windows = []
for key in data['negatives'].keys():
    neg_windows.append(key)

# generate features
neg_feature_matrix = torch.from_numpy(generate_feature_array(neg_windows))

# define number of training cases based on holdout (unbalanced)
pos_train_k = round((1-test_holdout_p) * pos_feature_matrix.size(0))
neg_train_k = round((1-test_holdout_p) * neg_feature_matrix.size(0))

# permute and split data
pos_perm = torch.randperm(pos_feature_matrix.size(0))
pos_train = pos_feature_matrix[pos_perm[:pos_train_k]]
pos_test = pos_feature_matrix[pos_perm[pos_train_k:]]

neg_perm = torch.randperm(neg_feature_matrix.size(0))
neg_train = neg_feature_matrix[neg_perm[:neg_train_k]]
# use same number of negative test examples as positives for balanced set
neg_test = neg_feature_matrix[neg_perm[neg_train_k:(pos_test.size(0) +
                                                    neg_train_k)]]

# pair training data with labels
pos_train_labeled = []
for i in range(len(pos_train)):
    pos_train_labeled.append([pos_train[i], torch.tensor(1)])
neg_train_labeled = []
for i in range(len(neg_train)):
    neg_train_labeled.append([neg_train[i], torch.tensor(0)])

# combine cleavage and non-cleavage train examples
train_data = pos_train_labeled + neg_train_labeled
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64,
                                           shuffle=True)

# pair test data with labels
pos_test_labeled = []
for i in range(len(pos_test)):
    pos_test_labeled.append([pos_test[i], torch.tensor(1)])
neg_test_labeled = []
for i in range(len(neg_test)):
    neg_test_labeled.append([neg_test[i], torch.tensor(0)])

# combine cleavage and non-cleavage test examples
test_data = pos_test_labeled + neg_test_labeled
# set batch size to give unique set with no re-use
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=int(len(test_data)/n_epoch), shuffle=True)

# establish training parameters
# inverse weighting for class imbalance in training set
mod_criterion = nn.NLLLoss(
    weight=torch.tensor([1, len(neg_train)/len(pos_train)]).type(dtype))
mod_optimizer = optim.Adam(model.parameters(), lr=.001)

# initialize tracking of optimal models and train
prev_mod_auc = 0

for epoch in range(n_epoch):
    # reset running loss for each epoch
    mod_running_loss = 0
    # load data, convert if needed
    for dat, labels in train_loader:
        # convert to proper data type
        dat = dat.type(dtype)
        if dtype == torch.cuda.FloatTensor:
            labels = labels.cuda()

        # reset gradients
        mod_optimizer.zero_grad()

        # generate model predictions
        mod_est = model(dat)  # one hot encoded sequences

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
            model.eval()
            dat, labels = next(iter(test_loader))
            # convert to proper data type
            dat = dat.type(dtype)

            # get est probability of cleavage event
            exp_mod_est = torch.exp(model(dat))[:, 1].cpu()

            # calculate AUC for each
            mod_auc = metrics.roc_auc_score(labels, exp_mod_est)

            # store model if perfomance is better than current best
            if mod_auc > prev_mod_auc:
                mod_state = model.state_dict()
                prev_seq_auc = mod_auc


            # print out performance
            print("Test Set Results:")
            print(f'Sequence Model AUC: {mod_auc}')
            print("\n")

    # return to train mode for next iteration
    model.train()


# save model states to file
torch.save(mod_state, out_dir + "/human_only_epitope_full_mod.pt")


