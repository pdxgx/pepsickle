#!/usr/bin/env python3
"""
epitope_based_ensemble_net.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script contains trains neural networks based on both the sequence identity
and physical property motifs of cleavage and non-cleavage examples. Exports
trained model wieghts and (optionally) aggregated weights for the first input
layer of the sequence based model.
"""
from sequence_featurization_tools import *
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import pandas as pd
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# set CPU or GPU
dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# prep data
# indir = "D:/Hobbies/Coding/proteasome_networks/data/"
in_dir = "/Users/weeder/PycharmProjects/proteasome/data/generated_training_sets"
out_dir = "/Users/weeder/PycharmProjects/proteasome/neochop/results"
file = "/cleavage_windows_human_only_13aa.pickle"
test_holdout_p = .2  # proportion of data held out for testing set
n_epoch = 26

# set seed for consistency
torch.manual_seed(123)

# define model structures
class SeqNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(p=0.3)
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
        self.drop = nn.Dropout(p=0.3)
        self.conv = nn.Conv1d(4, 4, 3, groups=4)
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

# convert models to cuda if on GPU
if dtype is torch.cuda.FloatTensor:
    sequence_model = sequence_model.cuda()
    motif_model = motif_model.cuda()

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
seq_criterion = nn.NLLLoss(
    weight=torch.tensor([1, len(neg_train)/len(pos_train)]).type(dtype))
seq_optimizer = optim.Adam(sequence_model.parameters(), lr=.001)

motif_criterion = nn.NLLLoss\
    (weight=torch.tensor([1, len(neg_train)/len(pos_train)]).type(dtype))
motif_optimizer = optim.Adam(motif_model.parameters(), lr=.001)

# initialize tracking of optimal models and train
prev_seq_auc = 0
prev_motif_auc = 0

for epoch in range(n_epoch):
    # reset running loss for each epoch
    seq_running_loss = 0
    motif_running_loss = 0
    # load data, convert if needed
    for dat, labels in train_loader:
        # convert to proper data type
        dat = dat.type(dtype)
        if dtype == torch.cuda.FloatTensor:
            labels = labels.cuda()

        # reset gradients
        seq_optimizer.zero_grad()
        motif_optimizer.zero_grad()

        # generate model predictions
        seq_est = sequence_model(dat[:, :, :20])  # one hot encoded sequences
        # with torch.no_grad():
        #     motif_dat = conv_pre(dat[:, :, 22:].transpose(1, 2))
        motif_est = motif_model(dat[:, :, 22:])  # physical properties (not side chains)

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
            sequence_model.eval()
            motif_model.eval()
            dat, labels = next(iter(test_loader))
            # convert to proper data type
            dat = dat.type(dtype)

            # get est probability of cleavage event
            exp_seq_est = torch.exp(sequence_model(dat[:, :, :20]))[:, 1].cpu()
            # motif_dat = conv_pre(dat[:, :, 22:].transpose(1, 2))
            exp_motif_est = torch.exp(motif_model(dat[:, :, 22:]))[:, 1].cpu()
            # take simple average
            consensus_est = (exp_seq_est +
                             exp_motif_est) / 2

            # calculate AUC for each
            seq_auc = metrics.roc_auc_score(labels, exp_seq_est)
            motif_auc = metrics.roc_auc_score(labels, exp_motif_est)
            consensus_auc = metrics.roc_auc_score(labels, consensus_est)

            # store model if perfomance is better than current best
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

    # return to train mode for next iteration
    sequence_model.train()
    motif_model.train()


# look at ultimate performance
t_dat, t_labels = next(iter(test_loader))
# reset model states to best performance
sequence_model.load_state_dict(seq_state)
sequence_model.eval()
motif_model.load_state_dict(motif_state)
motif_model.eval()

# performance with seq only was best so use to determine performance
seq_est = torch.exp(sequence_model(t_dat.type(dtype)[:, :, :20]))[:, 1].cpu()
# call classes so that sensitivity and specificity can be calculated
seq_guess_class = []
for est in seq_est:
    if est >= .5:  # changing threshold alters se and sp, .5 = default
        seq_guess_class.append(1)
    else:
        seq_guess_class.append(0)

# print AUC
seq_auc = metrics.roc_auc_score(t_labels.detach().numpy(),
                                seq_est.detach().numpy())
print(seq_auc)

# Print classification report
seq_report = metrics.classification_report(t_labels.detach().numpy(),
                                           seq_guess_class)
print(seq_report)

# calculate and print se and sp
tn, fp, fn, tp = metrics.confusion_matrix(t_labels.detach().numpy(),
                                          seq_guess_class).ravel()
sensitivity = tp/(tp + fn)
specificity = tn/(tn+fp)

print("Sensitivity: ", sensitivity)
print("Specificity: ", specificity)

# save model states to file
torch.save(seq_state, out_dir + "/human_only_epitope_sequence_mod.pt")
torch.save(motif_state, out_dir + "/human_only_epitope_motif_mod.pt")


# generate plot of weights
# look at position weights to see what areas are weighted most
in_layer_weights = sequence_model.input.weight
# sum across all second layer connections
in_layer_weights = in_layer_weights.abs().sum(dim=0)
# reshape to match original input
in_layer_weights = in_layer_weights.reshape(13, -1)
# sum across values per position
input_sums = in_layer_weights.sum(dim=1).detach().numpy()

# plot
positions = range(-6, 7)
y_pos = np.arange(len(positions))
plt.bar(y_pos, input_sums, align='center', alpha=0.5)
plt.xticks(y_pos, positions)
plt.ylabel('weight')
plt.xlabel('distance from cleavage point')
plt.title('')

plt.show()


physical_mod_weights = motif_model.fc1.weight.abs().sum(dim=0)
physical_mod_weights = physical_mod_weights.reshape(13, -1)
test = physical_mod_weights[:, 2].detach().numpy()

# generate weight table for export and plotting
pos_list = []
weights = []
grouping = []
for i in range(2, 6):
    tmp = physical_mod_weights[:, i].detach().numpy()
    for val in tmp:
        weights.append(val)
        grouping.append(i)
    for pos in positions:
        pos_list.append(pos)

out_df = pd.DataFrame(zip(pos_list, weights, grouping),
                      columns=['position', 'weight', 'group'])

# uncomment to export aggregate weights for first layer
# out_df.to_csv(out_dir + "/physical_property_weights.csv", index=False)
