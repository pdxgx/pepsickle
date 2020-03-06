from sequence_featurization_tools import *
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics

# torch.manual_seed(123)
torch.manual_seed(12)

dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# indir = "D:/Hobbies/Coding/proteasome_networks/data/"
indir = "/Users/weeder/PycharmProjects/proteasome/data_processing/merged_data/"
file = "tmp_data_full_negative.pickle"

handle = open(indir + file, "rb")
data = pickle.load(handle)

phys_conv_layer = nn.Conv1d(4, 4, 3, groups=4).type(torch.FloatTensor) # was 3
# phys_maxpool = nn.MaxPool1d(kernel_size=3, stride=1).type(torch.FloatTensor) # was none

pos_windows = list(data['positives'].keys())
pos_feature_matrix = torch.from_numpy(generate_feature_array(pos_windows))
with torch.no_grad():
    pos_phys_properties = phys_conv_layer(pos_feature_matrix[:, :, 22:].transpose(
        1, 2).type(torch.FloatTensor))
    # pos_phys_properties = phys_maxpool(pos_phys_properties)
    pos_phys_properties = pos_phys_properties.reshape(pos_phys_properties.size(0), -1)

neg_windows = list(data['negatives'].keys())
neg_feature_matrix = torch.from_numpy(generate_feature_array(neg_windows))
with torch.no_grad():
    neg_phys_properties = phys_conv_layer(neg_feature_matrix[:, :, 22:].transpose(
        1, 2).type(torch.FloatTensor))
    # neg_phys_properties = phys_maxpool(neg_phys_properties)
    neg_phys_properties = neg_phys_properties.reshape(neg_phys_properties.size(0), -1)


test_holdout_p = .2
pos_k = round((1-test_holdout_p) * pos_phys_properties.size(0))
# for balanced data
# neg_k = pos_k

neg_k = round((1-test_holdout_p) * neg_phys_properties.size(0))

# permute and split data
pos_perm = torch.randperm(pos_phys_properties.size(0))
pos_train = pos_phys_properties[pos_perm[:pos_k]]
pos_test = pos_phys_properties[pos_perm[pos_k:]]

neg_perm = torch.randperm(neg_phys_properties.size(0))
neg_train = neg_phys_properties[neg_perm[:neg_k]]
# for balanced data
neg_test = neg_phys_properties[neg_perm[neg_k:(pos_test.size(0) + neg_k)]]
# neg_test = neg_phys_properties[neg_perm[neg_k:]]

# pair training data with labels
pos_train_labeled = []
for i in range(len(pos_train)):
    pos_train_labeled.append([pos_train[i], torch.tensor(1)])
neg_train_labeled = []
for i in range(len(neg_train)):
    neg_train_labeled.append([neg_train[i], torch.tensor(0)])
train_data = pos_train_labeled + neg_train_labeled
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True) # was 20

# pair test data with labels
pos_test_labeled = []
for i in range(len(pos_test)):
    pos_test_labeled.append([pos_test[i], torch.tensor(1)])
neg_test_labeled = []
for i in range(len(neg_test)):
    neg_test_labeled.append([neg_test[i], torch.tensor(0)])
test_data = pos_test_labeled + neg_test_labeled
test_loader = torch.utils.data.DataLoader(test_data, batch_size=200, shuffle=True)

# train
properties_model = nn.Sequential(nn.Linear(76, 38), # was 76 (68 if maxpool)
                           nn.BatchNorm1d(38),
                           nn.ReLU(),
                           nn.Dropout(p=.3),
                           nn.Linear(38, 20),
                           nn.BatchNorm1d(20),
                           nn.ReLU(),
                           nn.Dropout(p=.3),
                           nn.Linear(20, 2),
                           nn.LogSoftmax(dim=1))

if dtype == torch.cuda.FloatTensor:
    properties_model = properties_model.cuda()

criterion = nn.NLLLoss(weight=torch.tensor([1, len(neg_train)/len(pos_train)]).type(dtype))
optimizer = optim.Adam(properties_model.parameters(), lr=.001)

n_epoch = 12
for epoch in range(n_epoch):
    running_loss = 0
    for dat, labels in train_loader:
        if dtype == torch.cuda.FloatTensor:
            labels = labels.cuda()
        optimizer.zero_grad()
        est = properties_model(dat.type(dtype))
        loss = criterion(est, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(epoch + 1)
        print(running_loss)
        with torch.no_grad():
            properties_model.eval()
            # dat, labels = next(iter(test_loader))
            # if dtype == torch.cuda.FloatTensor:
                # labels = labels.cuda()
            pos_out = torch.exp(properties_model(pos_test.type(dtype)))
            neg_out = torch.exp(properties_model(neg_test.type(dtype)))
            # output = torch.exp(properties_model(dat.type(dtype)))
            top_p, top_pos_class = pos_out.topk(1, dim=1)
            top_p, top_neg_class = neg_out.topk(1, dim=1)
            # match = top_class == labels.view(*top_class.shape)
            pos_match = top_pos_class == 1
            neg_match = top_neg_class == 0
            # accuracy = torch.mean(match.type(dtype))
            pos_accuracy = torch.mean(pos_match.type(dtype))
            neg_accuracy = torch.mean(neg_match.type(dtype))
            print(f'Positive set accuracy: {pos_accuracy.item() * 100}%')
            print(f'Negative set accuracy: {neg_accuracy.item() * 100}%')
    properties_model.train()

# calculate end AUC
conv_pos_p = [ex[1] for ex in pos_out[:2000]]
conv_neg_p = [ex[1] for ex in neg_out[:2000]]
conv_examples = conv_pos_p + conv_neg_p
conv_true_labs = [1] * 2000 + [0] * 2000
print(metrics.roc_auc_score(conv_true_labs, conv_examples))


"""
conv_tensor = torch.tensor(conv_examples)
seq_examples = torch.tensor(seq_examples)
consensus = (conv_tensor + seq_examples)/2
print(metrics.roc_auc_score(conv_true_labs, consensus))
"""
