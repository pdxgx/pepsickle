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
indir = "/Users/weeder/PycharmProjects/proteasome/data_modeling/merged_data/"
file = "tmp_data_full_negative.pickle"

handle = open(indir + file, "rb")
data = pickle.load(handle)


pos_windows = list(data['positives'].keys())
pos_feature_matrix = torch.from_numpy(generate_feature_array(pos_windows))
with torch.no_grad():
    pos_seq_matrix = pos_feature_matrix[:, :, :20]
    pos_seq_matrix = pos_seq_matrix.reshape(pos_seq_matrix.size(0), -1)

neg_windows = list(data['negatives'].keys())
neg_feature_matrix = torch.from_numpy(generate_feature_array(neg_windows))
with torch.no_grad():
    neg_seq_matrix = neg_feature_matrix[:, :, :20]
    neg_seq_matrix = neg_seq_matrix.reshape(neg_seq_matrix.size(0), -1)


test_holdout_p = .2
pos_k = round((1-test_holdout_p) * pos_seq_matrix.size(0))
# for balanced data
# neg_k = pos_k

neg_k = round((1-test_holdout_p) * neg_seq_matrix.size(0))

# permute and split data
pos_perm = torch.randperm(pos_seq_matrix.size(0))
pos_train = pos_seq_matrix[pos_perm[:pos_k]]
pos_test = pos_seq_matrix[pos_perm[pos_k:]]

neg_perm = torch.randperm(neg_seq_matrix.size(0))
neg_train = neg_seq_matrix[neg_perm[:neg_k]]
# for balanced data
neg_test = neg_seq_matrix[neg_perm[neg_k:(pos_test.size(0) + neg_k)]]
# neg_test = neg_seq_matrix[neg_perm[neg_k:]]

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
sequence_model = nn.Sequential(nn.Linear(420, 136),
                           nn.BatchNorm1d(136),
                           nn.ReLU(),
                           nn.Dropout(p=.3),
                           nn.Linear(136, 68),
                           nn.BatchNorm1d(68),
                           nn.ReLU(),
                           nn.Dropout(p=.3),
                           nn.Linear(68, 34),
                           nn.BatchNorm1d(34),
                           nn.ReLU(),
                           nn.Dropout(p=.3),
                           nn.Linear(34, 2),
                           nn.LogSoftmax(dim=1))

if dtype == torch.cuda.FloatTensor:
    sequence_model = sequence_model.cuda()

criterion = nn.NLLLoss(weight=torch.tensor([1, len(neg_train)/len(pos_train)]).type(dtype))
optimizer = optim.Adam(sequence_model.parameters(), lr=.001)

n_epoch = 12
for epoch in range(n_epoch):
    running_loss = 0
    for dat, labels in train_loader:
        if dtype == torch.cuda.FloatTensor:
            labels = labels.cuda()
        optimizer.zero_grad()
        est = sequence_model(dat.type(dtype))
        loss = criterion(est, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(epoch + 1)
        print(running_loss)
        with torch.no_grad():
            sequence_model.eval()
            # dat, labels = next(iter(test_loader))
            # if dtype == torch.cuda.FloatTensor:
                # labels = labels.cuda()
            pos_out = torch.exp(sequence_model(pos_test.type(dtype)))
            neg_out = torch.exp(sequence_model(neg_test.type(dtype)))
            # output = torch.exp(sequence_model(dat.type(dtype)))
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
    sequence_model.train()

seq_pos_p = [ex[1] for ex in pos_out[:2000]]
seq_neg_p = [ex[1] for ex in neg_out[:2000]]
seq_examples = seq_pos_p + seq_neg_p
seq_true_labs = [1] * 2000 + [0] * 2000
print(metrics.roc_auc_score(seq_true_labs, seq_examples))
