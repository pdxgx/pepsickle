from sequence_featurization_tools import *
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import torch.nn.functional as F

# set CPU or GPU
dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# prep data
# indir = "D:/Hobbies/Coding/proteasome_networks/data/"
indir = "/Users/weeder/PycharmProjects/proteasome/data_processing/generated_training_sets/"
file = "proteasome_data_1.6.20.pickle"

torch.manual_seed(123)

class SeqNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(p=0.3)
        self.input = nn.Linear(420, 136)
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
        self.fc1 = nn.Linear(76, 38)
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


class MotifNetNoConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(84, 38)
        self.bn1 = nn.BatchNorm1d(38)
        self.fc2 = nn.Linear(38, 20)
        self.bn2 = nn.BatchNorm1d(20)
        self.out = nn.Linear(20, 2)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.reshape(x.shape[0], -1)

        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        x = F.log_softmax(self.out(x), dim=1)

        return x


# initialize networks
sequence_model = SeqNet()
motif_model = MotifNet()
# motif_model = MotifNetNoConv()
# conv_pre = nn.Conv1d(4, 4, 3, groups=4)

# convert to cuda models if on GPU
if dtype is torch.cuda.FloatTensor:
    sequence_model = sequence_model.cuda()
    motif_model = motif_model.cuda()


handle = open(indir + file, "rb")
data = pickle.load(handle)

# for human only...
pos_windows = []
for key in data['positives'].keys():
    entry = data['positives'][key]
    if any('human' in i for i in entry):
        if any('cleavage map' in i for i in entry):
            pos_windows.append(key)

# for all mammals
# pos_windows = list(data['positives'].keys())
pos_feature_matrix = torch.from_numpy(generate_feature_array(pos_windows))

neg_windows = []
for key in data['negatives'].keys():
    entry = data['negatives'][key]
    if any('human' in i for i in entry):
        if any('cleavage map' in i for i in entry):
            neg_windows.append(key)

# neg_windows = list(data['negatives'].keys())
neg_feature_matrix = torch.from_numpy(generate_feature_array(neg_windows))

test_holdout_p = .2
pos_train_k = round((1-test_holdout_p) * pos_feature_matrix.size(0))
neg_train_k = round((1-test_holdout_p) * neg_feature_matrix.size(0))

# permute and split data
pos_perm = torch.randperm(pos_feature_matrix.size(0))
pos_train = pos_feature_matrix[pos_perm[:pos_train_k]]
pos_test = pos_feature_matrix[pos_perm[pos_train_k:]]

neg_perm = torch.randperm(neg_feature_matrix.size(0))
neg_train = neg_feature_matrix[neg_perm[:neg_train_k]]
neg_test = neg_feature_matrix[neg_perm[neg_train_k:(pos_test.size(0) + neg_train_k)]]  # for balanced testing set

# pair training data with labels
pos_train_labeled = []
for i in range(len(pos_train)):
    pos_train_labeled.append([pos_train[i], torch.tensor(1)])
neg_train_labeled = []
for i in range(len(neg_train)):
    neg_train_labeled.append([neg_train[i], torch.tensor(0)])

train_data = pos_train_labeled + neg_train_labeled
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)  # was 20, then 64

# pair test data with labels
pos_test_labeled = []
for i in range(len(pos_test)):
    pos_test_labeled.append([pos_test[i], torch.tensor(1)])
neg_test_labeled = []
for i in range(len(neg_test)):
    neg_test_labeled.append([neg_test[i], torch.tensor(0)])

test_data = pos_test_labeled + neg_test_labeled
test_loader = torch.utils.data.DataLoader(test_data, batch_size=2000, shuffle=True)  # was len * 0.10

pos_test_loader = torch.utils.data.DataLoader(pos_test_labeled, batch_size=2000, shuffle=True)
neg_test_loader = torch.utils.data.DataLoader(neg_test_labeled, batch_size=2000, shuffle=True)

# establish training parameters
# inverse weighting for class imbalance in training set
seq_criterion = nn.NLLLoss(weight=torch.tensor([1, len(neg_train)/len(pos_train)]).type(dtype))
seq_optimizer = optim.Adam(sequence_model.parameters(), lr=.001)

motif_criterion = nn.NLLLoss(weight=torch.tensor([1, len(neg_train)/len(pos_train)]).type(dtype))
motif_optimizer = optim.Adam(motif_model.parameters(), lr=.001)

# train
n_epoch = 12
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
            exp_seq_est = torch.exp(sequence_model(dat[:, :, :20]))[:, 1].cpu()  # one hot encoded sequences
            # motif_dat = conv_pre(dat[:, :, 22:].transpose(1, 2))
            exp_motif_est = torch.exp(motif_model(dat[:, :, 22:]))[:, 1].cpu()  # not including side chains
            # take simple average
            consensus_est = (exp_seq_est + exp_motif_est) / 2

            # calculate AUC for each
            seq_auc = metrics.roc_auc_score(labels, exp_seq_est)
            motif_auc = metrics.roc_auc_score(labels, exp_motif_est)
            consensus_auc = metrics.roc_auc_score(labels, consensus_est)

            # print out performance
            print("Test Set Results:")
            print(f'Sequence Model AUC: {seq_auc}')
            print(f'Motif Model AUC: {motif_auc}')
            print(f'Consensus Model AUC: {consensus_auc}')
            print("\n")

    sequence_model.train()
    motif_model.train()
