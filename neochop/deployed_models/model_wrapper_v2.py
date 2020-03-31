from sequence_featurization_tools import *
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import pandas as pd
import torch.nn.functional as F

model_dir = "/Users/weeder/PycharmProjects/proteasome/neochop/deployed_models"
handle = model_dir + '/trained_model_dict.pickle'
all_mammal = False
_model_dict = pickle.load(open(handle, "rb"))


class epitope_SeqNet(nn.Module):
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


class epitope_MotifNet(nn.Module):
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


class proteasome_SeqNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(p=0.4)
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


class proteasome_MotifNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(p=0.4)
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


def initialize_epitope_model(all_mammal=False):
    # set proper model file
    if all_mammal:
        mod_state = _model_dict['all_mammal_epitope_sequence_mod']
    else:
        mod_state = _model_dict['human_only_epitope_sequence_mod']

    # initialize model
    mod = epitope_SeqNet()
    mod.load_state_dict(mod_state)
    mod.eval()
    return mod


def initialize_proteasome_model():
    # set file paths
    mod1_file = _model_dict['all_mammal_proteasome_sequence_mod']
    mod2_file = _model_dict['all_mammal_proteasome_motif_mod']

    # initialize 1st mod
    mod1 = proteasome_SeqNet()
    mod1.load_state_dict(torch.load(mod1_file))
    mod1.eval()

    # initialize 2nd mod
    mod2 = proteasome_MotifNet()
    mod2.load_state_dict(torch.load(mod2_file))
    mod2.eval()

    return [mod1, mod2]


def predict_epitope_mod(model, features):
    features = torch.from_numpy(features)
    with torch.no_grad():
        p_cleavage = torch.exp(
            model(features[:, :, :20].type(torch.FloatTensor))[:, 1]
        )

    output_p = [float(x) for x in p_cleavage]
    return output_p


def predict_proteasome_mod(model_list, features, proteasome_type="C"):
    # assert features.shape[2] == 24
    features = torch.from_numpy(features)
    mod1 = model_list[0]
    mod2 = model_list[1]

    if proteasome_type == "C":
        c_prot = torch.tensor([1] * features.shape[0]).type(torch.FloatTensor)
        i_prot = torch.tensor([0] * features.shape[0]).type(torch.FloatTensor)

    with torch.no_grad():
        log_p1 = mod1(features[:, :, :20].type(torch.FloatTensor),
                      c_prot, i_prot)[:, 1]
        # NOTE: change if feature matrix is updated
        log_p2 = mod2(features[:, :, 22:].type(torch.FloatTensor),
                      c_prot, i_prot)[:, 1]
        log_avg = (log_p1 + log_p2)/2
        p_cleavage = torch.exp(log_avg)

    output_p = [float(x) for x in p_cleavage]
    return output_p


def create_windows_from_protein(protein_seq):
    # NOTE: last AA not made into window since c-terminal would be cleavage pos
    protein_windows = []
    for pos in range(len(protein_seq)-1):
        start_pos = pos + 1
        end_pos = pos + 2
        tmp_window = get_peptide_window(protein_seq,
                                        starting_position=start_pos,
                                        ending_position=end_pos, upstream=6,
                                        downstream=6)
        protein_windows.append(tmp_window)

    return protein_windows


# create output function that generates table...








handle = "/Users/weeder/PycharmProjects/proteasome/data/validation_data/" \
         "epitope_val_filtered.pickle"
epitope_val_dict = pickle.load(open(handle, "rb"))
epitope_positives = list(epitope_val_dict['positives'].keys())
epitope_positive_features = generate_feature_array(epitope_positives)
epitope_negatives = list(epitope_val_dict['negatives'].keys())
epitope_negative_features =generate_feature_array(epitope_negatives)

epitope_model = initialize_epitope_model(all_mammal=False)
pos_preds = predict_epitope_mod(epitope_model, epitope_positive_features)
neg_preds = predict_epitope_mod(epitope_model, epitope_negative_features)

true_labels = [1] * len(pos_preds) + [0] * len(neg_preds)
true_prob = pos_preds + neg_preds
predicted_label = [p > .5 for p in true_prob]

report = metrics.classification_report(true_labels, predicted_label)
epitope_auc = metrics.roc_auc_score(true_labels, true_prob)
tn, fp, fn, tp = metrics.confusion_matrix(true_labels,
                                          predicted_label).ravel()
sensitivity = tp/(tp + fn)
specificity = tn/(tn+fp)

print("Sensitivity: ", sensitivity)
print("Specificity: ", specificity)
print("AUC: ", epitope_auc)
