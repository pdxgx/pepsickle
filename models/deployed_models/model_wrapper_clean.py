from sequence_featurization_tools import *
import pandas as pd
from Bio import SeqIO
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

model_dir = "/models/deployed_models"
handle = model_dir + '/trained_model_dict.pickle'
all_mammal = False
_model_dict = pickle.load(open(handle, "rb"))


class epitope_FullNet(nn.Module):
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


class proteasome_FullNet(nn.Module):
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


def initialize_epitope_model(all_mammal=False):
    # set proper model file
    if all_mammal:
        mod_state = _model_dict['all_mammal_epitope_full_mod']
    else:
        mod_state = _model_dict['human_only_epitope_full_mod']

    # initialize model
    mod = epitope_FullNet()
    mod.load_state_dict(mod_state)
    mod.eval()
    return mod


def initialize_digestion_model(all_mammal=True):
    # set proper model file
    if all_mammal:
        mod_state = _model_dict['all_mammal_cleavage_map_full_mod']
    else:
        mod_state = _model_dict['human_only_cleavage_map_full_mod']

    # initialize model
    mod = proteasome_FullNet()
    mod.load_state_dict(mod_state)
    mod.eval()
    return mod


def predict_epitope_mod(model, features):
    features = torch.from_numpy(features)
    with torch.no_grad():
        p_cleavage = torch.exp(
            model(features.type(torch.FloatTensor))[:, 1]
        )

    output_p = [float(x) for x in p_cleavage]
    return output_p


def predict_digestion_mod(model, features, proteasome_type="C"):
    # assert features.shape[2] == 24
    features = torch.from_numpy(features)
    mod1 = model

    if proteasome_type == "C":
        c_prot = torch.tensor([1] * features.shape[0]).type(torch.FloatTensor)
        i_prot = torch.tensor([0] * features.shape[0]).type(torch.FloatTensor)
    if proteasome_type == "I":
        c_prot = torch.tensor([0] * features.shape[0]).type(torch.FloatTensor)
        i_prot = torch.tensor([1] * features.shape[0]).type(torch.FloatTensor)

    with torch.no_grad():
        log_p1 = mod1(features.type(torch.FloatTensor),
                      c_prot, i_prot)[:, 1]
        p_cleavage = torch.exp(log_p1)

    output_p = [float(x) for x in p_cleavage]
    return(output_p)


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


def predict_protein_cleavage_locations(protein_id, protein_seq, model,
                                       mod_type="epitope",
                                       proteasome_type=None,
                                       threshold=.5):
    protein_windows = create_windows_from_protein(protein_seq)
    window_features = generate_feature_array(protein_windows)

    if mod_type == "epitope":
        preds = predict_epitope_mod(model, window_features)
    if mod_type == "digestion":
        preds = predict_digestion_mod(model, window_features,
                                      proteasome_type=proteasome_type)
    positions = range(1, len(preds)+1)
    cleave = [p > threshold for p in preds]
    out_df = pd.DataFrame(zip(positions, preds, cleave),
                          columns=["pos", "p_cleavage", "cleaved"])
    out_df['prot_id'] = protein_id
    return out_df


# start
active_mod = initialize_epitope_model()
file_handle = "/Users/weeder/Downloads/human_proteome.fasta"

protein_list = SeqIO.to_dict(SeqIO.parse(file_handle, "fasta"))

end = len(protein_list)
protein_preds = []
counter = 0
for protein in protein_list:
    if counter % 1000 == 0:
        print("completion:", counter, "of", end)
    tmp_out = predict_protein_cleavage_locations(protein,
                                                 protein_list[protein],
                                                 active_mod)
    protein_preds.append(tmp_out)
    counter += 1

out_df = pd.concat(protein_preds)

out_df.to_csv("/Users/weeder/Downloads/human_proteome_our_cleavage_preds.csv",
              index=False)
