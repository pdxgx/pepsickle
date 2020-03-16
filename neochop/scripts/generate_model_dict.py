import torch
import os
import pickle

in_dir = "/Users/weeder/PycharmProjects/proteasome/neochop/results"
out_dir = "/Users/weeder/PycharmProjects/proteasome/neochop/deployed_models"
model_dict = {}
for f in os.listdir(in_dir):
    if not f.startswith("."):
        path = in_dir + "/" + f
        mod_name, ext = os.path.splitext(f)
        mod = torch.load(path)
        model_dict[mod_name] = mod

out_file = out_dir + "/trained_model_dict.pickle"
pickle.dump(model_dict, open(out_file, 'wb'))
