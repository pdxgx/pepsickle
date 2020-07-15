import pandas as pd
infile = "/Users/weeder/PycharmProjects/pepsickle/data/merged_data/merged_data_all_mammal_indices_repaired.csv"

dat = pd.read_csv(infile, low_memory=False)
dat.dropna(subset=['Proteasome'], inplace=True)
dat = dat[dat['Proteasome'] != "M"]




dat.to_csv("/Users/weeder/PycharmProjects/pepsickle/data/merged_data/merged_data_all_mammal_no_mix_only_digestion.csv", index=False)
