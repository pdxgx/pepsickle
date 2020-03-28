import pandas as pd
from extraction_functions import *

in_dir = "/Users/weeder/PycharmProjects/proteasome/data/validation_data"
out_dir = "/Users/weeder/PycharmProjects/proteasome/data/validation_data"
in_file = "/41467_2016_BFncomms13404_MOESM1318_ESM.csv"
handle = in_dir + in_file
dat = pd.read_csv(handle, low_memory=False, skiprows=0, header=1)

class_I_columns = []
for i in range(len(dat.columns)):
    col = dat.columns[i]
    if "_HLA-I " in col:
        class_I_columns.append(i)

total_class_I_intensity = dat[dat.columns[class_I_columns]].sum(axis=1)
dat = dat[total_class_I_intensity > 0]
cols_to_keep = ['Start position', 'End position', 'Sequence', 'Proteins']
dat = dat[cols_to_keep]
dat = dat.dropna()

unique_maps = []
for p_id in dat['Proteins']:
    id_count = p_id.count(";")
    if id_count == 0:
        unique_maps.append(True)
    else:
        unique_maps.append(False)

unambiguous_dat = dat[unique_maps]
unambiguous_dat.reset_index(inplace=True)

source_sequences = {}
failed_entries = []
unique_prots = unambiguous_dat["Proteins"].unique()
for p in range(len(unique_prots)):
    p_id = unique_prots[p]
    try:
        entry = retrieve_UniProt_seq(p_id)
        source_sequences[p_id] = entry
    except:
        failed_entries.append(p_id)

    if p % 100 == 0:
        print(round(p/len(unique_prots), 3))

unambiguous_dat["full_sequence"] = None
for e in range(len(unambiguous_dat)):
    # pull protein ID
    prot_id = str(unambiguous_dat.loc[e, "Proteins"])
    # if sequence was found for given ID, store full sequence
    if prot_id in source_sequences.keys():
        unambiguous_dat.loc[e, 'full_sequence'] = source_sequences[prot_id]

unambiguous_dat = unambiguous_dat.dropna()


new_cols = ['index', 'start_pos',  'end_pos', 'fragment', 'full_seq_accession',
            'full_sequence']
unambiguous_dat.columns = new_cols
unambiguous_dat['entry_source'] = "validation"

unambiguous_dat.to_csv(out_dir + "/validation_epitopes_w_source.csv",
                       index=False)
