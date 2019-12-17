import os
import re
import pandas as pd

# Later replace file_dir with -i argparse
file_dir = "/Users/weeder/PycharmProjects/proteasome/data_processing/" \
           "un-merged_data/"

SYF_df = pd.read_csv(file_dir + "SYF_data_w_sequences.csv")
IEDB_df = pd.read_csv(file_dir + "IEDB_data_w_sequences.csv")
digestion_df = pd.read_csv(file_dir + "edited_digestion.csv")
IEDB_df['entry_source'] = "IEDB"

new_col_names = ['IEDB_id', 'fragment', 'start_pos', 'end_pos',
                 'UniProt_parent_id', 'origin_species', 'lit_reference',
                 'full_sequence', 'entry_source']

IEDB_df.columns = new_col_names

# syf
syf_start = []
syf_end = []
for pos in SYF_df['Position']:
    entry = pos.split("-")
    syf_start.append(entry[0])
    syf_end.append(entry[1])

SYF_df['start_pos'] = syf_start
SYF_df['end_pos'] = syf_end
SYF_df['IEDB_id'] = None
SYF_df = SYF_df[['IEDB_id','epitope', 'start_pos', 'end_pos', 'UniProt_id',
                 'Human', 'reference', 'full_sequence', 'Origin']]

SYF_df.columns = new_col_names
SYF_df['origin_species'] = SYF_df['origin_species'].astype(str)
for i in range(len(SYF_df)):
    entry = SYF_df.iloc[i]['origin_species']
    if entry == "True":
        SYF_df.at[i, 'origin_species'] = "human"
    else:
        SYF_df.at[i, 'origin_species'] = "mammal_other"


IEDB_df['origin_species'] = IEDB_df['origin_species'].astype(str)
for i in range(len(IEDB_df)):
    entry = IEDB_df.iloc[i]['origin_species']
    if entry == "9606":
        IEDB_df.at[i, 'origin_species'] = "human"
    else:
        IEDB_df.at[i, 'origin_species'] = "mammal_other"

out_df = IEDB_df.append(SYF_df, ignore_index=True, sort=True)
out_df = out_df.append(digestion_df, ignore_index=True, sort=True)

out_df.to_csv(file_dir + "tmp_merged_v2.csv", index=False)
