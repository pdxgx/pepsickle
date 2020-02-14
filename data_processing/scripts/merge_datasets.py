#! usr/bin/env python3
"""
merge_datasets.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script mergest data from SYFPEITHI, IEDB, AntiJen, the breast cancer
study, and from all digestion maps into one cohesive csv.

options:
-i, --in_dir: directory where all the above listed data sources are stored.
These files names should all end with "...w_sequences.csv" indicating that the
corresponding source sequence data for each has been appended appropriately
-o, --out_dir: directory where merged cvs will be exported
"""
import pandas as pd
import os
from optparse import OptionParser


# define command line parameters
parser = OptionParser()
parser.add_option("-i", "--in_dir", dest="in_dir",
                  help="directory holding database csv's with source sequence"
                       "information")
parser.add_option("-o", "--out_dir", dest="out_dir",
                  help="output directory where merged CSV is exported")

(options, args) = parser.parse_args()

# import each database
# TODO: make assertion that all datasets are present?
SYF_df = pd.read_csv(options.in_dir + "/SYF_data_w_sequences.csv")
IEDB_df = pd.read_csv(options.in_dir + "/unique_iedb_epitopes.csv")
bc_df = pd.read_csv(options.in_dir + "/breast_cancer_data_w_sequences.csv")
antijen_df = pd.read_csv(options.in_dir + "/AntiJen_Tcell_w_sequences.csv")
digestion_df = pd.read_csv(options.in_dir + "/compiled_digestion_df.csv")

# identify names for IEDB to match others... this may need to be moved to
# previous script
IEDB_df['entry_source'] = "IEDB"
new_col_names = ['epitope_id', 'fragment', 'start_pos', 'end_pos',
                 'full_sequence', 'full_seq_database', 'full_seq_accession',
                 'mhc_allele_name', 'origin_species', 'lit_reference',
                 'entry_source']
IEDB_df.columns = new_col_names

# move to IEDB script?
IEDB_df['origin_species'] = IEDB_df['origin_species'].astype(str)
for i in range(len(IEDB_df)):
    entry = IEDB_df.iloc[i]['origin_species']
    if entry == "9606":
        IEDB_df.at[i, 'origin_species'] = "human"
    else:
        IEDB_df.at[i, 'origin_species'] = "mammal_other"


# move to SYFPEITHI script?
syf_start = []
syf_end = []
for pos in SYF_df['Position']:
    entry = pos.split("-")
    syf_start.append(entry[0])
    syf_end.append(entry[1])

SYF_df['start_pos'] = syf_start
SYF_df['end_pos'] = syf_end
SYF_df['epitope_id'] = None
SYF_df['full_seq_database'] = "UniProt"

SYF_df = SYF_df[['epitope_id', 'epitope', 'start_pos', 'end_pos',
                 'full_sequence', 'full_seq_database', 'UniProt_id', 'allele',
                 'Human', 'reference', 'Origin']]
SYF_df.columns = new_col_names

SYF_df['origin_species'] = SYF_df['origin_species'].astype(str)
for i in range(len(SYF_df)):
    entry = SYF_df.iloc[i]['origin_species']
    if entry == "True":
        SYF_df.at[i, 'origin_species'] = "human"
    else:
        SYF_df.at[i, 'origin_species'] = "mammal_other"


# move to AntiJen script?
antijen_df["epitope_id"] = None
antijen_df['full_seq_database'] = "UniProt"
antijen_df['full_seq_accession'] = antijen_df["UniProt_parent_id"]
antijen_df['mhc_allele_name'] = antijen_df['MHC_types']


for i in range(len(antijen_df)):
    entry = antijen_df.iloc[i]['origin_species']
    if 'HUMAN' in entry:
        antijen_df.at[i, 'origin_species'] = "human"
    else:
        antijen_df.at[i, 'origin_species'] = "mammal_other"

antijen_df.drop(columns=["MHC_types", 'UniProt_parent_id'], inplace=True)


# breast cancer data, this will likely stay
bc_df['epitope_id'] = None
bc_df['full_seq_accession'] = bc_df['Protein_ref']
bc_df['full_seq_database'] = bc_df['Ref_type']
bc_df['lit_reference'] = "10.1016/j.jprot.2018.01.004"
bc_df.drop(columns=['Protein_ref', 'Ref_type'], inplace=True)


# drop all non-human data from epitope entries
# IEDB_df = IEDB_df[IEDB_df['origin_species'] == 'human']
# SYF_df = SYF_df[SYF_df['origin_species'] == 'human']
# antijen_df = antijen_df[antijen_df['origin_species'] == 'human']


# rename digestion_df columns for consistency
new_digestion_cols = ['lit_reference', 'protein_name', 'origin_species',
                      'Proteasome', "Subunit", 'full_seq_accession', 'end_pos',
                      'fragment', 'full_sequence', 'start_pos', 'entry_source']
digestion_df.columns = new_digestion_cols
digestion_df['epitope_id'] = None
digestion_df['full_seq_database'] = "UniProt"
digestion_df['mhc_allele_name'] = None
digestion_df.drop(columns=['protein_name'], inplace=True)


out_df = IEDB_df.append(SYF_df, ignore_index=True, sort=True)
out_df = out_df.append(antijen_df, ignore_index=True, sort=True)
out_df = out_df.append(digestion_df, ignore_index=True, sort=True)
out_df = out_df.append(bc_df, ignore_index=True, sort=True)


# print summary info
print(out_df['entry_source'].value_counts())
print(out_df['origin_species'].value_counts())

out_df.to_csv(options.out_dir + "/merged_proteasome_data.csv", index=False)
