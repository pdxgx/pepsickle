#! usr/bin/env python3
"""
get_breast_cancer_source_sequences.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script takes data from a CSV of Breast Cancer epitopes results and returns
an updated CSV that includes source protein sequences. Column names are also
updated for consistency with other databases and easier downstream merging

options:
-i, --in_file: CSV of AntiJen results from database
-o, --out_dir: location to export updated CSV
"""
from extraction_functions import *
import pandas as pd
import numpy as np
from optparse import OptionParser

# define command line parameters
parser = OptionParser()
parser.add_option("-i", "--in_file", dest="in_file",
                  help="CSV of AntiJen database results")
parser.add_option("-o", "--out_dir", dest="out_dir",
                  help="output directory where CSV is exported")

(options, args) = parser.parse_args()

# load in breast cancer data
bc_df = pd.read_csv(options.in_file, low_memory=False)
bc_df['full_sequence'] = None

# identify unique protein ID's
unique_protein_ids = list(
    bc_df['Parent Protein IRI (Uniprot)'].dropna().unique()
)

# create variables to store data
sequence_dict = {}
error_index = []

progress = 0
# iterate through unique proteins
for entry in unique_protein_ids:
    try:
        # try to query
        sequence_dict[entry] = retrieve_UniProt_seq(entry)
    except:
        # store errors
        error_index.append(progress)

    progress += 1
    # periodically print progress
    if progress % 100 == 0:
        print(round(progress/len(unique_protein_ids)*100, 3), "% done")


# attempt to repair null queries
progress = 0
for e in error_index:
    tmp_id = unique_protein_ids[e]
    # expand query parameters
    query = compile_UniProt_url(tmp_id, include_experimental=True)
    buffer = extract_UniProt_table(query)
    new_id = buffer["Entry"][0]
    sequence_dict[unique_protein_ids[e]] = retrieve_UniProt_seq(new_id)

    progress += 1
    # periodically print progress
    if progress % 100 == 0:
        print(round(progress/len(error_index)*100, 3), "% done")

# if protein ID has associated sequence, enter in df
for e in range(len(bc_df)):
    prot_id = str(bc_df.at[e, 'Parent Protein IRI (Uniprot)'])
    if prot_id in sequence_dict.keys():
        bc_df.at[e, 'full_sequence'] = sequence_dict[prot_id]


# drop entries with missing ref sequence and add additional columns
bc_df.dropna(subset=['full_sequence'], inplace=True)
bc_df['entry_source'] = "BC_study"
bc_df['origin_species'] = "human"
bc_df['start_pos'] = np.nan
bc_df['end_pos'] = np.nan

# export data
bc_df.to_csv(options.out_dir + "/breast_cancer_data_w_sequences.csv",
             index=False)
