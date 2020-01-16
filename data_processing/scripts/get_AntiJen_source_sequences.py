#! usr/bin/env python3
"""
get_AntiJen_source_sequences.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script takes data from a CSV of AntiJen database results and returns an
updated CSV that includes source protein sequences. Column names are also
updated for consistency with other databases and easier downstream merging

options:
-i, --in_file: CSV of AntiJen results from database
-o, --out_dir: location to export updated CSV
"""

from extraction_functions import *
import pandas as pd
from optparse import OptionParser


# define command line parameters
parser = OptionParser()
parser.add_option("-i", "--in_file", dest="in_file",
                  help="CSV of AntiJen database results")
parser.add_option("-o", "--out_dir", dest="out_dir",
                  help="output directory where CSV is exported")

(options, args) = parser.parse_args()

# import DB data
antijen_df = pd.read_csv(options.in_file, low_memory=False)

# pull only unique protein references
unique_protein_ids = list(antijen_df['Protein_refs'].dropna().unique())

# set up to store queried sequences, errors, and progress
sequence_dict = {}
error_index = []
progress = 0

# for each unique entry
print("Querying protein ID's")
for entry in unique_protein_ids:
    # try to query
    try:
        sequence_dict[entry] = retrieve_UniProt_seq(entry)
    # if query fails, store index
    except:
        error_index.append(progress)
    progress += 1
    # periodically output progress
    if progress % 100 == 0:
        print(round(progress/len(unique_protein_ids)*100, 3), "% completed")


# attempt to repair null queries
progress = 0
print("Attempting to repair failed queries")
for e in error_index:
    # pull id of failed query
    tmp_id = unique_protein_ids[e]
    # ignore those with no real id
    if tmp_id != "not applicable":
        try:
            # re-query with broader parameters
            query = compile_UniProt_url(tmp_id, include_experimental=True)
            buffer = extract_UniProt_table(query)
            new_id = buffer["Entry"][0]
            sequence_dict[unique_protein_ids[e]] = retrieve_UniProt_seq(new_id)
        except IndexError:
            # if empty results table, pass
            pass
    progress += 1
    # periodically output progress
    if progress % 10 == 0:
        print(round(progress/len(error_index)*100, 3), "% completed")


# create new column for storing sequences
antijen_df['full_sequence'] = None

# for each entry in the antijen results table
for e in range(len(antijen_df)):
    # pull protein ID
    prot_id = str(antijen_df.at[e, 'Protein_refs'])
    # if sequence was found for given ID, store full sequence
    if prot_id in sequence_dict.keys():
        antijen_df.at[e, 'full_sequence'] = sequence_dict[prot_id]

# drop rows where no sequence was retrieved
antijen_df.dropna(subset=['full_sequence'], inplace=True)

# add columns needed in downstream processing
antijen_df['entry_source'] = "AntiJen_data"
antijen_df['start_pos'] = None
antijen_df['end_pos'] = None

# rename columns for downstream merging and consistency
new_col_names = ['fragment', 'MHC_classes', 'Serotype', 'MHC_alleles',
                 'origin_species', 'category', 'UniProt_parent_id', 'ref_type',
                 'lit_reference', 'full_sequence', 'entry_source',
                 'start_pos', 'end_pos']
antijen_df.columns = new_col_names

# subset relevant columns for export
antijen_df = antijen_df[['fragment', 'MHC_classes', 'Serotype', 'MHC_alleles',
                         'origin_species', 'UniProt_parent_id',
                         'lit_reference', 'full_sequence', 'entry_source',
                         'start_pos', 'end_pos']]

# write out new CSV
antijen_df.to_csv(options.out_dir + "/AntiJen_data_w_sequences.csv",
                  index=False)
