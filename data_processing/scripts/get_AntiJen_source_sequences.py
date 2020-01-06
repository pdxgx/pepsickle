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

antijen_df = pd.read_csv(options.infile, low_memory=False)

unique_protein_ids = list(antijen_df['Protein_refs'].dropna().unique())
sequence_dict = {}
error_index = []

progress = 0
for entry in unique_protein_ids:
    try:
        sequence_dict[entry] = retrieve_UniProt_seq(entry)
    except:
        error_index.append(progress)
    progress += 1
    if progress % 100 == 0:
        print(round(progress/len(unique_protein_ids)*100, 3), "% done")


# attempt to repair null queries
progress = 0
for e in error_index:
    tmp_id = unique_protein_ids[e]
    if tmp_id != "not applicable":
        try:
            query = compile_UniProt_url(tmp_id, include_experimental=True)
            buffer = extract_UniProt_table(query)
            new_id = buffer["Entry"][0]
            sequence_dict[unique_protein_ids[e]] = retrieve_UniProt_seq(new_id)
        except IndexError:
            pass
    progress += 1
    if progress % 10 == 0:
        print(round(progress/len(error_index)*100, 3), "% done")


antijen_df['full_sequence'] = None

for e in range(len(antijen_df)):
    prot_id = str(antijen_df.at[e, 'Protein_refs'])
    if prot_id in sequence_dict.keys():
        antijen_df.at[e, 'full_sequence'] = sequence_dict[prot_id]

antijen_df.dropna(subset=['full_sequence'], inplace=True)
antijen_df['entry_source'] = "AntiJen_data"
antijen_df['start_pos'] = None
antijen_df['end_pos'] = None

new_col_names = ['fragment', 'MHC_types', 'origin_species', 'category',
                 'UniProt_parent_id', 'ref_type', 'lit_reference',
                 'full_sequence', 'entry_source', 'start_pos', 'end_pos']

# fix column names on antijen df...
antijen_df.columns = new_col_names

# subset relevant columns
antijen_df = antijen_df[['fragment', 'MHC_types', 'origin_species',
                                     'UniProt_parent_id',  'lit_reference',
                                     'full_sequence', 'entry_source',
                                     'start_pos', 'end_pos']]
antijen_df.to_csv(options.out_dir + "AntiJen_data_w_sequences.csv",
                  index=False)
