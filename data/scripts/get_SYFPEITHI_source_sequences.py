#! usr/bin/env python3
"""
get_SYFPEITHI_source_sequences.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script takes data from a CSV of breast_cancer_data epitopes results and returns
an updated CSV that includes source protein sequences. Column names are also
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

# load in SYF data, subset to only those with protein ID's
SYF_df = pd.read_csv(options.in_file, low_memory=False)
SYF_df.dropna(subset=['UniProt_id'], inplace=True)
SYF_df.index = range(len(SYF_df))  # re-index

# pull list of ";" separated ID entries
uniprot_ids = list(SYF_df['UniProt_id'])
parsed_ids = []

# parse id's into full list of entries
for i in uniprot_ids:
    # if more than one entry...
    if ";" in i:
        # split
        tmp = i.split(";")
        # store new ID's
        if tmp[0] not in parsed_ids:
            parsed_ids.append(tmp)
    else:
        # if only one entry, store if new
        if i not in parsed_ids:
            parsed_ids.append(i)

# set up variables to store protein data
sequence_dict = {}
error_index = []

progress = 0
# for each unique ID
for entry in parsed_ids:
    try:
        # query
        sequence_dict[entry] = retrieve_UniProt_seq(entry)
    except:
        # store failed queries
        error_index.append(progress)

    progress += 1
    # periodically print progress
    if progress % 100 == 0:
        print(round(progress/len(parsed_ids), 3)*100, "% done")


SYF_df['full_sequence'] = None
# iterate through entries and add protein sequence if ID was found
for e in range(len(SYF_df)):
    # store current entry
    tmp_entry = str(SYF_df.at[e, 'UniProt_id'])
    # if multiple alternate id's
    if ";" in tmp_entry:
        # store id list
        id_list = tmp_entry.split(";")
        # iterate through each id
        for p_id in id_list:
            prot_id = p_id
            # if id has associated sequence
            if prot_id in sequence_dict.keys():
                # store
                SYF_df.at[e, 'full_sequence'] = sequence_dict[prot_id]
                # go to next table entry
                continue
    # if only one entry
    else:
        prot_id = str(SYF_df.at[e, 'UniProt_id'])
        # store if ID has associated sequence
        if prot_id in sequence_dict.keys():
            SYF_df.at[e, 'full_sequence'] = sequence_dict[prot_id]

# drop any entries with no associated source sequence
SYF_df.dropna(subset=['full_sequence'], inplace=True)
SYF_df['Origin'] = "SYFPEITHI_database"

# export data
SYF_df.to_csv(options.out_dir + "/SYFPEITHI_epitopes_w_source.csv", index=False)
