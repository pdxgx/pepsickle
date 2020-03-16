#!/usr/bin/env python3
"""
extract_SYF_data.py

This script extracts the epitope/ligand information from the SYFPEITHI
database (http://www.syfpeithi.de) and converts it into a csv file

options:
--allele_file: a list of alleles to be queried in the SYFPEITHI database
- o, --out_dir: destination for CSV with returned SYFPEITHI queries
"""
from extraction_functions import *
import pandas as pd
import numpy as np
import re
from optparse import OptionParser

# define command line options
parser = OptionParser()
parser.add_option("-a", "--allele_file", dest="allele_file",
                  help="input csv of alleles to query.")
parser.add_option("-o", "--out_dir", dest="out_dir",
                  help="output csv with SYFPEITHI entries")


(options, args) = parser.parse_args()

# load in alleles
mammal_allele_df = pd.read_csv(options.allele_file)
mammal_allele_df = mammal_allele_df[mammal_allele_df.Class != "II"]
allele_list = mammal_allele_df.Allele

# set up column headers
full_SYF_df = pd.DataFrame(columns=['epitope', 'prot_name', 'ebi_id',
                                    'reference', 'allele'])

# loop through alleles and query epitopes for each
for a in allele_list:
    print(a)
    allele_query = compile_SYF_url(a)
    try:
        tmp_df = extract_SYF_table(allele_query)
    except EmptyQueryError:
        continue
    # if entry exists, append
    if len(tmp_df) > 0:
        tmp_df['allele'] = a
        full_SYF_df = full_SYF_df.append(tmp_df, ignore_index=True, sort=True)

# fill in columns to stay consistent with other databases
full_SYF_df['UniProt_id'] = None
full_SYF_df['UniProt_reviewed'] = None
full_SYF_df['Position'] = None

# set up tracking vars
n_entries = []
flags = []

for e in range(len(full_SYF_df)):
    entry = full_SYF_df.iloc[e]
    if entry['prot_name']:
        prot_name = entry['prot_name'].strip()
        try:
            pos = re.search("([0-9]+-[0-9]+)", prot_name).group()
            clean_prot_name = re.sub(pos, "", prot_name)
        except AttributeError:
            pos == np.nan
            clean_prot_name = re.sub(pos, "", prot_name)

        try:
            # make base query with id and protein name
            query = compile_UniProt_url(clean_prot_name, entry['ebi_id'])
            tmp_df = extract_UniProt_table(query)
            # if no results try without protein name
            if len(tmp_df) == 0 and entry['ebi_id'] is not np.nan:
                query = compile_UniProt_url("", entry['ebi_id'])
                tmp_df = extract_UniProt_table(query)
                # if nothing, repeat with only protein name
                if len(tmp_df) == 0 and entry['ebi_id'] is not np.nan:
                    query = compile_UniProt_url(clean_prot_name)
                    tmp_df = extract_UniProt_table(query)
                    # if nothing, repeat with experimental included
                    if len(tmp_df) == 0 and entry['ebi_id'] is not np.nan:
                        query = compile_UniProt_url(clean_prot_name,
                                                       entry['ebi_id'],
                                                       include_experimental=True)
                        tmp_df = extract_UniProt_table(query)
                        # if nothing repeat with exp. and no protein name
                        if len(tmp_df) == 0 and entry['ebi_id'] is not np.nan:
                            query = compile_UniProt_url("",
                                                           entry['ebi_id'],
                                                           include_experimental=True)
                            tmp_df = extract_UniProt_table(query)

            ids = ";".join(tmp_df['Entry'])
            full_SYF_df.at[e, 'UniProt_id'] = ids
            full_SYF_df.at[e, 'UniProt_reviewed'] = "reviewed" in list(tmp_df['Reviewed'])
            full_SYF_df.at[e, 'Position'] = pos
            # potentially add search for experimental if no results for reviewed
        except:
            flags.append(e)
    print(e, "/", len(full_SYF_df), " complete", sep="")

# print tracking vars
print(flags)
print("Total entries: ", len(full_SYF_df))
print("Queried ID's: ", full_SYF_df['UniProt_id'].count())

# identify missing ID's
missing_id = []
for entry in full_SYF_df['UniProt_id']:
    if str(entry) == "":
        missing_id.append(True)
    else:
        missing_id.append(False)

missing_id_index = [i for i, val in enumerate(missing_id) if val]
print("Missing: ",len(missing_id_index))
print(missing_id_index)

# export compiled data
full_SYF_df.dropna(subset=["UniProt_id"], inplace=True)
full_SYF_df['Human'] = ["HLA-" in a for a in full_SYF_df['allele']]
full_SYF_df = full_SYF_df[['allele', 'epitope', 'UniProt_id', 'Position',
                           'Human', 'reference']]

full_SYF_df.to_csv(options.out_dir + "/SYFPEITHI_epitopes.csv", index=False)
