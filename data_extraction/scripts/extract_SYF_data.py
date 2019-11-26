#!/usr/bin/env python3
"""
This script extracts the epitope/ligand information from the SYFPEITHI
database (http://www.syfpeithi.de) and converts it into a csv file
"""
import pandas as pd
import numpy as np
import re
import extraction_functions

indir = "/Users/weeder/PycharmProjects/proteasome/data_extraction/raw_data/"
outdir = "/Users/weeder/PycharmProjects/proteasome/data_processing/" \
         "un-merged_data/positives/"

# NOTE: Pulls all epitopes, for class I & II and different organisms... need
# to restrict this further to save time/downstream filtering issues...
# raw_allele_list = ef.get_SYF_alleles()
# allele_series = pd.Series(raw_allele_list)
# allele_series.to_csv("../raw_allele_series.csv", index=False, header=False)

mammal_allele_df = pd.read_csv(indir + "raw_mammal_allele_series.csv")
mammal_allele_df = mammal_allele_df[mammal_allele_df.Class != "II"]

allele_list = mammal_allele_df.Allele

full_SYF_df = pd.DataFrame(columns=['epitope', 'prot_name', 'ebi_id',
                                    'reference', 'allele'])

for a in allele_list:
    print(a)
    allele_query = compile_SYF_url(a)
    try:
        tmp_df = extract_SYF_table(allele_query)
    except EmptyQueryError:
        continue
    if len(tmp_df) > 0:
        tmp_df['allele'] = a
        full_SYF_df = full_SYF_df.append(tmp_df, ignore_index=True, sort=True)

full_SYF_df['UniProt_id'] = None
full_SYF_df['UniProt_reviewed'] = None
full_SYF_df['Position'] = None
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

print(flags)
print("Total entries: ", len(full_SYF_df))
print("Queried ID's: ", full_SYF_df['UniProt_id'].count())

missing_id = []
for entry in full_SYF_df['UniProt_id']:
    if str(entry) == "":
        missing_id.append(True)
    else:
        missing_id.append(False)

missing_id_index = [i for i, val in enumerate(missing_id) if val]
print("Missing: ",len(missing_id_index))
print(missing_id_index)

full_SYF_df.dropna(subset=["UniProt_id"], inplace=True)
full_SYF_df['Human'] = ["HLA-" in a for a in full_SYF_df['allele']]
full_SYF_df = full_SYF_df[['allele', 'epitope', 'UniProt_id','Position', 'Human', 'reference']]

full_SYF_df.to_csv(outdir+"tmp_SYFPEITHI_epitopes.csv", index=False)
