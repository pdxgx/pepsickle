#!/usr/bin/env python3
"""
This script extracts the epitope/ligand information from the SYFPEITHI
database (http://www.syfpeithi.de) and converts it into a csv file
"""
import pandas as pd
import numpy as np
import re
import extraction_functions as ef

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
    allele_query = ef.compile_SYF_url(a)
    try:
        tmp_df = ef.extract_SYF_table(allele_query)
    except ef.EmptyQueryError:
        continue
    if len(tmp_df) > 0:
        tmp_df['allele'] = a
        full_SYF_df = full_SYF_df.append(tmp_df, ignore_index=True, sort=True)

full_SYF_df['UniProt_id'] = None
full_SYF_df['UniProt_reviewed'] = None
full_SYF_df['Position'] = None
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
            query = ef.compile_UniProt_url(clean_prot_name, entry['ebi_id'])
            tmp_df = ef.extract_UniProt_table(query)
            ids = ";".join(tmp_df['Entry'])
            full_SYF_df.at[e, 'UniProt_id'] = ids
            full_SYF_df.at[e, 'UniProt_reviewed'] = "reviewed" in tmp_df['Reviewed']
            full_SYF_df.at[e, 'Position'] = pos
            # potentially add search for exp if no results for reviewed
        except:
            flags.append(e)
            print(flags)
    print(round(e/len(full_SYF_df)*100, 2), "% complete")




# full_SYF_df.to_csv(outdir+"SYFPEITHI_epitopes.csv", index=False)
