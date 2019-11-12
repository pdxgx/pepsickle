#!/usr/bin/env python3
"""
This script extracts the epitope/ligand information from the SYFPEITHI
database (http://www.syfpeithi.de) and converts it into a csv file
"""
import pandas as pd
import extraction_functions as ef
import os

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

full_SYF_df = pd.DataFrame(columns=['epitope', 'prot_name', 'ebi_id'
                                    'reference', 'allele'])

for a in allele_list:
    allele_query = ef.compile_SYF_url(a)
    try:
        tmp_df = ef.extract_SYF_table(allele_query)
    except ef.EmptyQueryError:
        continue
    if len(tmp_df) > 0:
        tmp_df['allele'] = a
        full_SYF_df = full_SYF_df.append(tmp_df, ignore_index=True, sort=True)


full_SYF_df.to_csv(outdir+"SYFPEITHI_epitopes.csv", index=False)
