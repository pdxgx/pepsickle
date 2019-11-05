#!/usr/bin/env python3
"""
This script extracts the epitope/ligand information from the SYFPEITHI
database (http://www.syfpeithi.de) and converts it into a csv file
"""

import pandas as pd
import extraction_functions as ef

outdir = "/Users/weeder/PycharmProjects/proteasome/data_processing/" \
         "un-merged_data/positives/"

raw_allele_list = ef.get_SYF_alleles()
allele_list = raw_allele_list[2:]

full_SYF_df = pd.DataFrame(columns=['epitope', 'source',
                                    'reference', 'allele'])

for a in allele_list:
    allele_query = ef.compile_SYF_url(a)
    try:
        tmp_df = ef.extract_SYF_table(allele_query)
    except:
        print(a)
        tmp_df = None
    if len(tmp_df) > 0:
        tmp_df['allele'] = a
        full_SYF_df = full_SYF_df.append(tmp_df, ignore_index=True)

print(full_SYF_df)

# full_SYF_df.to_csv(outdir+"SYFPEITHI_epitopes.csv", index=False)
