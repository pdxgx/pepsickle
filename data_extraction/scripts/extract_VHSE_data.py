#!/usr/bin/env Python3
"""

"""
import pandas as pd
import numpy as np

indir = "/Users/weeder/PycharmProjects/proteasome/data_extraction/" \
        "raw_data/VHSE_data/"

s1_df = pd.read_excel(indir + "Dataset_s1.xlsx")
s3_df = pd.read_excel(indir + "Dataset_s3.xls")
s5_df = pd.read_excel(indir + "Dataset_s5.xlsx")
# s7 relevant for digesion map data?


s1_cleaned_df = pd.DataFrame(columns=["Epitope", "MHC_types", "Species",
                                      "Categories", "Protein_refs",
                                      "Ref_type", "Journal_refs"])
for e in range(len(s1_df)):
    entry = s1_df.iloc[e]

    if entry['Epitope'] is not np.nan:
        entry_n = entry['Number of results']
        tmp_df = s1_df.iloc[e:int(e+entry_n)]

        # make empty lists
        # append entries if novel
        Class = []
        Species = []
        Category = []
        Protein_ref = []
        Journal_ref = []

        for i in range(len(tmp_df)):
            dat = tmp_df.iloc[i]
            if dat['Class'] not in Class:
                Class.append(str(dat['Class']))
            if dat['MHC species'] not in Species:
                Species.append(str(dat['MHC species']))
            if dat['Category'] not in Category:
                Category.append(str(dat['Category']))
            if dat['Swiss Prot Ref'] not in Protein_ref:
                Protein_ref.append(str(dat['Swiss Prot Ref']))
            if dat['Journal Ref'] not in Journal_ref:
                Journal_ref.append(str(dat['Journal Ref']))

        # generate single entries separated with semicolons
        epitope = entry['Epitope']
        MHC_types = "; ".join(Class)
        MHC_species = "; ".join(Species)
        Categories = "; ".join(Category)
        Protein_ref = "; ".join(Protein_ref)
        Ref_type = "Uniprot"  # define reference database used
        Journal_refs = "; ".join(Journal_ref)

        new_entry = pd.Series([epitope, MHC_types, MHC_species, Categories,
                              Protein_ref, Ref_type, Journal_refs],
                              index=s1_cleaned_df.columns)
        # append to out_df
        s1_cleaned_df = s1_cleaned_df.append(new_entry, ignore_index=True)


# some fully duplicated rows...
s3_df.drop_duplicates(inplace=True)

# some duplicated sequences still, but rows are unique... looks like p mods?
