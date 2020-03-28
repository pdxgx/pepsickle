#!/usr/bin/env Python3
"""extract_breast_cancer_data.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script converts epitope information from a xlsx file into a csv file.
The information was derived from a study on breast cancer cell lines
(doi: 10.1016/j.jprot.2018.01.004)

This script requires that `pandas` and `numpy` be installed within the Python
environment you are running this script in.

Inputs:
    The location of the xlsx file of interest
    The location of a csv file containing two columns:
        ["Input"]: A column containing all entry names of UniProt proteins
                   which were derived from the constructed dataframe's
                   ["Protein Link"]
        ["Entry"]: The column of corresponding accession number (found using
                   UniProt's Retrieve/ID mapping function)

Outputs:
    The corresponding csv file containing the epitope's description and its
    protein's UniProt IRI
"""
import pandas as pd

df = pd.DataFrame()
in_dir = "/Users/weeder/PycharmProjects/proteasome/data/raw_data/" \
         "breast_cancer_data"

out_dir = "/Users/weeder/PycharmProjects/proteasome/data/raw_data"

""" Data Extraction """
# the range(2, 24) corresponds to the different excel sheets which contain
# epitope information
base_file = pd.ExcelFile(in_dir + "/breast_cancer.xlsx")
for i in range(2, 24):
    tmp_df = pd.read_excel(in_dir + "/breast_cancer.xlsx", sheet_name=i,
                           header=2)[["Sequence", "Protein Link", "Unique"]]
    tmp_df['cell_line'] = base_file.sheet_names[i]
    df = df.append(tmp_df)

key = pd.read_csv(in_dir + "/breast_cancer_key.csv")
dict_key = {k: v for k, v in zip(key["Input"].to_list(),
                                 key["Entry"].tolist())}
df["Protein_ref"] = df["Protein Link"].map(dict_key)
df['Ref_type'] = "Uniprot"
df.dropna(subset=["Sequence"], inplace=True)
print("N entries: ", df.shape[0])

df.drop_duplicates(inplace=True)
print("N unique entries: ", df.shape[0])

df["Sequence"] = df["Sequence"].apply(lambda x: x.split(".")[1])

df.drop(columns=["Protein Link", "Unique"], inplace=True)
df.rename(columns={"Sequence": "fragment"}, inplace=True)

df.to_csv(out_dir + "/breast_cancer_table.csv", index=False)
