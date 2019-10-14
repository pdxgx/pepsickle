#!/usr/bin/env Python3
"""Breast_Cancer_extraction.py

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

""" Data Extraction """
# the range(2, 24) corresponds to the different excel sheets which contain
# epitope information
for i in range(2, 24):
    df = df.append(pd.read_excel("/Users/weeder/PycharmProjects/proteasome/"
                                 "data_extraction/raw_data/Breast Cancer/"
                                 "breast_cancer.xlsx",
                                 sheet_name=i, header=2)[["Sequence",
                                                          "Protein Link",
                                                          "Unique"]])

key = pd.read_csv("breast_cancer_key.csv")
dict_key = {k: v for k, v in zip(key["Input"].to_list(),
                                 key["Entry"].tolist())}
df["Parent_Protein_IRI"] = df["Protein Link"].map(dict_key)
df['IRI_type'] = "Uniprot"
df.dropna(subset=["Sequence"], inplace=True)

df = df[df["Unique"] == 1]
# print(df.shape)

df["Sequence"] = df["Sequence"].apply(lambda x: x.split(".")[1])

df.drop(columns=["Protein Link", "Unique"], inplace=True)
df.rename(columns={"Sequence": "Description"}, inplace=True)

# print(df.shape)

df.to_csv("/Users/weeder/PycharmProjects/proteasome/data_processing/"
          "un-merged_data/positives/breast_cancer.csv", index=False)
