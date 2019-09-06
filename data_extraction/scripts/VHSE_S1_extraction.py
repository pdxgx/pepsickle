#!/usr/bin/env Python3
"""VHSE_S1_extraction.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script extracts the epitope information from the S1
supplementary of _The VHSE-Based Prediction of Proteasomal
Cleavage Sites_ (doi: 10.1371/journal.pone.0074506) into a csv file

This script requires that `pandas` be installed within the Python
environment you are running this script in.

Inputs:
    The location of the supplementary xlsx file

Outputs:
    The corresponding csv file
"""
import pandas as pd

df = pd.read_excel("S1.xlsx")

columns = ["Description", "Parent Protein IRI (Uniprot)","Allele Name",
           "Epitope Comments"]


df.rename(
    columns={
        "Epitope": "Description",
        "Allele": "Allele Name",
        "Category": "Epitope Comments",
        "Swiss Prot Ref": "Parent Protein IRI (Uniprot)"
    },
    inplace=True
)

df["Allele Name"] = "HLA-" + df["Allele Name"].astype(str)
df["Allele Name"].replace("HLA-nan", df["Serotype"], inplace=True)

df.dropna(subset=["Parent Protein IRI (Uniprot)"], inplace=True)
df = df[~df["Parent Protein IRI (Uniprot)"].str.contains("not applicable")]
df = df[df["MHC species"].str.contains("HUMAN")]

df = df.drop(columns=["Number of results", "Class", "Journal Ref",
                      "MHC species", "Serotype"])

df["Description"].fillna(method="ffill", inplace=True)
df["Allele Name"].fillna("", inplace=True)

f = {c: ", ".join if c == "Allele Name" else 'first' for c in columns}
df = df.groupby("Description").agg(f)


df["Allele Name"] = df["Allele Name"].apply(
    lambda x: ", ".join(sorted(list((dict.fromkeys(x.split(", ")))))))

df.to_csv("S1.csv", index=False)
