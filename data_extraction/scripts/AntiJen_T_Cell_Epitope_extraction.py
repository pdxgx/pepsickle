#!/usr/bin/env Python3
"""AntiJen_T_Cell_Epitope_extraction.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script extracts the epitopes found from the T Cell Epitope AntiJen
Database (http://www.ddg-pharmfac.net/antijen/AntiJen/antijenhomepage.htm)
and converts it into a csv file.  It filters to make sure the epitopes
are from human origin (meaning that they were digested by the human 
proteasome).

This script requires that `pandas`, `numpy`, and urllib.request be installed
within the Python environment you are running this script in.

Inputs:
    The location of the csv file containing a list of epitopes found with a
    specific search on the online AntiJen website
    (The most general searches being filtering by class I or class II)

Outputs:
    A csv file containing the information from the epitopes extracted - 
    including an epitope's description, allele class, and UniProt IRI
"""
import pandas as pd
import extraction_functions as ef

# in future replace with argparse input
file_name = "/Users/weeder/PycharmProjects/proteasome/data_extraction/raw_data/AntiJen/T C2.csv"

# add argparse with call type for t-cell vs. TAP... then antijen extraction
# can be a single script with 2 processing options

""" Data Extraction """
df = pd.read_csv(file_name)[["Description"]]
# print(df.shape)


df["Buffer"] = df["Description"].apply(ef.get_script_page, call="T_cell")
df["Allele_Name"] = df.apply(ef.get_alleles, axis=1)
df["Parent_Protein_IRI"] = df["Buffer"].apply(ef.get_sprot_IRI)
df['IRI_type'] = "Uniprot"
# seems to be a parsing error here...some have \n in name
df["host_org"] = df.apply(ef.get_mhc_organism, axis=1)

# we want to keep all mammal for now, so comment this out
# df = df[[m == "HUMAN" for m in df["MHC"]]]
df = df[[org is not None for org in df['host_org']]]

df.drop(columns=["Buffer"], inplace=True)
df.dropna(subset=["Parent_Protein_IRI"], inplace=True)

df.to_csv("/Users/weeder/PycharmProjects/proteasome/data_extraction/raw_data/AntiJen/T_Cell_C2.csv", index=False)
