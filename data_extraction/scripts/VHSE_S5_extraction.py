#!/usr/bin/env Python3
"""VHSE_S5_extraction.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script reformats the information from the S5 supplementary of _The
VHSE-Based Prediction of Proteasomal Cleavage Sites_ 
(doi: 10.1371/journal.pone.0074506)

This script requires that `pandas` be installed within the Python
environment you are running this script in.

Inputs:
    The location of the S5 supplementary csv file 

Outputs:
    The reformatted csv file
"""
import pandas as pd

df = pd.read_csv("S5.csv")

df.rename(columns={"Peptides": "Description",
                   "SWISSPROT accession number": "Parent Protein IRI (Uniprot)"},
          inplace=True)

df["Parent Protein IRI (Uniprot)"] = df["Parent Protein IRI (Uniprot)"].apply(
    lambda x: "_".join(x.split()[0:len(x.split()) - 1]))

df.to_csv("S5_.csv")