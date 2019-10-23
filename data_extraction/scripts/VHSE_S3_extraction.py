#!/usr/bin/env Python3
"""VSHE_S3_extraction.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script extracts the information from the S3 supplementary of _The
VHSE_data-Based Prediction of Proteasomal Cleavage Sites_ (doi: 10.1371/
journal.pone.0074506) and converts it into a csv file

This script requires that `pandas`, `biopython`, and `numpy` be installed 
within the Python environment you are running this script in.

Inputs:
    The location of the S3 supplementary file

Outputs:
    The corresponding csv file
"""

import pandas as pd
# from Bio import Entrez
import numpy as np
# Entrez.email = "lie@ohsu.edu"
#
#
# def j(x):
#     if str(x).isdigit():
#         handle = Entrez.esummary(db="protein", id=x)
#         record = Entrez.read(handle)
#         return record[0]["Caption"]
#     else:
#         return np.nan


df = pd.read_csv("S3.csv")

df.rename(columns={"Epitope ID": "Epitope IRI (IEDB)",
                   "Object Description": "Description",
                   "Source Molecule Accession": "Parent Protein IRI (NCBI)",
                   "Source Molecule Name": "Parent Protein",
                   "Source Organism ID": "Organism IRI (NCBITaxon)",
                   "Source Organism Name": "Organism Name"}, inplace=True)

df.drop(columns=["Linear Sequence", "Modification", "Modified Residues"],
        inplace=True)

df.dropna(subset=["Parent Protein IRI (NCBI)"], inplace=True)
print(df.shape)
df.to_csv("S3_.csv", index=False)
