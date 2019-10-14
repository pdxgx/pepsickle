#!/usr/bin/env Python3
"""AntiJen_TAP_extraction.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script extracts the epitopes found from the TAP AntiJen Database
(http://www.ddg-pharmfac.net/antijen/AntiJen/antijenhomepage.htm) and
converts it into a csv file.  It filters to make sure the epitopes are from
human origin (meaning that they were digested by the human proteasome).

This script requires that `pandas`, `numpy`, urllib.request, and itertools be
installed within the Python environment you are running this script in.
The program creates a comprehensive list of all TAP ligands by searching for
all possible two-amino acid combinations.

Inputs:
    The location of the csv file containing a list of epitopes found with a
    specific search on the online AntiJen website
    (The most general searches being filtering by class I or class II)

Outputs:
    A csv file containing the information from the epitopes extracted -
    including an epitope's description, allele class, and UniProt IRI
"""

""" Data Extraction """
df = pd.read_csv("TAP.csv")[["Description"]]

df["Buffer"] = df["Description"].apply(ef.get_script_page, call="TAP")
df.to_csv("TAP_.csv", index=False)

df = pd.read_csv("TAP_.csv")

df["Parent_Protein_IRI"] = df["Buffer"].apply(ef.get_sprot_IRI)
df['IRI_type'] = "Uniprot"
df["MHC"] = df.apply(ef.get_mhc_types, axis=1)

df = df[[m == "HUMAN" for m in df["MHC"]]]
df.drop(columns=["Buffer", "MHC"], inplace=True)
df.dropna(subset=["Parent Protein IRI"], inplace=True)
df.to_csv("TAP.csv", index=False)
