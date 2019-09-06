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

import urllib.request
import pandas as pd
import numpy as np
from itertools import product

# Generate combinations and their respective lists
amino_acids = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "O", "I", "L", "K",
               "M", "F", "P", "U", "S", "T", "W", "Y", "V"]

combos = list(product(amino_acids, repeat=2))
running_epitopes = []

for i in combos:
    print("".join(i))
    with urllib.request.urlopen("http://www.ddg-pharmfac.net/antijen/scripts/"
                                + "aj_scripts/aj_tapcalc.pl?epitope="
                                + "".join(i)
                                + "&MIN=&MAX=&allele=&CATEGORY=TAP&ic50MIN="
                                + "&ic50MAX=&KDNMMIN=&KDNMMAX=&TAP=Search+AntiJen") as h:
        epitope_buffer = str(h.read()).split("epitope value=")
        for j in epitope_buffer[1:]:
            running_epitopes.append(j.split(">")[0])

running_epitopes = list(dict.fromkeys(running_epitopes))

pd.DataFrame({"Description": running_epitopes}).to_csv("TAP.csv", index=False)


""" Data Extraction """
df = pd.read_csv("TAP.csv")[["Description"]]


def f(x):
    """Obtains the script of the page related to the given description
       Arguments:
           x (str): a certain epitope description in the dataframe
       Returns:
           str: the full script of the page
    """
    with urllib.request.urlopen("http://www.ddg-pharmfac.net/antijen/scripts/"
                                + "aj_scripts/aj_tapcalc2.pl?epitope=" + x
                                + "&CAT=TAP+Ligand&detailinfo=no&detailmin="
                                + "&detailmax=") as h:
        return str(h.read())


df["Buffer"] = df["Description"].apply(f)
df.to_csv("TAP_.csv", index=False)

df = pd.read_csv("TAP_.csv")


def get_sprot(buffer):
    """Obtains the UniProt IRI of the protein an epitope is derived from
       Arguments:
           x (int): the directory of the dataframe
       Returns:
           str: the UniProt IRI
    """
    if len(buffer.split("sprot-entry?")) > 1:
        return buffer.split("sprot-entry?")[1].split("\"")[0].split("http")[0]
    else:
        return np.nan


def get_mhc(x):
    """Obtains the organism name of the MHC species the epitope came from\

         Used to filter for human MHC species later

         Arguments:
             x (int): the directory of the dataframe
         Returns:
             str: the MHC species of the epitope
      """
    try:
        # Manually check any epitopes which may have a HUMAN MHC (just not in the first row)
        if "HUMAN" in x["Buffer"] \
                and x["Buffer"].split(x["Description"] + "</td>\\n\\t<td>")[1].split("<")[0] != "HUMAN" \
                and len(x["Buffer"].split(x["Description"] + "</td>\\n\\t<td>")) > 2:
            print(x["Description"])
        return x["Buffer"].split(x["Description"] + "</td>\\n\\t<td>")[1].split("<")[0]
    except IndexError:
        return np.nan


df["Parent Protein IRI (Uniprot)"] = df["Buffer"].apply(get_sprot)
df["MHC"] = df.apply(get_mhc, axis=1)

df = df[[x == "HUMAN" for x in df["MHC"]]]
df.drop(columns=["Buffer", "MHC"], inplace=True)
df.dropna(subset=["Parent Protein IRI (Uniprot)"], inplace=True)
df.to_csv("TAP.csv", index=False)
