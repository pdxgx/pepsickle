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
import urllib.request
import pandas as pd
import numpy as np

file_name = "T C2.csv"

""" Data Extraction """
df = pd.read_csv(file_name)[["Description"]]

print(df.shape)


def f(x):
    """Obtains the script of the page related to the given description
       Arguments:
           x (str): a certain epitope description in the dataframe
       Returns:
           str: the full script of the page
    """
    with urllib.request.urlopen("http://www.ddg-pharmfac.net/antijen/scripts/"
                                + "aj_scripts/aj_tcellcalc2.pl?epitope="
                                + x + "&AL=%25&ST=%25&CAT="
                                + "T+Cell+Epitope") as h:
        return str(h.read())


df["Buffer"] = df["Description"].apply(f)


def get_alleles(x):
    """Obtains the allele types of a certain epitope using the script
       Arguments:
           x (int): the directory of the dataframe
       Returns:
           str: the allele type(s) of a certain epitope
    """
    split_allele = x["Buffer"].split("allele.cgi?")
    alleles = []
    for i in range(len(split_allele)):
        if i != 0:
            alleles.append(split_allele[i][:6])

    if len(alleles) > 0:
        alleles = list(dict.fromkeys(alleles))
        return ", ".join(alleles)
    else:
        buffer = split_allele[0].split("</td>\\n\\t<td>CLASS-2")[0].split(">")
        return buffer[len(buffer) - 1]


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
        if "HUMAN" in x["Buffer"] and x["Buffer"].split("CLASS-2</td>\\n\\t<td>")[1].split("<")[0] != "HUMAN" and len(x["Buffer"].split("CLASS-1</td>\\n\\t<td>")) > 2:
            print(x["Description"])
        return x["Buffer"].split("CLASS-2</td>\\n\\t<td>")[1].split("<")[0]
    except IndexError:
        return np.nan


df["Allele Name"] = df.apply(get_alleles, axis=1)
df["Parent Protein IRI (Uniprot)"] = df["Buffer"].apply(get_sprot)
df["MHC"] = df.apply(get_mhc, axis=1)

df = df[[x == "HUMAN" for x in df["MHC"]]]

df.drop(columns=["Buffer", "MHC"], inplace=True)
df.dropna(subset=["Parent Protein IRI (Uniprot)"], inplace=True)

df.to_csv("T_Cell_C2.csv", index=False)
