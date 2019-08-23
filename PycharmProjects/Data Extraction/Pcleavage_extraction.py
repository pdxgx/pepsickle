#!/usr/bin/env Python3
"""Pcleavage_extraction.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script obtains epitope information from _Predicting proteasomal cleavage
sites: a comparison of available methods_ (doi: 10.1093/intimm/dxg084) which
was partially used by the algorithm Pcleavage in its test set.

This script requires that `pandas` and `numpy` be installed within the Python
environment you are running this script in.
Before running the program, the information from the paper was extracted with
an ORC and saved as a csv file. However, due to some issues, some of the
information was saved into one column instead of two - requiring additional
processing.


Inputs:
    The location of the csv file containing the following columns:
        [Description]: the description of the epitope
        [Parent Protein IRI (Uniprot)]: the Uniprot IRI of the protein
        [Starting Position]: the position of the epitope in its protein

Outputs:
    The numpy 3D array containing the negative data set for the model saved as
    a np file
"""
import pandas as pd
import numpy as np


def get_start_pos(x):
    """Separates and obtains the starting position from the Parent Protein IRI
       (Uniprot) column
       Arguments:
           x (int): directory of the dataframe
       Returns:
           int: the position of starting position
    """
    split = x["Parent Protein IRI (Uniprot)"].split()
    end = split[len(split) - 1]
    if float(x["Starting Position"]) > 0:
        return x["Starting Position"]
    elif str(end).isdigit():
        return end
    else:
        return np.nan


def get_sprot(x):
    """Separates and obtains the entry number from the Parent Protein IRI
       (Uniprot) column
       Arguments:
           x (int): directory of the dataframe
       Returns:
           str: the entry number of the protein
    """
    split = x["Parent Protein IRI (Uniprot)"].split()
    end = split[len(split) - 1]
    if str(end).isdigit():
        return "".join(split[0:len(split) - 1])
    else:
        return "".join(x["Parent Protein IRI (Uniprot)"].split())


df = pd.read_csv("pcleavage.csv")

df["Starting Position"] = df.apply(get_start_pos, axis=1)
df["Parent Protein IRI (Uniprot)"] = df.apply(get_sprot, axis=1)
df["Description"] = df["Description"].apply(lambda x: "".join(x.split()))

df.to_csv("pcleavage_entry.csv", index=False)