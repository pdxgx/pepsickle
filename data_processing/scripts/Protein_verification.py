#!/usr/bin/env Python3
"""Protein_verification.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script gets the protein sequence for each description when necessary and
verifies whether the epitope/product is in the protein sequence

This script requires that `pandas`, `numpy`, and `biopython` be installed
within the Python environment you are running this script in.

Inputs:
    The location of the csv file to be verified
    The locations of the protein databases used to get the protein sequences:
        SwissProt, trembl, and the NCBI

Outputs:
    csv files which have gone through the filtering process and have
    failed/passed certain criteria:
        Protein Error:  where a protein sequence could not be found
        Dropped: where the description was not found in the sequence
        Filtered: those which passed both criteria
"""

import pandas as pd
import numpy as np
from Bio import SeqIO


def get_index_seq(x, index):
    """Tries to obtain the protein sequence of a given description
       Arguments:
           x (int): directory of the dataframe
           index: the specific protein database to investigate in
       Returns:
           str: the protein sequence
    """
    try:
        seq = index[x].seq
    except KeyError:
        seq = np.nan
    except AttributeError:
        seq = index[x]
    return seq


df = pd.read_csv("csv/converged.csv", low_memory=False)

print(df.shape)

sprot_index = SeqIO.index_db(
    "idx/sprot_index.idx", "idx/uniprot_sprot.fasta", "fasta",
    key_function=lambda x: x.split("|")[1])
df["Buffer"] = df["Parent Protein IRI (Uniprot)"].apply(
    lambda x: get_index_seq(x, sprot_index))

if "Protein Sequence" not in df.keys():
    df["Protein Sequence"] = np.nan

df["Protein Sequence"].fillna(df["Buffer"], inplace=True)

trembl_index = SeqIO.index_db(
    "idx/trembl_index.idx", "idx/uniprot_trembl.fasta",
    "fasta", key_function=lambda x: x.split("|")[1])

df["Buffer"] = df["Parent Protein IRI (Uniprot)"].apply(
    lambda x: get_index_seq(x, trembl_index))
df["Protein Sequence"].fillna(df["Buffer"], inplace=True)

nr_index = SeqIO.index_db("idx/nr_index.idx", "idx/nr.fasta", "fasta",
                          key_function=lambda x: x.split()[0])

df["Buffer"] = df["Parent Protein IRI (NCBI)"].apply(
    lambda x: get_index_seq(x, nr_index))
df["Protein Sequence"].fillna(df["Buffer"], inplace=True)

""" Further Protein Sequence finding via Biopython """
# NCBI: a csv file containing all dropped protein sequences was generated
# using the online Retrieve/ID mapping program containing the entry name and
# sequence (https://www.uniprot.org/uploadlists/)

df["Parent Protein IRI (NCBI)"] = df["Parent Protein IRI (NCBI)"].apply(
    lambda x: x[:-2] if type(x) == str else np.nan)

# df[[type(x) == float and type(y) == str for x, y in zip(
#     df["Protein Sequence"], df["Parent Protein IRI (NCBI)"])
#     ]]["Parent Protein IRI (NCBI)"].to_csv("ncbi_dropped.csv",
#                                            index=False, header=False)

ncbi_dropped_df = pd.read_csv("ncbi.csv")
ncbi_dropped_index = dict(
    zip(ncbi_dropped_df["Entry"].to_list(),
        ncbi_dropped_df["Sequence"].to_list()))

df["Buffer"] = df["Parent Protein IRI (NCBI)"].apply(
    lambda x: get_index_seq(x, ncbi_dropped_index))

df["Protein Sequence"].fillna(df["Buffer"], inplace=True)

# UniProt: a csv file containing all dropped protein sequences was generated
# using the online Retrieve/ID mapping program containing the entry name and
# sequence (https://www.uniprot.org/uploadlists/)

# df[[type(x) == float and type(y) == str for x, y in zip(
#     df["Protein Sequence"], df["Parent Protein IRI (Uniprot)"])
#     ]]["Parent Protein IRI (Uniprot)"].to_csv("uniprot_dropped.csv",
#                                               index=False, header=False)

uniprot_dropped_df = pd.read_csv("uniprot.csv").dropna(subset=["Sequence"])

uniprot_dropped_index = dict(
    zip(uniprot_dropped_df["Entry"].to_list(),
        uniprot_dropped_df["Sequence"].to_list()))

df["Buffer"] = df["Parent Protein IRI (Uniprot)"].apply(
    lambda x: get_index_seq(x, uniprot_dropped_index))

df["Protein Sequence"].fillna(df["Buffer"], inplace=True)

df.drop(columns=["Buffer"], inplace=True)

df[[type(x) == float
    for x in df["Protein Sequence"]]].to_csv("converged_error.csv")
df.dropna(subset=["Protein Sequence"], inplace=True)

df[[type(z) == float and y not in x
    for x, y, z in zip(df["Protein Sequence"], df["Description"],
                       df["Window"])]].to_csv("converged_dropped.csv",
                                              index=False)

drop_index = df[[type(z) == float and y not in x
                 for x, y, z in zip(df["Protein Sequence"], df["Description"],
                                    df["Window"])]].index
df.drop(drop_index, inplace=True)

print(df.shape)

df.to_csv("converged_filtered.csv", index=False)
