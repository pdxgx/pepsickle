#!/usr/bin/env Python3
"""Mutation_sorting.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script, given a csv file containing epitopes which do not perfectly
match with their corresponding proteins sequences, sorts the epitopes into
separate csv files depending on how many mismatches there are

This script requires that `pandas` and `Levenshtein` be installed within the
Python environment you are running this script in.
The program works using Levenshtein distance and determines based off a 
critical value using the Levenshtein ratio between the expected epitope and
the actual one

Inputs:
    The location of the csv file containing the mismatched epitopes

Outputs:
    Two different csv files:
        One csv for the simpler mismatches/substitutions
        Another csv file with more
"""

import pandas as pd
import Levenshtein as lev

crit_value = 0.6

df = pd.read_csv("non_IEDB_dropped.csv")

print(df.shape)


def get_expected_description(x):
    if type(x["Starting Position"]) != float:
        best_start = int(x["Starting Position"]) - 1
        best_distance = lev.distance(str(x["Description"]), x["Protein Sequence"]
                                     [int(x["Starting Position"]-1):int(x["Ending Position"])])
    else:
        best_start = 0
        best_distance = 100
    length = len(x["Description"])

    for i in range(len(x["Protein Sequence"]) - length):
        sequence = x["Protein Sequence"][i:i + length]

        distance = lev.distance(str(x["Description"]), sequence)
        if distance < best_distance:
            best_distance = distance
            best_start = i

    return x["Protein Sequence"][best_start: best_start + length]


df["Expected Description"] = df.apply(get_expected_description, axis=1)

for index, row in df.iterrows():
    print(row["Description"], "|", row["Expected Description"])
    print("Distance:", lev.distance(str(row["Description"]), 
                                    str(row["Expected Description"])),
          "Ratio:", lev.ratio(str(row["Description"]), 
                              str(row["Expected Description"])))

substitutions = df[[lev.ratio(str(x), str(y)) > crit_value for x, y in 
                    zip(df["Description"], df["Expected Description"])]]

print(substitutions.shape)

columns = ["Epitope IRI (IEDB)", "Description", "Expected Description",
           "Starting Position", "Ending Position", "Antigen Name",
           "Antigen IRI (NCBI)", "Parent Protein",
           "Parent Protein IRI (Uniprot)", "Parent Protein IRI (NCBI)",
           "Organism Name", "Organism IRI (NCBITaxon)", "Parent Species",
           "Parent Species IRI (NCBITaxon)", "Allele Name", "Allele IRI (MRO)",
           "Epitope Comments", "Protein Sequence"]

substitutions = substitutions[columns]

substitutions.to_csv("non_IEDB_substitutions.csv", index=False)

df.drop(df[[lev.ratio(str(x), str(y)) > 0.6 
            for x, y in zip(df["Description"], 
                            df["Expected Description"])]].index, inplace=True)
df.to_csv("non_IEDB_complex.csv", index=False)