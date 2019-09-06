#!/usr/bin/env Python3
"""Convergence.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script combines all separate csv files containing information on epitopes,
ligands, and digestion products into one comprehensive csv file

This script requires that `pandas` be installed within the Python
environment you are running this script in.

Inputs:
    The locations of all the different csv files to be converged

Outputs:
    The converged csv file
"""
import pandas as pd

files = ["IEDB.csv", "AntiJen.csv", "Breast_cancer.csv", "CTL_HIV.csv", "CTL_SYF.csv",
         "Digestion.csv", "Pcleavage.csv", "SYFPEITHI.csv",
         "VHSE_S1.csv", "VHSE_S3.csv", "VHSE_S5.csv"]

cumulative_df = pd.read_csv("csv/" + files[0], low_memory=False)


def calculate_results(df, file_name):
    """Calculate and print/write the results of each convergence

       This calculation includes:
       - The number of epitopes/sequences to be added from the added file
       - The number of new sequences from the added file

       Arguments:
           df (pandas Dataframe): the cumulative dataframe
           file_name (str): the name of the file to be added
       Returns:
           pandas Dataframe: the new df
    """
    df = df.append(pd.read_csv("csv/" + file_name, low_memory=False),
                   sort=False)
    print("Sequences in " + file_name + ":")
    print(len(pd.read_csv("csv/" + file_name).index))
    print()
    return df


for i in range(1, len(files)):
    cumulative_df = calculate_results(cumulative_df, files[i])

print("Final Size:")
length, _ = cumulative_df.shape
print(length)

cumulative_df.to_csv("converged.csv", index=False)
