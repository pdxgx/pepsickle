#!/usr/bin/env Python3
"""CTL_extraction.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script converts information on epitopes from supplementaries of NetCTL
(doi: 10.1002/eji.200425811) into csv files.

This script requires that `pandas` be installed within the Python environment
you are running this script in.

Inputs:
    The location of the text files containing the NetCTL information

Outputs:
    The csv files corresponding to the text files
"""
import pandas as pd
import os

columns = ["Parent_Protein_IRI", "Starting_Position",
           "Description", "Allele_Name"]
file_names = os.listdir("/Users/weeder/PycharmProjects/proteasome/"
                        "data_extraction/raw_data/CTL")
# file_names = ["HIV1.raw_data.fsa.txt", "HIV2.raw_data.fsa.txt",
# "SYF1.raw_data.fsa.txt", "SYF2.raw_data.fsa.txt"]

export_names = ["HIV1.csv", "HIV2.csv", "SYF1.csv", "SYF2.csv"]


def load_CTL_data(file_name):
    """Extracts the raw_data from the text files into a pandas Dataframe
       Arguments:
           file_name (str): the name and location of the file to be extracted
       Returns:
           pandas Dataframe: the corresponding Dataframe
    """
    df = pd.DataFrame(columns=columns)

    with open(file_name) as f:
        buffer = f.readline()
        row = []
        while buffer:
            running_row = buffer.split()
            running_row[0] = running_row[0][1:]
            buffer = f.readline().strip()
            while buffer and buffer[0] != ">":
                buffer = f.readline()
            row.append(running_row)
        df = df.append(pd.DataFrame(row, columns=columns))
        df = df.append(pd.DataFrame([], columns=columns))

    df["Allele_Name"] = "HLA-" + df["Allele_Name"].astype(str)
    df["IRI_type"] = "Uniprot"
    return df

# this should probably not output files...
# either remove final "full" statements or don't output in this function
def process_CTL_data(file_name, export_name):
    """Process the dataframe by converging all epitopes with identical
       descriptions together and combining the different allele names
       Arguments:
           file_name (str): the name and location of the file
           export_name (str): the name and location of the csv file to be 
                              exported
       Returns:
           pandas Dataframe: the corresponding Dataframe
    """
    indir = "/Users/weeder/PycharmProjects/proteasome/data_extraction/" \
            "raw_data/CTL/"
    outdir = "/Users/weeder/PycharmProjects/proteasome/data_processing/" \
             "un-merged_data/positives/"
    df = load_CTL_data(indir + file_name)
    g = {c: ", ".join if c == "Allele_Name" else 'first' for c in columns}
    df = df.groupby("Description").agg(g)
    df["Allele_Name"] = df["Allele_Name"].apply(lambda x: ", ".join(
        sorted(list((dict.fromkeys(x.split(", ")))))))

    df.to_csv(outdir + export_name, index=False)
    return df


data_frames = []

for i in range(len(file_names)):
    data_frames.append(process_CTL_data(file_names[i], export_names[i]))

# temporarily comment out
"""
Entire_HIV = data_frames[0].append(data_frames[1], ignore_index=True)
f = {c: ", ".join if c == "Allele Name" else 'first' for c in columns}
Entire_HIV = Entire_HIV.groupby("Description").agg(f)
Entire_HIV["Allele Name"] = Entire_HIV["Allele Name"].apply(
    lambda x: ", ".join(sorted(list((dict.fromkeys(x.split(", ")))))))
Entire_HIV.to_csv("Entire HIV.csv", index=False)

Entire_SYF = data_frames[2].append(data_frames[3], ignore_index=True)
f = {c: ", ".join if c == "Allele Name" else 'first' for c in columns}
Entire_SYF = Entire_SYF.groupby("Description").agg(f)
Entire_SYF["Allele Name"] = Entire_SYF["Allele Name"].apply(
    lambda x: ", ".join(sorted(list((dict.fromkeys(x.split(", ")))))))
Entire_SYF.to_csv("Entire SYF.csv", index=False)
"""
