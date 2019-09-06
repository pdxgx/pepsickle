#!/usr/bin/env Python3
"""SYFPEITHI_extraction.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script extracts the epitope/ligand information from the SYFPEITHI
database (http://www.syfpeithi.de) and converts it into a csv file

This script requires that `pandas` and `urllib.request` be installed within
the Python environment you are running this script in.

Inputs:
    The location/name of the text file containing all HLA types to search by
    in the SYFPEITHI database

Outputs:
    The corresponding csv file
"""
import urllib.request
import pandas as pd


def get_ligand(string):
    """ Obtains a ligand from a given line from SYFPEITHI.
        Arguments:
            string (str): the given line of text containing the ligand
        Returns: 
            str: the ligand sequence
    """
    by_peptide = string.split(";")
    ligand = ""
    on_ligand = True
    i = len(by_peptide) - 1

    def remove_typeface(fragment):
        """ Removes any typefacing (bolding/underlining) from a string.
            fragment: the given string to remove typefacing from
            Arguments:
                n/a
            Returns:
                str: the edited fragment
        """
        fragment = fragment.replace("<B>", "")
        fragment = fragment.replace("</B>", "")
        fragment = fragment.replace("<U>", "")
        fragment = fragment.replace("</U>", "")
        return fragment

    fragment = remove_typeface(by_peptide[i])
    while on_ligand is True:
        ligand = fragment[0] + ligand
        i -= 1
        fragment = remove_typeface(by_peptide[i])

        if len(fragment) != 6 and ~fragment[0].isalpha():
            on_ligand = False
    return ligand


def get_protein(string):
    """Obtains a Uniprot entry number and peptide sequence from a given line
        from SYFPEITHI.
        Arguments:
            string (str): the given line of text containing the EBI ID
        Returns:
            str: the ncbi entry number
            str: the peptide sequence
    """
    gene_id = string.split("\"")[0]
    with urllib.request.urlopen("https://www.ebi.ac.uk/ena/data/view/"
                                + gene_id + "&display=text") as h:
        buffer = str(h.read())

    id, sequence = "", ""
    if len((buffer.split("/protein_id=\""))) > 1:
        id = buffer.split("/protein_id=\"")[1].split("\"")[0]
    if len(buffer.split("/translation=\"")) > 1:
        sequence = "".join(buffer.split("/translation=\"")[1].split("\"")[0].split()).replace("\\nFT", "")
    return id, sequence


def get_data(hla_name):
    """ Obtains a info from functions from an entire HLA page on SYFPEITHI
        Arguments:
            hla_name (str): the hla of the page to search/extract info from
        Returns: 
            str: a list of ligands
            str: a list of ncbi entry numbers
            str: a list of protein sequences
    """
    with urllib.request.urlopen(
            "http://www.syfpeithi.de/bin/MHCServer.dll/FindYourMotif?HLA_TYPE="
            + hla_name + "&AASequence=&OP1=AND&select1=002&content1=&OP2=AND&"
                         "select2=004&content2=&OP3=AND&select3=003&content3="
                         "&OP4=AND") as g:
        by_ebi = str(g.read()).split("emblfetch?")

    running_ids = []
    running_ligands = []
    running_sequences = []
    for i in range(len(by_ebi)):
        if i != 0:
            x, y = get_protein(by_ebi[i])
            running_ids.append(x)
            running_sequences.append(y)

        if i != len(by_ebi) - 1:
            running_ligands.append(get_ligand(by_ebi[i]))

    return running_ligands, running_ids, running_sequences


with open("hla.txt") as f:
    hla_list = f.read().split("&HLA_TYPE=")


def adjust_hla_name(old_hla):
    """ Converts the hla names used from the urls to their original names
        Arguments:
            old_hla (str): hla name found in the url
        Returns: 
            str: the original hla name (without any special characters)
    """
    new_hla = old_hla.replace("%3A", ":")
    new_hla = new_hla.replace("+%28", "(")
    new_hla = new_hla.replace("%28", "(")
    new_hla = new_hla.replace("%29", ")")
    new_hla = new_hla.replace("%2F", "/")
    new_hla = new_hla.replace("+", " ")
    return new_hla


df = pd.DataFrame()
for hla in hla_list:
    ligands, ids, sequences = get_data(hla)
    df = df.append(pd.DataFrame({"Allele Name": adjust_hla_name(hla),
                                 "Description": ligands,
                                 "Parent Protein IRI (NCBI)": ids,
                                 "Protein Sequence": sequences}))

df = pd.read_csv("")

df.dropna(subset=["Parent Protein IRI (NCBI)"], inplace=True)
df.drop(df[[y not in x for x, y in zip(df["Protein Sequence"], df["Description"])]].index, inplace=True)
df.drop(df[[len(x) <= 6 for x in df["Description"]]].index, inplace=True)
df.drop(df[["X" in x for x in df["Protein Sequence"]]].index, inplace=True)
df.drop(df[["U" in x for x in df["Protein Sequence"]]].index, inplace=True)

df.to_csv("syfpeithi_ncbi.csv", index=False)
