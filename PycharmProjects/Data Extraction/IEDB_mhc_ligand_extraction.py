#!/usr/bin/env Python3
"""IEDB_mhc_ligand_extraction.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script extracts epitope information from the IEDB mhc ligand database 
(http://www.iedb.org/database_export_v3.php) and filters it into a more 
simplistic csv file

This script requires that `pandas` be installed within the Python
environment you are running this script in.
The program filters for a specific MHC class, the epitopes are derived from
humans, and simplifies the IDs for the epitope, proteins, and organisms

Inputs:
    The location of the IEDB csv file of interest

Outputs:
    A csv file
"""
import pandas as pd

temp = pd.read_csv("mhc_ligand_full.csv", low_memory=False)

print(temp.shape)

columns = ["Epitope IRI", "Description", "Starting Position",
           "Ending Position", "Antigen Name", "Antigen IRI", "Parent Protein",
           "Parent Protein IRI", "Organism Name",	"Organism IRI",
           "Parent Species", "Parent Species IRI", "Name", "Allele Name",
           "Allele IRI", "MHC allele class", "Epitope Comments"]

url_columns = ["Epitope IRI", "Antigen IRI", "Parent Protein IRI",
               "Organism IRI", "Organism IRI", "Parent Species IRI",
               "Allele IRI"]

urls = ["http://www.iedb.org/epitope/", "http://www.ncbi.nlm.nih.gov/protein/",
        "http://www.uniprot.org/uniprot/",
        "http://purl.obolibrary.org/obo/NCBITaxon_",
        "https://ontology.iedb.org/ontology/",
        "http://purl.obolibrary.org/obo/NCBITaxon_",
        "http://purl.obolibrary.org/obo/MRO_"]

data = {}
for i in columns:
    data[i] = temp[i]

df = pd.DataFrame(columns=columns, data=data)


def remove_url(column_name, url):
    """Removes the urls of the corresponding columns
       Arguments:
           column_name (str): the name of the column to be edited
           url (str): the url to remove
       Returns:
           n/a
    """
    df[column_name] = df[column_name].str.replace(url, "")


for i in range(len(urls)):
    remove_url(url_columns[i], urls[i])

df["Description"] = df["Description"].str.split(" +").str[0]

f = {c: ", ".join if c == "Allele Name" or c == "Allele IRI"
     else 'first' for c in columns}
df = df.groupby("Description").agg(f)

df["Allele Name"] = df["Allele Name"].apply(lambda x: ", ".join(
    sorted(list((dict.fromkeys(x.split(", ")))))))
df["Allele IRI"] = df["Allele IRI"].apply(lambda x: ", ".join(
    sorted(list((dict.fromkeys(x.split(", ")))))))

df.rename(
    columns={
        "Epitope IRI": "Epitope IRI (IEDB)",
        "Antigen IRI": "Antigen IRI (NCBI)",
        "Parent Protein IRI": "Parent Protein IRI (Uniprot)",
        "Organism IRI": "Organism IRI (NCBITaxon)",
        "Parent Species IRI": "Parent Species IRI (NCBITaxon)",
        "Allele IRI": "Allele IRI (MRO)"
    },
    inplace=True
)

df = df.dropna(subset=["Epitope IRI (IEDB)", "Description",
                       "Parent Protein IRI (Uniprot)"])
# df = df[df["MHC allele class"] == "I"]
df = df[df["MHC allele class"] == "II"]
df = df[df["Name"].str.contains("Homo sapiens")]
df = df.drop(columns=["Name", "MHC allele class"])

df.to_csv("class_II_file.csv", index=False)
