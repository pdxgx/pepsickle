#!/usr/bin/env Python3
"""Digestion_extraction.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script allows the user to obtain a 3D numpy array containing positive
data derived from digestion products to be inputted into a model. It only
accepts comma separated value files (.csv) at the moment. Sources of each
text file can be found from files_dict where each doi is included

This script requires that `pandas`, `numpy`, and 'biopython' be installed
within the Python environment you are running this script in.
The script includes user input when multiple different sites for a given
digestion product are present in the corresponding protein sequence. In the
case of current files being inputted, the corresponding positions are:
0, 0, 37, 92, 37, 37, 8

Inputs:
    The location of the UniProt SwissProt database (saved as a fasta file)

    A list with all the locations of the text files containing the information
    of the positive data set (the epitopes/proteasome products and their
    respective parent proteins.
    The text files are formatted in the following way:
        Epitope Comment to signify the data is from a digestion map
        Parent Protein Description
        Parent Protein IRI (UniProt)
        A list of the digested products (one per line)



Outputs:
    A csv file containing the positive data set information (some with
    window sequences instead of digestion products)

    A numpy array containing the feature set for each generated window from
    the digestion products which can be directly inputed into the model
"""
import pandas as pd
import numpy as np
from Bio import SeqIO

expand = 10

# X below denotes an incomplete window
_features = {
    'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6,     29.5,  -0.495],
    'C': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.07,  51.6,  0.081],
    'D': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.77,  44.2,  9.573],
    'E': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.22,  70.6,  3.173],
    'F': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 5.48,  135.2, -0.37],
    'G': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.97,  0,     0.386],
    'H': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 7.59,  96.3,  2.029],
    'I': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6.02,  108.5, -0.528],
    'K': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9.74,  98,    2.101],
    'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.98,  108.6, -0.342],
    'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.74,  104.9, -0.324],
    'N': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.41,  58.8,  2.354],
    'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6.3,   54.1,  -0.322],
    'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 5.65,  81.5,  2.176],
    'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 10.76, 110.5, 4.383],
    'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 5.68,  29.9,  0.936],
    'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 5.6,   56.8,  0.853],
    'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 5.96,  80.5,  -0.308],
    'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 5.89,  164,   -0.27],
    'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 5.66,  137,   1.677],
    'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     -1,    -1000]
}
files_dict = {
    "cbeta-casein.txt": "10.1074/jbc.M000740200",
    "ibeta-casein.txt": "10.1074/jbc.M000740200",
    "HBVcAg.txt": "10.1006/jmbi.1998.2530",
    "HIV-1 Nef-1.txt": "10.1084/jem.191.2.239",
    "Insulin B chain.txt": "10.1006/jmbi.1998.2530",
    "Jak 1.txt": "10.1006/jmbi.1998.2530",
    "Ova 239-281.txt": "10.1006/jmbi.1998.2530",
    "OvaY 51-71.txt": "10.1006/jmbi.1998.2530",
    "OvaY 249-269.txt": "10.1006/jmbi.1998.2530",
    "P53wt.txt": "10.1006/jmbi.1998.2530",
    "pp89.txt": "10.1006/jmbi.1998.2530",
    "RU1.txt": "10.1016/S1074-7613(00)80163-6",
    "SSX2-1.txt": "10.4049/jimmunol.168.4.1717",
    "SSX2-2.txt": "10.4049/jimmunol.168.4.1717",
    "SSX2-3.txt": "10.4049/jimmunol.168.4.1717",
    "SSX2-4.txt": "10.4049/jimmunol.168.4.1717",
    "SSX2-5.txt": "10.4049/jimmunol.168.4.1717",
    "SSX2-6.txt": "10.4049/jimmunol.168.4.1717",
    "SSX2-7.txt": "10.4049/jimmunol.168.4.1717",
    "SSX2-8.txt": "10.4049/jimmunol.168.4.1717",
    "SSX2-9.txt": "10.4049/jimmunol.168.4.1717",
    "SSX2-10.txt": "10.4049/jimmunol.168.4.1717",
    "SSX2-11.txt": "10.4049/jimmunol.168.4.1717",
    "SSX2-12.txt": "10.4049/jimmunol.168.4.1717",
    "SSX2-13.txt": "10.4049/jimmunol.168.4.1717",
    "tyrosinase.txt": "10.1073/pnas.1101892108",
    "SP110.txt": "10.1126/science.1130660",
    "GP100-2.txt": "10.4049/jimmunol.1302032",
    "GP100-1.txt": "10.1126/science.1095522",
    "cPrP.txt": "10.4049/jimmunol.172.2.1083",
    "iPrP.txt": "10.4049/jimmunol.172.2.1083",
    "HIV RT-1.txt": "10.1128/JVI.00968-06",
    "HIV RT-2.txt": "10.1128/JVI.00968-06",
    "HIV-1 Nef-2.txt": "10.1073/pnas.1232228100",
    "PRAME 90–116.txt": "10.1084/jem.193.1.73",
    "PRAME 133–159.txt": "10.1084/jem.193.1.73",
    "PRAME 290–316.txt": "10.1084/jem.193.1.73",
    "PRAME 415–441.txt": "10.1084/jem.193.1.73",
    "NS4B-Mu.txt": "10.1128/JVI.79.8.4870–4876.2005",
    "NS4B-WT.txt": "10.1128/JVI.79.8.4870–4876.2005",
    "GP100 209-217.txt": "10.4049/jimmunol.1103213",
    "MAGE-A3 114-122.txt": "10.4049/jimmunol.1103213",
    "MAGE-A3 271-279.txt": "10.4049/jimmunol.1103213",
    "MAGE-A10 254-262.txt": "10.4049/jimmunol.1103213",
    "MAGE-C2 191-200.txt": "10.4049/jimmunol.1103213",
    "MAGE-C2 336-344.txt": "10.4049/jimmunol.1103213",
    "Melan-A 26-35.txt": "10.4049/jimmunol.1103213",
    "Tyrosinase 369-377.txt": "10.4049/jimmunol.1103213",
    "MV 438–446-1.txt": "10.1099/0022-1317-82-9-2131",
    "MV 438–446-2.txt": "10.1099/0022-1317-82-9-2131",
    "MV 438–446-3.txt": "10.1099/0022-1317-82-9-2131",
    "MV 438–446-4.txt": "10.1099/0022-1317-82-9-2131",
    "Insulin B chain-2.txt": "10.1073/pnas.0508621102",
    "LLO 291-317.txt": "10.1182/blood-2010-12-325035",
    "TEL-AML1 319-348.txt": "10.1182/blood-2010-12-325035",
    "pLLO 91-99.txt": "10.1093/intimm/dxh352",
    "pLLO 99A.txt": "10.1093/intimm/dxh352",
    "Proinsulin.txt": "10.2337/diabetes.54.7.2053",
    "Proteasome C5 120–146.txt": "10.4049/jimmunol.164.1.329",
    "iWT1 313-336.txt": "10.1158/1078-0432.CCR-06-1337",
    "cWT1 313-336.txt": "10.1158/1078-0432.CCR-06-1337",
    "iEnolase.txt": "10.1084/jem.194.1.1",
    "cEnolase.txt": "10.1084/jem.194.1.1",
    "ENO1.txt": "10.1073/pnas.95.21.12504",
    "Snca.txt": "10.1016/j.bbamcr.2013.11.018",
    "PARK7.txt": "10.1016/j.bbamcr.2011.11.010",
    "ALB 1-24.txt": "10.1681/ASN.2007111233",
    "HLA-B27 165-194.txt": "10.1074/jbc.M308816200",
    "Histone 2A 77-105.txt": "10.1074/jbc.M308816200",
    "Fatty Acid Synthase 1689-1718.txt": "10.1074/jbc.M308816200",
    "Beta-2m 1-24.txt": "10.1074/jbc.M308816200",
    "OvaRFP.txt": "10.1016/j.molcel.2012.08.029",
    "p21RFP-26S.txt": "10.1016/j.molcel.2012.08.029",
    "p21RFP-20S.txt": "10.1016/j.molcel.2012.08.029"
}

sprot_files = ["cbeta-casein.txt", "ibeta-casein.txt", "ENO1.txt",
               "HIV-1 Nef-2.txt", "cEnolase.txt", "iEnolase.txt", "Snca.txt"]

sprot_index = SeqIO.index_db("sprot/sprot_index.idx",
                             "sprot/uniprot_sprot.fasta", "fasta",
                             key_function=lambda x: x.split("|")[1])


def load_data(file_name):
    """Load data from the text file into a pandas Dataframe
       Arguments:
            file_name (str): name of file
       Returns:
            pd.Dataframe: Dataframe created from text file
    """
    with open("files/" + file_name) as f:
        e_comments = f.readline().strip()
        protein = f.readline().strip()
        if file_name in sprot_files:
            protein_id = f.readline().strip()
            try:
                protein_seq = str(sprot_index[protein_id].seq)
            except KeyError:
                protein_seq = np.nan
        else:
            buffer = f.readline().strip()
            if buffer != ">":
                protein_seq = buffer
            else:
                protein_seq = ""
                buffer = f.readline().strip()
                while buffer != ">":
                    protein_seq += buffer
                    buffer = f.readline().strip()
            protein_id = np.nan

        peptides = []
        buffer = f.readline().strip()
        while buffer:
            if buffer == ">":
                peptide_buffer = ""
                buffer = f.readline().strip()
                while buffer != ">":
                    peptide_buffer += buffer
                    buffer = f.readline().strip()
                buffer = peptide_buffer
            if protein_seq.find(buffer) + len(buffer) != len(protein_seq):
                peptides.append(buffer)
            buffer = f.readline().strip()

        df = pd.DataFrame()
        df["Description"] = peptides
        df["Parent Protein"] = protein
        df["Parent Protein IRI (Uniprot)"] = protein_id
        df["Protein Sequence"] = protein_seq
        df["Epitope Comments"] = e_comments
        df["Source"] = files_dict[file_name]
    return df


df = pd.DataFrame()
for file in files_dict.keys():
    df = df.append(load_data(file))


def get_position(x):
    """Obtains the position of a given digestion product from the protein
       context

       Gets the position of the product from the entire protein sequence;
       if multiple different positions are found, asks the user to input the
       correct position and continues to ask until a correct position is given

       Arguments:
            x (pd.Series): directory of the dataframe

       Returns:
            int: the position (from the N-terminus)
    """
    if x["Protein Sequence"].count(x["Description"]) > 1:
        position = int(input("Multiple positions found for " + x["Description"]
                             + " in " + x["Protein Sequence"]
                             + "!\nPlease input the correct position: "))
        while type(position) != int \
                or x["Protein Sequence"][position:
        position + len(x["Description"])] != x["Description"]:
            position = int(input("Multiple positions found for " + x["Description"]
                                 + " in " + x["Protein Sequence"]
                                 + "!\nPlease input the correct position: "))
        return position
    else:
        return x["Protein Sequence"].find(x["Description"])


def get_window(x, position=-1):
    """Obtains the window of interest (with the cleavage site at the center)
       from the information given; when necessary, adds "X" to the ends of the
       window if cleavage site is at the beginning/end of protein
       (meaning the window is incomplete)
       Arguments:
            x (pd.Series): directory of the dataframe
            position (int): position of the start of the description in the
                            protein sequence
       Returns:
            str: the window sequence
    """
    window = ""
    position = int(position)
    incomplete = False

    if position == -1:
        position = get_position(x)

    if position < expand:
        incomplete = True
        for i in range(expand - position):
            window += "X"
        window += x["Protein Sequence"][:position + expand + 1]
    if position >= len(x["Protein Sequence"]) - expand:
        incomplete = True
        if len(window) == 0:
            window += x["Protein Sequence"][position - expand:
                                            len(x["Protein Sequence"]) + 1]
        for i in range(expand - (len(x["Protein Sequence"])
                                 - position - 1)):
            window += "X"
    if incomplete is False:
        window = x["Protein Sequence"][position - expand: position
                                                          + expand
                                                          + 1]
    return window


def get_upstream_positives(x):
    """Obtains the upstream positive site for each digestion product if the
       product is not at the beginning of the protein sequence
       Arguments:
            x (int): the directory of the dataframe
       Returns:
            int/float: the position of the upstream positive site
                       (if the product is at the very beginning, returns 
                       np.nan)
    """
    product_position = get_position(x)
    if product_position > 0:
        return product_position - 1
    else:
        return np.nan


df["Window"] = df.apply(lambda x: get_window(x), axis=1)

buffer = df.copy()
buffer["Positions"] = buffer.apply(get_upstream_positives, axis=1)
buffer.dropna(subset=["Positions"], inplace=True)
buffer["Window"] = buffer.apply(lambda x: get_window(x, x["Positions"]), axis=1)
buffer["Description"] = np.nan

df = df.append(buffer, sort=False)
df.drop_duplicates(subset=["Window"], inplace=True)
df.drop(columns=["Positions"], inplace=True)

print(df.shape)

df.to_csv("csv/digestion.csv", index=False)

for i in range(expand * 2 + 1):
    df[i] = df["Window"].apply(lambda x: _features[x[i]])

np_positives = np.array(df.apply(lambda x: pd.DataFrame(
            {y: x[y] for y in range(expand * 2 + 1)}).to_numpy(),
                          axis=1).to_list())

n_samples, nx, ny = np_positives.shape
np.save("npy/digestion_positives_2d.npy", np_positives.reshape(
    (n_samples, nx*ny)))
