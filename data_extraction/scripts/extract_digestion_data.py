#!/usr/bin/env Python3
"""Digestion_extraction.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script allows the user to obtain a 3D numpy array containing positive
data derived from digestion products to be inputted into a model. It only
accepts comma separated value files (.csv) at the moment. Sources of each
text file can be found from files_dict where each doi is included

This script requires that `pandas` be installed within the Python environment
you are running this script in. The script includes user input when multiple
different sites for a given digestion product are present in the corresponding
protein sequence. In the case of current files being inputted,
the corresponding positions are:
0, 0, 37, 92, 37, 37, 8, 296, 296, 296, 295, 295, 294, 292, 291, 290, 290, 288,
287, 286, 285, 118

Inputs:
    A list with all the locations of the text files containing the information
    of the positive data set (the epitopes/proteasome products and their
    respective parent proteins.
    The text files are formatted in the following way:
        Epitope Comments describing the digestion map
        Full Parent Protein Description (surrounded by '>' if the sequence is
        too long)
        Uniprot Parent ID (if applicable)
        A list of the digested fragments (one per line)

Outputs:
    A csv file containing the fragment information
"""
import pandas as pd
import numpy as np

files_dict = {
    "26Sbeta-casein.txt": "10.1074/jbc.M000740200",
    "20Sbeta-casein.txt": "10.1074/jbc.M000740200",
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
    "PRAME 90-116.txt": "10.1084/jem.193.1.73",
    "PRAME 133-159.txt": "10.1084/jem.193.1.73",
    "PRAME 290-316.txt": "10.1084/jem.193.1.73",
    "PRAME 415-441.txt": "10.1084/jem.193.1.73",
    "NS4B-Mu.txt": "10.1128/JVI.79.8.4870–4876.2005",
    "NS4B-WT.txt": "10.1128/JVI.79.8.4870–4876.2005",
    "GP100 209-217.txt": "10.4049/jimmunol.1103213",  # Issue... this study uses an intermediate constit/immuno proteasome
    "MAGE-A3 114-122.txt": "10.4049/jimmunol.1103213",
    "MAGE-A3 271-279.txt": "10.4049/jimmunol.1103213",
    "MAGE-A10 254-262.txt": "10.4049/jimmunol.1103213",
    "MAGE-C2 191-200.txt": "10.4049/jimmunol.1103213",
    "MAGE-C2 336-344.txt": "10.4049/jimmunol.1103213",
    "Melan-A 26-35.txt": "10.4049/jimmunol.1103213",
    "Tyrosinase 369-377.txt": "10.4049/jimmunol.1103213",
    "MV 438-446-1.txt": "10.1099/0022-1317-82-9-2131",
    "MV 438-446-2.txt": "10.1099/0022-1317-82-9-2131",
    "MV 438-446-3.txt": "10.1099/0022-1317-82-9-2131",
    "MV 438-446-4.txt": "10.1099/0022-1317-82-9-2131",
    "Insulin B chain-2.txt": "10.1073/pnas.0508621102",
    "LLO 291-317.txt": "10.1182/blood-2010-12-325035",
    "TEL-AML1 319-348.txt": "10.1182/blood-2010-12-325035",
    "pLLO 91-99.txt": "10.1093/intimm/dxh352",
    "pLLO 99A.txt": "10.1093/intimm/dxh352",
    "Proinsulin.txt": "10.2337/diabetes.54.7.2053",
    "Proteasome C5 120-146.txt": "10.4049/jimmunol.164.1.329",
    "iWT1 313-336.txt": "10.1158/1078-0432.CCR-06-1337",
    "cWT1 313-336.txt": "10.1158/1078-0432.CCR-06-1337",
    "iEnolase.txt": "10.1084/jem.194.1.1",
    "cEnolase.txt": "10.1084/jem.194.1.1",
    "Snca.txt": "10.1016/j.bbamcr.2013.11.018",
    "PARK7.txt": "10.1016/j.bbamcr.2011.11.010",
    "ALB 1-24.txt": "10.1681/ASN.2007111233",
    "HLA-B27 165-194.txt": "10.1074/jbc.M308816200",
    "Histone 2A 77-105.txt": "10.1074/jbc.M308816200",
    "Fatty Acid Synthase 1689-1718.txt": "10.1074/jbc.M308816200",
    "Beta-2m 1-24.txt": "10.1074/jbc.M308816200",
    "OvaRFP.txt": "10.1016/j.molcel.2012.08.029",
    "p21RFP-20S.txt": "10.1016/j.molcel.2012.08.029",
    "p21RFP-26S.txt": "10.1016/j.molcel.2012.08.029",
    "cGP100 204-222.txt": "10.4049/jimmunol.176.2.1053",
    "iGP100 204-222.txt": "10.4049/jimmunol.176.2.1053",
    "cTyrosinase 364-382.txt": "10.4049/jimmunol.176.2.1053",
    "iTyrosinase 364-382.txt": "10.4049/jimmunol.176.2.1053",
    "cMAGE-C2 331-349.txt": "10.4049/jimmunol.176.2.1053",
    "iMAGE-C2 331-349.txt": "10.4049/jimmunol.176.2.1053"
}

sprot_files = ["26Sbeta-casein.txt", "20Sbeta-casein.txt", "HIV-1 Nef-2.txt",
               "cEnolase.txt", "iEnolase.txt", "Snca.txt"]

mammal_other_files = ["ALB 1-24.txt", "PARK7.txt", "Snca.txt"]

def load_data(file_name):
    """Load data from the text file into a pandas Dataframe
       Arguments:
            file_name (str): name of file
       Returns:
            pd.Dataframe: Dataframe created from text file
    """
    with open("files/" + file_name) as f:
        e_comments = f.readline().strip()
        if "20S" in e_comments:
            complex_type = "20S"
        elif "26S" in e_comments:
            complex_type = "26S"
        elif "19S" in e_comments:
            complex_type = "19S"
        else:
            complex_type = np.nan

        if "(i)" in e_comments:
            immune_type = "immuno"
        elif "(c)" in e_comments:
            immune_type = "constitutive"
        else:
            immune_type = np.nan

        protein = f.readline().strip()
        if file_name in sprot_files:
            protein_id = f.readline().strip()
        else:
            protein_id = np.nan
        buffer = f.readline().strip()
        if buffer != ">":
            protein_seq = buffer
        else:
            protein_seq = ""
            buffer = f.readline().strip()
            while buffer != ">":
                protein_seq += buffer
                buffer = f.readline().strip()
        if file_name in mammal_other_files:
            origin_species = "mammal_other"
        else:
            origin_species = "human"

        peptides = []
        ids = []
        start_pos = []
        end_pos = []
        count = 0
        buffer = f.readline().strip()
        while buffer:
            if buffer == ">":
                peptide_buffer = ""
                buffer = f.readline().strip()
                while buffer != ">":
                    peptide_buffer += buffer
                    buffer = f.readline().strip()
                buffer = peptide_buffer
            if buffer in protein_seq:
                peptides.append(buffer)
                ids.append(file_name[:-4] + "-" + str(count))
                start_pos.append(get_position(peptides[count], protein_seq))
                end_pos.append(start_pos[count] + len(peptides[count]))
                count += 1
            buffer = f.readline().strip()

        df = pd.DataFrame()
        df["epitope_id"] = ids
        df["full_seq_accession"] = protein_id
        df['full_seq_database'] = "UniProt"
        df["end_pos"] = end_pos
        df["entry_source"] = "cleavage_map"
        df["fragment"] = peptides
        df["full_sequence"] = protein_seq
        df["lit_reference"] = files_dict[file_name]
        df["origin_species"] = origin_species
        df["start_pos"] = start_pos
        df["complex_type"] = complex_type
        df["immune_type"] = immune_type
    return df


def get_position(fragment, full_sequence):
    """Obtains the start position of a given digestion product from the protein
       context

       Gets the position of the product from the entire protein sequence;
       if multiple different positions are found, asks the user to input the
       correct position and continues to ask until a correct position is given

       Arguments:
            fragment (string): sequence of the fragment
            full_sequence (string): sequence of the full sequence

       Returns:
            int: the position (from the start/N-terminus), 1-based
    """
    if full_sequence.count(fragment) > 1:
        position = int(input("Multiple positions found for " + fragment
                             + " in " + full_sequence
                             + "!\nPlease input the correct position: "))
        while type(position) != int or \
                full_sequence[position:position + len(fragment)] != fragment:
            position = int(input("Multiple positions found for " + fragment
                                 + " in " + full_sequence
                                 + "!\nPlease input the correct position: "))
        return position
    else:
        return full_sequence.find(fragment)


df = pd.DataFrame()
for file in files_dict.keys():
    df = df.append(load_data(file))

df.to_csv("edited_digestion.csv", index=False)
