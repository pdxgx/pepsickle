#!usr/bin/env python3
"""
extraction_functions.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script consolidates functions previously written for use in epitope and
cleavage site extraction scripts.
"""
import urllib.request
import numpy as np


def get_script_page(epitope_description, call="T_cell"):
    """Obtains the script of the page related to the given description
       Arguments:
           x (str): a certain epitope description in the dataframe
       Returns:
           str: the full script of the page
    """
    if call == "T_cell":
        with urllib.request.urlopen("http://www.ddg-pharmfac.net/antijen/scripts/"
                                    + "aj_scripts/aj_tcellcalc2.pl?epitope="
                                    + x + "&AL=%25&ST=%25&CAT="
                                    + "T+Cell+Epitope") as h:
            return str(h.read())
    if call == "TAP":
        with urllib.request.urlopen("http://www.ddg-pharmfac.net/antijen/scripts/"
                                    + "aj_scripts/aj_tapcalc2.pl?epitope=" + x
                                    + "&CAT=TAP+Ligand&detailinfo=no&detailmin="
                                    + "&detailmax=") as h:
            return str(h.read())


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


def get_sprot_IRI(buffer):
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


def get_mhc_types(x):
    """Obtains the organism name of the MHC species the epitope came from\

         Used to filter for human MHC species later

         Arguments:
             x (int): the directory of the dataframe
         Returns:
             str: the MHC species of the epitope
      """
    try:
        # Manually check any epitopes which may have a HUMAN MHC (just not in the first row)
        if "HUMAN" in x["Buffer"] and \
                x["Buffer"].split("CLASS-2</td>\\n\\t<td>")[1].split("<")[0] \
                != "HUMAN" \
                and len(x["Buffer"].split("CLASS-1</td>\\n\\t<td>")) > 2:
            print(x["Description"])
        return x["Buffer"].split("CLASS-2</td>\\n\\t<td>")[1].split("<")[0]
    except IndexError:
        return np.nan
