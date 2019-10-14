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
    """
    Obtains the script of the page related to the given description (epitope).
    from the Antigen site.
    :param epitope_description: string containing the epitope sequence
    :param call: type of epitope script to pull... either T_cell or TAP
    :return: buffer with info on specific epitope used for query
    """
    if call == "T_cell":
        with urllib.request.urlopen("http://www.ddg-pharmfac.net/antijen/scripts/"
                                    + "aj_scripts/aj_tcellcalc2.pl?epitope="
                                    + epitope_description + "&AL=%25&ST=%25&CAT="
                                    + "T+Cell+Epitope") as h:
            return str(h.read())
    if call == "TAP":
        with urllib.request.urlopen("http://www.ddg-pharmfac.net/antijen/scripts/"
                                    + "aj_scripts/aj_tapcalc2.pl?epitope=" + epitope_description
                                    + "&CAT=TAP+Ligand&detailinfo=no&detailmin="
                                    + "&detailmax=") as h:
            return str(h.read())


def get_alleles(x):
    """
    gets the alleles associated with a given epitope based on the buffer (from
    get_script_page()) for that epitope.
    :param x: index of the epitope/buffer
    :return: list of alleles associated with epitope
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
    """
    gets the uniprot IRI associated with a given epitope based on the buffer
    (from get_script_page()) for that epitope.
    :param buffer: index of the epitope/buffer
    :return: uniprot IRI
    """
    if len(buffer.split("sprot-entry?")) > 1:
        return buffer.split("sprot-entry?")[1].split("\"")[0].split("http")[0]
    else:
        return None


def get_mhc_organism(x):
    """
    gets the organism associated with a given epitope based on the buffer (from
    get_script_page()) for that epitope.
    :param x: index of the epitope/buffer
    :return: name of organism associated with the epitope
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
        return None
