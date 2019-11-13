#!usr/bin/env python3
"""
extraction_functions.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script houses and consolidates functions for extracting cleavage data
from various sources.
"""
import urllib.request
import re
import ssl
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


class Error(Exception):
    """Base class for other exceptions"""
    pass


class EmptyQueryError(Error):
    """
    Raised when query returns no content table

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """
    def __init__(self, n_tables, message):
        self.n_tables = n_tables
        self.message = message


def compile_AntiJen_url(aa_sequence, query_type="T_cell"):
    """
    compiles a query url for a given sequence or subsequence based on the
    desired query type and returns the search URL link.
    :param aa_sequence: amino acid sequence or sub sequence
    :param query_type: type of sequence query to be made
    :return: URL query link
    """
    query_types = ['T_cell', 'TAP', 'TAP_substring']
    if query_type not in query_types:
        raise ValueError("Invalid. Expected either: %s" % query_types)

    if query_type == "T_cell":
        base_url = "http://www.ddg-pharmfac.net/antijen/scripts/aj_scripts/" \
                   "aj_tcellcalc2.pl?epitope={}&AL=%25&ST=%25&" \
                   "CAT=T+Cell+Epitope"
    if query_type == "TAP":
        base_url = "http://www.ddg-pharmfac.net/antijen/scripts/aj_scripts/" \
                   "aj_tapcalc2.pl?epitope={}&CAT=TAP+Ligand&detailinfo=no&" \
                   "detailmin=&detailmax="
    if query_type == "TAP_substring":
        base_url = "http://www.ddg-pharmfac.net/antijen/scripts/aj_scripts/" \
                   "aj_tapcalc.pl?epitope={}&MIN=&MAX=&allele=&CATEGORY=TAP&" \
                   "ic50MIN=&ic50MAX=&KDNMMIN=&KDNMMAX=&TAP=Search+AntiJen"

    full_query = base_url.format(aa_sequence)
    return full_query


def extract_AntiJen_table(antijen_url, page_type="T_cell"):
    """
    given a URL for the AntiJen database, this function parses the table and
    returns a pandas df with the relevant information (depending on page type)
    :param antijen_url: URL of a table page on the AntiJen website:
    http://www.ddg-pharmfac.net/antijen/AntiJen/antijenhomepage.htm
    :param page_type: type of table page to be parsed... currently takes TAP,
    T_cell, and Summary entry tables
    :return: pandas df with parsed table info
    """
    page_types = ['T_cell', 'TAP', 'Summary']
    if page_type not in page_types:
        raise ValueError("Invalid Expected either: %s" % page_types)

    # open file and parse html structure
    handle = urllib.request.urlopen(antijen_url)
    buffer = BeautifulSoup(str(handle.read()), 'html.parser')
    # find all tables
    tables = buffer.find_all("table")
    if len(tables) < 2:
        raise EmptyQueryError(len(tables), "is too few. Missing content table")
    # first tables[0] is formatted header, tables[1] is nested epitope info
    epitope_table = tables[1]

    # initialize df size based on page type
    if page_type == "T_cell":
        out_df = pd.DataFrame(columns=range(0, 8))
    if page_type == "TAP":
        out_df = pd.DataFrame(columns=range(0, 5))
    if page_type == "Summary":
        out_df = pd.DataFrame(columns=range(0, 3))

    row_num = 0  # initialize
    # iterate through "tr" entries (table rows)
    for row in epitope_table.find_all("tr"):
        columns = row.find_all('td')

        # handle if not expected column number
        if page_type == "T_cell":
            # all cols should be 8
            assert len(columns) == 8
        if page_type == "TAP":
            # columns not == 5 are table footers
            if len(columns) != 5:
                row_num += 1
                continue
        if page_type == "Summary":
            # all cols should be 3
            assert len(columns) == 3

        # init empty list to be appended
        col_entries = []
        for column in columns:
            # get text and clean HTML artifacts
            col_text = column.get_text()
            col_text = col_text.strip()
            clean_text = re.sub("\\\.*", "", col_text)
            # creates list of entries to append
            col_entries.append(clean_text)
        # if first row, use as column header
        if row_num == 0:
            out_df.columns = col_entries
        # if column matches col header, ignore
        elif set(out_df.columns) == set(col_entries):
            pass
        # append
        else:
            out_df.loc[len(out_df), :] = col_entries
        row_num += 1
    return out_df


def get_SYF_alleles():
    query = "http://www.syfpeithi.de/bin/MHCServer.dll/FindYourMotif.htm"
    handle = urllib.request.urlopen(query)
    buffer = BeautifulSoup(str(handle.read()), 'html.parser')
    tables = buffer.find_all("table")
    MHC_tab = tables[1]
    options = []
    for option in MHC_tab.find_all('option'):
        if option.text not in options:
            options.append(option.text)
    return options


def compile_SYF_url(hla_type, html_encoded=False):
    base_url = "http://www.syfpeithi.de/bin/MHCServer.dll/FindYourMotif?" \
               "HLA_TYPE={}&AASequence=&OP1=AND&select1=002&content1=&" \
               "OP2=AND&select2=004&content2=&OP3=AND&select3=003&" \
               "content3=&OP4=AND"

    if html_encoded:
        hla_query = hla_type
    else:
        hla_query = urllib.request.pathname2url(hla_type)

    query = base_url.format(hla_query)
    return query


def extract_SYF_table(query):
    handle = urllib.request.urlopen(query)
    buffer = BeautifulSoup(str(handle.read()), 'html.parser')
    tables = buffer.find_all("table")
    if len(tables) < 2:
        raise EmptyQueryError(len(tables), "is too few. Missing content table")

    epitope_table = tables[1]
    out_df = pd.DataFrame(columns=['epitope', 'prot_name', 'ebi_id',
                                   'reference'])
    start_flag = False
    for row in epitope_table.find_all("tr"):
        if start_flag is False:
            if len(row) < 4:
                if "Example for Ligand" in row.text or \
                        "T-cell epitope" in row.text:
                    start_flag = True
            else:
                pass

        if start_flag is True:
            if len(row) == 4:
                epitope, source, ref, _ = row

                epitope = epitope.text.replace("\xa0", "")
                # source links may be depreciated... if so pull name
                prot_name = source.text.replace("\xa0", " ").strip()
                if source.find('a', href=True):
                    source = source.find('a', href=True)['href']
                    prot_id = re.search("emblfetch\?(.*)", source).groups()[0]
                else:
                    source = np.nan
                    prot_id = np.nan
                if ref.find('a', href=True):
                    ref = ref.find('a', href=True)['href']
                    pubmed_id = re.search("uids=(\d*)&", ref).groups()[0]
                else:
                    pubmed_id = np.nan

                tmp_row = pd.Series([epitope, prot_name, prot_id, pubmed_id],
                                    index=out_df.columns)
                out_df = out_df.append(tmp_row, ignore_index=True)
            else:
                pass
    return out_df


def compile_UniProt_url(clean_prot_name, prot_id=np.nan,
                        include_experimental=False, html_encoded=False):
    # add flag for column=reviewed
    if include_experimental:
        base_url = "https://www.uniprot.org/uniprot/?query={}&" \
               "sort=score&columns=id,reviewed,length,organism&format=tab"
    else:
        base_url = "https://www.uniprot.org/uniprot/?query=reviewed:yes&{}&" \
               "sort=score&columns=id,reviewed, length,organism&format=tab"

    if prot_id is np.nan:
        base_entry = clean_prot_name
    if prot_id is not np.nan:
        base_entry = clean_prot_name + prot_id

    if html_encoded:
        base_entry = base_entry
    else:
        base_entry = urllib.request.pathname2url(base_entry)
    query = base_url.format(base_entry)
    return query


def extract_UniProt_table(query):
    out_df = pd.DataFrame(columns=["Entry", "Length", "Organism"])
    context = ssl._create_unverified_context()
    handle = urllib.request.urlopen(query, context=context)
    buffer = BeautifulSoup(handle, "html.parser")

    if len(buffer) > 0:
        rows = buffer.contents[0].split("\n")
        for entry in range(len(rows)):
            if rows[entry]:
                values = rows[entry].split("\t")
                if entry == 0:
                    pass
                else:
                    tmp_row = pd.Series(values, index=out_df.columns)
                    out_df = out_df.append(tmp_row, ignore_index=True)
    return out_df
