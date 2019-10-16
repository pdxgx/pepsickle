#!usr/bin/env python3
"""
extraction_functions.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script houses and consolidates functions for extracting cleavage data
from various sources.
"""
import urllib.request
import re
import pandas as pd
from bs4 import BeautifulSoup


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
    if len(tables) <= 2:
        print("No Query Results found")
        return None
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
