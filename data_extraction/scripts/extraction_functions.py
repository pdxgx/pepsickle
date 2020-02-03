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
    """
    performs a web query that returns all allele options in the SYF database
    :return: list of MHC options
    """
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
    """
    creates a SYF query for all epitopes associated with the given hla and
    returns a formatted query string.
    :hla_type: the name of the HLA to search in the SYF database
    :html_encoded: flag, true if special characters in hla_type should be html
    encoded in the output
    :return: query string for SYF database
    """
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
    """
    returns a parsed pandas df of entries for a given SYF query.
    :query: SYF html query
    :return: a pandas data frame with epitope entries and relevant supporting
    information
    """
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


def compile_UniProt_url(prot_name, prot_id=np.nan,
                        include_experimental=False, html_encoded=False):
    """
    compiles uniprot HTML query
    :prot_name: plain text name of the protein to be queried
    :prot_id: optional, uniprot ID or alias ID
    :include_experimental: flag, whether to include experimental proteins along
    with reviewed proteins
    :html_encoded: flag, true if output query should be html encoded.
    :return: UniProt query string for the protein requested
    """
    # add flag for column=reviewed
    if include_experimental:
        base_url = "https://www.uniprot.org/uniprot/?query={}&" \
               "sort=score&columns=id,reviewed,length,organism&format=tab"
    else:
        base_url = "https://www.uniprot.org/uniprot/?query={}&fil=reviewed" \
                   "%3Ayes&sort=score&columns=id,reviewed,length,organism&" \
                   "format=tab"

    if prot_id is np.nan:
        base_entry = prot_name
    if prot_id is not np.nan:
        base_entry = prot_name + prot_id

    if html_encoded:
        base_entry = base_entry
    else:
        base_entry = urllib.request.pathname2url(base_entry)

    query = base_url.format(base_entry)
    return query


def extract_UniProt_table(query):
    """
    extracts a pandas data frame of relevant UniProt data for a given UniProt
    query
    :query: a UniProt HTML query
    :return: pandas data frame with parsed info from UniProt
    """
    out_df = pd.DataFrame(columns=["Entry", "Reviewed", "Length", "Organism"])
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


def retrieve_UniProt_seq(UniProt_id):
    """
    returns the full UniProt reference sequence for a given UniProt ID.
    :UniProt_id: uniprot identifier
    :return: full AA sequence of a protein
    """
    base_query = "https://www.uniprot.org/uniprot/?query={}&format=fasta"
    query = base_query.format(UniProt_id)
    context = ssl._create_unverified_context()
    handle = urllib.request.urlopen(query, context=context)
    buffer = BeautifulSoup(handle, "html.parser").prettify()
    buffer_split = buffer.split("\n")
    sequence = "".join(buffer_split[1:])
    return sequence


def parse_cleavage_header(line):
    """
    takes in a header line from custom cleavage map txt files and returns key
    entry pairs.
    :line: header line from txt file
    :return: key - the meta data entry label, entry - the meta variable value
    """
    line = line.strip("#")
    key, entry = line.split("=")
    return key.strip(), entry.strip()


def parse_cleavage_map(lines):
    """
    takes in all non-header lines from a custom cleavage map file and returns
    a dictionary of source sequences and their associated cleaved fragments
    with start positions.

    :lines: list of all non-header lines in file
    :return: a dictionary of key, value pairs where the key is the source
    protein sequence and the values are tuples of (fragment, start pos) within
    the source sequence
    """
    # create iterable for file parsing
    line_iter = iter(lines)
    # initiate variables
    fragment_dict = {}
    source_seq = None
    fragment_list = []
    # set first iter
    line = next(line_iter, None)
    # while more lines in file...
    while line:
        # ignore blank lines
        if line == " " or line == "\n":
            line = next(line_iter, None)
            continue
        # ignore any comments or header lines
        if "#" not in line:
            # If start of a source sequence annotated as >
            if ">" in line:
                # if multiple source sequences, store source-fragment
                # pairs that are already collected
                if source_seq is not None:
                    fragment_dict[source_seq] = fragment_list
                # store source sequence
                source_seq = next(line_iter).strip()
                # skip ending >
                end = next(line_iter)
                # reset fragment list to collect fragments for this source
                fragment_list = []
            # if fragment line
            if ">" not in line:
                # if not annotated with specific start pos.
                if "@" not in line:
                    fragment = line.strip()
                    # if fragment not identified in source, warn
                    if len(re.findall(fragment, source_seq)) == 0:
                        print("Warning: query string not found in source "
                              "sequence")
                        print("Query: ", fragment)
                        print("Source: ", source_seq)
                        line = next(line_iter, None)
                        continue
                    # if only one occurence of the fragment in source
                    if len(re.findall(fragment, source_seq)) == 1:
                        # annotate start position
                        start_pos = re.search(fragment, source_seq).span()[0]
                    # if ambiguity raise error
                    else:
                        print(fragment, "Occurs multiple times in: ",
                              source_seq)
                        print("exact position not annotated with @")
                        raise NotImplementedError
                    # add fragment, start to list
                    fragment_list.append((fragment, start_pos))
                # if exact position is annotated
                if "@" in line:
                    # split values
                    fragment, start_pos = line.split("@")
                    fragment = fragment.strip()
                    start_pos = int(start_pos.strip())
                    # verify that fragment matches expected seq
                    assert(fragment == source_seq[start_pos:start_pos +
                                                            len(fragment)]), \
                        "Warning: fragment does not match annotated " \
                        "position in source sequence"
                    # add fragment, start to list
                    fragment_list.append((fragment, start_pos))
        line = next(line_iter, None)
    # add last source - fragment pairs to list
    fragment_dict[source_seq] = fragment_list
    return fragment_dict


def generate_cleavage_df(meta_dict, seq_dict):
    """
    takes meta data and source-fragment pairs from the same text file and
    returns a pandas df of the compiled information.
    :meta_dict: dictionary of meta variables and their values for the given
    cleavage text file.
    :seq_dict: dictionary of source sequences and their associated fragments
    :return: compiled pandas df with matched meta data and fragment examples
    """
    fragment = []
    start_pos = []
    source_seq = []
    # for each source sequence
    for source in seq_dict.keys():
        # iterate through associated entries
        for entry in seq_dict[source]:
            # append fragment
            fragment.append(entry[0])
            # append start pos
            start_pos.append(entry[1])
            # append
            source_seq.append(source)
    # zip together to make row entries
    out_df = pd.DataFrame(zip(fragment, start_pos, source_seq),
                          columns=['Fragment', 'start_pos', "source_seq"])
    # for each meta variable, populate across all rows
    for meta_key in meta_dict:
        out_df[meta_key] = meta_dict[meta_key]

    return out_df


def parse_digestion_file(file):
    """
    wrapper that combines functions to fully read and parse cleavage map
    custom files.
    :file: custom cleavage map txt file
    :return: pandas data frame with associated fragment examples and meta data
    """
    with open(file, "r") as f:
        meta_dict = {}
        lines = f.readlines()
        for i in range(len(lines)):
            # while in header, add to meta dict
            if "#" in lines[i]:
                key, entry = parse_cleavage_header(lines[i])
                meta_dict[key] = entry
            # after header, pass to seq_lines
            if "#" not in lines[i]:
                seq_lines = lines[i:]
                break
    # parse seq lines
    seq_dict = parse_cleavage_map(seq_lines)
    # generate df
    out_df = generate_cleavage_df(meta_dict, seq_dict)
    return out_df
