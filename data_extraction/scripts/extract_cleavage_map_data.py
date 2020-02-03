#!usr/bin/env python3
"""
extract_cleavage_map_data.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script extracts parses custom made text files with cleavage map
annotations from primary literature, and compiles a csv containing all
annotated examples in a given file set (directory).

options:
-i, --in_dir: Directory of cleavage map text files to be parsed
-o, --out_dir: Directory where cleavage map CSV results are exported
"""
from extraction_functions import *
import re
import os
import pandas as pd
from optparse import OptionParser


# define command line parameters
parser = OptionParser()
parser.add_option("-i", "--in_dir", dest="in_dir",
                  help="input directory of cleavage map raw txt files to be"
                       "parsed.")
parser.add_option("-o", "--out_dir", dest="out_dir",
                  help="output directory where antigen csv's are exported")

(options, args) = parser.parse_args()

file_list = os.listdir(options.in_dir)


def parse_header(line):
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


def generate_pandas_df(meta_dict, seq_dict):
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
                key, entry = parse_header(lines[i])
                meta_dict[key] = entry
            # after header, pass to seq_lines
            if "#" not in lines[i]:
                seq_lines = lines[i:]
                break
    # parse seq lines
    seq_dict = parse_cleavage_map(seq_lines)
    # generate df
    out_df = generate_pandas_df(meta_dict, seq_dict)
    return out_df


# initiate data frame
digestion_df = pd.DataFrame(columns=['Fragment', 'start_pos', 'source_seq',
                                     'Name', 'DOI', 'Subunit', 'Proteasome',
                                     'Organism', 'UniProt'])

# iterate through and parse each file
for file in file_list:
    print("parsing: ", file)
    file_path = options.in_dir + "/" + file
    tmp_df = parse_digestion_file(file_path)
    digestion_df = digestion_df.append(tmp_df)

# for now drop all non-20S and all missing proteasome type
digestion_df = digestion_df[digestion_df['Subunit'] == "20S"]
digestion_df = digestion_df[digestion_df['Proteasome'] != "?"]

# export
digestion_df.to_csv(options.out_dir + "/compiled_digestion_df.csv",
                    index=False)
