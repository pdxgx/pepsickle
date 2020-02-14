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

# initiate data frame
digestion_df = pd.DataFrame(columns=['fragment', 'start_pos', 'end_pos',
                                     'full_sequence', 'Name', 'DOI', 'Subunit',
                                     'Proteasome', 'Organism', 'UniProt'])

# iterate through and parse each file
for file in file_list:
    print("parsing: ", file)
    file_path = options.in_dir + "/" + file
    tmp_df = parse_digestion_file(file_path)
    digestion_df = digestion_df.append(tmp_df, sort=True)

digestion_df['entry_source'] = "cleavage_map"

# for now drop all non-20S and all missing proteasome type
digestion_df = digestion_df[digestion_df['Subunit'] == "20s"]
digestion_df = digestion_df[digestion_df['Proteasome'] != "?"]

# print out summary info
# print(len(digestion_df['DOI'].unique()))
# print(len(digestion_df))
# print(digestion_df['Proteasome'].value_counts())
# print(digestion_df['DOI'].unique())

# export
digestion_df.to_csv(options.out_dir + "/compiled_digestion_df.csv",
                    index=False)
