#! usr/bin/env python3
"""
iedb_mysql_processing.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script takes the full IEDB query table created by `iedb_mysql_query.sql`
and returns a processed csv of results with the following columns:
- curated_epitope_id: IEDB ID
- linear_peptide_seq: fragment sequence
- starting_position: start position of fragment within source protein
- ending_position: end position of fragment within source protein
- sequence: full source sequence
- pubmed_id: Lit reference ID

Results can be exported for either human epitopes only, or for all mammals in
IEDB (including human).

Options:
-p, --password: mysql user password
-u, --user: mysql user to run commands as, default is root
-o, --out_dir: output directory where CSV is exported
-h, --human_only: flag that allows for export of only human data. default is
export of all mammal data (including human)
"""
import mysql.connector
import pandas as pd
from optparse import OptionParser

# define command line parameters
parser = OptionParser()
parser.add_option("-u", "--user", dest="user", default="root",
                  help="user to run mysql query as, defaults to root.")
parser.add_option("-p", "--password", dest="password",
                  help="mysql password for user", default=None)
parser.add_option("-o", "--out_dir", dest="out_dir",
                  help="output directory where CSV is exported")
parser.add_option("--human_only", action="store_true", dest="human_only",
                  default=False, help="Flags export of only human based data. "
                                      "Default is all mammal (including human)"
                  )

(options, args) = parser.parse_args()

# connect to mysql database
mydb = mysql.connector.connect(
  host="localhost",
  user=options.user,
  passwd=options.password,
  database="iedb_public",
  auth_plugin='mysql_native_password'
)

# generate code entry interface
mycursor = mydb.cursor()

# retrieve column names
mycursor.execute("DESCRIBE full_epitope_output")
feo_desc = mycursor.fetchall()
col_names = [desc[0] for desc in feo_desc]

# return full table of sequence info
mycursor.execute("SELECT * FROM full_epitope_output")
myresult = mycursor.fetchall()

# generate pandas df from results list
results_pd = pd.DataFrame(myresult, columns=col_names)
results_pd = results_pd.drop_duplicates()
results_pd = results_pd.dropna(subset=['linear_peptide_seq', 'sequence'])


# Note: this is cursory check, filtering is performed downstream downstream
mismatch = 0
no_pos = 0
seq_not_in_string = 0
for ind in range(len(results_pd)):
    # pull each entry
    row = results_pd.iloc[ind]
    # check if position info is missing
    if row['starting_position'] is None or row['ending_position'] is None:
        # tally if missing position
        no_pos += 1
        # convert to comparable strings
        fragment = str(row['linear_peptide_seq'])
        ref = str(row['sequence'])

        if fragment.upper() in ref.upper():
            # if fragment contained in reference, pass
            pass
        else:
            # if fragment not contained in reference, tally
            seq_not_in_string += 1

    # if position info is available
    else:
        # pull expected fragment sequence
        est_seq = str(row['sequence'][int(row['starting_position'])-1:
                                      int(row['ending_position'])])
        # pull listed fragment sequence
        true_seq = str(row['linear_peptide_seq'])

        if est_seq.upper() == true_seq.upper():
            # if sequences match, pass
            pass
        else:
            # if sequences do not match, tally and pring mismatch
            mismatch += 1
            print("Expected fragment sequence: ", est_seq)
            print("Listed fragment sequence: ", true_seq)

# print quality check
print("Entries with missing position info: ", no_pos)
print("Entries not in reference string: ", seq_not_in_string)
print("Entries with mismatch in listed vs. reference fragment: ", mismatch)

results_pd = results_pd.reset_index()
agg_cols = results_pd.columns.drop(['linear_peptide_seq', 'sequence',
                                    'starting_position', 'ending_position',
                                    'h_organism_id'])
agg_dict = {}
for col in agg_cols:
    agg_dict[col] = ";".join

# convert all data to
results_pd[agg_cols] = results_pd[agg_cols].astype("str")
results_pd = results_pd.groupby(['linear_peptide_seq', 'sequence',
                                 'starting_position', 'ending_position',
                                 'h_organism_id']).agg(agg_dict).reset_index()

# define columns to export
cols_to_export = ['curated_epitope_id',
                  'linear_peptide_seq',
                  'starting_position',
                  'ending_position',
                  'sequence',
                  'database',
                  'accession',
                  'mhc_allele_name',
                  'h_organism_id',
                  'pubmed_id']

# export csv based on command line parameters
if options.human_only:
    human_pd = results_pd[results_pd['h_organism_id'] == 9606]
    human_pd[cols_to_export].to_csv(
        options.out_dir + "/unique_iedb_epitopes_human_only.csv",
        index=False
    )
else:
    results_pd[cols_to_export].to_csv(
        options.out_dir + "/unique_iedb_epitopes.csv",
        index=False
    )
