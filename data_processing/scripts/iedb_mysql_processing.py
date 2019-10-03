#! usr/bin/env python3
"""

"""
import mysql.connector
import pandas as pd
import re

# connect to mysql database
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="password",
  database="iedb_public",
  auth_plugin='mysql_native_password'
)

# generate code entry interface
mycursor = mydb.cursor()

# retrieve column names
mycursor.execute("DESCRIBE full_epitope_output")
foo_desc = mycursor.fetchall()
col_names = [desc[0] for desc in foo_desc]

# return full table of sequence info
mycursor.execute("SELECT * FROM full_epitope_output")
myresult = mycursor.fetchall()

# generate pandas df from results list
results_pd = pd.DataFrame(myresult, columns=col_names)
results_pd = results_pd.drop_duplicates()
results_pd = results_pd.dropna(subset=['linear_peptide_seq', 'sequence'])


mismatch = 0
no_pos = 0
seq_not_in_string = 0
for ind in range(len(results_pd)):
    row = results_pd.iloc[ind]
    if row['starting_position'] is None or row['ending_position'] is None:
        no_pos += 1
        if str(row['linear_peptide_seq']) in str(row['sequence']):
            search = re.search(str(row['linear_peptide_seq']), str(row['sequence']))
            results_pd['starting_position'][ind] = search.span()[0]
            results_pd['ending_position'][ind] = search.span()[1]
        else:
            seq_not_in_string += 1
    else:
        est_seq = row['sequence'][int(row['starting_position'])-1:int(row['ending_position'])]
        true_seq = row['linear_peptide_seq']
        if est_seq == true_seq:
            pass
        elif str(est_seq).upper() == str(true_seq).upper():
            results_pd['linear_peptide_seq'][ind] = str(true_seq).upper()
            results_pd['sequence'][ind] = str(est_seq).upper()
        else:
            mismatch += 1
            print(str(row['linear_peptide_seq']))
            print(str(row['sequence']))

# subset human only sequences
human_pd = results_pd[results_pd['h_organism_id'] == 9606]

# get unique indices (unique epitope, protein combos) to prevent redundancy
unique_seq_index = results_pd[['linear_peptide_seq',
                               'sequence']].drop_duplicates().index
unique_human_index = human_pd[['linear_peptide_seq',
                               'sequence', ]].drop_duplicates().index

# use indices to create output tables
results_unique_seq = results_pd.loc[unique_seq_index]
results_unique_human_seq = results_pd.loc[unique_human_index]

# write out tables
cols_to_export = ['curated_epitope_id',
                  'linear_peptide_seq',
                  'starting_position',
                  'ending_position',
                  'sequence',
                  'pubmed_id']

results_unique_seq[cols_to_export].to_csv("/Users/weeder/Data/Proteasome/unique_mammal_epitopes.csv")
results_unique_human_seq[cols_to_export].to_csv("/Users/weeder/Data/Proteasome/unique_human_epitopes.csv")
