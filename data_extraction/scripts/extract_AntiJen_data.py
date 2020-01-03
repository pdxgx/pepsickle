#!usr/bin/env python3
"""
extract_AntiJen_data.py

For issues contact Ben Weeder (weeder@ohsu.edu)

This script extracts epitope examples from the AntiJen database and outputs
results as CSV files.
(http://www.ddg-pharmfac.net/antijen/AntiJen/antijenhomepage.htm)

options:
-o, --out_dir: Directory where AntiJen CSV results are exported
"""
from itertools import product
import pandas as pd
from optparse import OptionParser
import extraction_functions as ef


# define command line parameters
parser = OptionParser()
parser.add_option("-o", "--out_dir", dest="out_dir",
                  help="output directory where antigen csv's are exported")

(options, args) = parser.parse_args()

# set url for summary tcell assay table
AntiJen_tcell_summary_url = "http://www.ddg-pharmfac.net/antijen/scripts/" \
                            "aj_scripts/aj_tcellcalc.pl?epitope=&MIN=&MAX=&" \
                            "allele=CLASS-1CL&CATEGORY=T+Cell&ic50MIN=&" \
                            "Tcell=Search+AntiJen"

# pull entire table of T-cell associated antigens
tcell_epitope_table = ef.extract_AntiJen_table(AntiJen_tcell_summary_url,
                                               page_type="Summary")

# extract list of epitopes from table
tcell_epitope_list = list(tcell_epitope_table["Epitope"])

# define AA's (necessary for TAP transport queries
amino_acids = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "O", "I", "L", "K",
               "M", "F", "P", "U", "S", "T", "W", "Y", "V"]

# generate 2 AA base queries
aa_tuples = list(product(amino_acids, repeat=2))
aa_base_queries = ["".join(t) for t in aa_tuples]

# compile list of web queries
query_list = []
for q in aa_base_queries:
    query_list.append(
        ef.compile_AntiJen_url(q, query_type="TAP_substring")
    )

# compile list of TAP peptide results
TAP_peptides = []
for query in query_list:
    # return query results, skip if query has no results
    try:
        tmp_df = ef.extract_AntiJen_table(query, page_type="Summary")
        tmp_epitopes = list(tmp_df['Epitope'])
        # append any novel entries onto TAP list
        for tmp_ep in tmp_epitopes:
            if tmp_ep not in TAP_peptides:
                TAP_peptides.append(tmp_ep)

    except ef.EmptyQueryError:
        continue

# define colunms to be pulled from epitope query
tcell_epitope_df = pd.DataFrame(columns=["Epitope", "MHC_types", "Species",
                                         "Categories", "Protein_refs",
                                         "Ref_type", "Journal_refs"])

# query relevant data for each epitope entry
for epitope in tcell_epitope_list:
    # construct query and pull results into table
    query = ef.compile_AntiJen_url(epitope, query_type="T_cell")
    tmp_df = ef.extract_AntiJen_table(query, page_type="T_cell")

    # create lists to append multiple unique entries
    mhc_class = []
    mhc_species = []
    category = []
    swissprot_refs = []
    journal_refs = []

    # add unique entries to list
    for i in range(len(tmp_df)):
        entry = tmp_df.iloc[i]
        if entry['Class'] not in mhc_class:
            mhc_class.append(entry['Class'])
        if entry['MHC species'] not in mhc_species:
            mhc_species.append(entry['MHC species'])
        if entry['Category'] not in category:
            category.append(entry['Category'])
        if entry['Swiss Prot Ref'] not in swissprot_refs:
            swissprot_refs.append(entry['Swiss Prot Ref'])
        if entry['Journal Ref'] not in journal_refs:
            journal_refs.append(entry['Journal Ref'])

    # generate single entries separated with semicolons
    MHC_types = "; ".join(mhc_class)
    MHC_species = "; ".join(mhc_species)
    Categories = "; ".join(category)
    Protein_ref = "; ".join(swissprot_refs)
    Ref_type = "Uniprot"  # define reference database used
    Journal_refs = "; ".join(journal_refs)
    # reformat
    tcell_entry = pd.Series([epitope, MHC_types, MHC_species, Categories,
                             Protein_ref, Ref_type, Journal_refs],
                            index=tcell_epitope_df.columns)
    # append
    tcell_epitope_df = tcell_epitope_df.append(tcell_entry, ignore_index=True)


# repeat above with TAP associated peptides
tap_epitope_df = pd.DataFrame(columns=["Epitope", "Species", "Categories",
                                       "Protein_ref", "Ref_type",
                                       "Journal_refs"])

# pull relevant data for each TAP peptide
for epitope in TAP_peptides:
    query = ef.compile_AntiJen_url(epitope, query_type="TAP")
    tmp_df = ef.extract_AntiJen_table(query, page_type="TAP")

    mhc_species = []
    category = []
    swissprot_refs = []
    journal_refs = []
    for i in range(len(tmp_df)):
        entry = tmp_df.iloc[i]
        if entry['MHC species'] not in mhc_species:
            mhc_species.append(entry['MHC species'])
        if entry['Category'] not in category:
            category.append(entry['Category'])
        if entry['Swiss Prot Ref'] not in swissprot_refs:
            swissprot_refs.append(entry['Swiss Prot Ref'])
        if entry['Journal Ref'] not in journal_refs:
            journal_refs.append(entry['Journal Ref'])

    MHC_species = "; ".join(mhc_species)
    Categories = "; ".join(category)
    Protein_ref = "; ".join(swissprot_refs)
    Ref_type = "Uniprot"
    Journal_refs = "; ".join(journal_refs)
    TAP_entry = pd.Series([epitope, MHC_species, Categories, Protein_ref,
                           Ref_type, Journal_refs],
                          index=tap_epitope_df.columns)
    tap_epitope_df = tap_epitope_df.append(TAP_entry, ignore_index=True)


# write out data
tcell_epitope_df.to_csv(options.out_dir+"/AntiJen_Tcell_epitopes.csv", index=False)
tap_epitope_df.to_csv(options.outdir+"/AntiJen_tap_epitopes.csv", index=False)
