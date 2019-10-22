#!usr/bin/env python3
"""
extract_AntiJen_data.py

For issues contact Ben Weeder (weeder@ohsu.edu)

[description]
"""
from itertools import product
import pandas as pd
import extraction_functions as ef

outdir = "/Users/weeder/PycharmProjects/proteasome/data_processing/" \
         "un-merged_data/positives/"

# set url for summary tcell assay table
AntiJen_tcell_summary_url = "http://www.ddg-pharmfac.net/antijen/scripts/" \
                            "aj_scripts/aj_tcellcalc.pl?epitope=&MIN=&MAX=&" \
                            "allele=CLASS-1CL&CATEGORY=T+Cell&ic50MIN=&" \
                            "Tcell=Search+AntiJen"

tcell_epitope_table = ef.extract_AntiJen_table(AntiJen_tcell_summary_url,
                                               page_type="Summary")

tcell_epitope_list = list(tcell_epitope_table["Epitope"])

# may be worth pulling AA names from keys of feature matrix in future
amino_acids = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "O", "I", "L", "K",
               "M", "F", "P", "U", "S", "T", "W", "Y", "V"]

aa_tuples = list(product(amino_acids, repeat=2))
aa_base_queries = ["".join(t) for t in aa_tuples]

query_list = []
for q in aa_base_queries:
    query_list.append(
        ef.compile_AntiJen_url(q, query_type="TAP_substring")
    )

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

# define colunms to be pulled
tcell_epitope_df = pd.DataFrame(columns=["Epitope", "MHC_types", "Species",
                                         "Categories", "Protein_refs",
                                         "Ref_type", "Journal_refs"])

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
    Protein_refs = "; ".join(swissprot_refs)
    Ref_type = "Uniprot"  # define reference database used
    Journal_refs = "; ".join(journal_refs)
    # reformat
    tcell_entry = pd.Series([epitope, MHC_types, MHC_species, Categories,
                             Protein_refs, Ref_type, Journal_refs],
                            index=tcell_epitope_df.columns)
    # append
    tcell_epitope_df = tcell_epitope_df.append(tcell_entry, ignore_index=True)


# repeat above with TAP associated peptides
tap_epitope_df = pd.DataFrame(columns=["Epitope", "Species", "Categories",
                                       "Protein_refs", "Ref_type",
                                       "Journal_refs"])

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
    Protein_refs = "; ".join(swissprot_refs)
    Ref_type = "Uniprot"
    Journal_refs = "; ".join(journal_refs)
    TAP_entry = pd.Series([epitope, MHC_species, Categories, Protein_refs,
                           Ref_type, Journal_refs],
                          index=tap_epitope_df.columns)
    tap_epitope_df = tap_epitope_df.append(TAP_entry, ignore_index=True)


# put write outs here

tcell_epitope_df.to_csv(outdir+"/AntiJen_Tcell_epitopes.csv", index=False)
tap_epitope_df.to_csv(outdir+"/AntiJen_tap_epitopes.csv", index=False)
