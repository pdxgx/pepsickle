#!usr/bin/env python3
"""
extract_AntiJen_data.py

For issues contact Ben Weeder (weeder@ohsu.edu)

[description]
"""
from itertools import product
import extraction_functions as ef

# set url for summary tcell assay table
AntiJen_tcell_summary_url = "http://www.ddg-pharmfac.net/antijen/scripts/" \
                            "aj_scripts/aj_tcellcalc.pl?epitope=&MIN=&MAX=&" \
                            "allele=CLASS-1CL&CATEGORY=T+Cell&ic50MIN=&" \
                            "Tcell=Search+AntiJen"

tcell_epitope_df = ef.extract_AntiJen_table(AntiJen_tcell_summary_url,
                                            page_type="Summary")

tcell_epitope_list = list(tcell_epitope_df["Epitope"])

# may be worth pulling AA names from keys of feature matrix
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
        print(query)
        continue

print(tcell_epitope_list)
print(TAP_peptides)

# next, query each entry, pull out/format relevant data, compile
