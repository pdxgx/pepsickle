#!usr/bin/env python3
"""
extract_AntiJen_data.py

For issues contact Ben Weeder (weeder@ohsu.edu)

[description]
"""
from itertools import product
import extraction_functions

# set url for summary tcell assay table
AntiJen_tcell_summary_url = "http://www.ddg-pharmfac.net/antijen/scripts/" \
                            "aj_scripts/aj_tcellcalc.pl?epitope=&MIN=&MAX=&" \
                            "allele=CLASS-1CL&CATEGORY=T+Cell&ic50MIN=&" \
                            "Tcell=Search+AntiJen"

##### add in code for generating TAP queries
amino_acids = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "O", "I", "L", "K",
               "M", "F", "P", "U", "S", "T", "W", "Y", "V"]

aa_tuples = list(product(amino_acids, repeat=2))
aa_base_queries = ["".join(t) for t in aa_tuples]

query_list = []
for q in aa_base_queries:
    query_list.append(
        extraction_functions.compile_AntiJen_url(q, query_type="TAP_substring")
    )

TAP_epitopes = []
for query in query_list:
    try:
        tmp_df = extraction_functions.extract_AntiJen_table(query,
                                                            page_type="Summary")
        tmp_epitopes = list(tmp_df['Epitope'])
        for tmp_ep in tmp_epitopes:
            if tmp_ep not in TAP_epitopes:
                TAP_epitopes.append(tmp_ep)
    except:
        print(query)
print(TAP_epitopes)
