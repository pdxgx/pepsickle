import extraction_functions
import pandas as pd
import numpy as np

indir = "/Users/weeder/PycharmProjects/proteasome/data_processing/un-merged_data/"
bc_df = pd.read_csv(indir + "Breast_cancer.csv", low_memory=False)
bc_df['full_sequence'] = None

unique_protein_ids = list(bc_df['Parent Protein IRI (Uniprot)'].dropna().unique())
sequence_dict = {}
error_index = []

progress = 0
for entry in unique_protein_ids:
    try:
        sequence_dict[entry] = retrieve_UniProt_seq(entry)
    except:
        error_index.append(progress)
    progress += 1
    if progress % 100 == 0:
        print(round(progress/len(unique_protein_ids)*100, 3), "% done")


# attempt to repair null queries
progress = 0
for e in error_index:
    tmp_id = unique_protein_ids[e]
    query = compile_UniProt_url(tmp_id, include_experimental=True)
    buffer = extract_UniProt_table(query)
    new_id = buffer["Entry"][0]
    sequence_dict[unique_protein_ids[e]] = retrieve_UniProt_seq(new_id)
    progress += 1
    if progress % 100 == 0:
        print(round(progress/len(error_index)*100, 3), "% done")

for e in range(len(bc_df)):
    prot_id = str(bc_df.at[e, 'Parent Protein IRI (Uniprot)'])
    if prot_id in sequence_dict.keys():
        bc_df.at[e, 'full_sequence'] = sequence_dict[prot_id]


bc_df.dropna(subset=['full_sequence'], inplace=True)
bc_df['entry_source'] = "BC_study"
bc_df['origin_species'] = "human"
bc_df['start_pos'] = np.nan
bc_df['end_pos'] = np.nan

bc_df.to_csv(indir+"breast_cancer_data_w_sequences.csv", index=False)
