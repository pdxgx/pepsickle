import pandas as pd
import extraction_functions

indir = "/Users/weeder/PycharmProjects/proteasome/data_processing/un-merged_data/positives/"
iedb_df = pd.read_csv(indir + "IEDB.csv", low_memory=False)

# iedb_df['full_sequence'] = None


unique_iedb_ids = list(iedb_df['Parent Protein IRI (Uniprot)'].dropna().unique())
unique_ncbi_ids = list(iedb_df['Parent Protein IRI (NCBI)'].dropna().unique())
unique_protein_ids = unique_iedb_ids + unique_ncbi_ids
sequence_dict = {}
error_index = []

progress = 0
# for entry in unique_iedb_ids:
# for entry in unique_ncbi_ids:
for entry in unique_protein_ids:
    try:
        sequence_dict[entry] = retrieve_UniProt_seq(entry)
    except:
        error_index.append(progress)
    progress +=1
    if progress % 100 == 0:
        print(round(progress/len(unique_protein_ids)*100, 3), "% done")


# attempt to repair null queries
progress = 0
for e in error_index:
    tmp_id = unique_iedb_ids[e]
    query = compile_UniProt_url(tmp_id, include_experimental=True)
    buffer = extract_UniProt_table(query)
    new_id = buffer["Entry"][0]
    sequence_dict[unique_iedb_ids[e]] = retrieve_UniProt_seq(new_id)
    progress+=1
    if progress % 100 == 0:
        print(round(progress/len(error_index)*100, 3), "% done")

for e in range(len(iedb_df)):
    prot_id = str(iedb_df.at[e, 'Parent Protein IRI (Uniprot)'])
    if prot_id in sequence_dict.keys():
        iedb_df.at[e, 'full_sequence'] = sequence_dict[prot_id]


iedb_df.dropna(subset=['full_sequence'], inplace=True)

iedb_df = iedb_df[['Epitope IRI (IEDB)', 'Description', 'Starting Position',
       'Ending Position', 'Parent Protein IRI (Uniprot)',
         'Parent Species IRI (NCBITaxon)', 'Source', 'full_sequence']]
iedb_df.to_csv(indir+"IEDB_data_w_sequences.csv", index=False)
