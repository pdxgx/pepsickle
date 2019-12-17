import pandas as pd
import extraction_functions

indir = "/Users/weeder/PycharmProjects/proteasome/data_processing/un-merged_data/"
antijen_tcell_df = pd.read_csv(indir + "AntiJen_Tcell_epitopes.csv", low_memory=False)

unique_protein_ids = list(antijen_tcell_df['Protein_refs'].dropna().unique())
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
    if tmp_id != "not applicable":
        try:
            query = compile_UniProt_url(tmp_id, include_experimental=True)
            buffer = extract_UniProt_table(query)
            new_id = buffer["Entry"][0]
            sequence_dict[unique_protein_ids[e]] = retrieve_UniProt_seq(new_id)
        except IndexError:
            pass
    progress += 1
    if progress % 10 == 0:
        print(round(progress/len(error_index)*100, 3), "% done")


antijen_tcell_df['full_sequence'] = None

for e in range(len(antijen_tcell_df)):
    prot_id = str(antijen_tcell_df.at[e, 'Protein_refs'])
    if prot_id in sequence_dict.keys():
        antijen_tcell_df.at[e, 'full_sequence'] = sequence_dict[prot_id]

antijen_tcell_df.dropna(subset=['full_sequence'], inplace=True)
antijen_tcell_df['entry_source'] = "AntiJen_Tcell"
antijen_tcell_df['start_pos'] = None
antijen_tcell_df['end_pos'] = None

new_col_names = ['fragment', 'MHC_types', 'origin_species', 'category', 'UniProt_parent_id',
                 'ref_type', 'lit_reference', 'full_sequence', 'entry_source',
                 'start_pos', 'end_pos']

antijen_tcell_df.columns = new_col_names

# fix column names on antijen df...
antijen_tcell_df = antijen_tcell_df[['fragment', 'MHC_types', 'origin_species',
                                     'UniProt_parent_id',  'lit_reference',
                                     'full_sequence', 'entry_source',
                                     'start_pos', 'end_pos']]
antijen_tcell_df.to_csv(indir+"AntiJen_Tcell_w_sequences.csv", index=False)
