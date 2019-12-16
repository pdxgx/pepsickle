import pandas as pd
import extraction_functions

indir = "/Users/weeder/PycharmProjects/proteasome/data_processing/un-merged_data/positives/"
SYF_df = pd.read_csv(indir + "tmp_SYFPEITHI_epitopes.csv", low_memory=False)
SYF_df.dropna(subset=['UniProt_id'], inplace=True)
SYF_df.index = range(len(SYF_df)) # only necessary if na's aren't already removed in prev script


uniprot_ids = list(SYF_df['UniProt_id'])
parsed_ids = []

for i in uniprot_ids:
    if ";" in i:
        tmp = i.split(";")
        if t[0] not in parsed_ids:
            parsed_ids.append(t)
    else:
        if i not in parsed_ids:
            parsed_ids.append(i)


sequence_dict = {}
error_index = []

progress = 0

for entry in parsed_ids:
    try:
        sequence_dict[entry] = retrieve_UniProt_seq(entry)
    except:
        error_index.append(progress)
    progress +=1
    if progress % 100 == 0:
        print(round(progress/len(parsed_ids), 3)*100, "% done")


SYF_df['full_sequence'] = None
for e in range(len(SYF_df)):
    tmp_entry = str(SYF_df.at[e, 'UniProt_id'])
    if ";" in tmp_entry:
        id_list = tmp_entry.split(";")
    else:
        prot_id = str(SYF_df.at[e, 'UniProt_id'])
    if prot_id in sequence_dict.keys():
        SYF_df.at[e, 'full_sequence'] = sequence_dict[prot_id]

SYF_df.dropna(subset=['full_sequence'], inplace=True) # none should drop
SYF_df['Origin'] = "SYFPEITHI_database"

SYF_df.to_csv(indir+"SYF_data_w_sequences.csv", index=False)
