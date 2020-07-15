import pandas as pd

in_file = "//data/validation_data/" \
          "digestion_data/compiled_digestion_df.csv"

out_file = "//data/validation_data/" \
           "digestion_data/digestion_val_data_columns_remapped.csv"
digestion_val = pd.read_csv(in_file)

new_digestion_cols = ['lit_reference', 'protein_name', 'origin_species',
                      'Proteasome', 'Subunit', 'full_seq_accession', 'end_pos',
                      'fragment', 'full_sequence', 'start_pos', 'entry_source']

digestion_val.columns = new_digestion_cols
digestion_val['exclusions'] = None

digestion_val.to_csv(out_file, index=False)
