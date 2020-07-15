import pandas as pd
from extraction_functions import *

in_dir = "//data/raw_data"
out_dir = "//data/un-merged_data"
in_file = "/Winter_et_al_results.csv"
handle = in_dir + in_file
dat = pd.read_csv(handle, low_memory=False, header=0)


def parse_cleavage_logo(row):
    source = row[0]
    logo = row[1]
    start_pos = 0
    end_pos = None
    entries = []
    excluded_positions = ""
    for pos in range(len(logo)):
        if logo[pos] == "?":
            if len(excluded_positions) == 0:
                excluded_positions += str(pos)
            else:
                excluded_positions += (";" + str(pos))

    for pos in range(len(logo)):
        if logo[pos] == "C":
            end_pos = pos + 1
            entry = [source[start_pos:end_pos], start_pos, end_pos, source,
                     excluded_positions, "C"]
            entries.append(entry)
        if logo[pos] == "M":
            end_pos = pos + 1
            entry = [source[start_pos:end_pos], start_pos, end_pos, source,
                     excluded_positions, "M"]
            entries.append(entry)
        if logo[pos] == "I":
            end_pos = pos + 1
            entry = [source[start_pos:end_pos], start_pos, end_pos, source,
                     excluded_positions, "I"]
            entries.append(entry)
        else:
            pass
    return entries


col_names = ['fragment', 'start_pos', 'end_pos', 'full_sequence',
             'exclusions', 'Proteasome']

out_df = pd.DataFrame(columns=col_names)

for i in range(len(dat)):
    row = dat.iloc[i]
    fragment_entries = parse_cleavage_logo(row)
    for fragment in fragment_entries:
        new_entry = pd.Series(fragment, index=col_names)
        out_df = out_df.append(new_entry, ignore_index=True)


out_df['Name'] = "NA"
out_df['Organism'] = "human"
out_df['Subunit'] = "20S"
out_df['entry_source'] = "cleavage_map"
out_df['DOI'] = "10.7554/eLife.27364"

out_df.to_csv(out_dir + "/winter_et_al_cleavage_fragments.csv", index=False)
