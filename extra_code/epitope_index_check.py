import re
import pandas as pd
# loop for checking all index cases and repairing indices when possible
# replace df with the full df of positive epitope examples
file_dir = "/Users/weeder/PycharmProjects/proteasome/data_processing/" \
           "un-merged_data/"

df = pd.read_csv(file_dir + 'tmp_merged.csv')
df = df.where(pd.notnull(df), None)
df = df[df.full_sequence.notnull()]
df = df.reset_index(drop=True)

failed_indices = []
for i in range(len(df)):
    # pull out a single row
    row = df.iloc[i]
    # convert the epitope into a regular expression for seq. search
    search_exp = create_sequence_regex(row['fragment'])
    # if there's a starting and ending position...
    if row['start_pos'] and row['end_pos']:
        # get start & stop, these may need shifted depending on ref. indexing
        start = int(row['start_pos']) - 1  # if base 0
        end = int(row['end_pos'])
        # pull out expected match from source
        source_seq = str(row['full_sequence'][start:end])
        # if the sequences match, go to next row iteration
        if re.match(search_exp, source_seq):
            df.at[i, 'start_pos'] = start
            continue
        # if doesn't match, but only occurs once in origin sequence...
        elif len(re.findall(search_exp, str(row['full_sequence']))) == 1:
            # find location
            s = re.search(search_exp, str(row['full_sequence']))
            # update start position with start of match
            df.at[i, 'start_pos'] = s.span()[0]
            # update end position with start of match
            df.at[i, 'end_pos'] = s.span()[1]
        # if 0 or >1 matches, fail
        else:
            failed_indices.append(i)
    # either start or end is missing but not both
    elif row['start_pos'] or row['end_pos']:
        # if start position is present
        if row['start_pos']:
            start = int(row['start_pos']) - 1  # if 0 base
            end = int(row['start_pos']) + \
                  len(row['fragment'])
        # if end position is present
        if row['end_pos']:
            end = int(row['end_pos'])
            start = end - len(row['fragment']) - 1  # if 0 base
        # with newly defined positions, repeat code block from above
        source_seq = str(row['fragment'][start:end])
        # if the sequences match, go to next row iteration
        if re.match(search_exp, source_seq):
            df.at[i, 'start_pos'] = start
            df.at[i, 'end_pos'] = end
            continue
        # if doesn't match, but only occurs once in origin sequence...
        elif len(re.findall(search_exp, str(row['full_sequence']))) == 1:
            # find location
            s = re.search(search_exp, str(row['full_sequence']))
            # update start position with start of match
            df.at[i, 'start_pos'] = s.span()[0]
            # update end position with start of match
            df.at[i, 'end_pos'] = s.span()[1]
        # if 0 or >1 matches, fail
        else:
            failed_indices.append(i)
    # if no start and no end position
    else:
        # if only one occurrence...
        if len(re.findall(search_exp, str(row['full_sequence']))) == 1:
            # find location
            s = re.search(search_exp, str(row['full_sequence']))
            # update start position with start of match
            df.at[i, 'start_pos'] = s.span()[0]
            # update end position with start of match
            df.at[i, 'end_pos'] = s.span()[1]
        # if 0 or >1 matches, fail
        else:
            failed_indices.append(i)

len(failed_indices)
df.drop(index=failed_indices, inplace=True)
df = df.reset_index(drop=True)

# double check...
mismatch_indices = []
for i in range(len(df)):
    row = df.iloc[i]
    start = int(row['start_pos'])
    end = int(row['end_pos'])
    source_seq = str(row['full_sequence'][start:end])
    if str(row['fragment']) == source_seq:
        pass
    else:
        mismatch_indices.append(i)

# for now, drop few with errors:
df.drop(index=mismatch_indices, inplace=True)
df = df.reset_index(drop=True)

df.to_csv(file_dir+"tmp_data_indices_repaired.csv", index=False)
