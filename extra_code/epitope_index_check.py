import re
# loop for checking all index cases and repairing indices when possible
# replace df with the full df of positive epitope examples
failed_indices = []
for i in range(len(df)):
    # pull out a single row
    row = df.iloc[i]
    # convert the epitope into a regular expression for seq. search
    search_exp = create_sequence_regex(row['linear_peptide_seq'])
    # if there's a starting and ending position...
    if row['starting_position'] and row['ending_position']:
        # get start & stop, these may need shifted depending on ref. indexing
        start = int(row['starting_position']) - 1  # if base 0
        end = int(row['ending_position'])
        # pull out expected match from source
        source_seq = str(row['sequence'][start:end])
        # if the sequences match, go to next row iteration
        if re.match(search_exp, source_seq):
            continue
        # if doesn't match, but only occurs once in origin sequence...
        elif len(re.findall(search_exp, str(row['sequence']))) == 1:
            # find location
            s = re.search(search_exp, str(row['sequence']))
            # update start position with start of match
            df.iloc[i]['starting_position'] = s.span()[0]
            # update end position with start of match
            df.iloc[i]['ending_position'] = s.span()[1]
        # if 0 or >1 matches, fail
        else:
            failed_indices.append(i)
    # either start or end is missing but not both
    elif row['starting_position'] or row['ending_position']:
        # if start position is present
        if row['starting_position']:
            start = int(row['starting_position']) - 1  # if 0 base
            end = int(row['starting_position']) + \
                  len(row['linear_peptide_seq'])
        # if end position is present
        if row['ending_position']:
            end = int(row['ending_position'])
            start = end - len(row['linear_peptide_seq']) - 1  # if 0 base
        # with newly defined positions, repeat code block from above
        source_seq = str(row['sequence'][start:end])
        # if the sequences match, go to next row iteration
        if re.match(search_exp, source_seq):
            continue
        # if doesn't match, but only occurs once in origin sequence...
        elif len(re.findall(search_exp, str(row['sequence']))) == 1:
            # find location
            s = re.search(search_exp, str(row['sequence']))
            # update start position with start of match
            df.iloc[i]['starting_position'] = s.span()[0]
            # update end position with start of match
            df.iloc[i]['ending_position'] = s.span()[1]
        # if 0 or >1 matches, fail
        else:
            failed_indices.append(i)
    # if no start and no end position
    else:
        # if only one occurrence...
        if len(re.findall(search_exp, str(row['sequence']))) == 1:
            # find location
            s = re.search(search_exp, str(row['sequence']))
            # update start position with start of match
            df.iloc[i]['starting_position'] = s.span()[0]
            # update end position with start of match
            df.iloc[i]['ending_position'] = s.span()[1]
        # if 0 or >1 matches, fail
        else:
            failed_indices.append(i)
