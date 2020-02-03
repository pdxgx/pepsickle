import re
import os
import pandas as pd

in_dir = "C:/Users/bweed/Desktop/digestion_map_files"
os.chdir(in_dir)


def parse_header(line):
    line = line.strip("#")
    key, entry = line.split("=")
    return key.strip(), entry.strip()


def parse_cleavage_map(lines):
    line_iter = iter(lines)
    fragment_dict = {}
    source_seq = None
    fragment_list = []
    line = next(line_iter, None)
    while line:
        if line == " " or line == "\n":
            line = next(line_iter, None)
            continue
        if "#" not in line:
            if ">" in line:
                if source_seq is not None:
                    fragment_dict[source_seq] = fragment_list
                source_seq = next(line_iter).strip()
                end = next(line_iter)
                fragment_list = []
            if ">" not in line:
                if "@" not in line:
                    fragment = line.strip()
                    if len(re.findall(fragment, source_seq)) == 0:
                        print("Warning: query string not found in source sequence")
                        print("Query: ", fragment)
                        print("Source: ", source_seq)
                        line = next(line_iter, None)
                        continue
                    if len(re.findall(fragment, source_seq)) == 1:
                        start_pos = re.search(fragment, source_seq).span()[0]
                    else:
                        print(fragment, "Occurs multiple times in: ", source_seq)
                        print("exact position not annotated with @")
                        raise NotImplementedError
                    fragment_list.append((fragment, start_pos))
                if "@" in line:
                    fragment, start_pos = line.split("@")
                    fragment = fragment.strip()
                    start_pos = int(start_pos.strip())
                    assert(fragment == source_seq[start_pos:start_pos+len(fragment)]), \
                        "Warning: fragment does not match annotated position in source sequence"
                    fragment_list.append((fragment, start_pos))
        line = next(line_iter, None)
    fragment_dict[source_seq] = fragment_list
    return fragment_dict


def generate_pandas_df(meta_dict, seq_dict):
    fragment = []
    start_pos = []
    source_seq = []
    for source in seq_dict.keys():
        for entry in seq_dict[source]:
            fragment.append(entry[0])
            start_pos.append(entry[1])
            source_seq.append(source)
    out_df = pd.DataFrame(zip(fragment, start_pos, source_seq), columns=['Fragment', 'start_pos', "source_seq"])

    for meta_key in meta_dict:
        out_df[meta_key] = meta_dict[meta_key]

    return out_df


def parse_digestion_file(file):
    with open(file, "r") as f:
        meta_dict = {}
        lines = f.readlines()
        for i in range(len(lines)):
            if "#" in lines[i]:
                key, entry = parse_header(lines[i])
                meta_dict[key] = entry
            if "#" not in lines[i]:
                seq_lines = lines[i:]
                break
    if meta_dict['UniProt'] != 'NA':
        # raise NotImplementedError
        seq_dict = parse_cleavage_map(seq_lines)
    else:
        seq_dict = parse_cleavage_map(seq_lines)

    out_df = generate_pandas_df(meta_dict, seq_dict)
    return out_df


file_list = os.listdir()
digestion_df = pd.DataFrame(columns=['Fragment', 'start_pos', 'source_seq', 'Name', 'DOI', 'Subunit', 'Proteasome',
                                     'Organism', 'UniProt'])
for file in file_list:
    print("parsing: ", file)
    tmp_df = parse_digestion_file(file)
    digestion_df = digestion_df.append(tmp_df)

digestion_df.to_csv("../compiled_digestion_df.csv", index=False)