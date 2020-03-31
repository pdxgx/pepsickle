#!usr/bin/env python3
import pickle

epitope_val_data = "/Users/weeder/PycharmProjects/proteasome/data/" \
                   "validation_data/epitope_val_windows_13aa_paired.pickle"
digestion_val_data = "/Users/weeder/PycharmProjects/proteasome/data/" \
                   "validation_data/digestion_val_windows_13aa_paired.pickle"

human_only_training_data = "/Users/weeder/PycharmProjects/proteasome/data/" \
                           "generated_training_sets/" \
                           "cleavage_windows_human_only_13aa.pickle"
all_mammal_training_data = "/Users/weeder/PycharmProjects/proteasome/data/" \
                           "generated_training_sets/" \
                           "cleavage_windows_all_mammal_13aa.pickle"


epitope_val_dict = pickle.load(open(epitope_val_data, "rb"))
digestion_val_dict = pickle.load(open(digestion_val_data, "rb"))
all_mammal_training_dict = pickle.load(open(all_mammal_training_data, "rb"))

epitope_positive_windows = dict()
for key in epitope_val_dict['epitope']['positives'].keys():
    if key not in all_mammal_training_dict['epitope']['positives'].keys():
        epitope_positive_windows[key] = epitope_val_dict['epitope']['positives'][key].copy()

epitope_negative_windows = dict()
for key in epitope_val_dict['epitope']['negatives'].keys():
    if key not in all_mammal_training_dict['epitope']['positives'].keys():
        if key not in all_mammal_training_dict['epitope']['negatives'].keys():
            if key not in all_mammal_training_dict['epitope']['unknowns'].keys():
                epitope_negative_windows[key] = epitope_val_dict['epitope']['negatives'][key].copy()


epitope_windows_filtered = dict()
epitope_windows_filtered['positives'] = epitope_positive_windows
epitope_windows_filtered['negatives'] = epitope_negative_windows


digestion_positive_windows = dict()
for key in digestion_val_dict['proteasome']['positives'].keys():
    if key not in all_mammal_training_dict['proteasome']['positives'].keys():
        digestion_positive_windows[key] = digestion_val_dict['proteasome']['positives'][key].copy()

digestion_negative_windows = dict()
for key in digestion_val_dict['proteasome']['negatives'].keys():
    if key not in all_mammal_training_dict['proteasome']['positives'].keys():
        if key not in all_mammal_training_dict['proteasome']['negatives'].keys():
            digestion_negative_windows[key] = digestion_val_dict['proteasome']['negatives'][key].copy()

digestion_windows_filtered = dict()
digestion_windows_filtered['positives'] = digestion_positive_windows
digestion_windows_filtered['negatives'] = digestion_negative_windows

print("Epitope Positives: ", len(epitope_positive_windows))
print("Epitope Negatives: ", len(epitope_negative_windows))
print("Digestion Positives: ", len(digestion_positive_windows))
print("Digestion negatives: ", len(digestion_negative_windows))

pickle.dump(epitope_windows_filtered, open("/Users/weeder/PycharmProjects/"
                                           "proteasome/data/validation_data/"
                                           "epitope_val_filtered.pickle", "wb"))
pickle.dump(digestion_windows_filtered, open("/Users/weeder/PycharmProjects/"
                                             "proteasome/data/validation_data/"
                                             "digestion_val_filtered.pickle", "wb"))

epitope_val_handle = "/Users/weeder/PycharmProjects/proteasome/data/" \
                    "validation_data/epitope_val_data.fasta"
epitope_val_fasta = open(epitope_val_handle, "w")

for i in range(len(epitope_positive_windows)):
    prot_name = ">pos_" + str(i)
    epitope_val_fasta.write(prot_name)
    epitope_val_fasta.write(epitope_positive_windows[i])

for i in range(len(epitope_negative_windows)):
    prot_name = ">neg_" + str(i)
    epitope_val_fasta.write(prot_name)
    epitope_val_fasta.write(epitope_negative_windows[i])

epitope_val_fasta.close()
