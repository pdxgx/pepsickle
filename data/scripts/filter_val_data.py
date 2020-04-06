#!usr/bin/env python3
import pickle

epitope_val_data = "/Users/weeder/PycharmProjects/proteasome/data/" \
                   "validation_data/epitope_val_windows_13aa_paired.pickle"
digestion_val_data = "/Users/weeder/PycharmProjects/proteasome/data/" \
                     "validation_data/" \
                     "digestion_val_windows_13aa.pickle"

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


digestion_constit_positive_windows = dict()
digestion_immuno_positive_windows = dict()
for key in digestion_val_dict['proteasome']['positives'].keys():
    if key not in all_mammal_training_dict['proteasome']['positives'].keys():
        if "C" in list(digestion_val_dict['proteasome']['positives'][key])[0]:
            digestion_constit_positive_windows[key] = digestion_val_dict['proteasome']['positives'][key].copy()

        if "I" in list(digestion_val_dict['proteasome']['positives'][key])[0]:
            digestion_immuno_positive_windows[key] = digestion_val_dict['proteasome']['positives'][key].copy()


digestion_constit_negative_windows = dict()
digestion_immuno_negative_windows = dict()
for key in digestion_val_dict['proteasome']['negatives'].keys():
    if key not in all_mammal_training_dict['proteasome']['positives'].keys():
        if key not in all_mammal_training_dict['proteasome']['negatives'].keys():
            if "C" in list(digestion_val_dict['proteasome']['negatives'][key])[0]:
                digestion_constit_negative_windows[key] = digestion_val_dict['proteasome']['negatives'][key].copy()
            if "I" in list(digestion_val_dict['proteasome']['negatives'][key])[0]:
                digestion_immuno_negative_windows[key] = digestion_val_dict['proteasome']['negatives'][key].copy()


digestion_constit_windows_filtered = dict()
digestion_constit_windows_filtered['positives'] = digestion_constit_positive_windows
digestion_constit_windows_filtered['negatives'] = digestion_constit_negative_windows

digestion_immuno_windows_filtered = dict()
digestion_immuno_windows_filtered['positives'] = digestion_immuno_positive_windows
digestion_immuno_windows_filtered['negatives'] = digestion_immuno_negative_windows


print("Epitope positives: ", len(epitope_positive_windows))
print("Epitope negatives: ", len(epitope_negative_windows))

print("Digestion constitutive positives: ", len(digestion_constit_positive_windows))
print("Digestion constitutive negatives: ", len(digestion_constit_negative_windows))

print("Digestion immuno positives: ", len(digestion_immuno_positive_windows))
print("Digestion immuno negatives: ", len(digestion_immuno_negative_windows))

pickle.dump(epitope_windows_filtered, open("/Users/weeder/PycharmProjects/"
                                           "proteasome/data/validation_data/"
                                           "epitope_val_filtered.pickle", "wb"))

pickle.dump(digestion_constit_windows_filtered, open("/Users/weeder/PycharmProjects/"
                                             "proteasome/data/validation_data/"
                                             "digestion_constitutive_validation_filtered.pickle", "wb"))

pickle.dump(digestion_constit_windows_filtered, open("/Users/weeder/PycharmProjects/"
                                             "proteasome/data/validation_data/"
                                             "digestion_immuno_validation_filtered.pickle", "wb"))

# make this a function?
epitope_val_handle = "/Users/weeder/PycharmProjects/proteasome/data/" \
                    "validation_data/epitope_val_data.fasta"
epitope_val_fasta = open(epitope_val_handle, "w")

for i in range(len(epitope_positive_windows)):
    prot_name = ">pos_" + str(i)
    epitope_val_fasta.write(prot_name)
    epitope_val_fasta.write("\n")
    epitope_val_fasta.write(list(epitope_positive_windows.keys())[i])
    epitope_val_fasta.write("\n")

for i in range(len(epitope_negative_windows)):
    prot_name = ">neg_" + str(i)
    epitope_val_fasta.write(prot_name)
    epitope_val_fasta.write("\n")
    epitope_val_fasta.write(list(epitope_negative_windows.keys())[i])
    epitope_val_fasta.write("\n")

epitope_val_fasta.close()


digestion_constitutive_val_handle = "/Users/weeder/PycharmProjects/proteasome/data/" \
                    "validation_data/digestion_constitutive_validation_data.fasta"
digestion_constit_val_fasta = open(digestion_constitutive_val_handle, "w")

for i in range(len(digestion_constit_positive_windows)):
    prot_name = ">pos_" + str(i)
    digestion_constit_val_fasta.write(prot_name)
    digestion_constit_val_fasta.write("\n")
    digestion_constit_val_fasta.write(list(digestion_constit_positive_windows.keys())[i])
    digestion_constit_val_fasta.write("\n")

for i in range(len(digestion_constit_negative_windows)):
    prot_name = ">neg_" + str(i)
    digestion_constit_val_fasta.write(prot_name)
    digestion_constit_val_fasta.write("\n")
    digestion_constit_val_fasta.write(list(digestion_constit_negative_windows.keys())[i])
    digestion_constit_val_fasta.write("\n")

digestion_constit_val_fasta.close()


digestion_immuno_val_handle = "/Users/weeder/PycharmProjects/proteasome/data/" \
                    "validation_data/digestion_immuno_validation_data.fasta"
digestion_immuno_val_fasta = open(digestion_immuno_val_handle, "w")

for i in range(len(digestion_immuno_positive_windows)):
    prot_name = ">pos_" + str(i)
    digestion_immuno_val_fasta.write(prot_name)
    digestion_immuno_val_fasta.write("\n")
    digestion_immuno_val_fasta.write(list(digestion_immuno_positive_windows.keys())[i])
    digestion_immuno_val_fasta.write("\n")

for i in range(len(digestion_immuno_negative_windows)):
    prot_name = ">neg_" + str(i)
    digestion_immuno_val_fasta.write(prot_name)
    digestion_immuno_val_fasta.write("\n")
    digestion_immuno_val_fasta.write(list(digestion_immuno_negative_windows.keys())[i])
    digestion_immuno_val_fasta.write("\n")

digestion_immuno_val_fasta.close()
