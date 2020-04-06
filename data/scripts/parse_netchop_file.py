import pandas as pd
from sklearn import metrics

handle = "/Users/weeder/PycharmProjects/proteasome/data/validation_data/" \
         "netchop_immuno_digestion_validation_preds.txt"


with open(handle, 'r') as netchop_file:
        line = netchop_file.readline()
        table_start = False
        table_list = []
        while line:
            if line == ' pos  AA  C      score      Ident\n':
                table_start = True
                # handles break line right under table headers
                line = netchop_file.readline()
                line = netchop_file.readline()
                table_lines = []
            while table_start:
                if line == '--------------------------------------\n':
                    table_start = False
                    table_list.append(table_lines)
                    break
                else:
                    table_lines.append(line)
                    line = netchop_file.readline()
            line = netchop_file.readline()

# extract mid lines (point of cleavage/non-cleavage
cleave_val_lines = []
for table in table_list:
    mid_point = int(len(table)/2)
    cleave_val_lines.append(table[mid_point])


# parse val lines
result_df = pd.DataFrame(columns=['pos', 'aa', 'cleave_pred', 'prob',
                                  'true_class', 'ident'])
for line in cleave_val_lines:
    newline = line.strip()
    pos, aa, cleave, prob, ident = newline.split()
    if cleave == "S":
        cleave_pred = 1
    else:
        cleave_pred = 0

    if "pos" in ident:
        true_class = 1
    else:
        true_class = 0
    out_series = pd.Series([pos, aa, cleave_pred, prob, true_class, ident],
                           index=result_df.columns)
    result_df = result_df.append(out_series, ignore_index=True)


report = metrics.classification_report(result_df['true_class'].astype(int),
                                       result_df['cleave_pred'].astype(int))
epitope_auc = metrics.roc_auc_score(result_df['true_class'].astype(int),
                                    result_df['prob'].astype(float))
tn, fp, fn, tp = metrics.confusion_matrix(result_df['true_class'].astype(int),
                                          result_df['cleave_pred'].astype(int)).ravel()
sensitivity = tp/(tp + fn)
specificity = tn/(tn+fp)

print("Sensitivity: ", sensitivity)
print("Specificity: ", specificity)
print("AUC: ", epitope_auc)
