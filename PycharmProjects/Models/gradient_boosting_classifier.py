#!/usr/bin/env Python3
"""random_forest_classifier.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script ...

This script requires that `numpy` and `scikit-learn` be installed within the
Python environment you are running this script in.

Inputs:
    The location of the npy files corresponding to the training and testing
    datasets

Outputs:
    none
"""

import numpy as np
import time
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_curve, auc, matthews_corrcoef
start_time = time.time()

np.random.seed(6255)
class_weight = {0: 1, 1: 1}
ratio = 0.1
file_name = "txt/gradient_boosting.txt"

X_pos = np.load("npy/converged_filtered_positives_2d.npy")
X_neg = np.append(np.load("npy/converged_filtered_negatives_2d.npy"),
                  np.load("npy/digestion_negatives_2d.npy"), axis=0)
total_X = preprocessing.scale(np.append(X_pos, X_neg, axis=0))
total_y = np.append(np.ones(len(X_pos)), np.zeros(len(X_neg)), axis=0)

non_cleavage_X = np.append(total_X[len(total_X) - len(X_neg):], total_X[:10],
                           axis=0)

indices = np.arange(total_X.shape[0])
np.random.shuffle(indices)
total_X = total_X[indices]
total_y = total_y[indices]

test_X = total_X[0:int(len(total_X)*ratio)]
test_y = total_y[0:int(len(total_X)*ratio)]
train_X = total_X[int(len(total_X)*ratio):]
train_y = total_y[int(len(total_X)*ratio):]

clf = GradientBoostingClassifier()
clf = clf.fit(train_X, train_y)


def print_results(X, y):
    predict_y = clf.predict(X)
    print("f1 score: " + str(f1_score(y, predict_y)))
    f.write("f1 score: " + str(f1_score(y, predict_y)) + "\n")

    fpr_gb, tpr_gb, _ = roc_curve(y, predict_y)
    print("AUC: " + str(auc(fpr_gb, tpr_gb)))
    f.write("AUC: " + str(auc(fpr_gb, tpr_gb)) + "\n")

    print("MCC: " + str(matthews_corrcoef(y, predict_y)))
    f.write("MCC: " + str(matthews_corrcoef(y, predict_y)) + "\n")

    print(confusion_matrix(y, predict_y))
    f.write(np.array2string(confusion_matrix(y, predict_y)) + "\n")

    print("Classification Report:")
    print(classification_report(y, predict_y))
    f.write(classification_report(y, predict_y) + "\n\n")


with open(file_name, "w") as f:
    print("Training Set:")
    f.write("Training Set:\n")
    print_results(train_X, train_y)
    print()

    print("Test Set:")
    f.write("Test Set:\n")
    print_results(test_X, test_y)
    print()

    print("Non-cleavage Set:")
    f.write("Non-cleavage Set:\n")
    print_results(non_cleavage_X, np.append(np.zeros(len(X_neg)), np.ones(10),
                                            axis=0))

print("--- %s minutes ---" % ((time.time() - start_time) / 60))
