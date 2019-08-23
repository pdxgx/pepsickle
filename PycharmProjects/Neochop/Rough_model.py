#!/usr/bin/env Python3
"""rough_model.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script trains a tentative model for the Neochop algorithm.

This script requires that `numpy`, `keras`, `tensorflow` and `sklearn` be
installed within the Python environment you are running this script in.

Inputs:
    The location the two npy files containing the positive and negative information

Outputs:
    n/a (reports the results of the model)
"""


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import set_random_seed
from sklearn import preprocessing
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_curve, auc, matthews_corrcoef

set_random_seed(10)
np.random.seed(10)
ratio = 0.1

X_pos = np.load("converged_filtered_positives.npy")
X_neg = np.load("converged_filtered_negatives.npy")

total_X = np.append(X_pos, X_neg, axis=0)
total_y = np.append(np.ones(len(X_pos)), np.zeros(len(X_neg)), axis=0)
n_samples, nx, ny = total_X.shape
total_X = preprocessing.scale(total_X.reshape((n_samples, nx*ny)))

indices = np.arange(total_X.shape[0])
np.random.shuffle(indices)
total_X = total_X[indices]
total_y = total_y[indices]

test_X = total_X[0:int(len(total_X) * ratio)]
test_y = total_y[0:int(len(total_X) * ratio)]
train_X = total_X[int(len(total_X) * ratio):]
train_y = total_y[int(len(total_X) * ratio):]

model = Sequential()
model.add(Dense(22, input_dim=697, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=1, batch_size=1, shuffle=True)

_, accuracy = model.evaluate(total_X, total_y)
print('Accuracy: %.2f' % (accuracy*100))
print()


def print_results(X, y):
    """Prints the metrics of the model in respects to the training and test
       sets

       Outputs the following information:
       f1 score: 2 * (precision * recall) / (precision + recall)
       AUC: the area under the ROC curve
       MCC: Matthews correlation coefficient
       Confusion Matrix: A 2D array which depicts the True Negatives, False
                         Negatives, False Positives, and True Positives
                         [[TN   FN]
                           FP   TP]]
       Classification Report: a report formatted in a tabular-like structure
                              which depicts the precision, recall, f1-score,
                              and support of the two classes, as well as
                              accuracy and macro/weighted average

       Arguments:
           X (numpy): the input data the trained model predicts
           y (numpy): the expected output for the input data
       Returns:
           None
    """
    predict_y = model.predict(X)
    for i in range(len(predict_y)):
        if predict_y[i] >= 0.7:
            predict_y[i] = 1
        else:
            predict_y[i] = 0

    print("f1 score: " + str(f1_score(y, predict_y)))

    fpr_gb, tpr_gb, _ = roc_curve(y, predict_y)
    print("AUC: " + str(auc(fpr_gb, tpr_gb)))

    print("MCC: " + str(matthews_corrcoef(y, predict_y)))

    print(confusion_matrix(y, predict_y))

    print("Classification Report:")
    print(classification_report(y, predict_y))


print("Training Set:")
print_results(train_X, train_y)
print()

print("Test Set:")
print_results(test_X, test_y)
print()
