#!/usr/bin/env Python3
"""two-class_neural_net.py

Ellysia Li (lie@ohsu.edu)

Python 3.7

This script ...

This script requires that `numpy`, `keras` and `tensorflow` be installed within
the Python environment you are running this script in.

Inputs:
    The location of the npy files corresponding to the training and testing
    datasets

Outputs:
    none
"""
import time
import numpy as np
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn import preprocessing
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_curve, auc, matthews_corrcoef

start_time = time.time()

file_name = "txt/two-class_neural_net"
np.random.seed(6255)
set_random_seed(6255)
ratio = 0.1
class_weight = {0: 1, 1: 1}

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

test_X = total_X[0:int(len(total_X) * ratio)]
test_y = total_y[0:int(len(total_X) * ratio)]
train_X = total_X[int(len(total_X) * ratio):]
train_y = total_y[int(len(total_X) * ratio):]

model = Sequential()
model.add(Dense(25, input_dim=525, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=1,
          batch_size=1, class_weight=class_weight, shuffle=True)

# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()

def print_results(X, y):
    predict_y = model.predict(X)
    for i in range(len(predict_y)):
        if predict_y[i] >= 0.7:
            predict_y[i] = 1
        else:
            predict_y[i] = 0

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