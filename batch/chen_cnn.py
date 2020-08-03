from __future__ import print_function

import argparse

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score

path = "/home/halliganbs/parsed_data.csv"

# one hot encodes the dataframe
# creates an np array (number_samples, length_sample, number_values)
def one_hot_encode(df):
    n_samples = len(df['window'])
    X = np.zeros((n_samples, 501, 4))
    for i, seq in enumerate(df['window']):
        for j, nucleotide in enumerate(seq):
        # Possible Nucleotides: A, C, G, T
            if nucleotide == 'A':
                X[i,j,0] = 1
            elif nucleotide == 'T':
                X[i,j,1] = 1
            elif nucleotide == 'C':
                X[i,j,2] = 1
            elif nucleotide == 'G':
                X[i, j, 3] = 1
    return X

def load_data(path):
    # load data
    df = pd.read_csv(path)
    x = one_hot_encode(df)

    # sanity check
    print(x.shape)
    print(x[0, :, :])

    y = df['enhancer'].to_numpy()

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)



# Graph Results
def get_scores(scores):
    acc = []
    precision =[]
    recall = []
    for scr in scores:
        acc.append(scr[0])
        precision.append(scr[1])
        recall.append(scr[2])

    return acc, precision, recall

def graph_scores(acc, prec, recall, values, type="N/A"):
    print(type, 'Values:',values)
    all_data = [np.asarray(acc), np.asarray(prec), np.asarray(recall)]
    labels = np.array(['Accuracy', 'Precision', 'Recall'])
    sns.lineplot(hue=labels, data=all_data)
    plt.legend(labels)




def build_model():
    model=Sequential()

    model.add(Conv1D(
        filters=256,
        kernel_size=16,
        strides=1,
        activation='relu'))

    model.add(MaxPooling1D(
        pool_size=4,
        strides=4))

    model.add(Conv1D(
        filters=128,
        kernel_size=8,
        strides=1,
        activation='relu'))

    model.add(MaxPooling1D(
        pool_size=4,
        strides=4))

    model.add(Conv1D(
        filters=128,
        kernel_size=16,
        strides=1,
        activation='relu'))

    model.add(Dense(32))

    model.add(Flatten())

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model
