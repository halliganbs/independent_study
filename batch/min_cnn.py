# MIN CNN

from __future__ import print_function

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras import optimizers
import pandas as pd
import seaborn as sns
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score

path = "/home/halliganbs/parsed_data.csv"


# One Hot Encodes the dataframe
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

def load_data():
    df = pd.read_csv(path)
    x = one_hot_encode(df)

    # sanity check
    print(x.shape)
    print(x[0, :, :])

    y = df['enhancer'].to_numpy()

    # might make n_splits = a variable so that when adding more I can just call it
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    return kf

# Function to optimize based on greedy algorithm
def build_model():
    # Create model

    model = Sequential()

    # Layer 1
    # convo 128 x 4 x 1 x 8
    # BatchNorm
    model.add(Conv1D(filters=128,kernel_size=8,strides=1))
    model.add(BatchNormalization())

    # Layer 2
    # convo 128x128x1x8
    # BatchNorm
    # max pool 1x2
    model.add(Conv1D(filters=128,kernel_size=8,strides=1))
    model.add(BatchNormalization())
    model.add(MaxPooling1D()) # pool_size=2, strides=1

    # Layer 3
    # convo 64x128x1x3
    # BatchNorm
    model.add(Conv1D(filters=64,kernel_size=3,strides=1))
    model.add(BatchNormalization())

    # Layer 4
    # convo 64x64x1x3
    # batch normalization
    # max pool 1x2
    model.add(Conv1D(filters=64, kernel_size=3, strides=1))
    model.add(BatchNormalization())
    model.add(MaxPooling1D()) # pool_size=2

    model.add(Flatten())

    # Output Layer
    # Dense 256
    # Droput some 0.5
    # Dense 128
    # Softmax 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
#     model.add(Activation('softmax'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model
