from __future__ import print_function


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

# Load Data
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



# build model based on paper
# Default values from the Koo Paper
# varaible name Layer #
def build_model(filter1=30, kernel_size1=19, strides1=1, drop1=0.1, pool1 = MaxPooling1D(pool_size=2, strides=2),
                filter2=128, kernel_size2=5, strides2=1, drop2=0.1 pool2 = MaxPooling1D(pool_size=50, stides=2), hidden_drop =0.5):

    hidden_dims = 250 # needs verification

    model = Sequential()
    '''
    1st Connvolutation layer
    30 filters
    size of filter 19
    stride 1
    '''
    model.add(Conv1D(
        filters=filter1
        kernel_size=kernel_size1,
        strides=strides1,
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(1e-6)
    ))

    # Dropout Layer 1
    model.add(Dropout(drop1))

    # Max Pooling layer 1
    model.add(pool1)

    '''
    2nd Connvoluation layer
    128 filters
    filter size 5
    stride 1
    '''
    model.add(Conv1D(
        filters=filter2,
        kernel_size=kernel_size2,
        strides=strides2,
        activation='relu',
        padding='same',
        kernel_regularizer=regularizers.l2(1e-6)
    ))

    # Dropout Layer 2
    model.add(Dropout(drop2))

    # Max Pooling layer 2
    model.add(pool2)

    model.add(Flatten())

    # Hidden Layer
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(hidden_drop))

    # Output Layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model
