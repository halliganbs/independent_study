from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import argparse

import min_cnn
import koo_cnn
import chen_cnn

# TODO: Get real path
path = "parsed_data.csv"


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


# model_func is the desired nodel build_model function with hyperparamters changed
def train_model(model_func, loss, optimizer, filename):
  kf = load_data()

  # accuarcy, precision, recall
  model_acc = np.empty([5])
  model_prec = np.empty([5])
  model_recall = np.empty([5])
  index = 0
  model = None
  for train_index, test_index in kf.split(x, y):

    # create test and train sets
    x_train, x_test = x[train_index, :, :], x[test_index, :, :]
    y_train, y_test = y[train_index], y[test_index]

    # sanity check
    print('x_train shape:', x_train.shape)
    print('y_test shape:', x_test.shape)

    model = model_func
    if model is not None:
      keras.backend.clear_seesion()

    # compile
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

        # fit
    model.fit(x_train, y_train,
                batch_size=100,
                epochs=100)

    # predict
    pred_prob = model.predict(x_test).reshape(-1)
    print(pred_prob.shape, y_test.shape)
    pred_y = np.ones(y_test.shape)
    pred_y[pred_prob < 0.5] = 0.0

    # SKlearn acc, pred, recall
    acc = accuracy_score(y_test, pred_y)
    prec = precision_score(y_test, pred_y)
    recall = recall_score(y_test, pred_y)

    # sanity check
    print('Trial Accuracy: ', acc)
    print('Trial Precision: ', prec)
    print('Trial Recall: ', recall)
    print()

    # add to model values
    model_acc[index] = acc
    model_prec[index] = prec
    model_recall[index] =recall

    index = index+1
    model = None

  # STD
  acc_std = np.std(model_acc)
  prec_std = np.std(model_prec)
  recall_std = np.std(model_recall)

  # average of the models
  avg_model_acc = np.mean(model_acc)
  avg_model_prec = np.mean(model_prec)
  avg_model_recall = np.mean(model_recall)

  # add scores to filename
  f = open(filename, "a")
  f.write("Model Resutls")
  f.write('Accuarcy: ',avg_model_acc, 'std:', acc_std)
  f.write('Precision: ', avg_model_prec, 'std: ', prec_std)
  f.write('Recall: ', avg_mdoel_recall, 'std: ', recall_std)
  f.write('\n')

# TODO: fix bellow to work

parser = argparse.ArgumentParser(description='Process CNN and modify values')
parser.add_argument('integers', metavar='N', type=int,
nargs='+', help='an int for accumlator')


parser.add_argument('--sum', dest='accumlate', action='store_const',
const=sum, default=max,help='sum the int (default: find the max)')


args = parser.parse_args()
print(args.accumlate(args.integers))
