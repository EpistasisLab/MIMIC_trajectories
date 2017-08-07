from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from keras.preprocessing import sequence
from sklearn.cross_validation import StratifiedKFold
import pickle as pkl
import argparse
import h5py
import tensorflow as tf
import os

batch_size = 256

def lstm_fit_predict(train_X, train_y, test_X, test_y, roc_auc):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, stateful=False,
                   input_shape=(train_X.shape[1], train_X.shape[2]),
                   recurrent_dropout=0.4, dropout=0.5))
    model.add(LSTM(50, return_sequences=True, dropout=0.5, recurrent_dropout=0.3,
                   stateful=False))
    model.add(LSTM(50, return_sequences=False, dropout=0.5, recurrent_dropout=0.3,
                   stateful=False))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()

    case_proportion = (1-(sum(train_y)/float(len(train_y))))

    class_weights = {0: 1-case_proportion,
                     1: case_proportion}
    print(class_weights)
    print(sum(train_y), len(train_y), float(sum(y))/len(y))

    early_stopping = EarlyStopping(monitor='loss', patience=3)
    model.fit(train_X, train_y, validation_data=(test_X, test_y),
              epochs=100, batch_size=batch_size, callbacks=[early_stopping],
              class_weight=class_weights)
    lstm_pred = model.predict(test_X)

    return roc_auc_score(test_y, lstm_pred, average='macro', sample_weight=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_length", type=int, default=30)
    args = parser.parse_args()

    encounters = pd.read_hdf('/data/MIMIC/encounter_vectors_processed.h5','encounters')

    sequence_length = args.sequence_length
    encounters['6MONTH'] = 0
    encounters['1YEAR'] = 0
    encounters.loc[(encounters['SURVIVAL'] > 0) & (encounters['SURVIVAL'] < 183), '6MONTH'] = 1
    encounters.loc[(encounters['SURVIVAL'] > 0) & (encounters['SURVIVAL'] < 366), '1YEAR'] = 1

    if not os.path.isfile('/data/MIMIC/Xy_seq' + str(sequence_length) +'.h5'):
        print('generate arrays')
        j = 0
        encounter_list = []
        y_list = []
        print('go through admissions')
        for unique in encounters['HADM_ID'].unique():
            adm = encounters[encounters['HADM_ID'] == unique].copy()

            if adm.shape[1] > 0:
                y_list.append(adm['1YEAR'].head(1).values[0])
                adm.drop(['SUBJECT_ID', 'ENCOUNTER_ID', '6MONTH', 'SURVIVAL', '1YEAR',
                          'HADM_ID'], axis=1, inplace=True)

                encounter_list.append(adm.values.tolist())
                j += 1

        print('arrays settled')
        encounter_array = np.asarray(encounter_list)
        X = sequence.pad_sequences(encounter_array, maxlen=sequence_length,
                                   padding='post', truncating='post')
        y = np.array(y_list)


        h5f = h5py.File('/data/MIMIC/Xy_seq' + str(sequence_length) +'.h5', 'w')
        h5f.create_dataset('X', data=X)
        h5f.create_dataset('y', data=y)
        h5f.close()

    else:
        h5f = h5py.File('/data/MIMIC/Xy_seq' + str(sequence_length) +'.h5', 'r')
        X = h5f['X'][:]
        y = h5f['y'][:]

    print('Train model')
    cv = StratifiedKFold(y, n_folds=5, random_state=123)
    roc_auc = {'lstm':[]}

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        for j, (train, test) in enumerate(cv):
            roc_auc['lstm'].append(lstm_fit_predict(X[train], y[train],
                                                    X[test], y[test],
                                                    roc_auc))
            print('Cross fold: ', j, roc_auc)
        pkl.dump(roc_auc, open('/data/MIMIC/lstm_encounter_scores_' +
                               str(sequence_length) + '.p', 'wb'))
