
from __future__ import print_function


## import libraries
import numpy as np
np.random.seed(123)

import pandas as pd
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU

from keras.layers.normalization import BatchNormalization
from keras import callbacks
import os

from datetime import datetime

## Batch generators ##################################################################################################################################
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(' Time taken: %i minutes and %s seconds.' %
              (tmin, round(tsec, 2)))

# note The generator is expected to loop over its data indefinitely. 
# An epoch finishes when samples_per_epoch samples have been seen by the model

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

########################################################################################################################################################

## read data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#train = train.iloc[0:10000]
#test = test.iloc[0:10000]

## set test loss to NaN
test['loss'] = np.nan

## response and IDs
# loss take log
y = train['loss'].values
#y = np.log(train['loss'].values)
id_train = train['id'].values
id_test = test['id'].values

## stack train test
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis = 0)


## Preprocessing and transforming to sparse data
tr_te["cont14"] = (np.maximum(tr_te["cont14"] - 0.179722, 0) / 0.665122) ** 0.25

sparse_data = []

f_cat = [f for f in tr_te.columns if 'cat' in f]
for f in f_cat:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

f_num = [f for f in tr_te.columns if 'cont' in f]
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
sparse_data.append(tmp)

del(tr_te, train, test)

## sparse train and test data
xtr_te = hstack(sparse_data, format = 'csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

del(xtr_te, sparse_data, tmp)

## neural net
## three hidden layers, each layer with BatchNormalization and dropout
## Use PReLu activation function

def nn_model():
    model = Sequential()
    model.add(Dense(400, input_dim = xtrain.shape[1], init = 'he_normal')) # 400 is output_dim of 1st layer
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.99, weights=None))
    model.add(PReLU())
    model.add(Dropout(0.4))
    
    model.add(Dense(200, init = 'he_normal'))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.99, weights=None))
    model.add(PReLU())
    model.add(Dropout(0.2))
    
    model.add(Dense(50, init = 'he_normal')) # add one hidden layer
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.99, weights=None))
    model.add(PReLU())
    model.add(Dropout(0.2))    
    
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'mae', optimizer = 'adadelta') # change loss from 'mae' to fair_obj
    return(model)

## cv-folds
nfolds = 5
folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 111)

#########################################################################
## train models
## for each model, train n bags. For each bag, implement k-folds cross validation, 
## compute the average of k models' prediction on test set

i = 0
nbags = 7
nepochs = 100 # for the nn struture, optimal nb_epoch should be 19 or 20.
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])

for (inTr, inTe) in folds:
    start_time = timer(None)
    
    xtr = xtrain[inTr]
    ytr = y[inTr]
    xte = xtrain[inTe]
    yte = y[inTe]
    pred = np.zeros(xte.shape[0])
    for j in range(nbags):
    
        earlyStopping = [callbacks.EarlyStopping(monitor='val_loss',patience = 5,
                                                verbose = 0,mode = 'auto'),
                         callbacks.ModelCheckpoint(os.getcwd() +  "/models_keras/"+'model.weights.best.hdf5', monitor='val_loss', verbose=0,
                                                   save_best_only=True, save_weights_only=False, mode='auto')]
        model = nn_model()        
        fit = model.fit_generator(generator = batch_generator(xtr, ytr, 128, True), # previously use 128
                                  nb_epoch = nepochs,
                                  samples_per_epoch = xtr.shape[0],
                                  verbose = 1,
                                  callbacks = earlyStopping,                                  
                                  validation_data = batch_generator(xte, yte, 800, False),
                                  nb_val_samples = xte.shape[0]
                                  )
        ## load the weights of best epoch
        model = nn_model()
        model.load_weights(os.getcwd() + '/models_keras/' + 'model.weights.best.hdf5')
        model.compile(loss = 'mae', optimizer = 'adadelta')
        
        pred += model.predict_generator(generator = batch_generatorp(xte, 800, False), val_samples = xte.shape[0])[:,0]
        pred_test += model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])[:,0]
    pred /= nbags
    pred_oob[inTe] = pred
    score = mean_absolute_error(yte, pred)
    #score = mean_absolute_error(np.exp(yte), np.exp(pred))
    i += 1
    print('Fold ', i, '- MAE:', score)
    timer(start_time)

#print('Total - MAE:', mean_absolute_error(np.exp(y), np.exp(pred_oob)))
print('Total - MAE:', mean_absolute_error(y, pred_oob))


## train predictions
#df = pd.DataFrame({'id': id_train, 'loss': pred_oob})
#df.to_csv('preds_oob.csv', index = False)

#####################################################################
## test predictions
pred_test /= (nfolds*nbags)
#pred_test = np.exp(pred_test)
df = pd.DataFrame({'id': id_test, 'loss': pred_test})

now = datetime.now()
file_name = 'submission_keras_' + str(nfolds)+'flolds_' + str(nbags)+'bags_' + str(nepochs) + 'epochs_' +\
str(mean_absolute_error(y, pred_oob)) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
df.to_csv(file_name, index = False)
