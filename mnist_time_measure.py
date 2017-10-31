
'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import sys
import json
import numpy as np
import pandas as pd
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from custom_callbacks import CriteriaStopping, ElapsedTime
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from hyperbolic_nonlinearities import AdaptativeAssymetricBiHyperbolic, AdaptativeBiHyperbolic, AdaptativeHyperbolicReLU, AdaptativeHyperbolic, PELU, BiHyperbolic, Hyperbolic, HyperbolicReLU
from keras.layers.advanced_activations import ParametricSoftplus, SReLU, PReLU, ELU, LeakyReLU, ThresholdedReLU

nb_classes = 10
batch_size = 128
nb_epoch = 20

def evaluate_model(model, dataset, name, n_layers, hals):
    X_train, Y_train, X_test, Y_test = dataset
    csv_logger = CSVLogger('output/mnist_time_measure_%dx800/%s.csv' % (n_layers, name))
    es = EarlyStopping(monitor='val_loss', patience=5)
    et = ElapsedTime()
    #mcp = ModelCheckpoint('output/mnist_adaptative_%dx800/%s.checkpoint' % (n_layers, name), save_weights_only=True)
    #tb = TensorBoard(log_dir='output/mnist_adaptative_%dx800' % n_layers, histogram_freq=1, write_graph=False, write_images=False)

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=1/6, callbacks=[csv_logger, et])
    score = model.evaluate(X_test, Y_test, verbose=1)

    timing = et.timing
    print(timing)
    epochs = len(history.epoch)

    return score[0], score[1], epochs, timing

def load_dataset():
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, Y_train, X_test, Y_test

def create_layer(name):
    if name == 'aabh':
        return AdaptativeAssymetricBiHyperbolic()
    elif name == 'abh':
        return AdaptativeBiHyperbolic()
    elif name == 'ah':
        return AdaptativeHyperbolic()
    elif name == 'ahrelu':
        return AdaptativeHyperbolicReLU()
    elif name == 'srelu':
        return SReLU()
    elif name == 'prelu':
        return PReLU()
    elif name == 'lrelu':
        return LeakyReLU()
    elif name == 'trelu':
        return ThresholdedReLU()
    elif name == 'elu':
        return ELU()
    elif name == 'pelu':
        return PELU()
    elif name == 'psoftplus':
        return ParametricSoftplus()
    elif name == 'sigmoid':
        return Activation('sigmoid')
    elif name == 'relu':
        return Activation('relu')
    elif name == 'tanh':
        return Activation('tanh')
    elif name == 'softplus':
        return Activation('softplus')
    elif name == 'bh':
        return BiHyperbolic(1, 1)
    elif name == 'h':
        return Hyperbolic(1, 1)
    elif name == 'hrelu':
        return HyperbolicReLU(1)

def __main__(argv):
    n_layers = int(argv[0])
    print(n_layers,'layers')

    dataset = load_dataset()

    nonlinearities = ['bh', 'h', 'hrelu', 'sigmoid', 'relu', 'tanh', 'softplus']

    with open("output/mnist_time_measure_%dx800/compare.csv" % n_layers, "a") as fp:
            fp.write("fn,test_loss,test_acc,epochs\n")

    hals = []
    times = pd.DataFrame()

    for name in nonlinearities:
        model = Sequential()

        model.add(Dense(800, input_shape=(784,)))
        HAL = create_layer(name)
        model.add(HAL)
        hals.append(HAL)
        for l in range(n_layers):
            model.add(Dense(800))
            HAL = create_layer(name)
            model.add(HAL)
            hals.append(HAL)
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        model.summary()

        loss, acc, epochs, timing = evaluate_model(model, dataset, name, n_layers, hals)

        times[name] = timing

        with open("output/mnist_time_measure_%dx800/compare.csv" % n_layers, "a") as fp:
            fp.write("%s,%f,%f,%d\n" % (name, loss, 100*acc, epochs))

        model = None

    times.to_csv("output/mnist_time_measure_%dx800/timing.csv")

if __name__ == "__main__":
   __main__(sys.argv[1:])
