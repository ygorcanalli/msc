
'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import sys
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from custom_callbacks import CriteriaStopping
from keras.callbacks import CSVLogger, EarlyStopping
from hyperbolic_nonlinearities import BiHyperbolic

def __main__(argv):

    batch_size = 128
    nb_classes = 10
    nb_epoch = 1000

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


    lambdas = np.arange(0.5, 6, 0.5)
    taus = np.arange(0.5, 6, 0.5)
    n_layers = int(argv[0])

    print("n_layers=",n_layers)

    nonlinearities = {
        "sigmoid": Activation('sigmoid'),
        "tanh": Activation('tanh'),
        "relu": Activation('relu')
    }

    for lmbda in lambdas:
        for tau in taus:
            #nonlinearities['bihip_%.1f_%.1f' % (lmbda, tau)] = BiHyperbolic(lmbda, tau, mode='basic')
            nonlinearities['extbihip_%.1f_%.1f' % (lmbda, tau)] = BiHyperbolic(lmbda, tau, mode='ext')

    with open("output/mnist_bihyperbolic_manual_extended_%dx800/compare.csv" % n_layers, "a") as fp:
            fp.write("fn,test_loss,test_acc,epochs\n")

    for name, fn in nonlinearities.items():
        model = Sequential()

        model.add(Dense(800, input_shape=(784,)))
        model.add(fn)
        for l in range(n_layers):
            model.add(Dense(800))
            model.add(fn)
        model.add(Dense(10))
        model.add(Activation('softmax'))
        model.summary()

        csv_logger = CSVLogger('output/mnist_bihyperbolic_manual_extended_%dx800/%s.csv' % (n_layers, name))
        es = EarlyStopping(monitor='val_loss', patience=5)

        sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=1/6, callbacks=[es, csv_logger])
        score = model.evaluate(X_test, Y_test, verbose=0)

        epochs = len(history.epoch)

        with open("output/mnist_bihyperbolic_manual_extended_%dx800/compare.csv" % n_layers, "a") as fp:
            fp.write("%s,%f,%f,%d\n" % (name, score[0], 100*score[1], epochs))

        model = None

if __name__ == "__main__":
   __main__(sys.argv[1:])
