
'''Train a simple deep CNN on the CIFAR10 small images dataset.
GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py
It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import os
import sys
import json
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import rmsprop
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from custom_callbacks import CriteriaStopping
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from hyperbolic_nonlinearities import AdaptativeAssymetricBiHyperbolic, AdaptativeBiHyperbolic, AdaptativeHyperbolicReLU, AdaptativeHyperbolic
from keras_contrib.layers.advanced_activations import PELU, SReLU
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, ThresholdedReLU

nb_classes = 10
batch_size = 128
nb_epoch = 200
dump_params = False

def evaluate_model(model, dataset, name, n_layers, hals, output_path):
    X_train, Y_train, X_test, Y_test = dataset
    csv_logger = CSVLogger(os.path.join(output_path, '%s.csv' % name))
    es = EarlyStopping(monitor='val_loss', patience=5)
    #mcp = ModelCheckpoint('output/cifar10_adaptative_%dx800/%s.checkpoint' % (n_layers, name), save_weights_only=True)
    #tb = TensorBoard(log_dir='output/cifar10_adaptative_%dx800' % n_layers, histogram_freq=1, write_graph=False, write_images=False)

    # initiate RMSprop optimizer
    opt = rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    if dump_params:
        for l in range(n_layers + 1):
            HAL = hals[l].get_weights()
            lmbda = HAL[0]
            tau_1 = HAL[1]
            tau_2 = HAL[2]

            np.savetxt(os.path.join(output_path, 'lambda_%d_start.csv' % l), lmbda, delimiter=",")
            np.savetxt(os.path.join(output_path, 'tau1_%d_start.csv' % l), tau_1, delimiter=",")
            np.savetxt(os.path.join(output_path, 'tau2_%d_start.csv' % l), tau_2, delimiter=",")

    history = model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_split=1/6,
              verbose=1,
              shuffle=True,
              callbacks=[csv_logger])
    score = model.evaluate(X_test, Y_test, verbose=1)

    if dump_params:
        for l in range(n_layers + 1):
            HAL = hals[l].get_weights()
            lmbda = HAL[0]
            tau_1 = HAL[1]
            tau_2 = HAL[2]

            np.savetxt(os.path.join(output_path, 'lambda_%d_stop.csv' % l), lmbda, delimiter=",")
            np.savetxt(os.path.join(output_path, 'tau1_%d_stop.csv' % l), tau_1, delimiter=",")
            np.savetxt(os.path.join(output_path, 'tau2_%d_stop.csv' % l), tau_2, delimiter=",")

    epochs = len(history.epoch)

    return score[0], score[1], epochs

def load_dataset():
    # the data, shuffled and split between train and test sets

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

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
#    elif name == 'psoftplus':
#        return ParametricSoftplus()
    elif name == 'sigmoid':
        return Activation('sigmoid')
    elif name == 'relu':
        return Activation('relu')
    elif name == 'tanh':
        return Activation('tanh')
    elif name == 'softplus':
        return Activation('softplus')

def __main__(argv):
    n_layers = int(argv[0])
    print(n_layers,'layers')

    output_path = 'output/cifar10_adaptative_cnn'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    (X_train, Y_train, X_test, Y_test) = load_dataset()
    dataset = (X_train, Y_train, X_test, Y_test)

    nonlinearities = ['relu', 'aabh', 'abh', 'ah', 'ahrelu', 'srelu', 'prelu', 'lrelu', 'trelu', 'elu', 'pelu', 'sigmoid', 'tanh', 'softplus']

    with open(os.path.join(output_path, 'compare.csv'), "a") as fp:
            fp.write("fn,test_loss,test_acc,epochs\n")

    hals = []

    for name in nonlinearities:
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:]))
        HAL = create_layer(name)
        model.add(HAL)
        model.add(Conv2D(32, (3, 3)))
        HAL = create_layer(name)
        model.add(HAL)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        HAL = create_layer(name)
        model.add(HAL)
        model.add(Conv2D(64, (3, 3)))
        HAL = create_layer(name)
        model.add(HAL)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        HAL = create_layer(name)
        model.add(HAL)
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.summary()

        loss, acc, epochs = evaluate_model(model, dataset, name, n_layers, hals, output_path)

        with open(os.path.join(output_path, 'compare.csv'), "a") as fp:
            fp.write("%s,%f,%f,%d\n" % (name, loss, 100*acc, epochs))

        model = None

if __name__ == "__main__":
   __main__(sys.argv[1:])
