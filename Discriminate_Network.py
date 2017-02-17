
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


def fit_discriminate_network(tree_structures, labels, rho, nb_epoch):
    K.set_image_dim_ordering('th')

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # load data
    # reshape to be [samples][pixels][width][height]
    X_train = np.array(tree_structures)
    y_train = labels

    X_train = X_train.reshape(X_train.shape[0], 1, np.size(tree_structures, 1), 3).astype('float32')

    # normalize inputs
    X_train = (X_train - np.mean(X_train))/np.std(X_train)

    # reshape labels
    y_train = np_utils.to_categorical(y_train)

    num_classes = y_train.shape[1]

    # Model Building
    model = Sequential()
    model.add(ZeroPadding2D(padding=(1, 7), input_shape=(1, np.size(tree_structures, 1), 3)))
    model.add(Convolution2D(32, 3, 3, border_mode='valid', activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2,2)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    # Fit the model
    model.fit(X_train, y_train, validation_split=rho,  nb_epoch=nb_epoch, batch_size=200, verbose=1)
    return model


if __name__ == '__main__':
    from Generate_Trees import *

    iris = load_iris()
    X = iris.data
    Y = iris.target
    max_depth = 3
    rho = 0.5
    nb_epoch = 2
    List_trees, List_scores = generate_trees(X, Y, max_depths)
    tree_structures, labels = labeling_trees(List_trees, List_scores, max_depth)

    fit_discriminate_network(tree_structures, labels, rho, nb_epoch)