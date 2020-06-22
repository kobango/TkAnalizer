from keras.datasets import mnist
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras import backend as K
from keras.optimizers import SGD
from keras.backend.tensorflow_backend import set_session

import tensorflow as tf

import pickle
import os
import keras
import keras.backend.tensorflow_backend as ts

import resetKeras

import cv2

from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt



from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns

import numpy as np

def Create_network(a,name,epoki):
    # load mnist data


    Inputset = a[0]
    Labelset = a[1]
    print(Inputset.shape)
    print(Labelset.shape)
    NUM_OF_SAMPLES,IMG_WIDTH, IMG_HEIGHT, DEM =  Inputset.shape
    NUM_OF_TEST_SAMPLES = 100
    NUM_OF_TRAIN_SAMPLES = NUM_OF_SAMPLES -  NUM_OF_TEST_SAMPLES
    I_train = Inputset[0:NUM_OF_TRAIN_SAMPLES]
    L_train = Labelset[0:NUM_OF_TRAIN_SAMPLES]
    I_test = Inputset[NUM_OF_TRAIN_SAMPLES:NUM_OF_SAMPLES]
    L_test = Labelset[NUM_OF_TRAIN_SAMPLES:NUM_OF_SAMPLES]

    I_train = I_train.reshape((NUM_OF_TRAIN_SAMPLES, IMG_WIDTH, IMG_HEIGHT, 4))
    L_train = L_train.reshape((NUM_OF_TRAIN_SAMPLES, 1))

    I_test = I_test.reshape((NUM_OF_TEST_SAMPLES, IMG_WIDTH, IMG_HEIGHT, 4))
    L_test = L_test.reshape((NUM_OF_TEST_SAMPLES, 1))

    print(f'Shape of our training data: {I_train.shape}')
    print(f'Shape of our training labels: {L_train.shape}')

    print(f'Shape of our test data: {I_test.shape}')
    print(f'Shape of our test labels: {L_test.shape}')

    I_train = I_train.astype('float32')
    I_test = I_test.astype('float32')

    # now normalize
    I_train /= 255.0
    I_test /= 255.0

    L_train = np_utils.to_categorical(L_train)
    L_test = np_utils.to_categorical(L_test)

    NUM_OF_CLASSES = L_test.shape[1]
    print(f'Number of classes/ Number of columns : {NUM_OF_CLASSES}')

    input_shape = (IMG_WIDTH, IMG_HEIGHT, 4)

    # create model

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_OF_CLASSES, activation='sigmoid'))
    opt = keras.optimizers.SGD(learning_rate=0.0002)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    print(model.summary())

    batch_size = 2
    epochs = epoki

    history = model.fit(x=I_train, y=L_train, epochs=epochs, batch_size=batch_size, verbose=1,
                        validation_data=(I_test, L_test))

    score = model.evaluate(I_test, L_test, verbose=0)

    print(f'Test loss: {score[0]}')
    print(f'Test accuracy: {score[1]}')

    #plot_model(model, show_shapes=True, show_layer_names=True)  # -> this will by default save image to png

    #img = cv2.imread('model.png', 1)
    #plt.figure(figsize=(30, 15))
    #plt.imshow(img)

    history_dict = history.history
    os.makedirs(os.path.dirname('D:/Task01_BrainTumour/'+name+'/trainHistoryDict'), exist_ok=True)
    os.makedirs(os.path.dirname('D:/Task01_BrainTumour/'+name+"/network"), exist_ok=True)
    with open('D:/Task01_BrainTumour/'+name+'/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history_dict, file_pi)
    model.save('D:/Task01_BrainTumour/'+name+"/network")

    print(os.path.dirname('/'+name))
    resetKeras.reset_keras(model)
    del history_dict

   # print(history_dict.keys())

    #loss_values = history_dict['loss']
    #val_loss_values = history_dict['val_loss']
    #epochs_as_list = range(1, len(loss_values) + 1)

    #print(type(epochs_as_list))

    #plt.style.use('seaborn-darkgrid')

    #train_loss_line = plt.plot(epochs_as_list, loss_values, label='Train loss')
    #test_loss_line = plt.plot(epochs_as_list, val_loss_values, label='Validation/Test loss')

    #plt.setp(train_loss_line, linewidth=2.0, marker='*', markersize=5.0)
    #plt.setp(test_loss_line, linewidth=2.0, marker='*', markersize=5.0)

    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.grid(True)
    #plt.legend()
    #plt.show()

    #acc_values = history_dict['acc']
    #val_acc_values = history_dict['val_acc']

    #train_acc_line = plt.plot(epochs_as_list, acc_values, label='Train accuracy')
    #test_acc_line = plt.plot(epochs_as_list, val_acc_values, label='Test accuracy')

    #plt.setp(train_acc_line, linewidth=2.0, marker='*', markersize=5.0)
    #plt.setp(test_acc_line, linewidth=2.0, marker='*', markersize=5.0)

    #plt.xlabel('Epochs')
    #plt.ylabel('Accuracy')
    #plt.grid(True)
    #plt.legend()
    #plt.show()

    #y_pred = model.predict_classes(I_test)

    #print(classification_report(np.argmax(L_test, axis=1), y_pred))

    #cm = confusion_matrix(np.argmax(L_test, axis=1), y_pred)  # np.argmax because our labels were one hot encoded
    #plt.figure(figsize=(20, 10))
    #sns.heatmap(cm, annot=True)