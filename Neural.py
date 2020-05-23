import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
from matplotlib import pyplot as plt
import med2image
import tensorflow as tf
import keras_segmentation
import keras

def NetworkGenerator(height,width):
    img_input = keras.layers.Input(shape=(height, width,1))

    ## Enkoder
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    conv1 = keras.layers.Dropout(0.2)(conv1)
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = keras.layers.MaxPooling2D((2, 2))(conv1)

    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = keras.layers.Dropout(0.2)(conv2)
    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = keras.layers.MaxPooling2D((2, 2))(conv2)

    ## Dekoder
    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = keras.layers.Dropout(0.2)(conv3)
    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = keras.layers.concatenate([keras.layers.UpSampling2D((2, 2))(conv3), conv2], axis=-1)
    conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = keras.layers.Dropout(0.2)(conv4)
    conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    up2 = keras.layers.concatenate([keras.layers.UpSampling2D((2, 2))(conv4), conv1], axis=-1)
    conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = keras.layers.Dropout(0.2)(conv5)
    conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    ## Wyjście
    out = keras.layers.Conv2D(5, (1, 1), padding='same')(conv5)

    from keras_segmentation.models.model_utils import get_segmentation_model

    model = get_segmentation_model(img_input,out)  # this would build the segmentation model
    return model

def main():
    example_filename = os.path.join(data_path, 'D:\\Task01_BrainTumour\\imagesTr\\BRATS_001.nii.gz')
    img = nib.load(example_filename)
    print(img.shape)
    obraz = img.get_fdata()
    first_vol = obraz[:, :, :, 0]
    test = first_vol[:, :, 70]


    print(test)
    fig = plt.figure(1)
    min = np.min(test)
    max = np.max(test)
    plt.imshow(test, interpolation='nearest', cmap='gray', vmin=min, vmax=max)

    example_filename = os.path.join(data_path, 'D:\\Task01_BrainTumour\\labelsTr\\BRATS_001.nii.gz')
    img = nib.load(example_filename)
    obraz = img.get_fdata()
    test = obraz[:, :, 70]

    print(test)
    fig2 = plt.figure(2)
    plt.imshow(test)
    plt.show()


def MasDataRead(path):

    Inputset = []
    Labelset = []
    for a in range (1,100):
        ## Read Imput Data
        example_filename = os.path.join(data_path, path+str(a).zfill(3)+'.nii.gz')
        img = nib.load(example_filename)
        obraz = img.get_fdata()
        first_vol = obraz[:, :, :, 0]
        test = first_vol[:, :, 70]
        Inputset += [test]

        ## Read labels Data
        example_filename = os.path.join(data_path, path+str(a).zfill(3)+'.nii.gz')
        img = nib.load(example_filename)
        obraz = img.get_fdata()
        test = obraz[:, :, 70]
        Labelset += [test]

    return Inputset, Labelset

def Training(model,Inputset,Labelset):
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(Inputset,Labelset,epochs=10)
    return model

def Generator():
    model = NetworkGenerator(240,240)
    Inputset,Labelset = MasDataRead('D:\\Task01_BrainTumour\\imagesTr\\BRATS_')
    Training(model,Inputset,Labelset)

if __name__ == "__main__":
    # execute only if run as a script
    ##main()
    Generator()


