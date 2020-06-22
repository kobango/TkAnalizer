
from keras.models import Sequential
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import TrainDataGen
from keras.models import load_model
from keras.utils import np_utils

def ModelVall(path,number):
    model = load_model(path)

    a = TrainDataGen.MasDataRead('D:\\Task01_BrainTumour\\imagesTr\\BRATS_',number)

    Inputset = a[0]
    Labelset = a[1]
    print(Inputset.shape)
    print(Labelset.shape)
    NUM_OF_SAMPLES, IMG_WIDTH, IMG_HEIGHT, DEM = Inputset.shape
    NUM_OF_TEST_SAMPLES = 100
    NUM_OF_TRAIN_SAMPLES = NUM_OF_SAMPLES - NUM_OF_TEST_SAMPLES
    I_test = Inputset[NUM_OF_TRAIN_SAMPLES:NUM_OF_SAMPLES]
    L_test = Labelset[NUM_OF_TRAIN_SAMPLES:NUM_OF_SAMPLES]

    I_test = I_test.astype('float32')
    I_test /= 255.0

    L_test = np_utils.to_categorical(L_test)


    y_pred = model.predict_classes(I_test)

    print(classification_report(np.argmax(L_test, axis=1), y_pred))

    cm = confusion_matrix(np.argmax(L_test, axis=1), y_pred)  # np.argmax because our labels were one hot encoded
    plt.figure(figsize=(20, 10))
    heat_map = sns.heatmap(cm, annot=True)
    plt.show()


def main():
    ModelVall('D:/Task01_BrainTumour/Prometeusz1/network',20)

if __name__ == "__main__":
    # execute only if run as a script
    main()


