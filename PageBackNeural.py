from keras.models import load_model
network = None;
import os
import PIL
import numpy as np
from PIL import Image
imageHeight = 240
imageWith = 240
numberOfCanal = 1
typeOfImageResizeAproxymation = PIL.Image.ANTIALIAS

def LoadModel(pathToModel):
    network = load_model(pathToModel)
    return network

def Loaddata(pathToImg):
    print(pathToImg)
    oneImg = Image.open(pathToImg)
    oneImg = oneImg.resize((imageWith, imageHeight), typeOfImageResizeAproxymation)
    tablizedImage = np.array(oneImg.getdata()).reshape(1,imageWith, imageHeight, numberOfCanal)
    return tablizedImage

def Predict(network,PATH_TO_IMG):
    dataset = Loaddata(PATH_TO_IMG)

    result = network.predict_classes(dataset)

    if(result[0]==1):
        return True
    else:
        return False

def main():
    PATH_TO_IMG = 'D:\\Task01_BrainTumour\\X_ImagesFromData\\71Oznaczenie1.jpg'
    path_to_network = 'D:\\Task01_BrainTumour\\Mono_27_basic_Ephos50_Slice2_extended\\network'
    print("Main function of the module")
    network = LoadModel(path_to_network) #To powinno byc na poczatku pracy serwera
    #

    dataset = Loaddata(PATH_TO_IMG)
    print(dataset)

    result = network.predict_classes(dataset)
    print(result)



if __name__ == "__main__":
    # execute only if run as a script
    main()