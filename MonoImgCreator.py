import Mono_DataGen
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


def CreateImgFromData(dataset,path):

    iter =0
    Inputset = dataset[0]
    Labelset = dataset[1]
    NUM_OF_TRAIN_SAMPLES =np.size(Labelset)
    print("Wczytywanie")
    print(NUM_OF_TRAIN_SAMPLES)
    print(np.shape(Inputset))
    print(sum(sum((Inputset))))




    while iter<NUM_OF_TRAIN_SAMPLES:

        tablica = Inputset[iter]
        tablica = tablica
        print(np.shape(tablica))
        min = np.min(tablica)
        max = np.max(tablica)
        fig = plt.figure(1)
        plt.imshow(tablica, interpolation='nearest', cmap='gray', vmin=min, vmax=max)
        if(Labelset[iter]==1):
            #plt.show()
            print("Guz")
        tablica = tablica/max
        tablica = tablica*255
        oneImg = Image.fromarray(tablica)
        oneImg = oneImg.convert('L')
        test = oneImg




        oneImg.save(path+"\\"+str(iter)+"Oznaczenie"+str(Labelset[iter])+".jpg")
        iter += 1

def main():
    PATH_TO_DATA = 'D:\\Task01_BrainTumour'
    path = 'D:\\Task01_BrainTumour\\X_ImagesFromData'
    print("Main function of the module")
    dataset = Mono_DataGen.MasDataRead(PATH_TO_DATA,2,2)
    CreateImgFromData(dataset,path)


if __name__ == "__main__":
    # execute only if run as a script
    main()