import scipy
import numpy as np


def SegmentationMachine(numpyArrayImg2D):
    resized = np.resize(numpyArrayImg2D,(240,240))

    return resized

def main():
    print("Test resizera")
    voidArray = np.zeros((300,300))
    print(voidArray.shape)
    resizedArray = SegmentationMachine(voidArray)
    print(resizedArray.shape)
    if(resizedArray.shape==(240,240)):
        print("Skalowanie w doł sprawne")
    else:
        print("Skalowanie w dol nie sprawne")
    voidArray = np.zeros((160, 160))
    print(voidArray.shape)
    resizedArray = SegmentationMachine(voidArray)
    print(resizedArray.shape)
    if (resizedArray.shape == (240, 240)):
        print("Skalowanie w gore sprawne")
    else:
        print("Skalowanie w gore nie sprawne")





if __name__ == "__main__":
    # execute only if run as a script
    main()