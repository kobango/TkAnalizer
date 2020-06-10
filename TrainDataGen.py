import numpy as np
import os
from nibabel.testing import data_path
import nibabel as nib
from matplotlib import pyplot as plt


def MasDataRead(path):

    Inputset = []
    Labelset = []
    for a in range (1,20):
        ## Read Imput Data
        example_filename = os.path.join(data_path, path+str(a).zfill(3)+'.nii.gz')
        img = nib.load(example_filename)
        obraz = img.get_fdata()
        first_vol = obraz[:, :, :, :]
        print(first_vol.shape)
        slice_number = first_vol.shape[2]
        for oneslice in range(1,slice_number):
            test = first_vol[:, :, oneslice]
            Inputset.append(test)


        ## Read labels Data
        example_filename2 = os.path.join(data_path, path+str(a).zfill(3)+'.nii.gz')
        img2 = nib.load(example_filename2)
        obraz2 = img2.get_fdata()

        slice_number = obraz2.shape[2]
        for oneslice in range(1, slice_number):
            test = obraz2[:, :, oneslice]
            size_of_cancer = np.sum(test==2) + np.sum(test==3)
            if (size_of_cancer>40):
                Labelset.append(1)
            else:
                Labelset.append(0)

    print(Inputset.__sizeof__())
    print(Labelset.__sizeof__())
    Inputset =  np.array(Inputset)
    Labelset =  np.array(Labelset)

    return [Inputset, Labelset]



def main():
    print("Main function of the module")
    Inputset, Labelset = MasDataRead('D:\\Task01_BrainTumour\\imagesTr\\BRATS_')


if __name__ == "__main__":
    # execute only if run as a script
    main()


