import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
from matplotlib import pyplot as plt


example_filename = os.path.join(data_path, 'D:\\Task01_BrainTumour\\imagesTr\\BRATS_001.nii.gz')
img = nib.load(example_filename)
print(img.shape)
obraz = img.get_fdata()

result = obraz[:,:,140,:]


print(result)
plt.imshow(result, interpolation='nearest',  cmap='gray', vmin=0, vmax=255)
plt.show()