import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
from matplotlib import pyplot as plt
import med2image
import tensorflow


example_filename = os.path.join(data_path, 'D:\\Task01_BrainTumour\\imagesTr\\BRATS_001.nii.gz')
img = nib.load(example_filename)
print(img.shape)
obraz = img.get_fdata()
first_vol = obraz[:, :, :, 0]
test = first_vol[:,:,70]

result = obraz[:,:,140,:]


print(test)
fig = plt.figure(1)
min = np.min(test)
max = np.max(test)
plt.imshow(test, interpolation='nearest',  cmap='gray', vmin=min, vmax=max)


example_filename = os.path.join(data_path, 'D:\\Task01_BrainTumour\\labelsTr\\BRATS_001.nii.gz')
img = nib.load(example_filename)
obraz = img.get_fdata()
test = obraz[:,:,70]

print(test)
fig2 = plt.figure(2)
plt.imshow(test)
plt.show()