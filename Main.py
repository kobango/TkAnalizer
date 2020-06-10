import TrainDataGen
import NetworkGen
import numpy


PATH_TO_DATA = 'D:\\Task01_BrainTumour\\imagesTr\\BRATS_'
Dataset = TrainDataGen.MasDataRead(PATH_TO_DATA)
NetworkGen.Create_network(Dataset)