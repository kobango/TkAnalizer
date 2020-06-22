import TrainDataGen
import NetworkGen
import numpy


PATH_TO_DATA = 'D:\\Task01_BrainTumour\\imagesTr\\BRATS_'
Dataset = TrainDataGen.MasDataRead(PATH_TO_DATA,20)
NetworkGen.Create_network(Dataset,"Prometeusz20",100)

#Dataset = TrainDataGen.MasDataRead(PATH_TO_DATA,15)
#NetworkGen.Create_network(Dataset,"Prometeusz2",20)

#Dataset = TrainDataGen.MasDataRead(PATH_TO_DATA,20)
#NetworkGen.Create_network(Dataset,"Prometeusz3",30)

#Dataset = TrainDataGen.MasDataRead(PATH_TO_DATA,25)
#NetworkGen.Create_network(Dataset,"Prometeusz4",40)