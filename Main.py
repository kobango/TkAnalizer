#import TrainDataGen
#import NetworkGen
import numpy
import gc
import Mono_DataGen
import MonoNetworkGen

#gc.enable()
PATH_TO_DATA = 'D:\\Task01_BrainTumour\\imagesTr\\BRATS_'
#Dataset = TrainDataGen.MasDataRead(PATH_TO_DATA,20)
#NetworkGen.Create_network(Dataset,"Test_21_20",100)

DatasetMono = Mono_DataGen.MasDataRead(PATH_TO_DATA,27,2)
MonoNetworkGen.Create_network(DatasetMono,"Mono_25_basic_Ephos30_Slice2",25)

#Dataset = TrainDataGen.MasDataRead(PATH_TO_DATA,15)
#NetworkGen.Create_network(Dataset,"Test_21_15",100)

#Dataset = TrainDataGen.MasDataRead(PATH_TO_DATA,25)
#NetworkGen.Create_network(Dataset,"Test_21_25",100)

#Dataset = TrainDataGen.MasDataRead(PATH_TO_DATA,30)
#NetworkGen.Create_network(Dataset,"Test_21_30",100)
#gc.disable()