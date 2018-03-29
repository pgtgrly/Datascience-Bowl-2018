import torch
import torch.nn as nn
import os
from torch.autograd import Variable
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re
from tensorboardX import SummaryWriter
import time
import glob
from networks import network1
from networks import network2
from networks import network3
from sys import argv

checkpoints_directory_network_1="checkpoints_network_1"
checkpoints_directory_network_2="checkpoints_network_2"
checkpoints_directory_network_3="checkpoints_network_3"

script, img_path, img_name = argv

checkpointsNet1= os.listdir(checkpoints_directory_network_1)
checkpointsNet1.sort(key=lambda x:int((x.split('_')[2]).split('.')[0]))
model_network_1 = torch.load(checkpoints_directory_network_1+'/'+checkpointsNet1[-1])
checkpointsNet2= os.listdir(checkpoints_directory_network_2)
checkpointsNet2.sort(key=lambda x:int((x.split('_')[2]).split('.')[0]))
model_network_2 = torch.load(checkpoints_directory_network_2+'/'+checkpointsNet2[-1])
checkpointsNet3= os.listdir(checkpoints_directory_network_3)
checkpointsNet3.sort(key=lambda x:int((x.split('_')[2]).split('.')[0]))
model_network_3 = torch.load(checkpoints_directory_network_3+'/'+checkpointsNet2[-1])

model_network_1.eval()
model_network_2.eval()
model_network_3.eval()

image = cv2.imread(os.path.join(img_path,img_name))
input_net1 = image
input_net2 = image
input_net3 = image

input_net3=cv2.resize(input_net3,(256,256), interpolation = cv2.INTER_CUBIC)
input_net3= input_net3.reshape((256,256,3,1))
input_net2=cv2.resize(input_net2,(128,128), interpolation = cv2.INTER_CUBIC)
input_net2= input_net2.reshape((128,128,3,1))
input_net1=cv2.resize(input_net1,(64,64), interpolation = cv2.INTER_CUBIC)
input_net1= input_net1.reshape((64,64,3,1))

input_net1 = input_net1.transpose((3, 2, 0, 1))
input_net2 = input_net2.transpose((3, 2, 0, 1))
input_net3 = input_net3.transpose((3, 2, 0, 1))

input_net1.astype(float)
input_net1=input_net1/255
input_net2.astype(float)
input_net2=input_net2/255
input_net3.astype(float)
input_net3=input_net3/255

input_net1 = torch.from_numpy(input_net1)
input_net2 = torch.from_numpy(input_net2)
input_net3 = torch.from_numpy(input_net3)

input_net1=input_net1.type(torch.FloatTensor)
input_net2=input_net2.type(torch.FloatTensor)
input_net3=input_net3.type(torch.FloatTensor)

input_net1 = Variable(input_net1)
input_net2 = Variable(input_net2)
input_net3 = Variable(input_net3)

out_net1 = model_network_1(input_net1)
out_net2 = model_network_2(input_net2, out_net1)
out_net3 = model_network_3(input_net3, out_net1, out_net2)

out_net1 =  out_net1.data.numpy()
out_net2 =  out_net2.data.numpy()
out_net3 =  out_net3.data.numpy()

out_net1 = out_net1*255
out_net2 = out_net2*255
out_net3 = out_net3*255

out_net1 = out_net1.transpose((2,3,0,1))
out_net1 = out_net1.reshape((64,64,1))
out_net2 = out_net2.transpose((2,3,0,1))
out_net2= out_net2.reshape((128,128,1))
out_net3 = out_net3.transpose((2,3,0,1))
out_net3= out_net3.reshape((256,256,1))

cv2.imwrite(os.path.join(img_path, "Output_net_1.png"), out_net1)
cv2.imwrite(os.path.join(img_path, "Output_net_2.png"), out_net2)
cv2.imwrite(os.path.join(img_path, "Output_net_3.png"), out_net3)