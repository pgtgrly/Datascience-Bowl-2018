'''
Script to overlay all masks in a single training point on one background. Output is
stored in each individual data point's folder in stage1_train

dataset_path :- Source of stage1_train

'''

# coding: utf-8

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

dataset_path="data/stage1/stage1_train/" 
train_data_folder=os.listdir(dataset_path)
train_data_folder.sort()

for ImageMaskFolder in train_data_folder:
    print(ImageMaskFolder)
    MaskList = os.listdir(dataset_path + ImageMaskFolder + "/masks")
    ImgSizeTemp = cv2.imread(dataset_path + ImageMaskFolder + "/masks/"+ MaskList[0])
    MaskTemp = np.zeros(ImgSizeTemp.shape)
    MaskTemp=MaskTemp.astype('uint8')
    i=0
    for imgs in MaskList:
        img = cv2.imread(dataset_path + ImageMaskFolder + "/masks/"+ imgs)
        MaskTemp=img+MaskTemp
        i+=1
    path = dataset_path+ImageMaskFolder
    cv2.imwrite(os.path.join(path, ImageMaskFolder+"_"+str(i)+"_mask.png"), MaskTemp)

