
# coding: utf-8

# In[1]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[2]:


dataset_path="/media/alu57202/30CA81DBCA819DAA/Projects/Kaggle/2018 Data science bowl/stage1_train" # Change the path to stage1_train
train_data_folder=os.listdir(dataset_path)
train_data_folder.sort()


# In[8]:


for ImageMaskFolder in train_data_folder:
    print(ImageMaskFolder)
    MaskList = os.listdir("/media/alu57202/30CA81DBCA819DAA/Projects/Kaggle/2018 Data science bowl/stage1_train/" + ImageMaskFolder + "/masks")
    ImgSizeTemp = cv2.imread("/media/alu57202/30CA81DBCA819DAA/Projects/Kaggle/2018 Data science bowl/stage1_train/" + ImageMaskFolder + "/masks/"+ MaskList[0])
    MaskTemp = np.zeros(ImgSizeTemp.shape)
    MaskTemp=MaskTemp.astype('uint8')
    i=0
    for imgs in MaskList:
        img = cv2.imread("/media/alu57202/30CA81DBCA819DAA/Projects/Kaggle/2018 Data science bowl/stage1_train/" + ImageMaskFolder + "/masks/"+ imgs)
        #print(img.dtype)
        MaskTemp=img+MaskTemp
        #print(imgs)
        i+=1
    MaskTemp=
    path= "/media/alu57202/30CA81DBCA819DAA/Projects/Kaggle/2018 Data science bowl/stage1_train/"+ImageMaskFolder
    cv2.imwrite(os.path.join(path, ImageMaskFolder+"_"+str(i)+"_mask.png"), MaskTemp)

