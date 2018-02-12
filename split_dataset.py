import numpy as np
import shutil
import random
import os

#May the devil bring us good luck
seed = 666
random.seed = seed
np.random.seed = seed

dataset_path="data/stage1/stage1_train/" 
train_data_folder=os.listdir(dataset_path)
train_data_folder = np.asarray(train_data_folder)
np.random.shuffle(train_data_folder)

x_test = list(train_data_folder[0:67])
x_validation = list(train_data_folder[68:135])
x_train = list(train_data_folder[136:])

for folder in x_train:
  shutil.copytree(dataset_path + folder, "data/stage1/train/" + folder)

for folder in x_validation:
  shutil.copytree(dataset_path + folder , "data/stage1/validation/" + folder)

for folder in x_test:
  shutil.copytree(dataset_path + folder , "data/stage1/test/" + folder)

