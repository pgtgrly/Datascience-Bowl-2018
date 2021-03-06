import torch
import torch.nn as nn
import os
from torch.autograd import Variable
from skimage import io, transform
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re
from tensorboardX import SummaryWriter
import time
import glob
import pandas as pd
from Networks import network1

writer = SummaryWriter()

batch_size = 10 #mini-batch size
n_iters = 50000 #total iterations
learning_rate = 0.001
train_directory="train"
validation_directory=""
test_directory=""
checkpoints_directory="checkpoints"
test_batch_size=50
threshold=128

class ImageDataset(Dataset): #Defining the class to load datasets
    def __init__(self,stage=1, input_dir='train',transform=None):
        self.input_dir = os.path.join("data/stage", str(stage), input_dir)        
        self.transform = transform
        self.dirlist = os.listdir(self.input_dir).sort()

    def __len__ (self):
        return len(os.listdir(self.input_dir))

    def __getitem__(self,idx):

        img_id= self.dirlist[idx]
        
        input_image=cv2.imread(os.path.join(self.input_dir,img_id, "images", img_id + ".png"))
        input_image=cv2.resize(input_image,(64,64), interpolation = cv2.INTER_CUBIC)
        input_image= input_image.reshape((64,64,3)).transpose((2, 0, 1)) #The convolution function in pytorch expects data in format (N,C,H,W) N is batch size , C are channels H is height and W is width. here we convert image from (H,W,C) to (C,H,W)
        
        mask_path = glob.glob(os.path.join(self.input_dir,img_id) + "*.png")     
        no_of_masks = int(mask_path.split("_")[1])

        output_image=cv2.imread(mask_path)
        output_image=cv2.resize(output_image,(64,64), interpolation = cv2.INTER_CUBIC)
        output_image= output_image.reshape((64,64,1)).transpose((2, 0, 1))                                                                             

        sample = {'image': input_image, 'masks': output_image}  

        if self.transform:
            sample= self.transform(sample)
        
        #As transforms do not involve random crop, number of masks must stay the same
        sample['count'] = no_of_masks
        
        return sample

train_dataset=ImageDataset(stage=1, input_dir="train",transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])) #Training Dataset
test_dataset=ImageDataset(stage=1, input_dir="test",transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])) #Testing Dataset
validation_dataset=ImageDataset(stage=1, input_dir="validation",transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])) #Validation Dataset

num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=test_batch_size, 
                                          shuffle=False)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, 
                                          batch_size=test_batch_size, 
                                          shuffle=False)
if not os.path.exists(directory):
    os.makedirs(directory)

model=network1()
iteri=0
iter_new=0 

#checking if checkpoints exist to resume training and create it if not
if not os.path.exists(checkpoints_directory):
    checkpoints = os.listdir(checkpoints_directory)
    checkpoints.sort(key=lambda x:int((x.split('_')[2]).split('.')[0]))
    model=torch.load(checkpoints_directory+'/'+check[-1])
    iteri=int(re.findall(r'\d+',check[-1])[0])
    iter_new=iteri
    print("Resuming from iteration " + str(iteri))
else:
    os.makedirs(checkpoints_directory)

if torch.cuda.is_available(): #use gpu if available
    model.cuda() 

criterion=nn.MSELoss()  #Loss Class
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate) #optimizer class
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)# this will decrease the learning rate by factor of 0.1
                                                                              
                                                                              # https://discuss.pytorch.org/t/can-t-import-torch-optim-lr-scheduler/5138/6 
beg=time.time() #time at the beginning of training
print("Training Started!")
for epoch in range(num_epochs):
    print("\nEPOCH " +str(epoch+1)+" of "+str(num_epochs)+"\n")
    for i,datapoint in enumerate(train_loader):
        datapoint['input_image']=datapoint['input_image'].type(torch.FloatTensor) #typecasting to FloatTensor as it is compatible with CUDA
        datapoint['output_image']=datapoint['output_image'].type(torch.FloatTensor)
        if torch.cuda.is_available(): #move to gpu if available
                input_image = Variable(datapoint['input_image'].cuda()) #Converting a Torch Tensor to Autograd Variable
                output_image = Variable(datapoint['output_image'].cuda())
        else:
                input_image = Variable(datapoint['input_image'])
                output_image = Variable(datapoint['output_image'])

        optimizer.zero_grad()  #https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/3
        outputs = model(input_image)
        loss = criterion(outputs, output_image)
        loss.backward() #Backprop
        optimizer.step()    #Weight update
        writer.add_scalar('Training Loss',loss.data[0], iteri)
        iteri=iteri+1
        if iteri % 10 == 0 or iteri==1:
            # Calculate Accuracy         
            test_loss = 0
            total = 0
            # Iterate through test dataset
            for j,datapoint_1 in enumerate(test_loader): #for testing
                datapoint_1['input_image']=datapoint_1['input_image'].type(torch.FloatTensor)
                datapoint_1['output_image']=datapoint_1['output_image'].type(torch.FloatTensor)
           
                if torch.cuda.is_available():
                    input_image_1 = Variable(datapoint_1['input_image'].cuda())
                    output_image_1 = Variable(datapoint_1['output_image'].cuda())
                else:
                    input_image_1 = Variable(datapoint_1['input_image'])
                    output_image_1 = Variable(datapoint_1['output_image'])
                
                # Forward pass only to get logits/output
                outputs = model(input_image_1)
                test_loss += criterion(outputs, output_image_1).data[0]
                total+=datapoint_1['output_image'].size(0)
            test_loss=test_loss/total   #sum of test loss for all test cases/total cases
            writer.add_scalar('Test Loss',test_loss, iteri) 
            # Print Loss
            time_since_beg=(time.time()-beg)/60
            print('Iteration: {}. Loss: {}. Test Loss: {}. Time(mins) {}'.format(iteri, loss.data[0], test_loss,time_since_beg))
        if iteri % 500 ==0:
            '''
            Apparently Optimizer saving is possible,but there is varied discussion
            on the issue with some suggestions that the recent update fixed this.

            https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3
            
            http://pytorch.org/docs/0.3.0/notes/serialization.html

            
            Workaround discussed in https://github.com/pytorch/pytorch/issues/2830

            model = Model()
            model.load_state_dict(checkpoint['model'])
            model.cuda()
            optimizer = optim.Adam(model.parameters())
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

            '''

            torch.save(model,'checkpoints/model_iter_'+str(iteri)+'.pt')
            
            # Alternative lower overhead save 
            # torch.save(model.state_dict(),'checkpoints/model_iter_'+str(iteri)+'.pt')

            print("model saved at iteration : "+str(iteri))
            writer.export_scalars_to_json("graphs/all_scalars_"+str(iter_new)+".json") #saving loss vs iteration data to be used by visualise.py
    scheduler.step()            
writer.close()
