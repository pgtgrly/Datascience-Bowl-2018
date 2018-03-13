import torch
import torch.nn as nn
from torch.autograd import Variable

class Network1(nn.Module):

    def __init__(self):
        super(Network1,self).__init__()

        #Input Tensor Dimensions = 64x64x3
        #Convolution 1
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16, kernel_size=5,stride=1, padding=2)
        nn.init.xavier_uniform(self.conv1.weight) #Xaviers Initialisation
        self.activ_1= nn.ELU()
        #Pooling 1
        self.pool1= nn.MaxPool2d(kernel_size=2)
        #Output Tensor Dimensions = 32x32x16


        #Input Tensor Dimensions = 32x32x16
        #Convolution 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,padding=1)
        nn.init.xavier_uniform(self.conv2.weight)
        self.activ_2 = nn.ELU()
        #Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        #Output Tensor Dimensions = 16x16x32

        #Input Tensor Dimensions = 16x16x32
        #Convolution 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1)
        nn.init.xavier_uniform(self.conv3.weight)
        self.activ_3 = nn.ELU()
        #Output Tensor Dimensions = 16x16x64
        
        #https://www.quora.com/How-do-you-calculate-the-output-dimensions-of-a-deconvolution-network-layer
        #Input Tensor Dimensions = 16x16x64
        #De Convolution 1
        self.deconv1=nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,padding=1)
        nn.init.xavier_uniform(self.deconv1.weight)
        self.activ_4=nn.ELU()
        #UnPooling 1
        self.unpool1=nn.MaxUnpool2d(kernel_size=2)
        #Output Tensor Dimensions = 32x32x32

        #Input Tensor Dimensions = 32x32x32
        #De Convolution 2
        self.deconv2=nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=3,padding=1)
        nn.init.xavier_uniform(self.deconv2.weight)
        self.activ_5=nn.ELU()
        #UnPooling 2
        self.unpool2=nn.MaxUnpool2d(kernel_size=2)
        #Output Tensor Dimensions = 64x64x16

        #Input Tensor Dimensions= 64x64x16
        #DeConvolution 3
        self.deconv3=nn.ConvTranspose2d(in_channels=16,out_channels=1,kernel_size=5,padding=2)
        nn.init.xavier_uniform(self.deconv3.weight)
        self.activ_6=nn.Sigmoid()
        ##Output Tensor Dimensions = 64x64x1

    def forward(self,x):

        out = self.conv1(x)
        out = self.activ_1(out)
        size1 = out.size()
        out,indices1=self.pool1(out)
        out = self.conv2(out)
        out = self.activ_2(out)
        size2 = out.size()
        out,indices2=self.pool2(out)
        out = self.conv3(out)
        out = self.activ_3(out)
        
        out=self.deconv1(out)
        out=self.activ_4(out)
        out=self.unpool1(out,indices2,size2)
        out=self.deconv2(out)
        out=self.activ_5(out)
        out=self.unpool2(out,indices1,size1)
        out=self.deconv3(out)
        out=self.activ_6(out)
        out=255*out
        return out

class Network2(nn.Module):

    def __init__(self):
        super(Network2,self).__init__()

        #Input Tensor Dimensions = 128x128x3
        #Convolution 1
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16, kernel_size=5,stride=1, padding=2)
        nn.init.xavier_uniform(self.conv1.weight) #Xaviers Initialisation
        self.activ_1= nn.ELU()
        #Pooling 1
        self.pool1= nn.MaxPool2d(kernel_size=2)
        #Output Tensor Dimensions = 64x64x16


        #Input Tensor Dimensions = 64x64x16
        #Convolution 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,padding=1)
        nn.init.xavier_uniform(self.conv2.weight)
        self.activ_2 = nn.ELU()
        #Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        #Output Tensor Dimensions = 32x32x32

        #Input Tensor Dimensions = 32x32x32
        #Convolution 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1)
        nn.init.xavier_uniform(self.conv3.weight)
        self.activ_3 = nn.ELU()
        #Output Tensor Dimensions = 32x32x64
        
        #https://www.quora.com/How-do-you-calculate-the-output-dimensions-of-a-deconvolution-network-layer
        #Input Tensor Dimensions = 32x32x64
        #De Convolution 1
        self.deconv1=nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,padding=1)
        nn.init.xavier_uniform(self.deconv1.weight)
        self.activ_4=nn.ELU()
        #UnPooling 1
        self.unpool1=nn.MaxUnpool2d(kernel_size=2)
        #Output Tensor Dimensions = 64x64x32

        #input from network1 here

        #Input Tensor Dimensions = 64x64x33
        #De Convolution 2
        self.deconv2=nn.ConvTranspose2d(in_channels=33,out_channels=16,kernel_size=3,padding=1)
        nn.init.xavier_uniform(self.deconv2.weight)
        self.activ_5=nn.ELU()
        #UnPooling 2
        self.unpool2=nn.MaxUnpool2d(kernel_size=2)
        #Output Tensor Dimensions = 128x128x16

        #Input Tensor Dimensions= 128x128x16
        #DeConvolution 3
        self.deconv3=nn.ConvTranspose2d(in_channels=16,out_channels=1,kernel_size=5,padding=2)
        nn.init.xavier_uniform(self.deconv3.weight)
        self.activ_6=nn.Sigmoid()
        ##Output Tensor Dimensions = 128x128x1

    def forward(self,x,network1_output):

        out = self.conv1(x)
        out = self.activ_1(out)
        size1 = out.size()
        out,indices1=self.pool1(out)
        out = self.conv2(out)
        out = self.activ_2(out)
        size2 = out.size()
        out,indices2=self.pool2(out)
        out = self.conv3(out)
        out = self.activ_3(out)
        
        out=self.deconv1(out)
        out=self.activ_4(out)
        out=self.unpool1(out,indices2,size2)

        out=torch.cat((out,network1_output),dim=0) #check if dim is correct

        out=self.deconv2(out)
        out=self.activ_5(out)
        out=self.unpool2(out,indices1,size1)
        out=self.deconv3(out)
        out=self.activ_6(out)
        out=255*out
        return out

class network3(nn.Module):

    def __init__(self):
        super(network3,self).__init__()

        #Input Tensor Dimensions = 256x256x3
        #Convolution 1
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16, kernel_size=5,stride=1, padding=2)
        nn.init.xavier_uniform(self.conv1.weight) #Xaviers Initialisation
        self.activ_1= nn.ELU()
        #Pooling 1
        self.pool1= nn.MaxPool2d(kernel_size=2)
        #Output Tensor Dimensions = 128x128x16


        #Input Tensor Dimensions = 128x128x16
        #Convolution 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,padding=1)
        nn.init.xavier_uniform(self.conv2.weight)
        self.activ_2 = nn.ELU()
        #Pooling 2
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        #Output Tensor Dimensions = 64x64x32

        #Input Tensor Dimensions = 64x64x32
        #Convolution 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1)
        nn.init.xavier_uniform(self.conv3.weight)
        self.activ_3 = nn.ELU()
        #Output Tensor Dimensions = 64x64x64

        #Input From Network1
        
        #https://www.quora.com/How-do-you-calculate-the-output-dimensions-of-a-deconvolution-network-layer
        #Input Tensor Dimensions = 64x64x65
        #De Convolution 1
        self.deconv1=nn.ConvTranspose2d(in_channels=65,out_channels=32,kernel_size=3,padding=1)
        nn.init.xavier_uniform(self.deconv1.weight)
        self.activ_4=nn.ELU()
        #UnPooling 1
        self.unpool1=nn.MaxUnpool2d(kernel_size=2)
        #Output Tensor Dimensions = 128x128x32

        #Input From Network2

        #Input Tensor Dimensions = 128x128x33
        #De Convolution 2
        self.deconv2=nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=3,padding=1)
        nn.init.xavier_uniform(self.deconv2.weight)
        self.activ_5=nn.ELU()
        #UnPooling 2
        self.unpool2=nn.MaxUnpool2d(kernel_size=2)
        #Output Tensor Dimensions = 256x256x16

        #Input Tensor Dimensions= 256x256x16
        #DeConvolution 3
        self.deconv3=nn.ConvTranspose2d(in_channels=16,out_channels=1,kernel_size=5,padding=2)
        nn.init.xavier_uniform(self.deconv3.weight)
        self.activ_6=nn.Sigmoid()
        ##Output Tensor Dimensions = 256x256x1

    def forward(self,x,network1_output,network2_output):

        out = self.conv1(x)
        out = self.activ_1(out)
        size1 = out.size()
        out,indices1=self.pool1(out)
        out = self.conv2(out)
        out = self.activ_2(out)
        size2 = out.size()
        out,indices2=self.pool2(out)
        out = self.conv3(out)
        out = self.activ_3(out)

        out=torch.cat((out,network1_output),dim=0) #check if dim is correct
        
        out=self.deconv1(out)
        out=self.activ_4(out)
        out=self.unpool1(out,indices2,size2)

        out=torch.cat((out,network2_output),dim=0) #check if dim is correct

        out=self.deconv2(out)
        out=self.activ_5(out)
        out=self.munpool2(out,indices1,size1)
        out=self.deconv3(out)
        out=self.activ_6(out)
        out=255*out
        return out

