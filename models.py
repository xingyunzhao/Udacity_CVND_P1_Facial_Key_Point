## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)  # input (1, 224, 224), output (32, 220, 220)        
        self.pool1 = nn.MaxPool2d(2, 2)  # input (32, 220, 220), output (32, 110, 110)
        self.conv1_drop = nn.Dropout(p=0.5)  # dropout, no changes to data dimention
        
        self.conv2 = nn.Conv2d(32, 64, 4)  # input (32, 110, 110), output (64, 107, 107)
        self.pool2 = nn.MaxPool2d(2, 2)  # input (64, 107, 107), output (64, 53, 53)
        self.conv2_drop = nn.Dropout(p=0.5)  # dropout, no changes to data dimention
        
        self.conv3 = nn.Conv2d(64, 128, 3)  # input (64, 53, 53), output(128, 51, 51)
        self.pool3 = nn.MaxPool2d(2, 2)  # input (128, 51, 51), output (128, 25, 25)
        self.conv3_drop = nn.Dropout(p=0.5)  # dropout, no changes to data dimention        
        
        self.conv4 = nn.Conv2d(128, 256, 2)  # input (128, 25, 25), output(256, 24, 24)
        self.pool4 = nn.MaxPool2d(2, 2)  # input (256, 24, 24), output (256, 12, 12)
        self.conv4_drop = nn.Dropout(p=0.5)  # dropout, no changes to data dimention
        
        self.conv5 = nn.Conv2d(256, 512, 1)  # input (256, 12, 12), output(512, 12, 12)
        self.pool5 = nn.MaxPool2d(2, 2)  # input (512, 12, 12), output (512, 6, 6)
        self.conv5_drop = nn.Dropout(p=0.5)  # dropout, no changes to data dimention
        
        self.fc1 = nn.Linear(512*6*6, 1000)
        self.fc1_drop = nn.Dropout(p=0.5)  # dropout, no changes to data dimention
        
        self.fc2 = nn.Linear(1000, 1000)
        self.fc2_drop = nn.Dropout(p=0.5)  # dropout, no changes to data dimention
        
        self.fc3= nn.Linear(1000, 68*2)  # final output layer
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.conv1_drop(self.pool1(F.relu(self.conv1(x))))
        x = self.conv2_drop(self.pool2(F.relu(self.conv2(x))))
        x = self.conv3_drop(self.pool3(F.relu(self.conv3(x))))
        x = self.conv4_drop(self.pool4(F.relu(self.conv4(x))))
        x = self.conv5_drop(self.pool5(F.relu(self.conv5(x))))
        
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = self.pool3(F.relu(self.conv3(x)))
#         x = self.pool4(F.relu(self.conv4(x)))
#         x = self.pool5(F.relu(self.conv5(x)))
    
        x = x.view(x.size(0), -1)
        
        x = self.fc1_drop(F.relu(self.fc1(x)))       
        x = self.fc2_drop(F.relu(self.fc2(x)))       
        x = self.fc3(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
    
    
    
