# %%
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from typing import Tuple, Union, List
import matplotlib.pyplot as plt
import numpy as np

from morphotorch.nn.morpholayers import DepthwiseDilationLayer, DepthwiseErosionLayer

# %%
args={}
kwargs={}
args['batch_size']=100
args['test_batch_size']=100
args['epochs']=10  #The number of Epochs is the number of times you go through the full dataset. 
args['lr']=0.01 #Learning rate is how fast it will decend. 
args['momentum']=0.5 #SGD momentum (default: 0.5) Momentum is a moving average of our gradients (helps to keep direction).

args['seed']=1 #random seed
args['log_interval']=10


# %%
#@title Downloading MNIST data

train_data = dsets.MNIST(root = './data', train = True,
                        transform = transforms.ToTensor(), download = True)

test_data = dsets.MNIST(root = './data', train = False,
                       transform = transforms.ToTensor())

#@title Loading the data

train_gen = torch.utils.data.DataLoader(dataset = train_data,
                                             batch_size = args['batch_size'],
                                             shuffle = True)

test_gen = torch.utils.data.DataLoader(dataset = test_data,
                                      batch_size = args['batch_size'], 
                                      shuffle = False)


# %%
class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.Dilation = DepthwiseDilationLayer(kernel_size=(5,5),in_units=10)
        self.Erosion = DepthwiseErosionLayer(kernel_size=(5,5),in_units=10)
        self.conv2_drop = nn.Dropout2d()  #Dropout
        self.fc1 = nn.Linear(10*24*24, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        #Convolutional Layer/Pooling Layer/Activation
        x = F.relu(self.conv1(x)) 
        #Convolutional Layer/Dropout/Pooling Layer/Activation
        x = (self.Dilation(x) + self.Erosion(x))/2
        x = x.view(-1, 10*24*24)
        #Fully Connected Layer/Activation
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        #Fully Connected Layer/Activation
        x = self.fc2(x)
        #Softmax gets probabilities. 
        return x #F.log_softmax(x, dim=1)
        
model=Net().to('cuda')

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

# %%
#@title Training the model
for epoch in range(args['epochs']):
    for i ,(images,labels) in enumerate(train_gen):
        images = images.to('cuda')
        labels = labels.to('cuda')
        images = Variable(images)
        #print(images.shape)
        labels = Variable(labels)
        #print(labels.shape)
        optimizer.zero_grad()
        outputs = model(images)
        #print(outputs.shape)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
    print(epoch+1, args['epochs'], i+1 / len(train_data)//args['batch_size'], loss.data.item())

# %%
#@title Testing the model
correct = 0
total = 0
for images,labels in test_gen:
  images = Variable(images.to('cuda'))
  labels = labels.to('cuda')
  
  output = model(images)
  _, predicted = torch.max(output,1)
  correct += (predicted == labels).sum()
  total += labels.size(0)

print('Accuracy of the model: %.3f %%' %((100*correct)/(total+1)))

# %%
#@title Visualizing the kernels
W=model.Dilation.kernel.cpu().detach().numpy()
print(W.shape)
plt.plot(W[0,:,:,0,0])
plt.show()

W=model.Erosion.kernel.cpu().detach().numpy()
print(W.shape)
plt.plot(W[0,:,:,0,0])
plt.show()