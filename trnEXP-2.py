
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:07:48 2020

@author: naveenpaluru
"""

import os,time
import sklearn.metrics as metrics
import scipy.io
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import tqdm
import torch.nn as nn
from datetime import datetime
from config import Config
from mydataset import myDataset
from anamnet import AnamNet
import torch.optim as optim
from torch.optim import lr_scheduler


print ('*******************************************************')
start_time=time.time()
saveDir='savedModels/'
cwd=os.getcwd()
directory=saveDir+datetime.now().strftime("%d%b_%I%M%P_")+'model'
print('Model will be saved to  :', directory)

if not os.path.exists(directory):
    os.makedirs(directory)

config  = Config()

# Training Data

data1 = scipy.io.loadmat('set2.mat')
inp1  = data1['inp']
lab1  = data1['lab']
traininp1 =np.reshape(np.transpose(inp1,(2,0,1)),(694,512,512,1))  # set2.mat has 694 slices
trainlab1 =np.transpose(lab1,(2,0,1))
data2 = scipy.io.loadmat('set3.mat')
inp2  = data2['inp']
lab2  = data2['lab']
traininp2 =np.reshape(np.transpose(inp2,(2,0,1)),(740,512,512,1))  # set3.mat has 740 slices
trainlab2 =np.transpose(lab2,(2,0,1))
traininp = np.concatenate((traininp1,traininp2), axis = 0)
trainlab = np.concatenate((trainlab1,trainlab2), axis = 0)

# Validation Data

data3 = scipy.io.loadmat('set1.mat')
Valinp  = data3['inp']
Vallab  = data3['lab']
Valinp =np.reshape( np.transpose(Valinp,(2,0,1)),(871,512,512,1))  # set1.mat has 871 slices
Vallab =np.transpose(Vallab,(2,0,1))
       

transform = transforms.Compose([transforms.ToTensor()])

# make the data iterator for training data
train_data = myDataset(traininp,trainlab, transform)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=config.batchsize, shuffle=True, num_workers=2)

val_data  = myDataset(Valinp, Vallab, transform)
valloader = torch.utils.data.DataLoader(val_data, batch_size=config.batchsize, shuffle=True, num_workers=2)

# weights for the class labels

labels = trainlab.flatten()
class_count = np.bincount(labels, minlength=3)
propensity_score = class_count/labels.size 
class_weights = 1 / propensity_score 

print('----------------------------------------------------------')
#%%
# Create the object for the network

if config.gpu == True:    
    net = AnamNet()
    net.cuda(config.gpuid)
    class_weights = torch.FloatTensor(class_weights).cuda(config.gpuid)    
else:
   net = AnamNet()
   
# Define the optimizer
optimizer = optim.Adam(net.parameters(),lr=5e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=33, gamma=0.1)

# Define the loss function
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Iterate over the training dataset
train_loss = []
val_loss = []

for j in range(config.epochs):  
    # Start epochs   
    runtrainloss = 0
    net.train() 
    for i,data in tqdm.tqdm(enumerate(trainloader)): 
        # start iterations
        images,trainLabels = Variable(data[0]),Variable(data[1])
        
        # ckeck if gpu is available
        if config.gpu == True:
            images  = images.cuda(config.gpuid )
            trainLabels = trainLabels.cuda(config.gpuid)
                    
        # make forward pass      
        output = net(images)
       
        #compute loss
        loss   = criterion(output, trainLabels)        
                
        # make gradients zero
        optimizer.zero_grad()
        
        # back propagate
        loss.backward()
        
        # Accumulate loss for current minibatch
        runtrainloss += loss.item()        
        
        # update the parameters
        optimizer.step()

    print('Training - Epoch {}/{}, loss:{:.4f} '.format(j+1, config.epochs, runtrainloss/len(trainloader)))
    
    runvalloss = 0
    net.eval()
    for i,data in tqdm.tqdm(enumerate(valloader)): 
        # start iterations
        images,Labels = Variable(data[0]),Variable(data[1])
        
        # ckeck if gpu is available
        if config.gpu == True:
            images  = images.cuda(config.gpuid )
            Labels =  Labels.cuda(config.gpuid)
                    
        # make forward pass      
        output = net(images)
       
        #compute loss
        loss   = criterion(output, Labels)                
             
        # Accumulate loss for current minibatch
        runvalloss += loss.item()
        
          
    # print loss after every epoch    
   
    print('Validatn - Epoch {}/{}, loss:{:.4f} '.format(j+1, config.epochs, runvalloss/len(valloader)))
    print('----------------------------------------------------------')

    train_loss.append(runtrainloss/len(trainloader))
    val_loss.append(runvalloss/len(valloader))     
    
    #save the model  
    torch.save(net.state_dict(),os.path.join(directory,"AnamNet_" + str(j+1) +"_model.pth"))            

# Save the train stats

np.save(directory+'/trnloss.npy',np.array(train_loss) )
np.save(directory+'/valloss.npy',np.array(val_loss) )

# plot the training loss

x = range(config.epochs)
plt.figure()
plt.plot(x,train_loss,label='Training')
plt.plot(x,val_loss,label='Validation')
plt.xlabel('epochs')
plt.ylabel('Train Loss ') 
plt.legend(loc="upper left")  
plt.show()                


