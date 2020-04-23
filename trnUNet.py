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
from model import UNet
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

# load the  data

data = scipy.io.loadmat('train.mat')
inp  = data['inp']
lab  = data['lab']

traininp =np.reshape( np.transpose(inp,(2,0,1)),(270,512,512,1))
trainlab =np.reshape( np.transpose(lab,(2,0,1)),(270,512,512,1))

transform = transforms.Compose([transforms.ToTensor(),
            ])

# make the data iterator for training data
train_data = myDataset(traininp,trainlab, transform)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=config.batchsize, shuffle=True, num_workers=2)

#

print('----------------------------------------------------------')
#%%
# Create the object for the network

if config.gpu == True:    
    net = UNet()
    net.cuda(config.gpuid)
    
else:
   net = UNet()
   
# Define the optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)
scheduler = lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.1)

# Define the loss function
criterion = nn.CrossEntropyLoss(reduction='none')


# Iterate over the training dataset
train_loss = []
sens = []
spec = []
acc  = []
img_rows = 512
img_cols = 512
numImgs  = 5 # should be same as mini batch size

for j in range(config.epochs):  
    # Start epochs   
    runtrainloss = 0
    net.train() 
    for i,data in tqdm.tqdm(enumerate(trainloader)): 
        # start iterations
        images,trainLabels = Variable(data[0]),Variable(data[1])
        ind2 = np.where(trainLabels==1)
        ind1 = np.where(trainLabels==0)
        ind3 = np.where(trainLabels==2)
        wImg = np.zeros(trainLabels.shape, dtype='uint8')
        w1 = ((len(ind1[0]))*1.0/(numImgs*img_rows*img_cols))+np.finfo('float').eps
        w2 = ((len(ind2[0]))*1.0/(numImgs*img_rows*img_cols))+np.finfo('float').eps
        w3 = ((len(ind3[0]))*1.0/(numImgs*img_rows*img_cols))+np.finfo('float').eps
        w1 = np.max([w1,w2,w3])/(w1+np.finfo('float').eps)
        w2 = np.max([w1,w2,w3])/(w2+np.finfo('float').eps)
        w3 = np.max([w1,w2,w3])/(w3+np.finfo('float').eps)
        if len(ind1)!=0:
	    # As background pixels are more, w1 will be very small, so give a small boost 
            wImg[ind1] = w1 * 10 
        if len(ind2)!=0:
            wImg[ind2] = w2     
        if len(ind3)!=0:
            wImg[ind3] = w3  
        # ckeck if gpu is available
        if config.gpu == True:
              images  = images.cuda(config.gpuid )
              imtruth = trainLabels.cuda(config.gpuid)
              wImg    = torch.FloatTensor(wImg).cuda(config.gpuid)
            
        # make forward pass      
        output = net(images)
       
        #compute loss per sample
        loss   = criterion(output, imtruth.squeeze())
	
        # Apply weights per sample 
        loss   = loss * wImg.squeeze()
        
        #compute acc
        out    = F.softmax(output,dim=1)
        _, pred= torch.max(out   ,dim=1)      
            
        # make gradients zero
        optimizer.zero_grad()
        
        # back propagate
        loss.mean().backward()
        
        # Accumulate loss for current minibatch
        runtrainloss += loss.mean().item()        
        
        # update the parameters
        optimizer.step()       
        
        if i==0:
            tmp = pred.cpu().detach()
            tmpl=imtruth.squeeze().cpu().detach()
        else:
            tmp = torch.cat((tmp ,pred.cpu().detach()),dim=0)
            tmpl= torch.cat((tmpl,imtruth.squeeze().cpu().detach()),dim=0)
            
    # These stats are only for abnormal and normal class. Background class is ignored.
    
    hh=np.reshape(tmp.numpy(), (-1,1))      
    gg=np.reshape(tmpl.numpy(),(-1,1))
    
    matrix = metrics.confusion_matrix(hh, gg)
    
    sens.append(matrix[1,1]/np.sum(matrix[:,1]))
    spec.append(matrix[2,2]/np.sum(matrix[:,2]))
    
    
    acc.append((matrix[1,1]+matrix[2,2])/(np.sum(matrix[:,1])+np.sum(matrix[:,2])))
    # print loss after every epoch
    
    print('Training - Epoch {}/{}, loss:{:.4f}, acc:{:.4f}, sens:{:.4f}, spec:{:.4f} '.format(j+1, 
                       config.epochs,  runtrainloss/len(trainloader),  acc[j],  sens[j], spec[j]))
    train_loss.append(runtrainloss/len(trainloader))
    
    #tmp  = tmp.numpy().astype(np.float64)  
    #tmpl = tmpl.numpy().astype(np.float64)  
        
    # Take a step for scheduler
    scheduler.step()    
    
    #save the model   
    torch.save(net.state_dict(),os.path.join(directory,"UNet_" + str(j+1) +"_model.pth"))
	    

# plot the training loss

x = range(config.epochs)
plt.figure()
plt.plot(x,train_loss,label='Training')
plt.xlabel('epochs')
plt.ylabel('Train Loss ') 
plt.legend(loc="upper left")  
plt.show()
plt.figure()
plt.plot(x,acc,label='acc')
plt.plot(x,sens,label='sens')
plt.plot(x,spec,label='spec')
plt.xlabel('epochs')
plt.ylabel('FOM ')  
plt.legend(loc="upper left") 
plt.show()
#plt.figure()
#plt.plot(x, vall_loss ,label='Validatn')
#plt.xlabel('epochs')
#plt.ylabel('Val Loss ')                                  
#plt.show()
#                                 


