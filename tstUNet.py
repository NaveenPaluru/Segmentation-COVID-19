#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:04:43 2020

@author: naveenpaluru
"""


import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import tqdm
from config import Config
from mydataset import myDataset
#import h5py as h5
from model import UNet
import torch.nn.functional as F
import sklearn.metrics as metrics
import seaborn as sn
import pandas  as pd
import scipy


def test(directory):
    
    net = UNet()
    net.load_state_dict(torch.load(directory))
    
    config  = Config()
    
    # load the  data here .....................................
    # load the testimages and test labels in form of numpy arrays
    
    data = scipy.io.loadmat('testVOL.mat')
    inp  = data['IMG']
    lab  = data['LAB']
    
    # 704 be replaced by 30 for test.mat
    testinp =np.reshape( np.transpose(inp,(2,0,1)),(704,512,512,1))
    testlab =np.reshape( np.transpose(lab,(2,0,1)),(704,512,512,1))
    
    transform = transforms.Compose([transforms.ToTensor(),
                ])        
    
    # make the data iterator for testing data . 
    test_data = myDataset(testinp, testlab, transform)
    testloader  = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)   
    
           
    if config.gpu == True:
        net.cuda(config.gpuid).eval()   
        
    
    for i,data in tqdm.tqdm(enumerate(testloader)):             
        # start iterations
        images,imtruth = Variable(data[0]),Variable(data[1])
        # ckeck if gpu is available
        if config.gpu == True:
            images  = images.cuda( config.gpuid)
            imtruth = imtruth.cuda(config.gpuid)
        # make forward pass      
        output = F.softmax(net(images),dim=1)
        _, pred= torch.max(output,dim=1)  
        if i==0:
            tmp = pred.cpu().detach()
            tmpl=imtruth.cpu().detach()
        else:
            tmp = torch.cat((tmp ,pred.cpu().detach()),dim=0)
            tmpl= torch.cat((tmpl,imtruth.cpu().detach()),dim=0)     
            
    return tmp,tmpl.squeeze()
                
            
           
if __name__ == '__main__':         
        
    saveDir='./savedModels/'
        
    # if want to test on a specific model
    directory=saveDir+"21Apr_1247am_model/"+ "UNet_100_model.pth"
    print('Loading the Model : ', directory)
    
    tmp, tmpl = test(directory)
    tmp  = tmp.numpy().astype(np.float64 )  
    tmpl = tmpl.numpy().astype(np.float64)  
    #scipy.io.savemat("test_pred.mat", {'pred':tmp })
    #scipy.io.savemat("test_labl.mat", {'labl':tmpl})
    
    hh=np.reshape(tmp, (-1,1))      
    gg=np.reshape(tmpl,(-1,1))        
    matrix = metrics.confusion_matrix(hh,  gg)
    matrix= matrix / matrix.astype(np.float).sum(axis=0)
    df_cm = pd.DataFrame(matrix, index = ['Background', 'Abnormal', 'Normal'],
                               columns = ['Background', 'Abnormal', 'Normal'])
    #plt.figure(figsize = (10,10))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},fmt = '.2f') # font sizesn.set(font_scale=1.4) # for label size
   
    
    plt.tight_layout()
    plt.xlabel('Target Class')    
    plt.ylabel('Output Class') 
    plt.show()
        
##    









































