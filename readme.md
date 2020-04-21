
# Segmentation of Abnormalities in COVID-19-CT Images by Cost Sensitive UNet (CSUNet)

Folders and Files Descritions

# data preparation

This folder contains two files : dataPrepCTSlice.m  for preparimg training data and test set 1 and  dataPrepCTVolume.m
for preparing test set 2. These files contain all necessary references for the datasets.

Dataset Link : http://medicalsegmentation.com/covid19/

# savedModels

Download the trained model CSUNet_100_model.pth from the google drive link below and place it inside the subfolder in the savedModels. https://drive.google.com/open?id=1ak-tFBpl0ulj3tigFhzLwQPAsP7c9kZs


# results

This folder contains the FOM of CSUNet.

# python files

% trn.py is the run file to train the model. % test.py is the testing file for evaluation % config.py is configure file.
% model.py has model definition and % myDataset.py is data iterator.

# Performance

Confusion Matrix Showing the Performance of CSUNet on Test Set 2 : 704 Slices
![test1VOL](https://github.com/NaveenPaluru/Segmentation-COVID-19/blob/master/results/test1VOL.png)



