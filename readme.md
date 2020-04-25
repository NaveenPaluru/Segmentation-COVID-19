
# Segmentation of the Abnormalities in  COVID-19-CT Images  by  Cost Sensitive Convolutional Neural Networks (CNNs)

Folders and Files Descritions

## data preparation

This folder contains two files : **dataPrepCTSlice.m**  for preparing training data and test set 1, **dataPrepCTVolume.m**
for preparing test set 2. These files contain all necessary references for the datasets.

Dataset : [Link](http://medicalsegmentation.com/covid19/)

## savedModels

Download the trained models from the google drive link below and place it inside the subfolder in the savedModels. Trained model are [here](https://drive.google.com/open?id=1wm3m-0Upjk6g8jxnNEIBWK686kf2SJZm) 


## results

This folder contains the FOM of UNet and ENet.

## python files

**trnENet+.py** is the run file to train the ENet+ model. **tstENet+.py** is the testing file for evaluation with ENet+, **enet.py** has model definition for enet and **myDataset.py** is data iterator.

## Quantitative Performance of UNet

Confusion Matrix Showing the Performance of UNet on Test Set 2 : 704 Slices
<p align="center">
  <img src="https://github.com/NaveenPaluru/Segmentation-COVID-19/blob/master/results/test1VOL.png">
</p>

## Quantitative Performance of ENet

Confusion Matrix Showing the Performance of ENet on Test Set 2 : 704 Slices
<p align="center">
  <img src="https://github.com/NaveenPaluru/Segmentation-COVID-19/blob/master/results/testVOL3.1.png">
</p>

## Quantitative Performance of ENet+ (ours)

Confusion Matrix Showing the Performance of ENet on Test Set 2 : 704 Slices
<p align="center">
  <img src="https://github.com/NaveenPaluru/Segmentation-COVID-19/blob/master/results/testVOL3.png">
</p>

## Qualitative Performance of UNet and ENet+

<p align="center">
  <img src="https://github.com/NaveenPaluru/Segmentation-COVID-19/blob/master/results/Visual.png">
</p>


#### Any query, please raise an issue or contact :

*Dr. Phannendra  K. Yalavarthy* 

*Assoc.Prof, CDS, IISc Bangalore, email : yalavarthy@iisc.ac.in*

*Naveen Paluru*

*(PhD) CDS, MIG, IISc Bangalore,  email : naveenp@iisc.ac.in*

#### References
 1. [UNet](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)
 2. [ENet](https://arxiv.org/abs/1606.02147)
