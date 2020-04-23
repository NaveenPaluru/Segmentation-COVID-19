
# Segmentation of Abnormalities in COVID-19-CT Images by UNet  

Folders and Files Descritions

## data preparation

This folder contains two files : **dataPrepCTSlice.m**  for preparing training data and test set 1, **dataPrepCTVolume.m**
for preparing test set 2. These files contain all necessary references for the datasets.

Dataset : [Link](http://medicalsegmentation.com/covid19/)

## savedModels

Download the trained model UNet_100_model.pth from the google drive link below and place it inside the subfolder in the savedModels. Trained model is [here](https://drive.google.com/open?id=1wm3m-0Upjk6g8jxnNEIBWK686kf2SJZm) 


## results

This folder contains the FOM of UNet.

## python files

% **trnUNet.py** is the run file to train the model. % **testUNet.py** is the testing file for evaluation % **config.py** is  configure file.% **model.py** has model definition and % **myDataset.py** is data iterator.

## Quantitative Performance of UNet

Confusion Matrix Showing the Performance of CSUNet on Test Set 2 : 704 Slices
<p align="center">
  <img src="https://github.com/NaveenPaluru/Segmentation-COVID-19/blob/master/results/test1VOL.png">
</p>

## Qualitative Performance of UNet

 Performance of CSUNet on one of the slices in Test Set 2 .
<p align="center">
  <img width = 500 height = 250 src="https://github.com/NaveenPaluru/Segmentation-COVID-19/blob/master/results/Visual.png">
</p>


#### Any difficulty, please raise an issue or contact :

*Dr. Phannendra  K. Yalavarthy* 

*Assoc.Prof, CDS, IISc Bangalore, email : yalavarthy@iisc.ac.in*

*Naveen Paluru*

*(PhD) CDS, MIG, IISc Bangalore,  email : naveenp@iisc.ac.in*

#### References
1.  O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in International Conference on Medical image computing and computer-assisted intervention. Springer, 2015, pp. 234–241.



