
# Anam-Net : Anamorphic Depth Embedding basedLight-Weight CNN for Segmentation of Anomaliesin COVID-19 Chest CT Images

Folders and Files Descritions

## data preparation

This folder contains two files : **dataPrepCTSlice.m**  for preparing training data and test set 1, **dataPrepCTVolume.m**
for preparing test set 2. These files contain all necessary references for the datasets.

Dataset : [Link](http://medicalsegmentation.com/covid19/)

## savedModels

Will be updated Soon.


## python files

**trnAnamNet.py** is the run file to train the ENet+ model. **tstAnamNet.py** is the testing file for evaluation, **AnamNet.py** has model definition and **myDataset.py** is data iterator. (Missing files will be updated soon.)

## Quantitative Performance of UNet


  <img src="https://github.com/NaveenPaluru/Segmentation-COVID-19/blob/master/results/ENet+ Test.png">
</p>



#### Any query, please raise an issue or contact :

*Dr. Phannendra  K. Yalavarthy* 

*Assoc.Prof, CDS, IISc Bangalore, email : yalavarthy@iisc.ac.in*

*Naveen Paluru*

*(PhD) CDS, MIG, IISc Bangalore,  email : naveenp@iisc.ac.in*

#### References
 1. [UNet](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)
 2. [ENet](https://arxiv.org/abs/1606.02147)
