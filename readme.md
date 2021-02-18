
# COVID-19 Anomaly Segmentation

Naveen Paluru, Aveen Dayal, Havard B. Jenssen, Tomas Sakinis, Linga R. Cenkeramaddi, Jaya Prakash, and Phaneendra K. Yalavarthy, ["Anam-Net : Anamorphic Depth Embedding based Light-Weight CNN for Segmentation of Anomalies in COVID-19 Chest CT Images,"](http://cds.iisc.ac.in/faculty/yalavarthy/Paluru_IEEETNNLS_2021.pdf) IEEE Transactions on Neural Networks and Learning Systems (Fast Track: COVID-19 Focused Papers) 2021 (in press).

Folders and Files Descritions

## data preparation

This folder contains two files : **dataPrepCTSlice.m**  for preparing training data (experiment 1 in manuscript) and ,**dataPrepCTVolume.m**
for preparing the test data ( for experiment 1 and experiment 3 in the manuscript). 

These files contain all necessary references for the datasets.

Dataset : [Link](http://medicalsegmentation.com/covid19/)

## CovSeg

This folder contains Android Application Details. Use mobiletorch.py to convert the trained model to its lite version. The app was developed in Android Studio.

### Disclaimer
<div class="red">
  The software and applications developed are not intended, nor should they be construed, as claims that this can be used to diagnose,treat, mitigate, cure, prevent or otherwise be used for any disease or medical condition. The software/application has not been clinically proven or evaluated.
</div>

##
<p align="center">
  <img src="https://github.com/NaveenPaluru/Segmentation-COVID-19/blob/master/CovSeg.gif" width="350" height="700">
</p>






## python files

**trnAnamNet.py** is the run file to train the model, **AnamNet.py** has model definition and **myDataset.py** is data iterator. 


## Segmentation Results
<p align="center">
  <img src="https://github.com/NaveenPaluru/Segmentation-COVID-19/blob/master/finalresults.png">
</p>


#### Any query, please raise an issue or contact :

*Dr. Phaneendra  K. Yalavarthy* 

*Assoc.Prof, CDS, IISc Bangalore, email : yalavarthy@iisc.ac.in*

*Naveen Paluru*

*(PhD) CDS, MIG, IISc Bangalore,  email : naveenp@iisc.ac.in*

#### References
[UNet](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28), [ENet](https://arxiv.org/abs/1606.02147), [UNet++](https://arxiv.org/abs/1807.10165),
[SegNet](https://arxiv.org/pdf/1511.00561.pdf), [Attention-UNet](https://arxiv.org/abs/1804.03999), [LedNet](https://arxiv.org/abs/1905.02423), [DeepLabV3+](https://arxiv.org/abs/1802.02611)
