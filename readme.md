
# COVID-19 Anomaly Segmentation        
<p align="justify" markdown="1">
Naveen Paluru, Aveen Dayal, Havard B. Jenssen, Tomas Sakinis, Linga R. Cenkeramaddi, Jaya Prakash, and Phaneendra K. Yalavarthy, "Anam-Net : Anamorphic Depth Embedding based Light-Weight CNN for Segmentation of Anomalies in COVID-19 Chest CT Images," IEEE Transactions on Neural Networks and Learning Systems (Fast Track: COVID-19 Focused Papers), 32(3), 932-946 (2021). 
</p>
<a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9349153&tag=1">[manuscript]</a>

### data preparation

<p align="justify" markdown="1">
This folder contain all the necessary details of pre-processing and annotations of the COVID-19 infection. Please go through the readme file in this folder for further details.
</p>

## python files
```md
trnEXP-1.py - train file experiment 1

trnEXP-2.py - train file experiment 2

anamNet.py has the  model definition 

myDataset.py is the the data iterator

```
## CovSeg

This folder contains Android Application Details. Use mobiletorch.py to convert the trained model to its lite version. The android app (CovSeg) was developed in Android Studio.

### Disclaimer
<div class="red">
  The software and applications developed are not intended, nor should they be construed, as claims that this can be used to diagnose,treat, mitigate, cure, prevent or otherwise be used for any disease or medical condition. The software/application has not been clinically proven or evaluated.
</div>

##
<p align="center">
  <img src="https://github.com/NaveenPaluru/Segmentation-COVID-19/blob/master/CovSeg.gif" width="350" height="700">
</p>



## Segmentation Results
<p align="center">
  <img src="https://github.com/NaveenPaluru/Segmentation-COVID-19/blob/master/finalresults.png">
</p>


#### Any query, please raise an issue or contact :

*Dr. Phaneendra  K. Yalavarthy* 

*Assoc.Prof, CDS, IISc Bangalore, email : yalavarthy@iisc.ac.in*

*Naveen Paluru*

*(PhD) CDS, MIG, IISc Bangalore,  email : naveenp@iisc.ac.in*

