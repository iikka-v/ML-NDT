# ML-NDT
Data and code for training deep convolutional neural network to detect cracks in phased-array ultrasonic data.

Please refer to https://arxiv.org/abs/1903.11399 for details. 

## Contents
The directory "data" contains ultrasonic data sets, containing various flaws. 

The directory "src" contains python code to train a deep CNN using the data provided. Use "make train" to run. 

To make inference, you can consult the sample code in src/inference.py.

