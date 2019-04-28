# ML-NDT
Data and code for training deep convolutional neural network to detect cracks in phased-array ultrasonic data.

Please refer to https://arxiv.org/abs/1903.11399 for details. 

## Contents
The directory "data" contains ultrasonic data sets, containing various flaws. Each batch file is named with an UUID and contains
    * .bins file, that contains the raw data
    * .meta file, that documents the raw data format, this is always UInt16, 256 x 256 x 100
    * .jsons file, that contains a json-formatted meta-data for each binary file. This includes the locations of all flaws, source flaw size and "equivalent size"
    * .labels file, that contain tab-separated data for flaw existence (0/1) and equivalent flaw size.

The directory "src" contains python code to train a deep CNN using the data provided. Use "make train" to run. 

To make inference, you can consult the sample code in src/inference.py.

