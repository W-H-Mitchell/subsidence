#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 12:22:34 2021

@author: whamitchell
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
#%matplotlib qt

datasources = sorted(glob.glob("rcp85outputs/*.tif"))
print(len(datasources))

def create_array(ds):
    data = gdal.Open(ds).ReadAsArray()
    data = data[data>-1]
    data = data.flatten()
    mean = data.mean()
    print(ds)
    print("Mean: {0} \nMedian: {1}\nQuantile: {2}".format(data.mean(), np.median(data), np.quantile(data, 0.25)))
    print('-'*40)
    return data, mean

_2015_2024, _2020mean = create_array(datasources[0])
_2020_2029, _2025mean = create_array(datasources[1])
_2025_2034, _2030mean = create_array(datasources[2])
_2030_2039, _2035mean = create_array(datasources[3])
_2035_2044, _2040mean = create_array(datasources[4])
_2055_2064, _2060mean = create_array(datasources[5])
_2060_2069, _2065mean = create_array(datasources[6])
_2065_2074, _2070mean = create_array(datasources[7])
_2070_2079, _2075mean = create_array(datasources[8])
