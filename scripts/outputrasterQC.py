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
%matplotlib qt

datasources = sorted(glob.glob("tifs/*.tif"))

print(len(datasources))
print(len(rcp26))

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
_2065_2074, _2075mean = create_array(datasources[7])


r2015_2024 = create_array(rcp26[0])
r2020_2029 = create_array(rcp26[1])
r2025_2034 = create_array(rcp26[2])
r2030_2039 = create_array(rcp26[3])
r2035_2044 = create_array(rcp26[4])
r2050_2059 = create_array(rcp26[5])
r2055_2064 = create_array(rcp26[6])
r2060_2069 = create_array(rcp26[7])
r2065_2074 = create_array(rcp26[8])


ax = plt.subplots()
ax = plt.hist(_2015_2024, cumulative=True, density=True, histtype='step', label='2020')
#ax = plt.hist(_2020_2029, cumulative=True, density=True, histtype='step', label='2025')
#ax = plt.hist(_2025_2034, cumulative=True, density=True, histtype='step', label='2030')
#ax = plt.hist(_2030_2039, cumulative=True, density=True, histtype='step', label='2035')
#ax = plt.hist(_2035_2044, cumulative=True, density=True, histtype='step', label='2040')
#ax = plt.hist(_2055_2064, cumulative=True, density=True, histtype='step', label='2060')
#ax = plt.hist(_2060_2069, cumulative=True, density=True, histtype='step', label='2065')
ax = plt.hist(_2065_2074, cumulative=True, density=True, histtype='step', label='2070')
plt.legend()
plt.show()
fig = plt.gcf()
fig.savefig('rcp85.png')
