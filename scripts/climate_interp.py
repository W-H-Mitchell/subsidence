import os
import numpy as np
from osgeo import gdal

os.chdir("tifs/climate_data")

def InterpolateBetweenRasters(raster1, raster2, deltaT, interval, out_fp):
    ds = gdal.Open(raster1)
    geo = ds.GetGeoTransform()
    arr1 = ds.ReadAsArray()
    arr2 = gdal.Open(raster2).ReadAsArray()
    nodata =ds.GetRasterBand(1).GetNoDataValue()
    arr2[arr2==nodata]=np.nan
    arr1[arr1==nodata]=np.nan
    diff = arr2 - arr1
    grad = diff / deltaT
    test = grad * interval
    out = np.add(arr1,test)


    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.CreateCopy(out_fp, ds, strict=0)
    dst_ds.GetRasterBand(1).WriteArray(out)
    dst_ds.SetGeoTransform(geo)
    band = dst_ds.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    band = None
    dst_ds = None


InterpolateBetweenRasters("rcp85_model_2030-2039_winter-summer_rainfall.tif",
                          "rcp85_model_2060-2069_winter-summer_rainfall.tif",
                          30, 30,
                          "rcp85_model_2060-2069_winter-summer_rainfall_linearinterp.tif")
"""
InterpolateBetweenRasters("rcp85_model_2030-2039_winter-summer_rainfall.tif",
                          "rcp85_model_2060-2069_winter-summer_rainfall.tif",
                          30., 5., "rcp85_model_2035-2044_winter-summer_rainfall_lin.tif")
InterpolateBetweenRasters("rcp85_model_2030-2039_winter-summer_rainfall.tif",
                          "rcp85_model_2060-2069_winter-summer_rainfall.tif",
                          30., 15., "rcp85_model_2045-2054_winter-summer_rainfall_lin.tif")
InterpolateBetweenRasters("rcp85_model_2030-2039_winter-summer_rainfall.tif",
                          "rcp85_model_2060-2069_winter-summer_rainfall.tif",
                          30., 25., "rcp85_model_2055-2064_winter-summer_rainfall_lin.tif")
InterpolateBetweenRasters("rcp85_model_2030-2039_summer_tas.tif",
                          "rcp85_model_2060-2069_summer_tas.tif",
                          30., 5., "rcp85_model_2035-2044_summer_tas_lin.tif")
InterpolateBetweenRasters("rcp85_model_2030-2039_summer_tas.tif",
                          "rcp85_model_2060-2069_summer_tas.tif",
                          30., 15., "rcp85_model_2045-2054_summer_tas_lin.tif")
InterpolateBetweenRasters("rcp85_model_2030-2039_summer_tas.tif",
                          "rcp85_model_2060-2069_summer_tas.tif",
                          30., 25., "rcp85_model_2055-2064_summer_tas_lin.tif")
"""
