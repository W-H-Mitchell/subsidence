# IMPORT PACKAGES
import os
from osgeo import gdal
#import xarray as xr
#import rioxarray as rxr
import matplotlib.pyplot as plt

# Check the pwd is correct
os.chdir("Documents/aws/subsidence")

def cdf_tiff(ncdf, lyr, tif):
    cmd = "gdalwarp -overwrite -of  GTiff NETCDF:'{0}':{1} {2}".format(ncdf, lyr, tif)
    os.system(cmd)
    
def reproj_climate(ncdf, lyr, tif):
    # dem reference raster
    reference_raster = gdal.Open("tifs/final_dem.tif") # raster to use as reference array
    geo = reference_raster.GetGeoTransform()
    xsize, ysize = reference_raster.RasterXSize, reference_raster.RasterYSize
    x_res = geo[1]
    y_res = geo[5]
    min_x = geo[0]
    min_y = geo[3] + xsize*geo[4] + ysize*geo[5] 
    max_x = geo[0] + xsize*geo[1] + ysize*geo[2]
    max_y = geo[3] 
    
    cmd = f"gdalwarp -overwrite -of  GTiff -t_srs 'epsg:4326' -tr {x_res} {y_res} -te {min_x} {min_y} {max_x} {max_y} NETCDF:'{ncdf}':{lyr} {tif}"
    os.system(cmd)

def interpolate_rasters(tif, outtif):
    # dem reference raster
    reference_raster = gdal.Open("tifs/uk_dem_wgs84_0.0008.tif")
    geo = reference_raster.GetGeoTransform()
    xsize, ysize = reference_raster.RasterXSize, reference_raster.RasterYSize
    x_res = geo[1]
    y_res = geo[5]
    min_x = geo[0]
    min_y = geo[3] + xsize * geo[4] + ysize * geo[5]
    max_x = geo[0] + xsize * geo[1] + ysize * geo[2]
    max_y = geo[3]
    cmd = "gdalwarp -overwrite -of  GTiff -r bilinear -t_srs 'epsg:4326' -tr {0} {1} -te {2} {3} {4} {5} {6} {7}".format(x_res, y_res, min_x, min_y, max_x, max_y, tif, outtif)
    os.system(cmd)

def add_rasters(in1, in2, in3, in4, out):
    cmd = "gdal_calc.py -A {0} -B {1} -C {2} -D {3} --outfile={4} --calc='A+B+C+D'".format(in1, in2, in3, in4, out)
    os.system(cmd)
    
# open the netcdf and check variable layer name
#nc1 = rxr.open_rasterio("netcdfs/rcp85_1981-2000_winter-summer_verification.nc") #tas
#nc2 = rxr.open_rasterio("netcdfs/obs_winter-summer_total_rainfall_2017-2018.nc") #unknown

"""
# temperature
reproj_climate("netcdfs/obs_summer_mean_tas_2018.nc", "tas", "tifs/train2018/summer_temperature.tif")
reproj_climate("netcdfs/rcp85_model_1981-2000_summer_tas.nc", "tas", "tifs/rcp85_baseline2020_tas.tif")
reproj_climate("netcdfs/rcp85_model_2020-2029_summer_tas.nc", "tas", "tifs/rcp85_model_2020-2029_summer_tas.tif")
reproj_climate("netcdfs/rcp85_model_2025-2034_summer_tas.nc", "tas", "tifs/rcp85_model_2025-2034_summer_tas.tif")
reproj_climate("netcdfs/rcp85_model_2030-2039_summer_tas.nc", "tas", "tifs/rcp85_model_2030-2039_summer_tas.tif")
reproj_climate("netcdfs/rcp85_model_2035-2044_summer_tas.nc", "tas", "tifs/rcp85_model_2035-2044_summer_tas.tif")
reproj_climate("netcdfs/rcp85_model_2040-2049_summer_tas.nc", "tas", "tifs/rcp85_model_2040-2049_summer_tas.tif")
reproj_climate("netcdfs/rcp85_model_2045-2054_summer_tas.nc", "tas", "tifs/rcp85_model_2045-2054_summer_tas.tif")
reproj_climate("netcdfs/rcp85_model_2050-2059_summer_tas.nc", "tas", "tifs/rcp85_model_2050-2059_summer_tas.tif")
reproj_climate("netcdfs/rcp85_model_2055-2064_summer_tas.nc", "tas", "tifs/rcp85_model_2055-2064_summer_tas.tif")
reproj_climate("netcdfs/rcp85_model_2060-2069_summer_tas.nc", "tas", "tifs/rcp85_model_2060-2069_summer_tas.tif")
reproj_climate("netcdfs/rcp85_model_2065-2074_summer_tas.nc", "tas", "tifs/predict/rcp85_model_2065-2074_summer_tas.tif")
reproj_climate("netcdfs/rcp85_model_2070-2079_summer_tas.nc", "tas", "tifs/predict/rcp85_model_2070-2079_summer_tas.tif")
reproj_climate("netcdfs/rcp85_model_2075-2084_summer_tas.nc", "tas", "tifs/predict/rcp85_model_2075-2084_summer_tas.tif")
reproj_climate("netcdfs/rcp85_model_2075-2084_summer_tas.nc", "tas", "tifs/predict/rcp85_model_2075-2084_summer_tas.tif")
"""
reproj_climate("netcdfs/rcp85_1981-2000_winter-summer_verification.nc", "unknown", "tifs/rainfall_verification.tif")

"""
# rainfall
reproj_climate("netcdfs/rcp85_model_1981-2000_winter-summer_rainfall.nc", 
               "unknown", "tifs/predict/rcp85_baseline2020_rainfall.tif")
reproj_climate("netcdfs/rcp85_model_2020-2029_winter-summer_rainfall.nc",
               "unknown", "tifs/predict/rcp85_model_2020-2029_winter-summer_rainfall.tif")
reproj_climate("netcdfs/rcp85_model_2025-2034_winter-summer_rainfall.nc",
               "unknown", "tifs/rcp85_model_2025-2034_winter-summer_rainfall.tif")
reproj_climate("netcdfs/rcp85_model_2030-2039_winter-summer_rainfall.nc",
               "unknown", "tifs/rcp85_model_2030-2039_winter-summer_rainfall.tif")
reproj_climate("netcdfs/rcp85_model_2035-2044_winter-summer_rainfall.nc",
               "unknown", "tifs/rcp85_model_2035-2044_winter-summer_rainfall.tif")
reproj_climate("netcdfs/rcp85_model_2040-2049_winter-summer_rainfall.nc",
               "unknown", "tifs/rcp85_model_2040-2049_winter-summer_rainfall.tif")
reproj_climate("netcdfs/rcp85_model_2045-2054_winter-summer_rainfall.nc",
               "unknown", "tifs/predict/rcp85_model_2045-2054_winter-summer_rainfall.tif")
reproj_climate("netcdfs/rcp85_model_2050-2059_winter-summer_rainfall.nc",
               "unknown", "tifs/predict/rcp85_model_2050-2059_winter-summer_rainfall.tif")
reproj_climate("netcdfs/rcp85_model_2055-2064_winter-summer_rainfall.nc",
               "unknown", "tifs/predict/rcp85_model_2055-2064_winter-summer_rainfall.tif")
reproj_climate("netcdfs/rcp85_model_2060-2069_winter-summer_rainfall.nc",
               "unknown", "tifs/predict/rcp85_model_2060-2069_winter-summer_rainfall.tif")
reproj_climate("netcdfs/rcp85_model_2065-2074_winter-summer_rainfall.nc",
               "unknown", "tifs/predict/rcp85_model_2065-2074_winter-summer_rainfall.tif")
reproj_climate("netcdfs/rcp85_model_2070-2079_winter-summer_rainfall.nc",
               "unknown", "tifs/predict/rcp85_model_2070-2079_winter-summer_rainfall.tif")
reproj_climate("netcdfs/rcp85_model_2075-2084_winter-summer_rainfall.nc",
               "unknown", "tifs/predict/rcp85_model_2075-2084_winter-summer_rainfall.tif")
"""
reproj_climate("netcdfs/rcp85_1981-2000_winter-summer_verification.nc", "unknown", 
               "tifs/temperature_verification.tif")

# soil
#cdf_tiff("netcdfs/rcp85_mod_summer_mean_soil_moisture_winter-summer_2018_layer_0.nc",
#            "unknown", "tifs/train2018/soilmoisture_lyr0.tif")
#cdf_tiff("netcdfs/rcp85_mod_summer_mean_soil_moisture_winter-summer_2018_layer_1.nc",
#            "unknown", "tifs/train2018/soilmoisture_lyr1.tif")
#cdf_tiff("netcdfs/rcp85_mod_summer_mean_soil_moisture_winter-summer_2018_layer_2.nc",
#            "unknown", "tifs/train2018/soilmoisture_lyr2.tif")
#cdf_tiff("netcdfs/rcp85_mod_summer_mean_soil_moisture_winter-summer_2018_layer_3.nc",
#            "unknown", "tifs/train2018/soilmoisture_lyr3.tif")
#reproj_soil("netcdfs/rcp85_model_2015-2024_winter-summer_soil_moisture_all_layer_sum.nc",
#            "unknown", "tifs/predict/rcp85_model_2015-2024_winter-summer_soil.tif")
#reproj_soil("netcdfs/rcp85_model_2020-2029_winter-summer_soil_moisture_all_layer_sum.nc",
#            "unknown", "tifs/predict/rcp85_model_2020-2029_winter-summer_soil.tif")
#reproj_soil("netcdfs/rcp85_model_2025-2034_winter-summer_soil_moisture_all_layer_sum.nc",
#            "unknown", "tifs/predict/rcp85_model_2025-2034_winter-summer_soil.tif")
#reproj_soil("netcdfs/rcp85_model_2030-2039_winter-summer_soil_moisture_all_layer_sum.nc",
#            "unknown", "tifs/predict/rcp85_model_2030-2039_winter-summer_soil.tif")
#reproj_soil("netcdfs/rcp85_model_2035-2044_winter-summer_soil_moisture_all_layer_sum.nc",
#            "unknown", "tifs/predict/rcp85_model_2035-2044_winter-summer_soil.tif")
#reproj_soil("netcdfs/rcp85_model_2040-2049_winter-summer_soil_moisture_all_layer_sum.nc",
#            "unknown", "tifs/predict/rcp85_model_2040-2049_winter-summer_soil.tif")
#reproj_soil("netcdfs/rcp85_model_2045-2054_winter-summer_soil_moisture_all_layer_sum.nc",
#            "unknown", "tifs/predict/rcp85_model_2045-2054_winter-summer_soil.tif")
#reproj_soil("netcdfs/rcp85_model_2050-2059_winter-summer_soil_moisture_all_layer_sum.nc",
#            "unknown", "tifs/predict/rcp85_model_2050-2059_winter-summer_soil.tif")
#reproj_soil("netcdfs/rcp85_model_2055-2064_winter-summer_soil_moisture_all_layer_sum.nc",
#            "unknown", "tifs/predict/rcp85_model_2055-2064_winter-summer_soil.tif")
#reproj_soil("netcdfs/rcp85_model_2060-2069_winter-summer_soil_moisture_all_layer_sum.nc",
#            "unknown", "tifs/predict/rcp85_model_2060-2069_winter-summer_soil.tif")
#reproj_soil("netcdfs/rcp85_model_2065-2074_winter-summer_soil_moisture_all_layer_sum.nc",
#            "unknown", "tifs/predict/rcp85_model_2065-2074_winter-summer_soil.tif")