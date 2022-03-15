# IMPORT PACKAGES
import os
from osgeo import gdal
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt

# Check the pwd is correct
#os.chdir("Documents/aws/subsidence")

def cdf_tiff(ncdf, lyr, tif):
    cmd = "gdalwarp -overwrite -of  GTiff NETCDF:'{0}':{1} {2}".format(ncdf, lyr, tif)
    os.system(cmd)
    
def reproj_climate(ncdf, lyr, tif):
    # dem reference raster
    reference_raster = gdal.Open("tifs/uk_dem_wgs84_0.0008.tif") # raster to use as reference array
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

ds = xr.open_dataset("/Users/whamitchell/Downloads/twi/ga2.nc")
print(ds)

# function call
reproj_climate("/Users/whamitchell/Downloads/twi/ga2.nc", "Band1", "/Users/whamitchell/Documents/aws/subsidence/tifs/twi.tif")
