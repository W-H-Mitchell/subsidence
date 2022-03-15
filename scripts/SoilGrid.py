# imports
import os
import glob
import requests
import subprocess
from osgeo import gdal, ogr

# lat long transform
def lat_lon_transform(input_filepath, reference_filepath, output_filepath):
    reference = gdal.Open(reference_filepath)  # reads in the reference raster
    referenceSRS = reference.GetProjection()
    xsize, ysize = reference.RasterXSize, reference.RasterYSize
    geo = reference.GetGeoTransform()
    x_res, y_res = geo[1], geo[5]
    min_x = geo[0]
    min_y = geo[3] + xsize * geo[4] + ysize * geo[5]
    max_x = geo[0] + xsize * geo[1] + ysize * geo[2]
    max_y = geo[3]
    extent = (min_x, min_y, max_x, max_y)
    options = gdal.WarpOptions(resampleAlg=gdal.GRA_NearestNeighbour, xRes=x_res, yRes=y_res,
                               outputBounds=extent, srcSRS=referenceSRS,
                               creationOptions=['COMPRESS=DEFLATE', 'BIGTIFF=YES', 'BLOCKXSIZE=512',
                                                'TILED=YES', 'NUM_THREADS=ALL_CPUS',
                                                'SPARSE_OK=TRUE', 'INTERLEAVE=PIXEL'])
    gdal.Warp(output_filepath, input_filepath, options=options)

def add_rasters(in1, in2, in3, in4, out):
    cmd = f"gdal_calc.py -A {in1} -B {in2} -C {in3} -D {in4} --outfile={out} --calc='A+B+C+D'"
    os.system(cmd)
"""
# downloading the soil type grid
k = 0
for lat in range(40, 60):
    for long in range(-20, 20):
        lat_max = lat+2
        long_max = long + 2
        os.chdir("/Users/whamitchell/Documents/aws/subsidence/clay/claytype")
        url = f"https://maps.isric.org/mapserv?map=/map/wrb.map&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage&COVERAGEID=MostProbable&FORMAT=image/tiff&SUBSET=long({long},{long_max})&SUBSET=lat({lat},{lat_max})&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
        print(url)
        with open(f'out_{k}.tif', 'wb') as out_file:
            content = requests.get(url, stream=True).content
            out_file.write(content)
        k+=1
filenames = glob.glob('*.tif')
cmd = "gdal_merge.py -o mergedRaster.tif"
subprocess.call(cmd.split()+filenames)

# download the soil grid percentage
os.chdir("/Users/whamitchell/Documents/aws/subsidence/clay/claypct")
# 0 - 5 cm clay percentage
k = 0
for lat in range(40, 60):
    for long in range(-20, 20):
        lat_max = lat+2
        long_max = long + 2
        url = f"https://maps.isric.org/mapserv?map=/map/clay.map&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage&COVERAGEID=clay_0-5cm_mean&FORMAT=image/tiff&SUBSET=long({long},{long_max})&SUBSET=lat({lat},{lat_max})&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
        print(url)
        with open(f'out_{k}.tif', 'wb') as out_file:
            content = requests.get(url, stream=True).content
            out_file.write(content)
        k+=1
filenames = glob.glob('*.tif')
cmd = f"gdal_merge.py -o claypct_clay_0-5cm_mean.tif"
subprocess.call(cmd.split() + filenames)

# 5 - 15 cm clay percentage
k = 0
for lat in range(40, 60):
    for long in range(-20, 20):
        lat_max = lat+2
        long_max = long + 2
        url = f"https://maps.isric.org/mapserv?map=/map/clay.map&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage&COVERAGEID=clay_5-15cm_mean&FORMAT=image/tiff&SUBSET=long({long},{long_max})&SUBSET=lat({lat},{lat_max})&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
        print(url)
        with open(f'out_{k}.tif', 'wb') as out_file:
            content = requests.get(url, stream=True).content
            out_file.write(content)
        k+=1
filenames = glob.glob('*.tif')
cmd = f"gdal_merge.py -o claypct_clay_5-15cm_mean.tif"
subprocess.call(cmd.split() + filenames)

# 15 - 30 cm clay percentage
k = 0
for lat in range(40, 60):
    for long in range(-20, 20):
        lat_max = lat+2
        long_max = long + 2
        url = f"https://maps.isric.org/mapserv?map=/map/clay.map&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage&COVERAGEID=clay_15-30cm_mean&FORMAT=image/tiff&SUBSET=long({long},{long_max})&SUBSET=lat({lat},{lat_max})&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
        print(url)
        with open(f'out_{k}.tif', 'wb') as out_file:
            content = requests.get(url, stream=True).content
            out_file.write(content)
        k+=1
filenames = glob.glob('*.tif')
cmd = f"gdal_merge.py -o claypct_clay_15-30cm_mean.tif"
subprocess.call(cmd.split() + filenames)

# 30 - 60 cm clay percentage
k = 0
for lat in range(40, 60):
    for long in range(-20, 20):
        lat_max = lat+2
        long_max = long + 2
        url = f"https://maps.isric.org/mapserv?map=/map/clay.map&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage&COVERAGEID=clay_30-60cm_mean&FORMAT=image/tiff&SUBSET=long({long},{long_max})&SUBSET=lat({lat},{lat_max})&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
        print(url)
        with open(f'out_{k}.tif', 'wb') as out_file:
            content = requests.get(url, stream=True).content
            out_file.write(content)
        k+=1
filenames = glob.glob('*.tif')
cmd = f"gdal_merge.py -o claypct_clay_30-60cm_mean.tif"
subprocess.call(cmd.split() + filenames)
# merging the soil grid files
add_rasters("clay/claypct/claypct_clay_0-5cm_mean.tif","clay/claypct/claypct_clay_5-15cm_mean.tif",
            "clay/claypct/claypct_clay_15-30cm_mean.tif","clay/claypct/claypct_clay_30-60cm_mean.tif",
            "clay/claypct/claypct_cum.tif")
lat_lon_transform("clay/claytype/mergedRaster.tif", "tifs/uk_dem_wgs84_0.0008.tif", "tifs/ClayTypes.tif")
lat_lon_transform("clay/claypct/claypct_cum.tif", "tifs/uk_dem_wgs84_0.0008.tif", "tifs/ClayPct.tif")
"""
lat_lon_transform("/Users/whamitchell/Documents/aws/inputs/tifs/superficial_encoded.tif",
                  "/Users/whamitchell/Documents/aws/inputs/tifs/uk_dem_wgs84_0.0008.tif",
                  "/Users/whamitchell/Documents/aws/inputs/tifs/superficial.tif")
lat_lon_transform("/Users/whamitchell/Documents/aws/inputs/tifs/geology.tif",
                  "/Users/whamitchell/Documents/aws/inputs/tifs/uk_dem_wgs84_0.0008.tif",
                  "/Users/whamitchell/Documents/aws/inputs/tifs/geology.tif")
