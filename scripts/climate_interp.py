import os
from osgeo import gdal

os.chdir("Documents/aws/subsidence/tifs/climate_data")
def InterpolateBetweenRasters(raster1, raster2, deltaT, interval, output_fp):
    ds = gdal.Open(raster1)
    geo = ds.GetGeoTransform()
    arr1 = ds.ReadAsArray()
    arr2 = gdal.Open(raster2).ReadAsArray()
    diff = arr2 - arr1
    grad = diff / deltaT
    out = arr1 + (grad * interval)

    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.CreateCopy(output_fp, ds1, strict=0,
                               options=['COMPRESS=DEFLATE','BIGTIFF=YES','BLOCKXSIZE=512',
                                        'PREDICTOR=3','TILED=YES','NUM_THREADS=ALL_CPUS',
                                        'SPARSE_OK=TRUE','INTERLEAVE=PIXEL'])
    dst_ds.GetRasterBand(1).WriteArray(out)
    dst_ds.SetGeoTransform(geo)
    band = dst_ds.GetRasterBand(1)
    band.SetNoDataValue(-9999)

InterpolateBetweenRasters("rcp85_model_2030-2039_winter-summer_rainfall.tif",
                          "rcp85_model_2060-2069_winter-summer_rainfall.tif",
                          30, 5, "rcp85_model_2035-2044_winter-summer_rainfall_lin.tif")
InterpolateBetweenRasters("rcp85_model_2030-2039_winter-summer_rainfall.tif",
                          "rcp85_model_2060-2069_winter-summer_rainfall.tif",
                          30, 15, "rcp85_model_2045-2054_winter-summer_rainfall_lin.tif")
InterpolateBetweenRasters("rcp85_model_2030-2039_winter-summer_rainfall.tif",
                          "rcp85_model_2060-2069_winter-summer_rainfall.tif",
                          30, 25, "rcp85_model_2055-2064_winter-summer_rainfall_lin.tif")
