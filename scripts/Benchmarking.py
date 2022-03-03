import numpy as np
from osgeo import ogr, gdal

# Clipping
def clip(shp, out_tif):
    # Path to reference file
    ref = gdal.Open("tifs/uk_dem_wgs84_0.0008.tif")  # reads in the reference raster
    xsize, ysize = ref.RasterXSize, ref.RasterYSize
    x_res = geo[1]
    y_res = geo[5]
    min_x = geo[0]
    min_y = geo[3] + xsize * geo[4] + ysize * geo[5]
    max_x = geo[0] + xsize * geo[1] + ysize * geo[2]
    max_y = geo[3]
    extent = [min_x, min_y, max_x, max_y]
    options = gdal.WarpOptions(resampleAlg=gdal.GRA_NearestNeighbour, xRes=x_res, yRes=y_res,
                               outputBounds=extent, cutlineDSName=shp, cropToCutline=True,
                               creationOptions=['COMPRESS=DEFLATE','BIGTIFF=YES','BLOCKXSIZE=512',
                                                'PREDICTOR=3','TILED=YES','NUM_THREADS=ALL_CPUS',
                                                'SPARSE_OK=TRUE','INTERLEAVE=PIXEL'])
    gdal.Warp(out_tif, "tifs/uk_dem_wgs84_0.0008.tif", options=options)



clip("tifs/eng&wales.shp", "tifs/eng_wales.tif")
clip("tifs/scotland.shp", "tifs/scotland.tif")
clip("tifs/ni.shp", "tifs/ni.tif")
