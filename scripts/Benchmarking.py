import pandas as pd
import seaborn as sns
from osgeo import ogr, gdal
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

# Clipping
def clip(shp, out_tif):
    # Path to reference file
    ref = gdal.Open("tifs/uk_dem_wgs84_0.0008.tif")  # reads in the reference raster
    xsize, ysize = ref.RasterXSize, ref.RasterYSize
    geo = ref.GetGeoTransform()
    x_res, y_res = geo[1], geo[5]
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
    gdal.Warp(out_tif,"tifs/uk_dem_wgs84_0.0008.tif",options=options)

def reproject(inp,out):
    # Path to reference file
    ref = gdal.Open("tifs/uk_dem_wgs84_0.0008.tif")  # reads in the reference raster
    xsize, ysize = ref.RasterXSize, ref.RasterYSize
    geo = ref.GetGeoTransform()
    x_res = geo[1]
    y_res = geo[5]
    min_x = geo[0]
    min_y = geo[3] + xsize * geo[4] + ysize * geo[5]
    max_x = geo[0] + xsize * geo[1] + ysize * geo[2]
    max_y = geo[3]
    extent = [min_x, min_y, max_x, max_y]
    options = gdal.WarpOptions(resampleAlg=gdal.GRA_NearestNeighbour,xRes=x_res,yRes=y_res,outputBounds=extent,
                               creationOptions=['COMPRESS=DEFLATE','BIGTIFF=YES','BLOCKXSIZE=512',
                                                'PREDICTOR=3','TILED=YES','NUM_THREADS=ALL_CPUS',
                                                'SPARSE_OK=TRUE','INTERLEAVE=PIXEL'])
    gdal.Warp(out,inp,options=options)

def GetPercentageBands(mask, subsidence, plt_name):
    sub = gdal.Open(subsidence).ReadAsArray()
    msk = gdal.Open(mask).ReadAsArray()
    nan = msk <= -1000
    sub[nan] = -9999
    sub = sub[sub > -9999]
    sub.flatten()

    sub[sub < -1.05] = 6
    sub[(sub >= -1.05) & (sub < -0.85)] = 5
    sub[(sub >= -0.85) & (sub < -0.65)] = 4
    sub[(sub >= -0.65) & (sub < -0.41)] = 3
    sub[(sub >= -0.41) & (sub < -0.15)] = 2
    sub[(sub >= -0.15) & (sub <= 0.15)] = 1
    sub[(sub > 0.15) & (sub <= 0.41)] = 2

    df = pd.DataFrame({'Subsidence':sub})
    sns.countplot(x='Subsidence', data=df)
    plt.gcf()
    plt.savefig(plt_name)

    outdict = {'F':len(sub[sub==6]), 'E':len(sub[sub==5]), 'D':len(sub[sub==4]),
               'C':len(sub[sub==3]), 'B':len(sub[sub==2]), 'A':len(sub[sub==1])}
    outframe = pd.DataFrame(list(outdict.items()))
    return outframe

# Calculate mask
#clip("tifs/eng&wales.shp","tifs/eng_wales.tif")
#reproject("tifs/eng_wales.tif", "tifs/eng_wales1.tif")
#clip("tifs/scotland.shp","tifs/scotland.tif")
#reproject("tifs/scotland.tif", "tifs/scotland1.tif")
#clip("tifs/ni.shp","tifs/ni.tif")
#reproject("tifs/ni.tif", "tifs/ni1.tif")
clip("tifs/GBcutline.shp", "tifs/Britain.tif")
reproject("tifs/Britain.tif", "tifs/Britain.tif")

"""
# Baseline get percentages
en_wal = GetPercentageBands("tifs/eng_wales1.tif", "tifs/rcp85_2020_baseline.tif", "tifs/rcp85_EngWales_GetPercentages_baseline.png")
scot = GetPercentageBands("tifs/scotland1.tif", "tifs/rcp85_2020_baseline.tif", "tifs/rcp85_Scot_GetPercentages_baseline.png")
ni = GetPercentageBands("tifs/ni1.tif", "tifs/rcp85_2020_baseline.tif", "tifs/rcp85_NI_GetPercentages_baseline.png")

# 2050 get percentages
en_wal2050 = GetPercentageBands()
scot2050 = GetPercentageBands()
ni2050 = GetPercentageBands()
"""