import pandas as pd
import seaborn as sns
from osgeo import ogr, gdal
import glob
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
                                                'TILED=YES','NUM_THREADS=ALL_CPUS',
                                                'SPARSE_OK=TRUE','INTERLEAVE=PIXEL']) # 'PREDICTOR=3'
    gdal.Warp(out,inp,options=options)

def GetPercentageBands(mask, subsidence, csv_out):
    sub = gdal.Open(subsidence).ReadAsArray()
    msk = gdal.Open(mask).ReadAsArray()
    nan = msk <= -1000
    sub[nan] = -9999
    sub = sub[sub > -9999]
    sub.flatten()

    sub[sub < -1.04] = 6
    sub[(sub >= -1.04) & (sub < -0.84)] = 5
    sub[(sub >= -0.84) & (sub < -0.64)] = 4
    sub[(sub >= -0.64) & (sub < -0.44)] = 3
    sub[(sub >= -0.44) & (sub < -0.24)] = 2
    sub[(sub >= -0.24) & (sub <= 0.24)] = 1
    sub[(sub > 0.24) & (sub <= 0.44)] = 2

    #df = pd.DataFrame({'Subsidence':sub})
    #sns.countplot(x='Subsidence', data=df)
    #plt.gcf()
    #plt.savefig(plt_name)

    outdict = {'F':len(sub[sub==6]), 'E':len(sub[sub==5]), 'D':len(sub[sub==4]),
               'C':len(sub[sub==3]), 'B':len(sub[sub==2]), 'A':len(sub[sub==1])}
    outframe = pd.DataFrame(list(outdict.items()))
    outframe.to_csv(csv_out)

def GetShrinkSwell(mask, shrink, csv_out):
    shr = gdal.Open(shrink).ReadAsArray()
    msk = gdal.Open(mask).ReadAsArray()
    nan = msk <= -1000
    shr[nan] = -9999
    shr = shr[shr > -9999]
    shr.flatten()
    outdict = {'E':len(shr[shr==4]), 'D':len(shr[shr==3]),'C':len(shr[shr==2]),
               'B':len(shr[shr==1]), 'A':len(shr[shr==0])}
    outframe = pd.DataFrame(list(outdict.items()))
    outframe.to_csv(csv_out)

# Calculate mask
#clip("tifs/eng&wales.shp","tifs/eng_wales.tif")
#reproject("tifs/eng_wales.tif", "tifs/eng_wales1.tif")
#clip("tifs/scotland.shp","tifs/scotland.tif")
#reproject("tifs/scotland.tif", "tifs/scotland1.tif")
#clip("tifs/ni.shp","tifs/ni.tif")
#reproject("tifs/ni.tif", "tifs/ni1.tif")
#clip("tifs/GBcutline.shp", "tifs/Britain.tif")
#reproject("tifs/Britain.tif", "tifs/Britain.tif")


# Baseline get percentages
GetPercentageBands("tifs/eng_wales1.tif", "tifs/with_tc/rcp85_2020_bs_soil.tif",
                    "tifs/engwales_bl_new1.csv")
GetPercentageBands("tifs/scotland1.tif", "tifs/with_tc/rcp85_2020_bs_soil.tif",
                    "tifs/scot_bl_new.csv")
GetPercentageBands("tifs/ni1.tif", "tifs/with_tc/rcp85_2020_bs_soil.tif",
                   "tifs/ni_bl_new.csv")

# 2050 get percentages
GetPercentageBands("tifs/eng_wales1.tif", "tifs/with_tc/rcp85_2045-2054_soil.tif",
                   "tifs/engwales_2050new.csv")
GetPercentageBands("tifs/scotland1.tif", "tifs/with_tc/rcp85_2045-2054_soil.tif",
                   "tifs/scot_2050new.csv")
GetPercentageBands("tifs/ni1.tif", "tifs/with_tc/rcp85_2045-2054_soil.tif",
                   "tifs/ni_2050new.csv")

# 2070 get percentages
GetPercentageBands("tifs/eng_wales1.tif", "tifs/with_tc/rcp85_2065-2074_soil.tif",
                   "tifs/engwales_2070new.csv")
GetPercentageBands("tifs/scotland1.tif", "tifs/with_tc/rcp85_2065-2074_soil.tif",
                   "tifs/scot_2070new.csv")
GetPercentageBands("tifs/ni1.tif", "tifs/with_tc/rcp85_2065-2074_soil.tif",
                   "tifs/ni_2070new.csv")

"""
# get BGS shrink swell
reproject("/Users/whamitchell/Documents/aws/inputs/tifs/geology_encoded.tif",
          "/Users/whamitchell/Documents/aws/inputs/tifs/geology.tif")
#GetShrinkSwell("tifs/eng_wales1.tif",
#               "/Users/whamitchell/Documents/aws/inputs/bgs/geo_ss/ShrinkSwell.gtiff",
#               "tifs/bgs_shrinkwswell.csv")
"""