import numpy as np
import pandas as pd
from osgeo import gdal
from joblib import load

def predict(model, data, outraster):
    rfr = load(model)  # load the model
    df = pd.read_hdf(data, key='df') # load prediction data with projected rainfall
    RandRF_predY = rfr.predict(df.drop(['row', 'col', 'Disp_cmyr'], axis=1))
    
    # save to raster
    print("\n Creating raster...")
    reference_raster = gdal.Open("tifs/final_dem.tif")
    geo = reference_raster.GetGeoTransform()
    dem = reference_raster.ReadAsArray()
    xsize, ysize = reference_raster.RasterXSize, reference_raster.RasterYSize
    array = np.zeros((ysize, xsize))
    row = df['row'].values
    col = df['col'].values
    array[row, col] = RandRF_predY
    mask = dem == -9999
    array[mask] = -9999
    
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.CreateCopy(outraster, reference_raster, strict=0)
    dst_ds.GetRasterBand(1).WriteArray(array)
    dst_ds.SetGeoTransform(geo)
    band = dst_ds.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    dst_ds = None

"""
#predict("climate_cond_models/rcp85_trained_no_cornwall.joblib",
#        "prediction/rcp85_2020_baseline.h5",
#        "rcp85outputs/rcp85_2020_baseline.tif")
predict("climate_cond_models/rcp85_trained_no_cornwall.joblib",
        "prediction/rcp85_2015-2024.h5",
        "rcp85outputs/rcp85_2015-2024.tif")
predict("climate_cond_models/rcp85_trained_no_cornwall.joblib",
        "prediction/rcp85_2020-2029.h5",
        "rcp85outputs/rcp85_2020-2029.tif")
predict("climate_cond_models/rcp85_trained_no_cornwall.joblib",
        "prediction/rcp85_2025-2034.h5",
        "rcp85outputs/rcp85_2025-2034.tif")
predict("climate_cond_models/rcp85_trained_no_cornwall.joblib",
        "prediction/rcp85_2030-2039.h5",
        "rcp85outputs/rcp85_2030-2039.tif")
predict("climate_cond_models/rcp85_trained_no_cornwall.joblib",
        "prediction/rcp85_2035-2044.h5",
        "rcp85outputs/rcp85_2035-2044.tif")
predict("climate_cond_models/rcp85_trained_no_cornwall.joblib",
        "prediction/rcp85_2040-2049.h5",
        "rcp85outputs/rcp85_2040-2049.tif")
predict("climate_cond_models/rcp85_trained_no_cornwall.joblib",
        "prediction/rcp85_2045-2054.h5",
        "rcp85outputs/rcp85_2045-2054.tif")
predict("climate_cond_models/rcp85_trained_no_cornwall.joblib",
        "prediction/rcp85_2045-2054.h5",
        "rcp85outputs/rcp85_2050-2059.tif")
predict("climate_cond_models/rcp85_trained_no_cornwall.joblib",
        "prediction/rcp85_2055-2064.h5",
        "rcp85outputs/rcp85_2055-2064.tif")
predict("climate_cond_models/rcp85_trained_no_cornwall.joblib",
        "prediction/rcp85_2060-2069.h5",
        "rcp85outputs/rcp85_2060-2069.tif")
predict("climate_cond_models/rcp85_trained_no_cornwall.joblib",
        "prediction/rcp85_2065-2074.h5",
        "rcp85outputs/rcp85_2065-2074.tif")
predict("climate_cond_models/rcp85_trained_no_cornwall.joblib",
        "prediction/rcp85_2070-2079.h5",
        "rcp85outputs/rcp85_2070-2079.tif")
"""