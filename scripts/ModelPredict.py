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
    dst_ds = driver.CreateCopy(outraster, reference_raster, strict=0,
                               options=['COMPRESS=DEFLATE','BIGTIFF=YES','BLOCKXSIZE=512',
                                        'PREDICTOR=3','TILED=YES','NUM_THREADS=ALL_CPUS',
                                        'SPARSE_OK=TRUE','INTERLEAVE=PIXEL'])
    dst_ds.GetRasterBand(1).WriteArray(array)
    dst_ds.SetGeoTransform(geo)
    band = dst_ds.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    dst_ds = None


def PredictNoLoss(model, data, outraster):
    rfr = load(model)  # load the model
    df = pd.read_hdf(data, key='df')  # load prediction data with projected rainfall
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

### TREE and FOREST INTERPRETATION ###
def LocalInterpretation(model, data, out_df):
    """
    model: tree-based classifier
    data: prediction data
    out_df: output dataframe filepath
    """
    rf = load(model)
    df = pd.read_hdf(data, key='df')
    X = df.drop(['Landslide', 'row', 'col'], axis=1)
    pred_y = rf.predict(X)
    instances = X[(pred_y > -0.1) & (pred_y < 0.6)][0:100] # conditional statement for values of interest

    prediction, bias, contributions = ti.predict(rf, instances) #prediction=bias+feature1contribution+…featurencontribution
    for i in range(len(instances)):
        temp = {}
        temp['Prediction'] = prediction[i][0]
        temp['Bias(transient_mean)'] = bias[i][0]
        for c, feat in sorted(zip(contributions[i], boston.feature_names), key=lambda x: -abs(x[0])):
            temp[feat] = c[0]
        df = df.append(temp, ignore_index=True)
    df.to_csv(out_df)


predict("climate_cond_models/rcp85__regularised.joblib",
        "prediction/rcp85_2020_baseline.h5",
        "rcp85outputs/rcp85_2020_bl_reg.tif")
predict("climate_cond_models/rcp85__regularised.joblib",
        "prediction/rcp85_2055-2064.h5",
        "rcp85outputs/rcp85_2055-2064_reg.tif")
"""
predict("climate_cond_models/rcp85_notemp.joblib",
        "prediction/rcp85_2025-2034_notemp.h5",
        "rcp85outputs/rcp85_2025-2034_notemp_geo.tif")
predict("climate_cond_models/rcp85_notemp.joblib",
        "prediction/rcp85_2035-2044_notemp.h5",
        "rcp85outputs/rcp85_2035-2044_notemp_geo.tif")
predict("climate_cond_models/rcp85_notemp.joblib",
        "prediction/rcp85_2045-2054_notemp.h5",
        "rcp85outputs/rcp85_2045-2054_notemp_geo.tif")
predict("climate_cond_models/rcp85_notemp.joblib",
        "prediction/rcp85_2055-2064_notemp.h5",
        "rcp85outputs/rcp85_2055-2064_notemp_geo.tif")
predict("climate_cond_models/rcp85_notemp.joblib",
        "prediction/rcp85_2065-2074_notemp.h5",
        "rcp85outputs/rcp85_2065-2074_notemp_geo.tif")
predict("climate_cond_models/rcp85_notemp.joblib",
        "prediction/rcp85_2075-2084_notemp.h5",
        "rcp85outputs/rcp85_2075-2084_notemp_geo.tif")
"""