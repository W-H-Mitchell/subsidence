import csv
import numpy as np
import pandas as pd
from joblib import dump
from osgeo import gdal, gdalconst
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 

def train(train_data, outmodel, featout, outraster):
    df = pd.read_hdf(train_data, 'df')
    baseDF = df[df['Disp_cmyr'].notnull()] # remove null from the target class for training
    y = baseDF.loc[:, 'Disp_cmyr'] # target class
    X = baseDF.drop(['Disp_cmyr', 'row', 'col'], axis=1) # predictors 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # model
    rfr = RandomForestRegressor(random_state=42, n_jobs=-1, min_samples_split=10000, verbose=100)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    dump(rfr, 'climate_cond_models/{0}.joblib'.format(outmodel))
    # evaluation metrics
    print('Verification set:\nMean Absolute Error: {0}; Mean Absolute Percentage Error: {1}'.format(mean_absolute_error(y_test, y_pred), mean_absolute_percentage_error(y_test, y_pred)))
    
    RandRF_predY = rfr.predict(df.drop(['row', 'col', 'Disp_cmyr'], axis=1))
    # prediction to fill target class
    print("\n Creating raster...")
    reference_raster = gdal.Open("tifs/dem_gb.tif")
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
    
    
    # feature importance
    importance = rfr.feature_importances_
    dictionary = {}
    # summarize feature importance
    for name, importance in zip(X.columns, rfr.feature_importances_):
        dictionary[name] = importance # plot feature importance
    # open file for writing, "w" is writing
    w = csv.writer(open("feat_importance/{0}.csv".format(featout), "w"))
    # loop over dictionary keys and values
    for key, val in dictionary.items():
        # write every key and value to file
        w.writerow([key, val])
        
train("training/rcp85_training.h5", "rcp85_trained", "rcp85", "training/rcp85_train_sm.tif")
#train("training/train_sep_soil.h5", "trained_sep_soil", 'separatesoil')