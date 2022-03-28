import csv
import numpy as np
import pandas as pd
from joblib import dump
from joblib import load
from osgeo import gdal, gdalconst
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 

def train(train_data, outmodel, outraster):

    ### Data
    df = pd.read_hdf(train_data, 'df')
    baseDF = df[df['Disp_cmyr'].notnull()] # remove null from the target class for training
    print("Data shape with DInSAR nan: {0}; data shape without NaNs: {1}".format(df.shape, baseDF.shape))
    y = baseDF.loc[:, 'Disp_cmyr'] # target class
    X = baseDF.drop(['Disp_cmyr', 'row', 'col'], axis=1) # predictors 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    ### Model Training
    rfr = RandomForestRegressor(random_state=42, n_jobs=-1,
                                min_samples_split=10000, verbose=100)
    rfr.fit(X_train, y_train)
    y_pred = rfr.predict(X_test)
    dump(rfr, 'climate_cond_models/{0}.joblib'.format(outmodel))
    # evaluation metrics
    print('Verification set:\nMean Absolute Error: {0}; Mean Absolute Percentage Error: {1}'.format(mean_absolute_error(y_test, y_pred), mean_absolute_percentage_error(y_test, y_pred)))

    ### Raster Generation
    RandRF_predY = rfr.predict(df.drop(['row', 'col', 'Disp_cmyr'], axis=1))
    # prediction to fill target class
    print("\n Creating raster...")
    reference_raster = gdal.Open("tifs/uk_dem_wgs84_0.0008.tif")
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



def feature_importance(model, train_data, rf_featout, perm_featout, perm_plt):
    rf = load(model)  # load the model
    df = pd.read_hdf(train_data, 'df')
    baseDF = df[df['Disp_cmyr'].notnull()]  # remove null from the target class for training
    y = baseDF.loc[:, 'Disp_cmyr'] # target class
    X = baseDF.drop(['Disp_cmyr', 'row', 'col'], axis=1) # predictors
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    ### Random Feature importance ###
    importance = rf.feature_importances_
    dictionary = {}
    # summarize feature importance
    for name, importance in zip(X.columns, rf.feature_importances_):
        dictionary[name] = importance # plot feature importance
    # open file for writing, "w" is writing
    w = csv.writer(open(rf_featout, "w"))
    # loop over dictionary keys and values
    for key, val in dictionary.items():
        # write every key and value to file
        w.writerow([key, val])

    ### Permutation feature importance ###
    result = permutation_importance(rf, X_test, y_test, n_repeats=3, random_state=0, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()
    # summarize feature importance
    # for i in result.importances_mean.argsort()[::-1]:
    #    dictionary[f"{X.columns[i]}"] = f"{result.importances_mean[i]:.3f}"
    for name, importance in zip(X_test.columns[sorted_idx], result.importances[sorted_idx]):
        dictionary[name] = importance # plot feature importance
    # open file for writing, "w" is writing
    w = csv.writer(open(perm_featout, "w"))
    for key, val in dictionary.items():
        # write every key and value to file
        w.writerow([key, val])

    # Plot the permutation importance
    fig, ax = plt.subplots(figsize=(50, 50))
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx])
    ax.set_title("Permutation Importances (Test Set)")
    plt.gcf()
    plt.savefig(perm_plt)



#train("training/rcp85train_nogeo.h5","rcp85_nogeo","rcp85_nogeo",
#     "training/Rcp85_NoGeo.tif")
feature_importance("climate_cond_models/rcp85_nogeo.joblib",
                   "training/rcp85train_nogeo.h5",
                   "feat_importance/rcp85_nogeology_rf.csv",
                   "feat_importance/rcp85_nogeology_perm.csv",
                   "feat_importance/rcp85_nogeo_perm.png")