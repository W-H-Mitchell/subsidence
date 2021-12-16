# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 13:35:09 2021

@author: HamishMitchell
"""
# Preprocessing 
import os
import pandas as pd
import numpy as np
import collections
import json
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from osgeo import gdal 

# change direction
# os.chdir("C:/Users/HamishMitchell/uk_geophys/subsidence")
# load dataset
df = pd.read_hdf('uk_clipped_subsidence_processed.h5', 'df')
baseDF = df[df['disp_cmyr'].notnull()] # remove null from the target class for training
print("Data shape with DInSAR nan: {0}; data shape without NaNs: {1}".format(df.shape, baseDF.shape))
df = None 

# separate target and the predictors 
y = baseDF.loc[:, 'disp_cmyr']
X = baseDF.drop(['disp_cmyr', 'row', 'col'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# model
rfr = RandomForestRegressor(random_state=42, n_jobs=-1, min_samples_split=1000, verbose=100)
print(rfr.get_params())
# Test on the verification set
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)
# evaluation metrics
print("Verification set:\n")
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Mean Absolute Percentage Error:', mean_absolute_percentage_error(y_test, y_pred))
print('-'*40)

# get importance
importance = rfr.feature_importances_
print(importance)
# summarize feature importance
dictionary = {}
for name, importance in zip(baseDF.drop(['disp_cmyr', 'row', 'col'], axis=1).columns, rfr.feature_importances_):
    dictionary["name"] = importance # plot feature importance
dictionary = collections.OrderedDict(sorted(dictionary.items()))
with open('Feature_importance.txt', 'w') as file:
     file.write(json.dumps(dictionary)) # use `json.loads` to do the reverse
    
"""
# HYPERPARAMETER TUNING
# number of trees in random forest
n_estimators = [int(x) for x in np.linspace(200, 2000, 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(30, 80, 5)]
max_depth.append(None)
# minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

# use the random grid to search for best hyperparameters
rfr_random = RandomizedSearchCV(estimator=rfr, param_distributions=random_grid, scoring='neg_mean_absolute_percentage_error', n_iter=10, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# fit the random search model
rfr_random.fit(X_train, y_train)
print("Randomised Search Cross Validation:\n Scores:", rfr_random.cv_results_)
print("Best parameters for Radom Forest Regressor:", rfr_random.best_params_)
print("Best score for the Radom Forest Regressor:", rfr_random.best_score_)
print('-'*40)
baseDF = None
"""
# Load subsidence dataset
df = pd.read_hdf('uk_clipped_subsidence_processed.h5', 'df')

# Predict the rest of the data
RandRF_predY = rfr.predict(df.drop(['row', 'col', 'disp_cmyr'], axis=1))
predict_y = pd.DataFrame(RandRF_predY.flatten(), columns=['cm/yr'], index=df.index)
predictedXy = pd.concat([df, predict_y], axis=1)
#predictedXy.to_pickle('UK_subsidence.pkl')
predictedXy.to_hdf('UK_clipped_displacement.h5', key='df')

#Read in the subsidence data 
df = pd.read_hdf('UK_clipped_displacement.h5', 'df')
# Save to raster
reference_raster = gdal.Open("tifs/dem_gb.tif")
dem = reference_raster.ReadAsArray()
geo = reference_raster.GetGeoTransform()
xsize, ysize = reference_raster.RasterXSize, reference_raster.RasterYSize

array = np.zeros((ysize, xsize))
row = df['row'].values
col = df['col'].values
array[row, col] = RandRF_predY
mask = dem == -9999
array[mask] = -9999

driver = gdal.GetDriverByName('GTiff')
dst_ds = driver.CreateCopy('clipped_displacement.tif', reference_raster, strict=0)
dst_ds.GetRasterBand(1).WriteArray(array)
dst_ds.SetGeoTransform(geo)
dst_ds = None