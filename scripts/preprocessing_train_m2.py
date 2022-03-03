# IMPORT PACKAGES
import ast
import pandas as pd
import numpy as np
from osgeo import gdal
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import TransformerMixin

# lat long
def lat_lon_transform(input_filepath, reference_filepath, output_filepath):
    inputfile = input_filepath  # Path to input file
    inpt = gdal.Open(inputfile)
    inputProj = inpt.GetProjection()

    referencefile = reference_filepath  # Path to reference file
    reference = gdal.Open(referencefile)  # reads in the reference raster
    referenceProj = reference.GetProjection()  # obtains the crs
    referenceTrans = reference.GetGeoTransform()  # geotransform which contains extent, resolution data
    bandreference = reference.GetRasterBand(1)  # dw about this, you can get multiband rasters but we only have one
    x = reference.RasterXSize  # get the raster array xsize, ie., number of pixels in x direction
    y = reference.RasterYSize

    outputfile = output_filepath  # Path to output file to be created
    driver = gdal.GetDriverByName('GTiff')  # type of driver needed for the output file
    output = driver.Create(outputfile, x, y, 1, bandreference.DataType)  # creates output
    output.SetGeoTransform(referenceTrans)
    output.SetProjection(referenceProj)
    output.GetRasterBand(1).SetNoDataValue(-9999)
    gdal.ReprojectImage(inpt, output, inputProj, referenceProj, gdal.GRA_Bilinear) # reprojects the output


    res = 90
    project = 'epsg:4326'
    bounds = [-10, 52, 10, 56]
    options = gdal.WarpOptions(resampleAlg='nearest', xRes=res, yRes=90)
    gdal.Warp(output_filepath,input_filepath,options=options )



# Create the dataframe imputer class
class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value
        in column.
        Columns of other types are imputed with mean of column.
        """
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

# Ordinal Encoder for Categorical Data
def ord_enc(df, col):
    ord_enc = OrdinalEncoder()
    data = df[col].values
    data = data.reshape(-1,1)
    data_enc = ord_enc.fit_transform(data)
    data_enc = pd.DataFrame(data_enc.flatten(), columns=[col], index=df.index)
    return data_enc

def data_gen(folder, rain, temp, output_filename):
    # rasters
    z = gdal.Open("tifs/uk_dem_wgs84_0.0008.tif")
    s = gdal.Open("tifs/slope_gb.tif")
    rock = gdal.Open("tifs/lithology_gb.tif")
    supf = gdal.Open("tifs/superficial_gb.tif")
    #swel = gdal.Open("tifs/shrinkswell_gb.tif")
    luc = gdal.Open("tifs/landuse_gb.tif")
    veg = gdal.Open("tifs/ndvi_gb.tif") #NDVI
    rivers = gdal.Open("tifs/rivers_gb.tif")
    disp = gdal.Open("tifs/reproject_v2dinsar copy.tif") # subsidence; target variable training data
    # climate data
    rain = gdal.Open(folder+rain)
    temp = gdal.Open(folder+temp)
    #soil = gdal.Open(folder+soil)

    
    # Stack the geotifs of each feature and merge into a dataframe
    dfs = [z, s, rock, supf, luc, veg,
           rivers, disp, rain, temp]
    for df in dfs: 
        print(df.RasterYSize, df.RasterXSize)
    merged = gdal.BuildVRT('', dfs, separate=True)
    merged_data = merged.ReadAsArray()
    ysize, xsize = merged.RasterYSize, merged.RasterXSize
    row, cols = np.mgrid[0:ysize:1, 0:xsize:1]
    column_names = ['z', 's', "RockClass", "SuperfDep", "LUC",
                    "NDVI", "DistRiv_m", "Disp_cmyr", "Rainfall", "deg_c",
                    ]
    df = pd.DataFrame(data=merged_data.reshape((len(dfs), -1)).T, columns=column_names)
    df['row'] = row.flatten()
    df['col'] = cols.flatten()
    
    # mask
    df = df[df['z'] > -61]
    row, col = df['row'].values.flatten(), df['col'].values.flatten()
    
    # map categorical data
    geo_file = open("geology_dictionary.txt", "r")
    contents = geo_file.read()
    geo_dict = ast.literal_eval(contents)
    geo_dict = dict((v,k) for k,v in geo_dict.items())

    #swell
    #swell_file = open("shrinkswell_dictionary.txt", "r")
    #contents = swell_file.read()
    #swell_dict = ast.literal_eval(contents)
    #swell_dict = dict((v,k) for k,v in swell_dict.items())

    # superficial 
    sup_file = open("superficial_dictionary.txt", "r")
    contents = sup_file.read()
    sup_dict = ast.literal_eval(contents)
    sup_dict = dict((v,k) for k,v in sup_dict.items())
    LUC_map = {10: 'Tree cover', 20: 'Shrubland', 30: 'Grassland', 40: 'Cropland',
               50: 'Built-up', 60: 'Sparse vegetation', 80: 'Permanent water bodies',
               90: 'Herbaceous wetland', 100: 'Moss and lichen'}

    # replace added new values
    df['RockClass'][~df['RockClass'].isin(geo_dict.keys())] = np.nan
    df['SuperfDep'][~df['SuperfDep'].isin(sup_dict.keys())] = np.nan
    #df['Swell'][~df['Swell'].isin(swell_dict.keys())] = np.nan
    #df['Swell'].replace(swell_dict, inplace=True)

    df['SuperfDep'].replace(sup_dict, inplace=True)
    df['RockClass'].replace(geo_dict, inplace=True)
    df['LUC'].replace(LUC_map, inplace=True)
    df['s'].replace(-9999, 0, inplace=True)
    df['Disp_cmyr'] = df["Disp_cmyr"].values * 10
    
    print(df.isna().sum(), f"Percentage missing data:\n{100*(df.isna().sum()/len(df))}") # No bgs data in Ireland
    print('imputing')
    # IMPUTER
    # Missing Values
    numDf = df.select_dtypes(include='number') # dataframe of numerical featues
    catDf = df.select_dtypes(exclude='number') # categorical features
    for d in numDf.columns:
        numDf[d][numDf[d] < -100] = np.nan
    for _ in catDf.columns:
        catDf[_][(catDf[_]==-9999.0)] = np.nan 
    df = pd.concat([numDf, catDf], axis=1, join='outer')
    # Drop columns that we do not want to impute missing values in
    data = df.drop(['RockClass', 'SuperfDep', "Disp_cmyr"], axis=1) #'Swell'
    imp_df = DataFrameImputer().fit_transform(data)
    assert len(imp_df.isna()==0), "Imputing Error: imputed dataframe contains NaN"
    imp_df = pd.concat([imp_df, df[["RockClass", "SuperfDep", "Disp_cmyr"]]], axis=1) #"Swell"
    
    print('encoding')
    # ENCODING
    enc_catDF = pd.DataFrame(index=catDf.index)
    cat_variables = imp_df[['RockClass', 'SuperfDep', 'LUC']] # 'Swell'
    cat_dummies = pd.get_dummies(cat_variables, drop_first=True)
    enc_catDF = pd.concat([enc_catDF, cat_dummies], axis=1) # Append encoded columns 
    temp_df = imp_df.drop(['RockClass', 'SuperfDep', 'LUC'], axis=1) #'Swell'
    processDF = pd.concat([temp_df, enc_catDF], axis=1)
    processDF.to_hdf(output_filename, key='df', mode='w')

# lat long transform

# training 
#data_gen("tifs/train2018/", "winter-summer_rain2018.tif", "summer_temperature.tif",
#"training/rcp85train_swell.h5")
# prediction
data_gen("tifs/predict/", "rcp85_baseline2020_rainfall.tif",
         "rcp85_baseline2020_tas.tif",
         "prediction/rcp85_2020_baseline.h5")
data_gen("tifs/predict/", "rcp85_model_2025-2034_winter-summer_rainfall.tif",
         "rcp85_model_2025-2034_summer_tas.tif",
         "prediction/rcp85_2025-2034.h5")
data_gen("tifs/predict/", "rcp85_model_2035-2044_winter-summer_rainfall.tif",
         "rcp85_model_2035-2044_summer_tas.tif",
         "prediction/rcp85_2035-2044.h5")
data_gen("tifs/predict/", "rcp85_model_2040-2049_winter-summer_rainfall.tif",
         "rcp85_model_2035-2044_summer_tas.tif",
         "prediction/rcp85_2045-2054.h5")
data_gen("tifs/predict/", "rcp85_model_2040-2049_winter-summer_rainfall.tif",
         "rcp85_model_2055-2064_summer_tas.tif",
         "prediction/rcp85_2055-2064.h5")
data_gen("tifs/predict/", "rcp85_model_2065-2074_winter-summer_rainfall.tif",
         "rcp85_model_2065-2074_summer_tas.tif",
         "prediction/rcp85_2065-2074.h5")
data_gen("tifs/predict/", "rcp85_model_2070-2079_winter-summer_rainfall.tif",
         "rcp85_model_2075-2084_summer_tas.tif",
         "prediction/rcp85_2075-2084.h5")