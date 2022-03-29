# IMPORT PACKAGES
import ast
import pandas as pd
import numpy as np
import seaborn as sns
from osgeo import gdal
import matplotlib.pyplot as plt
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
    x, y = reference.RasterXSize, reference.RasterYSize  # get the raster array xsize, ie., number of pixels in x direction

    outputfile = output_filepath  # Path to output file to be created
    driver = gdal.GetDriverByName('GTiff')  # type of driver needed for the output file
    output = driver.Create(outputfile, x, y, 1, bandreference.DataType)  # creates output
    output.SetGeoTransform(referenceTrans)
    output.SetProjection(referenceProj)
    output.GetRasterBand(1).SetNoDataValue(-9999)
    gdal.ReprojectImage(inpt, output, inputProj, referenceProj, gdal.GRA_NearestNeighbour) # reprojects the output

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

def data_gen(folder, rain, dem, temp, output_filename):
    # rasters
    z = gdal.Open(dem)
    s = gdal.Open("tifs/slope_gb.tif")
    rock = gdal.Open("tifs/lithology_reclassified.tif")
    #twi =  gdal.Open("tifs/twi.tif")
    supf = gdal.Open("tifs/superficial_gb.tif")
    luc = gdal.Open("tifs/landuse_gb.tif")
    veg = gdal.Open("tifs/ndvi_gb.tif") #NDVI
    rivers = gdal.Open("tifs/rivers_gb.tif")
    disp = gdal.Open("tifs/reproject_v2dinsar copy.tif") # subsidence; target variable training data
    soiltyp = gdal.Open("tifs/ClayTypes.tif")
    soilpct = gdal.Open("tifs/ClayPct.tif")
    # climate data
    rain = gdal.Open(folder+rain)
    temp = gdal.Open(folder+temp)
    #soil = gdal.Open(folder+soil)

    
    # Stack the geotifs of each feature and merge into a dataframe
    dfs = [z, s, rock, supf, luc, veg, rivers,
           disp, soiltyp, soilpct, rain] #temp
    for df in dfs: 
        print(df.RasterYSize, df.RasterXSize)
    merged = gdal.BuildVRT('', dfs, separate=True)
    merged_data = merged.ReadAsArray()
    ysize, xsize = merged.RasterYSize, merged.RasterXSize
    row, cols = np.mgrid[0:ysize:1, 0:xsize:1]
    column_names = ['z', 's', "Rock", "SuperfDep", "LUC", "NDVI", "DistRiv_m",
                    "Disp_cmyr", "SoilType", "SoilPct", "Rainfall"] #"deg_C"
    df = pd.DataFrame(data=merged_data.reshape((len(dfs), -1)).T, columns=column_names)
    df['row'] = row.flatten()
    df['col'] = cols.flatten()
    
    # mask
    df = df[df['z'] > -61]
    row, col = df['row'].values.flatten(), df['col'].values.flatten()
    
    # map categorical data
    # geology
    #geo_file = open("geology_dictionary.txt", "r")
    #contents = geo_file.read()
    #geo_dict = ast.literal_eval(contents)
    #geo_dict = dict((v,k) for k,v in geo_dict.items())

    # superficial 
    sup_file = open("superficial_dictionary.txt", "r")
    contents = sup_file.read()
    sup_dict = ast.literal_eval(contents)
    sup_dict = dict((v,k) for k,v in sup_dict.items())
    lith_map = {1: 'IGNEOUS or METAMORPHIC', 2: 'CLAY', 3: 'LIAS GROUP', 4: 'LONDON CLAY',
                 5: 'LAMBETH GROUP', 6: 'MERCIA - MUDSTONE, SILTSTONE AND SANDSTONE',
                 7: 'MERCIA - MUDSTONE WITH GYPSUM-STONE AND/OR ANHYDRITE-STONE',
                 8: 'MERCIA - MUDSTONE AND HALITE-STONE', 9: 'MERCIA - MUDSTONE',
                 10: 'MERCIA - MUDSTONE AND SILTSTONE', 11: 'GAULT FM', 12: 'GAULT - MUDSTONE',
                 13: 'KELLAWAYS FM', 14: 'WEALD CLAY', 15: 'OXFORD CLAY', 16: 'WADHURST CLAY',
                 17: 'BARTON GROUP', 18: 'WEALDEN GROUP', 19: 'KIMMERIDGE CLAY', 20: 'AMPTHILL CLAY'}
    LUC_map = {10: 'Tree cover', 20: 'Shrubland', 30: 'Grassland', 40: 'Cropland',
               50: 'Built-up', 60: 'Sparse vegetation', 80: 'Permanent water bodies',
               90: 'Herbaceous wetland', 100: 'Moss and lichen'}
    soil_dict = {2:'Alisols',3:'Leptosols',4:'Arenosols',5:'Calcisols',6:'Cambisols',11:'Fluvisols',
                12:'Gleysols',14:'Histosols',16:'Andosols',17:'Lixisols',18:'Luvisols',20:'Phaeozems',
                21:'Planosols',23:'Podzols',27:'Stagnosols'}
    # replace added new values
    #df['Rock'][~df['Rock'].isin(geo_dict.keys())] = np.nan
    df['SuperfDep'][~df['SuperfDep'].isin(sup_dict.keys())] = np.nan
    df['SoilType'].replace(soil_dict, inplace=True)
    df['SuperfDep'].replace(sup_dict, inplace=True)
    df['Rock'].replace(lith_map, inplace=True) #geo_dict
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
        numDf[d][numDf[d] < -300] = np.nan
    for _ in catDf.columns:
        catDf[_][(catDf[_]==-9999.0)] = np.nan 
    df = pd.concat([numDf, catDf], axis=1, join='outer')
    # Drop columns that we do not want to impute missing values in
    data = df.drop(['SuperfDep','Disp_cmyr','SoilType',"Rock"], axis=1)
    imp_df = DataFrameImputer().fit_transform(data)
    assert len(imp_df.isna()==0), "Imputing Error: imputed dataframe contains NaN"
    imp_df = pd.concat([imp_df, df[["SuperfDep","Disp_cmyr","SoilType", "Rock"]]], axis=1)
    
    print('encoding')
    # ENCODING
    enc_catDF = pd.DataFrame(index=catDf.index)
    cat_variables = imp_df[['SuperfDep','LUC','SoilType', 'Rock']]
    cat_dummies = pd.get_dummies(cat_variables, drop_first=True)
    enc_catDF = pd.concat([enc_catDF, cat_dummies], axis=1) # Append encoded columns 
    temp_df = imp_df.drop(['SuperfDep', 'LUC','SoilType', 'Rock'], axis=1)
    processDF = pd.concat([temp_df, enc_catDF], axis=1)
    processDF.to_hdf(output_filename, key='df', mode='w')
    print(processDF.shape)
    print("SAVED\n" + ("-")*30)

def multicollinearity_check(data, matrix, plot):
    # correlation matrix
    df = pd.read_hdf(data)
    corrmatrix = df.corr()
    corrmatrix.to_csv(matrix)

    # plot
    fig, ax = plt.figure(figsize=(40, 40))
    sns.heatmap(corrmatrix, ax=ax)
    plt.gcf()
    plt.savefig(plot)



# training 
data_gen("tifs/train2018/", "winter-summer_rain2018.tif",
         "tifs/Britain.tif",
         "summer_temperature.tif",
         "training/rcp85train_notemp.h5")
# prediction
data_gen("tifs/predict/", "rcp85_baseline2020_rainfall.tif",
         "tifs/uk_dem_wgs84_0.0008.tif",
         "rcp85_baseline2020_tas.tif",
         "prediction/rcp85_2020_baseline_notemp.h5")
data_gen("tifs/predict/", "rcp85_model_2020-2029_winter-summer_rainfall.tif",
         "tifs/uk_dem_wgs84_0.0008.tif",
         "rcp85_model_2020-2029_summer_tas.tif",
         "prediction/rcp85_2025-2034_notemp.h5")
data_gen("tifs/predict/", "rcp85_model_2035-2044_winter-summer_rainfall_lin.tif",
         "tifs/uk_dem_wgs84_0.0008.tif",
         "rcp85_model_2035-2044_summer_tas.tif",
         "prediction/rcp85_2035-2044_notemp.h5")
data_gen("tifs/predict/", "rcp85_model_2045-2054_winter-summer_rainfall_lin.tif",
         "tifs/uk_dem_wgs84_0.0008.tif",
         "rcp85_model_2045-2054_summer_tas_lin.tif",
         "prediction/rcp85_2045-2054_notemp.h5")
data_gen("tifs/predict/", "rcp85_model_2055-2064_winter-summer_rainfall_lin.tif",
         "tifs/uk_dem_wgs84_0.0008.tif",
         "rcp85_model_2055-2064_summer_tas_lin.tif",
         "prediction/rcp85_2055-2064_notemp.h5")
data_gen("tifs/predict/", "rcp85_model_2065-2074_winter-summer_rainfall.tif",
         "tifs/uk_dem_wgs84_0.0008.tif",
         "rcp85_model_2065-2074_summer_tas.tif",
         "prediction/rcp85_2065-2074_notemp.h5")
data_gen("tifs/predict/", "rcp85_model_2070-2079_winter-summer_rainfall.tif",
         "tifs/uk_dem_wgs84_0.0008.tif",
         "rcp85_model_2075-2084_summer_tas.tif",
         "prediction/rcp85_2075-2084_notemp.h5")
"""
multicollinearity_check("training/rcp85train_nopr.h5",
                        "training/correlation_matrix_nopr.csv",
                        "training/corr_heatmap_nopr.png")
multicollinearity_check("training/rcp85train_nogeo.h5",
                        "training/correlation_matrix_nogeo.csv",
                        "training/corr_heatmap_nogeo.png")
"""