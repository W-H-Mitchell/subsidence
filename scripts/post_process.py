import xarray as xr
import numpy as np
from osgeo import gdal
from affine import Affine
from tqdm import tqdm
ds = xr.open_dataset("JBt81.zarr")

PROJECTION = 'epsg:4326'



def get_min_max(ts, error_threshold=50):
    med = df.rolling(window='60d', center=True).median()
    std = df.rolling(window='60d', center=True).std()
    relative_error = (100*std/med).abs()
    # remove obs above relative error
    ts[relative_error>error_threshold]=np.nan

def min_max(ts):
    med =  df.rolling(window='60d', center=True).median()
    year_max = med.groupby(ts.index.year).max()
    year_min = med.groupby(ts.index.year).min()
    return year_max, year_min

if __name__ == "__main__":
    ds = xr.open_dataset("JBt81.zarr")

    nT, ysize, xsize=  ds['displacement'].shape

    nyears = 3
    max = np.ones((nyears, ysize, xsize))
    min = np.ones((nyears, ysize, xsize))
    
    
    
    #ff = ds['displacement'][:, y:y+5, x:x+5].rolling(time=10, center=True).median().dropna("time")


    FILL_VALUE = -9999
    mask = ds['displacement'][0].values != FILL_VALUE
    ys, xs = np.where(mask)
    k = 0
    npixels = mask.sum()
    for y, x in tqdm(zip(ys, xs)):
        if mask[y,x]:
            df = ds['displacement'][:, y, x].to_series()
            _max, _min = min_max(df)
            max[:, y, x]=_max.values
            min[:, y, x]=_min.values
            #print(f'{(float(k/npixels)*100):.03f}%')
            k+=1

    # write the data
    outDrv = gdal.GetDriverByName('GTiff')
    ds_max = outDrv.Create("max.tif",
                        xsize, ysize,
                        3, gdal.GDT_Float32, options=['COMPRESS=LZW', 'BIGTIFF=YES' ] )
    ds_max.SetGeoTransform( ds.rio.transform().to_gdal())
    ds_max.SetProjection(PROJECTION)

    ds_min = outDrv.Create("min.tif",
                        xsize, ysize,
                        3, gdal.GDT_Float32, options=['COMPRESS=LZW','BIGTIFF=YES'] )   

    ds_min.SetGeoTransform( ds.rio.transform().to_gdal())
    ds_min.SetProjection(PROJECTION)

    for k in range(3):
        ds_max.GetRasterBand(k+1).WriteArray(max[k])
        ds_min.GetRasterBand(k+1).WriteArray(min[k])

    ds_max = None 
    ds_min = None