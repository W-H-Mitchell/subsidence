import os
import glob
import zarr
import shutil
import hvplot
import zipfile
import imageio
import numpy as np
import pandas as pd
import xarray as xr
import hvplot.xarray
from tqdm import tqdm
import holoviews as hv
from osgeo import gdal
from datetime import datetime
import matplotlib.pyplot as plt
#os.chdir("Documents/aws/subsidence/sbas")


def DateToColumnHeader(data, outcsv):
    f = open(data, 'r')
    cols = ['lat', 'long', 'topo', 'Vel', 'coh', 'cosN', 'cosE', 'cosU',]
    headers = [f'header{k}' for k in range(43)] 
    dic = {}
    for n in headers:
        dic[n] = f.readline()
    dates = (dic['header39'])
    dates = dates[15:].split(', ')
    cols += dates
    df = pd.read_csv(data, skiprows=43, index_col=0, names=cols)
    acquisitions = []
    for date in dates:
        df[date[:-1]] = df[date]*df['cosU']
        df = df.drop(columns=date)
        acquisitions.append(date[:-1])
    df.to_csv(outcsv, index=False)
    return df, acquisitions

def LOStoVertical(data, dates, out):
    for day in dates:
        df[day] = df[f"DT_{day}"]*df['cosU']
        df = df.drop(column=f"DT_{day}")
    df.to_csv(out, index=False)
    return df, dates

def RandomToTimeSeries(indf, rand_samplesize):
    cols = ['long', 'lat', 'topo', 'Vel', 'coh', 'cosN', 'cosE', 'cosU',]
    indf = indf.drop(columns=cols)
    indf = indf.sample(rand_samplesize)
    outdf = indf.T
    outdf.index = pd.to_datetime(outdf.index)
    return outdf

def SelectedToTimeSeries(indf, min_lat, min_long, max_lat, max_long):
    indf = indf[(indf['lat']>min_lat) & (indf['lat']<max_lat)]
    indf = indf[(indf['long']>min_long) & (indf['long']<max_long)]
    cols = ['long', 'lat', 'topo', 'Vel', 'coh', 'cosN', 'cosE', 'cosU',]
    indf = indf.drop(columns=cols)
    outdf = indf.T
    outdf.index = pd.to_datetime(outdf.index)
    return outdf

def LoopTimeSeriesPlot(TransposeDf, outplot):
    for col in TransposeDf.columns:
        TransposeDf[col].plot(legend=False, rot=90, c='b', lw=1,  ylim=(-2,2))
        plt.gcf()
        plt.show()
        #plt.save(f"sbas/{outplot}")
    
def TimeSeriesPlot(TransposeDf, outplot):
    TransposeDf.plot(legend=False, rot=90, c='b', alpha=0.1, lw=1, ylim=(-2,2))
    plt.gcf()
    plt.show()
    #plt.save(f"sbas/{outplot}")

def gif(df, dates, outgif):
    filenames = []
    for t in dates:
        plt.scatter(x=df['lat'], y=df['long'], c=df[t], 
                    cmap='RdYlBu_r', s=0.3)
        filename = f"sbas/Step_{t}.png"
        filenames.append(filename)
        plt.gcf()
        plt.savefig(filename) #save the current figure
        plt.close()
    
    # build gif
    with imageio.get_writer(f'sbas/{outgif}', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
        
def find_filenames(path_to_dir, suffix=".csv"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]

def DisplacementRasters(csv_folder, dates, ref_raster):
    # Open raster, get extent and resolution (raster will not be overwritten)
    ras_ds = gdal.Open(ref_raster)
    geo = ras_ds.GetGeoTransform()
    srs = ras_ds.GetProjection()
    x, y = ras_ds.RasterXSize, ras_ds.RasterYSize 
    x_res = geo[1]
    y_res = geo[5]
    min_x = geo[0]
    min_y = geo[3] + x * geo[4] + y * geo[5]
    max_x = geo[0] + x * geo[1] + y * geo[2]
    max_y = geo[3]
    extent = [min_x, min_y, max_x, max_y]
    csvfiles = find_filenames(csv_folder)
    for fn in csvfiles:
        vrt_fn = fn.replace(".csv", ".vrt") #this must be the same as the layer below
        lyr_name = fn.replace('.csv', '')
        with open(vrt_fn, 'w') as fn_vrt:
            fn_vrt.write('<OGRVRTDataSource>\n')
            fn_vrt.write('\t<OGRVRTLayer name="%s">\n' %lyr_name)
            fn_vrt.write('\t\t<SrcDataSource>%s</SrcDataSource>\n' %fn)
            fn_vrt.write('\t\t<GeometryType>wkbPoint</GeometryType>\n')
            fn_vrt.write('\t\t<GeometryField encoding="PointFromColumns" x="long" y="lat"/>\n')
            fn_vrt.write('\t</OGRVRTLayer>\n')
            fn_vrt.write('</OGRVRTDataSource>\n')
    print("Rasterising vrt")
    for date in dates:
        rs_options = gdal.RasterizeOptions(format="GTiff", outputBounds=extent, outputSRS=srs, 
                                           noData=-9999, attribute=f'{date}', xRes=x_res, yRes=y_res, 
                                           creationOptions=['BIGTIFF=YES','BLOCKXSIZE=512','TILED=YES',
                                                            'NUM_THREADS=ALL_CPUS','SPARSE_OK=TRUE'])
        out_tif = f"tifs/{lyr_name}_{date}.tif"
        output = gdal.Rasterize(f"{out_tif}", f"{vrt_fn}", options=rs_options)

def RasterStackToXrDs(tif_folder, nc, zr, zip_nc, zip_zr):
    tifs = find_filenames(tif_folder, '.tif')
    tifs = sorted(tifs)
    
    datasets = []
    
    for file in tifs:
        dt = np.datetime64(file[38:57])
        da = xr.open_rasterio(f"tifs/{file}")[0]
        da = da.astype('float32')
        ds = da.to_dataset(name='displacement')
        ds.coords['time'] = dt
        
        datasets.append(ds)
        
    dset = xr.concat(datasets, dim='time')
    dset = dset.where(dset['displacement'] != -9999.)
    #comp = dict(zlib=True, complevel=5)
    #encoding = {var: comp for var in ds.data_vars}
    dset.to_netcdf(f"{nc}.nc") #encoding=encoding
    with zipfile.ZipFile(zip_nc,'w') as fn_zip:
        fn_zip.write(f"{nc}.nc", compress_type=zipfile.ZIP_DEFLATED)
    dset.to_zarr(f"{zr}.zarr")
    shutil.make_archive(zip_zr,"zip", f"{zr}.zarr")

def min_max(ts):
    med = ts.rolling(window='60d', center=True).median()
    year_max = med.groupby(ts.index.year).max()
    year_min = med.groupby(ts.index.year).min()
    return year_max, year_min

def TimeSeriesMedian(xr_zarr):
    ds = xr.open_dataset(xr_zarr)
    nT, ysize, xsize = ds['displacement'].shape

    nyears = 3
    max = np.ones((nyears, ysize, xsize))
    min = np.ones((nyears, ysize, xsize))

    # ff = ds['displacement'][:, y:y+5, x:x+5].rolling(time=10, center=True).median().dropna("time")

    FILL_VALUE = -9999
    mask = ds['displacement'][0].values != FILL_VALUE
    ys, xs = np.where(mask)
    k = 0
    npixels = mask.sum()
    for y, x in tqdm(zip(ys, xs)):
        if mask[y, x]:
            df = ds['displacement'][:, y, x].to_series()
            _max, _min = min_max(df) # function called
            max[:, y, x] = _max.values
            min[:, y, x] = _min.values
            # print(f'{(float(k/npixels)*100):.03f}%')
            k += 1

    # write the data
    outDrv = gdal.GetDriverByName('GTiff')
    ds_max = outDrv.Create("max.tif",
                           xsize, ysize,
                           3, gdal.GDT_Float32, options=['COMPRESS=LZW', 'BIGTIFF=YES'])
    ds_max.SetGeoTransform(ds.rio.transform().to_gdal())
    ds_max.SetProjection(PROJECTION)

    ds_min = outDrv.Create("min.tif",
                           xsize, ysize,
                           3, gdal.GDT_Float32, options=['COMPRESS=LZW', 'BIGTIFF=YES'])

    ds_min.SetGeoTransform(ds.rio.transform().to_gdal())
    ds_min.SetProjection(PROJECTION)

    for k in range(3):
        ds_max.GetRasterBand(k + 1).WriteArray(max[k])
        ds_min.GetRasterBand(k + 1).WriteArray(min[k])

    ds_max = None
    ds_min = None



### Function Calls ###
csv_root = 'csvs/'
files = ['DTLOS_hmitchellclimatex_20161018_20190921_C5HF.csv',
         'DTLOS_hmitchellclimatex_20161229_20190912_C5HF.csv']

RasterStackToXrDs('tifs/', 'JBTk81', 'JBt81','JBTk81nc.zip', 'JBt81zr.zip')
