import os
import glob
import zarr
import shutil
import imageio
import numpy as np
import pandas as pd
import xarray as xr
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
    df.to_csv(outcsv, index=False)
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
    
def TimeSeriesPlot(TransposeDf, outplot):
    TransposeDf.plot(legend=False, rot=90, c='b', alpha=0.01, lw=1)
    plt.gcf()
    plt.show()
    #plt.save(f"sbas/{outplot}")

def gif(df, dates, outgif):
    filenames = []
    for t in dates:
        plt.scatter(x=df['lat'], y=df['long'], c=df[t], cmap='RdYlBu_r', s=0.3)
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

def RasterStackToXrDs(tif_folder, track_prefix, zipped_fn):
    tifs = find_filenames(tif_folder, '.tif')
    tifs = sorted(tifs)
    
    datasets = []
    
    for file in tifs:
        dt = np.datetime64(file[38:57])
        da = xr.open_rasterio(f"tifs/{file}")
        da = da.astype('float32')
        ds = da.to_dataset(name='displacement')
        ds.coords['time'] = dt
        
        datasets.append(ds)
        
    dset = xr.concat(datasets, dim='time')
    #comp = dict(zlib=True, complevel=5)
    #encoding = {var: comp for var in ds.data_vars}
    dset.to_netcdf(f"{track_prefix}.nc") #encoding=encoding
    dset.to_zarr(f"{track_prefix}.zarr")
    shutil.make_archive(zipped_fn,'zip',f"{track_prefix}.zarr")
    return dset


def RasterStackToNetCDF(tif_folder, track_prefix, zipped_fn):
    files = find_filenames(tif_folder, '.tif')
    sort_files = sorted(files)
    
    #collecting datasets when looping over your files
    list_da = []
    
    for path in sort_files:
        #path = "tifs/DTSLOS_20170122_20190828_D79H_2017-02-15T06:13:38Z.tif"
        da = xr.open_rasterio(f"tifs/{path}")
        
        time = path.split("_")[-1].split("Z")[0]
        dt = datetime.strptime(time,"%Y-%m-%dT%H:%M:%S")
        dt = pd.to_datetime(dt)
        
        da = da.assign_coords(time = dt)
        da = da.expand_dims(dim="time")
        
        list_da.append(da)
    
    ds = xr.combine_by_coords(list_da)
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(f"{track_prefix}.nc", encoding=encoding) #
    #shutil.make_archive(zipped_fn,'zip',f"{track_prefix}.nc")


### Calls ###
# test, dates = DateToColumnHeader("DTSLOS_fpacini_20170122_20190828_D79H.csv",
#                                  "DTSLOS_fpacini_20170122_20190828_D79H.csv")
# cwd = str(os.getcwd())
# DisplacementRasters(cwd, dates, "uk_dem_wgs84_0.0008.tif")
RasterStackToXrDs('tifs/', 'Track81JB_new', 'Track81Test')   
