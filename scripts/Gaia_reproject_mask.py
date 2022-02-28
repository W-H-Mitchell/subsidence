import os
from osgeo import gdal

def reproj_climate(ncdf, lyr, tif, clipped_tif):
    """
    :param ncdf: input netcdf to be transformed; if raster remove the ":lyr" from command (.nc)
    :param lyr: layer within netcdf to be transformed
    :param tif: output reprojected tif filename (.tif)
    :param clipped_tif: the clipped version of the climate raster output
    :return:
    """
    # dem reference raster
    reference_raster = gdal.Open(***DEM GOES HERE***)  # raster to use as reference array
    geo = reference_raster.GetGeoTransform()
    xsize, ysize = reference_raster.RasterXSize, reference_raster.RasterYSize
    x_res = geo[1]
    y_res = geo[5]
    min_x = geo[0]
    min_y = geo[3] + xsize * geo[4] + ysize * geo[5]
    max_x = geo[0] + xsize * geo[1] + ysize * geo[2]
    max_y = geo[3]
    # resample the file
    cmd = f"gdalwarp -overwrite -of  GTiff -t_srs 'epsg:4326' -tr {x_res} {y_res} -te {min_x} {min_y} {max_x} {max_y} NETCDF:'{ncdf}':{lyr} {tif}"
    os.system(cmd)

    # mask the output file
    dem = reference_raster.ReadAsArray()
    clim = gdal.Open(tif).ReadAsArray() # load the above output as an array
    mask = dem == -32768 # mask is array positions where dem = no data
    clim[mask] = -32768 # set same no data for array positions in climate tif

    # save the masked array as a raster
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.CreateCopy(clipped_tif, reference_raster, strict=0)
    dst_ds.GetRasterBand(1).WriteArray(clim)
    dst_ds.SetGeoTransform(geo)
    band = dst_ds.GetRasterBand(1)
    band.SetNoDataValue(-32768)
    dst_ds = None
