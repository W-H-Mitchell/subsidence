import os
from osgeo import gdal

# Functions
def reproj_climate(inp, tif):
    # dem reference raster
    reference_raster = gdal.Open("tifs/uk_dem_wgs84_0.0008.tif")  # raster to use as reference array
    geo = reference_raster.GetGeoTransform()
    xsize, ysize = reference_raster.RasterXSize, reference_raster.RasterYSize
    x_res = geo[1]
    y_res = geo[5]
    min_x = geo[0]
    min_y = geo[3] + xsize * geo[4] + ysize * geo[5]
    max_x = geo[0] + xsize * geo[1] + ysize * geo[2]
    max_y = geo[3]
    cmd = f"gdalwarp -t_srs 'epsg:4326' -tr {x_res} {y_res} -te {min_x} {min_y} {max_x} {max_y} {inp} {tif}"
    os.system(cmd)

def add_rasters(in1, in2, in3, in4, out):
    cmd = f"gdal_calc.py -A {in1} -B {in2} -C {in3} -D {in4} --outfile={out} --calc='A+B+C+D'"
    os.system(cmd)

def mask(clay_raster, dem_raster, output):
    clay = gdal.Open(clay_raster).ReadAsArray()
    ds = gdal.Open(dem_raster)
    geo = ds.GetGeoTransform()
    dem = ds.ReadAsArray()
    mean = clay.mean()
    nan = dem <= -1000
    clay[nan] = -9999
    negs = clay == -32768
    clay[negs] = mean

    # save the masked array as a raster
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.CreateCopy(output, ds, strict=0)
    dst_ds.GetRasterBand(1).WriteArray(clay)
    dst_ds.SetGeoTransform(geo)
    band = dst_ds.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    dst_ds = None

add_rasters("clay/clay_0-5cm1000wsg84.tif", "clay/clay_5-15cm1000wsg84.tif",
            "clay/clay_15-30cm1000wsg84.tif", "clay/clay_30-60cm1000wsg84.tif",
            "clay/clay_60cm_1000.tif")
reproj_climate("clay/clay_60cm_1000.tif", "clay/clay_60cm_wgs84.tif")
mask("clay/clay_60cm_wgs84.tif", "tifs/uk_dem_wgs84_0.0008.tif", "clay/clay_wgs84_clip.tif")