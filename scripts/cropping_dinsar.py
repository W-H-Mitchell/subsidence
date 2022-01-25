import os
from osgeo import gdal

os.chdir("Documents/aws/subsidence")

# dem reference raster
reference_raster = gdal.Open("tifs/final_dem.tif")
geo = reference_raster.GetGeoTransform()
xsize, ysize = reference_raster.RasterXSize, reference_raster.RasterYSize
x_res = geo[1]
y_res = geo[5]
min_x = geo[0]
min_y = geo[3] + xsize * geo[4] + ysize * geo[5]
max_x = geo[0] + xsize * geo[1] + ysize * geo[2]
max_y = geo[3]

#cmd = f"gdalwarp -overwrite -srcnodata 0 -dstnodata -9999 -tr {x_res} {y_res} -cutline tifs/hi_coh_clip.shp -crop_to_cutline tifs/mean_displacement.tif tifs/crop1.tif"
#os.system(cmd)
cmd = f"gdalwarp -overwrite -dstnodata -9999 -tr {x_res} {y_res} -cutline tifs/dis_clip.shp -crop_to_cutline tifs/displacement_descending.tif tifs/crop2_adjusted.tif"
os.system(cmd)
#cmd = f"gdal_merge.py -ps {x_res} {y_res} -n 0 -a_nodata -9999 -o tifs/merg_disp.tif tifs/crop1_adj.tif ttifs/crop2_adjusted.tif"
#os.system(cmd)
cmd = f"gdalwarp -overwrite -srcnodata 0 -dstnodata -9999 -tr {x_res} {y_res} -te {min_x} {min_y} {max_x} {max_y} tifs/v2_dinsar_merg.tif tifs/reproj_merg_adj.tif"
os.system(cmd)