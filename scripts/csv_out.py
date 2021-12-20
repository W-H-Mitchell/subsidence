#import os
from osgeo import gdal
import pandas as pd
import time

#os.chdir('C:/Users/HamishMitchell/uk_geophys/landslide')
def xyz(tif, xyz):
    ds = gdal.Open(tif)
    ly = ds.GetRasterBand(1)
    ly.SetNoDataValue(0)
    f = gdal.Translate(xyz, ds)
    f = None

def output(xyz, disp_tiff, yr, rcp):
    data_in = pd.read_csv(xyz, delimiter=' ', )
    data_in.columns = ['Longitude','Latitude','severity value']
    out_data_frame = data_in[data_in['severity value']>-10]
    print("CSV modelled displacement shape shape: {0}".format(out_data_frame.shape))
    # masked hazard reliability
    displacement = gdal.Open(disp_tiff).ReadAsArray()
    dem = gdal.Open('tifs/dem_gb.tif').ReadAsArray()
    sea = dem == -9999
    displacement[sea] = -100000
    displacement = displacement[displacement > -100000]
    displacement = displacement.flatten()
    mask = displacement == -9999
    print("Displacment masking array shape: {0}".format(displacement.shape))

    reliability = 0.97
    no_data_reliability = 0.73
    out_frame=pd.DataFrame(columns=
                          ['Latitude',
                           'Longitude',
                           'grid size (m)',
                           'year',
                           'month or season',
                           'scenario',
                           'hazard type',
                           'severity metric',
                           'severity value',
                           'severity range',
                           'likelihood',
                           'return time',
                           'confidence percentile',
                           'likelihood percentile value',
                           'return time percentile value',
                           'climate reliability',
                           'hazard reliability',
                           'metadata filename'
                           ])
    out_date = time.strftime('%Y%m%d') 
    outname= "{0}_subsidence_UK_climate_conditioned_{1}".format(out_date, yr)
    out_data_frame['grid size (m)']= 90
    out_data_frame['hazard type']= 'subsidence'
    out_data_frame['severity metric']= 'displacement cm/yr'
    out_data_frame['severity range']= 'nan'
    out_data_frame['hazard reliability'] = reliability
    out_data_frame['hazard reliability'][mask] = no_data_reliability
    out_data_frame['metadata filename']= outname+'.txt'
    out_data_frame['confidence percentile']=95
    out_data_frame['month or season']='annual'
    out_data_frame['scenario']=rcp   
    out_data_frame['year']=yr    
    out_data_frame['likelihood']= 0.9
    out_data_frame['return time'] ='nan'
    out_data_frame['likelihood percentile value'] = 0.1
    out_data_frame['return time percentile value'] = 'nan'
    out_data_frame['climate reliability'] = 0.9
    out_frame=out_frame.append(out_data_frame)

    # TESTS
    assert len(out_frame.columns) == 18
    assert out_frame['Latitude'].min() >= 40
    assert out_frame['Longitude'].max() <= 10
    assert out_frame['severity value'].max() <= 5 # check the subsidence range
    assert out_frame['severity value'].min() >= -10 # check the subsidence range
    
    # OUTPUT
    out_frame.to_csv("s3outputs/{0}_{1}.csv".format(outname, rcp), index=False)

# generate xyz rcp26
#xyz('rcp85outputs/rcp85_2015-2024.tif', 'rcp85xyz/2015-2024.xyz')
#xyz('rcp85outputs/rcp85_2020-2029.tif', 'rcp85xyz/2020-2029.xyz')
#xyz('rcp85outputs/rcp85_2025-2034.tif', 'rcp85xyz/2025-2034.xyz')
#xyz('rcp85outputs/rcp85_2030-2039.tif', 'rcp85xyz/2030-2039.xyz')
#xyz('rcp85outputs/rcp85_2035-2044.tif', 'rcp85xyz/2035-2044.xyz')
#xyz('rcp85outputs/rcp85_2040-2049.tif', 'rcp85xyz/2040-2049.xyz')
#xyz('rcp85outputs/rcp85_2045-2054.tif', 'rcp85xyz/2045-2054.xyz')
#xyz('rcp85outputs/rcp85_2050-2059.tif', 'rcp85xyz/2050-2059.xyz')
#xyz('rcp85outputs/rcp85_2055-2064.tif', 'rcp85xyz/2055-2064.xyz')
#xyz('rcp85outputs/rcp85_2060-2069.tif', 'rcp85xyz/2060-2069.xyz')
#xyz('rcp85outputs/rcp85_2065-2074.tif', 'rcp85xyz/2065-2074.xyz')
#xyz('rcp85outputs/rcp85_2070-2079.tif', 'rcp85xyz/2070-2079.xyz')
#output('rcp85xyz/2015-2024.xyz', 'tifs/displacement_cmyr.tif', '2020', 'rcp85')
#output('rcp85xyz/2020-2029.xyz', 'tifs/displacement_cmyr.tif', '2025', 'rcp85')
#output('rcp85xyz/2025-2034.xyz', 'tifs/displacement_cmyr.tif', '2030', 'rcp85')
#output('rcp85xyz/2030-2039.xyz', 'tifs/displacement_cmyr.tif', '2035', 'rcp85')
#output('rcp85xyz/2035-2044.xyz', 'tifs/displacement_cmyr.tif', '2040', 'rcp85')
#output('rcp85xyz/2040-2049.xyz', 'tifs/displacement_cmyr.tif', '2045', 'rcp85')
#output('rcp85xyz/2045-2054.xyz', 'tifs/displacement_cmyr.tif', '2050', 'rcp85')
#output('rcp85xyz/2050-2059.xyz', 'tifs/displacement_cmyr.tif', '2055', 'rcp85')
#output('rcp85xyz/2055-2064.xyz', 'tifs/displacement_cmyr.tif', '2060', 'rcp85')
#output('rcp85xyz/2060-2069.xyz', 'tifs/displacement_cmyr.tif', '2065', 'rcp85')
#output('rcp85xyz/2065-2074.xyz', 'tifs/displacement_cmyr.tif', '2070', 'rcp85')
#output('rcp85xyz/2070-2079.xyz', 'tifs/displacement_cmyr.tif', '2075', 'rcp85')
#output('rcp85xyz/2070-2079.xyz', 'tifs/displacement_cmyr.tif', '2080', 'rcp85')

# rcp26
output('rcp85xyz/2015-2024.xyz', 'tifs/displacement_cmyr.tif', '2020', 'rcp26')
output('rcp85xyz/2015-2024.xyz', 'tifs/displacement_cmyr.tif', '2025', 'rcp26')
output('rcp85xyz/2015-2024.xyz', 'tifs/displacement_cmyr.tif', '2030', 'rcp26')
output('rcp85xyz/2015-2024.xyz', 'tifs/displacement_cmyr.tif', '2035', 'rcp26')
output('rcp85xyz/2015-2024.xyz', 'tifs/displacement_cmyr.tif', '2040', 'rcp26')
output('rcp85xyz/2015-2024.xyz', 'tifs/displacement_cmyr.tif', '2045', 'rcp26')
output('rcp85xyz/2015-2024.xyz', 'tifs/displacement_cmyr.tif', '2050', 'rcp26')
output('rcp85xyz/2015-2024.xyz', 'tifs/displacement_cmyr.tif', '2055', 'rcp26')
output('rcp85xyz/2015-2024.xyz', 'tifs/displacement_cmyr.tif', '2060', 'rcp26')
output('rcp85xyz/2015-2024.xyz', 'tifs/displacement_cmyr.tif', '2065', 'rcp26')
output('rcp85xyz/2015-2024.xyz', 'tifs/displacement_cmyr.tif', '2070', 'rcp26')
output('rcp85xyz/2015-2024.xyz', 'tifs/displacement_cmyr.tif', '2075', 'rcp26')
output('rcp85xyz/2015-2024.xyz', 'tifs/displacement_cmyr.tif', '2080', 'rcp26')
output('rcp85xyz/2015-2024.xyz', 'tifs/displacement_cmyr.tif', '2085', 'rcp26')
output('rcp85xyz/2015-2024.xyz', 'tifs/displacement_cmyr.tif', '2090', 'rcp26')
output('rcp85xyz/2015-2024.xyz', 'tifs/displacement_cmyr.tif', '2095', 'rcp26')
output('rcp85xyz/2015-2024.xyz', 'tifs/displacement_cmyr.tif', '2100', 'rcp26')