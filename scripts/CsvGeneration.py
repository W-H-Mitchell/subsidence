#import os
from osgeo import gdal
import numpy as np
import pandas as pd
import time
import scipy.stats as st
from scipy.stats import t

def normalize(data, accuracy, nodata_accuracy):
    data[data <= -9999] = 0 # make no data values equal to zero
    return (accuracy-nodata_accuracy)*((data - np.min(data))/(np.max(data)-np.min(data)))+nodata_accuracy

def confidence_interval(data, confidence=0.95):
    m = x.mean()
    s = data.std()
    n = len(x)-1
    t_crit = np.abs(t.ppf((1-confidence)/2,n))
    return m-s*t_crit/np.sqrt(len(x)), m+s*t_crit/np.sqrt(len(x))

def lat_lon(tif):
    ds = gdal.Open(tif)
    dem = gdal.Open("tifs/uk_dem_wgs84_0.0008.tif").ReadAsArray()
    geot = ds.GetGeoTransform()
    subs =  ds.ReadAsArray()
    y, x = np.where(dem > -1000)
    print("Displacement shape shape: {0}".format(subs.shape))
    sea = dem <= -1000
    subs[sea] = -9999
    subs = subs[subs > -9999]
    subs.flatten()
    print("Displacement masked array shape: {0}".format(subs.shape))
    xoff, a, b, yoff, d, e = geot
    longitude = (a * x + b * y + xoff).astype(np.float32)
    latitude = (d * x + e * y + yoff).astype(np.float32)
    return subs, longitude, latitude

def output(disp_tif, yr, rcp):
    subs, longitude, latitude = lat_lon(disp_tif)
    dinsar = gdal.Open("tifs/reproject_v2dinsar copy.tif").ReadAsArray()
    dem = gdal.Open("tifs/uk_dem_wgs84_0.0008.tif").ReadAsArray()
    clim = gdal.Open("tifs/rainfall_verification.tif").ReadAsArray()
    sea = dem <= -1000
    clim[sea] = -9999
    dinsar[sea] = -100000 # dinsar has no data val of -9999 so mask needs to be different
    clim = clim[clim > -9999]
    dinsar = dinsar[dinsar > -100000]
    mask = dinsar == -9999 # no data value
    print("Latitude, longitude shape: {0}, {1}".format(latitude.shape, longitude.shape))
    print("Climate masked array shape: {0}".format(clim.shape))

    coher = gdal.Open("tifs/coherence_quality_sum.tif").ReadAsArray()
    coher[sea] = -10000
    coher = coher[coher > -10000]
    coher = normalize(coher, 0.947, 0.72) # normalise between the accuracy and the no data accuracy
    print("Coherence masked array shape: {0}".format(coher.shape))
    out_data_frame = pd.DataFrame({'Longitude': longitude, 'Latitude': latitude, 'climate reliability':clim,
                                   'severity value': subs, 'hazard reliability':coher})
    out_data_frame['climate reliability'][out_data_frame['climate reliability'] > 1] = 0.94683767253075

    # confidence interval
    confidence_limits = st.t.interval(0.95, len(subs)-1, loc=np.mean(subs), scale=st.sem(subs)) # sem=standard error of measurement
    lower, upper = confidence_limits[0], confidence_limits[1]
    half_confidence_range = (upper - lower)/2

    # output dataframe
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
    out_data_frame['severity range']= half_confidence_range
    out_data_frame['metadata filename']= outname+'.txt'
    out_data_frame['confidence percentile']= 95
    out_data_frame['month or season']='annual'
    out_data_frame['scenario']= rcp
    out_data_frame['year']= yr
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
    assert out_frame['severity value'].max() <= 3 # check the subsidence range
    assert out_frame['severity value'].min() >= -3 # check the subsidence range
    
    # OUTPUT
    out_frame.to_csv("s3outputs/{0}_{1}.csv".format(outname, rcp), index=False)

#rcp85
output('rcp85outputs/rcp85_2020_bl_nopr.tif', '2020', 'rcp85')
output('rcp85outputs/rcp85_2025-2034_notemp_geo.tif', '2030', 'rcp85')
output('rcp85outputs/rcp85_2035-2044_notemp_geo.tif', '2040', 'rcp85')
output('rcp85outputs/rcp85_2045-2054_notemp_geo.tif', '2050', 'rcp85')
output('rcp85outputs/rcp85_2055-2064_notemp_geo.tif', '2060', 'rcp85')
output('rcp85outputs/rcp85_2065-2074_notemp_geo.tif', '2070', 'rcp85')
output('rcp85outputs/rcp85_2075-2084_notemp_geo.tif', '2080', 'rcp85')
#rcp26
output('rcp85outputs/rcp85_2020_bl_nopr.tif', '2020', 'rcp26')
output('rcp85outputs/rcp85_2020_bl_nopr.tif', '2030', 'rcp26')
output('rcp85outputs/rcp85_2020_bl_nopr.tif', '2040', 'rcp26')
output('rcp85outputs/rcp85_2020_bl_nopr.tif', '2050', 'rcp26')
output('rcp85outputs/rcp85_2020_bl_nopr.tif', '2060', 'rcp26')
output('rcp85outputs/rcp85_2020_bl_nopr.tif', '2070', 'rcp26')
output('rcp85outputs/rcp85_2020_bl_nopr.tif', '2080', 'rcp26')
output('rcp85outputs/rcp85_2020_bl_nopr.tif', '2090', 'rcp26')
output('rcp85outputs/rcp85_2020_bl_nopr.tif', '2100', 'rcp26')
