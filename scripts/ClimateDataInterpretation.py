# imports
#import geopandas as gpd
#import descartes
import matplotlib as mpl
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import glob
import numpy as np
import iris
import iris.analysis as ia
import iris.quickplot as qplt
import pandas as pd
import pylab
import time

root_path = "/home/ubuntu/climate_data" # root path to folder
# ensemble file paths
temp_model = "/tas_monthly"
rain_model = "/monthly_pr_mod"
temp_obs = "/obs_tas"
rain_obs = "/obs_pr"


### Mean rainfall from historical observed ###
ens_path = glob.glob(root_path+rain_obs+'/*.nc')
months = np.arange(1, 13, 1)
mn_name = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
           'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
for month, name in zip(months, mn_name):
    obs_pr = iris.load(ens_path)
    obs_pr = obs_pr.concatenate(iris.util.equalise_attributes(obs_pr))[0]
    mon_constraint = iris.Constraint(coord_values={'month_number': month})
    mn_pr = obs_pr.extract(mon_constraint).aggregated_by('month_number', ia.MEAN)
    iris.save(mn_pr,'outputs/HistoricalAverageRainfall/PrecipitationObservedHistorical_'+str(month)+name+'.nc')


### Deviatoric modelled rainfall - monthly but want to downscale to daily ###
ens_path = glob.glob(root_path+rain_model+'/*.nc')
mn_avers = sorted(glob.glob("outputs/HistoricalAverageRainfall/PrecipitationObservedHistorical_*"))
years=np.arange(2020,2080,5)
for year in years:
    yr_constraint = iris.Constraint(coord_values={'year':lambda yr:year-5<=yr<=year+4})
    obs_pr = iris.load(ens_path, yr_constraint)
    obs_pr = obs_pr.concatenate(iris.util.equalise_attributes(obs_pr))[0]
    months = np.arange(1, 13, 1)
    for month in months:
        mon_constraint = iris.Constraint(coord_values={'month_number': month})
        mn_pr = obs_pr.extract(mon_constraint).aggregated_by('month_number', ia.MEAN)
        mn_av = iris.load(mn_avers[month-1])

        dev_pr=mn_pr-mn_av

        iris.save(dev_pr,'outputs/DeviatoricRainfall/PrecipitationObservedHistorical_'+str(year)+str(month)+name+'.nc')


"""
### Days over 20 degrees over 3 years ###
ens_path = glob.glob(root_path+temp_model+'/*.nc')

# mon_constraint=iris.Constraint(coord_values={'month_number': lambda mn: 3<= mn<=11})
# years=np.arange(2020,2080,5)
for year in years:
    yr_constraint = iris.Constraint(coord_values={'year': lambda yr: year-5<=yr<=year+4})
    tri_yr_constraint = iris.Constraint(coord_values={'year': lambda yr: year-3 <= yr})
    ens_tas = iris.load(ens_path, yr_constraint) # yr_constraint, mon_constraint
    ens_tas = ens_tas.concatenate(iris.util.equalise_attributes(ens_tas))[0]
    ndays_yr = ens_tas.aggregated_by('year', iris.analysis.COUNT, function=lambda values: values <= 20)
    tot_days = ndays_yr.extract(tri_yr_constraint).aggregated_by('year', ia.SUM)
    ens_10yr_mean = ndays_yr.collapsed(['time','ensemble_member'], ia.MEAN)

    iris.save(ndays_yr, root_path+out_test+'/'+'test_tas20deg_nodays.nc')


### Total precipitation ###
ens_path = glob.glob(root_path+rain_model+'/*.nc')
years = np.arange(2020, 2101, 5)
for year in years:

    # Calculate the precipitation for 2 years before the target year
    YrConstraint = iris.Constraint(coord_values={'year': lambda yr: year - 5 <= yr <= year + 4})
    TwoYrConstraint = iris.Constraint(coord_values={'year': lambda yr: year - 3 <= yr <= year - 1})
    ens_yr = iris.load(ens_path, YrConstraint)
    ens_yr = ens_yr.concatenate(iris.util.equalise_attributes(ens_yr))[0]
    ens_tot = ens_yr.extract(TwoYrConstraint).aggregated_by('year', ia.SUM)
    ens_2yr_tot = ens_tot.collapsed(['time', 'ensemble_member'], ia.MEAN)
    # ens_10yr_mean.units = 'mm/year'

    # Calculate the precipitation for the first 9 months of the target year
    mon_constraint=iris.Constraint(coord_values={'month_number': lambda mn: mn<=9})
    ens_yr = iris.load(ens_path, yr_constraint)
    ens_yr = ens_yr.concatenate(iris.util.equalise_attributes(ens_yr))[0]
    ens_yr_tot = ens_yr.extract(mon_constraint).aggregated_by('year', ia.SUM)
    ens_9mn_tot = ens_tot.collapsed(['time', 'ensemble_member'])
"""