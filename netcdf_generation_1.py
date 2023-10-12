"""
Code to create the files from ERA5 and CMIP6 models 

Part 1

Author: Isabel 
"""

#libreries
import netCDF4 as nc
import xarray as xr
import numpy as np
import numpy.ma as ma
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import datetime as dt
import pandas as pd
import os
from scipy import interpolate
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy import integrate, stats
from matplotlib.pyplot import cm
from windspharm.standard import VectorWind
import matplotlib.patches as mpatches
from Functions import *
import pickle
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
sns.set(style="white")
sns.set_context('notebook', font_scale=1.5)

#Path save is the path to save all the files from the calculations
#CHANGE path save
# Modified by PHWeill 2023/07/15
#path_save='/home/iccorreasa/Documentos/Paper_CMIP6_models/PLOTS_paper/PAPER_FINAL/npz/' #CHANGE
path_save='/scratchx/lfita/'
# End Modif

#-------------------------------------------------------------------------------------------------------
#1. ERA5

"""
path_entry_ini='/bdd/ERA5/NETCDF/GLOBAL_025/1xmonthly/'

#-------------------------------------------------------------------------------------------------------  

#list_variables=['u','v','q','msl','geopt','sstk','ta','t2m', 'mtpr','w']
list_variables=['u','v','q','msl','geopt','sstk','ta','t2m', 'w']

lat_limits_global=[-70,50]
lon_limits_global=[-155,-5]

lat_limits_full=[-90,90]
lon_limits_full=[0,360]

initial_time='19790101'
final_time='20141201'

p_level_interest_upper=10000.0
p_level_interest_lower=100000.0


for i in range(len(list_variables)):

    var_sp=list_variables[i]

    if var_sp=='u' or var_sp=='v' or var_sp=='w':
        path_entry=path_entry_ini+'AN_PL/'
        gridsize_x, gridsize_y=netcdf_creation_original_ERA5(path_entry,var_sp,lat_limits_full,lon_limits_full,\
                                                        p_level_interest_lower,p_level_interest_upper,\
                                                            'ERA5','Yes',path_save)
        
    elif var_sp=='q' or var_sp=='ta' or var_sp=='geopt':
        path_entry=path_entry_ini+'AN_PL/'
        gridsize_x, gridsize_y=netcdf_creation_original_ERA5(path_entry,var_sp,lat_limits_global,lon_limits_global,\
                                                        p_level_interest_lower,p_level_interest_upper,\
                                                            'ERA5','Yes',path_save)
            
    elif var_sp=='msl' or var_sp=='mtpr' or var_sp=='t2m' or var_sp=='sstk':
# Modified by PHWeill 2023/07/15
#        path_entry=path_entry_ini+'AN_SL/'
        path_entry=path_entry_ini+'AN_SF/'
# End Modif
        gridsize_x, gridsize_y=netcdf_creation_original_ERA5(path_entry,var_sp,lat_limits_global,lon_limits_global,\
                                                        p_level_interest_lower,p_level_interest_upper,\
                                                            'ERA5','No',path_save)
    
    else:
        print('Status Variable: Not found')
        
print('The generation of the netCDF from ERA5 is ready')

"""
#-----------------------------------------------------------------------------------------------------
#2. MODELS
# ls /bdd/CMIP6/CMIP/*/*/amip/r1i1p1f1/Amon/

path_entry_m='/bdd/CMIP6/CMIP/'
Ensemble='r1i1p1f1'
"""
list_variables=['ua','va','hus','psl','zg','tos','ta','pr']

dict_models={}

for i in range(len(list_variables)):

    if list_variables[i]=='tos':
        dom_str='Omon'
    else:
        dom_str='Amon'

    models_var_sp=variables_availability(path_entry_m,list_variables[i],dom_str)

    dict_models[list_variables[i]]=models_var_sp[:]

with open(path_save+'Models_var_availability.pkl', 'wb') as fp:
    pickle.dump(dict_models, fp)
"""
#------------------------------------------------------------------------------------------------
#Create the netcdf with the original gridsize of each model and extracting the grid size of each 
#model 
#list_variables=['ua','va','hus','psl','zg','tos','ta','pr']
list_variables=['tos']

with open(path_save+'Models_var_availability.pkl', 'rb') as fp:
    dict_models = pickle.load(fp)

lat_limits_global=[-70,50]
lon_limits_global=[-155,-5]

lat_limits_full=[-90,90]
lon_limits_full=[0,360]

initial_time='19790101'
final_time='20141201'

p_level_interest_upper=10000.0
p_level_interest_lower=100000.0

#gridsize_df=pd.DataFrame(columns=['Model','Longitude','Latitude'])
gridsize_df_tos=pd.DataFrame(columns=['Model','Longitude','Latitude'])
#gridsize_df.to_csv(path_save+'CMIP6_models_GridSize_lat_lon_Amon.csv')
gridsize_df_tos.to_csv(path_save+'CMIP6_models_GridSize_lat_lon_Omon.csv')

path_entry_m='/bdd/CMIP6/CMIP/'
Ensemble='r1i1p1f1'

for i in range(len(list_variables)):

    var_sp=list_variables[i]

    models_ava=list(dict_models[var_sp])

    #---------------------------------------------------------------------------------------------------------------------
    list_f1=os.listdir(path_entry_m)

    for n in range(len(models_ava)):

        for e in range(len(list_f1)):

            path_folder_sp=path_entry_m+list_f1[e]+'/'

            list_mod_v=os.listdir(path_folder_sp)

            if models_ava[n] in list_mod_v:

                path_entry=path_folder_sp+models_ava[n]+'/'

                try:

                    if var_sp=='ua' or var_sp=='va':
                        gridsize_x, gridsize_y, path_check=netcdf_creation_original(path_entry,var_sp,'Amon',lat_limits_full,lon_limits_full,\
                                                                        initial_time,final_time,p_level_interest_lower,p_level_interest_upper,\
                                                                            'model','Yes',path_save,models_ava[n])
                        print(path_check)
                    
                    elif var_sp=='hus' or var_sp=='ta' or var_sp=='zg':
                        gridsize_x, gridsize_y, path_check=netcdf_creation_original(path_entry,var_sp,'Amon',lat_limits_global,lon_limits_global,\
                                                                        initial_time,final_time,p_level_interest_lower,p_level_interest_upper,\
                                                                            'model','Yes',path_save,models_ava[n])
                        
                        print(path_check)
                            
                    elif var_sp=='psl' or var_sp=='pr':
                        gridsize_x, gridsize_y, path_check=netcdf_creation_original(path_entry,var_sp,'Amon',lat_limits_global,lon_limits_global,\
                                                                        initial_time,final_time,p_level_interest_lower,p_level_interest_upper,\
                                                                            'model','No',path_save,models_ava[n])
                        
                        print(path_check)
                        if var_sp=='psl':
                            dt_row={'Model':models_ava[n], 'Longitude':gridsize_x, 'Latitude':gridsize_y}

                            #Reading the dataframe to save the gridsize 
                            grid_amon=pd.read_csv(path_save+'CMIP6_models_GridSize_lat_lon_Amon.csv', index_col=[0])

                            grid_amon=grid_amon.append(dt_row,ignore_index=True)

                            grid_amon.to_csv(path_save+'CMIP6_models_GridSize_lat_lon_Amon.csv')

                    elif var_sp=='tos':
                        cdo_remapbill(path_entry,models_ava[n],path_save)
                        gridsize_x, gridsize_y, path_check=netcdf_creation_original(path_save,var_sp,'Omon',lat_limits_global,lon_limits_global,\
                                                                        initial_time,final_time,p_level_interest_lower,p_level_interest_upper,\
                                                                            'model','No',path_save,models_ava[n])
                        print(path_check)
                        
                        dt_row=pd.DataFrame({'Model':[models_ava[n]], 'Longitude':[gridsize_x], 'Latitude':[gridsize_y]})

                        #Reading the dataframe to save the gridsize 
                        grid_omon=pd.read_csv(path_save+'CMIP6_models_GridSize_lat_lon_Omon.csv', index_col=[0])

                        grid_omon = pd.concat([grid_omon, dt_row], ignore_index=True)

                        grid_omon.to_csv(path_save+'CMIP6_models_GridSize_lat_lon_Omon.csv')
                        
                except:
                    print('Error: ',var_sp,models_ava[n])
            
            else:
                pass

print('The generation of Original netCDF is ready')

print('##########################################################')
print('Code finished')
print('##########################################################')
