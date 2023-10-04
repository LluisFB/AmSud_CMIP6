"""
This code extracts the grid size of each model and save them into a .csv file 
to use in future calculations
Author: Isabel 
"""

#libreries
import xarray as xr
import numpy as np
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


#path where all files that have been generated are saved

#path_save='/home/iccorreasa/Documentos/Paper_CMIP6_models/PLOTS_paper/PAPER_FINAL/npz/' #CHANGE
path_save='/scratchx/lfita/'

#Generating the list of files in the path

files_path=list(os.listdir(path_save))

files_to_read=[]

for h in range(len(files_path)):

    if '_original_seasonal_mean' in files_path[h]:
        if ('psl' in files_path[h]) or ('tos' in files_path[h]):
            files_to_read.append(files_path[h])

#Ontaining the grid size of each model 
for h in range(len(files_to_read)):
    data=xr.open_dataset(path_save+files_to_read[h])

    if 'lat' in data.dims:
        Lat_grid=list(np.array(data.lat))
        Lon_grid=list(np.array(data.lon))
    else:
        Lat_grid=list(np.array(data.latitude))
        Lon_grid=list(np.array(data.longitude))

    dx_data=np.round(abs(Lon_grid[1])-abs(Lon_grid[0]),2)
    dy_data=np.round(abs(Lat_grid[-1:][0])-  abs(Lat_grid[-2:][0]),2)

    if 'psl' in files_to_read[h]:
        Model_name=files_to_read[h].split('_psl')[0]

        if Model_name!='ERA5':

            dt_row=pd.DataFrame({'Model':[Model_name], 'Longitude':[dx_data], 'Latitude':[dy_data]})

            #Reading the dataframe to save the gridsize 
            grid_amon=pd.read_csv(path_save+'CMIP6_models_GridSize_lat_lon_Amon.csv', index_col=[0])

            grid_amon = pd.concat([grid_amon, dt_row], ignore_index=True)

            grid_amon.to_csv(path_save+'CMIP6_models_GridSize_lat_lon_Amon.csv')
    
    elif 'tos' in files_to_read[h]:
        Model_name=files_to_read[h].split('_tos')[0]

        if Model_name!='ERA5':

            dt_row=pd.DataFrame({'Model':[Model_name], 'Longitude':[dx_data], 'Latitude':[dy_data]})

            #Reading the dataframe to save the gridsize 
            grid_omon=pd.read_csv(path_save+'CMIP6_models_GridSize_lat_lon_Omon.csv', index_col=[0])

            grid_omon = pd.concat([grid_omon, dt_row], ignore_index=True)

            grid_omon.to_csv(path_save+'CMIP6_models_GridSize_lat_lon_Omon.csv')



