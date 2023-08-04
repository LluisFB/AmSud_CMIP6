"""
Code to calculate the the reference fields of the calculations 
from ERA5 reanalysis 

Part 2

Author:Isabel
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


#------------------------------------------------------------------------------------------
#Path_save is the path of the folder that contains all the files
path_save='/home/iccorreasa/Documentos/Paper_CMIP6_models/PLOTS_paper/PAPER_FINAL/npz/' #CHANGE

#-------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------
ref_std_DataFrame=pd.DataFrame(columns=['Characteristic','std_DJF', 'std_JJA', 'std_MAM', 'std_SON'])
ref_std_DataFrame.to_csv(path_save+'reference_std_original.csv')
#------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#Performing the calculations and obtaining the standard deviation of the fields of ERA5
lon_limits_F=[-140,-10]
lat_limits_F=[-60,40]

lat_limits_B=[-60,20]
lon_limits_B=[-100,-20]

#Hadley cell 
lat_limits_Had=[-40,40]
lon_limits_Had=[-70,-50]
#Walker cell
lat_limits_Wal=[-15,15]
lon_limits_Wal=[-150,-20]

north_boundaries_lat=[15,20]
north_boundaries_lon=[-95,-25]

south_boundaries_lat=[-60,-55]
south_boundaries_lon=[-95,-25]

west_boundaries_lat=[-60,20]
west_boundaries_lon=[-100,-95]

east_boundaries_lat=[-60,20]
east_boundaries_lon=[-25,-20]

gridsize_df=pd.read_csv(path_save+'CMIP6_models_GridSize_lat_lon_Amon.csv', index_col=[0])
gridsize_df_tos=pd.read_csv(path_save+'CMIP6_models_GridSize_lat_lon_Omon.csv', index_col=[0])


#Finding the grid size to perform the interpolation 
dx_common=gridsize_df['Longitude'].max()
dy_common=gridsize_df['Latitude'].max()

dx_common_tos=gridsize_df_tos['Longitude'].max()
dy_common_tos=gridsize_df_tos['Latitude'].max()

Lat_common_b=np.arange(lat_limits_B[0],lat_limits_B[1], dy_common)
Lon_common_b=np.arange(lon_limits_B[0],lon_limits_B[1], dx_common)


list_calculation=['wind_850','wind_200','Subtropical_highs','Precipitation','Regional_cells',\
                  'SST','Wind_indices','Bolivian_high','VIMF','qu_qv','MSE','tu_tv']

for i in range(len(list_calculation)):

    if list_calculation[i]=='Wind_indices':

        try:

            lon_limits_subtropical=[-140,-70]
            lat_limits_subtropical=[-40,-20]

            lon_limits_westerlies=[-140,-70]
            lat_limits_westerlies=[-60,-30]

            lat_limits_trade=[-5,5]
            lon_limits_trade=[-35,-20]

            
            jet_strength_ref, jet_latitude_ref=subtropical_jet(path_save,'u','ERA5', None,lon_limits_subtropical,\
                                                        lat_limits_subtropical, 20000.0,20000.0)
            
            westerly_strength_ref, westerly_latitude_ref=westerlies(path_save,'u','ERA5', None,lon_limits_westerlies,\
                                                        lat_limits_westerlies, 85000.0,85000.0)
            
            trade_index_arr_ref=tradeWind(path_save,'u','ERA5', None,lon_limits_trade,lat_limits_trade,  85000.0,85000.0)[:,0]

            #saving the npz
            np.savez_compressed(path_save+'subtropical_jet_strength_ERA5.npz',jet_strength_ref)
            np.savez_compressed(path_save+'subtropical_jet_latitude_ERA5.npz',jet_latitude_ref)
            np.savez_compressed(path_save+'westerlies_strength_ERA5.npz',westerly_strength_ref)
            np.savez_compressed(path_save+'westerlies_latitude_ERA5.npz',westerly_latitude_ref)
            np.savez_compressed(path_save+'trade_winds_ERA5.npz',trade_index_arr_ref)
        
        except:
            print('Error ERA5 Wind Indices')
    
    elif list_calculation[i]=='Subtropical_highs':

        NASH_domains_lat=[20,43]
        NASH_domains_lon=[-70,-16]

        SASH_domains_lat=[-45,0]
        SASH_domains_lon=[-35,-2]

        Pacific_domains_lat=[-50,-15]
        Pacific_domains_lon=[-150,-79]

        fields_domains_lon=lon_limits_F
        fields_domains_lat=lat_limits_F

        try:
       
            #Monthly
            southAtlantic_strength_r, southAtlantic_lat_r,southAtlantic_lon_r=subtropicalHighs(path_save,'msl',\
                                                                                                'ERA5','reference',SASH_domains_lon,SASH_domains_lat)

            southPacific_strength_r, southPacific_lat_r,southPacific_lon_r=subtropicalHighs(path_save,'msl',\
                                                                                            'ERA5', 'reference',Pacific_domains_lon,Pacific_domains_lat)

            nash_strength_r, nash_lat_r,nash_lon_r=subtropicalHighs(path_save,'msl','ERA5', 'reference',\
                                                                    NASH_domains_lon,NASH_domains_lat)

            #Seasonal
            southAtlantic_strength_r_seasonal, southAtlantic_lat_r_seasonal,\
                southAtlantic_lon_r_seasonal=subtropicalHighs_core_Seasonal(path_save,'msl','ERA5','reference',SASH_domains_lon,SASH_domains_lat)

            southPacific_strength_r_seasonal, southPacific_lat_r_seasonal,\
                southPacific_lon_r_seasonal=subtropicalHighs_core_Seasonal(path_save,'msl','ERA5', 'reference',Pacific_domains_lon,Pacific_domains_lat)

            nash_strength_r_seasonal, nash_lat_r_seasonal,\
                nash_lon_r_seasonal=subtropicalHighs_core_Seasonal(path_save,'msl','ERA5', 'reference',NASH_domains_lon,NASH_domains_lat)
            
            #Seasonal fields 

            psl_array,Lat_list_psl,Lon_list_psl,dx_data_psl, dy_data_psl=var_field_calc(path_save,'msl','ERA5',\
                                                                                    fields_domains_lat,fields_domains_lon,None,None,\
                                                                                    None,None,'ERA5','No')
            
            psl_array=psl_array/100

            var_sum=np.sum(psl_array)

            if np.isnan(var_sum)==True :
                var_array_r=NaNs_interp(psl_array, '3D', 'cubic')
            else:
                var_array_r=psl_array

            np.savez_compressed(path_save+'southAtlantic_high_strength_ERA5.npz',southAtlantic_strength_r)
            np.savez_compressed(path_save+'southAtlantic_high_latitude_ERA5.npz',southAtlantic_lat_r)
            np.savez_compressed(path_save+'southAtlantic_high_longitude_ERA5.npz',southAtlantic_lon_r)

            np.savez_compressed(path_save+'southPacific_high_strength_ERA5.npz',southPacific_strength_r)
            np.savez_compressed(path_save+'southPacific_high_latitude_ERA5.npz',southPacific_lat_r)
            np.savez_compressed(path_save+'southPacific_high_longitude_ERA5.npz',southPacific_lon_r)

            np.savez_compressed(path_save+'northAtlantic_high_strength_ERA5.npz',nash_strength_r)
            np.savez_compressed(path_save+'northAtlantic_high_latitude_ERA5.npz',nash_lat_r)
            np.savez_compressed(path_save+'northAtlantic_high_longitude_ERA5.npz',nash_lon_r)

            np.savez_compressed(path_save+'southAtlantic_high_strength_seasonal_ERA5.npz',southAtlantic_strength_r_seasonal)
            np.savez_compressed(path_save+'southAtlantic_high_latitude_seasonal_ERA5.npz',southAtlantic_lat_r_seasonal)
            np.savez_compressed(path_save+'southAtlantic_high_longitude_seasonal_ERA5.npz',southAtlantic_lon_r_seasonal)

            np.savez_compressed(path_save+'southPacific_high_strength_seasonal_ERA5.npz',southPacific_strength_r_seasonal)
            np.savez_compressed(path_save+'southPacific_high_latitude_seasonal_ERA5.npz',southPacific_lat_r_seasonal)
            np.savez_compressed(path_save+'southPacific_high_longitude_seasonal_ERA5.npz',southPacific_lon_r_seasonal)

            np.savez_compressed(path_save+'northAtlantic_high_strength_seasonal_ERA5.npz',nash_strength_r_seasonal)
            np.savez_compressed(path_save+'northAtlantic_high_latitude_seasonal_ERA5.npz',nash_lat_r_seasonal)
            np.savez_compressed(path_save+'northAtlantic_high_longitude_seasonal_ERA5.npz',nash_lon_r_seasonal)

            #-----------------------------------------------------------------------------------------------------------------------
            #Saving the spatial fields 
            np.savez_compressed(path_save+'ERA5_slp_fields.npz',var_array_r)
            np.savez_compressed(path_save+'ERA5_slp_fields_Lat.npz',Lat_list_psl)
            np.savez_compressed(path_save+'ERA5_slp_fields_Lon.npz',Lon_list_psl)

            std_ref(var_array_r, path_save, 'SLP')
        
        except:
            print('Error ERA5 Subtropical Highs')

    elif list_calculation[i]=='Bolivian_high':

        lon_limits_bol=[-70,-60]
        lat_limits_bol=[-25,-15]

        try:
            
            var_array_r=Bolivian_High(path_save,'geopt','ERA5', 'reference',lon_limits_bol,lat_limits_bol, 20000,20000)[:,0]

            var_array_r=(var_array_r/9.8)/1000

            np.savez_compressed(path_save+'Bolivian_High_index_monthly_ERA5.npz',var_array_r)
        except:
            print('Erros ERA5 Bolivian High')
    
    elif list_calculation[i]=='Precipitation':

        lon_limits_pr=lon_limits_F
        lat_limits_pr=lat_limits_F

        try:

            pr_array,Lat_list_pr,Lon_list_pr,dx_data_pr, dy_data_pr=var_field_calc(path_save,'mtpr','ERA5',\
                                                                                    lat_limits_pr,lon_limits_pr,None,None,\
                                                                                    None,None,'ERA5','No')
            
            pr_array=pr_array*86400

            var_sum=np.sum(pr_array)

            if np.isnan(var_sum)==True :
                var_array_r=NaNs_interp(pr_array, '3D', 'cubic')
            else:
                var_array_r=pr_array
            
            #-----------------------------------------------------------------------------------------------------------------------
            #Saving the spatial fields 
            np.savez_compressed(path_save+'ERA5_mtpr_fields.npz',var_array_r)
            np.savez_compressed(path_save+'ERA5_mtpr_fields_Lat.npz',Lat_list_pr)
            np.savez_compressed(path_save+'ERA5_mtpr_fields_Lon.npz',Lon_list_pr)

            std_ref(var_array_r, path_save, 'PPT')
        
        except:
            print('Error ERA5 Precipitation')
    
    elif list_calculation[i]=='SST':

        lon_limits_tos=lon_limits_F
        lat_limits_tos=lat_limits_F

        try:

            tos_array,Lat_list_tos,Lon_list_tos,dx_data_tos, dy_data_tos=var_field_calc(path_save,'sst','ERA5',\
                                                                                    lat_limits_tos,lon_limits_tos,None,None,\
                                                                                    None,None,'ERA5','No')
            
            var_array_r=tos_array-273.15

            #-----------------------------------------------------------------------------------------------------------------------
            #Saving the spatial fields 
            np.savez_compressed(path_save+'ERA5_sst_fields.npz',var_array_r)
            np.savez_compressed(path_save+'ERA5_sst_fields_Lat.npz',Lat_list_tos)
            np.savez_compressed(path_save+'ERA5_sst_fields_Lon.npz',Lon_list_tos)

            std_ref(var_array_r, path_save, 'SST')
        
        except:
            print('Error ERA5 SST')
    
    elif list_calculation[i]=='wind_200':

        lon_limits_w200=lon_limits_F
        lat_limits_w200=lat_limits_F

        p_level_interest=20000.0

        try:

            ua200_array,va200_array,mag200_array,Lat_list_w200,Lon_list_w200,\
                dx_data_w200, dy_data_w200=wind_field_calc(path_save,'u','v','ERA5',\
                                                    lat_limits_w200,lon_limits_w200,None,None,\
                                                    p_level_interest,p_level_interest,'ERA5')
            
            ############################################################################
            #Evaluating if there are any NaNs values in the matrices
            u_sum=np.sum(ua200_array)
            v_sum=np.sum(va200_array)
            mag_sum=np.sum(mag200_array)

            if np.isnan(u_sum)==True or np.isnan(v_sum)==True or np.isnan(mag_sum)==True:
                u_array_r=NaNs_interp(ua200_array, '3D', 'cubic')
                v_array_r=NaNs_interp(va200_array, '3D', 'cubic')
                mag_array_r=NaNs_interp(mag200_array, '3D', 'cubic')
            else:
                u_array_r=ua200_array
                v_array_r=va200_array
                mag_array_r=mag200_array

            #-----------------------------------------------------------------------------------------------------------------------
            #Saving the spatial fields 
            np.savez_compressed(path_save+'ERA5_u200_fields.npz',u_array_r)
            np.savez_compressed(path_save+'ERA5_v200_fields.npz',v_array_r)
            np.savez_compressed(path_save+'ERA5_W200_fields.npz',mag_array_r)
            np.savez_compressed(path_save+'ERA5_W200_fields_Lat.npz',Lat_list_w200)
            np.savez_compressed(path_save+'ERA5_W200_fields_Lon.npz',Lon_list_w200)

            std_ref(mag_array_r, path_save, 'W200')
        
        except:
            print('Error ERA5 wind_200')
    
    elif list_calculation[i]=='wind_850':

        lon_limits_w850=lon_limits_F
        lat_limits_w850=lat_limits_F

        p_level_interest=85000.0

        try:

            ua850_array,va850_array,mag850_array,Lat_list_w850,Lon_list_w850,\
                dx_data_w850, dy_data_w850=wind_field_calc(path_save,'u','v','ERA5',\
                                                    lat_limits_w850,lon_limits_w850,None,None,\
                                                    p_level_interest,p_level_interest,'ERA5')
            
            ############################################################################
            #Evaluating if there are any NaNs values in the matrices
            u_sum=np.sum(ua850_array)
            v_sum=np.sum(va850_array)
            mag_sum=np.sum(mag850_array)

            if np.isnan(u_sum)==True or np.isnan(v_sum)==True or np.isnan(mag_sum)==True:
                u_array_r=NaNs_interp(ua850_array, '3D', 'cubic')
                v_array_r=NaNs_interp(va850_array, '3D', 'cubic')
                mag_array_r=NaNs_interp(mag850_array, '3D', 'cubic')
            else:
                u_array_r=ua850_array
                v_array_r=va850_array
                mag_array_r=mag850_array

            #-----------------------------------------------------------------------------------------------------------------------
            #Saving the spatial fields 
            np.savez_compressed(path_save+'ERA5_u850_fields.npz',u_array_r)
            np.savez_compressed(path_save+'ERA5_v850_fields.npz',v_array_r)
            np.savez_compressed(path_save+'ERA5_W850_fields.npz',mag_array_r)
            np.savez_compressed(path_save+'ERA5_W850_fields_Lat.npz',Lat_list_w850)
            np.savez_compressed(path_save+'ERA5_W850_fields_Lon.npz',Lon_list_w850)

            std_ref(mag_array_r, path_save, 'W850')
        
        except:
            print('Error ERA5 wind_850')
    
    elif list_calculation[i]=='Regional_cells':

        #Hadley cell 
        lat_limits_h=lat_limits_Had
        lon_limits_h=lon_limits_Had
        #Walker cell
        lat_limits_w=lat_limits_Wal
        lon_limits_w=lon_limits_Wal

        p_level_interest_top=10000.0
        p_level_interest_bottom=100000.0

        try:

            hadley_stream_r, std_h_r, lat_had_r,  walker_stream_r,std_w_r,lon_wal_r, p_level_r, dx_h_r, dx_w_r=regional_cells(path_save,\
                    'u','v','ERA5',lat_limits_h,lon_limits_h,lat_limits_w,lon_limits_w,None,None,p_level_interest_bottom,p_level_interest_top,\
                        'reference')
            
            #-----------------------------------------------------------------------------------------------------------------------
            #Saving the spatial fields 
            np.savez_compressed(path_save+'ERA5_HadCell_fields.npz',hadley_stream_r)
            np.savez_compressed(path_save+'ERA5_cells_fields_plevel.npz',p_level_r)
            np.savez_compressed(path_save+'ERA5_HadCell_fields_Lat.npz',lat_had_r)
            np.savez_compressed(path_save+'ERA5_WalCell_fields.npz',walker_stream_r)
            np.savez_compressed(path_save+'ERA5_WalCell_fields_Lon.npz',lon_wal_r)

            std_ref(hadley_stream_r, path_save, 'HadCell')
            std_ref(walker_stream_r, path_save, 'WalCell')
        
        except:
            print('Error ERA5 Regional Cells')

    elif list_calculation[i]=='VIMF':

        #Delimiting the longitude and latitudes bands
        north_boundaries_lat=[15,20]
        north_boundaries_lon=[-95,-25]

        south_boundaries_lat=[-60,-55]
        south_boundaries_lon=[-95,-25]

        west_boundaries_lat=[-60,20]
        west_boundaries_lon=[-100,-95]

        east_boundaries_lat=[-60,20]
        east_boundaries_lon=[-25,-20]


        #Delimiting the longitude and latitudes bands
        lat_limits=lat_limits_B
        lon_limits=lon_limits_B

        p_level_interest_upper=10000.0
        p_level_interest_lower=100000.0

        try:

            VIMF_ref,Lat_ref,Lon_ref,dx_ref,\
            dy_ref=VIMF_calc(path_save, 'u', 'v', 'q', 'ERA5',lat_limits,lon_limits,None, None,\
            p_level_interest_lower,p_level_interest_upper,'ERA5')

            #-----------------------------------------------------------------------------
            #Interpolating to the common grid size 
            VIMF_ref_interpolated=interpolation_fields(VIMF_ref,Lat_ref,Lon_ref,Lat_common_b,Lon_common_b,dx_common,dy_common)

            ############################################################################
            #Obtaining the series of each boundary
            #north
            lat_loc_m_n=np.where(np.logical_and(Lat_common_b>=north_boundaries_lat[0], \
            Lat_common_b<=north_boundaries_lat[1]))
            lon_loc_m_n=np.where(np.logical_and(Lon_common_b>=north_boundaries_lon[0], \
            Lon_common_b<=north_boundaries_lon[1]))
            VIMF_ref_n=VIMF_ref_interpolated[:,lat_loc_m_n[0][0]:lat_loc_m_n[0][-1]+1,\
            lon_loc_m_n[0][0]:lon_loc_m_n[0][-1]+1]
            VIMF_ref_northern=np.nanmean(VIMF_ref_n,axis=1)
            #south
            lat_loc_m_s=np.where(np.logical_and(Lat_common_b>=south_boundaries_lat[0], \
            Lat_common_b<=south_boundaries_lat[1]))
            lon_loc_m_s=np.where(np.logical_and(Lon_common_b>=south_boundaries_lon[0], \
            Lon_common_b<=south_boundaries_lon[1]))
            VIMF_ref_s=VIMF_ref_interpolated[:,lat_loc_m_s[0][0]:lat_loc_m_s[0][-1]+1,\
            lon_loc_m_s[0][0]:lon_loc_m_s[0][-1]+1]
            VIMF_ref_southern=np.nanmean(VIMF_ref_s,axis=1)
            #west
            lat_loc_m_w=np.where(np.logical_and(Lat_common_b>=west_boundaries_lat[0],\
            Lat_common_b<=west_boundaries_lat[1]))
            lon_loc_m_w=np.where(np.logical_and(Lon_common_b>=west_boundaries_lon[0],\
            Lon_common_b<=west_boundaries_lon[1]))
            VIMF_ref_w=VIMF_ref_interpolated[:,lat_loc_m_w[0][0]:lat_loc_m_w[0][-1]+1,\
            lon_loc_m_w[0][0]:lon_loc_m_w[0][-1]+1]
            VIMF_ref_western=np.nanmean(VIMF_ref_w,axis=2)
            #east
            lat_loc_m_e=np.where(np.logical_and(Lat_common_b>=east_boundaries_lat[0],\
            Lat_common_b<=east_boundaries_lat[1]))
            lon_loc_m_e=np.where(np.logical_and(Lon_common_b>=east_boundaries_lon[0],\
            Lon_common_b<=east_boundaries_lon[1]))
            VIMF_ref_e=VIMF_ref_interpolated[:,lat_loc_m_e[0][0]:lat_loc_m_e[0][-1]+1,\
            lon_loc_m_e[0][0]:lon_loc_m_e[0][-1]+1]
            VIMF_ref_eastern=np.nanmean(VIMF_ref_e,axis=2)


            ############################################################################
            ############################################################################
            #SAVING THE INFORMATION

            np.savez_compressed(path_save+'VIMF_ERA5_northern_Boundary.npz',VIMF_ref_northern)
            np.savez_compressed(path_save+'VIMF_ERA5_southern_Boundary.npz',VIMF_ref_southern)
            np.savez_compressed(path_save+'VIMF_ERA5_western_Boundary.npz',VIMF_ref_western)
            np.savez_compressed(path_save+'VIMF_ERA5_eastern_Boundary.npz',VIMF_ref_eastern)

        except:
            print('Error ERA5 VIMF')

    
    elif list_calculation[i]=='MSE':

        #Delimiting the longitude and latitudes bands
        north_boundaries_lat=[15,20]
        north_boundaries_lon=[-95,-25]

        south_boundaries_lat=[-60,-55]
        south_boundaries_lon=[-95,-25]

        west_boundaries_lat=[-60,20]
        west_boundaries_lon=[-100,-95]

        east_boundaries_lat=[-60,20]
        east_boundaries_lon=[-25,-20]


        #Delimiting the longitude and latitudes bands
        lat_limits=lat_limits_B
        lon_limits=lon_limits_B

        p_level_interest_upper=10000.0
        p_level_interest_lower=100000.0

        #Applying the function

        try:

            MSE_ref,Lat_ref,Lon_ref,dx_ref,\
            dy_ref=MSE_calc(path_save, 'v', 'q', 't','geopt', 'ERA5',lat_limits,lon_limits,None, None,\
            p_level_interest_lower,p_level_interest_upper,'ERA5')

            #-----------------------------------------------------------------------------
            #Interpolating to the common grid size 
            MSE_ref_interpolated=interpolation_fields(MSE_ref,Lat_ref,Lon_ref,Lat_common_b,Lon_common_b,dx_common,dy_common)

            ############################################################################
            #Obtaining the series of each boundary
            #north
            lat_loc_m_n=np.where(np.logical_and(Lat_common_b>=north_boundaries_lat[0], \
            Lat_common_b<=north_boundaries_lat[1]))
            lon_loc_m_n=np.where(np.logical_and(Lon_common_b>=north_boundaries_lon[0], \
            Lon_common_b<=north_boundaries_lon[1]))
            MSE_ref_n=MSE_ref_interpolated[:,lat_loc_m_n[0][0]:lat_loc_m_n[0][-1]+1,\
            lon_loc_m_n[0][0]:lon_loc_m_n[0][-1]+1]
            MSE_ref_northern=np.nanmean(MSE_ref_n,axis=1)
            #south
            lat_loc_m_s=np.where(np.logical_and(Lat_common_b>=south_boundaries_lat[0], \
            Lat_common_b<=south_boundaries_lat[1]))
            lon_loc_m_s=np.where(np.logical_and(Lon_common_b>=south_boundaries_lon[0], \
            Lon_common_b<=south_boundaries_lon[1]))
            MSE_ref_s=MSE_ref_interpolated[:,lat_loc_m_s[0][0]:lat_loc_m_s[0][-1]+1,\
            lon_loc_m_s[0][0]:lon_loc_m_s[0][-1]+1]
            MSE_ref_southern=np.nanmean(MSE_ref_s,axis=1)
            #west
            lat_loc_m_w=np.where(np.logical_and(Lat_common_b>=west_boundaries_lat[0],\
            Lat_common_b<=west_boundaries_lat[1]))
            lon_loc_m_w=np.where(np.logical_and(Lon_common_b>=west_boundaries_lon[0],\
            Lon_common_b<=west_boundaries_lon[1]))
            MSE_ref_w=MSE_ref_interpolated[:,lat_loc_m_w[0][0]:lat_loc_m_w[0][-1]+1,\
            lon_loc_m_w[0][0]:lon_loc_m_w[0][-1]+1]
            MSE_ref_western=np.nanmean(MSE_ref_w,axis=2)
            #east
            lat_loc_m_e=np.where(np.logical_and(Lat_common_b>=east_boundaries_lat[0],\
            Lat_common_b<=east_boundaries_lat[1]))
            lon_loc_m_e=np.where(np.logical_and(Lon_common_b>=east_boundaries_lon[0],\
            Lon_common_b<=east_boundaries_lon[1]))
            MSE_ref_e=MSE_ref_interpolated[:,lat_loc_m_e[0][0]:lat_loc_m_e[0][-1]+1,\
            lon_loc_m_e[0][0]:lon_loc_m_e[0][-1]+1]
            MSE_ref_eastern=np.nanmean(MSE_ref_e,axis=2)

            ############################################################################
            ############################################################################
            #SAVING THE INFORMATION
            np.savez_compressed(path_save+'MSE_ERA5_northern_Boundary.npz',MSE_ref_northern)
            np.savez_compressed(path_save+'MSE_ERA5_southern_Boundary.npz',MSE_ref_southern)
            np.savez_compressed(path_save+'MSE_ERA5_western_Boundary.npz',MSE_ref_western)
            np.savez_compressed(path_save+'MSE_ERA5_eastern_Boundary.npz',MSE_ref_eastern)
        
        except:
            print('Error ERA5 MSE')

    elif list_calculation[i]=='qu_qv':

        #Delimiting the longitude and latitudes bands
        north_boundaries_lat=[15,20]
        north_boundaries_lon=[-95,-25]

        south_boundaries_lat=[-60,-55]
        south_boundaries_lon=[-95,-25]

        west_boundaries_lat=[-60,20]
        west_boundaries_lon=[-100,-95]

        east_boundaries_lat=[-60,20]
        east_boundaries_lon=[-25,-20]

        p_level_interest_upper=10000.0
        p_level_interest_lower=100000.0

        #Applying the function

        try:

            #Northern boundary 
            qv_r_northern,Lat_r_north,Lon_r_north,dx_r, \
            dy_r,levels_r=boundaries_fluxes(path_save, 'v', 'q',  'ERA5',\
            north_boundaries_lat,north_boundaries_lon, None, None,p_level_interest_lower,\
            p_level_interest_upper,'reference')

            #Southern boundary
            qv_r_southern,Lat_r_south,Lon_r_south=boundaries_fluxes(path_save, 'v', 'q',  'ERA5',\
            south_boundaries_lat,south_boundaries_lon,None, None,p_level_interest_lower,\
            p_level_interest_upper,'reference')[0:3]

            #Western boundary
            qu_r_western,Lat_r_west,Lon_r_west=boundaries_fluxes(path_save, 'u', 'q',  'ERA5',\
            west_boundaries_lat,west_boundaries_lon,None, None,p_level_interest_lower,\
            p_level_interest_upper,'reference')[0:3]
            
            #Eastern boundary
            qu_r_eastern,Lat_r_east,Lon_r_east=boundaries_fluxes(path_save, 'u', 'q',  'ERA5',\
            east_boundaries_lat,east_boundaries_lon,None, None,p_level_interest_lower,\
            p_level_interest_upper,'reference')[0:3]
            
            ############################################################################
            ############################################################################
            #Averaging in the corresponding axis
            qv_r_northern_av=np.nanmean(qv_r_northern,axis=2)*1000
            qv_r_southern_av=np.nanmean(qv_r_southern,axis=2)*1000
            qu_r_western_av=np.nanmean(qu_r_western,axis=3)*1000
            qu_r_eastern_av=np.nanmean(qu_r_eastern,axis=3)*1000

            #CHECKING IF THERE ARE ANY NANs IN THE MATRICES 
            qv_r_northern_av, indices_northern, text_northern=NaNs_land(qv_r_northern_av)
            qv_r_southern_av, indices_southern, text_southern=NaNs_land(qv_r_southern_av)
            qu_r_western_av, indices_western, text_western=NaNs_land(qu_r_western_av)
            qu_r_eastern_av, indices_eastern, text_eastern=NaNs_land(qu_r_eastern_av)

            #Saving the information
            np.savez_compressed(path_save+'ERA5_qu_qv_north.npz',qv_r_northern_av)

            np.savez_compressed(path_save+'ERA5_qu_qv_south.npz',qv_r_southern_av)

            np.savez_compressed(path_save+'ERA5_qu_qv_west.npz',qu_r_western_av)

            np.savez_compressed(path_save+'ERA5_qu_qv_east.npz',qu_r_eastern_av)

            np.savez_compressed(path_save+'ERA5_qu_qv_north_Lat.npz',Lat_r_north)

            np.savez_compressed(path_save+'ERA5_qu_qv_north_Lon.npz',Lon_r_north)

            np.savez_compressed(path_save+'ERA5_qu_qv_south_Lat.npz',Lat_r_south)

            np.savez_compressed(path_save+'ERA5_qu_qv_south_Lon.npz',Lon_r_south)

            np.savez_compressed(path_save+'ERA5_qu_qv_east_Lat.npz',Lat_r_east)

            np.savez_compressed(path_save+'ERA5_qu_qv_east_Lon.npz',Lon_r_east)

            np.savez_compressed(path_save+'ERA5_qu_qv_west_Lat.npz',Lat_r_west)

            np.savez_compressed(path_save+'ERA5_qu_qv_west_Lon.npz',Lon_r_west)

            np.savez_compressed(path_save+'ERA5_qu_qv_p_level.npz',levels_r)

            std_ref(qv_r_northern_av, path_save, 'qu_qv_north')
            std_ref(qv_r_southern_av, path_save, 'qu_qv_south')
            std_ref(qu_r_eastern_av, path_save, 'qu_qv_east')
            std_ref(qu_r_western_av, path_save, 'qu_qv_west')
        
        except:
            print('Error ERA5 qu_qv')
    
    elif list_calculation[i]=='tu_tv':

        #Delimiting the longitude and latitudes bands
        north_boundaries_lat=[15,20]
        north_boundaries_lon=[-95,-25]

        south_boundaries_lat=[-60,-55]
        south_boundaries_lon=[-95,-25]

        west_boundaries_lat=[-60,20]
        west_boundaries_lon=[-100,-95]

        east_boundaries_lat=[-60,20]
        east_boundaries_lon=[-25,-20]

        p_level_interest_upper=10000.0
        p_level_interest_lower=100000.0

        try:


            tv_r_northern,Lat_r_north,Lon_r_north,dx_r, \
            dy_r,levels_r=boundaries_fluxes(path_save, 'v', 't',  'ERA5',\
            north_boundaries_lat,north_boundaries_lon, None, None,p_level_interest_lower,\
            p_level_interest_upper,'reference')

            #Southern boundary
            tv_r_southern,Lat_r_south,Lon_r_south=boundaries_fluxes(path_save, 'v', 't',  'ERA5',\
            south_boundaries_lat,south_boundaries_lon,None, None,p_level_interest_lower,\
            p_level_interest_upper,'reference')[0:3]

            #Western boundary
            tu_r_western,Lat_r_west,Lon_r_west=boundaries_fluxes(path_save, 'u', 't',  'ERA5',\
            west_boundaries_lat,west_boundaries_lon,None, None,p_level_interest_lower,\
            p_level_interest_upper,'reference')[0:3]
            
            #Eastern boundary
            tu_r_eastern,Lat_r_east,Lon_r_east=boundaries_fluxes(path_save, 'u', 't',  'ERA5',\
            east_boundaries_lat,east_boundaries_lon,None, None,p_level_interest_lower,\
            p_level_interest_upper,'reference')[0:3]
            
            ############################################################################
            ############################################################################
            #Averaging in the corresponding axis
            tv_r_northern_av=np.nanmean(tv_r_northern,axis=2)
            tv_r_southern_av=np.nanmean(tv_r_southern,axis=2)
            tu_r_western_av=np.nanmean(tu_r_western,axis=3)
            tu_r_eastern_av=np.nanmean(tu_r_eastern,axis=3)

            #CHECKING IF THERE ARE ANY NANs IN THE MATRICES 
            tv_r_northern_av, indices_northern, text_northern=NaNs_land(tv_r_northern_av)
            tv_r_southern_av, indices_southern, text_southern=NaNs_land(tv_r_southern_av)
            tu_r_western_av, indices_western, text_western=NaNs_land(tu_r_western_av)
            tu_r_eastern_av, indices_eastern, text_eastern=NaNs_land(tu_r_eastern_av)

            #Saving the information
            np.savez_compressed(path_save+'ERA5_tu_tv_north.npz',tv_r_northern_av)

            np.savez_compressed(path_save+'ERA5_tu_tv_south.npz',tv_r_southern_av)

            np.savez_compressed(path_save+'ERA5_tu_tv_west.npz',tu_r_western_av)

            np.savez_compressed(path_save+'ERA5_tu_tv_east.npz',tu_r_eastern_av)

            np.savez_compressed(path_save+'ERA5_tu_tv_north_Lat.npz',Lat_r_north)

            np.savez_compressed(path_save+'ERA5_tu_tv_north_Lon.npz',Lon_r_north)

            np.savez_compressed(path_save+'ERA5_tu_tv_south_Lat.npz',Lat_r_south)

            np.savez_compressed(path_save+'ERA5_tu_tv_south_Lon.npz',Lon_r_south)

            np.savez_compressed(path_save+'ERA5_tu_tv_east_Lat.npz',Lat_r_east)

            np.savez_compressed(path_save+'ERA5_tu_tv_east_Lon.npz',Lon_r_east)

            np.savez_compressed(path_save+'ERA5_tu_tv_west_Lat.npz',Lat_r_west)

            np.savez_compressed(path_save+'ERA5_tu_tv_west_Lon.npz',Lon_r_west)

            np.savez_compressed(path_save+'ERA5_tu_tv_p_level.npz',levels_r)

            std_ref(tv_r_northern_av, path_save, 'tu_tv_north')
            std_ref(tv_r_southern_av, path_save, 'tu_tv_south')
            std_ref(tu_r_eastern_av, path_save, 'tu_tv_east')
            std_ref(tu_r_western_av, path_save, 'tu_tv_west')
        
        except:
            print('Error ERA5 tu_tv')

#-----------------------------------------------------------------------------------------
print('#################################################')
print('Finished')
print('#################################################')





          
            
        

            
            
            
           
        

