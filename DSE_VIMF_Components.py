"""
Code to calculate the MSE from zonal wind and the plots 
of the fluxes throught the boundaries 
Author: iccorreasa
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


#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#Path_save is the path that contains all the created files
path_entry='/scratchx/lfita/'
path_save='/scratchx/lfita/'
path_save_ind=path_save+'model_season_ind/'
#-----------------------------------------------------------------------------------------------------------------------
gridsize_df=pd.read_csv(path_save+'CMIP6_models_GridSize_lat_lon_Amon.csv', index_col=[0])
gridsize_df_tos=pd.read_csv(path_save+'CMIP6_models_GridSize_lat_lon_Omon.csv', index_col=[0])

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#Defining the parameters of the plots 
fig_title_font=25
title_str_size=23
xy_label_str=20
tick_labels_str=18
tick_labels_str_regcell=18
legends_str=16

#CMAP
cmap_plot=['gist_rainbow_r','terrain','terrain_r','PuOr_r','RdBu_r','RdBu','rainbow']
cmap_bias=cmap_plot[4]
cmap_slp=cmap_plot[6]
cmap_pr=cmap_plot[2]
cmap_sst=cmap_plot[6]
cmap_VIMF=cmap_plot[6]
cmap_MSE=cmap_plot[3]
cmap_w200=cmap_plot[6]
cmap_w850=cmap_plot[6]
cmap_regcell=cmap_plot[3]
cmap_flux=cmap_plot[3]


#Finding the grid size to perform the interpolation 
dx_common=gridsize_df['Longitude'].max()
dy_common=gridsize_df['Latitude'].max()

dx_common_tos=gridsize_df_tos['Longitude'].max()
dy_common_tos=gridsize_df_tos['Latitude'].max()

p_level_common=[10000.0,15000.0,20000.0,25000.0,30000.0,40000.0,50000.0,60000.0,\
70000.0,85000.0,92500.0,100000.0]

lon_limits_F=[-140,-10]
lat_limits_F=[-60,40]

Lat_common_tos=np.arange(lat_limits_F[0],lat_limits_F[1],dy_common_tos)
Lon_common_tos=np.arange(lon_limits_F[0],lon_limits_F[1],dx_common_tos)

Lat_common=np.arange(lat_limits_F[0],lat_limits_F[1],dy_common)
Lon_common=np.arange(lon_limits_F[0],lon_limits_F[1],dx_common)

lat_limits_B=[-60,20]
lon_limits_B=[-100,-20]

Lat_common_b=np.arange(lat_limits_B[0],lat_limits_B[1], dy_common)
Lon_common_b=np.arange(lon_limits_B[0],lon_limits_B[1], dx_common)


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

#boundaries 
Lon_common_fl=np.arange(south_boundaries_lon[0],south_boundaries_lon[1],dx_common)
Lat_common_fl=np.arange(west_boundaries_lat[0],west_boundaries_lat[1],dy_common)

####################################################################################################
####################################################################################################
#Calculating the MSE from zonal wind 
def DSE_calc(path_entry, var_sp1, var_sp2, var_sp3, model_name,lat_d,lon_d,time_0,time_1,level_lower,level_upper,type_data):
    
    var_data1=xr.open_mfdataset(path_entry+model_name+'_'+var_sp1+'_original_seasonal_mean.nc')

    u_field=var_data1[var_sp1]

    var_data2=xr.open_mfdataset(path_entry+model_name+'_'+var_sp2+'_original_seasonal_mean.nc')

    t_field=var_data2[var_sp2]

    var_data3=xr.open_mfdataset(path_entry+model_name+'_'+var_sp3+'_original_seasonal_mean.nc')

    z_field=var_data3[var_sp3]

    print('#########################')
    print('MSE_calc: read files OK')
    print('#########################')

    #obtaining the domain bnds
    lat_bnd,lon_bnd,lat_slice,lon_slice=lat_lon_bds(lon_d,lat_d,u_field)
    lat_bnd_t,lon_bnd_t=lat_lon_bds(lon_d,lat_d,t_field)[0:2]
    lat_bnd_z,lon_bnd_z=lat_lon_bds(lon_d,lat_d,z_field)[0:2]
    #delimiting the spatial domain

    u_delimited=time_lat_lon_positions(time_0,time_1,lat_bnd,\
    lon_bnd,lat_slice,u_field,'ERA5','Yes')

    t_delimited=time_lat_lon_positions(time_0,time_1,lat_bnd_t,\
    lon_bnd_t,lat_slice,t_field,'ERA5','Yes')

    z_delimited=time_lat_lon_positions(time_0,time_1,lat_bnd_z,\
    lon_bnd_z,lat_slice,z_field,'ERA5','Yes')

    print('#########################')
    print('MSE_calc: var_delimited OK')
    print('#########################')

    #Comparing the delimited fields of all variables 
    t_delimited_1, lon_t_new=domain_comparison(lat_slice,u_delimited,t_delimited, lon_bnd_t,time_0,time_1,lat_bnd_t,t_field,'ERA5','longitude')
    z_delimited_1, lon_z_new=domain_comparison(lat_slice,u_delimited,z_delimited, lon_bnd_z,time_0,time_1,lat_bnd_z,z_field,'ERA5','longitude')

    t_delimited_f, lat_t_new=domain_comparison(lat_slice,u_delimited,t_delimited_1, lon_t_new,time_0,time_1,lat_bnd_t,t_field,'ERA5','latitude')
    z_delimited_f, lat_z_new=domain_comparison(lat_slice,u_delimited,z_delimited_1, lon_z_new,time_0,time_1,lat_bnd_z,z_field,'ERA5','latitude')

    print('#########################')
    print('MSE_calc: rdomain_comparison OK')
    print('#########################')

    #selecting the pressure level
    ini_level,fin_level=levels_limit(u_delimited,level_lower,level_upper)
    u_levels=u_delimited[:,int(ini_level):int(fin_level+1),:,:]
    t_levels=t_delimited_f[:,int(ini_level):int(fin_level+1),:,:]
    z_levels=z_delimited_f[:,int(ini_level):int(fin_level+1),:,:]

    #converting into array
    u_array_t=np.array(u_levels)
    t_array_t=np.array(t_levels)

    if type_data=='ERA5':
        z_array_t=np.array(z_levels)
    else:
        z_array_t=np.array(z_levels)*9.8

    #Evaluating the missing values in the pressure levels
    u_array=NaNs_levels(u_array_t,'3D', 'cubic')
    t_array=NaNs_levels(t_array_t,'3D', 'cubic')
    z_array=NaNs_levels(z_array_t,'3D', 'cubic')

    print('#########################')
    print('MSE_calc: NaNs_levels OK')
    print('#########################')

    #defining Lat and Lon lists
    if lat_slice=='lat':
        Lat_list=list(np.array(u_delimited.lat))
        Lon_list=list(np.array(u_delimited.lon))
    else:
        Lat_list=list(np.array(u_delimited.latitude))
        Lon_list=list(np.array(u_delimited.longitude))

    #For the list of pressure levels (They have to be in Pa)
    if type_data=='ERA5':
        if u_delimited.level.units=='Pa':
            p_levels=list(np.array(u_levels.level))
        else:
            p_levels=list(np.array(u_levels.level)*100)
    else:
        if u_delimited.plev.units=='Pa':
            p_levels=list(np.round(np.array(u_levels.plev),1))
        else:
            p_levels=list(np.round(np.array(u_levels.plev),1)*100)

    ############################################################################
    #Performing the VIMF calculation

    #geneating the integral
    DSE_column=np.empty((4,len(p_levels)-1,len(Lat_list),len(Lon_list)))

    #Generating the listo of latitudes in radians
    lat_radians=np.radians(Lat_list)
    pi=np.pi
    g=9.8 #m/s
    a=6.371*1000 #m
    lv= 22.6 * 105 #J/Kg
    cp= 1.005 *1000 #J/kg-K
    

    for p in range(4):
        for i in range(len(p_levels)-1):
            for j in range(len(Lat_list)):

                #----------------------------------------------------------
                #Calculating the term c
                c=(2*pi*a*np.cos(lat_radians[j]))/g

                for k in range(len(Lon_list)):
                    u_level=(u_array[p,i,j,k]+u_array[p,i+1,j,k])/2
                    t_level=(t_array[p,i,j,k]+t_array[p,i+1,j,k])/2
                    z_level=(t_array[p,i,j,k]+z_array[p,i+1,j,k])/2

                    m= cp*t_level + z_level

                    dp=(p_levels[i]-p_levels[i+1])

                    dse_level_grid=m *u_level*dp

                    DSE_column[p,i,j,k]=dse_level_grid

    # adding up in the pressure levels
    DSE_integral=np.nansum(DSE_column,axis=1)

    print('#########################')
    print('MSE_calc: DSE_integral OK')
    print('#########################')

    #defining the grid size
    dx_data=np.round(abs(Lon_list[1])-abs(Lon_list[0]),2)
    dy_data=np.round(abs(Lat_list[-1:][0])-  abs(Lat_list[-2:][0]),2)
    

    return DSE_integral,Lat_list,Lon_list,dx_data, dy_data

def VIMF_calc_field(path_entry, var_sp1, var_sp2, var_sp3, model_name,lat_d,lon_d,time_0,time_1,level_lower,level_upper,type_data):

    var_data1=xr.open_mfdataset(path_entry+model_name+'_'+var_sp1+'_original_seasonal_mean.nc')

    u_field=var_data1[var_sp1]

    var_data2=xr.open_mfdataset(path_entry+model_name+'_'+var_sp2+'_original_seasonal_mean.nc')

    v_field=var_data2[var_sp2]

    var_data3=xr.open_mfdataset(path_entry+model_name+'_'+var_sp3+'_original_seasonal_mean.nc')

    q_field=var_data3[var_sp3]

    print('#########################')
    print('VIMF_calc: read files OK')
    print('#########################')

    #obtaining the domain bnds
    lat_bnd,lon_bnd,lat_slice,lon_slice=lat_lon_bds(lon_d,lat_d,u_field)
    lat_bnd_q,lon_bnd_q=lat_lon_bds(lon_d,lat_d,q_field)[0:2]
    #delimiting the spatial domain
    u_delimited=time_lat_lon_positions(time_0,time_1,lat_bnd,\
    lon_bnd,lat_slice,u_field,'ERA5','Yes')

    v_delimited=time_lat_lon_positions(time_0,time_1,lat_bnd,\
    lon_bnd,lat_slice,v_field,'ERA5','Yes')

    q_delimited=time_lat_lon_positions(time_0,time_1,lat_bnd_q,\
    lon_bnd_q,lat_slice,q_field,'ERA5','Yes')

    print('#########################')
    print('VIMF_calc: var_delimited OK')
    print('#########################')

    #Comparing the delimited fields of all variables 
    q_delimited_1,lon_var_new=domain_comparison(lat_slice,v_delimited,q_delimited, lon_bnd_q,time_0,time_1,lat_bnd_q,q_field,'ERA5','longitude')
    q_delimited_f,lat_var_new=domain_comparison(lat_slice,v_delimited,q_delimited_1, lon_var_new,time_0,time_1,lat_bnd_q,q_field,'ERA5','latitude')

    print('#########################')
    print('VIMF_calc: domain_comparison OK')
    print('#########################')

    #selecting the pressure level
    ini_level,fin_level=levels_limit(u_delimited,level_lower,level_upper)
    u_levels=u_delimited[:,int(ini_level):int(fin_level+1),:,:]
    v_levels=v_delimited[:,int(ini_level):int(fin_level+1),:,:]
    q_levels=q_delimited_f[:,int(ini_level):int(fin_level+1),:,:]

    #converting into array
    u_array_t=np.array(u_levels)
    v_array_t=np.array(v_levels)
    q_array_t=np.array(q_levels)

    #Evaluating the missing values in the pressure levels
    u_array=NaNs_levels(u_array_t,'3D', 'cubic')
    v_array=NaNs_levels(v_array_t,'3D', 'cubic')
    q_array=NaNs_levels(q_array_t,'3D', 'cubic')

    print('#########################')
    print('VIMF_calc: NaNs_levels OK')
    print('#########################')

    #defining Lat and Lon lists
    if lat_slice=='lat':
        Lat_list=list(np.array(v_delimited.lat))
        Lon_list=list(np.array(v_delimited.lon))
    else:
        Lat_list=list(np.array(v_delimited.latitude))
        Lon_list=list(np.array(v_delimited.longitude))

    #For the list of pressure levels (They have to be in Pa)
    if type_data=='ERA5':
        if u_delimited.level.units=='Pa':
            p_levels=list(np.array(u_levels.level))
        else:
            p_levels=list(np.array(u_levels.level)*100)
    else:
        if u_delimited.plev.units=='Pa':
            p_levels=list(np.round(np.array(u_levels.plev),1))
        else:
            p_levels=list(np.round(np.array(u_levels.plev),1)*100)

    ############################################################################
    #Performing the VIMF calculation

    #geneating the integral
    qu_column=np.empty((4,len(p_levels)-1,len(Lat_list),len(Lon_list)))
    qv_column=np.empty((4,len(p_levels)-1,len(Lat_list),len(Lon_list)))

    for p in range(4):
        for i in range(len(p_levels)-1):
            for j in range(len(Lat_list)):
                for k in range(len(Lon_list)):
                    u_level=(u_array[p,i,j,k]+u_array[p,i+1,j,k])/2
                    v_level=(v_array[p,i,j,k]+v_array[p,i+1,j,k])/2
                    q_level=(q_array[p,i,j,k]+q_array[p,i+1,j,k])/2
                    dp=(p_levels[i]-p_levels[i+1])

                    qu_level_grid=q_level*u_level*dp
                    qv_level_grid=q_level*v_level*dp

                    qu_column[p,i,j,k]=qu_level_grid
                    qv_column[p,i,j,k]=qv_level_grid

    # adding up in the pressure levels
    qu_integral=np.nansum(qu_column,axis=1)/9.8*(-1)
    qv_integral=np.nansum(qv_column,axis=1)/9.8*(-1)

    print('#########################')
    print('VIMF_calc: integrals OK')
    print('#########################')

    #defining the grid size
    dx_data=np.round(abs(Lon_list[1])-abs(Lon_list[0]),2)
    dy_data=np.round(abs(Lat_list[-1:][0])-  abs(Lat_list[-2:][0]),2)

    VIMF_estimation=np.sqrt(qu_integral**2+qv_integral**2)

    return VIMF_estimation,qu_integral,qv_integral,Lat_list,Lon_list,dx_data, dy_data

def plot_boundary(axs,title_str, ref_arr,models_list,models_arr,x_label_str,arange_x,labels_x,legend_status,ylabel_str,title_font2,label_font, ticks_font,lower_st,sep,y_l,list_q):
    axs.set_title(title_str,fontsize=title_font2,loc='left')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    #iterating in the models to obtain the serie
    colors=iter(cm.rainbow(np.linspace(0,1,len(models_list))))
    for m in range(len(models_list)):
        if models_list[m] in list_q:
            pass
        else:
            c=next(colors)
            if legend_status=='yes':
                label_serie=models_list[m]
                label_era='Reference [ERA5]'
            else:
                label_serie=None
                label_era=None
            axs.plot(models_arr[m] ,color =c ,linewidth=1.5,label=label_serie)
    axs.plot(ref_arr, color = 'k', linewidth=2.2,label=label_era)
    axs.set_xticks(arange_x[::sep])
    axs.set_xticklabels(labels_x[::sep],fontsize=ticks_font)
    axs.set_yticks(y_l)
    axs.set_yticklabels(y_l,fontsize=ticks_font)
    axs.set_ylabel(ylabel_str,fontsize=label_font)
    axs.axhline(y=0, lw=2.5, color='grey',ls='--')
    if lower_st=='Yes':
        axs.set_xlabel(x_label_str,fontsize=label_font)

#-------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#Performing the calculations 
with open(path_save+'Models_var_availability.pkl', 'rb') as fp:
    dict_models = pickle.load(fp)


#1. ERA5

list_calculation=['VIMF','DSE']


for i in range(len(list_calculation)):

    if list_calculation[i]=='DSE':

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

            DSE_ref_u,Lat_ref,Lon_ref,dx_ref,\
            dy_ref=DSE_calc(path_save, 'u', 'ta','geopt', 'ERA5',lat_limits,lon_limits,None, None,\
            p_level_interest_lower,p_level_interest_upper,'ERA5')

            DSE_ref_v,Lat_ref,Lon_ref,dx_ref,\
            dy_ref=DSE_calc(path_save, 'v', 'ta','geopt', 'ERA5',lat_limits,lon_limits,None, None,\
            p_level_interest_lower,p_level_interest_upper,'ERA5')

            print('####################################')
            print('DSE: var_array OK')
            print('####################################')

            #-----------------------------------------------------------------------------
            #Interpolating to the common grid size 
            DSE_u_ref_interpolated=interpolation_fields(DSE_ref_u,Lat_ref,Lon_ref,Lat_common_b,Lon_common_b,dx_common,dy_common)
            DSE_v_ref_interpolated=interpolation_fields(DSE_ref_v,Lat_ref,Lon_ref,Lat_common_b,Lon_common_b,dx_common,dy_common)

            print('####################################')
            print('DSE: interpolation OK')
            print('####################################')

            ############################################################################
            #Obtaining the series of each boundary
            #north
            lat_loc_m_n=np.where(np.logical_and(Lat_common_b>=north_boundaries_lat[0], \
            Lat_common_b<=north_boundaries_lat[1]))
            lon_loc_m_n=np.where(np.logical_and(Lon_common_b>=north_boundaries_lon[0], \
            Lon_common_b<=north_boundaries_lon[1]))

            DSE_v_ref_n=DSE_v_ref_interpolated[:,lat_loc_m_n[0][0]:lat_loc_m_n[0][-1]+1,\
            lon_loc_m_n[0][0]:lon_loc_m_n[0][-1]+1]
            DSE_v_ref_northern=np.nanmean(DSE_v_ref_n,axis=1)

            DSE_u_ref_n=DSE_u_ref_interpolated[:,lat_loc_m_n[0][0]:lat_loc_m_n[0][-1]+1,\
            lon_loc_m_n[0][0]:lon_loc_m_n[0][-1]+1]
            DSE_u_ref_northern=np.nanmean(DSE_u_ref_n,axis=1)

            #south
            lat_loc_m_s=np.where(np.logical_and(Lat_common_b>=south_boundaries_lat[0], \
            Lat_common_b<=south_boundaries_lat[1]))
            lon_loc_m_s=np.where(np.logical_and(Lon_common_b>=south_boundaries_lon[0], \
            Lon_common_b<=south_boundaries_lon[1]))

            DSE_v_ref_s=DSE_v_ref_interpolated[:,lat_loc_m_s[0][0]:lat_loc_m_s[0][-1]+1,\
            lon_loc_m_s[0][0]:lon_loc_m_s[0][-1]+1]
            DSE_v_ref_southern=np.nanmean(DSE_v_ref_s,axis=1)

            DSE_u_ref_s=DSE_u_ref_interpolated[:,lat_loc_m_s[0][0]:lat_loc_m_s[0][-1]+1,\
            lon_loc_m_s[0][0]:lon_loc_m_s[0][-1]+1]
            DSE_u_ref_southern=np.nanmean(DSE_u_ref_s,axis=1)

            #west
            lat_loc_m_w=np.where(np.logical_and(Lat_common_b>=west_boundaries_lat[0],\
            Lat_common_b<=west_boundaries_lat[1]))
            lon_loc_m_w=np.where(np.logical_and(Lon_common_b>=west_boundaries_lon[0],\
            Lon_common_b<=west_boundaries_lon[1]))

            DSE_u_ref_w=DSE_u_ref_interpolated[:,lat_loc_m_w[0][0]:lat_loc_m_w[0][-1]+1,\
            lon_loc_m_w[0][0]:lon_loc_m_w[0][-1]+1]
            DSE_u_ref_western=np.nanmean(DSE_u_ref_w,axis=2)

            DSE_v_ref_w=DSE_v_ref_interpolated[:,lat_loc_m_w[0][0]:lat_loc_m_w[0][-1]+1,\
            lon_loc_m_w[0][0]:lon_loc_m_w[0][-1]+1]
            DSE_v_ref_western=np.nanmean(DSE_v_ref_w,axis=2)

            #east
            lat_loc_m_e=np.where(np.logical_and(Lat_common_b>=east_boundaries_lat[0],\
            Lat_common_b<=east_boundaries_lat[1]))
            lon_loc_m_e=np.where(np.logical_and(Lon_common_b>=east_boundaries_lon[0],\
            Lon_common_b<=east_boundaries_lon[1]))

            DSE_u_ref_e=DSE_u_ref_interpolated[:,lat_loc_m_e[0][0]:lat_loc_m_e[0][-1]+1,\
            lon_loc_m_e[0][0]:lon_loc_m_e[0][-1]+1]
            DSE_u_ref_eastern=np.nanmean(DSE_u_ref_e,axis=2)

            DSE_v_ref_e=DSE_v_ref_interpolated[:,lat_loc_m_e[0][0]:lat_loc_m_e[0][-1]+1,\
            lon_loc_m_e[0][0]:lon_loc_m_e[0][-1]+1]
            DSE_v_ref_eastern=np.nanmean(DSE_v_ref_e,axis=2)

            print('####################################')
            print('DSE: DSE boundaries OK')
            print('####################################')

            ############################################################################
            ############################################################################
            #SAVING THE INFORMATION
            np.savez_compressed(path_save+'DSE_u_ERA5_northern_Boundary.npz',DSE_u_ref_northern)
            np.savez_compressed(path_save+'DSE_u_ERA5_southern_Boundary.npz',DSE_u_ref_southern)
            np.savez_compressed(path_save+'DSE_u_ERA5_western_Boundary.npz',DSE_u_ref_western)
            np.savez_compressed(path_save+'DSE_u_ERA5_eastern_Boundary.npz',DSE_u_ref_eastern)

            np.savez_compressed(path_save+'DSE_v_ERA5_northern_Boundary.npz',DSE_v_ref_northern)
            np.savez_compressed(path_save+'DSE_v_ERA5_southern_Boundary.npz',DSE_v_ref_southern)
            np.savez_compressed(path_save+'DSE_v_ERA5_western_Boundary.npz',DSE_v_ref_western)
            np.savez_compressed(path_save+'DSE_v_ERA5_eastern_Boundary.npz',DSE_v_ref_eastern)
        
        except Exception as e:
            print('Error ERA5 DSE')
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
    

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

            VIMF_ref,u_comp,v_comp,Lat_ref,Lon_ref,dx_ref,\
            dy_ref=VIMF_calc_field(path_save, 'u', 'v', 'q', 'ERA5',lat_limits,lon_limits,None, None,\
            p_level_interest_lower,p_level_interest_upper,'ERA5')


            print('####################################')
            print('VIMF: var_array OK')
            print('####################################')


            #-----------------------------------------------------------------------------
            #Interpolating to the common grid size 
            VIMF_u_ref_interpolated=interpolation_fields(u_comp,Lat_ref,Lon_ref,Lat_common_b,Lon_common_b,dx_common,dy_common)

            VIMF_v_ref_interpolated=interpolation_fields(v_comp,Lat_ref,Lon_ref,Lat_common_b,Lon_common_b,dx_common,dy_common)

            print('####################################')
            print('VIMF: interpolation OK')
            print('####################################')

            ############################################################################
            #Obtaining the series of each boundary
            #north
            lat_loc_m_n=np.where(np.logical_and(Lat_common_b>=north_boundaries_lat[0], \
            Lat_common_b<=north_boundaries_lat[1]))
            lon_loc_m_n=np.where(np.logical_and(Lon_common_b>=north_boundaries_lon[0], \
            Lon_common_b<=north_boundaries_lon[1]))

            VIMF_u_ref_n=VIMF_u_ref_interpolated[:,lat_loc_m_n[0][0]:lat_loc_m_n[0][-1]+1,\
            lon_loc_m_n[0][0]:lon_loc_m_n[0][-1]+1]
            VIMF_u_ref_northern=np.nanmean(VIMF_u_ref_n,axis=1)

            VIMF_v_ref_n=VIMF_v_ref_interpolated[:,lat_loc_m_n[0][0]:lat_loc_m_n[0][-1]+1,\
            lon_loc_m_n[0][0]:lon_loc_m_n[0][-1]+1]
            VIMF_v_ref_northern=np.nanmean(VIMF_v_ref_n,axis=1)

            #south
            lat_loc_m_s=np.where(np.logical_and(Lat_common_b>=south_boundaries_lat[0], \
            Lat_common_b<=south_boundaries_lat[1]))
            lon_loc_m_s=np.where(np.logical_and(Lon_common_b>=south_boundaries_lon[0], \
            Lon_common_b<=south_boundaries_lon[1]))

            VIMF_u_ref_s=VIMF_u_ref_interpolated[:,lat_loc_m_s[0][0]:lat_loc_m_s[0][-1]+1,\
            lon_loc_m_s[0][0]:lon_loc_m_s[0][-1]+1]
            VIMF_u_ref_southern=np.nanmean(VIMF_u_ref_s,axis=1)

            VIMF_v_ref_s=VIMF_v_ref_interpolated[:,lat_loc_m_s[0][0]:lat_loc_m_s[0][-1]+1,\
            lon_loc_m_s[0][0]:lon_loc_m_s[0][-1]+1]
            VIMF_v_ref_southern=np.nanmean(VIMF_v_ref_s,axis=1)

            #west
            lat_loc_m_w=np.where(np.logical_and(Lat_common_b>=west_boundaries_lat[0],\
            Lat_common_b<=west_boundaries_lat[1]))
            lon_loc_m_w=np.where(np.logical_and(Lon_common_b>=west_boundaries_lon[0],\
            Lon_common_b<=west_boundaries_lon[1]))

            VIMF_u_ref_w=VIMF_u_ref_interpolated[:,lat_loc_m_w[0][0]:lat_loc_m_w[0][-1]+1,\
            lon_loc_m_w[0][0]:lon_loc_m_w[0][-1]+1]
            VIMF_u_ref_western=np.nanmean(VIMF_u_ref_w,axis=2)

            VIMF_v_ref_w=VIMF_v_ref_interpolated[:,lat_loc_m_w[0][0]:lat_loc_m_w[0][-1]+1,\
            lon_loc_m_w[0][0]:lon_loc_m_w[0][-1]+1]
            VIMF_v_ref_western=np.nanmean(VIMF_v_ref_w,axis=2)

            #east
            lat_loc_m_e=np.where(np.logical_and(Lat_common_b>=east_boundaries_lat[0],\
            Lat_common_b<=east_boundaries_lat[1]))
            lon_loc_m_e=np.where(np.logical_and(Lon_common_b>=east_boundaries_lon[0],\
            Lon_common_b<=east_boundaries_lon[1]))

            VIMF_u_ref_e=VIMF_u_ref_interpolated[:,lat_loc_m_e[0][0]:lat_loc_m_e[0][-1]+1,\
            lon_loc_m_e[0][0]:lon_loc_m_e[0][-1]+1]
            VIMF_u_ref_eastern=np.nanmean(VIMF_u_ref_e,axis=2)

            VIMF_v_ref_e=VIMF_v_ref_interpolated[:,lat_loc_m_e[0][0]:lat_loc_m_e[0][-1]+1,\
            lon_loc_m_e[0][0]:lon_loc_m_e[0][-1]+1]
            VIMF_v_ref_eastern=np.nanmean(VIMF_v_ref_e,axis=2)

            print('####################################')
            print('VIMF: VIMF boundaries OK')
            print('####################################')


            ############################################################################
            ############################################################################
            #SAVING THE INFORMATION

            np.savez_compressed(path_save+'VIMF_u_ERA5_northern_Boundary.npz',VIMF_u_ref_northern)
            np.savez_compressed(path_save+'VIMF_u_ERA5_southern_Boundary.npz',VIMF_u_ref_southern)
            np.savez_compressed(path_save+'VIMF_u_ERA5_western_Boundary.npz',VIMF_u_ref_western)
            np.savez_compressed(path_save+'VIMF_u_ERA5_eastern_Boundary.npz',VIMF_u_ref_eastern)

            np.savez_compressed(path_save+'VIMF_v_ERA5_northern_Boundary.npz',VIMF_v_ref_northern)
            np.savez_compressed(path_save+'VIMF_v_ERA5_southern_Boundary.npz',VIMF_v_ref_southern)
            np.savez_compressed(path_save+'VIMF_v_ERA5_western_Boundary.npz',VIMF_v_ref_western)
            np.savez_compressed(path_save+'VIMF_v_ERA5_eastern_Boundary.npz',VIMF_v_ref_eastern)
    
        except Exception as e:
            print('Error ERA5 VIMF')
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")



############################################################################################################
############################################################################################################
#2. CMIP6 MODELS 
############################################################################################################
############################################################################################################

list_calculation=['VIMF','DSE']

for i in range(len(list_calculation)):

    if list_calculation[i]=='VIMF':

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

        models_ua=list(dict_models['ua'])
        models_va=list(dict_models['va'])
        models_hus=list(dict_models['hus'])

        models_wind=np.intersect1d(models_ua, models_va)
        models=np.intersect1d(models_wind, models_hus)

        models_app=np.array([])
        np.savez_compressed(path_save+'VIMF_components_models_N.npz',models_app)

        #------------------------------------------------------------------------------------------------
        #Reading the reference information
        VIMF_ref_northern=np.load(path_save+'VIMF_ERA5_northern_Boundary.npz')['arr_0']
        VIMF_ref_southern=np.load(path_save+'VIMF_ERA5_southern_Boundary.npz')['arr_0']
        VIMF_ref_western=np.load(path_save+'VIMF_ERA5_western_Boundary.npz')['arr_0']
        VIMF_ref_eastern=np.load(path_save+'VIMF_ERA5_eastern_Boundary.npz')['arr_0']

        #Creating the empty arrays to save the models information
        northern_boundary_models=np.empty((len(models),4,VIMF_ref_northern.shape[1]))
        southern_boundary_models=np.empty((len(models),4,VIMF_ref_southern.shape[1]))
        western_boundary_models=np.empty((len(models),4,VIMF_ref_western.shape[1]))
        eastern_boundary_models=np.empty((len(models),4,VIMF_ref_eastern.shape[1]))

        np.savez_compressed(path_save+'VIMF_u_CMIP6_northern_Boundary.npz',northern_boundary_models)
        np.savez_compressed(path_save+'VIMF_u_CMIP6_southern_Boundary.npz',southern_boundary_models)
        np.savez_compressed(path_save+'VIMF_u_CMIP6_western_Boundary.npz',western_boundary_models)
        np.savez_compressed(path_save+'VIMF_u_CMIP6_eastern_Boundary.npz',eastern_boundary_models)

        np.savez_compressed(path_save+'VIMF_v_CMIP6_northern_Boundary.npz',northern_boundary_models)
        np.savez_compressed(path_save+'VIMF_v_CMIP6_southern_Boundary.npz',southern_boundary_models)
        np.savez_compressed(path_save+'VIMF_v_CMIP6_western_Boundary.npz',western_boundary_models)
        np.savez_compressed(path_save+'VIMF_v_CMIP6_eastern_Boundary.npz',eastern_boundary_models)

        for p in range(len(models)):

            try:

                VIMF_model,u_comp_model,v_comp_model,Lat_model,Lon_model,dx_model,\
                dy_model=VIMF_calc_field(path_save, 'ua', 'va', 'hus', models[p],lat_limits,lon_limits,None, None,\
                p_level_interest_lower,p_level_interest_upper,'model')

                #-----------------------------------------------------------------------------
                #Interpolating to the common grid size 
                VIMF_u_model_interpolated=interpolation_fields(u_comp_model,Lat_model,Lon_model,Lat_common_b,Lon_common_b,dx_common,dy_common)

                VIMF_v_model_interpolated=interpolation_fields(v_comp_model,Lat_model,Lon_model,Lat_common_b,Lon_common_b,dx_common,dy_common)

                ############################################################################
                #Obtaining the series of each boundary
                #north
                lat_loc_m_n=np.where(np.logical_and(Lat_common_b>=north_boundaries_lat[0], \
                Lat_common_b<=north_boundaries_lat[1]))
                lon_loc_m_n=np.where(np.logical_and(Lon_common_b>=north_boundaries_lon[0], \
                Lon_common_b<=north_boundaries_lon[1]))


                VIMF_u_model_n=VIMF_u_model_interpolated[:,lat_loc_m_n[0][0]:lat_loc_m_n[0][-1]+1,\
                lon_loc_m_n[0][0]:lon_loc_m_n[0][-1]+1]
                VIMF_u_model_northern=np.nanmean(VIMF_u_model_n,axis=1)

                VIMF_v_model_n=VIMF_v_model_interpolated[:,lat_loc_m_n[0][0]:lat_loc_m_n[0][-1]+1,\
                lon_loc_m_n[0][0]:lon_loc_m_n[0][-1]+1]
                VIMF_v_model_northern=np.nanmean(VIMF_v_model_n,axis=1)

                #south
                lat_loc_m_s=np.where(np.logical_and(Lat_common_b>=south_boundaries_lat[0], \
                Lat_common_b<=south_boundaries_lat[1]))
                lon_loc_m_s=np.where(np.logical_and(Lon_common_b>=south_boundaries_lon[0], \
                Lon_common_b<=south_boundaries_lon[1]))

                VIMF_u_model_s=VIMF_u_model_interpolated[:,lat_loc_m_s[0][0]:lat_loc_m_s[0][-1]+1,\
                lon_loc_m_s[0][0]:lon_loc_m_s[0][-1]+1]
                VIMF_u_model_southern=np.nanmean(VIMF_u_model_s,axis=1)

                VIMF_v_model_s=VIMF_v_model_interpolated[:,lat_loc_m_s[0][0]:lat_loc_m_s[0][-1]+1,\
                lon_loc_m_s[0][0]:lon_loc_m_s[0][-1]+1]
                VIMF_v_model_southern=np.nanmean(VIMF_v_model_s,axis=1)

                #west
                lat_loc_m_w=np.where(np.logical_and(Lat_common_b>=west_boundaries_lat[0],\
                Lat_common_b<=west_boundaries_lat[1]))
                lon_loc_m_w=np.where(np.logical_and(Lon_common_b>=west_boundaries_lon[0],\
                Lon_common_b<=west_boundaries_lon[1]))

                VIMF_u_model_w=VIMF_u_model_interpolated[:,lat_loc_m_w[0][0]:lat_loc_m_w[0][-1]+1,\
                lon_loc_m_w[0][0]:lon_loc_m_w[0][-1]+1]
                VIMF_u_model_western=np.nanmean(VIMF_u_model_w,axis=2)

                VIMF_v_model_w=VIMF_v_model_interpolated[:,lat_loc_m_w[0][0]:lat_loc_m_w[0][-1]+1,\
                lon_loc_m_w[0][0]:lon_loc_m_w[0][-1]+1]
                VIMF_v_model_western=np.nanmean(VIMF_v_model_w,axis=2)

                #east
                lat_loc_m_e=np.where(np.logical_and(Lat_common_b>=east_boundaries_lat[0],\
                Lat_common_b<=east_boundaries_lat[1]))
                lon_loc_m_e=np.where(np.logical_and(Lon_common_b>=east_boundaries_lon[0],\
                Lon_common_b<=east_boundaries_lon[1]))

                VIMF_u_model_e=VIMF_u_model_interpolated[:,lat_loc_m_e[0][0]:lat_loc_m_e[0][-1]+1,\
                lon_loc_m_e[0][0]:lon_loc_m_e[0][-1]+1]
                VIMF_u_model_eastern=np.nanmean(VIMF_u_model_e,axis=2)

                VIMF_v_model_e=VIMF_v_model_interpolated[:,lat_loc_m_e[0][0]:lat_loc_m_e[0][-1]+1,\
                lon_loc_m_e[0][0]:lon_loc_m_e[0][-1]+1]
                VIMF_v_model_eastern=np.nanmean(VIMF_v_model_e,axis=2)

                ############################################################################
                ############################################################################
                #SAVING THE INFORMATION

                #U component
                north_u_m_arr=np.load(path_save+'VIMF_u_CMIP6_northern_Boundary.npz',\
                allow_pickle=True)['arr_0']
                south_u_m_arr=np.load(path_save+'VIMF_u_CMIP6_southern_Boundary.npz',\
                allow_pickle=True)['arr_0']
                west_u_m_arr=np.load(path_save+'VIMF_u_CMIP6_western_Boundary.npz',\
                allow_pickle=True)['arr_0']
                east_u_m_arr=np.load(path_save+'VIMF_u_CMIP6_eastern_Boundary.npz',\
                allow_pickle=True)['arr_0']

                north_u_m_arr[p,:,:]=VIMF_u_model_northern
                south_u_m_arr[p,:,:]=VIMF_u_model_southern
                west_u_m_arr[p,:,:]=VIMF_u_model_western
                east_u_m_arr[p,:,:]=VIMF_u_model_eastern

                np.savez_compressed(path_save+'VIMF_u_CMIP6_northern_Boundary.npz',north_u_m_arr)
                np.savez_compressed(path_save+'VIMF_u_CMIP6_southern_Boundary.npz',south_u_m_arr)
                np.savez_compressed(path_save+'VIMF_u_CMIP6_western_Boundary.npz',west_u_m_arr)
                np.savez_compressed(path_save+'VIMF_u_CMIP6_eastern_Boundary.npz',east_u_m_arr)

                #V component
                north_v_m_arr=np.load(path_save+'VIMF_v_CMIP6_northern_Boundary.npz',\
                allow_pickle=True)['arr_0']
                south_v_m_arr=np.load(path_save+'VIMF_v_CMIP6_southern_Boundary.npz',\
                allow_pickle=True)['arr_0']
                west_v_m_arr=np.load(path_save+'VIMF_v_CMIP6_western_Boundary.npz',\
                allow_pickle=True)['arr_0']
                east_v_m_arr=np.load(path_save+'VIMF_v_CMIP6_eastern_Boundary.npz',\
                allow_pickle=True)['arr_0']

                north_v_m_arr[p,:,:]=VIMF_v_model_northern
                south_v_m_arr[p,:,:]=VIMF_v_model_southern
                west_v_m_arr[p,:,:]=VIMF_v_model_western
                east_v_m_arr[p,:,:]=VIMF_v_model_eastern

                np.savez_compressed(path_save+'VIMF_v_CMIP6_northern_Boundary.npz',north_v_m_arr)
                np.savez_compressed(path_save+'VIMF_v_CMIP6_southern_Boundary.npz',south_v_m_arr)
                np.savez_compressed(path_save+'VIMF_v_CMIP6_western_Boundary.npz',west_v_m_arr)
                np.savez_compressed(path_save+'VIMF_v_CMIP6_eastern_Boundary.npz',east_v_m_arr)

                models_vimf_calc=np.load(path_save+'VIMF_components_models_N.npz',allow_pickle=True)['arr_0']

                models_vimf_calc=np.append(models_vimf_calc,models[p])
                np.savez_compressed(path_save+'VIMF_components_models_N.npz',models_vimf_calc)
            
            except:
                print('Error VIMF ', models[p])
        
    elif list_calculation[i]=='DSE':

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

        models_ua=list(dict_models['ua'])
        models_va=list(dict_models['va'])
        models_ta=list(dict_models['ta'])
        models_zg=list(dict_models['zg'])

        models_1=np.intersect1d(models_ua, models_va)
        models_2=np.intersect1d(models_1, models_ta)
        models=np.intersect1d(models_2, models_zg)

        #------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------
        #Reading the reference information
        DSE_ref_northern=np.load(path_save+'DSE_u_ERA5_northern_Boundary.npz')['arr_0']
        DSE_ref_southern=np.load(path_save+'DSE_u_ERA5_southern_Boundary.npz')['arr_0']
        DSE_ref_western=np.load(path_save+'DSE_u_ERA5_western_Boundary.npz')['arr_0']
        DSE_ref_eastern=np.load(path_save+'DSE_u_ERA5_eastern_Boundary.npz')['arr_0']

        models_app=np.array([])
        np.savez_compressed(path_save+'DSE_models_N.npz',models_app)

        #Creating the empty arrays to save the models information
        northern_boundary_models=np.empty((len(models),4,DSE_ref_northern.shape[1]))
        southern_boundary_models=np.empty((len(models),4,DSE_ref_southern.shape[1]))
        western_boundary_models=np.empty((len(models),4,DSE_ref_western.shape[1]))
        eastern_boundary_models=np.empty((len(models),4,DSE_ref_eastern.shape[1]))

        np.savez_compressed(path_save+'DSE_u_CMIP6_northern_Boundary.npz',northern_boundary_models)
        np.savez_compressed(path_save+'DSE_u_CMIP6_southern_Boundary.npz',southern_boundary_models)
        np.savez_compressed(path_save+'DSE_u_CMIP6_western_Boundary.npz',western_boundary_models)
        np.savez_compressed(path_save+'DSE_u_CMIP6_eastern_Boundary.npz',eastern_boundary_models)

        np.savez_compressed(path_save+'DSE_v_CMIP6_northern_Boundary.npz',northern_boundary_models)
        np.savez_compressed(path_save+'DSE_v_CMIP6_southern_Boundary.npz',southern_boundary_models)
        np.savez_compressed(path_save+'DSE_v_CMIP6_western_Boundary.npz',western_boundary_models)
        np.savez_compressed(path_save+'DSE_v_CMIP6_eastern_Boundary.npz',eastern_boundary_models)

        for p in range(len(models)):

            try:

                #Applying the function
                DSE_u_model,Lat_model,Lon_model,dx_model,\
                dy_model=DSE_calc(path_save, 'ua', 'ta','zg', models[p],lat_limits,lon_limits,None, None,\
                p_level_interest_lower,p_level_interest_upper,'model')

                DSE_v_model,Lat_model,Lon_model,dx_model,\
                dy_model=DSE_calc(path_save, 'va', 'ta','zg', models[p],lat_limits,lon_limits,None, None,\
                p_level_interest_lower,p_level_interest_upper,'model')

                #-----------------------------------------------------------------------------
                #Interpolating to the common grid size 
                DSE_u_model_interpolated=interpolation_fields(DSE_u_model,Lat_model,Lon_model,Lat_common_b,Lon_common_b,dx_common,dy_common)

                DSE_v_model_interpolated=interpolation_fields(DSE_v_model,Lat_model,Lon_model,Lat_common_b,Lon_common_b,dx_common,dy_common)

                ############################################################################
                #Obtaining the series of each boundary
                #north
                lat_loc_m_n=np.where(np.logical_and(Lat_common_b>=north_boundaries_lat[0], \
                Lat_common_b<=north_boundaries_lat[1]))
                lon_loc_m_n=np.where(np.logical_and(Lon_common_b>=north_boundaries_lon[0], \
                Lon_common_b<=north_boundaries_lon[1]))

                DSE_v_model_n=DSE_v_model_interpolated[:,lat_loc_m_n[0][0]:lat_loc_m_n[0][-1]+1,\
                lon_loc_m_n[0][0]:lon_loc_m_n[0][-1]+1]
                DSE_v_model_northern=np.nanmean(DSE_v_model_n,axis=1)

                DSE_u_model_n=DSE_u_model_interpolated[:,lat_loc_m_n[0][0]:lat_loc_m_n[0][-1]+1,\
                lon_loc_m_n[0][0]:lon_loc_m_n[0][-1]+1]
                DSE_u_model_northern=np.nanmean(DSE_u_model_n,axis=1)

                #south
                lat_loc_m_s=np.where(np.logical_and(Lat_common_b>=south_boundaries_lat[0], \
                Lat_common_b<=south_boundaries_lat[1]))
                lon_loc_m_s=np.where(np.logical_and(Lon_common_b>=south_boundaries_lon[0], \
                Lon_common_b<=south_boundaries_lon[1]))

                DSE_v_model_s=DSE_v_model_interpolated[:,lat_loc_m_s[0][0]:lat_loc_m_s[0][-1]+1,\
                lon_loc_m_s[0][0]:lon_loc_m_s[0][-1]+1]
                DSE_v_model_southern=np.nanmean(DSE_v_model_s,axis=1)

                DSE_u_model_s=DSE_u_model_interpolated[:,lat_loc_m_s[0][0]:lat_loc_m_s[0][-1]+1,\
                lon_loc_m_s[0][0]:lon_loc_m_s[0][-1]+1]
                DSE_u_model_southern=np.nanmean(DSE_u_model_s,axis=1)

                #west
                lat_loc_m_w=np.where(np.logical_and(Lat_common_b>=west_boundaries_lat[0],\
                Lat_common_b<=west_boundaries_lat[1]))
                lon_loc_m_w=np.where(np.logical_and(Lon_common_b>=west_boundaries_lon[0],\
                Lon_common_b<=west_boundaries_lon[1]))

                DSE_u_model_w=DSE_u_model_interpolated[:,lat_loc_m_w[0][0]:lat_loc_m_w[0][-1]+1,\
                lon_loc_m_w[0][0]:lon_loc_m_w[0][-1]+1]
                DSE_u_model_western=np.nanmean(DSE_u_model_w,axis=2)

                DSE_v_model_w=DSE_v_model_interpolated[:,lat_loc_m_w[0][0]:lat_loc_m_w[0][-1]+1,\
                lon_loc_m_w[0][0]:lon_loc_m_w[0][-1]+1]
                DSE_v_model_western=np.nanmean(DSE_v_model_w,axis=2)

                #east
                lat_loc_m_e=np.where(np.logical_and(Lat_common_b>=east_boundaries_lat[0],\
                Lat_common_b<=east_boundaries_lat[1]))
                lon_loc_m_e=np.where(np.logical_and(Lon_common_b>=east_boundaries_lon[0],\
                Lon_common_b<=east_boundaries_lon[1]))

                DSE_u_model_e=DSE_u_model_interpolated[:,lat_loc_m_e[0][0]:lat_loc_m_e[0][-1]+1,\
                lon_loc_m_e[0][0]:lon_loc_m_e[0][-1]+1]
                DSE_u_model_eastern=np.nanmean(DSE_u_model_e,axis=2)

                DSE_v_model_e=DSE_v_model_interpolated[:,lat_loc_m_e[0][0]:lat_loc_m_e[0][-1]+1,\
                lon_loc_m_e[0][0]:lon_loc_m_e[0][-1]+1]
                DSE_v_model_eastern=np.nanmean(DSE_v_model_e,axis=2)

                ############################################################################
                ############################################################################
                #SAVING THE INFORMATION

                #U
                north_u_m_arr=np.load(path_save+'DSE_u_CMIP6_northern_Boundary.npz',\
                allow_pickle=True)['arr_0']
                south_u_m_arr=np.load(path_save+'DSE_u_CMIP6_southern_Boundary.npz',\
                allow_pickle=True)['arr_0']
                west_u_m_arr=np.load(path_save+'DSE_u_CMIP6_western_Boundary.npz',\
                allow_pickle=True)['arr_0']
                east_u_m_arr=np.load(path_save+'DSE_u_CMIP6_eastern_Boundary.npz',\
                allow_pickle=True)['arr_0']

                north_u_m_arr[p,:,:]=DSE_u_model_northern
                south_u_m_arr[p,:,:]=DSE_u_model_southern
                west_u_m_arr[p,:,:]=DSE_u_model_western
                east_u_m_arr[p,:,:]=DSE_u_model_eastern

                np.savez_compressed(path_save+'DSE_u_CMIP6_northern_Boundary.npz',north_u_m_arr)
                np.savez_compressed(path_save+'DSE_u_CMIP6_southern_Boundary.npz',south_u_m_arr)
                np.savez_compressed(path_save+'DSE_u_CMIP6_western_Boundary.npz',west_u_m_arr)
                np.savez_compressed(path_save+'DSE_u_CMIP6_eastern_Boundary.npz',east_u_m_arr)

                #V
                north_v_m_arr=np.load(path_save+'DSE_v_CMIP6_northern_Boundary.npz',\
                allow_pickle=True)['arr_0']
                south_v_m_arr=np.load(path_save+'DSE_v_CMIP6_southern_Boundary.npz',\
                allow_pickle=True)['arr_0']
                west_v_m_arr=np.load(path_save+'DSE_v_CMIP6_western_Boundary.npz',\
                allow_pickle=True)['arr_0']
                east_v_m_arr=np.load(path_save+'DSE_v_CMIP6_eastern_Boundary.npz',\
                allow_pickle=True)['arr_0']

                north_v_m_arr[p,:,:]=DSE_v_model_northern
                south_v_m_arr[p,:,:]=DSE_v_model_southern
                west_v_m_arr[p,:,:]=DSE_v_model_western
                east_v_m_arr[p,:,:]=DSE_v_model_eastern

                np.savez_compressed(path_save+'DSE_v_CMIP6_northern_Boundary.npz',north_v_m_arr)
                np.savez_compressed(path_save+'DSE_v_CMIP6_southern_Boundary.npz',south_v_m_arr)
                np.savez_compressed(path_save+'DSE_v_CMIP6_western_Boundary.npz',west_v_m_arr)
                np.savez_compressed(path_save+'DSE_v_CMIP6_eastern_Boundary.npz',east_v_m_arr)

                models_mse_calc=np.load(path_save+'DSE_models_N.npz',allow_pickle=True)['arr_0']

                models_mse_calc=np.append(models_mse_calc,models[p])
                np.savez_compressed(path_save+'DSE_models_N.npz',models_mse_calc)
            
            except:
                print('Error DSE ', models[p])



#---------------------------------------------------------------------------------------------------------------------------
############################################################################################################################
#CREATING THE PLOTS 
#---------------------------------------------------------------------------------------------------------------------------
############################################################################################################################

list_calculation=['VIMF','DSE']

for i in range(len(list_calculation)):

    if list_calculation[i]=='VIMF':

        try:

            #input
            models=np.load(path_entry+'VIMF_components_models_N.npz',allow_pickle=True)['arr_0']

            #reference data
            east_era5=np.load(path_entry+'VIMF_u_ERA5_eastern_Boundary.npz',\
            allow_pickle=True)['arr_0']
            west_era5=np.load(path_entry+'VIMF_u_ERA5_western_Boundary.npz',\
            allow_pickle=True)['arr_0']
            north_era5=np.load(path_entry+'VIMF_v_ERA5_northern_Boundary.npz',\
            allow_pickle=True)['arr_0']
            south_era5=np.load(path_entry+'VIMF_v_ERA5_southern_Boundary.npz',\
            allow_pickle=True)['arr_0']

            #models
            north_cmip6=np.load(path_entry+'VIMF_v_CMIP6_northern_Boundary.npz',\
            allow_pickle=True)['arr_0']
            south_cmip6=np.load(path_entry+'VIMF_v_CMIP6_southern_Boundary.npz',\
            allow_pickle=True)['arr_0']
            west_cmip6=np.load(path_entry+'VIMF_u_CMIP6_western_Boundary.npz',\
            allow_pickle=True)['arr_0']
            east_cmip6=np.load(path_entry+'VIMF_u_CMIP6_eastern_Boundary.npz',\
            allow_pickle=True)['arr_0']

            print('-----------------------------------------------------------------------------------')
            print('VIMF: files read OK')
            print('-----------------------------------------------------------------------------------')

            #Generating the individual plots for each model 
            #individual_plots_boundary('VIMF',models,path_entry,path_save_ind,dx_common,dy_common)

            series_metrics_bound(east_era5,east_cmip6,models,'VIMF_components_Eastern',path_entry)
            series_metrics_bound(west_era5,west_cmip6,models,'VIMF_components_Western',path_entry)
            series_metrics_bound(north_era5,north_cmip6,models,'VIMF_components_Northern',path_entry)
            series_metrics_bound(south_era5,south_cmip6,models,'VIMF_components_Southern',path_entry)
            print('-----------------------------------------------------------------------------------')
            print('VIMF: filter OK')
            print('-----------------------------------------------------------------------------------')

            ################################################################################
            #Delimiting the longitude and latitudes bands
            north_boundaries_lat=[15,20]
            north_boundaries_lon=[-95,-25]

            south_boundaries_lat=[-60,-55]
            south_boundaries_lon=[-95,-25]

            west_boundaries_lat=[-60,20]
            west_boundaries_lon=[-100,-95]

            east_boundaries_lat=[-60,20]
            east_boundaries_lon=[-25,-20]

            y_label_str='VIMF [Kg/ms]'

            #-------------------------------------------------------------------------------

            labels_x_n=np.round(np.arange(north_boundaries_lon[0],north_boundaries_lon[1],dx_common),0)
            labels_plot_northern=labels_str(labels_x_n,'northern')
            arange_x_n=np.arange(0,labels_x_n.shape[0],1)

            labels_x_s=np.round(np.arange(south_boundaries_lon[0],south_boundaries_lon[1],dx_common),0)
            labels_plot_southern=labels_str(labels_x_s,'southern')
            arange_x_s=np.arange(0,labels_x_s.shape[0],1)

            labels_x_w=np.round(np.arange(west_boundaries_lat[0],west_boundaries_lat[1],dy_common),0)
            labels_plot_western=labels_str(labels_x_w,'western')
            arange_x_w=np.arange(0,labels_x_w.shape[0],1)

            labels_x_e=np.round(np.arange(east_boundaries_lat[0],east_boundaries_lat[1],dy_common),0)
            labels_plot_eastern=labels_str(labels_x_e,'eastern')
            arange_x_e=np.arange(0,labels_x_e.shape[0],1)

            y_limits_pl_u=np.arange(-400,600,100)
            y_limits_pl_v=np.arange(-150,200,50)

            print('-----------------------------------------------------------------------------------')
            print('VIMF: labels OK')
            print('-----------------------------------------------------------------------------------')

            #---------------------------------------------------------------------------------
            #Creating the plot 
            fig = plt.figure(figsize=(15.5,22))
            ax1 = fig.add_subplot(4, 2, 1)
            plot_boundary(ax1,'a. ', north_era5[0],models,north_cmip6[:,0,:],\
            '[Longitude]',arange_x_n,labels_plot_northern,'yes',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'No',8,y_limits_pl_v,['CESM2','IITM-ESM','E3SM-1-1-ECA','GISS-E2-1-H','INM-CM4-8'])

            ax2 = fig.add_subplot(4, 2, 2)
            plot_boundary(ax2,'b. ', north_era5[1],models,north_cmip6[:,1,:],\
            '[Longitude]',arange_x_n,labels_plot_northern,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'No',8,y_limits_pl_v,['CESM2','IITM-ESM','E3SM-1-1-ECA','GISS-E2-1-H','INM-CM4-8'])

            ax3 = fig.add_subplot(4, 2, 3)
            plot_boundary(ax3,'c. ', south_era5[0],models,south_cmip6[:,0,:],\
            '[Longitude]',arange_x_s,labels_plot_southern,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'Yes',8,y_limits_pl_v,['CESM2','IITM-ESM','E3SM-1-1-ECA','GISS-E2-1-H','INM-CM4-8'])

            ax4 = fig.add_subplot(4, 2, 4)
            plot_boundary(ax4,'d. ', south_era5[1],models,south_cmip6[:,1,:],\
            '[Longitude]',arange_x_s,labels_plot_southern,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'Yes',8,y_limits_pl_v,['CESM2','IITM-ESM','E3SM-1-1-ECA','GISS-E2-1-H','INM-CM4-8'])

            ax5 = fig.add_subplot(4, 2, 5)
            plot_boundary(ax5,'e. ', west_era5[0],models,west_cmip6[:,0,:],\
            '[Latitude]',arange_x_w,labels_plot_western,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'No',11,y_limits_pl_u,['CESM2','IITM-ESM','E3SM-1-1-ECA','GISS-E2-1-H','INM-CM4-8'])

            ax6 = fig.add_subplot(4, 2, 6)
            plot_boundary(ax6,'f. ', west_era5[1],models,west_cmip6[:,1,:],\
            '[Latitude]',arange_x_w,labels_plot_western,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'No',11,y_limits_pl_u,['CESM2','IITM-ESM','E3SM-1-1-ECA','GISS-E2-1-H','INM-CM4-8'])

            ax7 = fig.add_subplot(4, 2, 7)
            plot_boundary(ax7,'g. ', east_era5[0],models,east_cmip6[:,0,:],\
            '[Latitude]',arange_x_e,labels_plot_eastern,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'Yes',11,y_limits_pl_u,['CESM2','IITM-ESM','E3SM-1-1-ECA','GISS-E2-1-H','INM-CM4-8'])

            ax8 = fig.add_subplot(4, 2, 8)
            plot_boundary(ax8,'h. ', east_era5[1],models,east_cmip6[:,1,:],\
            '[Latitude]',arange_x_e,labels_plot_eastern,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'Yes',11,y_limits_pl_u,['CESM2','IITM-ESM','E3SM-1-1-ECA','GISS-E2-1-H','INM-CM4-8'])
            
            nrows = 20
            ncols = int(np.ceil(len(models) / float(nrows)))

            #fig.legend(bbox_to_anchor=(0.95, 0.90), ncol=2,loc='upper left', fontsize=str(legends_str))

            plt.text(0.4,1.3,'DJF', fontsize=fig_title_font,rotation='horizontal',transform=ax1.transAxes)
            plt.text(0.4,1.3,'JJA', fontsize=fig_title_font,rotation='horizontal',transform=ax2.transAxes)

            plt.text(-0.3,0.3,'Northern', fontsize=title_str_size,rotation='vertical',transform=ax1.transAxes)
            plt.text(-0.3,0.3,'Southern', fontsize=title_str_size,rotation='vertical',transform=ax3.transAxes)
            plt.text(-0.3,0.3,'Western', fontsize=title_str_size,rotation='vertical',transform=ax5.transAxes)
            plt.text(-0.3,0.3,'Eastern', fontsize=title_str_size,rotation='vertical',transform=ax7.transAxes)

            fig.savefig(path_save+'VIMF_Boundaries_Series_components.png', format = 'png',\
            bbox_inches='tight')
            
            #To save the legend in an independent plot 

            legend=ax1.legend(ncol=ncols,loc='lower left', fontsize=str(legends_str))

            fig.canvas.draw()
            legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
            legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
            legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width, legend_bbox.height))
            legend_squared = legend_ax.legend(
                *ax1.get_legend_handles_labels(), 
                bbox_transform=legend_fig.transFigure,
                bbox_to_anchor=(0,0,1,1),
                frameon=False,
                fancybox=None,
                shadow=False,
                ncol=ncols,
                mode='expand',
            )
            legend_ax.axis('off')
            legend_fig.savefig(
                path_save+'VIMF_Boundaries_Series_components_legend.png', format = 'png',\
            bbox_inches='tight',bbox_extra_artists=[legend_squared],
            )
            plt.close()
        
        except Exception as e:
            print('Error plot VIMF')
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
    
    elif list_calculation[i]=='DSE':

        try:

            #input
            models=np.load(path_entry+'DSE_models_N.npz',allow_pickle=True)['arr_0']

            #reference data
            east_era5=np.load(path_entry+'DSE_u_ERA5_eastern_Boundary.npz',\
            allow_pickle=True)['arr_0']*(1e-10)
            west_era5=np.load(path_entry+'DSE_u_ERA5_western_Boundary.npz',\
            allow_pickle=True)['arr_0']*(1e-10)
            north_era5=np.load(path_entry+'DSE_v_ERA5_northern_Boundary.npz',\
            allow_pickle=True)['arr_0']*(1e-10)
            south_era5=np.load(path_entry+'DSE_v_ERA5_southern_Boundary.npz',\
            allow_pickle=True)['arr_0']*(1e-10)

            #models
            north_cmip6=np.load(path_entry+'DSE_v_CMIP6_northern_Boundary.npz',\
            allow_pickle=True)['arr_0']*(1e-10)
            south_cmip6=np.load(path_entry+'DSE_v_CMIP6_southern_Boundary.npz',\
            allow_pickle=True)['arr_0']*(1e-10)
            west_cmip6=np.load(path_entry+'DSE_u_CMIP6_western_Boundary.npz',\
            allow_pickle=True)['arr_0']*(1e-10)
            east_cmip6=np.load(path_entry+'DSE_u_CMIP6_eastern_Boundary.npz',\
            allow_pickle=True)['arr_0']*(1e-10)

            print('-----------------------------------------------------------------------------------')
            print('DSE: files read OK')
            print('-----------------------------------------------------------------------------------')

            #Generating the individual plots for each model 
            #individual_plots_boundary('MSE',models,path_entry,path_save_ind,dx_common,dy_common)

            series_metrics_bound(east_era5,east_cmip6,models,'DSE_Eastern',path_entry)
            series_metrics_bound(west_era5,west_cmip6,models,'DSE_Western',path_entry)
            series_metrics_bound(north_era5,north_cmip6,models,'DSE_Northern',path_entry)
            series_metrics_bound(south_era5,south_cmip6,models,'DSE_Southern',path_entry)
        
        
            ################################################################################
            #Delimiting the longitude and latitudes bands
            north_boundaries_lat=[15,20]
            north_boundaries_lon=[-95,-25]

            south_boundaries_lat=[-60,-55]
            south_boundaries_lon=[-95,-25]

            west_boundaries_lat=[-60,20]
            west_boundaries_lon=[-100,-95]

            east_boundaries_lat=[-60,20]
            east_boundaries_lon=[-25,-20]

            y_label_str='VIHF [ x 10¹⁰ W]'

            labels_x_n=np.round(np.arange(north_boundaries_lon[0],north_boundaries_lon[1],dx_common),0)
            labels_plot_northern=labels_str(labels_x_n,'northern')
            arange_x_n=np.arange(0,labels_x_n.shape[0],1)

            labels_x_s=np.round(np.arange(south_boundaries_lon[0],south_boundaries_lon[1],dx_common),0)
            labels_plot_southern=labels_str(labels_x_s,'southern')
            arange_x_s=np.arange(0,labels_x_s.shape[0],1)

            labels_x_w=np.round(np.arange(west_boundaries_lat[0],west_boundaries_lat[1],dy_common),0)
            labels_plot_western=labels_str(labels_x_w,'western')
            arange_x_w=np.arange(0,labels_x_w.shape[0],1)

            labels_x_e=np.round(np.arange(east_boundaries_lat[0],east_boundaries_lat[1],dy_common),0)
            labels_plot_eastern=labels_str(labels_x_e,'eastern')
            arange_x_e=np.arange(0,labels_x_e.shape[0],1)

            y_limits_pl_u=np.arange(-70,60,20)
            y_limits_pl_v=np.arange(-15,25,5)

            print('-----------------------------------------------------------------------------------')
            print('DSE: labels OK')
            print('-----------------------------------------------------------------------------------')

            #---------------------------------------------------------------------------------
            #Creating the plot 
            fig = plt.figure(figsize=(15.5,22))
            ax1 = fig.add_subplot(4, 2, 1)
            plot_boundary(ax1,'a. ', north_era5[0],models,north_cmip6[:,0,:],\
            '[Longitude]',arange_x_n,labels_plot_northern,'yes',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'No',8,y_limits_pl_v,['CESM2','IITM-ESM','E3SM-1-1-ECA','INM-CM4-8','GISS-E2-1-H'])

            ax2 = fig.add_subplot(4, 2, 2)
            plot_boundary(ax2,'b. ', north_era5[1],models,north_cmip6[:,1,:],\
            '[Longitude]',arange_x_n,labels_plot_northern,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'No',8,y_limits_pl_v,['CESM2','IITM-ESM','E3SM-1-1-ECA','INM-CM4-8','GISS-E2-1-H'])

            ax3 = fig.add_subplot(4, 2, 3)
            plot_boundary(ax3,'c. ', south_era5[0],models,south_cmip6[:,0,:],\
            '[Longitude]',arange_x_s,labels_plot_southern,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'Yes',8,y_limits_pl_v,['CESM2','IITM-ESM','E3SM-1-1-ECA','INM-CM4-8','GISS-E2-1-H'])

            ax4 = fig.add_subplot(4, 2, 4)
            plot_boundary(ax4,'d. ', south_era5[1],models,south_cmip6[:,1,:],\
            '[Longitude]',arange_x_s,labels_plot_southern,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'Yes',8,y_limits_pl_v,['CESM2','IITM-ESM','E3SM-1-1-ECA','INM-CM4-8','GISS-E2-1-H'])

            ax5 = fig.add_subplot(4, 2, 5)
            plot_boundary(ax5,'e. ', west_era5[0],models,west_cmip6[:,0,:],\
            '[Latitude]',arange_x_w,labels_plot_western,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'No',11,y_limits_pl_u,['CESM2','IITM-ESM','E3SM-1-1-ECA','INM-CM4-8','GISS-E2-1-H'])

            ax6 = fig.add_subplot(4, 2, 6)
            plot_boundary(ax6,'f. ', west_era5[1],models,west_cmip6[:,1,:],\
            '[Latitude]',arange_x_w,labels_plot_western,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'No',11,y_limits_pl_u,['CESM2','IITM-ESM','E3SM-1-1-ECA','INM-CM4-8','GISS-E2-1-H'])

            ax7 = fig.add_subplot(4, 2, 7)
            plot_boundary(ax7,'g. ', east_era5[0],models,east_cmip6[:,0,:],\
            '[Latitude]',arange_x_e,labels_plot_eastern,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'Yes',11,y_limits_pl_u,['CESM2','IITM-ESM','E3SM-1-1-ECA','INM-CM4-8','GISS-E2-1-H'])

            ax8 = fig.add_subplot(4, 2, 8)
            plot_boundary(ax8,'h. ', east_era5[1],models,east_cmip6[:,1,:],\
            '[Latitude]',arange_x_e,labels_plot_eastern,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'Yes',11,y_limits_pl_u,['CESM2','IITM-ESM','E3SM-1-1-ECA','INM-CM4-8','GISS-E2-1-H'])
            
            nrows = 20
            ncols = int(np.ceil(len(models) / float(nrows)))

            #fig.legend(bbox_to_anchor=(0.95, 0.90), ncol=2,loc='upper left', fontsize=str(legends_str))

            plt.text(0.4,1.3,'DJF', fontsize=fig_title_font,rotation='horizontal',transform=ax1.transAxes)
            plt.text(0.4,1.3,'JJA', fontsize=fig_title_font,rotation='horizontal',transform=ax2.transAxes)

            plt.text(-0.3,0.3,'Northern', fontsize=title_str_size,rotation='vertical',transform=ax1.transAxes)
            plt.text(-0.3,0.3,'Southern', fontsize=title_str_size,rotation='vertical',transform=ax3.transAxes)
            plt.text(-0.3,0.3,'Western', fontsize=title_str_size,rotation='vertical',transform=ax5.transAxes)
            plt.text(-0.3,0.3,'Eastern', fontsize=title_str_size,rotation='vertical',transform=ax7.transAxes)

            fig.savefig(path_save+'DSE_Boundaries_Series.png', format = 'png',\
            bbox_inches='tight')
            
            legend=ax1.legend(ncol=ncols,loc='lower left', fontsize=str(legends_str))

            fig.canvas.draw()
            legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
            legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
            legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width, legend_bbox.height))
            legend_squared = legend_ax.legend(
                *ax1.get_legend_handles_labels(), 
                bbox_transform=legend_fig.transFigure,
                bbox_to_anchor=(0,0,1,1),
                frameon=False,
                fancybox=None,
                shadow=False,
                ncol=ncols,
                mode='expand',
            )
            legend_ax.axis('off')
            legend_fig.savefig(
                path_save+'DSE_Boundaries_Series_legend.png', format = 'png',\
            bbox_inches='tight',bbox_extra_artists=[legend_squared],
            )
            plt.close()
        
        except Exception as e:
            print('Error plot DSE')
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")

