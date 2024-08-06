"""
Code to create the maps of the fields of VIMF and VIMHF 
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


#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#Path_save is the path that contains all the created files
path_entry='/scratchx/lfita/'
path_save='/scratchx/lfita/'

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

####################################################################################################################
#####################################################################################################################
#FUNCTIONS TO USE 
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

#-------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
#Performing the calculations 
with open(path_save+'Models_var_availability.pkl', 'rb') as fp:
    dict_models = pickle.load(fp)


list_calculation=['VIMF','MSE']

#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
#1. ERA5
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------

for i in range(len(list_calculation)):

    #--------------------------------------------------------------------------------------------------------------
    if list_calculation[i]=='VIMF':


        #Delimiting the longitude and latitudes bands
        lat_limits=lat_limits_F
        lon_limits=lon_limits_F

        p_level_interest_upper=10000.0
        p_level_interest_lower=100000.0

        try:

            VIMF_ref,u_comp,v_comp,Lat_ref,Lon_ref,dx_ref,\
            dy_ref=VIMF_calc_field(path_save, 'u', 'v', 'q', 'ERA5',lat_limits,lon_limits,None, None,\
            p_level_interest_lower,p_level_interest_upper,'ERA5')

            #-----------------------------------------------------------------------------------------------------------------------
            #Saving the spatial fields 
            np.savez_compressed(path_save+'ERA5_VIMF_fields.npz',VIMF_ref)
            np.savez_compressed(path_save+'ERA5_uVIMF_fields.npz',u_comp)
            np.savez_compressed(path_save+'ERA5_vVIMF_fields.npz',v_comp)
            np.savez_compressed(path_save+'ERA5_VIMF_fields_Lat.npz',Lat_ref)
            np.savez_compressed(path_save+'ERA5_VIMF_fields_Lon.npz',Lon_ref)

            std_ref(VIMF_ref, path_save, 'VIMF')

            print('####################################')
            print('VIMF: std_ref OK')
            print('####################################')
        
        except Exception as e:
            print('Error ERA5 VIMF')
            # FROM: https://stackoverflow.com/questions/1483429/how-do-i-print-an-exception-in-python
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
    
    elif list_calculation[i]=='MSE':


        #Delimiting the longitude and latitudes bands
        lat_limits=lat_limits_F
        lon_limits=lon_limits_F

        p_level_interest_upper=10000.0
        p_level_interest_lower=100000.0

        #Applying the function

        try:

            MSE_ref,Lat_ref,Lon_ref,dx_ref,\
            dy_ref=MSE_calc(path_save, 'v', 'q', 'ta','geopt', 'ERA5',lat_limits,lon_limits,None, None,\
            p_level_interest_lower,p_level_interest_upper,'ERA5')

            print('####################################')
            print('MSE: var_array OK')
            print('####################################')

            #-----------------------------------------------------------------------------------------------------------------------
            #Saving the spatial fields 
            np.savez_compressed(path_save+'ERA5_MSE_fields.npz',MSE_ref)
            np.savez_compressed(path_save+'ERA5_MSE_fields_Lat.npz',Lat_ref)
            np.savez_compressed(path_save+'ERA5_MSE_fields_Lon.npz',Lon_ref)

            std_ref(MSE_ref, path_save, 'MSE')

            print('####################################')
            print('MSE: std_ref OK')
            print('####################################')
        
        except Exception as e:
            print('Error ERA5 MSE')
            # FROM: https://stackoverflow.com/questions/1483429/how-do-i-print-an-exception-in-python
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
        
    else: 
        pass 


#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
#2. CMIP6 MODELS 
#---------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------
list_calculation=['VIMF','MSE']

for i in range(len(list_calculation)):
        
    if list_calculation[i]=='VIMF':

        #Delimiting the longitude and latitudes bands
        lat_limits=lat_limits_F
        lon_limits=lon_limits_F

        p_level_interest_upper=10000.0
        p_level_interest_lower=100000.0

        models_ua=list(dict_models['ua'])
        models_va=list(dict_models['va'])
        models_hus=list(dict_models['hus'])

        models_wind=np.intersect1d(models_ua, models_va)
        models=np.intersect1d(models_wind, models_hus)

        #---------------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------
        #creating and saving the matrices of the fields 
        models_app=np.array([])

        taylor_diagram_metrics=pd.DataFrame(columns=['Model','corr_DJF','corr_JJA',\
        'corr_MAM','corr_SON','std_DJF','std_JJA','std_MAM','std_SON'])

        taylor_diagram_metrics.to_csv(path_save+'taylorDiagram_metrics_VIMF.csv')
        np.savez_compressed(path_entry+'VIMF_fields_models_N.npz',models_app)

        var_array_ref=np.load(path_save+'ERA5_VIMF_fields.npz')['arr_0']
        lat_refe=np.load(path_save+'ERA5_VIMF_fields_Lat.npz')['arr_0']
        lon_refe=np.load(path_save+'ERA5_VIMF_fields_Lon.npz')['arr_0']

        for p in range(len(models)):

            try:

                VIMF_model,u_comp_model,v_comp_model,Lat_model,Lon_model,dx_model,\
                dy_model=VIMF_calc_field(path_save, 'ua', 'va', 'hus', models[p],lat_limits,lon_limits,None, None,\
                p_level_interest_lower,p_level_interest_upper,'model')

                #-----------------------------------------------------------------------------------------------------------------------            
                #ERA5 interpolation to the model's gridsize 
                era5_field_interp=interpolation_fields(var_array_ref,lat_refe,lon_refe,Lat_model,Lon_model,dx_model, dy_model)
                #calculating the metrics
                corr_m_o,std_m=taylor_diagram_metrics_def(era5_field_interp,VIMF_model)
                #Calculating the bias
                bias_model= VIMF_model - era5_field_interp
                #Model's interpolation to a common gridsize
                VIMF_field_interp=interpolation_fields(VIMF_model,Lat_model,Lon_model,Lat_common,Lon_common,dx_common, dy_common)
                model_ua_field_interp=interpolation_fields(u_comp_model,Lat_model,Lon_model,Lat_common,Lon_common,dx_common, dy_common)
                model_va_field_interp=interpolation_fields(v_comp_model,Lat_model,Lon_model,Lat_common,Lon_common,dx_common, dy_common)
                model_bias_field_interp=interpolation_fields(bias_model,Lat_model,Lon_model,Lat_common,Lon_common,dx_common, dy_common)

                #-------------------------------------------------------------------------------------------------------------------
                #Saving the npz
                models_vimf_calc=np.load(path_save+'VIMF_fields_models_N.npz',allow_pickle=True)['arr_0']

                models_vimf_calc=np.append(models_vimf_calc,models[p])
                np.savez_compressed(path_save+'VIMF_fields_models_N.npz',models_vimf_calc)

                np.savez_compressed(path_save+models[p]+'_VIMF_MMM_meanFields.npz',VIMF_field_interp)
                np.savez_compressed(path_save+models[p]+'_uVIMF_MMM_meanFields.npz',model_ua_field_interp)
                np.savez_compressed(path_save+models[p]+'_vVIMF_MMM_meanFields.npz',model_va_field_interp)
                np.savez_compressed(path_save+models[p]+'_VIMF_MMM_biasFields.npz',model_bias_field_interp)

                #Saving the performance metrics 
                taylor_diagram_metrics_DT=pd.read_csv(path_save+'taylorDiagram_metrics_VIMF.csv', index_col=[0])

                newRow_metrics=pd.DataFrame({'Model':[models[p]],'corr_DJF': [corr_m_o[0]],\
                'corr_JJA': [corr_m_o[1]],'corr_MAM': [corr_m_o[2]],'corr_SON': [corr_m_o[3]],\
                'std_DJF':[std_m[0]],'std_JJA':[std_m[1]],'std_MAM':[std_m[2]],'std_SON':[std_m[3]]})

                #taylor_diagram_metrics_DT=taylor_diagram_metrics_DT.append(newRow_metrics,\
                #ignore_index=True)
                taylor_diagram_metrics_DT = pd.concat([taylor_diagram_metrics_DT, newRow_metrics], ignore_index=True)

                taylor_diagram_metrics_DT.to_csv(path_save+'taylorDiagram_metrics_VIMF.csv')
            
            except Exception as e:
                print('Error VIMF ', models[p])
                # FROM: https://stackoverflow.com/questions/1483429/how-do-i-print-an-exception-in-python
                print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
        

    elif list_calculation[i]=='MSE':


        #Delimiting the longitude and latitudes bands
        lat_limits=lat_limits_F
        lon_limits=lon_limits_F


        p_level_interest_upper=10000.0
        p_level_interest_lower=100000.0

        models_va=list(dict_models['va'])
        models_hus=list(dict_models['hus'])
        models_ta=list(dict_models['ta'])
        models_zg=list(dict_models['zg'])

        models_1=np.intersect1d(models_va, models_hus)
        models_2=np.intersect1d(models_1, models_ta)
        models=np.intersect1d(models_2, models_zg)

        #---------------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------
        #creating and saving the matrices of the fields 
        models_app=np.array([])

        taylor_diagram_metrics=pd.DataFrame(columns=['Model','corr_DJF','corr_JJA',\
        'corr_MAM','corr_SON','std_DJF','std_JJA','std_MAM','std_SON'])

        taylor_diagram_metrics.to_csv(path_save+'taylorDiagram_metrics_MSE.csv')
        np.savez_compressed(path_save+'MSE_fields_models_N.npz',models_app)

        var_array_ref=np.load(path_save+'ERA5_MSE_fields.npz')['arr_0']
        lat_refe=np.load(path_save+'ERA5_MSE_fields_Lat.npz')['arr_0']
        lon_refe=np.load(path_save+'ERA5_MSE_fields_Lon.npz')['arr_0']

        for p in range(len(models)):

            try:

                #Applying the function
                MSE_model,Lat_model,Lon_model,dx_model,\
                dy_model=MSE_calc(path_save, 'va', 'hus', 'ta','zg', models[p],lat_limits,lon_limits,None, None,\
                p_level_interest_lower,p_level_interest_upper,'model')

                #-----------------------------------------------------------------------------------------------------------------------            
                #ERA5 interpolation to the model's gridsize 
                era5_field_interp=interpolation_fields(var_array_ref,lat_refe,lon_refe,Lat_model,Lon_model,dx_model, dy_model)
                #calculating the metrics
                corr_m_o,std_m=taylor_diagram_metrics_def(era5_field_interp,MSE_model)
                #Calculating the bias
                bias_model= MSE_model - era5_field_interp
                #Model's interpolation to a common gridsize
                MSE_field_interp=interpolation_fields(MSE_model,Lat_model,Lon_model,Lat_common,Lon_common,dx_common, dy_common)
                model_bias_field_interp=interpolation_fields(bias_model,Lat_model,Lon_model,Lat_common,Lon_common,dx_common, dy_common)

                #-------------------------------------------------------------------------------------------------------------------
                #Saving the npz
                models_MSE_calc=np.load(path_save+'MSE_fields_models_N.npz',allow_pickle=True)['arr_0']

                models_MSE_calc=np.append(models_MSE_calc,models[p])
                np.savez_compressed(path_save+'MSE_fields_models_N.npz',models_MSE_calc)

                np.savez_compressed(path_save+models[p]+'_MSE_MMM_meanFields.npz',MSE_field_interp)
                np.savez_compressed(path_save+models[p]+'_MSE_MMM_biasFields.npz',model_bias_field_interp)

                #Saving the performance metrics 
                taylor_diagram_metrics_DT=pd.read_csv(path_save+'taylorDiagram_metrics_MSE.csv', index_col=[0])

                newRow_metrics=pd.DataFrame({'Model':[models[p]],'corr_DJF': [corr_m_o[0]],\
                'corr_JJA': [corr_m_o[1]],'corr_MAM': [corr_m_o[2]],'corr_SON': [corr_m_o[3]],\
                'std_DJF':[std_m[0]],'std_JJA':[std_m[1]],'std_MAM':[std_m[2]],'std_SON':[std_m[3]]})

                #taylor_diagram_metrics_DT=taylor_diagram_metrics_DT.append(newRow_metrics,\
                #ignore_index=True)
                taylor_diagram_metrics_DT = pd.concat([taylor_diagram_metrics_DT, newRow_metrics], ignore_index=True)

                taylor_diagram_metrics_DT.to_csv(path_save+'taylorDiagram_metrics_MSE.csv')
            
            except Exception as e:
                print('Error MSE ', models[p])
                # FROM: https://stackoverflow.com/questions/1483429/how-do-i-print-an-exception-in-python
                print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")


#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#PLOTTING 
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
list_calculation=['VIMF','MSE']
for i in range(len(list_calculation)):

    if list_calculation[i]=='VIMF':

        try:

            #input
            models_o=np.load(path_save+'VIMF_fields_models_N.npz',allow_pickle=True)['arr_0']

            models=[]

            for o in range(len(models_o)):
                if models_o[o]=='CESM2' or models_o[o]=='IITM-ESM' or models_o[o]=='E3SM-1-1-ECA' or models_o[o]=='GISS-E2-1-H' or models_o[o]=='INM-CM4-8':
                    pass 
                else:
                    models.append(models_o[o])
            
            models=np.array(models)

            

            ################################################################################
            #Obtaining the ensamble

            print ('  Obteniendo VIMF mean ...')
            mag_mmm=seasonal_ensamble(models,path_save,\
            'VIMF_MMM_meanFields',len(Lat_common),len(Lon_common))

            print ('  Obteniendo VIMF bias ...')
            mag_mmm_bias=seasonal_ensamble(models,path_save,\
            'VIMF_MMM_biasFields',len(Lat_common),len(Lon_common))

            print ('  Obteniendo uVIMF mean ...')
            ua_mmm=seasonal_ensamble(models,path_save,\
            'uVIMF_MMM_meanFields',len(Lat_common),len(Lon_common))

            print ('  Obteniendo vVIMF mean ...')
            va_mmm=seasonal_ensamble(models,path_save,\
            'vVIMF_MMM_meanFields',len(Lat_common),len(Lon_common))

            print ('  agreement_sign VIMF ...')
            bias_mmm_agreement=agreement_sign(models,path_save,'VIMF_MMM_biasFields',\
                                            len(Lat_common),len(Lon_common))
            
            print('-----------------------------------------------------------------------------------')
            print('VIMF: files read OK')
            print('-----------------------------------------------------------------------------------')

            #PLOT
            print ("  Leyendo '" + path_save+"taylorDiagram_metrics_VIMF.csv'")
            models_metrics=pd.read_csv(path_save+'taylorDiagram_metrics_VIMF.csv', index_col=[0])

            models_metrics=models_metrics.drop(models_metrics[models_metrics['Model']=='CESM2'].index).reset_index(drop=True)
            models_metrics=models_metrics.drop(models_metrics[models_metrics['Model']=='IITM-ESM'].index).reset_index(drop=True)
            models_metrics=models_metrics.drop(models_metrics[models_metrics['Model']=='E3SM-1-1-ECA'].index).reset_index(drop=True)
            models_metrics=models_metrics.drop(models_metrics[models_metrics['Model']=='GISS-E2-1-H'].index).reset_index(drop=True)
            models_metrics=models_metrics.drop(models_metrics[models_metrics['Model']=='INM-CM4-8'].index).reset_index(drop=True)

            print ("  Leyendo '" + path_save+"reference_std_original.csv'")
            ref_std=pd.read_csv(path_save+'reference_std_original.csv',index_col=[0])

            print ('Ploteando ...')

            plot_label='VIMF [Kg/ms]'
            limits_var=np.arange(0,425,25)
            limits_bias=np.arange(-100,125,5)

            fig=plt.figure(figsize=(24,14))
            colorbar_attributes=[0.92, 0.37,  0.017,0.24]

            colorbar_attributes_bias=[0.92, 0.1,  0.017,0.24]

            print('-----------------------------------------------------------------------------------')
            print('VIMF fields: files metrics OK')
            print('-----------------------------------------------------------------------------------')

            lon2D, lat2D = np.meshgrid(Lon_common, Lat_common)
            projection=ccrs.PlateCarree()

            print ('  extent')
            extent = [min(Lon_common),max(Lon_common),min(Lat_common),max(Lat_common)]

            print ('  ploteando DJF Taylor')
            taylor=td_plots(fig,'DJF',ref_std,models_metrics,'VIMF',len(models),341,'a.',title_str_size,'no',None)

            taylor=td_plots(fig,'MAM',ref_std,models_metrics,'VIMF',len(models),342,'b.',title_str_size,'no',None)

            taylor=td_plots(fig,'JJA',ref_std,models_metrics,'VIMF',len(models),343,'c.',title_str_size,'no',None)

            taylor=td_plots(fig,'SON',ref_std,models_metrics,'VIMF',len(models),344,'d.',title_str_size,'no',None)

            print('-----------------------------------------------------------------------------------')
            print('VIMF fields: td plots OK')
            print('-----------------------------------------------------------------------------------')

            print ('    shapes 0 mag_mmm', mag_mmm[0].shape, 'ua', ua_mmm[0].shape,  \
              'va', va_mmm[0].shape, 'lon2D', lon2D.shape, 'lat2D', lat2D.shape)
            ax5 = fig.add_subplot(3, 4, 5, projection=projection)
            cs=plotMap_vector(ax5,mag_mmm[0],ua_mmm[0],va_mmm[0],lon2D,lat2D,cmap_VIMF,limits_var,'e.',extent, projection,title_str_size)

            print ('    shapes 2 mag_mmm', mag_mmm[2].shape, 'ua', ua_mmm[2].shape,  \
              'va', va_mmm[2].shape, 'lon2D', lon2D.shape, 'lat2D', lat2D.shape)
            ax6 = fig.add_subplot(3, 4, 6, projection=projection)
            cs=plotMap_vector(ax6,mag_mmm[2],ua_mmm[2],va_mmm[2],lon2D,lat2D,cmap_VIMF,limits_var,'f.',extent, projection,title_str_size)

            print ('    shapes 1 mag_mmm', mag_mmm[1].shape, 'ua', ua_mmm[1].shape,  \
              'va', va_mmm[1].shape, 'lon2D', lon2D.shape, 'lat2D', lat2D.shape)
            ax7 = fig.add_subplot(3, 4, 7, projection=projection)
            cs=plotMap_vector(ax7,mag_mmm[1],ua_mmm[1],va_mmm[1],lon2D,lat2D,cmap_VIMF,limits_var,'g.',extent, projection,title_str_size,)

            print ('    shapes 3 mag_mmm', mag_mmm[3].shape, 'ua', ua_mmm[3].shape,  \
              'va', va_mmm[3].shape, 'lon2D', lon2D.shape, 'lat2D', lat2D.shape)
            ax8 = fig.add_subplot(3, 4, 8, projection=projection)
            cs=plotMap_vector(ax8,mag_mmm[3],ua_mmm[3],va_mmm[3],lon2D,lat2D,cmap_VIMF,limits_var,'h.',extent, projection,title_str_size)

            print('-----------------------------------------------------------------------------------')
            print('VIMF fields: plot vector OK')
            print('-----------------------------------------------------------------------------------')

            ax9 = fig.add_subplot(3, 4, 9, projection=projection)
            print ('  plotting ax9')
            #lotMap(axs,var_data,lonPlot,latPlot,colorMap,limits,title_label,extent, projection,title_font2,scatter_status,points_scatter,land_cov):
            csb=plotMap(ax9,mag_mmm_bias[0],lon2D,lat2D,cmap_bias,limits_bias,'i.',extent, projection,title_str_size,'yes',bias_mmm_agreement[0],'no')

            print ('    shapes 2 mag_bias', mag_mmm_bias[2].shape)
            ax10 = fig.add_subplot(3, 4, 10, projection=projection)
            csb=plotMap(ax10,mag_mmm_bias[2],lon2D,lat2D,cmap_bias,limits_bias,'j.',extent, projection,title_str_size,'yes',bias_mmm_agreement[2],'no')

            print ('    shapes 1 mag_bias', mag_mmm_bias[1].shape)
            ax11 = fig.add_subplot(3, 4, 11, projection=projection)
            csb=plotMap(ax11,mag_mmm_bias[1],lon2D,lat2D,cmap_bias,limits_bias,'k.',extent, projection,title_str_size,'yes',bias_mmm_agreement[1],'no')

            print ('    shapes 3 mag_bias', mag_mmm_bias[3].shape)
            ax12 = fig.add_subplot(3, 4, 12, projection=projection)
            csb=plotMap(ax12,mag_mmm_bias[3],lon2D,lat2D,cmap_bias,limits_bias,'l.',extent, projection,title_str_size,'yes',bias_mmm_agreement[3],'no')

            print ('  añadiendo colorbars')

            cbar_ax = fig.add_axes(colorbar_attributes)
            cb = fig.colorbar(cs,cax=cbar_ax, orientation="vertical")
            cb.ax.tick_params(labelsize=tick_labels_str)
            cb.set_label(plot_label,fontsize=xy_label_str)

            cbar_ax_b = fig.add_axes(colorbar_attributes_bias)
            cbb = fig.colorbar(csb,cax=cbar_ax_b, orientation="vertical")
            cbb.ax.tick_params(labelsize=tick_labels_str)
            cbb.set_label(plot_label,fontsize=xy_label_str)

            plt.text(0.3,2.42,'DJF', fontsize=title_str_size,rotation='horizontal',transform=ax5.transAxes)
            plt.text(0.3,2.42,'MAM', fontsize=title_str_size,rotation='horizontal',transform=ax6.transAxes)
            plt.text(0.3,2.42,'JJA', fontsize=title_str_size,rotation='horizontal',transform=ax7.transAxes)
            plt.text(0.3,2.42,'SON', fontsize=title_str_size,rotation='horizontal',transform=ax8.transAxes)
            #fig.subplots_adjust(hspace=0.3)
            plt.savefig(path_save+'VIMF_fields.png', \
            format = 'png', bbox_inches='tight')
            
            #To save the legend independently
            dia=taylor

            legend= fig.legend(dia.samplePoints,
            [ p.get_label() for p in dia.samplePoints ],
            numpoints=1, prop=dict(size='small'),bbox_to_anchor=(1.11, 0.85) \
            ,ncol=2,loc='right')

            ncols=2

            fig.canvas.draw()
            legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
            legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
            legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width, legend_bbox.height))
            legend_squared = legend_ax.legend(
            *dia._ax.get_legend_handles_labels(), 
            bbox_transform=legend_fig.transFigure,
            bbox_to_anchor=(0,0,1.1,1),
            frameon=False,
            fancybox=None,
            shadow=False,
            ncol=ncols,
            mode='expand',
            )
            legend_ax.axis('off')
            legend_fig.savefig(
            path_save+'VIMF_fields_legend.png', format = 'png',\
            bbox_inches='tight',bbox_extra_artists=[legend_squared],
            ) 

            plt.close()

        except Exception as e:
            print('Error plot VIMF')
            # FROM: https://stackoverflow.com/questions/1483429/how-do-i-print-an-exception-in-python
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")


    elif list_calculation[i]=='MSE':

        try:

            #input
            models_o=np.load(path_save+'MSE_fields_models_N.npz',allow_pickle=True)['arr_0']

            models=[]

            for o in range(len(models_o)):
                if models_o[o]=='CESM2' or models_o[o]=='IITM-ESM' or models_o[o]=='E3SM-1-1-ECA' or models_o[o]=='GISS-E2-1-H' or models_o[o]=='INM-CM4-8':
                    pass 
                else:
                    models.append(models_o[o])
            
            models=np.array(models)

            ################################################################################
            #Obtaining the ensamble

            var_mmm=seasonal_ensamble(models,path_save,\
            'MSE_MMM_meanFields',len(Lat_common),len(Lon_common))*(1e-10)

            var_mmm_bias=seasonal_ensamble(models,path_save,\
            'MSE_MMM_biasFields',len(Lat_common),len(Lon_common))*(1e-10)

            bias_mmm_agreement=agreement_sign(models,path_save,'MSE_MMM_biasFields',\
                                            len(Lat_common),len(Lon_common))
            
            print('-----------------------------------------------------------------------------------')
            print('MSE: files read OK')
            print('-----------------------------------------------------------------------------------')

            ################################################################################
            #PLOT
            models_metrics=pd.read_csv(path_save+'taylorDiagram_metrics_MSE.csv', index_col=[0])

            models_metrics=models_metrics.drop(models_metrics[models_metrics['Model']=='CESM2'].index).reset_index(drop=True)
            models_metrics=models_metrics.drop(models_metrics[models_metrics['Model']=='IITM-ESM'].index).reset_index(drop=True)
            models_metrics=models_metrics.drop(models_metrics[models_metrics['Model']=='E3SM-1-1-ECA'].index).reset_index(drop=True)
            models_metrics=models_metrics.drop(models_metrics[models_metrics['Model']=='GISS-E2-1-H'].index).reset_index(drop=True)
            models_metrics=models_metrics.drop(models_metrics[models_metrics['Model']=='INM-CM4-8'].index).reset_index(drop=True)

            ref_std=pd.read_csv(path_save+'reference_std_original.csv',index_col=[0])

            
            plot_label='VIHF [ x 10¹⁰ W]'
            limits_var=np.arange(-10,11,1)
            limits_bias=np.arange(-9,10,1)

            cmap_plot=['gist_rainbow_r','terrain','terrain_r','rainbow','RdBu']

            fig=plt.figure(figsize=(24,14))
            colorbar_attributes=[0.92, 0.37,  0.017,0.24]

            colorbar_attributes_bias=[0.92, 0.1,  0.017,0.24]

            print('-----------------------------------------------------------------------------------')
            print('MSE: files metrics OK')
            print('-----------------------------------------------------------------------------------')

            lon2D, lat2D = np.meshgrid(Lon_common, Lat_common)
            projection=ccrs.PlateCarree()
            extent = [min(Lon_common),max(Lon_common),min(Lat_common),max(Lat_common)]

            taylor=td_plots(fig,'DJF',ref_std,models_metrics,'MSE',len(models),341,'a.',title_str_size,'no',None)

            taylor=td_plots(fig,'MAM',ref_std,models_metrics,'MSE',len(models),342,'b.',title_str_size,'no',None)

            taylor=td_plots(fig,'JJA',ref_std,models_metrics,'MSE',len(models),343,'c.',title_str_size,'no',None)

            taylor=td_plots(fig,'SON',ref_std,models_metrics,'MSE',len(models),344,'d.',title_str_size,'no',None)

            ax5 = fig.add_subplot(3, 4, 5, projection=projection)
            cs=plotMap(ax5,var_mmm[0],lon2D,lat2D,cmap_MSE,limits_var,'e.',extent, projection,title_str_size,'no',None,'no')

            ax6 = fig.add_subplot(3, 4, 6, projection=projection)
            cs=plotMap(ax6,var_mmm[2],lon2D,lat2D,cmap_MSE,limits_var,'f.',extent, projection,title_str_size,'no',None,'no')

            ax7 = fig.add_subplot(3, 4, 7, projection=projection)
            cs=plotMap(ax7,var_mmm[1],lon2D,lat2D,cmap_MSE,limits_var,'g.',extent, projection,title_str_size,'no',None,'no')

            ax8 = fig.add_subplot(3, 4, 8, projection=projection)
            cs=plotMap(ax8,var_mmm[3],lon2D,lat2D,cmap_MSE,limits_var,'h.',extent, projection,title_str_size,'no',None,'no')

            ax9 = fig.add_subplot(3, 4, 9, projection=projection)
            csb=plotMap(ax9,var_mmm_bias[0],lon2D,lat2D,cmap_bias,limits_bias,'i.',extent, projection,title_str_size,'yes',bias_mmm_agreement[0],'no')

            ax10 = fig.add_subplot(3, 4, 10, projection=projection)
            csb=plotMap(ax10,var_mmm_bias[2],lon2D,lat2D,cmap_bias,limits_bias,'j.',extent, projection,title_str_size,'yes',bias_mmm_agreement[2],'no')

            ax11 = fig.add_subplot(3, 4, 11, projection=projection)
            csb=plotMap(ax11,var_mmm_bias[1],lon2D,lat2D,cmap_bias,limits_bias,'k.',extent, projection,title_str_size,'yes',bias_mmm_agreement[1],'no')

            ax12 = fig.add_subplot(3, 4, 12, projection=projection)
            csb=plotMap(ax12,var_mmm_bias[3],lon2D,lat2D,cmap_bias,limits_bias,'l.',extent, projection,title_str_size,'yes',bias_mmm_agreement[2],'no')

            cbar_ax = fig.add_axes(colorbar_attributes)
            cb = fig.colorbar(cs,cax=cbar_ax, orientation="vertical")
            cb.ax.tick_params(labelsize=tick_labels_str)
            cb.set_label(plot_label,fontsize=xy_label_str)

            cbar_ax_b = fig.add_axes(colorbar_attributes_bias)
            cbb = fig.colorbar(csb,cax=cbar_ax_b, orientation="vertical")
            cbb.ax.tick_params(labelsize=tick_labels_str)
            cbb.set_label(plot_label,fontsize=xy_label_str)

            plt.text(0.3,2.42,'DJF', fontsize=title_str_size,rotation='horizontal',transform=ax5.transAxes)
            plt.text(0.3,2.42,'MAM', fontsize=title_str_size,rotation='horizontal',transform=ax6.transAxes)
            plt.text(0.3,2.42,'JJA', fontsize=title_str_size,rotation='horizontal',transform=ax7.transAxes)
            plt.text(0.3,2.42,'SON', fontsize=title_str_size,rotation='horizontal',transform=ax8.transAxes)
            #fig.subplots_adjust(hspace=0.3)
            plt.savefig(path_save+'MSE_fields.png', \
            format = 'png', bbox_inches='tight')
            
            #To save the legend independently
            dia=taylor

            legend= fig.legend(dia.samplePoints,
            [ p.get_label() for p in dia.samplePoints ],
            numpoints=1, prop=dict(size='small'),bbox_to_anchor=(1.11, 0.85) \
            ,ncol=2,loc='right')

            ncols=2

            fig.canvas.draw()
            legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
            legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
            legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width, legend_bbox.height))
            legend_squared = legend_ax.legend(
            *dia._ax.get_legend_handles_labels(), 
            bbox_transform=legend_fig.transFigure,
            bbox_to_anchor=(0,0,1.1,1),
            frameon=False,
            fancybox=None,
            shadow=False,
            ncol=ncols,
            mode='expand',
            )
            legend_ax.axis('off')
            legend_fig.savefig(
            path_save+'MSE_fields_legend.png', format = 'png',\
            bbox_inches='tight',bbox_extra_artists=[legend_squared],
            ) 
            plt.close()
        
        except Exception as e:
            print('Error plot MSE')
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")




