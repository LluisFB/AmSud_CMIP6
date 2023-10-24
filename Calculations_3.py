"""
Code to perform the calculations of the paper from the set of models CMIP6
models 

Part 3

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
#path_save='/home/iccorreasa/Documentos/Paper_CMIP6_models/PLOTS_paper/PAPER_FINAL/npz/' #CHANGE
path_save='/scratchx/lfita/'

#-----------------------------------------------------------------------------------------------------------------------
gridsize_df=pd.read_csv(path_save+'CMIP6_models_GridSize_lat_lon_Amon.csv', index_col=[0])
gridsize_df_tos=pd.read_csv(path_save+'CMIP6_models_GridSize_lat_lon_Omon.csv', index_col=[0])


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

#-------------------------------------------------------------------------------------------------------------------
#3------------------------------------------------------------------------------------------------------------------
#Performing the calculations 
with open(path_save+'Models_var_availability.pkl', 'rb') as fp:
    dict_models = pickle.load(fp)


list_calculation=['wind_850','wind_200','Subtropical_highs','Precipitation','Regional_cells',\
                  'SST','Wind_indices','Bolivian_high','VIMF','qu_qv','MSE','tu_tv']


for i in range(len(list_calculation)):

    #--------------------------------------------------------------------------------------------------------------
    if list_calculation[i]=='Wind_indices':

        lon_limits_subtropical=[-140,-70]
        lat_limits_subtropical=[-40,-20]

        lon_limits_westerlies=[-140,-70]
        lat_limits_westerlies=[-60,-30]

        lat_limits_trade=[-5,5]
        lon_limits_trade=[-35,-20]

        models=list(dict_models['ua'])

        #Models
        subtropical_strength_models=np.empty((len(models),12))
        subtropical_latitude_models=np.empty((len(models),12))

        westerly_strength_models=np.empty((len(models),12))
        westerly_latitude_models=np.empty((len(models),12))

        trade_index_models=np.empty((len(models),12))

        models_app=np.array([])

        #saving the npz
        np.savez_compressed(path_save+'subtropical_jet_strength_models.npz',subtropical_strength_models)
        np.savez_compressed(path_save+'subtropical_jet_latitude_models.npz',subtropical_latitude_models)
        np.savez_compressed(path_save+'westerlies_strength_models.npz',westerly_strength_models)
        np.savez_compressed(path_save+'westerlies_latitude_models.npz',westerly_latitude_models)
        np.savez_compressed(path_save+'trade_winds_models.npz',trade_index_models)
        np.savez_compressed(path_save+'wind_indices_models_N.npz',models_app)

        for p in range(len(models)):

            try: 
                jet_strength_m, jet_latitude_m=subtropical_jet(path_save,'ua',models[p], None,lon_limits_subtropical,\
                                                            lat_limits_subtropical, 20000.0,20000.0)
                
                westerly_strength_m, westerly_latitude_m=westerlies(path_save,'ua',models[p], None,lon_limits_westerlies,\
                                                            lat_limits_westerlies, 85000.0,85000.0)
                
                trade_index_arr_m=tradeWind(path_save,'ua',models[p], None,lon_limits_trade,lat_limits_trade,  85000.0,85000.0)[:,0]

                #---------------------------------------------------------------------------------------------------------

                #reading and saving the existing npz
                subtropical_Sm=np.load(path_save+'subtropical_jet_strength_models.npz',\
                allow_pickle=True)['arr_0']

                subtropical_Lm=np.load(path_save+'subtropical_jet_latitude_models.npz',\
                allow_pickle=True)['arr_0']

                westerlies_Sm=np.load(path_save+'westerlies_strength_models.npz',\
                allow_pickle=True)['arr_0']

                westerlies_Lm=np.load(path_save+'westerlies_latitude_models.npz',\
                allow_pickle=True)['arr_0']

                trades_Sm=np.load(path_save+'trade_winds_models.npz',\
                allow_pickle=True)['arr_0']

                models_wind_calc=np.load(path_save+'wind_indices_models_N.npz',\
                allow_pickle=True)['arr_0']

                #Saving the indices from the models in the arrays
                subtropical_Sm[p,:]=jet_strength_m
                subtropical_Lm[p,:]=jet_latitude_m

                westerlies_Sm[p,:]=westerly_strength_m
                westerlies_Lm[p,:]=westerly_latitude_m

                trades_Sm[p,:]=trade_index_arr_m

                models_wind_calc=np.append(models_wind_calc,models[p])

                np.savez_compressed(path_save+'subtropical_jet_strength_models.npz',subtropical_Sm)
                np.savez_compressed(path_save+'subtropical_jet_latitude_models.npz',subtropical_Lm)
                np.savez_compressed(path_save+'westerlies_strength_models.npz',westerlies_Sm)
                np.savez_compressed(path_save+'westerlies_latitude_models.npz',westerlies_Lm)
                np.savez_compressed(path_save+'trade_winds_models.npz',trades_Sm)
                np.savez_compressed(path_save+'wind_indices_models_N.npz',models_wind_calc)

            except:
                print('Error: Wind indices ', models[p])
    
    elif list_calculation[i]=='Subtropical_highs':
        
        NASH_domains_lat=[20,43]
        NASH_domains_lon=[-70,-16]

        SASH_domains_lat=[-45,0]
        SASH_domains_lon=[-35,-2]

        Pacific_domains_lat=[-50,-15]
        Pacific_domains_lon=[-150,-79]

        fields_domains_lon=lon_limits_F
        fields_domains_lat=lat_limits_F

        models=list(dict_models['psl'])

        #Models
        southAtlan_strength_models=np.empty((len(models),12))
        southAtlan_latitude_models=np.empty((len(models),12))
        southAtlan_longitude_models=np.empty((len(models),12))

        southPaci_strength_models=np.empty((len(models),12))
        southPaci_latitude_models=np.empty((len(models),12))
        southPaci_longitude_models=np.empty((len(models),12))

        nash_strength_models=np.empty((len(models),12))
        nash_latitude_models=np.empty((len(models),12))
        nash_longitude_models=np.empty((len(models),12))

        southAtlan_strength_models_seasonal=np.empty((len(models),4))
        southAtlan_latitude_models_seasonal=np.empty((len(models),4))
        southAtlan_longitude_models_seasonal=np.empty((len(models),4))

        southPaci_strength_models_seasonal=np.empty((len(models),4))
        southPaci_latitude_models_seasonal=np.empty((len(models),4))
        southPaci_longitude_models_seasonal=np.empty((len(models),4))

        nash_strength_models_seasonal=np.empty((len(models),4))
        nash_latitude_models_seasonal=np.empty((len(models),4))
        nash_longitude_models_seasonal=np.empty((len(models),4))

        taylor_diagram_metrics=pd.DataFrame(columns=['Model','corr_DJF','corr_JJA',\
        'corr_MAM','corr_SON','std_DJF','std_JJA','std_MAM','std_SON'])

        taylor_diagram_metrics.to_csv(path_save+'taylorDiagram_metrics_psl.csv')


        models_app=np.array([])

        #saving the npz
        np.savez_compressed(path_save+'southAtlantic_high_strength_models.npz',southAtlan_strength_models)
        np.savez_compressed(path_save+'southAtlantic_high_latitude_models.npz',southAtlan_latitude_models)
        np.savez_compressed(path_save+'southAtlantic_high_longitude_models.npz',southAtlan_longitude_models)

        np.savez_compressed(path_save+'southPacific_high_strength_models.npz',southPaci_strength_models)
        np.savez_compressed(path_save+'southPacific_high_latitude_models.npz',southPaci_latitude_models)
        np.savez_compressed(path_save+'southPacific_high_longitude_models.npz',southPaci_longitude_models)

        np.savez_compressed(path_save+'northAtlantic_high_strength_models.npz',nash_strength_models)
        np.savez_compressed(path_save+'northAtlantic_high_latitude_models.npz',nash_latitude_models)
        np.savez_compressed(path_save+'northAtlantic_high_longitude_models.npz',nash_longitude_models)


        np.savez_compressed(path_save+'southAtlantic_high_strength_seasonal_models.npz',southAtlan_strength_models_seasonal)
        np.savez_compressed(path_save+'southAtlantic_high_latitude_seasonal_models.npz',southAtlan_latitude_models_seasonal)
        np.savez_compressed(path_save+'southAtlantic_high_longitude_seasonal_models.npz',southAtlan_longitude_models_seasonal)

        np.savez_compressed(path_save+'southPacific_high_strength_seasonal_models.npz',southPaci_strength_models_seasonal)
        np.savez_compressed(path_save+'southPacific_high_latitude_seasonal_models.npz',southPaci_latitude_models_seasonal)
        np.savez_compressed(path_save+'southPacific_high_longitude_seasonal_models.npz',southPaci_longitude_models_seasonal)

        np.savez_compressed(path_save+'northAtlantic_high_strength_seasonal_models.npz',nash_strength_models_seasonal)
        np.savez_compressed(path_save+'northAtlantic_high_latitude_seasonal_models.npz',nash_latitude_models_seasonal)
        np.savez_compressed(path_save+'northAtlantic_high_longitude_seasonal_models.npz',nash_longitude_models_seasonal)


        np.savez_compressed(path_save+'subtropicalHighs_models_N.npz',models_app)

        #Reading the reference files 
        var_array_ref=np.load(path_save+'ERA5_slp_fields.npz')['arr_0']
        lat_refe=np.load(path_save+'ERA5_slp_fields_Lat.npz')['arr_0']
        lon_refe=np.load(path_save+'ERA5_slp_fields_Lon.npz')['arr_0']

        for p in range(len(models)):

            try:

                #Monthly
                southAtlantic_strength_m, southAtlantic_lat_m,southAtlantic_lon_m=subtropicalHighs(path_save,'psl',\
                                                                                                models[p],'model',SASH_domains_lon,SASH_domains_lat)

                southPacific_strength_m, southPacific_lat_m,southPacific_lon_m=subtropicalHighs(path_save,'psl',\
                                                                                                models[p], 'model',Pacific_domains_lon,Pacific_domains_lat)

                nash_strength_m, nash_lat_m,nash_lon_m=subtropicalHighs(path_save,'psl',models[p], 'model',\
                                                                        NASH_domains_lon,NASH_domains_lat)

                #Seasonal
                southAtlantic_strength_m_seasonal, southAtlantic_lat_m_seasonal,\
                    southAtlantic_lon_m_seasonal=subtropicalHighs_core_Seasonal(path_save,'psl',models[p],'model',SASH_domains_lon,SASH_domains_lat)

                southPacific_strength_m_seasonal, southPacific_lat_m_seasonal,\
                    southPacific_lon_m_seasonal=subtropicalHighs_core_Seasonal(path_save,'psl',models[p], 'model',Pacific_domains_lon,Pacific_domains_lat)

                nash_strength_m_seasonal, nash_lat_m_seasonal,\
                    nash_lon_m_seasonal=subtropicalHighs_core_Seasonal(path_save,'psl',models[p], 'model',NASH_domains_lon,NASH_domains_lat)
                
                #Seasonal fields 

                psl_array,Lat_list_psl,Lon_list_psl,dx_data_psl, dy_data_psl=var_field_calc(path_save,'psl',models[p],\
                                                                                    fields_domains_lat,fields_domains_lon,None,None,\
                                                                                        None,None,'ERA5','No')
                
                psl_array=psl_array/100

                var_sum=np.sum(psl_array)

                if np.isnan(var_sum)==True :
                    var_array_model=NaNs_interp(psl_array, '3D', 'cubic')
                else:
                    var_array_model=psl_array
                
                #-----------------------------------------------------------------------------------------------------------------------            
                #ERA5 interpolation to the model's gridsize 
                era5_field_interp=interpolation_fields(var_array_ref,lat_refe,lon_refe,Lat_list_psl,Lon_list_psl,dx_data_psl, dy_data_psl)
                #calculating the metrics
                corr_m_o,std_m=taylor_diagram_metrics_def(era5_field_interp,var_array_model)
                #Calculating the bias
                bias_model= var_array_model - era5_field_interp
                #Model's interpolation to a common gridsize
                model_field_interp=interpolation_fields(var_array_model,Lat_list_psl,Lon_list_psl,Lat_common,Lon_common,dx_common, dy_common)
                model_bias_field_interp=interpolation_fields(bias_model,Lat_list_psl,Lon_list_psl,Lat_common,Lon_common,dx_common, dy_common)

                #-----------------------------------------------------------------------------------------------------------------------
                #Saving the npz 

                np.savez_compressed(path_save+models[p]+'_psl_MMM_meanFields.npz',model_field_interp)
                np.savez_compressed(path_save+models[p]+'_psl_MMM_biasFields.npz',model_bias_field_interp)

                #Saving the performance metrics 
                taylor_diagram_metrics_DT=pd.read_csv(path_save+'taylorDiagram_metrics_psl.csv', index_col=[0])

                newRow_metrics=pd.DataFrame({'Model':[models[p]],'corr_DJF': [corr_m_o[0]],\
                'corr_JJA': [corr_m_o[1]],'corr_MAM': [corr_m_o[2]],'corr_SON': [corr_m_o[3]],\
                'std_DJF':[std_m[0]],'std_JJA':[std_m[1]],'std_MAM':[std_m[2]],'std_SON':[std_m[3]]})

                #taylor_diagram_metrics_DT=taylor_diagram_metrics_DT.append(newRow_metrics,\
                #ignore_index=True)
                taylor_diagram_metrics_DT = pd.concat([taylor_diagram_metrics_DT, newRow_metrics], ignore_index=True)

                taylor_diagram_metrics_DT.to_csv(path_save+'taylorDiagram_metrics_psl.csv')

                #----------------------------------------------------------------------------------------------------------------------------------------
                #reading and saving the existing npz
                southAtlan_Sm=np.load(path_save+'southAtlantic_high_strength_models.npz',allow_pickle=True)['arr_0']
                southAtlan_Latm=np.load(path_save+'southAtlantic_high_latitude_models.npz',allow_pickle=True)['arr_0']
                southAtlan_Lonm=np.load(path_save+'southAtlantic_high_longitude_models.npz',allow_pickle=True)['arr_0']

                southPaci_Sm=np.load(path_save+'southPacific_high_strength_models.npz',allow_pickle=True)['arr_0']
                southPaci_Latm=np.load(path_save+'southPacific_high_latitude_models.npz',allow_pickle=True)['arr_0']
                southPaci_Lonm=np.load(path_save+'southPacific_high_longitude_models.npz',allow_pickle=True)['arr_0']

                northAtlan_Sm=np.load(path_save+'northAtlantic_high_strength_models.npz',allow_pickle=True)['arr_0']
                northAtlan_Latm=np.load(path_save+'northAtlantic_high_latitude_models.npz',allow_pickle=True)['arr_0']
                northAtlan_Lonm=np.load(path_save+'northAtlantic_high_longitude_models.npz',allow_pickle=True)['arr_0']

                southAtlan_seasonal_Sm=np.load(path_save+'southAtlantic_high_strength_seasonal_models.npz',allow_pickle=True)['arr_0']
                southAtlan_seasonal_Latm=np.load(path_save+'southAtlantic_high_latitude_seasonal_models.npz',allow_pickle=True)['arr_0']
                southAtlan_seasonal_Lonm=np.load(path_save+'southAtlantic_high_longitude_seasonal_models.npz',allow_pickle=True)['arr_0']

                southPaci_seasonal_Sm=np.load(path_save+'southPacific_high_strength_seasonal_models.npz',allow_pickle=True)['arr_0']
                southPaci_seasonal_Latm=np.load(path_save+'southPacific_high_latitude_seasonal_models.npz',allow_pickle=True)['arr_0']
                southPaci_seasonal_Lonm=np.load(path_save+'southPacific_high_longitude_seasonal_models.npz',allow_pickle=True)['arr_0']

                northAtlan_seasonal_Sm=np.load(path_save+'northAtlantic_high_strength_seasonal_models.npz',allow_pickle=True)['arr_0']
                northAtlan_seasonal_Latm=np.load(path_save+'northAtlantic_high_latitude_seasonal_models.npz',allow_pickle=True)['arr_0']
                northAtlan_seasonal_Lonm=np.load(path_save+'northAtlantic_high_longitude_seasonal_models.npz',allow_pickle=True)['arr_0']

                models_subHighs_calc=np.load(path_save+'subtropicalHighs_models_N.npz',allow_pickle=True)['arr_0']


                #-------------------------------------------------------------------------------------------------------------------------
                southAtlan_Sm[p,:]=southAtlantic_strength_m
                southAtlan_Latm[p,:]=southAtlantic_lat_m
                southAtlan_Lonm[p,:]=southAtlantic_lon_m

                southPaci_Sm[p,:]=southPacific_strength_m
                southPaci_Latm[p,:]=southPacific_lat_m
                southPaci_Lonm[p,:]=southPacific_lon_m

                northAtlan_Sm[p,:]=nash_strength_m
                northAtlan_Latm[p,:]=nash_lat_m
                northAtlan_Lonm[p,:]=nash_lon_m

                southAtlan_seasonal_Sm[p,:]=southAtlantic_strength_m_seasonal
                southAtlan_seasonal_Latm[p,:]=southAtlantic_lat_m_seasonal
                southAtlan_seasonal_Lonm[p,:]=southAtlantic_lon_m_seasonal

                southPaci_seasonal_Sm[p,:]=southPacific_strength_m_seasonal
                southPaci_seasonal_Latm[p,:]=southPacific_lat_m_seasonal
                southPaci_seasonal_Lonm[p,:]=southPacific_lon_m_seasonal

                northAtlan_seasonal_Sm[p,:]=nash_strength_m_seasonal
                northAtlan_seasonal_Latm[p,:]=nash_lat_m_seasonal
                northAtlan_seasonal_Lonm[p,:]=nash_lon_m_seasonal

                #-----------------------------------------------------------------------------------------------------------------------------------------
                models_subHighs_calc=np.append(models_subHighs_calc,models[p])

                np.savez_compressed(path_save+'southAtlantic_high_strength_models.npz',southAtlan_Sm)
                np.savez_compressed(path_save+'southAtlantic_high_latitude_models.npz',southAtlan_Latm)
                np.savez_compressed(path_save+'southAtlantic_high_longitude_models.npz',southAtlan_Lonm)

                np.savez_compressed(path_save+'southPacific_high_strength_models.npz',southPaci_Sm)
                np.savez_compressed(path_save+'southPacific_high_latitude_models.npz',southPaci_Latm)
                np.savez_compressed(path_save+'southPacific_high_longitude_models.npz',southPaci_Lonm)

                np.savez_compressed(path_save+'northAtlantic_high_strength_models.npz',northAtlan_Sm)
                np.savez_compressed(path_save+'northAtlantic_high_latitude_models.npz',northAtlan_Latm)
                np.savez_compressed(path_save+'northAtlantic_high_longitude_models.npz',northAtlan_Lonm)

                np.savez_compressed(path_save+'southAtlantic_high_strength_seasonal_models.npz',southAtlan_seasonal_Sm)
                np.savez_compressed(path_save+'southAtlantic_high_latitude_seasonal_models.npz',southAtlan_seasonal_Latm)
                np.savez_compressed(path_save+'southAtlantic_high_longitude_seasonal_models.npz',southAtlan_seasonal_Lonm)

                np.savez_compressed(path_save+'southPacific_high_strength_seasonal_models.npz',southPaci_seasonal_Sm)
                np.savez_compressed(path_save+'southPacific_high_latitude_seasonal_models.npz',southPaci_seasonal_Latm)
                np.savez_compressed(path_save+'southPacific_high_longitude_seasonal_models.npz',southPaci_seasonal_Lonm)

                np.savez_compressed(path_save+'northAtlantic_high_strength_seasonal_models.npz',northAtlan_seasonal_Sm)
                np.savez_compressed(path_save+'northAtlantic_high_latitude_seasonal_models.npz',northAtlan_seasonal_Latm)
                np.savez_compressed(path_save+'northAtlantic_high_longitude_seasonal_models.npz',northAtlan_seasonal_Lonm)

                np.savez_compressed(path_save+'subtropicalHighs_models_N.npz',models_subHighs_calc)
            
            except:
                print('Error Subtropical Highs ', models[p])
           

    elif list_calculation[i]=='Bolivian_high':

        lon_limits_bol=[-70,-60]
        lat_limits_bol=[-25,-15]

        models=list(dict_models['zg'])

        bolhigh_models=np.empty((len(models),12))

        models_app=np.array([])

        np.savez_compressed(path_save+'Bolivian_High_index_monthly.npz',bolhigh_models)

        np.savez_compressed(path_save+'Bolivian_High_models_N.npz',models_app)

        for p in range(len(models)):

            try:
            
                var_array_model=Bolivian_High(path_save,'zg',models[p], 'model',lon_limits_bol,lat_limits_bol, 20000,20000)[:,0]


                BH_index=np.load(path_save+'Bolivian_High_index_monthly.npz',allow_pickle=True)['arr_0']
                models_BH_calc=np.load(path_save+'Bolivian_High_models_N.npz',allow_pickle=True)['arr_0']

                BH_index[p,:]=var_array_model
                models_BH_calc=np.append(models_BH_calc,models[p])

                np.savez_compressed(path_save+'Bolivian_High_index_monthly.npz',BH_index)

                np.savez_compressed(path_save+'Bolivian_High_models_N.npz',models_BH_calc)
            except:
                print('Error: Bolivian High ',models[p])
        
    
    elif list_calculation[i]=='Precipitation':

        lon_limits_pr=lon_limits_F
        lat_limits_pr=lat_limits_F

        models=list(dict_models['pr'])

        #---------------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------
        #creating and saving the matrices of the fields 
        models_app=np.array([])

        taylor_diagram_metrics=pd.DataFrame(columns=['Model','corr_DJF','corr_JJA',\
        'corr_MAM','corr_SON','std_DJF','std_JJA','std_MAM','std_SON'])

        taylor_diagram_metrics.to_csv(path_save+'taylorDiagram_metrics_pr.csv')
        np.savez_compressed(path_save+'pr_fields_models_N.npz',models_app)

        #Reading the reference files 
        var_array_ref=np.load(path_save+'ERA5_mtpr_fields.npz')['arr_0']
        lat_refe=np.load(path_save+'ERA5_mtpr_fields_Lat.npz')['arr_0']
        lon_refe=np.load(path_save+'ERA5_mtpr_fields_Lon.npz')['arr_0']


        for p in range(len(models)):

            try:

                pr_array,Lat_list_pr,Lon_list_pr,dx_data_pr, dy_data_pr=var_field_calc(path_save,'pr',models[p],\
                                                                                    lat_limits_pr,lon_limits_pr,None,None,\
                                                                                        None,None,'ERA5','No')
                
                pr_array=pr_array*86400

                var_sum=np.sum(pr_array)

                if np.isnan(var_sum)==True :
                    var_array_model=NaNs_interp(pr_array, '3D', 'cubic')
                else:
                    var_array_model=pr_array
                
                #-----------------------------------------------------------------------------------------------------------------------            
                #ERA5 interpolation to the model's gridsize 
                era5_field_interp=interpolation_fields(var_array_ref,lat_refe,lon_refe,Lat_list_pr,Lon_list_pr,dx_data_pr, dy_data_pr)
                #calculating the metrics
                corr_m_o,std_m=taylor_diagram_metrics_def(era5_field_interp,var_array_model)
                #Calculating the bias
                bias_model= var_array_model - era5_field_interp
                #Model's interpolation to a common gridsize
                model_field_interp=interpolation_fields(var_array_model,Lat_list_pr,Lon_list_pr,Lat_common,Lon_common,dx_common, dy_common)
                model_bias_field_interp=interpolation_fields(bias_model,Lat_list_pr,Lon_list_pr,Lat_common,Lon_common,dx_common, dy_common)

                #-----------------------------------------------------------------------------------------------------------------------
                #Saving the npz 
                
                models_pr_calc=np.load(path_save+'pr_fields_models_N.npz',allow_pickle=True)['arr_0']

            
                models_pr_calc=np.append(models_pr_calc,models[p])

                np.savez_compressed(path_save+models[p]+'_pr_MMM_meanFields.npz',model_field_interp)
                np.savez_compressed(path_save+models[p]+'_pr_MMM_biasFields.npz',model_bias_field_interp)
                np.savez_compressed(path_save+'pr_fields_models_N.npz',models_pr_calc)

                #Saving the performance metrics 
                taylor_diagram_metrics_DT=pd.read_csv(path_save+'taylorDiagram_metrics_pr.csv', index_col=[0])

                newRow_metrics=pd.DataFrame({'Model':[models[p]],'corr_DJF': [corr_m_o[0]],\
                'corr_JJA': [corr_m_o[1]],'corr_MAM': [corr_m_o[2]],'corr_SON': [corr_m_o[3]],\
                'std_DJF':[std_m[0]],'std_JJA':[std_m[1]],'std_MAM':[std_m[2]],'std_SON':[std_m[3]]})

                #taylor_diagram_metrics_DT=taylor_diagram_metrics_DT.append(newRow_metrics,\
                #ignore_index=True)
                taylor_diagram_metrics_DT = pd.concat([taylor_diagram_metrics_DT, newRow_metrics], ignore_index=True)


                taylor_diagram_metrics_DT.to_csv(path_save+'taylorDiagram_metrics_pr.csv')
            
            except:
                print('Error Precipitation ',models[p])
    
    elif list_calculation[i]=='SST':

        lon_limits_tos=lon_limits_F
        lat_limits_tos=lat_limits_F

        models=list(dict_models['tos'])

        #---------------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------
        #creating and saving the matrices of the fields 
        models_app=np.array([])

        taylor_diagram_metrics=pd.DataFrame(columns=['Model','corr_DJF','corr_JJA',\
        'corr_MAM','corr_SON','std_DJF','std_JJA','std_MAM','std_SON'])

        taylor_diagram_metrics.to_csv(path_save+'taylorDiagram_metrics_tos.csv')
        np.savez_compressed(path_save+'tos_fields_models_N.npz',models_app)

        #Reading the reference files 
        var_array_ref_nans=np.load(path_save+'ERA5_sst_fields.npz')['arr_0']
        lat_refe=np.load(path_save+'ERA5_sst_fields_Lat.npz')['arr_0']
        lon_refe=np.load(path_save+'ERA5_sst_fields_Lon.npz')['arr_0']

        var_array_ref=NaNs_interp(var_array_ref_nans,'3D', 'nearest')

        for p in range(len(models)):

            try:

                tos_array,Lat_list_tos,Lon_list_tos,dx_data_tos, dy_data_tos=var_field_calc(path_save,'tos',models[p],\
                                                                                    lat_limits_tos,lon_limits_tos,None,None,\
                                                                                        None,None,'ERA5','No')
                
                if len(np.where(tos_array<0)[0])>0:
                    var_array_model=tos_array
                else:
                    var_array_model=tos_array-273.15
                
                #-----------------------------------------------------------------------------------------------------------------------            
                #ERA5 interpolation to the model's gridsize 
                era5_field_interp=interpolation_fields(var_array_ref,lat_refe,lon_refe,Lat_list_tos,Lon_list_tos,dx_data_tos, dy_data_tos)
                #calculating the metrics
                corr_m_o,std_m=taylor_diagram_metrics_def(era5_field_interp,var_array_model)
                #Calculating the bias
                bias_model= var_array_model - era5_field_interp
                #Model's interpolation to a common gridsize
                var_array_model=NaNs_interp(var_array_model,'3D', 'nearest')
                bias_model=NaNs_interp(bias_model,'3D', 'nearest')
                model_field_interp=interpolation_fields(var_array_model,Lat_list_tos,Lon_list_tos,Lat_common_tos,Lon_common_tos,dx_common_tos, dy_common_tos)
                model_bias_field_interp=interpolation_fields(bias_model,Lat_list_tos,Lon_list_tos,Lat_common_tos,Lon_common_tos,dx_common_tos, dy_common_tos)

                #-----------------------------------------------------------------------------------------------------------------------
                #Saving the npz 
            
                models_tos_calc=np.load(path_save+'tos_fields_models_N.npz',allow_pickle=True)['arr_0']

                models_tos_calc=np.append(models_tos_calc,models[p])

                np.savez_compressed(path_save+models[p]+'_tos_MMM_meanFields.npz',model_field_interp)
                np.savez_compressed(path_save+models[p]+'_tos_MMM_biasFields.npz',model_bias_field_interp)
                np.savez_compressed(path_save+'tos_fields_models_N.npz',models_tos_calc)

                #Saving the performance metrics 
                taylor_diagram_metrics_DT=pd.read_csv(path_save+'taylorDiagram_metrics_tos.csv', index_col=[0])

                newRow_metrics=pd.DataFrame({'Model':[models[p]],'corr_DJF': [corr_m_o[0]],\
                'corr_JJA': [corr_m_o[1]],'corr_MAM': [corr_m_o[2]],'corr_SON': [corr_m_o[3]],\
                'std_DJF':[std_m[0]],'std_JJA':[std_m[1]],'std_MAM':[std_m[2]],'std_SON':[std_m[3]]})

                #taylor_diagram_metrics_DT=taylor_diagram_metrics_DT.append(newRow_metrics,\
                #ignore_index=True)
                taylor_diagram_metrics_DT = pd.concat([taylor_diagram_metrics_DT, newRow_metrics], ignore_index=True)

                taylor_diagram_metrics_DT.to_csv(path_save+'taylorDiagram_metrics_tos.csv')
            
            except:
                print('Error SST ',models[p])

    elif list_calculation[i]=='wind_200':

        lon_limits_w200=lon_limits_F
        lat_limits_w200=lat_limits_F

        p_level_interest=20000.0

        models_ua=list(dict_models['ua'])
        models_va=list(dict_models['va'])

        models=np.intersect1d(models_ua, models_va)

        #---------------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------
        #creating and saving the matrices of the fields 
        models_app=np.array([])

        taylor_diagram_metrics=pd.DataFrame(columns=['Model','corr_DJF','corr_JJA',\
        'corr_MAM','corr_SON','std_DJF','std_JJA','std_MAM','std_SON'])

        taylor_diagram_metrics.to_csv(path_save+'taylorDiagram_metrics_wind200.csv')
        np.savez_compressed(path_save+'wind200_fields_models_N.npz',models_app)

        #Reading the reference files 
        var_array_ref=np.load(path_save+'ERA5_W200_fields.npz')['arr_0']
        lat_refe=np.load(path_save+'ERA5_W200_fields_Lat.npz')['arr_0']
        lon_refe=np.load(path_save+'ERA5_W200_fields_Lon.npz')['arr_0']

        for p in range(len(models)):

            try:

                ua200_array,va200_array,mag200_array,Lat_list_w200,Lon_list_w200,\
                    dx_data_w200, dy_data_w200=wind_field_calc(path_save,'ua','va',models[p],\
                                                    lat_limits_w200,lon_limits_w200,None,None,\
                                                        p_level_interest,p_level_interest,'ERA5')
                
                ############################################################################
                #Evaluating if there are any NaNs values in the matrices
                u_sum=np.sum(ua200_array)
                v_sum=np.sum(va200_array)
                mag_sum=np.sum(mag200_array)

                if np.isnan(u_sum)==True or np.isnan(v_sum)==True or np.isnan(mag_sum)==True:
                    u_array_model=NaNs_interp(ua200_array, '3D', 'cubic')
                    v_array_model=NaNs_interp(va200_array, '3D', 'cubic')
                    mag_array_model=NaNs_interp(mag200_array, '3D', 'cubic')
                else:
                    u_array_model=ua200_array
                    v_array_model=va200_array
                    mag_array_model=mag200_array
                
                #-----------------------------------------------------------------------------------------------------------------------            
                #ERA5 interpolation to the model's gridsize 
                era5_field_interp=interpolation_fields(var_array_ref,lat_refe,lon_refe,Lat_list_w200,Lon_list_w200,dx_data_w200, dy_data_w200)
                #calculating the metrics
                corr_m_o,std_m=taylor_diagram_metrics_def(era5_field_interp,mag_array_model)
                #Calculating the bias
                bias_model= mag_array_model - era5_field_interp
                #Model's interpolation to a common gridsize
                model_field_interp=interpolation_fields(mag_array_model,Lat_list_w200,Lon_list_w200,Lat_common,Lon_common,dx_common, dy_common)
                model_ua_field_interp=interpolation_fields(u_array_model,Lat_list_w200,Lon_list_w200,Lat_common,Lon_common,dx_common, dy_common)
                model_va_field_interp=interpolation_fields(v_array_model,Lat_list_w200,Lon_list_w200,Lat_common,Lon_common,dx_common, dy_common)
                model_bias_field_interp=interpolation_fields(bias_model,Lat_list_w200,Lon_list_w200,Lat_common,Lon_common,dx_common, dy_common)

                #-----------------------------------------------------------------------------------------------------------------------
                #Saving the npz 
                models_wind200_calc=np.load(path_save+'wind200_fields_models_N.npz',allow_pickle=True)['arr_0']

                models_wind200_calc=np.append(models_wind200_calc,models[p])

                np.savez_compressed(path_save+models[p]+'_mag200_MMM_meanFields.npz',model_field_interp)
                np.savez_compressed(path_save+models[p]+'_ua200_MMM_meanFields.npz',model_ua_field_interp)
                np.savez_compressed(path_save+models[p]+'_va200_MMM_meanFields.npz',model_va_field_interp)
                np.savez_compressed(path_save+models[p]+'_mag200_MMM_biasFields.npz',model_bias_field_interp)
                np.savez_compressed(path_save+'wind200_fields_models_N.npz',models_wind200_calc)

                #Saving the performance metrics 
                taylor_diagram_metrics_DT=pd.read_csv(path_save+'taylorDiagram_metrics_wind200.csv', index_col=[0])

                newRow_metrics=pd.DataFrame({'Model':[models[p]],'corr_DJF': [corr_m_o[0]],\
                'corr_JJA': [corr_m_o[1]],'corr_MAM': [corr_m_o[2]],'corr_SON': [corr_m_o[3]],\
                'std_DJF':[std_m[0]],'std_JJA':[std_m[1]],'std_MAM':[std_m[2]],'std_SON':[std_m[3]]})

                #taylor_diagram_metrics_DT=taylor_diagram_metrics_DT.append(newRow_metrics,\
                #ignore_index=True)
                taylor_diagram_metrics_DT = pd.concat([taylor_diagram_metrics_DT, newRow_metrics], ignore_index=True)

                taylor_diagram_metrics_DT.to_csv(path_save+'taylorDiagram_metrics_wind200.csv')
            
            except:
                print('Error W200 ',models[p])

    elif list_calculation[i]=='wind_850':

        lon_limits_w850=lon_limits_F
        lat_limits_w850=lat_limits_F

        p_level_interest=85000.0

        models_ua=list(dict_models['ua'])
        models_va=list(dict_models['va'])

        models=np.intersect1d(models_ua, models_va)

        #---------------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------
        #creating and saving the matrices of the fields 
        models_app=np.array([])

        taylor_diagram_metrics=pd.DataFrame(columns=['Model','corr_DJF','corr_JJA',\
        'corr_MAM','corr_SON','std_DJF','std_JJA','std_MAM','std_SON'])

        taylor_diagram_metrics.to_csv(path_save+'taylorDiagram_metrics_wind850.csv')
        np.savez_compressed(path_save+'wind850_fields_models_N.npz',models_app)

        var_array_ref=np.load(path_save+'ERA5_W850_fields.npz')['arr_0']
        lat_refe=np.load(path_save+'ERA5_W850_fields_Lat.npz')['arr_0']
        lon_refe=np.load(path_save+'ERA5_W850_fields_Lon.npz')['arr_0']

        for p in range(len(models)):

            try:

                ua850_array,va850_array,mag850_array,Lat_list_w850,Lon_list_w850,\
                    dx_data_w850, dy_data_w850=wind_field_calc(path_save,'ua','va',models[p],\
                                                    lat_limits_w850,lon_limits_w850,None,None,\
                                                        p_level_interest,p_level_interest,'ERA5')
                
                ############################################################################
                #Evaluating if there are any NaNs values in the matrices
                u_sum=np.sum(ua850_array)
                v_sum=np.sum(va850_array)
                mag_sum=np.sum(mag850_array)

                print ('  ' + list_calculation[i] + ': ' + models[p] +               \
                  ' checking for NaNs')

                if np.isnan(u_sum)==True or np.isnan(v_sum)==True or np.isnan(mag_sum)==True:
                    u_array_model=NaNs_interp(ua850_array, '3D', 'cubic')
                    v_array_model=NaNs_interp(va850_array, '3D', 'cubic')
                    mag_array_model=NaNs_interp(mag850_array, '3D', 'cubic')
                else:
                    u_array_model=ua850_array
                    v_array_model=va850_array
                    mag_array_model=mag850_array

                print ('  ' + list_calculation[i] + ': ' + models[p] +               \
                  ' NaNs re-interpolated')
                
                #-----------------------------------------------------------------------------------------------------------------------            
                #ERA5 interpolation to the model's gridsize 
                era5_field_interp=interpolation_fields(var_array_ref,lat_refe,lon_refe,Lat_list_w850,Lon_list_w850,dx_data_w850, dy_data_w850)
                print ('  ' + list_calculation[i] + ': ERA5 interpolated to ' +      \
                  models[p])
                #calculating the metrics
                corr_m_o,std_m=taylor_diagram_metrics_def(era5_field_interp,mag_array_model)
                print ('  ' + list_calculation[i] + ': ERA5 - ' + models[p] + 'metrics')
                #Calculating the bias
                bias_model= mag_array_model - era5_field_interp
                #Model's interpolation to a common gridsize
                print ('  ' + list_calculation[i] + ': ERA5 - ' + models[p] +        \
                  ' common grid _______')
                model_field_interp=interpolation_fields(mag_array_model,Lat_list_w850,Lon_list_w850,Lat_common,Lon_common,dx_common, dy_common)
                print ('   mag_array_model ')
                model_ua_field_interp=interpolation_fields(u_array_model,Lat_list_w850,Lon_list_w850,Lat_common,Lon_common,dx_common, dy_common)
                print ('   u_array_model ')
                model_va_field_interp=interpolation_fields(v_array_model,Lat_list_w850,Lon_list_w850,Lat_common,Lon_common,dx_common, dy_common)
                print ('   v_array_model ')
                model_bias_field_interp=interpolation_fields(bias_model,Lat_list_w850,Lon_list_w850,Lat_common,Lon_common,dx_common, dy_common)
                print ('   bias_model ')

                #-----------------------------------------------------------------------------------------------------------------------
                #Saving the npz 
                print ('  Saving outputs')
                models_wind850_calc=np.load(path_save+'wind850_fields_models_N.npz',allow_pickle=True)['arr_0']
                print ("    load: '" + path_save+"wind850_fields_models_N.npz'")

                models_wind850_calc=np.append(models_wind850_calc,models[p])

                np.savez_compressed(path_save+models[p]+'_mag850_MMM_meanFields.npz',model_field_interp)
                print ("    saved: '" + path_save+models[p]+"_mag850_MMM_meanFields.npz'")
                np.savez_compressed(path_save+models[p]+'_ua850_MMM_meanFields.npz',model_ua_field_interp)
                print ("    saved: '" + path_save+models[p]+"_ua850_MMM_meanFields.npz'")
                np.savez_compressed(path_save+models[p]+'_va850_MMM_meanFields.npz',model_va_field_interp)
                print ("    saved: '" + path_save+models[p]+"_va850_MMM_meanFields.npz'")
                np.savez_compressed(path_save+models[p]+'_mag850_MMM_biasFields.npz',model_bias_field_interp)
                print ("    saved: '" + path_save+models[p]+"_mag850_MMM_meanFields.npz'")
                np.savez_compressed(path_save+'wind850_fields_models_N.npz',models_wind850_calc)
                print ("    saved: '" + path_save+"wind850_fields_models_N.npz'")

                #Saving the performance metrics 
                print ('  saving performance metrics ...')
                taylor_diagram_metrics_DT=pd.read_csv(path_save+'taylorDiagram_metrics_wind850.csv', index_col=[0])
                print ("    read: '" + path_save+"taylorDiagram_metrics_wind850.csv'")

                newRow_metrics=pd.DataFrame({'Model':[models[p]],'corr_DJF': [corr_m_o[0]],\
                'corr_JJA': [corr_m_o[1]],'corr_MAM': [corr_m_o[2]],'corr_SON': [corr_m_o[3]],\
                'std_DJF':[std_m[0]],'std_JJA':[std_m[1]],'std_MAM':[std_m[2]],'std_SON':[std_m[3]]})
                print ("    filling taylor")

                #taylor_diagram_metrics_DT=taylor_diagram_metrics_DT.append(newRow_metrics,\
                #ignore_index=True)
                taylor_diagram_metrics_DT = pd.concat([taylor_diagram_metrics_DT, newRow_metrics], ignore_index=True)
                print ("    writting taylor into csv")

                taylor_diagram_metrics_DT.to_csv(path_save+'taylorDiagram_metrics_wind850.csv')
                print ("    written taylor into csv")
            
            except:
                print('Error W850 ',models[p])
    
    elif list_calculation[i]=='Regional_cells':

        #Hadley cell 
        lat_limits_h=lat_limits_Had
        lon_limits_h=lon_limits_Had
        #Walker cell
        lat_limits_w=lat_limits_Wal
        lon_limits_w=lon_limits_Wal

        lat_had_common=np.flip(np.arange(lat_limits_h[0],lat_limits_h[1],dy_common))
        lon_wal_common=np.arange(lon_limits_w[0],lon_limits_w[1],dx_common)

        shape_interp_wal=np.empty((4,lon_wal_common.shape[0],len(p_level_common)))
        shape_interp_had=np.empty((4,lat_had_common.shape[0],len(p_level_common)))


        p_level_interest_top=10000.0
        p_level_interest_bottom=100000.0

        models_ua=list(dict_models['ua'])
        models_va=list(dict_models['va'])

        models=np.intersect1d(models_ua, models_va)

        #--------------------------------------------------------------------------------------------------------
        #Creating the files to save the information 
        taylor_diagram_metrics_h=pd.DataFrame(columns=['Model','corr_DJF','corr_JJA',\
        'corr_MAM','corr_SON','std_DJF','std_JJA','std_MAM','std_SON'])
        taylor_diagram_metrics_h.to_csv(path_save+'taylorDiagram_metrics_hadleycell.csv')

        #Walker
        taylor_diagram_metrics_w=pd.DataFrame(columns=['Model','corr_DJF','corr_JJA',\
        'corr_MAM','corr_SON','std_DJF','std_JJA','std_MAM','std_SON'])
        taylor_diagram_metrics_w.to_csv(path_save+'taylorDiagram_metrics_walkercell.csv')

        models_app=np.array([])
        np.savez_compressed(path_save+'regionalcells_models_N.npz',models_app)

        #Reading the information of the reference 
        era_hadley_cell=np.load(path_save+'ERA5_HadCell_fields.npz')['arr_0']
        era5_walker_cell=np.load(path_save+'ERA5_WalCell_fields.npz')['arr_0']
        p_level_ref=np.load(path_save+'ERA5_cells_fields_plevel.npz')['arr_0']
        lat_had_ref=np.load(path_save+'ERA5_HadCell_fields_Lat.npz')['arr_0']
        lon_wal_ref=np.load(path_save+'ERA5_WalCell_fields_Lon.npz')['arr_0']
        p_level_ref=np.round(p_level_ref*100,1)


        for p in range(len(models)):

            try:

                hadley_stream_m, std_h_m, lat_had_m,  walker_stream_m,std_w_m,lon_wal_m, p_level_m, dx_h_m, dx_w_m=regional_cells(path_save,\
                        'ua','va',models[p],lat_limits_h,lon_limits_h,lat_limits_w,lon_limits_w,None,None,p_level_interest_bottom,p_level_interest_top,\
                            'model')
                #----------------------------------------------------------------------------------------------------------
                #Interpolation 
                p_level_m=np.round(p_level_m*100,1)

                #1. ERA5 to model's grid size 
                matrix_interpolated_h_ref, matrix_interpolated_w_ref=interpolation_cells(era_hadley_cell,\
                era5_walker_cell, dx_h_m, dx_w_m, p_level_m, lat_had_m,lon_wal_m,p_level_ref,lat_had_ref, lon_wal_ref, hadley_stream_m, walker_stream_m)
                
                #calculating the metrics
                corr_m_h,std_m_h=taylor_diagram_metrics_def(matrix_interpolated_h_ref,hadley_stream_m)
                corr_m_w,std_m_w=taylor_diagram_metrics_def(matrix_interpolated_w_ref,walker_stream_m)
                #Calculating the bias
                bias_model_h= hadley_stream_m - matrix_interpolated_h_ref
                bias_model_w= walker_stream_m - matrix_interpolated_w_ref

                #2. Model's to the common grid size 
                matrix_interpolated_h_model, matrix_interpolated_w_model=interpolation_cells(hadley_stream_m,\
                walker_stream_m, dx_common, dx_common, p_level_common, lat_had_common,lon_wal_common,p_level_m,lat_had_m, lon_wal_m, \
                    shape_interp_had, shape_interp_wal)
                
                matrix_interpolated_h_model_bias, matrix_interpolated_w_model_bias=interpolation_cells(bias_model_h,\
                bias_model_w, dx_common, dx_common, p_level_common, lat_had_common,lon_wal_common,p_level_m,lat_had_m, lon_wal_m, \
                    shape_interp_had, shape_interp_wal)

                #------------------------------------------------------------------------------------------------------------------
                #------------------------------------------------------------------------------------------------------------------
                #Saving the information 

                #1. Metrics
                #hadley
                taylor_diagram_metrics_DT_H=pd.read_csv(path_save+'taylorDiagram_metrics_hadleycell.csv', index_col=[0])

                newRow_metrics_H=pd.DataFrame({'Model':[models[p]],'corr_DJF': [corr_m_h[0]],\
                'corr_JJA': [corr_m_h[1]],'corr_MAM': [corr_m_h[2]],'corr_SON': [corr_m_h[3]],\
                'std_DJF':[std_m_h[0]],'std_JJA':[std_m_h[1]],'std_MAM':[std_m_h[2]],'std_SON':[std_m_h[3]]})

                taylor_diagram_metrics_DT_H = pd.concat([taylor_diagram_metrics_DT_H, newRow_metrics_H], ignore_index=True)

                taylor_diagram_metrics_DT_H.to_csv(path_save+'taylorDiagram_metrics_hadleycell.csv')

                #walker
                taylor_diagram_metrics_DT_W=pd.read_csv(path_save+'taylorDiagram_metrics_walkercell.csv', index_col=[0])

                newRow_metrics_W=pd.DataFrame({'Model':[models[p]],'corr_DJF': [corr_m_w[0]],\
                'corr_JJA': [corr_m_w[1]],'corr_MAM': [corr_m_w[2]],'corr_SON': [corr_m_w[3]],\
                'std_DJF':[std_m_w[0]],'std_JJA':[std_m_w[1]],'std_MAM':[std_m_w[2]],'std_SON':[std_m_w[3]]})

                taylor_diagram_metrics_DT_W = pd.concat([taylor_diagram_metrics_DT_W, newRow_metrics_W], ignore_index=True)

                taylor_diagram_metrics_DT_W.to_csv(path_save+'taylorDiagram_metrics_walkercell.csv')

                ##2. Ensamble matrix
                np.savez_compressed(path_save+models[p]+'_regCirc_hadleycell.npz',matrix_interpolated_h_model)
                np.savez_compressed(path_save+models[p]+'_regCirc_walkercell.npz',matrix_interpolated_w_model)
                np.savez_compressed(path_save+models[p]+'_regCirc_hadleycell_bias.npz',matrix_interpolated_h_model_bias)
                np.savez_compressed(path_save+models[p]+'_regCirc_walkercell_bias.npz',matrix_interpolated_w_model_bias)

                models_regce_calc=np.load(path_save+'regionalcells_models_N.npz',allow_pickle=True)['arr_0']

                models_regce_calc=np.append(models_regce_calc,models[p])
                np.savez_compressed(path_save+'regionalcells_models_N.npz',models_regce_calc)
            
            except:
                print('Error RegCell ', models[p])
        

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

        models_ua=list(dict_models['ua'])
        models_va=list(dict_models['va'])
        models_hus=list(dict_models['hus'])

        models_wind=np.intersect1d(models_ua, models_va)
        models=np.intersect1d(models_wind, models_hus)

        #------------------------------------------------------------------------------------------------
        #Reading the reference information
        VIMF_ref_northern=np.load(path_save+'VIMF_ERA5_northern_Boundary.npz')['arr_0']
        VIMF_ref_southern=np.load(path_save+'VIMF_ERA5_southern_Boundary.npz')['arr_0']
        VIMF_ref_western=np.load(path_save+'VIMF_ERA5_western_Boundary.npz')['arr_0']
        VIMF_ref_eastern=np.load(path_save+'VIMF_ERA5_eastern_Boundary.npz')['arr_0']

        models_app=np.array([])
        np.savez_compressed(path_save+'VIMF_models_N.npz',models_app)

        #Creating the empty arrays to save the models information
        northern_boundary_models=np.empty((len(models),4,VIMF_ref_northern.shape[1]))
        southern_boundary_models=np.empty((len(models),4,VIMF_ref_southern.shape[1]))
        western_boundary_models=np.empty((len(models),4,VIMF_ref_western.shape[1]))
        eastern_boundary_models=np.empty((len(models),4,VIMF_ref_eastern.shape[1]))

        np.savez_compressed(path_save+'VIMF_CMIP6_northern_Boundary.npz',northern_boundary_models)
        np.savez_compressed(path_save+'VIMF_CMIP6_southern_Boundary.npz',southern_boundary_models)
        np.savez_compressed(path_save+'VIMF_CMIP6_western_Boundary.npz',western_boundary_models)
        np.savez_compressed(path_save+'VIMF_CMIP6_eastern_Boundary.npz',eastern_boundary_models)

        for p in range(len(models)):

            try:

                VIMF_model,Lat_model,Lon_model,dx_model,\
                dy_model=VIMF_calc(path_save, 'ua', 'va', 'hus', models[p],lat_limits,lon_limits,None, None,\
                p_level_interest_lower,p_level_interest_upper,'model')

                #-----------------------------------------------------------------------------
                #Interpolating to the common grid size 
                VIMF_model_interpolated=interpolation_fields(VIMF_model,Lat_model,Lon_model,Lat_common_b,Lon_common_b,dx_common,dy_common)

                ############################################################################
                #Obtaining the series of each boundary
                #north
                lat_loc_m_n=np.where(np.logical_and(Lat_common_b>=north_boundaries_lat[0], \
                Lat_common_b<=north_boundaries_lat[1]))
                lon_loc_m_n=np.where(np.logical_and(Lon_common_b>=north_boundaries_lon[0], \
                Lon_common_b<=north_boundaries_lon[1]))
                VIMF_model_n=VIMF_model_interpolated[:,lat_loc_m_n[0][0]:lat_loc_m_n[0][-1]+1,\
                lon_loc_m_n[0][0]:lon_loc_m_n[0][-1]+1]
                VIMF_model_northern=np.nanmean(VIMF_model_n,axis=1)
                #south
                lat_loc_m_s=np.where(np.logical_and(Lat_common_b>=south_boundaries_lat[0], \
                Lat_common_b<=south_boundaries_lat[1]))
                lon_loc_m_s=np.where(np.logical_and(Lon_common_b>=south_boundaries_lon[0], \
                Lon_common_b<=south_boundaries_lon[1]))
                VIMF_model_s=VIMF_model_interpolated[:,lat_loc_m_s[0][0]:lat_loc_m_s[0][-1]+1,\
                lon_loc_m_s[0][0]:lon_loc_m_s[0][-1]+1]
                VIMF_model_southern=np.nanmean(VIMF_model_s,axis=1)
                #west
                lat_loc_m_w=np.where(np.logical_and(Lat_common_b>=west_boundaries_lat[0],\
                Lat_common_b<=west_boundaries_lat[1]))
                lon_loc_m_w=np.where(np.logical_and(Lon_common_b>=west_boundaries_lon[0],\
                Lon_common_b<=west_boundaries_lon[1]))
                VIMF_model_w=VIMF_model_interpolated[:,lat_loc_m_w[0][0]:lat_loc_m_w[0][-1]+1,\
                lon_loc_m_w[0][0]:lon_loc_m_w[0][-1]+1]
                VIMF_model_western=np.nanmean(VIMF_model_w,axis=2)
                #east
                lat_loc_m_e=np.where(np.logical_and(Lat_common_b>=east_boundaries_lat[0],\
                Lat_common_b<=east_boundaries_lat[1]))
                lon_loc_m_e=np.where(np.logical_and(Lon_common_b>=east_boundaries_lon[0],\
                Lon_common_b<=east_boundaries_lon[1]))
                VIMF_model_e=VIMF_model_interpolated[:,lat_loc_m_e[0][0]:lat_loc_m_e[0][-1]+1,\
                lon_loc_m_e[0][0]:lon_loc_m_e[0][-1]+1]
                VIMF_model_eastern=np.nanmean(VIMF_model_e,axis=2)


                ############################################################################
                ############################################################################
                #SAVING THE INFORMATION

                north_m_arr=np.load(path_save+'VIMF_CMIP6_northern_Boundary.npz',\
                allow_pickle=True)['arr_0']
                south_m_arr=np.load(path_save+'VIMF_CMIP6_southern_Boundary.npz',\
                allow_pickle=True)['arr_0']
                west_m_arr=np.load(path_save+'VIMF_CMIP6_western_Boundary.npz',\
                allow_pickle=True)['arr_0']
                east_m_arr=np.load(path_save+'VIMF_CMIP6_eastern_Boundary.npz',\
                allow_pickle=True)['arr_0']

                north_m_arr[p,:,:]=VIMF_model_northern
                south_m_arr[p,:,:]=VIMF_model_southern
                west_m_arr[p,:,:]=VIMF_model_western
                east_m_arr[p,:,:]=VIMF_model_eastern

                np.savez_compressed(path_save+'VIMF_CMIP6_northern_Boundary.npz',north_m_arr)
                np.savez_compressed(path_save+'VIMF_CMIP6_southern_Boundary.npz',south_m_arr)
                np.savez_compressed(path_save+'VIMF_CMIP6_western_Boundary.npz',west_m_arr)
                np.savez_compressed(path_save+'VIMF_CMIP6_eastern_Boundary.npz',east_m_arr)

                models_vimf_calc=np.load(path_save+'VIMF_models_N.npz',allow_pickle=True)['arr_0']

                models_vimf_calc=np.append(models_vimf_calc,models[p])
                np.savez_compressed(path_save+'VIMF_models_N.npz',models_vimf_calc)
            
            except:
                print('Error VIMF ', models[p])
    
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

        models_va=list(dict_models['va'])
        models_hus=list(dict_models['hus'])
        models_ta=list(dict_models['ta'])
        models_zg=list(dict_models['zg'])

        models_1=np.intersect1d(models_va, models_hus)
        models_2=np.intersect1d(models_1, models_ta)
        models=np.intersect1d(models_2, models_zg)

        #------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------
        #Reading the reference information
        MSE_ref_northern=np.load(path_save+'MSE_ERA5_northern_Boundary.npz')['arr_0']
        MSE_ref_southern=np.load(path_save+'MSE_ERA5_southern_Boundary.npz')['arr_0']
        MSE_ref_western=np.load(path_save+'MSE_ERA5_western_Boundary.npz')['arr_0']
        MSE_ref_eastern=np.load(path_save+'MSE_ERA5_eastern_Boundary.npz')['arr_0']

        models_app=np.array([])
        np.savez_compressed(path_save+'MSE_models_N.npz',models_app)

        #Creating the empty arrays to save the models information
        northern_boundary_models=np.empty((len(models),4,MSE_ref_northern.shape[1]))
        southern_boundary_models=np.empty((len(models),4,MSE_ref_southern.shape[1]))
        western_boundary_models=np.empty((len(models),4,MSE_ref_western.shape[1]))
        eastern_boundary_models=np.empty((len(models),4,MSE_ref_eastern.shape[1]))

        np.savez_compressed(path_save+'MSE_CMIP6_northern_Boundary.npz',northern_boundary_models)
        np.savez_compressed(path_save+'MSE_CMIP6_southern_Boundary.npz',southern_boundary_models)
        np.savez_compressed(path_save+'MSE_CMIP6_western_Boundary.npz',western_boundary_models)
        np.savez_compressed(path_save+'MSE_CMIP6_eastern_Boundary.npz',eastern_boundary_models)

        for p in range(len(models)):

            try:

                #Applying the function
                MSE_model,Lat_model,Lon_model,dx_model,\
                dy_model=MSE_calc(path_save, 'va', 'hus', 'ta','zg', models[p],lat_limits,lon_limits,None, None,\
                p_level_interest_lower,p_level_interest_upper,'model')

                #-----------------------------------------------------------------------------
                #Interpolating to the common grid size 
                MSE_model_interpolated=interpolation_fields(MSE_model,Lat_model,Lon_model,Lat_common_b,Lon_common_b,dx_common,dy_common)

                ############################################################################
                #Obtaining the series of each boundary
                #north
                lat_loc_m_n=np.where(np.logical_and(Lat_common_b>=north_boundaries_lat[0], \
                Lat_common_b<=north_boundaries_lat[1]))
                lon_loc_m_n=np.where(np.logical_and(Lon_common_b>=north_boundaries_lon[0], \
                Lon_common_b<=north_boundaries_lon[1]))
                MSE_model_n=MSE_model_interpolated[:,lat_loc_m_n[0][0]:lat_loc_m_n[0][-1]+1,\
                lon_loc_m_n[0][0]:lon_loc_m_n[0][-1]+1]
                MSE_model_northern=np.nanmean(MSE_model_n,axis=1)
                #south
                lat_loc_m_s=np.where(np.logical_and(Lat_common_b>=south_boundaries_lat[0], \
                Lat_common_b<=south_boundaries_lat[1]))
                lon_loc_m_s=np.where(np.logical_and(Lon_common_b>=south_boundaries_lon[0], \
                Lon_common_b<=south_boundaries_lon[1]))
                MSE_model_s=MSE_model_interpolated[:,lat_loc_m_s[0][0]:lat_loc_m_s[0][-1]+1,\
                lon_loc_m_s[0][0]:lon_loc_m_s[0][-1]+1]
                MSE_model_southern=np.nanmean(MSE_model_s,axis=1)
                #west
                lat_loc_m_w=np.where(np.logical_and(Lat_common_b>=west_boundaries_lat[0],\
                Lat_common_b<=west_boundaries_lat[1]))
                lon_loc_m_w=np.where(np.logical_and(Lon_common_b>=west_boundaries_lon[0],\
                Lon_common_b<=west_boundaries_lon[1]))
                MSE_model_w=MSE_model_interpolated[:,lat_loc_m_w[0][0]:lat_loc_m_w[0][-1]+1,\
                lon_loc_m_w[0][0]:lon_loc_m_w[0][-1]+1]
                MSE_model_western=np.nanmean(MSE_model_w,axis=2)
                #east
                lat_loc_m_e=np.where(np.logical_and(Lat_common_b>=east_boundaries_lat[0],\
                Lat_common_b<=east_boundaries_lat[1]))
                lon_loc_m_e=np.where(np.logical_and(Lon_common_b>=east_boundaries_lon[0],\
                Lon_common_b<=east_boundaries_lon[1]))
                MSE_model_e=MSE_model_interpolated[:,lat_loc_m_e[0][0]:lat_loc_m_e[0][-1]+1,\
                lon_loc_m_e[0][0]:lon_loc_m_e[0][-1]+1]
                MSE_model_eastern=np.nanmean(MSE_model_e,axis=2)

                ############################################################################
                ############################################################################
                #SAVING THE INFORMATION

                north_m_arr=np.load(path_save+'MSE_CMIP6_northern_Boundary.npz',\
                allow_pickle=True)['arr_0']
                south_m_arr=np.load(path_save+'MSE_CMIP6_southern_Boundary.npz',\
                allow_pickle=True)['arr_0']
                west_m_arr=np.load(path_save+'MSE_CMIP6_western_Boundary.npz',\
                allow_pickle=True)['arr_0']
                east_m_arr=np.load(path_save+'MSE_CMIP6_eastern_Boundary.npz',\
                allow_pickle=True)['arr_0']

                north_m_arr[p,:,:]=MSE_model_northern
                south_m_arr[p,:,:]=MSE_model_southern
                west_m_arr[p,:,:]=MSE_model_western
                east_m_arr[p,:,:]=MSE_model_eastern

                np.savez_compressed(path_save+'MSE_CMIP6_northern_Boundary.npz',north_m_arr)
                np.savez_compressed(path_save+'MSE_CMIP6_southern_Boundary.npz',south_m_arr)
                np.savez_compressed(path_save+'MSE_CMIP6_western_Boundary.npz',west_m_arr)
                np.savez_compressed(path_save+'MSE_CMIP6_eastern_Boundary.npz',east_m_arr)

                models_mse_calc=np.load(path_save+'MSE_models_N.npz',allow_pickle=True)['arr_0']

                models_mse_calc=np.append(models_mse_calc,models[p])
                np.savez_compressed(path_save+'MSE_models_N.npz',models_mse_calc)
            
            except:
                print('Error MSE ', models[p])
    
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

        models_ua=list(dict_models['ua'])
        models_va=list(dict_models['va'])
        models_hus=list(dict_models['hus'])

        models_1=np.intersect1d(models_ua, models_va)
        models=np.intersect1d(models_1, models_hus)

        p_level_interest_upper=10000.0
        p_level_interest_lower=100000.0

        #------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------
        models_app=np.array([])
        np.savez_compressed(path_save+'qu_qv_models_N.npz',models_app)

        taylor_diagram_metrics=pd.DataFrame(columns=['Model','Boundary','corr_DJF','corr_JJA',\
        'corr_MAM','corr_SON','std_DJF','std_JJA','std_MAM','std_SON'])
        taylor_diagram_metrics.to_csv(path_save+'taylorDiagram_metrics_qu_qv.csv')

        #Reading the reference information
        Lon_ref_north=np.load(path_save+'ERA5_qu_qv_north_Lon.npz')['arr_0']
        qv_ref_northern_av=np.load(path_save+'ERA5_qu_qv_north.npz')['arr_0']
        levels_ref=np.load(path_save+'ERA5_qu_qv_p_level.npz')['arr_0']
        qv_ref_southern_av=np.load(path_save+'ERA5_qu_qv_south.npz')['arr_0']
        qu_ref_eastern_av=np.load(path_save+'ERA5_qu_qv_east.npz')['arr_0']
        qu_ref_western_av=np.load(path_save+'ERA5_qu_qv_west.npz')['arr_0']
        Lat_ref_east=np.load(path_save+'ERA5_qu_qv_east_Lat.npz')['arr_0']

        for p in range(len(models)):

            try:

                #Applying the function

                #Northern boundary 
                qv_m_northern,Lat_m_north,Lon_m_north,dx_m, \
                dy_m,levels_m=boundaries_fluxes(path_save, 'va', 'hus',  models[p],\
                north_boundaries_lat,north_boundaries_lon, None, None,p_level_interest_lower,\
                p_level_interest_upper,'model')

                #Southern boundary
                qv_m_southern,Lat_m_south,Lon_m_south=boundaries_fluxes(path_save, 'va', 'hus',  models[p],\
                south_boundaries_lat,south_boundaries_lon,None, None,p_level_interest_lower,\
                p_level_interest_upper,'model')[0:3]

                #Western boundary
                qu_m_western,Lat_m_west,Lon_m_west=boundaries_fluxes(path_save, 'ua', 'hus',  models[p],\
                west_boundaries_lat,west_boundaries_lon,None, None,p_level_interest_lower,\
                p_level_interest_upper,'model')[0:3]
                
                #Eastern boundary
                qu_m_eastern,Lat_m_east,Lon_m_east=boundaries_fluxes(path_save, 'ua', 'hus',  models[p],\
                east_boundaries_lat,east_boundaries_lon,None, None,p_level_interest_lower,\
                p_level_interest_upper,'model')[0:3]
                
                ############################################################################
                ############################################################################
                #Averaging in the corresponding axis
                qv_m_northern_av=np.nanmean(qv_m_northern,axis=2)*1000
                qv_m_southern_av=np.nanmean(qv_m_southern,axis=2)*1000
                qu_m_western_av=np.nanmean(qu_m_western,axis=3)*1000
                qu_m_eastern_av=np.nanmean(qu_m_eastern,axis=3)*1000

                #CHECKING IF THERE ARE ANY NANs IN THE MATRICES 
                qv_m_northern_av, indices_northern, text_northern=NaNs_land(qv_m_northern_av)
                qv_m_southern_av, indices_southern, text_southern=NaNs_land(qv_m_southern_av)
                qu_m_western_av, indices_western, text_western=NaNs_land(qu_m_western_av)
                qu_m_eastern_av, indices_eastern, text_eastern=NaNs_land(qu_m_eastern_av)

                #------------------------------------------------------------------------------------------------
                #------------------------------------------------------------------------------------------------
                #INTERPOLATION 
                #1. ERA5 to the model's resolution 
                xnew_ref_ns=np.arange(Lon_ref_north[0],Lon_ref_north[-1],dx_m)
                xnew_ref_ew=Lat_m_east
                ynew_ref=levels_m

                ERA_interp_northern=interpolation_boundaries(qv_ref_northern_av, levels_ref,levels_m, \
                                                            Lon_ref_north,Lon_m_north, dx_m,dy_m,ynew_ref, xnew_ref_ns)
                
                ERA_interp_southern=interpolation_boundaries(qv_ref_southern_av, levels_ref,levels_m, \
                                                            Lon_ref_north,Lon_m_north, dx_m,dy_m,ynew_ref, xnew_ref_ns)
                
                ERA_interp_eastern=interpolation_boundaries(qu_ref_eastern_av, levels_ref,levels_m, \
                                                            Lat_ref_east,Lat_m_east, dx_m,dy_m,ynew_ref, xnew_ref_ew)
                
                ERA_interp_western=interpolation_boundaries(qu_ref_western_av, levels_ref,levels_m, \
                                                            Lat_ref_east,Lat_m_east, dx_m,dy_m,ynew_ref, xnew_ref_ew)
                
                #calculing the metrics
                corr_m_o_north,std_m_north=taylor_diagram_metrics_def(ERA_interp_northern,qv_m_northern_av)
                corr_m_o_south,std_m_south=taylor_diagram_metrics_def(ERA_interp_southern,qv_m_southern_av)
                corr_m_o_east,std_m_east=taylor_diagram_metrics_def(ERA_interp_eastern,qu_m_eastern_av)
                corr_m_o_west,std_m_west=taylor_diagram_metrics_def(ERA_interp_western,qu_m_western_av)

                #Calculating the bias of the models
                northern_bias= qv_m_northern_av - ERA_interp_northern
                southern_bias= qv_m_southern_av - ERA_interp_southern
                eastern_bias= qu_m_eastern_av - ERA_interp_eastern
                western_bias= qu_m_western_av - ERA_interp_western

                #--------------------------------------------------------------------------------------------------
                #2. Model to common gridsize 
                Lon_common_0=360+Lon_common_fl

                if Lon_m_south[0]<0:
                    Lon_common_bF=Lon_common_fl
                else:
                    Lon_common_bF=Lon_common_0

                model_interp_northern=interpolation_boundaries(qv_m_northern_av, levels_m,p_level_common, \
                                                            Lon_m_north,Lon_common_bF, dx_common,dy_common,levels_m,Lon_common_bF)
                
                model_interp_northern_bias=interpolation_boundaries(northern_bias, levels_m,p_level_common, \
                                                            Lon_m_north,Lon_common_bF, dx_common,dy_common,levels_m,Lon_common_bF)
                
                model_interp_southern=interpolation_boundaries(qv_m_southern_av, levels_m,p_level_common, \
                                                            Lon_m_north,Lon_common_bF, dx_common,dy_common,levels_m,Lon_common_bF)
                
                model_interp_southern_bias=interpolation_boundaries(southern_bias, levels_m,p_level_common, \
                                                            Lon_m_north,Lon_common_bF, dx_common,dy_common,levels_m,Lon_common_bF)
                
                model_interp_eastern=interpolation_boundaries(qu_m_eastern_av, levels_m,p_level_common, \
                                                            Lat_m_east,Lat_common_fl, dx_common,dy_common,levels_m,Lat_common_fl)
                
                model_interp_eastern_bias=interpolation_boundaries(eastern_bias, levels_m,p_level_common, \
                                                            Lat_m_east,Lat_common_fl, dx_common,dy_common,levels_m,Lat_common_fl)
                
                model_interp_western=interpolation_boundaries(qu_m_western_av, levels_m,p_level_common, \
                                                            Lat_m_east,Lat_common_fl, dx_common,dy_common,levels_m,Lat_common_fl)
                
                model_interp_western_bias=interpolation_boundaries(western_bias, levels_m,p_level_common, \
                                                            Lat_m_east,Lat_common_fl, dx_common,dy_common,levels_m,Lat_common_fl)
                
                ############################################################################
                ############################################################################
                #SAVING THE INFORMATION
                #1. Metrics
                taylor_diagram_metrics_DT=pd.read_csv(path_save+'taylorDiagram_metrics_qu_qv.csv', index_col=[0])

                newRow_metrics_north_m=pd.DataFrame({'Model':[models[p]],'Boundary':['north'],'corr_DJF': [corr_m_o_north[0]],\
                'corr_JJA': [corr_m_o_north[1]],'corr_MAM': [corr_m_o_north[2]],'corr_SON': [corr_m_o_north[3]],\
                'std_DJF':[std_m_north[0]],'std_JJA':[std_m_north[1]],'std_MAM':[std_m_north[2]],'std_SON':[std_m_north[3]]})

                newRow_metrics_south_m=pd.DataFrame({'Model':[models[p]],'Boundary':['south'],'corr_DJF': [corr_m_o_south[0]],\
                'corr_JJA': [corr_m_o_south[1]],'corr_MAM': [corr_m_o_south[2]],'corr_SON': [corr_m_o_south[3]],\
                'std_DJF':[std_m_south[0]],'std_JJA':[std_m_south[1]],'std_MAM':[std_m_south[2]],'std_SON':[std_m_south[3]]})

                newRow_metrics_east_m=pd.DataFrame({'Model':[models[p]],'Boundary':['east'],'corr_DJF': [corr_m_o_east[0]],\
                'corr_JJA': [corr_m_o_east[1]],'corr_MAM': [corr_m_o_east[2]],'corr_SON': [corr_m_o_east[3]],\
                'std_DJF':[std_m_east[0]],'std_JJA':[std_m_east[1]],'std_MAM':[std_m_east[2]],'std_SON':[std_m_east[3]]})

                newRow_metrics_west_m=pd.DataFrame({'Model':[models[p]],'Boundary':['west'],'corr_DJF': [corr_m_o_west[0]],\
                'corr_JJA': [corr_m_o_west[1]],'corr_MAM': [corr_m_o_west[2]],'corr_SON': [corr_m_o_west[3]],\
                'std_DJF':[std_m_west[0]],'std_JJA':[std_m_west[1]],'std_MAM':[std_m_west[2]],'std_SON':[std_m_west[3]]})

                taylor_diagram_metrics_DT = pd.concat([taylor_diagram_metrics_DT, newRow_metrics_north_m], ignore_index=True)

                taylor_diagram_metrics_DT = pd.concat([taylor_diagram_metrics_DT, newRow_metrics_south_m], ignore_index=True)

                taylor_diagram_metrics_DT = pd.concat([taylor_diagram_metrics_DT, newRow_metrics_east_m], ignore_index=True)

                taylor_diagram_metrics_DT = pd.concat([taylor_diagram_metrics_DT, newRow_metrics_west_m], ignore_index=True)

                taylor_diagram_metrics_DT.to_csv(path_save+'taylorDiagram_metrics_qu_qv.csv')

                #2. Ensamble matrix
                np.savez_compressed(path_save+models[p]+'_qu_qv_north.npz',model_interp_northern)

                np.savez_compressed(path_save+models[p]+'_qu_qv_south.npz',model_interp_southern)

                np.savez_compressed(path_save+models[p]+'_qu_qv_east.npz',model_interp_eastern)

                np.savez_compressed(path_save+models[p]+'_qu_qv_west.npz',model_interp_western)

                #3. Bias matrix
                np.savez_compressed(path_save+models[p]+'_qu_qv_north_bias.npz',model_interp_northern_bias)

                np.savez_compressed(path_save+models[p]+'_qu_qv_south_bias.npz',model_interp_southern_bias)

                np.savez_compressed(path_save+models[p]+'_qu_qv_east_bias.npz',model_interp_eastern_bias)

                np.savez_compressed(path_save+models[p]+'_qu_qv_west_bias.npz',model_interp_western_bias)

                models_qu_qv_calc=np.load(path_save+'qu_qv_models_N.npz',allow_pickle=True)['arr_0']

                models_qu_qv_calc=np.append(models_qu_qv_calc,models[p])
                np.savez_compressed(path_save+'qu_qv_models_N.npz',models_qu_qv_calc)
            
            except:
                print('Error qu_qv ', models[p])
    
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

        models_ua=list(dict_models['ua'])
        models_va=list(dict_models['va'])
        models_ta=list(dict_models['ta'])

        models_1=np.intersect1d(models_ua, models_va)
        models=np.intersect1d(models_1, models_ta)

        p_level_interest_upper=10000.0
        p_level_interest_lower=100000.0

        #------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------
        models_app=np.array([])
        np.savez_compressed(path_save+'tu_tv_models_N.npz',models_app)

        taylor_diagram_metrics=pd.DataFrame(columns=['Model','Boundary','corr_DJF','corr_JJA',\
        'corr_MAM','corr_SON','std_DJF','std_JJA','std_MAM','std_SON'])
        taylor_diagram_metrics.to_csv(path_save+'taylorDiagram_metrics_tu_tv.csv')

        #Reading the reference information
        Lon_ref_north=np.load(path_save+'ERA5_tu_tv_north_Lon.npz')['arr_0']
        tv_ref_northern_av=np.load(path_save+'ERA5_tu_tv_north.npz')['arr_0']
        levels_ref=np.load(path_save+'ERA5_tu_tv_p_level.npz')['arr_0']
        tv_ref_southern_av=np.load(path_save+'ERA5_tu_tv_south.npz')['arr_0']
        tu_ref_eastern_av=np.load(path_save+'ERA5_tu_tv_east.npz')['arr_0']
        tu_ref_western_av=np.load(path_save+'ERA5_tu_tv_west.npz')['arr_0']
        Lat_ref_east=np.load(path_save+'ERA5_tu_tv_east_Lat.npz')['arr_0']

        for p in range(len(models)):

            try:

                #Applying the function

                #Northern boundary 
                tv_m_northern,Lat_m_north,Lon_m_north,dx_m, \
                dy_m,levels_m=boundaries_fluxes(path_save, 'va', 'ta',  models[p],\
                north_boundaries_lat,north_boundaries_lon, None, None,p_level_interest_lower,\
                p_level_interest_upper,'model')

                #Southern boundary
                tv_m_southern,Lat_m_south,Lon_m_south=boundaries_fluxes(path_save, 'va', 'ta',  models[p],\
                south_boundaries_lat,south_boundaries_lon,None, None,p_level_interest_lower,\
                p_level_interest_upper,'model')[0:3]

                #Western boundary
                tu_m_western,Lat_m_west,Lon_m_west=boundaries_fluxes(path_save, 'ua', 'ta',  models[p],\
                west_boundaries_lat,west_boundaries_lon,None, None,p_level_interest_lower,\
                p_level_interest_upper,'model')[0:3]
                
                #Eastern boundary
                tu_m_eastern,Lat_m_east,Lon_m_east=boundaries_fluxes(path_save, 'ua', 'ta',  models[p],\
                east_boundaries_lat,east_boundaries_lon,None, None,p_level_interest_lower,\
                p_level_interest_upper,'model')[0:3]
                
                ############################################################################
                ############################################################################
                #Averaging in the corresponding axis
                tv_m_northern_av=np.nanmean(tv_m_northern,axis=2)
                tv_m_southern_av=np.nanmean(tv_m_southern,axis=2)
                tu_m_western_av=np.nanmean(tu_m_western,axis=3)
                tu_m_eastern_av=np.nanmean(tu_m_eastern,axis=3)

                #CHECKING IF THERE ARE ANY NANs IN THE MATRICES 
                tv_m_northern_av, indices_northern, text_northern=NaNs_land(tv_m_northern_av)
                tv_m_southern_av, indices_southern, text_southern=NaNs_land(tv_m_southern_av)
                tu_m_western_av, indices_western, text_western=NaNs_land(tu_m_western_av)
                tu_m_eastern_av, indices_eastern, text_eastern=NaNs_land(tu_m_eastern_av)

                #------------------------------------------------------------------------------------------------
                #------------------------------------------------------------------------------------------------
                #INTERPOLATION 
                #1. ERA5 to the model's resolution 
                xnew_ref_ns=np.arange(Lon_ref_north[0],Lon_ref_north[-1],dx_m)
                xnew_ref_ew=Lat_m_east
                ynew_ref=levels_m

                ERA_interp_northern=interpolation_boundaries(tv_ref_northern_av, levels_ref,levels_m, \
                                                            Lon_ref_north,Lon_m_north, dx_m,dy_m,ynew_ref, xnew_ref_ns)
                
                ERA_interp_southern=interpolation_boundaries(tv_ref_southern_av, levels_ref,levels_m, \
                                                            Lon_ref_north,Lon_m_north, dx_m,dy_m,ynew_ref, xnew_ref_ns)
                
                ERA_interp_eastern=interpolation_boundaries(tu_ref_eastern_av, levels_ref,levels_m, \
                                                            Lat_ref_east,Lat_m_east, dx_m,dy_m,ynew_ref, xnew_ref_ew)
                
                ERA_interp_western=interpolation_boundaries(tu_ref_western_av, levels_ref,levels_m, \
                                                            Lat_ref_east,Lat_m_east, dx_m,dy_m,ynew_ref, xnew_ref_ew)
                
                #calculing the metrics
                corr_m_o_north,std_m_north=taylor_diagram_metrics_def(ERA_interp_northern,tv_m_northern_av)
                corr_m_o_south,std_m_south=taylor_diagram_metrics_def(ERA_interp_southern,tv_m_southern_av)
                corr_m_o_east,std_m_east=taylor_diagram_metrics_def(ERA_interp_eastern,tu_m_eastern_av)
                corr_m_o_west,std_m_west=taylor_diagram_metrics_def(ERA_interp_western,tu_m_western_av)

                #Calculating the bias of the models
                northern_bias= tv_m_northern_av - ERA_interp_northern
                southern_bias= tv_m_southern_av - ERA_interp_southern
                eastern_bias= tu_m_eastern_av - ERA_interp_eastern
                western_bias= tu_m_western_av - ERA_interp_western

                #--------------------------------------------------------------------------------------------------
                #2. Model to common gridsize 
                Lon_common_0=360+Lon_common_fl

                if Lon_m_south[0]<0:
                    Lon_common_bF=Lon_common_fl
                else:
                    Lon_common_bF=Lon_common_0

                model_interp_northern=interpolation_boundaries(tv_m_northern_av, levels_m,p_level_common, \
                                                            Lon_m_north,Lon_common_bF, dx_common,dy_common,levels_m,Lon_common_bF)
                
                model_interp_northern_bias=interpolation_boundaries(northern_bias, levels_m,p_level_common, \
                                                            Lon_m_north,Lon_common_bF, dx_common,dy_common,levels_m,Lon_common_bF)
                
                model_interp_southern=interpolation_boundaries(tv_m_southern_av, levels_m,p_level_common, \
                                                            Lon_m_north,Lon_common_bF, dx_common,dy_common,levels_m,Lon_common_bF)
                
                model_interp_southern_bias=interpolation_boundaries(southern_bias, levels_m,p_level_common, \
                                                            Lon_m_north,Lon_common_bF, dx_common,dy_common,levels_m,Lon_common_bF)
                
                model_interp_eastern=interpolation_boundaries(tu_m_eastern_av, levels_m,p_level_common, \
                                                            Lat_m_east,Lat_common_fl, dx_common,dy_common,levels_m,Lat_common_fl)
                
                model_interp_eastern_bias=interpolation_boundaries(eastern_bias, levels_m,p_level_common, \
                                                            Lat_m_east,Lat_common_fl, dx_common,dy_common,levels_m,Lat_common_fl)
                
                model_interp_western=interpolation_boundaries(tu_m_western_av, levels_m,p_level_common, \
                                                            Lat_m_east,Lat_common_fl, dx_common,dy_common,levels_m,Lat_common_fl)
                
                model_interp_western_bias=interpolation_boundaries(western_bias, levels_m,p_level_common, \
                                                            Lat_m_east,Lat_common_fl, dx_common,dy_common,levels_m,Lat_common_fl)
                
                ############################################################################
                ############################################################################
                #SAVING THE INFORMATION
                #1. Metrics
                taylor_diagram_metrics_DT=pd.read_csv(path_save+'taylorDiagram_metrics_tu_tv.csv', index_col=[0])

                newRow_metrics_north_m=pd.DataFrame({'Model':[models[p]],'Boundary':['north'],'corr_DJF': [corr_m_o_north[0]],\
                'corr_JJA': [corr_m_o_north[1]],'corr_MAM': [corr_m_o_north[2]],'corr_SON': [corr_m_o_north[3]],\
                'std_DJF':[std_m_north[0]],'std_JJA':[std_m_north[1]],'std_MAM':[std_m_north[2]],'std_SON':[std_m_north[3]]})

                newRow_metrics_south_m=pd.DataFrame({'Model':[models[p]],'Boundary':['south'],'corr_DJF': [corr_m_o_south[0]],\
                'corr_JJA': [corr_m_o_south[1]],'corr_MAM': [corr_m_o_south[2]],'corr_SON': [corr_m_o_south[3]],\
                'std_DJF':[std_m_south[0]],'std_JJA':[std_m_south[1]],'std_MAM':[std_m_south[2]],'std_SON':[std_m_south[3]]})

                newRow_metrics_east_m=pd.DataFrame({'Model':[models[p]],'Boundary':['east'],'corr_DJF': [corr_m_o_east[0]],\
                'corr_JJA': [corr_m_o_east[1]],'corr_MAM': [corr_m_o_east[2]],'corr_SON': [corr_m_o_east[3]],\
                'std_DJF':[std_m_east[0]],'std_JJA':[std_m_east[1]],'std_MAM':[std_m_east[2]],'std_SON':[std_m_east[3]]})

                newRow_metrics_west_m=pd.DataFrame({'Model':[models[p]],'Boundary':['west'],'corr_DJF': [corr_m_o_west[0]],\
                'corr_JJA': [corr_m_o_west[1]],'corr_MAM': [corr_m_o_west[2]],'corr_SON': [corr_m_o_west[3]],\
                'std_DJF':[std_m_west[0]],'std_JJA':[std_m_west[1]],'std_MAM':[std_m_west[2]],'std_SON':[std_m_west[3]]})

                taylor_diagram_metrics_DT = pd.concat([taylor_diagram_metrics_DT, newRow_metrics_north_m], ignore_index=True)

                taylor_diagram_metrics_DT = pd.concat([taylor_diagram_metrics_DT, newRow_metrics_south_m], ignore_index=True)

                taylor_diagram_metrics_DT = pd.concat([taylor_diagram_metrics_DT, newRow_metrics_east_m], ignore_index=True)

                taylor_diagram_metrics_DT = pd.concat([taylor_diagram_metrics_DT, newRow_metrics_west_m], ignore_index=True)

                taylor_diagram_metrics_DT.to_csv(path_save+'taylorDiagram_metrics_tu_tv.csv')

                #2. Ensamble matrix
                np.savez_compressed(path_save+models[p]+'_tu_tv_north.npz',model_interp_northern)

                np.savez_compressed(path_save+models[p]+'_tu_tv_south.npz',model_interp_southern)

                np.savez_compressed(path_save+models[p]+'_tu_tv_east.npz',model_interp_eastern)

                np.savez_compressed(path_save+models[p]+'_tu_tv_west.npz',model_interp_western)

                #3. Bias matrix
                np.savez_compressed(path_save+models[p]+'_tu_tv_north_bias.npz',model_interp_northern_bias)

                np.savez_compressed(path_save+models[p]+'_tu_tv_south_bias.npz',model_interp_southern_bias)

                np.savez_compressed(path_save+models[p]+'_tu_tv_east_bias.npz',model_interp_eastern_bias)

                np.savez_compressed(path_save+models[p]+'_tu_tv_west_bias.npz',model_interp_western_bias)

                models_tu_tv_calc=np.load(path_save+'tu_tv_models_N.npz',allow_pickle=True)['arr_0']

                models_tu_tv_calc=np.append(models_tu_tv_calc,models[p])
                np.savez_compressed(path_save+'tu_tv_models_N.npz',models_tu_tv_calc)
            
            except:
                print('Error tu_tv ',models[p])

#--------------------------------------------------------------------------------------------------
print('#################################################')
print('Finished')
print('#################################################')
        







            











        







            











            








