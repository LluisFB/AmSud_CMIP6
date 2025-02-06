""" 
Code to calculate some additiona suggestions for the review of the paper 

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


#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#Path_save is the path that contains all the created files
path_save='/scratchx/lfita/'
path_entry='/scratchx/lfita/'

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#Defining the parameters of the plots 
fig_title_font=25
title_str_size=23
xy_label_str=20
tick_labels_str=18
tick_labels_str_regcell=18
legends_str=16

#-----------------------------------------------------------------------------------------------------------------------
gridsize_df=pd.read_csv(path_save+'CMIP6_models_GridSize_lat_lon_Amon.csv', index_col=[0])
gridsize_df_tos=pd.read_csv(path_save+'CMIP6_models_GridSize_lat_lon_Omon.csv', index_col=[0])


#Finding the grid size to perform the interpolation 
dx_common=gridsize_df['Longitude'].max()
dy_common=gridsize_df['Latitude'].max()


p_level_common=[10000.0,15000.0,20000.0,25000.0,30000.0,40000.0,50000.0,60000.0,\
70000.0,85000.0,92500.0,100000.0]

lon_limits_F=[-83,-35]
lat_limits_F=[-57,15]

Lat_common=np.arange(lat_limits_F[0],lat_limits_F[1],dy_common)
Lon_common=np.arange(lon_limits_F[0],lon_limits_F[1],dx_common)


############################################################################################################
############################################################################################################
#1. FOR PRECIPITATION

#ERA5

lon_limits_pr=lon_limits_F
lat_limits_pr=lat_limits_F

try:

    pr_array,Lat_list_pr,Lon_list_pr,dx_data_pr, dy_data_pr=var_field_calc(path_save,'tp','ERA5',\
                                                                            lat_limits_pr,lon_limits_pr,None,None,\
                                                                            None,None,'ERA5','No')
    
    print('####################################')
    print('Precipitation: var_array OK')
    print('####################################')
    
    pr_array=pr_array*1000

    var_sum=np.sum(pr_array)

    if np.isnan(var_sum)==True :
        var_array_r=NaNs_interp(pr_array, '3D', 'cubic')
    else:
        var_array_r=pr_array
    
    print('####################################')
    print('Precipitation: var_array NaNs OK')
    print('####################################')
    
    #-----------------------------------------------------------------------------------------------------------------------
    #Saving the spatial fields 
    np.savez_compressed(path_save+'ERA5_tp_fields_SA.npz',var_array_r)
    np.savez_compressed(path_save+'ERA5_tp_fields_Lat_SA.npz',Lat_list_pr)
    np.savez_compressed(path_save+'ERA5_tp_fields_Lon_SA.npz',Lon_list_pr)

    std_ref(var_array_r, path_save, 'PPT_SA')

    print('####################################')
    print('Precipitation: std_ref OK')
    print('####################################')

except Exception as e:
    print('Error ERA5 Precipitation')
    print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")


#Models 

with open(path_save+'Models_var_availability.pkl', 'rb') as fp:
    dict_models = pickle.load(fp)

lon_limits_pr=lon_limits_F
lat_limits_pr=lat_limits_F

models=list(dict_models['pr'])

#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#creating and saving the matrices of the fields 
models_app=np.array([])

taylor_diagram_metrics=pd.DataFrame(columns=['Model','corr_DJF','corr_JJA',\
'corr_MAM','corr_SON','std_DJF','std_JJA','std_MAM','std_SON'])

taylor_diagram_metrics.to_csv(path_save+'taylorDiagram_metrics_pr_SA.csv')
np.savez_compressed(path_save+'pr_fields_models_N_SA.npz',models_app)

#Reading the reference files 
var_array_ref=np.load(path_save+'ERA5_tp_fields_SA.npz')['arr_0']
lat_refe=np.load(path_save+'ERA5_tp_fields_Lat_SA.npz')['arr_0']
lon_refe=np.load(path_save+'ERA5_tp_fields_Lon_SA.npz')['arr_0']


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
        
        models_pr_calc=np.load(path_save+'pr_fields_models_N_SA.npz',allow_pickle=True)['arr_0']

    
        models_pr_calc=np.append(models_pr_calc,models[p])

        np.savez_compressed(path_save+models[p]+'_pr_MMM_meanFields_SA.npz',model_field_interp)
        np.savez_compressed(path_save+models[p]+'_pr_MMM_biasFields_SA.npz',model_bias_field_interp)
        np.savez_compressed(path_save+'pr_fields_models_N_SA.npz',models_pr_calc)

        #Saving the performance metrics 
        taylor_diagram_metrics_DT=pd.read_csv(path_save+'taylorDiagram_metrics_pr_SA.csv', index_col=[0])

        newRow_metrics=pd.DataFrame({'Model':[models[p]],'corr_DJF': [corr_m_o[0]],\
        'corr_JJA': [corr_m_o[1]],'corr_MAM': [corr_m_o[2]],'corr_SON': [corr_m_o[3]],\
        'std_DJF':[std_m[0]],'std_JJA':[std_m[1]],'std_MAM':[std_m[2]],'std_SON':[std_m[3]]})

        #taylor_diagram_metrics_DT=taylor_diagram_metrics_DT.append(newRow_metrics,\
        #ignore_index=True)
        taylor_diagram_metrics_DT = pd.concat([taylor_diagram_metrics_DT, newRow_metrics], ignore_index=True)


        taylor_diagram_metrics_DT.to_csv(path_save+'taylorDiagram_metrics_pr_SA.csv')
    
    except Exception as e:
        print('Error plot precipitation',models[p])
        print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")

    


#Creating the plot 

#CMAP
cmap_plot=['gist_rainbow_r','terrain','terrain_r','PuOr_r','RdBu_r','RdBu','rainbow']

cmap_pr=cmap_plot[2]
cmap_bias=cmap_plot[4]

try:

    #input
    models=np.load(path_entry+'pr_fields_models_N_SA.npz',allow_pickle=True)['arr_0']

    ################################################################################
    #Obtaining the ensamble

    var_mmm=seasonal_ensamble(models,path_entry,\
    'pr_MMM_meanFields_SA',len(Lat_common),len(Lon_common))

    var_mmm_bias=seasonal_ensamble(models,path_entry,\
    'pr_MMM_biasFields_SA',len(Lat_common),len(Lon_common))

    bias_mmm_agreement=agreement_sign(models,path_entry,'pr_MMM_biasFields_SA',\
                                    len(Lat_common),len(Lon_common))
    
    print('-----------------------------------------------------------------------------------')
    print('Precipitation: files read OK')
    print('-----------------------------------------------------------------------------------')

    ################################################################################
    #PLOT
    models_metrics=pd.read_csv(path_entry+'taylorDiagram_metrics_pr_SA.csv', index_col=[0])

    ref_std=pd.read_csv(path_entry+'reference_std_original.csv',index_col=[0])

    
    plot_label='Precipitation rate\n [mm/day]'
    limits_var=np.arange(2,18,1)
    limits_bias=np.arange(-9,10,1)

    cmap_plot=['gist_rainbow_r','terrain','terrain_r','rainbow','RdBu']

    fig=plt.figure(figsize=(20,14))
    colorbar_attributes=[0.92, 0.37,  0.017,0.24]

    colorbar_attributes_bias=[0.92, 0.1,  0.017,0.24]

    print('-----------------------------------------------------------------------------------')
    print('Precipitation: files metrics OK')
    print('-----------------------------------------------------------------------------------')

    lon2D, lat2D = np.meshgrid(Lon_common, Lat_common)
    projection=ccrs.PlateCarree()
    extent = [min(Lon_common),max(Lon_common),min(Lat_common),max(Lat_common)]

    taylor=td_plots(fig,'DJF',ref_std,models_metrics,'PPT_SA',len(models),341,'a.',title_str_size,'no',None)

    taylor=td_plots(fig,'MAM',ref_std,models_metrics,'PPT_SA',len(models),342,'b.',title_str_size,'no',None)

    taylor=td_plots(fig,'JJA',ref_std,models_metrics,'PPT_SA',len(models),343,'c.',title_str_size,'no',None)

    taylor=td_plots(fig,'SON',ref_std,models_metrics,'PPT_SA',len(models),344,'d.',title_str_size,'no',None)

    ax5 = fig.add_subplot(3, 4, 5, projection=projection)
    cs=plotMap(ax5,var_mmm[0],lon2D,lat2D,cmap_pr,limits_var,'e.',extent, projection,title_str_size,'no',None,'no')

    ax6 = fig.add_subplot(3, 4, 6, projection=projection)
    cs=plotMap(ax6,var_mmm[2],lon2D,lat2D,cmap_pr,limits_var,'f.',extent, projection,title_str_size,'no',None,'no')

    ax7 = fig.add_subplot(3, 4, 7, projection=projection)
    cs=plotMap(ax7,var_mmm[1],lon2D,lat2D,cmap_pr,limits_var,'g.',extent, projection,title_str_size,'no',None,'no')

    ax8 = fig.add_subplot(3, 4, 8, projection=projection)
    cs=plotMap(ax8,var_mmm[3],lon2D,lat2D,cmap_pr,limits_var,'h.',extent, projection,title_str_size,'no',None,'no')

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
    fig.subplots_adjust(hspace=0.3)
    plt.savefig(path_save+'pr_mm_day_SA.png', \
    format = 'png', bbox_inches='tight')
    
    #To save the legend independently
    dia=taylor

    legend= fig.legend(dia.samplePoints,
    [ p.get_label() for p in dia.samplePoints ],
    numpoints=1, prop=dict(size='small'),bbox_to_anchor=(1.11, 0.85) \
    ,ncol=4,loc='right')

    ncols=4

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
    path_save+'pr_mm_day_legend_SA.png', format = 'png',\
    bbox_inches='tight',bbox_extra_artists=[legend_squared],
    ) 
    plt.close()

except Exception as e:
    print('Error plot precipitation')
    print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")


####################################################################################################################
####################################################################################################################
#2. POLAR JET 

#ERA5 

lon_limits_polar=[-140,-70]
lat_limits_polar=[-60,-50]

try: 
    jet_strength_ref, jet_latitude_ref=subtropical_jet(path_save,'u','ERA5', None,lon_limits_polar,\
                                                lat_limits_polar, 20000.0,20000.0)


    #saving the npz
    np.savez_compressed(path_save+'polar_jet_strength_ERA5.npz',jet_strength_ref)
    np.savez_compressed(path_save+'polar_jet_latitude_ERA5.npz',jet_latitude_ref)


except Exception as e:
    print('Error ERA5 Wind Indices')
    print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")


#Models 

models=list(dict_models['ua'])

#Models
polar_strength_models=np.empty((len(models),12))
polar_latitude_models=np.empty((len(models),12))

models_app=np.array([])

#saving the npz
np.savez_compressed(path_save+'polar_jet_strength_models.npz',polar_strength_models)
np.savez_compressed(path_save+'polar_jet_latitude_models.npz',polar_latitude_models)
np.savez_compressed(path_save+'polar_indices_models_N.npz',models_app)

for p in range(len(models)):

    try: 
        jet_strength_m, jet_latitude_m=subtropical_jet(path_save,'ua',models[p], None,lon_limits_polar,\
                                                    lat_limits_polar, 20000.0,20000.0)
        
        #---------------------------------------------------------------------------------------------------------

        #reading and saving the existing npz
        polar_Sm=np.load(path_save+'polar_jet_strength_models.npz',\
        allow_pickle=True)['arr_0']

        polar_Lm=np.load(path_save+'polar_jet_latitude_models.npz',\
        allow_pickle=True)['arr_0']

        models_wind_calc=np.load(path_save+'polar_indices_models_N.npz',\
        allow_pickle=True)['arr_0']

        #Saving the indices from the models in the arrays
        polar_Sm[p,:]=jet_strength_m
        polar_Lm[p,:]=jet_latitude_m

        models_wind_calc=np.append(models_wind_calc,models[p])

        np.savez_compressed(path_save+'polar_jet_strength_models.npz',polar_Sm)
        np.savez_compressed(path_save+'polar_jet_latitude_models.npz',polar_Lm)
        np.savez_compressed(path_save+'polar_indices_models_N.npz',models_wind_calc)

    except:
        print('Error: Wind indices ', models[p])

#------------------------------------------------------------------------------------
#Creating the plot 
#------------------------------------------------------------------------------------

try:

    #Inputs

    polar_strength_ref=np.load(path_entry+'polar_jet_strength_ERA5.npz',allow_pickle=True)['arr_0']
    polar_Str_models=np.load(path_entry+'polar_jet_strength_models.npz',allow_pickle=True)['arr_0']
    polar_latitude_ref=np.load(path_entry+'polar_jet_latitude_ERA5.npz',allow_pickle=True)['arr_0']
    polar_Lat_models=np.load(path_entry+'polar_jet_latitude_models.npz',allow_pickle=True)['arr_0']

    models=np.load(path_entry+'polar_indices_models_N.npz',allow_pickle=True)['arr_0']

    print('-----------------------------------------------------------------------------------')
    print('wind_indices: files read OK')
    print('-----------------------------------------------------------------------------------')

    #Obtaining the metrics 
    
    series_metrics(polar_strength_ref,polar_Str_models,models,'polarJet_strength',path_entry)
    series_metrics(polar_latitude_ref,polar_Lat_models,models,'polarJet_latitude',path_entry)


    print('-----------------------------------------------------------------------------------')
    print('wind_indices: series OK')
    print('-----------------------------------------------------------------------------------')
    
    #Polar jet stream
    wind_indices('Southern Hemisphere Polar Jet Stream',polar_strength_ref,\
    polar_Str_models,polar_latitude_ref,polar_Lat_models,\
    'a. Polar jet stream mean strength','b. Polar jet stream mean location',\
    np.arange(10,64,4),np.arange(-60,-46,2),models,'polar_Jet_200hPa',path_save,2,1,10,14,fig_title_font,\
        title_str_size, xy_label_str, tick_labels_str, legends_str,[])
    

except Exception as e:
    print('Error plot wind indices')
    print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")


