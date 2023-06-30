"""
Code to calculate the performance of the models in the 
simulation of ENSO patterns 

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
from scipy import integrate, stats, signal
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

#######################################################################################################
#Defining the paths 
#path_save is the path were all files are being saved (.nc, .npz, .csv files)
#path_save_plots is the path were the plots are being saved (It can be the same path of path_save)

#This is the only part of the code to be changed

path_save='/home/iccorreasa/Documentos/Paper_CMIP6_models/PLOTS_paper/PAPER_FINAL/npz/' #CHANGE
path_save_plots='/home/iccorreasa/Documentos/Paper_CMIP6_models/PLOTS_paper/PAPER_FINAL/plots/' #CHANGE

#-------------------------------------------------------------------------------------------------------

gridsize_df_tos=pd.read_csv(path_save+'CMIP6_models_GridSize_lat_lon_Omon.csv', index_col=[0])
gridsize_df=pd.read_csv(path_save+'CMIP6_models_GridSize_lat_lon_Amon.csv', index_col=[0])

#Finding the grid size to perform the interpolation 
dx_common=gridsize_df['Longitude'].max()
dy_common=gridsize_df['Latitude'].max()

dx_common_tos=gridsize_df_tos['Longitude'].max()
dy_common_tos=gridsize_df_tos['Latitude'].max()

p_level_common=[10000.0,15000.0,20000.0,25000.0,30000.0,40000.0,50000.0,60000.0,\
70000.0,85000.0,92500.0,100000.0]

ninno3_lat=[-5,5]
ninno3_lon=[-150,-90]

lat_regre=[-60,40]
lon_regre=[-153,-30]

Lat_common_tos=np.arange(lat_regre[0],lat_regre[1],dy_common_tos)
Lon_common_tos=np.arange(lon_regre[0],lon_regre[1],dx_common_tos)

def ENSO_calculations(path_entry,model_name,type_data,lat_bnds_field,lon_bnds_field,lat_bnd_index,lon_bnd_index):
    if type_data=='model':

        file_data=xr.open_dataset(path_entry+model_name+'_tos_original_mon_clim_LT.nc')

        data=file_data['tos']
    
    else:

        file_data=xr.open_dataset(path_entry+'ERA5_sst_original_mon_clim_LT.nc')

        data=file_data['sst']
    
    if data.units=='K' or data.units=='k':
        data=data-273.15
    
    else:
        pass

    ###########################################################################################
    #1. Calculating the El Niño 3 index (standirzed anomalies)
    n_years=int(data.shape[0]/12)

    lat_bnd,lon_bnd,lat_slice,lon_slice=lat_lon_bds(lon_bnd_index,lat_bnd_index,data)
    #delimiting the spatial domain
    var_delimited=time_lat_lon_positions(None,None,lat_bnd,\
    lon_bnd,lat_slice,data,'ERA5','No')

    var_levels=var_delimited

    #defining Lat and Lon lists
    if lat_slice=='lat':
        var_lat=var_levels.mean(dim='lat')
        var_spatial=var_lat.mean(dim='lon')
    else:
        var_lat=var_levels.mean(dim='latitude')
        var_spatial=var_lat.mean(dim='longitude')

    #------------------------------------------------------------------------------------------
    #Obtaining the climatology 
    var_clim=var_spatial.groupby('time.month').mean('time')

    var_arr_longterm=np.array(var_spatial)

    clim_arr=np.array(var_clim)
    var_clim_arr=np.tile(clim_arr,n_years)


    #Obtaining the standar deviation of the long-term series 
    std_serie=np.nanstd(var_arr_longterm)


    #Obtaining the standarized anomalies 
    std_anom=(var_arr_longterm - var_clim_arr)/std_serie


    ##########################################################################################
    #2. Calculating the regressed anomalies of SST 

    lat_bnd_r,lon_bnd_r=lat_lon_bds(lon_bnds_field,lat_bnds_field,data)[0:2]

    #delimiting the spatial domain
    var_delimited_r=time_lat_lon_positions(None,None,lat_bnd_r,\
    lon_bnd_r,lat_slice,data,'ERA5','No')

    var_levels_r=var_delimited_r

    #defining Lat and Lon lists
    if lat_slice=='lat':
        lat_list_d=np.array(var_delimited_r.lat)
        lon_list_d=np.array(var_delimited_r.lon)
    else:
        lat_list_d=np.array(var_delimited_r.latitude)
        lon_list_d=np.array(var_delimited_r.longitude)
    
    #defining the grid size
    dx_data=np.round(abs(lon_list_d[1])-abs(lon_list_d[0]),2)
    dy_data=np.round(abs(lat_list_d[-1:][0])-  abs(lat_list_d[-2:][0]),2)


    #Obtaining the climatology
    var_clim_r=var_levels_r.groupby('time.month').mean('time')

    var_r_arr=np.array(var_levels_r)

    var_r_clim_arr=np.array(var_clim_r)
    var_r_clim_arr=np.tile(var_r_clim_arr,(n_years,1,1))

    #Obtaining the monthly anomalies 
    var_anom_r= var_r_arr - var_r_clim_arr


    #---------------------------------------------------------------------------------------------
    #Generating the linnear regression 
    slope_matrix=np.empty((var_anom_r.shape[1],var_anom_r.shape[2]))

    for h in range(len(lat_list_d)):
        for g in range(len(lon_list_d)):

            anom_serie=var_anom_r[:,h,g]

            slope, intercept, r, p, se = stats.linregress(std_anom, anom_serie)

            slope_matrix[h,g]=slope


    ###############################################################################################
    #----------------------------------------------------------------------------------------------
    #3. Calculating the power spectrum 

    n        = 150 
    alpha    = 0.5 
    noverlap = 75 
    nfft     = 256 #default value 
    fs       = 1   #default value 
    win      = signal.tukey(n, alpha)
    ssta     = std_anom.reshape(len(std_anom)) # convert vector

    f1, pxx1  = signal.welch(ssta, nfft=nfft, fs=fs, window=win, noverlap=noverlap)
    #f1, pxx1  = signal.welch(ssta)

    # process frequencies and psd
    pxx1 = pxx1/np.max(pxx1) # noralize the psd values    


    return std_anom, slope_matrix, lat_list_d, lon_list_d, f1, pxx1, dx_data, dy_data

def interpolation_fields_ENSO(var_array_ref,lat_refe,lon_refe,lat_mo,lon_mo,dx_mo,dy_mo):
    xnew_ref=np.arange(lon_refe[0],lon_refe[-1:][0], dx_mo)
   
    ynew_ref=np.arange(lat_refe[0],lat_refe[-1:][0], dy_mo)
    

    #Comparing the lists
    lon_interpolation,lat_interpolation=lat_lon_lengthComparison(lat_mo,\
    lon_mo,ynew_ref,xnew_ref,lat_refe,lon_refe,dx_mo,dy_mo)

    #Performing the interpolation of the reference data to the grid size of the
    #model
    var_ref_interpolated=data_interpolation(var_array_ref,'2D',lon_refe,\
    lat_refe,lon_interpolation,lat_interpolation)

    return var_ref_interpolated

def power_spectrum_plot(axs,freq_ref, power_ref, freq_models,power_models,list_models,title_subplot,legend_font,title_font2,label_font,ticks_font):

    axs.set_title(title_subplot,fontsize=title_font2,loc='left')

    axs.plot(1.0/freq_ref[1:]/12, power_ref[1:],color='k', linewidth=2,label='ERA5')
    colors=iter(cm.rainbow(np.linspace(0,1,len(list_models))))

    for m in range(len(list_models)):
        
        labels_plots=list_models[m]
        
        c=next(colors)

        axs.plot(1.0/freq_models[m,1:]/12, power_models[m,1:],color=c, linewidth=1.5,label=labels_plots)
    # adjust spines
    axs = plt.gca()
    axs.spines['top'].set_color('none')
    axs.spines['right'].set_color('none')
    axs.xaxis.set_ticks_position('bottom')
    axs.spines['bottom'].set_position(('data',0))
    plt.xlabel('Years', size=label_font)
    plt.ylabel('Power spectrum', size=label_font)
    plt.xlim(0.9,12)
    plt.xticks([1,2,3,4,5,6,7,8,9,10],['1','','3','','5','','7','','','10'], size=ticks_font)
    plt.legend( bbox_to_anchor=(0.7, -0.5), loc='lower right', fontsize=str(legend_font),ncol=5,frameon=False)

def plotCells_scatter(axs,cell_data,horPlot,pressPlot,colorMap,limits,xlabel,labels_x_cross,step_hor, title_label,y_label_status,title_font2,label_font, ticks_font,scatter_status,points_scatter):
    """
    This function creates the maps of the seasonal regional circulations
    """
    axs.invert_yaxis()
    axs.set_title(title_label,fontsize=title_font2,loc='left')
    cs=axs.contourf(horPlot,pressPlot,cell_data,limits,cmap=colorMap,extend='both')

    if scatter_status=='yes':
        for x in range(len(horPlot)):
            for y in range(len(pressPlot)):
                axs.scatter(horPlot[x],pressPlot[y],points_scatter[y,x]*3,zorder=2, c='grey')
                #axs.scatter(x, y ,points_scatter*3,zorder=2, c='grey')

    axs.set_xlabel(xlabel,fontsize=label_font)
    if y_label_status=='yes':
        axs.set_ylabel('Pressure [hPa]',fontsize=label_font)
    axs.set_xticks(horPlot[::step_hor])
    axs.set_yticks(pressPlot)
    axs.set_yticklabels(pressPlot,fontsize=ticks_font)
    axs.set_xticklabels(labels_x_cross[::step_hor],fontsize=ticks_font)

    return cs

def power_freq_ensamble(path_entry,list_models, str_1,str_2,list_len):

    str1_arr=np.empty((len(list_models),list_len))
    str2_arr=np.empty((len(list_models),list_len))

    for y in range(len(list_models)):
        var_1=np.load(path_entry+list_models[y]+str_1)['arr_0']
        var_2=np.load(path_entry+list_models[y]+str_2)['arr_0']

        str1_arr[y,:]=var_1
        str2_arr[y,:]=var_2

    return str1_arr, str2_arr



with open(path_save+'Models_var_availability.pkl', 'rb') as fp:
    dict_models = pickle.load(fp)

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
cmap_w200=cmap_plot[6]
cmap_w850=cmap_plot[6]
cmap_regcell=cmap_plot[3]
cmap_flux=cmap_plot[3]

#Hadley cell 
lat_limits_Had=[-40,40]
lon_limits_Had=[-70,-50]
#Walker cell
lat_limits_Wal=[-15,15]
lon_limits_Wal=[-150,-20]

Lat_had_common=np.arange(lat_limits_Had[0],lat_limits_Had[1],dy_common)
Lon_wal_common=np.arange(lon_limits_Wal[0],lon_limits_Wal[1],dx_common)

#Fluxes
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


#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#ENSO CALCULATIONS 
#ERA5
try:
    ninno3_era5, slope_matrix_era5, lat_list_era5, lon_list_era5, f_era5, p_era5, dx_era5, dy_era5=ENSO_calculations(path_save,\
                            None,'ERA5',lat_regre,lon_regre,ninno3_lat,ninno3_lon)

    #Obtaining the reference standar deviation 
    std_ref_enso=np.nanstd(slope_matrix_era5)
    new_row_reference={'Characteristic': 'ENSO',\
    'std_DJF':std_ref_enso, 'std_JJA':std_ref_enso, 'std_MAM':std_ref_enso, 'std_SON':std_ref_enso}

    reference_std_DT=pd.read_csv(path_save+'reference_std_original.csv',\
    index_col=[0])
    reference_std_DT=reference_std_DT.append(new_row_reference, ignore_index=True)
    reference_std_DT.to_csv(path_save+'reference_std_original.csv')

    np.savez_compressed(path_save+'ENSO_frequency_ERA5.npz',f_era5)
    np.savez_compressed(path_save+'ENSO_power_ERA5.npz',p_era5)
    np.savez_compressed(path_save+'ERA5_ENSO_fields.npz',slope_matrix_era5)
    np.savez_compressed(path_save+'ERA5_ENSO_fields_Lon.npz',lon_list_era5)
    np.savez_compressed(path_save+'ERA5_ENSO_fields_Lat.npz',lat_list_era5)
except:
    print('Error ERA5 ENSO')

#Models
models=list(dict_models['tos'])
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#Generating the empty arrays to save the information 
models_app=np.array([])

taylor_diagram_metrics=pd.DataFrame(columns=['Model','corr_DJF','std_DJF'])

taylor_diagram_metrics.to_csv(path_save+'taylorDiagram_metrics_ENSO.csv')
np.savez_compressed(path_save+'ENSO_fields_models_N.npz',models_app)

ref_matrix=np.load(path_save+'ERA5_ENSO_fields.npz')['arr_0']
ref_lat=np.load(path_save+'ERA5_ENSO_fields_Lat.npz')['arr_0']
ref_lon=np.load(path_save+'ERA5_ENSO_fields_Lon.npz')['arr_0']

for p in range(len(models)):

    try:

        ninno3_m, slope_matrix_m, lat_list_m, lon_list_m, f_m, p_m, dx_m, dy_m=ENSO_calculations(path_save,\
                            models[p],'model',lat_regre,lon_regre,ninno3_lat,ninno3_lon)


        #--------------------------------------------------------------------------------------------
        #Interpolating ERA5 to the model's gridsize
        era5_field_interp=interpolation_fields_ENSO(ref_matrix,ref_lat,ref_lon,\
                                                    lat_list_m, lon_list_m,dx_m, dy_m)

        corr_m=ma.corrcoef(ma.masked_invalid(era5_field_interp.flatten()), \
                ma.masked_invalid(slope_matrix_m.flatten()))[0,1]
        std_model=np.nanstd(slope_matrix_m)


        #Model's interpolation to a common gridsize
        var_array_model=NaNs_interp(slope_matrix_m,'2D', 'nearest')
                    
        model_field_interp=interpolation_fields_ENSO(var_array_model,lat_list_m, lon_list_m,Lat_common_tos,Lon_common_tos,dx_common_tos, dy_common_tos)

        #-----------------------------------------------------------------------------------------------------------------------
        #Saving the npz 

        models_enso_calc=np.load(path_save+'ENSO_fields_models_N.npz',allow_pickle=True)['arr_0']

        models_enso_calc=np.append(models_enso_calc,models[p])

        np.savez_compressed(path_save+models[p]+'_ENSO_MMM_meanFields.npz',model_field_interp)
        np.savez_compressed(path_save+'ENSO_fields_models_N.npz',models_enso_calc)
        np.savez_compressed(path_save+models[p]+'_ENSO_frequency.npz',f_m)
        np.savez_compressed(path_save+models[p]+'_ENSO_power.npz',p_m)

        #Saving the performance metrics 
        taylor_diagram_metrics_DT=pd.read_csv(path_save+'taylorDiagram_metrics_ENSO.csv', index_col=[0])

        newRow_metrics={'Model':models[p],'corr_DJF': corr_m,'std_DJF':std_model}

        taylor_diagram_metrics_DT=taylor_diagram_metrics_DT.append(newRow_metrics,\
        ignore_index=True)

        taylor_diagram_metrics_DT.to_csv(path_save+'taylorDiagram_metrics_ENSO.csv')
    
    except:
        print('Error ENSO ', models[p])
                
#----------------------------------------------------------------------------------------------------------
print('Calculations - Finished')

len_series=len(f_era5)

try:
    #Creating the plot 
    #-----------------------------------------------------------------------------------------------------------
    #input
    models_calc=np.load(path_save+'ENSO_fields_models_N.npz',allow_pickle=True)['arr_0']
    enso_frequency_models,enso_power_models=power_freq_ensamble(path_save,models_calc, '_ENSO_frequency.npz','_ENSO_power.npz',len_series)
    enso_power_ref=np.load(path_save+'ENSO_power_ERA5.npz',allow_pickle=True)['arr_0']
    enso_frequency_ref=np.load(path_save+'ENSO_frequency_ERA5.npz',allow_pickle=True)['arr_0']

    ################################################################################
    #Obtaining the ensamble of the fields
    mmm_models=np.empty((len(models_calc),len(Lat_common_tos),len(Lon_common_tos)))

    for t in range(len(models_calc)):
        field_model=np.load(path_save+models_calc[t]+'_ENSO_MMM_meanFields.npz')['arr_0']

        mmm_models[t,:,:]=field_model

    mean_field_models=np.nanmean(mmm_models,axis=0)

    #---------------------------------------------------------------------------------
    #obtaining the performance of the series of the power spectrum 
    series_metrics(enso_power_ref,enso_power_models,models_calc,'ENSO_power_spectrum',path_save)

    ################################################################################
    #PLOT
    models_metrics=pd.read_csv(path_save+'taylorDiagram_metrics_ENSO.csv', index_col=[0])

    ref_std=pd.read_csv(path_save+'reference_std_original.csv',index_col=[0])

    plot_label='SSTA [°C]'
    limits_var=np.arange(-2.,2.1,0.1)

    fig=plt.figure(figsize=(13,13))
    colorbar_attributes=[0.027, 0.57,  0.02,0.27]

    lon2D, lat2D = np.meshgrid(Lon_common_tos, Lat_common_tos)
    projection=ccrs.PlateCarree()
    extent = [min(Lon_common_tos),max(Lon_common_tos),min(Lat_common_tos),max(Lat_common_tos)]

    ax1 = fig.add_subplot(2, 2, 1, projection=projection)
    cs=plotMap(ax1,mean_field_models,lon2D,lat2D,cmap_bias,limits_var,'a.',extent, projection,title_str_size,'no',None,'yes',None)

    taylor=td_plots(fig,'DJF',ref_std,models_metrics,'ENSO',len(models_calc),222,'b.',title_str_size,'no',None)

    ax3=fig.add_subplot(2, 2, 3)
    power_spectrum_plot(ax3,enso_frequency_ref, enso_power_ref, enso_frequency_models,enso_power_models,models_calc,'c. ',legends_str,title_str_size,xy_label_str,tick_labels_str)

    ax3.set_position([0.22,0.27,0.55,0.2])

    ax5 = fig.add_subplot(2, 2, 4)
    ax5.set_visible(False)

    cbar_ax = fig.add_axes(colorbar_attributes)
    cb = fig.colorbar(cs,cax=cbar_ax, orientation="vertical",location='left')
    cb.ax.tick_params(labelsize=tick_labels_str)
    cb.set_label(plot_label,fontsize=xy_label_str)

    #fig.subplots_adjust(hspace=0.3)
    plt.savefig(path_save_plots+'ENSO.png', \
    format = 'png', bbox_inches='tight')
    plt.close()

except:
    print('ENSO plot error ')

    

#-------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------
#SECOND PART OF THE CODE 
#PLOTTING THE CROSS SECTION WITH THE POINT OF THE AGREEMENT 

#-------------------------------------------------------------------------------------------------------------

list_calculation=['Regional_cells','qu_qv','tu_tv']

for i in range(len(list_calculation)):

    if list_calculation[i]=='Regional_cells':

        try:

            #input
            models=np.load(path_save+'regionalcells_models_N.npz',allow_pickle=True)['arr_0']

            hadley_mmm=seasonal_ensamble(models,path_save,'regCirc_hadleycell',len(Lat_had_common),len(p_level_common))

            walker_mmm=seasonal_ensamble(models,path_save,'regCirc_walkercell',len(Lon_wal_common),len(p_level_common))

            hadley_mmm_bias=seasonal_ensamble(models,path_save,'regCirc_hadleycell_bias',len(Lat_had_common),len(p_level_common))

            walker_mmm_bias=seasonal_ensamble(models,path_save,'regCirc_walkercell_bias',len(Lon_wal_common),len(p_level_common))

            bias_mmm_agreement_had=agreement_sign(models,path_save,'regCirc_hadleycell_bias',\
                                            len(Lat_had_common),len(p_level_common))
            
            bias_mmm_agreement_wal=agreement_sign(models,path_save,'regCirc_walkercell_bias',\
                                            len(Lon_wal_common),len(p_level_common))

            models_metrics_h=pd.read_csv(path_save+'taylorDiagram_metrics_hadleycell.csv', index_col=[0])

            models_metrics_w=pd.read_csv(path_save+'taylorDiagram_metrics_walkercell.csv', index_col=[0])

            ref_std=pd.read_csv(path_save+'reference_std_original.csv',index_col=[0])

            #--------------------------------------------------------------------------------------------------------------------------
            #--------------------------------------------------------------------------------------------------------------------------
            plot_label='10⁹ [Kg/s]'
            limits_var=np.arange(-11,12,1)
            limits_bias=np.arange(-6,7,1)

            #Creating the input to the plots
            p_level_plot=[]
            for v in range(len(p_level_common)):
                str_level=str(np.round(p_level_common[v]/100,0)).replace('.0','')
                p_level_plot.append(str_level)


            lat_plot=[]
            for v in range(len(Lat_had_common)):
                if Lat_had_common[v]<0:
                    lat_str=str(np.abs(np.round(Lat_had_common[v],0))).replace('.0','')+'°S'
                elif Lat_had_common[v]>0:
                    lat_str=str(np.abs(np.round(Lat_had_common[v],0))).replace('.0','')+'°N'
                else:
                    lat_str=str(np.abs(np.round(Lat_had_common[v],0))).replace('.0','')

                lat_plot.append(lat_str)

            lon_plot=[]
            for v in range(len(Lon_wal_common)):
                lon_str=str(np.abs(np.round(Lon_wal_common[v],0))).replace('.0','')+'°W'
                lon_plot.append(lon_str)

            #------------------------------------------------------------------------------------------------------------------
            #------------------------------------------------------------------------------------------------------------------
            #Creating the plot for the Hadley cell 

            cmap_plot=['gist_rainbow_r','terrain','rainbow','RdBu','RdBu_r']

            fig=plt.figure(figsize=(24,17))
            colorbar_attributes=[0.92, 0.37,  0.017,0.24]

            colorbar_attributes_bias=[0.92, 0.1,  0.017,0.24]

            taylor=td_plots(fig,'DJF',ref_std,models_metrics_h,'HadCell',len(models),341,'a.',title_str_size,'no',None)

            taylor=td_plots(fig,'MAM',ref_std,models_metrics_h,'HadCell',len(models),342,'b.',title_str_size,'no',None)

            taylor=td_plots(fig,'JJA',ref_std,models_metrics_h,'HadCell',len(models),343,'c.',title_str_size,'no',None)

            taylor=td_plots(fig,'SON',ref_std,models_metrics_h,'HadCell',len(models),344,'d.',title_str_size,'no',None)

            ax5 = fig.add_subplot(3, 4, 5)
            cs=plotCells(ax5,(hadley_mmm[0].T)/10e9,np.flip(Lat_had_common),p_level_plot,cmap_regcell,limits_var,\
                        'Latitude',np.flip(lat_plot),12, 'e. ','yes',title_str_size,xy_label_str, tick_labels_str_regcell,'no',None)
            
            ax6 = fig.add_subplot(3, 4, 6)
            cs=plotCells(ax6,(hadley_mmm[2].T)/10e9,np.flip(Lat_had_common),p_level_plot,cmap_regcell,limits_var,\
                        'Latitude',np.flip(lat_plot),12, 'f. ','no',title_str_size,xy_label_str, tick_labels_str_regcell,'no',None)
            
            ax7 = fig.add_subplot(3, 4, 7)
            cs=plotCells(ax7,(hadley_mmm[1].T)/10e9,np.flip(Lat_had_common),p_level_plot,cmap_regcell,limits_var,\
                        'Latitude',np.flip(lat_plot),12, 'g. ','no',title_str_size,xy_label_str, tick_labels_str_regcell,'no',None)
            
            ax8 = fig.add_subplot(3, 4, 8)
            cs=plotCells(ax8,(hadley_mmm[3].T)/10e9,np.flip(Lat_had_common),p_level_plot,cmap_regcell,limits_var,\
                        'Latitude',np.flip(lat_plot),12, 'h. ','no',title_str_size,xy_label_str, tick_labels_str_regcell,'no',None)
            
            ax9 = fig.add_subplot(3, 4, 9)
            csb=plotCells_scatter(ax9,(hadley_mmm_bias[0].T)/10e9,np.flip(Lat_had_common),p_level_plot,cmap_bias,limits_bias,\
                        'Latitude',np.flip(lat_plot),12, 'i. ','yes',title_str_size,xy_label_str, tick_labels_str_regcell,'yes',bias_mmm_agreement_had[0].T)
            
            ax10 = fig.add_subplot(3, 4, 10)
            csb=plotCells_scatter(ax10,(hadley_mmm_bias[2].T)/10e9,np.flip(Lat_had_common),p_level_plot,cmap_bias,limits_bias,\
                        'Latitude',np.flip(lat_plot),12, 'j. ','no',title_str_size,xy_label_str, tick_labels_str_regcell,'yes',bias_mmm_agreement_had[2].T)
            
            ax11 = fig.add_subplot(3, 4, 11)
            csb=plotCells_scatter(ax11,(hadley_mmm_bias[1].T)/10e9,np.flip(Lat_had_common),p_level_plot,cmap_bias,limits_bias,\
                        'Latitude',np.flip(lat_plot),12, 'k. ','no',title_str_size,xy_label_str, tick_labels_str_regcell,'yes',bias_mmm_agreement_had[1].T)
            
            ax12 = fig.add_subplot(3, 4, 12)
            csb=plotCells_scatter(ax12,(hadley_mmm_bias[3].T)/10e9,np.flip(Lat_had_common),p_level_plot,cmap_bias,limits_bias,\
                        'Latitude',np.flip(lat_plot),12, 'l. ','no',title_str_size,xy_label_str, tick_labels_str_regcell,'yes',bias_mmm_agreement_had[3].T)

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
            plt.savefig(path_save_plots+'HadleyCell_fields.png', \
            format = 'png', bbox_inches='tight')
            plt.close()


            #Creating the plot for the Walker cell 
            cmap_plot=['gist_rainbow_r','terrain','rainbow','RdBu','RdBu_r']

            fig=plt.figure(figsize=(24,17))
            colorbar_attributes=[0.92, 0.37,  0.017,0.24]

            colorbar_attributes_bias=[0.92, 0.1,  0.017,0.24]

            taylor=td_plots(fig,'DJF',ref_std,models_metrics_w,'WalCell',len(models),341,'a.',title_str_size,'no',None)

            taylor=td_plots(fig,'MAM',ref_std,models_metrics_w,'WalCell',len(models),342,'b.',title_str_size,'no',None)

            taylor=td_plots(fig,'JJA',ref_std,models_metrics_w,'WalCell',len(models),343,'c.',title_str_size,'no',None)

            taylor=td_plots(fig,'SON',ref_std,models_metrics_w,'WalCell',len(models),344,'d.',title_str_size,'no',None)

            ax5 = fig.add_subplot(3, 4, 5)
            cs=plotCells(ax5,(walker_mmm[0].T)/10e9,Lon_wal_common,p_level_plot,cmap_regcell,limits_var,'Longitude',\
                        lon_plot,16, 'e. ','yes',title_str_size,xy_label_str, tick_labels_str_regcell,'no',None)
            
            ax6 = fig.add_subplot(3, 4, 6)
            cs=plotCells(ax6,(walker_mmm[2].T)/10e9,Lon_wal_common,p_level_plot,cmap_regcell,limits_var,'Longitude',\
                        lon_plot,16, 'f. ','no',title_str_size,xy_label_str, tick_labels_str_regcell,'no',None)
            
            ax7 = fig.add_subplot(3, 4, 7)
            cs=plotCells(ax7,(walker_mmm[1].T)/10e9,Lon_wal_common,p_level_plot,cmap_regcell,limits_var,'Longitude',\
                        lon_plot,16, 'g. ','no',title_str_size,xy_label_str, tick_labels_str_regcell,'no',None)
            
            ax8 = fig.add_subplot(3, 4, 8)
            cs=plotCells(ax8,(walker_mmm[3].T)/10e9,Lon_wal_common,p_level_plot,cmap_regcell,limits_var,'Longitude',\
                        lon_plot,16, 'h. ','no',title_str_size,xy_label_str, tick_labels_str_regcell,'no',None)
            
            ax9 = fig.add_subplot(3, 4, 9)
            csb=plotCells_scatter(ax9,(walker_mmm_bias[0].T)/10e9,Lon_wal_common,p_level_plot,cmap_bias,limits_bias,'Longitude',\
                        lon_plot,16, 'i. ','yes',title_str_size,xy_label_str, tick_labels_str_regcell,'yes',bias_mmm_agreement_wal[0].T)
            
            ax10 = fig.add_subplot(3, 4, 10)
            csb=plotCells_scatter(ax10,(walker_mmm_bias[2].T)/10e9,Lon_wal_common,p_level_plot,cmap_bias,limits_bias,'Longitude',\
                        lon_plot,16, 'j. ','no',title_str_size,xy_label_str, tick_labels_str_regcell,'yes',bias_mmm_agreement_wal[2].T)
            
            ax11 = fig.add_subplot(3, 4, 11)
            csb=plotCells_scatter(ax11,(walker_mmm_bias[1].T)/10e9,Lon_wal_common,p_level_plot,cmap_bias,limits_bias,'Longitude',\
                        lon_plot,16, 'k. ','no',title_str_size,xy_label_str, tick_labels_str_regcell,'yes',bias_mmm_agreement_wal[1].T)
            
            ax12 = fig.add_subplot(3, 4, 12)
            csb=plotCells_scatter(ax12,(walker_mmm_bias[3].T)/10e9,Lon_wal_common,p_level_plot,cmap_bias,limits_bias,'Longitude',\
                        lon_plot,16, 'l. ','no',title_str_size,xy_label_str, tick_labels_str_regcell,'yes',bias_mmm_agreement_wal[3].T)

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
            plt.savefig(path_save_plots+'WalkerCell_fields.png', \
            format = 'png', bbox_inches='tight')
            plt.close()
        
        except:
            print('Error plot regional cells')
    
    elif list_calculation[i]=='qu_qv':

        try:

            #input
            models=np.load(path_save+'qu_qv_models_N.npz',allow_pickle=True)['arr_0']

            var_mmm_north=seasonal_ensamble(models,path_save,'qu_qv_north',len(p_level_common),len(Lon_common_fl))

            var_mmm_south=seasonal_ensamble(models,path_save,'qu_qv_south',len(p_level_common),len(Lon_common_fl))

            var_mmm_east=seasonal_ensamble(models,path_save,'qu_qv_east',len(p_level_common),len(Lat_common_fl))

            var_mmm_west=seasonal_ensamble(models,path_save,'qu_qv_west',len(p_level_common),len(Lat_common_fl))

            var_mmm_bias_north=seasonal_ensamble(models,path_save,'qu_qv_north_bias',len(p_level_common),len(Lon_common_fl))

            var_mmm_bias_south=seasonal_ensamble(models,path_save,'qu_qv_south_bias',len(p_level_common),len(Lon_common_fl))

            var_mmm_bias_east=seasonal_ensamble(models,path_save,'qu_qv_east_bias',len(p_level_common),len(Lat_common_fl))

            var_mmm_bias_west=seasonal_ensamble(models,path_save,'qu_qv_west_bias',len(p_level_common),len(Lat_common_fl))

            bias_mmm_agreement_north=agreement_sign(models,path_save,'qu_qv_north_bias',len(p_level_common),len(Lon_common_fl))

            bias_mmm_agreement_south=agreement_sign(models,path_save,'qu_qv_south_bias',len(p_level_common),len(Lon_common_fl))

            bias_mmm_agreement_east=agreement_sign(models,path_save,'qu_qv_east_bias',len(p_level_common),len(Lat_common_fl))

            bias_mmm_agreement_west=agreement_sign(models,path_save,'qu_qv_west_bias',len(p_level_common),len(Lat_common_fl))

            ################################################################################
            #PLOT
            models_metrics=pd.read_csv(path_save+'taylorDiagram_metrics_qu_qv.csv', index_col=[0])

            ref_std=pd.read_csv(path_save+'reference_std_original.csv',index_col=[0])

            #Creating the input to the plots
            p_level_plot=[]
            for v in range(len(p_level_common)):
                str_level=str(np.round(p_level_common[v]/100,0)).replace('.0','')
                p_level_plot.append(str_level)


            lat_plot=[]
            for v in range(len(Lat_common_fl)):
                if Lat_common_fl[v]<0:
                    lat_str=str(np.round(np.abs(Lat_common_fl[v]),0)).replace('.0','')+'°S'
                elif Lat_common_fl[v]>0:
                    lat_str=str(np.round(np.abs(Lat_common_fl[v]),0)).replace('.0','')+'°N'
                else:
                    lat_str=str(np.round(np.abs(Lat_common_fl[v],0))).replace('.0','')

                lat_plot.append(lat_str)

            lon_plot=[]
            for v in range(len(Lon_common_fl)):
                lon_str=str(np.round(np.abs(Lon_common_fl[v]),0)).replace('.0','')+'°W'
                lon_plot.append(lon_str)

            seasons_labels_i=['DJF','JJA','MAM','SON']

            plot_label='[m * g/ Kg *s]'

            for n in range(4):

                index_season=n
                seasons_labels=seasons_labels_i[index_season]

                cmap_plot=['gist_rainbow_r','terrain','terrain_r','PuOr','RdBu_r']

                fig=plt.figure(figsize=(24,17))
                colorbar_attributes=[0.92, 0.37,  0.017,0.24]

                colorbar_attributes_bias=[0.92, 0.1,  0.017,0.24]

                levels=np.arange(-70,80,10)

                levels_bias=np.arange(-20,22,2)


                taylor=td_plots(fig,seasons_labels,ref_std,models_metrics,'qu_qv',len(models),341,'a.',title_str_size,'yes','north')

                taylor=td_plots(fig,seasons_labels,ref_std,models_metrics,'qu_qv',len(models),342,'b.',title_str_size,'yes','south')

                taylor=td_plots(fig,seasons_labels,ref_std,models_metrics,'qu_qv',len(models),343,'c.',title_str_size,'yes','east')

                taylor=td_plots(fig,seasons_labels,ref_std,models_metrics,'qu_qv',len(models),344,'d.',title_str_size,'yes','west')


                ax5 = fig.add_subplot(3, 4, 5)
                cs=plotCells(ax5,var_mmm_north[index_season],Lon_common_fl,p_level_plot,cmap_flux,levels,'Longitude',lon_plot,10,'e.','yes',\
                            title_str_size,xy_label_str, tick_labels_str,'no',None)

                ax6 = fig.add_subplot(3, 4, 6)
                cs=plotCells(ax6,var_mmm_south[index_season],Lon_common_fl,p_level_plot,cmap_flux,levels,'Longitude',lon_plot,10,'f.','no',\
                            title_str_size,xy_label_str, tick_labels_str,'no',None)

                ax7 = fig.add_subplot(3, 4, 7)
                cs=plotCells(ax7,var_mmm_east[index_season],Lat_common_fl,p_level_plot,cmap_flux,levels,'Latitude',lat_plot,13,'g.','no',\
                            title_str_size,xy_label_str, tick_labels_str,'no',None)

                ax8 = fig.add_subplot(3, 4, 8)
                cs=plotCells(ax8,var_mmm_west[index_season],Lat_common_fl,p_level_plot,cmap_flux,levels,'Latitude',lat_plot,13,'h.','no',\
                            title_str_size,xy_label_str, tick_labels_str,'no',None)


                ax9 = fig.add_subplot(3, 4, 9)
                csb=plotCells_scatter(ax9,var_mmm_bias_north[index_season],Lon_common_fl,p_level_plot,cmap_bias,levels_bias,'Longitude',lon_plot,10,'i.','yes',\
                            title_str_size,xy_label_str, tick_labels_str,'yes',bias_mmm_agreement_north[index_season])


                ax10 = fig.add_subplot(3, 4, 10)
                csb=plotCells_scatter(ax10,var_mmm_bias_south[index_season],Lon_common_fl,p_level_plot,cmap_bias,levels_bias,'Longitude',lon_plot,10,'j.','no',\
                            title_str_size,xy_label_str, tick_labels_str,'yes',bias_mmm_agreement_south[index_season])


                ax11 = fig.add_subplot(3, 4, 11)
                csb=plotCells_scatter(ax11,var_mmm_bias_east[index_season],Lat_common_fl,p_level_plot,cmap_bias,levels_bias,'Latitude',lat_plot,13,'k.','no',\
                            title_str_size,xy_label_str, tick_labels_str,'yes',bias_mmm_agreement_east[index_season])

                ax12 = fig.add_subplot(3, 4, 12)
                csb=plotCells_scatter(ax12,var_mmm_bias_west[index_season],Lat_common_fl,p_level_plot,cmap_bias,levels_bias,'Latitude',lat_plot,13,'l.','no',\
                            title_str_size,xy_label_str, tick_labels_str,'yes',bias_mmm_agreement_west[index_season])

                cbar_ax = fig.add_axes(colorbar_attributes)
                cb = fig.colorbar(cs,cax=cbar_ax, orientation="vertical")
                cb.ax.tick_params(labelsize=tick_labels_str)
                cb.set_label(plot_label,fontsize=xy_label_str)

                cbar_ax_b = fig.add_axes(colorbar_attributes_bias)
                cbb = fig.colorbar(csb,cax=cbar_ax_b, orientation="vertical")
                cbb.ax.tick_params(labelsize=tick_labels_str)
                cbb.set_label(plot_label,fontsize=xy_label_str)

                plt.text(0.3,2.42,'Northern', fontsize=title_str_size,rotation='horizontal',transform=ax5.transAxes)
                plt.text(0.3,2.42,'Southern', fontsize=title_str_size,rotation='horizontal',transform=ax6.transAxes)
                plt.text(0.3,2.42,'Eastern', fontsize=title_str_size,rotation='horizontal',transform=ax7.transAxes)
                plt.text(0.3,2.42,'Western', fontsize=title_str_size,rotation='horizontal',transform=ax8.transAxes)
                #fig.subplots_adjust(hspace=0.3)
                plt.savefig(path_save_plots+'qu_qv_'+seasons_labels+'.png', \
                format = 'png', bbox_inches='tight')
                plt.close()

        except:
            print('Error plot qu_qv')
    
    elif list_calculation[i]=='tu_tv':

        try:

            #input
            models=np.load(path_save+'tu_tv_models_N.npz',allow_pickle=True)['arr_0']

            var_mmm_north=seasonal_ensamble(models,path_save,'tu_tv_north',len(p_level_common),len(Lon_common_fl))

            var_mmm_south=seasonal_ensamble(models,path_save,'tu_tv_south',len(p_level_common),len(Lon_common_fl))

            var_mmm_east=seasonal_ensamble(models,path_save,'tu_tv_east',len(p_level_common),len(Lat_common_fl))

            var_mmm_west=seasonal_ensamble(models,path_save,'tu_tv_west',len(p_level_common),len(Lat_common_fl))

            var_mmm_bias_north=seasonal_ensamble(models,path_save,'tu_tv_north_bias',len(p_level_common),len(Lon_common_fl))

            var_mmm_bias_south=seasonal_ensamble(models,path_save,'tu_tv_south_bias',len(p_level_common),len(Lon_common_fl))

            var_mmm_bias_east=seasonal_ensamble(models,path_save,'tu_tv_east_bias',len(p_level_common),len(Lat_common_fl))

            var_mmm_bias_west=seasonal_ensamble(models,path_save,'tu_tv_west_bias',len(p_level_common),len(Lat_common_fl))

            bias_mmm_agreement_north=agreement_sign(models,path_save,'tu_tv_north_bias',len(p_level_common),len(Lon_common_fl))

            bias_mmm_agreement_south=agreement_sign(models,path_save,'tu_tv_south_bias',len(p_level_common),len(Lon_common_fl))

            bias_mmm_agreement_east=agreement_sign(models,path_save,'tu_tv_east_bias',len(p_level_common),len(Lat_common_fl))

            bias_mmm_agreement_west=agreement_sign(models,path_save,'tu_tv_west_bias',len(p_level_common),len(Lat_common_fl))

            ################################################################################
            #PLOT
            models_metrics=pd.read_csv(path_save+'taylorDiagram_metrics_tu_tv.csv', index_col=[0])

            ref_std=pd.read_csv(path_save+'reference_std_original.csv',index_col=[0])

            #Creating the input to the plots
            p_level_plot=[]
            for v in range(len(p_level_common)):
                str_level=str(np.round(p_level_common[v]/100,0)).replace('.0','')
                p_level_plot.append(str_level)


            lat_plot=[]
            for v in range(len(Lat_common_fl)):
                if Lat_common_fl[v]<0:
                    lat_str=str(np.round(np.abs(Lat_common_fl[v]),0)).replace('.0','')+'°S'
                elif Lat_common_fl[v]>0:
                    lat_str=str(np.round(np.abs(Lat_common_fl[v]),0)).replace('.0','')+'°N'
                else:
                    lat_str=str(np.round(np.abs(Lat_common_fl[v],0))).replace('.0','')

                lat_plot.append(lat_str)

            lon_plot=[]
            for v in range(len(Lon_common_fl)):
                lon_str=str(np.round(np.abs(Lon_common_fl[v]),0)).replace('.0','')+'°W'
                lon_plot.append(lon_str)

            seasons_labels_i=['DJF','JJA','MAM','SON']

            plot_label='[ x 10³ K * m/ s]'

            for n in range(4):

                index_season=n
                seasons_labels=seasons_labels_i[index_season]

                cmap_plot=['gist_rainbow_r','terrain','terrain_r','PuOr','RdBu_r']

                fig=plt.figure(figsize=(24,17))
                colorbar_attributes=[0.92, 0.37,  0.017,0.24]

                colorbar_attributes_bias=[0.92, 0.1,  0.017,0.24]

                levels=np.arange(-5,5.5,0.5)

                levels_bias=np.arange(-2,2.1,0.1)


                taylor=td_plots(fig,seasons_labels,ref_std,models_metrics,'tu_tv',len(models),341,'a.',title_str_size,'yes','north')

                taylor=td_plots(fig,seasons_labels,ref_std,models_metrics,'tu_tv',len(models),342,'b.',title_str_size,'yes','south')

                taylor=td_plots(fig,seasons_labels,ref_std,models_metrics,'tu_tv',len(models),343,'c.',title_str_size,'yes','east')

                taylor=td_plots(fig,seasons_labels,ref_std,models_metrics,'tu_tv',len(models),344,'d.',title_str_size,'yes','west')


                ax5 = fig.add_subplot(3, 4, 5)
                cs=plotCells(ax5,var_mmm_north[index_season]/1000,Lon_common_fl,p_level_plot,cmap_flux,levels,'Longitude',lon_plot,10,'e.','yes',\
                            title_str_size,xy_label_str, tick_labels_str,'no',None)

                ax6 = fig.add_subplot(3, 4, 6)
                cs=plotCells(ax6,var_mmm_south[index_season]/1000,Lon_common_fl,p_level_plot,cmap_flux,levels,'Longitude',lon_plot,10,'f.','no',\
                            title_str_size,xy_label_str, tick_labels_str,'no',None)

                ax7 = fig.add_subplot(3, 4, 7)
                cs=plotCells(ax7,var_mmm_east[index_season]/1000,Lat_common_fl,p_level_plot,cmap_flux,levels,'Latitude',lat_plot,13,'g.','no',\
                            title_str_size,xy_label_str, tick_labels_str,'no',None)

                ax8 = fig.add_subplot(3, 4, 8)
                cs=plotCells(ax8,var_mmm_west[index_season]/1000,Lat_common_fl,p_level_plot,cmap_flux,levels,'Latitude',lat_plot,13,'h.','no',\
                            title_str_size,xy_label_str, tick_labels_str,'no',None)


                ax9 = fig.add_subplot(3, 4, 9)
                csb=plotCells_scatter(ax9,var_mmm_bias_north[index_season]/1000,Lon_common_fl,p_level_plot,cmap_bias,levels_bias,'Longitude',lon_plot,10,'i.','yes',\
                            title_str_size,xy_label_str, tick_labels_str,'yes',bias_mmm_agreement_north[index_season])


                ax10 = fig.add_subplot(3, 4, 10)
                csb=plotCells_scatter(ax10,var_mmm_bias_south[index_season]/1000,Lon_common_fl,p_level_plot,cmap_bias,levels_bias,'Longitude',lon_plot,10,'j.','no',\
                            title_str_size,xy_label_str, tick_labels_str,'yes',bias_mmm_agreement_south[index_season])


                ax11 = fig.add_subplot(3, 4, 11)
                csb=plotCells_scatter(ax11,var_mmm_bias_east[index_season]/1000,Lat_common_fl,p_level_plot,cmap_bias,levels_bias,'Latitude',lat_plot,13,'k.','no',\
                            title_str_size,xy_label_str, tick_labels_str,'yes',bias_mmm_agreement_east[index_season])

                ax12 = fig.add_subplot(3, 4, 12)
                csb=plotCells_scatter(ax12,var_mmm_bias_west[index_season]/1000,Lat_common_fl,p_level_plot,cmap_bias,levels_bias,'Latitude',lat_plot,13,'l.','no',\
                            title_str_size,xy_label_str, tick_labels_str,'yes',bias_mmm_agreement_west[index_season])

                cbar_ax = fig.add_axes(colorbar_attributes)
                cb = fig.colorbar(cs,cax=cbar_ax, orientation="vertical")
                cb.ax.tick_params(labelsize=tick_labels_str)
                cb.set_label(plot_label,fontsize=xy_label_str)

                cbar_ax_b = fig.add_axes(colorbar_attributes_bias)
                cbb = fig.colorbar(csb,cax=cbar_ax_b, orientation="vertical")
                cbb.ax.tick_params(labelsize=tick_labels_str)
                cbb.set_label(plot_label,fontsize=xy_label_str)

                plt.text(0.3,2.42,'Northern', fontsize=title_str_size,rotation='horizontal',transform=ax5.transAxes)
                plt.text(0.3,2.42,'Southern', fontsize=title_str_size,rotation='horizontal',transform=ax6.transAxes)
                plt.text(0.3,2.42,'Eastern', fontsize=title_str_size,rotation='horizontal',transform=ax7.transAxes)
                plt.text(0.3,2.42,'Western', fontsize=title_str_size,rotation='horizontal',transform=ax8.transAxes)
                #fig.subplots_adjust(hspace=0.3)
                plt.savefig(path_save_plots+'tu_tv_'+seasons_labels+'.png', \
                format = 'png', bbox_inches='tight')
                plt.close()
        
        except:
            print('Error plot tu_tv')


print('############################################')
print('Plots Finished')
print('############################################')


#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
#THIRD PART OF THE CODE: MODEL'S AVAILABILITY OF VARIABLE WAP AND CREATION OF THE NETCDFS OF THAT VARIABLE
#----------------------------------------------------------------------------------------------------------
#Generating the netcdfs for variable wap 

path_entry_m='/bdd/CMIP6/CMIP/'
Ensemble='r1i1p1f1'
dom_str='Amon'

list_variables='wap'

dict_models_wap={}

models_var_sp=variables_availability(path_entry_m,list_variables,dom_str)

dict_models_wap[list_variables]=models_var_sp[:]

with open(path_save+'Models_var_availability_wap.pkl', 'wb') as fp:
    pickle.dump(dict_models_wap, fp)

#---------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#Create the netcdf with the original gridsize of each model and extracting the grid size of each 
#model 
list_variables='wap'

with open(path_save+'Models_var_availability_wap.pkl', 'rb') as fp:
    dict_models_wap = pickle.load(fp)

lat_limits_full=[-90,90]
lon_limits_full=[0,360]

initial_time='19790101'
final_time='20141201'

p_level_interest_upper=10000.0
p_level_interest_lower=100000.0

path_entry_m='/bdd/CMIP6/CMIP/'
Ensemble='r1i1p1f1'

var_sp=list_variables

models_ava_wap=list(dict_models_wap[var_sp])

#---------------------------------------------------------------------------------------------------------------------
list_f1=os.listdir(path_entry_m)

for n in range(len(models_ava_wap)):

    for e in range(len(list_f1)):

        path_folder_sp=path_entry_m+list_f1[e]+'/'

        list_mod_v=os.listdir(path_folder_sp)

        if models_ava_wap[n] in list_mod_v:

            path_entry=path_folder_sp+models_ava_wap[n]+'/'

            try:

                gridsize_x, gridsize_y=netcdf_creation_original(path_entry,var_sp,'Amon',lat_limits_full,lon_limits_full,\
                                                                initial_time,final_time,p_level_interest_lower,p_level_interest_upper,\
                                                                    'model','Yes',path_save,models_ava_wap[n])
                    
            except:
                print('Error: ',var_sp,models_ava_wap[n])
        
        else:
            pass

print('The generation of Original netCDF is ready')

print('##########################################################')
print('Code finished')
print('##########################################################')
