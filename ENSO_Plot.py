"""
Code to create the plot of the ENSO calculations after applying 
some changes on the colors of the legend
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

#path_save='/home/iccorreasa/Documentos/Paper_CMIP6_models/PLOTS_paper/PAPER_FINAL/npz/' #CHANGE
#path_save_plots='/home/iccorreasa/Documentos/Paper_CMIP6_models/PLOTS_paper/PAPER_FINAL/plots/' #CHANGE
path_save = '/scratchx/lfita/'
path_save_plots = '/scratchx/lfita/'

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

        file_data=xr.open_dataset(path_entry+'ERA5_sstk_original_mon_clim_LT.nc')

        data=file_data['sstk']
    
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

    colors=iter(cm.rainbow(np.linspace(0,1,len(list_models))))

    for m in range(len(list_models)):
        
        labels_plots=list_models[m]
        
        c=next(colors)

        axs.plot(1.0/freq_models[m,1:]/12, power_models[m,1:],color=c, linewidth=1.5,label=labels_plots)
    
    axs.plot(1.0/freq_ref[1:]/12, power_ref[1:],color='k', linewidth=2.5,label='ERA5')
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

def td_plots(fig,season_str,ref_table,models_table,characteristic,number_models,rects,title_label,title_size,bound_status,boundary):

    #to obtain the reference standar deviation
    if bound_status=='yes':
        ref_standar=float(ref_table[ref_table['Characteristic']==characteristic+'_'+boundary]['std_'+season_str])
    else:
        ref_standar=float(ref_table[ref_table['Characteristic']==characteristic]['std_'+season_str])

    ############################################################################
    #To obtain the metrics of the models

    if bound_status=='yes':
        models_table=models_table[models_table['Boundary']==boundary]
        models_indices=list(models_table.index)
        sample_models=[]
        for i in range(number_models):
            index_row_sp=models_indices[i]
            new_row=[models_table['std_'+season_str].loc[index_row_sp]/ref_standar,\
            models_table['corr_'+season_str].loc[index_row_sp],models_table['Model'].loc[index_row_sp]]

            #appending the new row
            sample_models.append(new_row)
    else:
        sample_models=[]
        for i in range(number_models):
            new_row=[models_table['std_'+season_str].loc[i]/ref_standar,\
            models_table['corr_'+season_str].loc[i],models_table['Model'].loc[i]]

            #appending the new row
            sample_models.append(new_row)

    #Creating the plot
    list_markers=['o','P','X','D','s','v','^','<','>']
    n_repeat=math.ceil(number_models/len(list_markers))
    list_markers_repeated=list_markers*n_repeat
    markers=list_markers_repeated[0:number_models]

    colors=iter(cm.rainbow(np.linspace(0,1,number_models)))

    dia = TaylorDiagram(ref_standar/ref_standar, fig=fig,rect=rects,label='Reference')


    # Add samples to Taylor diagram
    for i,(stddev,corrcoef,name) in enumerate(sample_models):
        c=next(colors)
        dia.add_sample(stddev, corrcoef,
                       marker=markers[i] , ms=10, ls='',
                       #mfc='k', mec='k', # B&W
                       mfc=c, mec=c, # Colors
                       label=name)

    # Add RMS contours, and label them
    contours = dia.add_contours(levels=5, colors='0.5') # 5 levels
    dia.ax.clabel(contours, inline=1, fontsize=10, fmt='%.1f')
    # Tricky: ax is the polar ax (used for plots), _ax is the
    # container (used for layout)
    dia._ax.set_title(title_label,fontsize=title_size,loc='left', pad=20)

    nrows = 16
    ncols = int(np.ceil(number_models / float(nrows)))


    #fig.legend(dia.samplePoints,
    #           [ p.get_label() for p in dia.samplePoints ],
    #           numpoints=1, prop=dict(size='small'),bbox_to_anchor=(1.11, 0.85) \
    #           ,ncol=2,loc='right')

    #fig.tight_layout()


    return dia

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

     
except Exception as e:
    print('Error ERA5 ENSO')
    print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")

#Models
models=list(dict_models['tos'])
#---------------------------------------------------------------------------------------                
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

    #Obtaining the agreement of the sign of the slope
    mmm_agreement_enso=agreement_sign_ENSO(models_calc,path_save,'ENSO_MMM_meanFields',\
                                            len(Lat_common_tos),len(Lon_common_tos))


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
    cs=plotMap(ax1,mean_field_models,lon2D,lat2D,cmap_bias,limits_var,'a.',extent, projection,title_str_size,'yes',mmm_agreement_enso,'yes')

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

    #---------------------------------------------------------------------
    
    #To save the legend independently Taylor diagram

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
    path_save_plots+'ENSO_taylor_legend.png', format = 'png',\
    bbox_inches='tight',bbox_extra_artists=[legend_squared],
    ) 

    #----------------------------------------------------------------------------------------------------
    #To save the legend of the power spectrum plot 
    legend=ax3.legend( bbox_to_anchor=(0.7, -1), loc='lower right', fontsize=str(legends_str),ncol=4,frameon=False)

    nrows = 20

    fig.canvas.draw()
    legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
    legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
    legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width, legend_bbox.height))
    legend_squared = legend_ax.legend(
        *ax3.get_legend_handles_labels(), 
        bbox_transform=legend_fig.transFigure,
        bbox_to_anchor=(0,0,1,1),
        frameon=False,
        fancybox=None,
        shadow=False,
        ncol=4,
        mode='expand',
    )
    legend_ax.axis('off')
    legend_fig.savefig(
        path_save_plots+'ENSO_power_legend.png', format = 'png',\
    bbox_inches='tight',bbox_extra_artists=[legend_squared],
    )
    plt.close()

except Exception as e:
    print('ENSO plot error ')
    print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
