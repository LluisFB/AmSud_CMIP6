"""
Code to generate the plots of the calculations from
the set of CMIP6 models 

Part 4 

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
import matplotlib.patches as mpatches
from Functions import *
import pickle
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
sns.set(style="white")
sns.set_context('notebook', font_scale=1.5)

#----------------------------------------------------------------------------------------
#path_entry: path where the npz, csv and.nc files were saved
#path_save= path to save the plots
#CHANGE THE PATHS

#path_entry='/home/iccorreasa/Documentos/Paper_CMIP6_models/PLOTS_paper/PAPER_FINAL/npz/' #CHANGE
#path_save='/home/iccorreasa/Documentos/Paper_CMIP6_models/PLOTS_paper/PAPER_FINAL/plots/' #CHANGE
path_entry='/scratchx/lfita/'
path_save='/scratchx/lfita/'


#Creating the path to save the figures of individual models 

#os.system('mkdir '+path_save+'model_season_ind') 

path_save_ind=path_save+'model_season_ind/'
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#Defining the parameters of the plots 
fig_title_font=25
title_str_size=23
xy_label_str=20
tick_labels_str=18
tick_labels_str_regcell=18
legends_str=16

gridsize_df=pd.read_csv(path_entry+'CMIP6_models_GridSize_lat_lon_Amon.csv', index_col=[0])
gridsize_df_tos=pd.read_csv(path_entry+'CMIP6_models_GridSize_lat_lon_Omon.csv', index_col=[0])


#Finding the grid size to perform the interpolation 
dx_common=gridsize_df['Longitude'].max()
dy_common=gridsize_df['Latitude'].max()

dx_common_tos=gridsize_df_tos['Longitude'].max()
dy_common_tos=gridsize_df_tos['Latitude'].max()

p_level_common=[10000.0,15000.0,20000.0,25000.0,30000.0,40000.0,50000.0,60000.0,\
70000.0,85000.0,92500.0,100000.0]

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

#Delimiting the boundaries of each calculation

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

#-------------------------------------------------------------------------------------------------------------
#Stating the limits for the plots of the indices 
nash_str_limit_lower=980
nash_str_limit_upper=1100

nash_lat_limit_lower=10
nash_lat_limit_upper=55

nash_lon_limit_lower=360-65
nash_lon_limit_upper=360-1

spsh_str_limit_lower=980
spsh_str_limit_upper=1100

spsh_lat_limit_lower=-55
spsh_lat_limit_upper=-10

spsh_lon_limit_lower=360-130
spsh_lon_limit_upper=360-65

sash_str_limit_lower=980
sash_str_limit_upper=1100

sash_lat_limit_lower=-55
sash_lat_limit_upper=-10

sash_lon_limit_lower=360-35
sash_lon_limit_upper=360-1

westerlies_str_limit_lower=0
westerlies_str_limit_upper=30

westerlies_lat_limit_lower=-70
westerlies_lat_limit_upper=-25

subjet_str_limit_lower=15
subjet_str_limit_upper=55

subjet_lat_limit_lower=-50
subjet_lat_limit_upper=-15

trade_str_lower=-4
trade_str_upper=15

vimf_lower=0
vimf_upper=550

vihf_lower=-25
vihf_upper=50

#-------------------------------------------------------------------------------------------------------------

#list_calculation=['wind_850','wind_200','Subtropical_highs','Precipitation','Regional_cells',\
#                  'SST','Wind_indices','Bolivian_high','VIMF','qu_qv','MSE','tu_tv']

list_calculation=['Subtropical_highs']

for i in range(len(list_calculation)):
    if list_calculation[i]=='Subtropical_highs':

        try:

            #Inputs for the plot 
            models=np.load(path_entry+'subtropicalHighs_models_N.npz',allow_pickle=True)['arr_0']
            
            southAtlantic_strength_ref=np.load(path_entry+'southAtlantic_high_strength_ERA5.npz',allow_pickle=True)['arr_0']
            southPacific_strength_ref=np.load(path_entry+'southPacific_high_strength_ERA5.npz',allow_pickle=True)['arr_0']
            nash_strength_ref=np.load(path_entry+'northAtlantic_high_strength_ERA5.npz',allow_pickle=True)['arr_0']

            southAtlantic_strength_models=np.load(path_entry+'southAtlantic_high_strength_models.npz',allow_pickle=True)['arr_0']
            southPacific_strength_models=np.load(path_entry+'southPacific_high_strength_models.npz',allow_pickle=True)['arr_0']
            nash_strength_models=np.load(path_entry+'northAtlantic_high_strength_models.npz',allow_pickle=True)['arr_0']

            southAtlantic_lat_ref=np.load(path_entry+'southAtlantic_high_latitude_ERA5.npz',allow_pickle=True)['arr_0']
            southAtlantic_lon_ref=np.load(path_entry+'southAtlantic_high_longitude_ERA5.npz',allow_pickle=True)['arr_0']
            southAtlantic_latitude_models=np.load(path_entry+'southAtlantic_high_latitude_models.npz',allow_pickle=True)['arr_0']
            southAtlantic_longitude_models=np.load(path_entry+'southAtlantic_high_longitude_models.npz',allow_pickle=True)['arr_0']

            southPacific_lat_ref=np.load(path_entry+'southPacific_high_latitude_ERA5.npz',allow_pickle=True)['arr_0']
            southPacific_lon_ref=np.load(path_entry+'southPacific_high_longitude_ERA5.npz',allow_pickle=True)['arr_0']
            southPacific_latitude_models=np.load(path_entry+'southPacific_high_latitude_models.npz',allow_pickle=True)['arr_0']
            southPacific_longitude_models=np.load(path_entry+'southPacific_high_longitude_models.npz',allow_pickle=True)['arr_0']

            nash_lat_ref=np.load(path_entry+'northAtlantic_high_latitude_ERA5.npz',allow_pickle=True)['arr_0']
            nash_lon_ref=np.load(path_entry+'northAtlantic_high_longitude_ERA5.npz',allow_pickle=True)['arr_0']
            nash_latitude_models=np.load(path_entry+'northAtlantic_high_latitude_models.npz',allow_pickle=True)['arr_0']
            nash_longitude_models=np.load(path_entry+'northAtlantic_high_longitude_models.npz',allow_pickle=True)['arr_0']

            #------------------------------------------------------------------------------------------------------
            #Performing the data quality of the series of the models 
            models, southAtlantic_strength_models=filter_series_plots(southAtlantic_strength_models,models,sash_str_limit_lower,sash_str_limit_upper)
            models, southPacific_strength_models=filter_series_plots(southPacific_strength_models,models,spsh_str_limit_lower,spsh_str_limit_upper)
            models, nash_strength_models=filter_series_plots(nash_strength_models,models,nash_str_limit_lower,nash_str_limit_upper)

            models, southAtlantic_latitude_models=filter_series_plots(southAtlantic_latitude_models,models,sash_lat_limit_lower,sash_lat_limit_upper)
            models, southPacific_latitude_models=filter_series_plots(southPacific_latitude_models,models,spsh_lat_limit_lower,spsh_lat_limit_upper)
            models, nash_latitude_models=filter_series_plots(nash_latitude_models,models,nash_lat_limit_lower,nash_lat_limit_upper)

            models, southAtlantic_longitude_models=filter_series_plots(southAtlantic_longitude_models,models,sash_lon_limit_lower,sash_lon_limit_upper)
            models, southPacific_longitude_models=filter_series_plots(southPacific_longitude_models,models,spsh_lon_limit_lower,spsh_lon_limit_upper)
            models, nash_longitude_models=filter_series_plots(nash_longitude_models,models,nash_lon_limit_lower,nash_lon_limit_upper)


            #----------------------------------------------------------------------------------------------------
            #Obtaining the metrics of the series 
            
            series_metrics(southAtlantic_strength_ref,southAtlantic_strength_models,models,'southAtlantic_high_strength',path_entry)
            series_metrics(southPacific_strength_ref,southPacific_strength_models,models,'southPacific_high_strength',path_entry)
            series_metrics(nash_strength_ref,nash_strength_models,models,'nash_high_strength',path_entry)

            series_metrics(southAtlantic_lat_ref,southAtlantic_latitude_models,models,'southAtlantic_high_lat',path_entry)
            series_metrics(southPacific_lat_ref,southPacific_latitude_models,models,'southPacific_high_lat',path_entry)
            series_metrics(nash_lat_ref,nash_latitude_models,models,'nash_high_lat',path_entry)

            series_metrics(southAtlantic_lon_ref,southAtlantic_longitude_models,models,'southAtlantic_high_lon',path_entry)
            series_metrics(southPacific_lon_ref,southPacific_longitude_models,models,'southPacific_high_lon',path_entry)
            series_metrics(nash_lon_ref,nash_longitude_models,models,'nash_high_lon',path_entry)
            

            #---------------------------------------------------------------------------------------------
            #1. Intensity, location (Lat, Lon)

            #South Atlantic subtropical High
            plot_series_int_loc('South Atlantic subtropical High (SASH)',southAtlantic_strength_ref,\
            southAtlantic_strength_models,southAtlantic_lat_ref*(-1),360-southAtlantic_lon_ref,\
            southAtlantic_latitude_models*(-1),360-southAtlantic_longitude_models,'a. SASH Intensity',\
            'b. SASH Latitude','c. SASH Longitude',models,\
            'SouthAtlanticSubtropicalHigh_SASH','Pressure [hPa]','[°S]', path_save,3,1,10,17,fig_title_font,\
                title_str_size, xy_label_str, tick_labels_str, legends_str)

            #Southern Pacific subtropical High
            plot_series_int_loc('Southern Pacific subtropical High (SPSH)',southPacific_strength_ref,\
            southPacific_strength_models,southPacific_lat_ref*(-1),360-southPacific_lon_ref,\
            southPacific_latitude_models*(-1),360-southPacific_longitude_models,'a. SPSH Intensity',\
            'b. SPSH Latitude','c. SPSH Longitude',models,\
            'SouthernPacificSubtropicalHigh_SASH','Pressure [hPa]','[°S]', path_save,3,1,10,17,fig_title_font,\
                title_str_size, xy_label_str, tick_labels_str, legends_str)

            #North Atlantic subtropical High
            plot_series_int_loc('North Atlantic subtropical High (NASH)',nash_strength_ref,\
            nash_strength_models,nash_lat_ref,360-nash_lon_ref,\
            nash_latitude_models,360-nash_longitude_models,'a. NASH Intensity',\
            'b. NASH Latitude','c. NASH Longitude',models,\
            'NorthAtlanticSubtropicalHigh_NASH','Pressure [hPa]','[°N]', path_save,3,1,10,17,fig_title_font,\
                title_str_size, xy_label_str, tick_labels_str, legends_str)

            #--------------------------------------------------------------------------------------------
            #2. Intensity 
            fig=plt.figure(figsize=(10, 20))
            fig.suptitle('Subtropical Highs Intensity',fontsize=fig_title_font,\
            fontweight='bold')

            ax1 = fig.add_subplot(3, 1, 1)
            plot_series(ax1,southAtlantic_strength_ref,southAtlantic_strength_models,\
            'a. South Atlantic subtropical High (SASH)',models,'Pressure [hPa]','Yes',title_str_size, xy_label_str, tick_labels_str,'no')

            ax2 = fig.add_subplot(3, 1, 2)
            plot_series(ax2,southPacific_strength_ref,southPacific_strength_models,\
            'b. Southern Pacific subtropical High (SPSH)',models,'Pressure [hPa]','No',title_str_size, xy_label_str, tick_labels_str,'no')

            ax3 = fig.add_subplot(3, 1, 3)
            plot_series(ax3,nash_strength_ref,nash_strength_models,\
            'c. North Atlantic subtropical High (NASH)',models,'Pressure [hPa]','No',title_str_size, xy_label_str, tick_labels_str,'yes')

            nrows = 20
            ncols = int(np.ceil(len(models) / float(nrows)))

            fig.legend( bbox_to_anchor=(0.92, 0.9), ncol=1,loc='upper left', fontsize=str(legends_str))

            fig.savefig(path_save+'SubtropicalHighs_intensity.png', \
            format = 'png', bbox_inches='tight')
            plt.close()

            #--------------------------------------------------------------------------------------------------------
            #3. Combined series
            plot_series_int_loc_combined('Subtropical Highs',southAtlantic_strength_ref,southAtlantic_strength_models,southAtlantic_lat_ref*(-1),360-southAtlantic_lon_ref,southAtlantic_latitude_models*(-1),360-southAtlantic_longitude_models,\
                               southPacific_strength_ref,southPacific_strength_models,southPacific_lat_ref*(-1),360-southPacific_lon_ref,southPacific_latitude_models*(-1),360-southPacific_longitude_models,\
                               nash_strength_ref,nash_strength_models,nash_lat_ref,360-nash_lon_ref,nash_latitude_models,360-nash_longitude_models,\
                               'a. SASH Intensity','d. SASH Latitude','g. SASH Longitude','b. SPSH Intensity','e. SPSH Latitude','h. SPSH Longitude',\
                               'c. NASH Intensity','f. NASH Latitude','i. NASH Longitude',models,'SubtropicalHighs_indices','Pressure [hPa]', path_save,3,3,27,21,fig_title_font,title_str_size, \
                                xy_label_str, tick_labels_str, legends_str)

            
            """
            #--------------------------------------------------------------------------------------------------------
            #4. Subtropical center 
            seasons_labels_i=['DJF','JJA','MAM','SON']
            
            #input for the plot 
            southAtlantic_lat_ref_seasonal=np.load(path_entry+'southAtlantic_high_latitude_seasonal_ERA5.npz',allow_pickle=True)['arr_0']
            southAtlantic_lon_ref_seasonal=np.load(path_entry+'southAtlantic_high_longitude_seasonal_ERA5.npz',allow_pickle=True)['arr_0']

            southPacific_lat_ref_seasonal=np.load(path_entry+'southPacific_high_latitude_seasonal_ERA5.npz',allow_pickle=True)['arr_0']
            southPacific_lon_ref_seasonal=np.load(path_entry+'southPacific_high_longitude_seasonal_ERA5.npz',allow_pickle=True)['arr_0']

            nash_lat_ref_seasonal=np.load(path_entry+'northAtlantic_high_latitude_seasonal_ERA5.npz',allow_pickle=True)['arr_0']
            nash_lon_ref_seasonal=np.load(path_entry+'northAtlantic_high_longitude_seasonal_ERA5.npz',allow_pickle=True)['arr_0']

            southAtlantic_latitude_models_seasonal=np.load(path_entry+'southAtlantic_high_latitude_seasonal_models.npz',allow_pickle=True)['arr_0']
            southAtlantic_longitude_models_seasonal=np.load(path_entry+'southAtlantic_high_longitude_seasonal_models.npz',allow_pickle=True)['arr_0']

            southPacific_latitude_models_seasonal=np.load(path_entry+'southPacific_high_latitude_seasonal_models.npz',allow_pickle=True)['arr_0']
            southPacific_longitude_models_seasonal=np.load(path_entry+'southPacific_high_longitude_seasonal_models.npz',allow_pickle=True)['arr_0']

            nash_latitude_models_seasonal=np.load(path_entry+'northAtlantic_high_latitude_seasonal_models.npz',allow_pickle=True)['arr_0']
            nash_longitude_models_seasonal=np.load(path_entry+'northAtlantic_high_longitude_seasonal_models.npz',allow_pickle=True)['arr_0']

            lat_common_plot=np.arange(-70,70,1)
            lon_common_plot=np.arange(200,360,1)

            for n in range(4):
                index_season=n
                seasons_labels=seasons_labels_i[index_season]

                #NASH
                nash_latitude_round=np.round(nash_latitude_models_seasonal[:,index_season],0)
                nash_longitude_round=np.round(nash_longitude_models_seasonal[:,index_season],0)
                #SASH
                sash_latitude_round=np.round(southAtlantic_latitude_models_seasonal[:,index_season],0)
                sash_longitude_round=np.round(southAtlantic_longitude_models_seasonal[:,index_season],0)
                #SPSH
                spsh_latitude_round=np.round(southPacific_latitude_models_seasonal[:,index_season],0)
                spsh_longitude_round=np.round(southPacific_longitude_models_seasonal[:,index_season],0)

                ################################################################################
                ################################################################################
                #Creating the plot
                list_markers=['o','P','X','D','v','<','s','H']
                n_repeat=math.ceil(len(models)/len(list_markers))
                list_markers_repeated=list_markers*n_repeat
                markers=list_markers_repeated[0:len(models)]
                colors=iter(cm.rainbow(np.linspace(0,1,len(models))))


                lon2D, lat2D = np.meshgrid(lon_common_plot, lat_common_plot)
                projection=ccrs.PlateCarree() #definiendo la proyeccion
                extent = [min(lon_common_plot),max(lon_common_plot),min(lat_common_plot),max(lat_common_plot)]


                fig1=plt.figure(figsize=(13, 13))
                ax = plt.axes(projection=ccrs.PlateCarree())

                ax.set_extent(extent, projection)
                ax.set_title('Subtropical Highs Core - '+seasons_labels,fontsize=fig_title_font)
                ax.add_feature(cfeature.OCEAN, edgecolor='k',facecolor=cfeature.COLORS['water'])
                #, zorder=100
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.LAND)
                ax.add_feature(cfeature.BORDERS, linestyle='-')
                #For the NASH
                for g in range(len(models)):
                    co=next(colors)
                    plt.scatter(nash_longitude_round[g],nash_latitude_round[g],s=70,zorder=500, c=co,marker=markers[g],label=models[g])
                    plt.scatter(sash_longitude_round[g],sash_latitude_round[g],s=70,zorder=500, c=co,marker=markers[g])
                    plt.scatter(spsh_longitude_round[g],spsh_latitude_round[g],s=70,zorder=500, c=co,marker=markers[g])

                plt.scatter(nash_lon_ref_seasonal[index_season],nash_lat_ref_seasonal[index_season],s=70,zorder=500, c='black',marker='*',label='Reference [ERA5]')
                plt.scatter(southAtlantic_lon_ref_seasonal[index_season],southAtlantic_lat_ref_seasonal[index_season],s=70,zorder=500, c='black',marker='*')
                plt.scatter(southPacific_lon_ref_seasonal[index_season],southPacific_lat_ref_seasonal[index_season],s=70,zorder=500, c='black',marker='*')
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,)
                gl.xlabels_top = False
                gl.ylabels_right = False
                gl.xlines = False
                gl.ylines = False
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                gl.xlabel_style = {'size': tick_labels_str}
                gl.ylabel_style = {'size': tick_labels_str}

                ax.add_patch(mpatches.Rectangle(xy=[210, -50], width=71, height=35,ec='red',
                                                    facecolor='red',
                                                    alpha=1,fill=None,lw=2.5,
                                                    transform=ccrs.PlateCarree()))
                ax.add_patch(mpatches.Rectangle(xy=[325, -45], width=33, height=45,ec='orange',
                                                    facecolor='orange',
                                                    alpha=1,fill=None,lw=2.5,
                                                    transform=ccrs.PlateCarree()))
                ax.add_patch(mpatches.Rectangle(xy=[290, 20], width=54, height=23,ec='blue',
                                                    facecolor='blue',
                                                    alpha=1,fill=None,lw=2.5,
                                                    transform=ccrs.PlateCarree()))
                
                nrows = 20
                ncols = int(np.ceil(len(models) / float(nrows)))

                fig1.legend( bbox_to_anchor=(0.92, 0.8), ncol=1,loc='upper left', fontsize=str(legends_str),frameon=False)

                fig1.savefig(path_save+seasons_labels+'_SubtropicalHighs_Core.png', \
                format = 'png', bbox_inches='tight')
                plt.close()
            
            #--------------------------------------------------------------------------------------------------------------------------
            #4. Spatial fields of SLP 
            ################################################################################
            #Obtaining the ensamble

            var_mmm=seasonal_ensamble(models,path_entry,\
            'psl_MMM_meanFields',len(Lat_common),len(Lon_common))

            var_mmm_bias=seasonal_ensamble(models,path_entry,\
            'psl_MMM_biasFields',len(Lat_common),len(Lon_common))

            bias_mmm_agreement=agreement_sign(models,path_entry,'psl_MMM_biasFields',\
                                            len(Lat_common),len(Lon_common))

            ################################################################################
            #PLOT
            models_metrics=pd.read_csv(path_entry+'taylorDiagram_metrics_psl.csv', index_col=[0])

            ref_std=pd.read_csv(path_entry+'reference_std_original.csv',index_col=[0])

            
            plot_label='SLP [hPa]'
            limits_var=np.arange(1000,1028,2)
            limits_bias=np.arange(-5,5.5,0.5)

            cmap_plot=['gist_rainbow_r','terrain','terrain_r','rainbow','RdBu']

            fig=plt.figure(figsize=(24,14))
            colorbar_attributes=[0.92, 0.37,  0.017,0.24]

            colorbar_attributes_bias=[0.92, 0.1,  0.017,0.24]

            lon2D, lat2D = np.meshgrid(Lon_common, Lat_common)
            projection=ccrs.PlateCarree()
            extent = [min(Lon_common),max(Lon_common),min(Lat_common),max(Lat_common)]

            taylor=td_plots(fig,'DJF',ref_std,models_metrics,'SLP',len(models),341,'a.',title_str_size,'no',None)

            taylor=td_plots(fig,'MAM',ref_std,models_metrics,'SLP',len(models),342,'b.',title_str_size,'no',None)

            taylor=td_plots(fig,'JJA',ref_std,models_metrics,'SLP',len(models),343,'c.',title_str_size,'no',None)

            taylor=td_plots(fig,'SON',ref_std,models_metrics,'SLP',len(models),344,'d.',title_str_size,'no',None)

            ax5 = fig.add_subplot(3, 4, 5, projection=projection)
            cs=plotMap(ax5,var_mmm[0],lon2D,lat2D,cmap_slp,limits_var,'e.',extent, projection,title_str_size,'no',None,'no')

            ax6 = fig.add_subplot(3, 4, 6, projection=projection)
            cs=plotMap(ax6,var_mmm[2],lon2D,lat2D,cmap_slp,limits_var,'f.',extent, projection,title_str_size,'no',None,'no')

            ax7 = fig.add_subplot(3, 4, 7, projection=projection)
            cs=plotMap(ax7,var_mmm[1],lon2D,lat2D,cmap_slp,limits_var,'g.',extent, projection,title_str_size,'no',None,'no')

            ax8 = fig.add_subplot(3, 4, 8, projection=projection)
            cs=plotMap(ax8,var_mmm[3],lon2D,lat2D,cmap_slp,limits_var,'h.',extent, projection,title_str_size,'no',None,'no')

            ax9 = fig.add_subplot(3, 4, 9, projection=projection)
            csb=plotMap(ax9,var_mmm_bias[0],lon2D,lat2D,cmap_bias,limits_bias,'i.',extent, projection,title_str_size,'yes',bias_mmm_agreement[0],'no')

            ax10 = fig.add_subplot(3, 4, 10, projection=projection)
            csb=plotMap(ax10,var_mmm_bias[2],lon2D,lat2D,cmap_bias,limits_bias,'j.',extent, projection,title_str_size,'yes',bias_mmm_agreement[2],'no')

            ax11 = fig.add_subplot(3, 4, 11, projection=projection)
            csb=plotMap(ax11,var_mmm_bias[1],lon2D,lat2D,cmap_bias,limits_bias,'k.',extent, projection,title_str_size,'yes',bias_mmm_agreement[1],'no')

            ax12 = fig.add_subplot(3, 4, 12, projection=projection)
            csb=plotMap(ax12,var_mmm_bias[3],lon2D,lat2D,cmap_bias,limits_bias,'l.',extent, projection,title_str_size,'yes',bias_mmm_agreement[3],'no')

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
            plt.savefig(path_save+'slp_fields.png', \
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
            path_save+'slp_fields_legend.png', format = 'png',\
            bbox_inches='tight',bbox_extra_artists=[legend_squared],
            ) 
            plt.close()
            """
        except Exception as e:
            print('Error plot subtropical highs')
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")

            
    elif list_calculation[i]=='Wind_indices':

        try:

            #Inputs

            subtropical_strength_ref=np.load(path_entry+'subtropical_jet_strength_ERA5.npz',allow_pickle=True)['arr_0']
            subtropical_Str_models=np.load(path_entry+'subtropical_jet_strength_models.npz',allow_pickle=True)['arr_0']
            subtropical_latitude_ref=np.load(path_entry+'subtropical_jet_latitude_ERA5.npz',allow_pickle=True)['arr_0']
            subtropical_Lat_models=np.load(path_entry+'subtropical_jet_latitude_models.npz',allow_pickle=True)['arr_0']

            westerly_strength_ref=np.load(path_entry+'westerlies_strength_ERA5.npz',allow_pickle=True)['arr_0']
            westerlies_Str_model=np.load(path_entry+'westerlies_strength_models.npz',allow_pickle=True)['arr_0']
            westerly_latitude_ref=np.load(path_entry+'westerlies_latitude_ERA5.npz',allow_pickle=True)['arr_0']
            westerlies_Lat_model=np.load(path_entry+'westerlies_latitude_models.npz',allow_pickle=True)['arr_0']

            models=np.load(path_entry+'wind_indices_models_N.npz',allow_pickle=True)['arr_0']

            trade_index_m=np.load(path_entry+'trade_winds_models.npz',allow_pickle=True)['arr_0']
            trade_index_r=np.load(path_entry+'trade_winds_ERA5.npz',allow_pickle=True)['arr_0']

            print('-----------------------------------------------------------------------------------')
            print('wind_indices: files read OK')
            print('-----------------------------------------------------------------------------------')

            #-------------------------------------------------------------------------------------------------------
            #Performing the filter of the series 
            models, subtropical_Str_models=filter_series_plots(subtropical_Str_models,models,subjet_str_limit_lower,subjet_str_limit_upper)
            models, subtropical_Lat_models=filter_series_plots(subtropical_Lat_models,models,subjet_lat_limit_lower,subjet_lat_limit_upper)

            models, westerlies_Str_model=filter_series_plots(westerlies_Str_model,models,westerlies_str_limit_lower,westerlies_str_limit_upper)
            models, westerlies_Lat_model=filter_series_plots(westerlies_Lat_model,models,westerlies_lat_limit_lower,westerlies_lat_limit_upper)

            models, trade_index_m=filter_series_plots(trade_index_m,models,trade_str_lower,trade_str_upper)

            print('-----------------------------------------------------------------------------------')
            print('wind_indices: filter series OK')
            print('-----------------------------------------------------------------------------------')
            #-------------------------------------------------------------------------------------------------------
            #Obtaining the metrics 
            series_metrics(subtropical_strength_ref,subtropical_Str_models,models,'subtropicalJet_strength',path_entry)
            series_metrics(subtropical_latitude_ref,subtropical_Lat_models,models,'subtropicalJet_latitude',path_entry)

            series_metrics(westerly_strength_ref,westerlies_Str_model,models,'westerlies_strength',path_entry)
            series_metrics(westerly_latitude_ref,westerlies_Lat_model,models,'westerlies_latitude',path_entry)

            series_metrics(trade_index_r,trade_index_m,models,'Trade_winds',path_entry)

            print('-----------------------------------------------------------------------------------')
            print('wind_indices: series OK')
            print('-----------------------------------------------------------------------------------')

            #Subtropical jet stream
            wind_indices('Southern Hemisphere Subtropical Jet Stream',subtropical_strength_ref,\
            subtropical_Str_models,subtropical_latitude_ref,subtropical_Lat_models,\
            'a. Subtropical jet stream mean strength','b. Subtropical jet stream mean location',\
            np.arange(20,50,2),np.arange(-40,-22,2),models,'Subtropical_Jet_200hPa',path_save,2,1,10,14,fig_title_font,\
                title_str_size, xy_label_str, tick_labels_str, legends_str,['IPSL-CM5A2-INCA'])

            #Westerlies
            wind_indices('Southern Hemisphere Westerlies',westerly_strength_ref,\
            westerlies_Str_model,westerly_latitude_ref,westerlies_Lat_model,\
            'a. Westerlies mean strength','b. Westerlies mean location',\
            np.arange(5,21,1),np.arange(-65,-25,5),models,'Westerlies_850hPa',path_save,2,1,10,14,fig_title_font,\
                title_str_size, xy_label_str, tick_labels_str, legends_str,['NESM3','EC-Earth3'])

            #Trade winds 
            plot_one_plot(models,'Trade_Wind_Index',path_save,trade_index_m,trade_index_r,None,'Wind [m/s]',-4,12,'[850 hPa] Trade Wind Index',\
                        fig_title_font, xy_label_str, tick_labels_str, legends_str,['NESM3','EC-Earth3'])
            
            print('-----------------------------------------------------------------------------------')
            print('wind_indices: plots OK')
            print('-----------------------------------------------------------------------------------')
        
        except Exception as e:
            print('Error plot wind indices')
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
    
    elif list_calculation[i]=='Bolivian_high':

        try:

            #input
            models=np.load(path_entry+'Bolivian_High_models_N.npz',allow_pickle=True)['arr_0']

            bolivian_index_m=np.load(path_entry+'Bolivian_High_index_monthly.npz',allow_pickle=True)['arr_0']
            bolivian_index_r=np.load(path_entry+'Bolivian_High_index_monthly_ERA5.npz',allow_pickle=True)['arr_0']

            #Obtaining the metrics 
            series_metrics(bolivian_index_r,bolivian_index_m,models,'Bolivian_High',path_entry)


            plot_one_plot(models,'Bolivian_High',path_save,bolivian_index_m,bolivian_index_r, None,'200 hPa GPH [Km]',12,13,'Bolivian High Index',\
                        fig_title_font, xy_label_str, tick_labels_str, legends_str,[])
            
        except Exception as e:
            print('Error plot bolivian high')
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
    
    elif list_calculation[i]=='Precipitation':

        try:

            #input
            models=np.load(path_entry+'pr_fields_models_N.npz',allow_pickle=True)['arr_0']

            ################################################################################
            #Obtaining the ensamble

            var_mmm=seasonal_ensamble(models,path_entry,\
            'pr_MMM_meanFields',len(Lat_common),len(Lon_common))

            var_mmm_bias=seasonal_ensamble(models,path_entry,\
            'pr_MMM_biasFields',len(Lat_common),len(Lon_common))

            bias_mmm_agreement=agreement_sign(models,path_entry,'pr_MMM_biasFields',\
                                            len(Lat_common),len(Lon_common))
            
            print('-----------------------------------------------------------------------------------')
            print('Precipitation: files read OK')
            print('-----------------------------------------------------------------------------------')

            ################################################################################
            #PLOT
            models_metrics=pd.read_csv(path_entry+'taylorDiagram_metrics_pr.csv', index_col=[0])

            ref_std=pd.read_csv(path_entry+'reference_std_original.csv',index_col=[0])

            
            plot_label='Precipitation rate\n [mm/day]'
            limits_var=np.arange(2,18,1)
            limits_bias=np.arange(-9,10,1)

            cmap_plot=['gist_rainbow_r','terrain','terrain_r','rainbow','RdBu']

            fig=plt.figure(figsize=(24,14))
            colorbar_attributes=[0.92, 0.37,  0.017,0.24]

            colorbar_attributes_bias=[0.92, 0.1,  0.017,0.24]

            print('-----------------------------------------------------------------------------------')
            print('Precipitation: files metrics OK')
            print('-----------------------------------------------------------------------------------')

            lon2D, lat2D = np.meshgrid(Lon_common, Lat_common)
            projection=ccrs.PlateCarree()
            extent = [min(Lon_common),max(Lon_common),min(Lat_common),max(Lat_common)]

            taylor=td_plots(fig,'DJF',ref_std,models_metrics,'PPT',len(models),341,'a.',title_str_size,'no',None)

            taylor=td_plots(fig,'MAM',ref_std,models_metrics,'PPT',len(models),342,'b.',title_str_size,'no',None)

            taylor=td_plots(fig,'JJA',ref_std,models_metrics,'PPT',len(models),343,'c.',title_str_size,'no',None)

            taylor=td_plots(fig,'SON',ref_std,models_metrics,'PPT',len(models),344,'d.',title_str_size,'no',None)

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
            #fig.subplots_adjust(hspace=0.3)
            plt.savefig(path_save+'pr_mm_day.png', \
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
            path_save+'pr_mm_day_legend.png', format = 'png',\
            bbox_inches='tight',bbox_extra_artists=[legend_squared],
            ) 
            plt.close()
        
        except Exception as e:
            print('Error plot precipitation')
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")

    elif list_calculation[i]=='SST':

        try:

            #input
            models_o=np.load(path_entry+'tos_fields_models_N.npz',allow_pickle=True)['arr_0']

            models=[]

            for o in range(len(models_o)):
                if models_o[o]=='MIROC6':
                    pass 
                else:
                    models.append(models_o[o])
            
            models=np.array(models)

            ################################################################################
            #Obtaining the ensamble

            var_mmm=seasonal_ensamble(models,path_entry,\
            'tos_MMM_meanFields',len(Lat_common_tos),len(Lon_common_tos))

            var_mmm_bias=seasonal_ensamble(models,path_entry,\
            'tos_MMM_biasFields',len(Lat_common_tos),len(Lon_common_tos))

            bias_mmm_agreement=agreement_sign(models,path_entry,'tos_MMM_biasFields',\
                                            len(Lat_common_tos),len(Lon_common_tos))

            ################################################################################
            #PLOT
            models_metrics=pd.read_csv(path_entry+'taylorDiagram_metrics_tos.csv', index_col=[0])

            models_metrics=models_metrics.drop(models_metrics[models_metrics['Model']=='MIROC6'].index).reset_index(drop=True)

            ref_std=pd.read_csv(path_entry+'reference_std_original.csv',index_col=[0])

            plot_label='SST [°C]'
            limits_var=np.arange(0,32,2)
            limits_bias=np.arange(-6,7,1)

            cmap_plot=['gist_rainbow_r','terrain','terrain_r','rainbow','RdBu']

            fig=plt.figure(figsize=(24,14))
            colorbar_attributes=[0.92, 0.37,  0.017,0.24]

            colorbar_attributes_bias=[0.92, 0.1,  0.017,0.24]

            lon2D, lat2D = np.meshgrid(Lon_common_tos, Lat_common_tos)
            projection=ccrs.PlateCarree()
            extent = [min(Lon_common_tos),max(Lon_common_tos),min(Lat_common_tos),max(Lat_common_tos)]

            taylor=td_plots(fig,'DJF',ref_std,models_metrics,'SST',len(models),341,'a.',title_str_size,'no',None)

            taylor=td_plots(fig,'MAM',ref_std,models_metrics,'SST',len(models),342,'b.',title_str_size,'no',None)

            taylor=td_plots(fig,'JJA',ref_std,models_metrics,'SST',len(models),343,'c.',title_str_size,'no',None)

            taylor=td_plots(fig,'SON',ref_std,models_metrics,'SST',len(models),344,'d.',title_str_size,'no',None)

            ax5 = fig.add_subplot(3, 4, 5, projection=projection)
            cs=plotMap(ax5,var_mmm[0],lon2D,lat2D,cmap_sst,limits_var,'e.',extent, projection,title_str_size,'no',None,'yes')

            ax6 = fig.add_subplot(3, 4, 6, projection=projection)
            cs=plotMap(ax6,var_mmm[2],lon2D,lat2D,cmap_sst,limits_var,'f.',extent, projection,title_str_size,'no',None,'yes')

            ax7 = fig.add_subplot(3, 4, 7, projection=projection)
            cs=plotMap(ax7,var_mmm[1],lon2D,lat2D,cmap_sst,limits_var,'g.',extent, projection,title_str_size,'no',None,'yes')

            ax8 = fig.add_subplot(3, 4, 8, projection=projection)
            cs=plotMap(ax8,var_mmm[3],lon2D,lat2D,cmap_sst,limits_var,'h.',extent, projection,title_str_size,'no',None,'yes')

            ax9 = fig.add_subplot(3, 4, 9, projection=projection)
            csb=plotMap(ax9,var_mmm_bias[0],lon2D,lat2D,cmap_bias,limits_bias,'i.',extent, projection,title_str_size,'yes',bias_mmm_agreement[0],'yes')

            ax10 = fig.add_subplot(3, 4, 10, projection=projection)
            csb=plotMap(ax10,var_mmm_bias[2],lon2D,lat2D,cmap_bias,limits_bias,'j.',extent, projection,title_str_size,'yes',bias_mmm_agreement[2],'yes')

            ax11 = fig.add_subplot(3, 4, 11, projection=projection)
            csb=plotMap(ax11,var_mmm_bias[1],lon2D,lat2D,cmap_bias,limits_bias,'k.',extent, projection,title_str_size,'yes',bias_mmm_agreement[1],'yes')

            ax12 = fig.add_subplot(3, 4, 12, projection=projection)
            csb=plotMap(ax12,var_mmm_bias[3],lon2D,lat2D,cmap_bias,limits_bias,'l.',extent, projection,title_str_size,'yes',bias_mmm_agreement[3],'yes')

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
            plt.savefig(path_save+'tos_fields_c.png', \
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
            path_save+'tos_fields_c_legend.png', format = 'png',\
            bbox_inches='tight',bbox_extra_artists=[legend_squared],
            ) 
            plt.close()

            """
            #----------------------------------------------------------------------------------------------
            #----------------------------------------------------------------------------------------------
            #Applying the function to check the models individually
            individual_plots('sst',models,path_entry,path_save_ind,'tos_MMM_meanFields','tos_MMM_biasFields',
                             None,None,Lat_common_tos, Lon_common_tos, np.arange(0,32,2),np.arange(-6,7,1),
                             [0.92, 0.51,  0.017,0.34],[0.92, 0.12,  0.017,0.34])
            """
        except Exception as e:
            print('Error plot SST')
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
    
    elif list_calculation[i]=='wind_200':

        try:

            #input
            models=np.load(path_entry+'wind200_fields_models_N.npz',allow_pickle=True)['arr_0']

            ################################################################################
            #Obtaining the ensamble

            mag_mmm=seasonal_ensamble(models,path_entry,\
            'mag200_MMM_meanFields',len(Lat_common),len(Lon_common))

            mag_mmm_bias=seasonal_ensamble(models,path_entry,\
            'mag200_MMM_biasFields',len(Lat_common),len(Lon_common))

            ua_mmm=seasonal_ensamble(models,path_entry,\
            'ua200_MMM_meanFields',len(Lat_common),len(Lon_common))

            va_mmm=seasonal_ensamble(models,path_entry,\
            'va200_MMM_meanFields',len(Lat_common),len(Lon_common))

            bias_mmm_agreement=agreement_sign(models,path_entry,'mag200_MMM_biasFields',\
                                            len(Lat_common),len(Lon_common))
            
            print('-----------------------------------------------------------------------------------')
            print('wind_200: files read OK')
            print('-----------------------------------------------------------------------------------')

            #PLOT
            models_metrics=pd.read_csv(path_entry+'taylorDiagram_metrics_wind200.csv', index_col=[0])

            ref_std=pd.read_csv(path_entry+'reference_std_original.csv',index_col=[0])

            plot_label='Wind magnitude [m/s]'
            limits_var=np.arange(1,45,5)
            limits_bias=np.arange(-10,11,1)

            cmap_plot=['gist_rainbow_r','terrain','terrain_r','rainbow','RdBu']

            print('-----------------------------------------------------------------------------------')
            print('wind_200: files metrics OK')
            print('-----------------------------------------------------------------------------------')

            fig=plt.figure(figsize=(24,14))
            colorbar_attributes=[0.92, 0.37,  0.017,0.24]

            colorbar_attributes_bias=[0.92, 0.1,  0.017,0.24]

            lon2D, lat2D = np.meshgrid(Lon_common, Lat_common)
            projection=ccrs.PlateCarree()
            extent = [min(Lon_common),max(Lon_common),min(Lat_common),max(Lat_common)]

            taylor=td_plots(fig,'DJF',ref_std,models_metrics,'W200',len(models),341,'a.',title_str_size,'no',None)

            taylor=td_plots(fig,'MAM',ref_std,models_metrics,'W200',len(models),342,'b.',title_str_size,'no',None)

            taylor=td_plots(fig,'JJA',ref_std,models_metrics,'W200',len(models),343,'c.',title_str_size,'no',None)

            taylor=td_plots(fig,'SON',ref_std,models_metrics,'W200',len(models),344,'d.',title_str_size,'no',None)

            print('-----------------------------------------------------------------------------------')
            print('wind_200: td plots OK')
            print('-----------------------------------------------------------------------------------')

            ax5 = fig.add_subplot(3, 4, 5, projection=projection)
            cs=plotMap_vector(ax5,mag_mmm[0],ua_mmm[0],va_mmm[0],lon2D,lat2D,cmap_w200,limits_var,'e.',extent, projection,title_str_size)

            ax6 = fig.add_subplot(3, 4, 6, projection=projection)
            cs=plotMap_vector(ax6,mag_mmm[2],ua_mmm[2],va_mmm[2],lon2D,lat2D,cmap_w200,limits_var,'f.',extent, projection,title_str_size)

            ax7 = fig.add_subplot(3, 4, 7, projection=projection)
            cs=plotMap_vector(ax7,mag_mmm[1],ua_mmm[1],va_mmm[1],lon2D,lat2D,cmap_w200,limits_var,'g.',extent, projection,title_str_size,)

            ax8 = fig.add_subplot(3, 4, 8, projection=projection)
            cs=plotMap_vector(ax8,mag_mmm[3],ua_mmm[3],va_mmm[3],lon2D,lat2D,cmap_w200,limits_var,'h.',extent, projection,title_str_size)

            print('-----------------------------------------------------------------------------------')
            print('wind_200: plot vector OK')
            print('-----------------------------------------------------------------------------------')

            ax9 = fig.add_subplot(3, 4, 9, projection=projection)
            csb=plotMap(ax9,mag_mmm_bias[0],lon2D,lat2D,cmap_bias,limits_bias,'i.',extent, projection,title_str_size,'yes',bias_mmm_agreement[0],'no')

            ax10 = fig.add_subplot(3, 4, 10, projection=projection)
            csb=plotMap(ax10,mag_mmm_bias[2],lon2D,lat2D,cmap_bias,limits_bias,'j.',extent, projection,title_str_size,'yes',bias_mmm_agreement[2],'no')

            ax11 = fig.add_subplot(3, 4, 11, projection=projection)
            csb=plotMap(ax11,mag_mmm_bias[1],lon2D,lat2D,cmap_bias,limits_bias,'k.',extent, projection,title_str_size,'yes',bias_mmm_agreement[1],'no')

            ax12 = fig.add_subplot(3, 4, 12, projection=projection)
            csb=plotMap(ax12,mag_mmm_bias[3],lon2D,lat2D,cmap_bias,limits_bias,'l.',extent, projection,title_str_size,'yes',bias_mmm_agreement[3],'no')

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
            plt.savefig(path_save+'wind200_fields.png', \
            format = 'png', bbox_inches='tight')
            
            #To save the legend independently
            dia=taylor

            legend= fig.legend(dia.samplePoints,
            [ p.get_label() for p in dia.samplePoints ],
            numpoints=1, prop=dict(size='small') \
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
            path_save+'wind200_legend.png', format = 'png',\
            bbox_inches='tight',bbox_extra_artists=[legend_squared],
            ) 
            plt.close()
        except Exception as e:
            print('Error plot wind 200')
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
    
    elif list_calculation[i]=='wind_850':

        try:

            #input
            models_o=np.load(path_entry+'wind850_fields_models_N.npz',allow_pickle=True)['arr_0']

            models=[]

            for o in range(len(models_o)):
                if models_o[o]=='E3SM-1-1':
                    pass 
                else:
                    models.append(models_o[o])
            
            models=np.array(models)

            ################################################################################
            #Obtaining the ensamble

            print ('  Obteniendo mag850 mean ...')
            mag_mmm=seasonal_ensamble(models,path_entry,\
            'mag850_MMM_meanFields',len(Lat_common),len(Lon_common))

            print ('  Obteniendo mag850 bias ...')
            mag_mmm_bias=seasonal_ensamble(models,path_entry,\
            'mag850_MMM_biasFields',len(Lat_common),len(Lon_common))

            print ('  Obteniendo u850 mean ...')
            ua_mmm=seasonal_ensamble(models,path_entry,\
            'ua850_MMM_meanFields',len(Lat_common),len(Lon_common))

            print ('  Obteniendo va850 mean ...')
            va_mmm=seasonal_ensamble(models,path_entry,\
            'va850_MMM_meanFields',len(Lat_common),len(Lon_common))

            print ('  agreement_sign mag 850 ...')
            bias_mmm_agreement=agreement_sign(models,path_entry,'mag850_MMM_biasFields',\
                                            len(Lat_common),len(Lon_common))
            
            print('-----------------------------------------------------------------------------------')
            print('wind_850: files read OK')
            print('-----------------------------------------------------------------------------------')

            #PLOT
            print ("  Leyendo '" + path_entry+"taylorDiagram_metrics_wind850.csv'")
            models_metrics=pd.read_csv(path_entry+'taylorDiagram_metrics_wind850.csv', index_col=[0])

            models_metrics=models_metrics.drop(models_metrics[models_metrics['Model']=='E3SM-1-1'].index).reset_index(drop=True)

            print ("  Leyendo '" + path_entry+"reference_std_original.csv'")
            ref_std=pd.read_csv(path_entry+'reference_std_original.csv',index_col=[0])

            print ('Ploteando ...')

            plot_label='Wind magnitude [m/s]'
            limits_var=np.arange(1.5,13.5,1.5)
            limits_bias=np.arange(-5,6,1)

            cmap_plot=['gist_rainbow_r','terrain','terrain_r','rainbow','RdBu']

            fig=plt.figure(figsize=(24,14))
            colorbar_attributes=[0.92, 0.37,  0.017,0.24]

            colorbar_attributes_bias=[0.92, 0.1,  0.017,0.24]

            print('-----------------------------------------------------------------------------------')
            print('wind_850: files metrics OK')
            print('-----------------------------------------------------------------------------------')

            lon2D, lat2D = np.meshgrid(Lon_common, Lat_common)
            projection=ccrs.PlateCarree()

            print ('  extent')
            extent = [min(Lon_common),max(Lon_common),min(Lat_common),max(Lat_common)]

            print ('  ploteando DJF Taylor')
            taylor=td_plots(fig,'DJF',ref_std,models_metrics,'W850',len(models),341,'a.',title_str_size,'no',None)

            taylor=td_plots(fig,'MAM',ref_std,models_metrics,'W850',len(models),342,'b.',title_str_size,'no',None)

            taylor=td_plots(fig,'JJA',ref_std,models_metrics,'W850',len(models),343,'c.',title_str_size,'no',None)

            taylor=td_plots(fig,'SON',ref_std,models_metrics,'W850',len(models),344,'d.',title_str_size,'no',None)

            print('-----------------------------------------------------------------------------------')
            print('wind_850: td plots OK')
            print('-----------------------------------------------------------------------------------')

            print ('    shapes 0 mag_mmm', mag_mmm[0].shape, 'ua', ua_mmm[0].shape,  \
              'va', va_mmm[0].shape, 'lon2D', lon2D.shape, 'lat2D', lat2D.shape)
            ax5 = fig.add_subplot(3, 4, 5, projection=projection)
            cs=plotMap_vector(ax5,mag_mmm[0],ua_mmm[0],va_mmm[0],lon2D,lat2D,cmap_w850,limits_var,'e.',extent, projection,title_str_size)

            print ('    shapes 2 mag_mmm', mag_mmm[2].shape, 'ua', ua_mmm[2].shape,  \
              'va', va_mmm[2].shape, 'lon2D', lon2D.shape, 'lat2D', lat2D.shape)
            ax6 = fig.add_subplot(3, 4, 6, projection=projection)
            cs=plotMap_vector(ax6,mag_mmm[2],ua_mmm[2],va_mmm[2],lon2D,lat2D,cmap_w850,limits_var,'f.',extent, projection,title_str_size)

            print ('    shapes 1 mag_mmm', mag_mmm[1].shape, 'ua', ua_mmm[1].shape,  \
              'va', va_mmm[1].shape, 'lon2D', lon2D.shape, 'lat2D', lat2D.shape)
            ax7 = fig.add_subplot(3, 4, 7, projection=projection)
            cs=plotMap_vector(ax7,mag_mmm[1],ua_mmm[1],va_mmm[1],lon2D,lat2D,cmap_w850,limits_var,'g.',extent, projection,title_str_size,)

            print ('    shapes 3 mag_mmm', mag_mmm[3].shape, 'ua', ua_mmm[3].shape,  \
              'va', va_mmm[3].shape, 'lon2D', lon2D.shape, 'lat2D', lat2D.shape)
            ax8 = fig.add_subplot(3, 4, 8, projection=projection)
            cs=plotMap_vector(ax8,mag_mmm[3],ua_mmm[3],va_mmm[3],lon2D,lat2D,cmap_w850,limits_var,'h.',extent, projection,title_str_size)

            print('-----------------------------------------------------------------------------------')
            print('wind_850: plot vector OK')
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
            plt.savefig(path_save+'wind850_fields.png', \
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
            path_save+'wind850_legend.png', format = 'png',\
            bbox_inches='tight',bbox_extra_artists=[legend_squared],
            ) 

            plt.close()

            """
            #-------------------------------------------------------------------------
            #-------------------------------------------------------------------------
            #Creating the individual plots to check the models individually 

            individual_plots('vector',models,path_entry,path_save_ind,'mag850_MMM_meanFields',
                             'mag850_MMM_biasFields','ua850_MMM_meanFields','va850_MMM_meanFields',
                             Lat_common, Lon_common, np.arange(1.5,13.5,1.5),np.arange(-5,6,1),
                             [0.92, 0.51,  0.017,0.34],[0.92, 0.12,  0.017,0.34])
            """

        except Exception as e:
            print('Error plot wind 850')
            # FROM: https://stackoverflow.com/questions/1483429/how-do-i-print-an-exception-in-python
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
    

    elif list_calculation[i]=='Regional_cells':

        try:

            #input
            models=np.load(path_entry+'regionalcells_models_N.npz',allow_pickle=True)['arr_0']

            hadley_mmm=seasonal_ensamble(models,path_entry,'regCirc_hadleycell',len(Lat_had_common),len(p_level_common))

            walker_mmm=seasonal_ensamble(models,path_entry,'regCirc_walkercell',len(Lon_wal_common),len(p_level_common))

            hadley_mmm_bias=seasonal_ensamble(models,path_entry,'regCirc_hadleycell_bias',len(Lat_had_common),len(p_level_common))

            walker_mmm_bias=seasonal_ensamble(models,path_entry,'regCirc_walkercell_bias',len(Lon_wal_common),len(p_level_common))

            bias_mmm_agreement_had=agreement_sign(models,path_entry,'regCirc_hadleycell_bias',\
                                            len(Lat_had_common),len(p_level_common))
            
            bias_mmm_agreement_wal=agreement_sign(models,path_entry,'regCirc_walkercell_bias',\
                                            len(Lon_wal_common),len(p_level_common))

            models_metrics_h=pd.read_csv(path_entry+'taylorDiagram_metrics_hadleycell.csv', index_col=[0])

            models_metrics_w=pd.read_csv(path_entry+'taylorDiagram_metrics_walkercell.csv', index_col=[0])

            ref_std=pd.read_csv(path_entry+'reference_std_original.csv',index_col=[0])

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
            csb=plotCells(ax9,(hadley_mmm_bias[0].T)/10e9,np.flip(Lat_had_common),p_level_plot,cmap_bias,limits_bias,\
                        'Latitude',np.flip(lat_plot),12, 'i. ','yes',title_str_size,xy_label_str, tick_labels_str_regcell,'no',bias_mmm_agreement_had[0].T)
            
            ax10 = fig.add_subplot(3, 4, 10)
            csb=plotCells(ax10,(hadley_mmm_bias[2].T)/10e9,np.flip(Lat_had_common),p_level_plot,cmap_bias,limits_bias,\
                        'Latitude',np.flip(lat_plot),12, 'j. ','no',title_str_size,xy_label_str, tick_labels_str_regcell,'no',bias_mmm_agreement_had[2].T)
            
            ax11 = fig.add_subplot(3, 4, 11)
            csb=plotCells(ax11,(hadley_mmm_bias[1].T)/10e9,np.flip(Lat_had_common),p_level_plot,cmap_bias,limits_bias,\
                        'Latitude',np.flip(lat_plot),12, 'k. ','no',title_str_size,xy_label_str, tick_labels_str_regcell,'no',bias_mmm_agreement_had[1].T)
            
            ax12 = fig.add_subplot(3, 4, 12)
            csb=plotCells(ax12,(hadley_mmm_bias[3].T)/10e9,np.flip(Lat_had_common),p_level_plot,cmap_bias,limits_bias,\
                        'Latitude',np.flip(lat_plot),12, 'l. ','no',title_str_size,xy_label_str, tick_labels_str_regcell,'no',bias_mmm_agreement_had[3].T)

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
            plt.savefig(path_save+'HadleyCell_fields.png', \
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
            path_save+'HadleyCell_fields_legend.png', format = 'png',\
            bbox_inches='tight',bbox_extra_artists=[legend_squared],
            ) 
            plt.close()

            #-----------------------------------------------------------------------------------------


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
            csb=plotCells(ax9,(walker_mmm_bias[0].T)/10e9,Lon_wal_common,p_level_plot,cmap_bias,limits_bias,'Longitude',\
                        lon_plot,16, 'i. ','yes',title_str_size,xy_label_str, tick_labels_str_regcell,'no',bias_mmm_agreement_wal[0].T)
            
            ax10 = fig.add_subplot(3, 4, 10)
            csb=plotCells(ax10,(walker_mmm_bias[2].T)/10e9,Lon_wal_common,p_level_plot,cmap_bias,limits_bias,'Longitude',\
                        lon_plot,16, 'j. ','no',title_str_size,xy_label_str, tick_labels_str_regcell,'no',bias_mmm_agreement_wal[2].T)
            
            ax11 = fig.add_subplot(3, 4, 11)
            csb=plotCells(ax11,(walker_mmm_bias[1].T)/10e9,Lon_wal_common,p_level_plot,cmap_bias,limits_bias,'Longitude',\
                        lon_plot,16, 'k. ','no',title_str_size,xy_label_str, tick_labels_str_regcell,'no',bias_mmm_agreement_wal[1].T)
            
            ax12 = fig.add_subplot(3, 4, 12)
            csb=plotCells(ax12,(walker_mmm_bias[3].T)/10e9,Lon_wal_common,p_level_plot,cmap_bias,limits_bias,'Longitude',\
                        lon_plot,16, 'l. ','no',title_str_size,xy_label_str, tick_labels_str_regcell,'no',bias_mmm_agreement_wal[3].T)

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
            plt.savefig(path_save+'WalkerCell_fields.png', \
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
            path_save+'WalkerCell_fields_legend.png', format = 'png',\
            bbox_inches='tight',bbox_extra_artists=[legend_squared],
            ) 
            plt.close()
        
        except Exception as e:
            print('Error plot regional cells')
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")

    elif list_calculation[i]=='VIMF':

        try:

            #input
            models=np.load(path_entry+'VIMF_models_N.npz',allow_pickle=True)['arr_0']

            #reference data
            east_era5=np.load(path_entry+'VIMF_ERA5_eastern_Boundary.npz',\
            allow_pickle=True)['arr_0']
            west_era5=np.load(path_entry+'VIMF_ERA5_western_Boundary.npz',\
            allow_pickle=True)['arr_0']
            north_era5=np.load(path_entry+'VIMF_ERA5_northern_Boundary.npz',\
            allow_pickle=True)['arr_0']
            south_era5=np.load(path_entry+'VIMF_ERA5_southern_Boundary.npz',\
            allow_pickle=True)['arr_0']

            #models
            north_cmip6=np.load(path_entry+'VIMF_CMIP6_northern_Boundary.npz',\
            allow_pickle=True)['arr_0']
            south_cmip6=np.load(path_entry+'VIMF_CMIP6_southern_Boundary.npz',\
            allow_pickle=True)['arr_0']
            west_cmip6=np.load(path_entry+'VIMF_CMIP6_western_Boundary.npz',\
            allow_pickle=True)['arr_0']
            east_cmip6=np.load(path_entry+'VIMF_CMIP6_eastern_Boundary.npz',\
            allow_pickle=True)['arr_0']

            print('-----------------------------------------------------------------------------------')
            print('VIMF: files read OK')
            print('-----------------------------------------------------------------------------------')

            #Generating the individual plots for each model 
            #individual_plots_boundary('VIMF',models,path_entry,path_save_ind,dx_common,dy_common)

            """
            # FROM: https://stackoverflow.com/questions/31368710/how-to-open-an-npz-file
            npz_east_cmip6 = np.load(path_entry+'VIMF_CMIP6_northern_Boundary.npz',\
              allow_pickle=True)
            print ('  models data...', type(npz_east_cmip6))
            print ('    files:', npz_east_cmip6.files)
            print ('    dir npz', dir(npz_east_cmip6))
            print ('    shape arr_0:', east_cmip6.shape)
            print ('    min', east_cmip6.min(axis=(1,2)))
            print ('    max', east_cmip6.max(axis=(1,2)))
            print ('    mean', east_cmip6.mean(axis=(1,2)))
            print ('    era5 max', east_era5.max())

            east_era5 = np.ma.masked_invalid(east_era5)
            west_era5 = np.ma.masked_invalid(west_era5)
            north_era5 = np.ma.masked_invalid(north_era5)
            south_era5 = np.ma.masked_invalid(south_era5)

            east_cmip6 = np.ma.masked_invalid(east_cmip6)
            west_cmip6 = np.ma.masked_invalid(west_cmip6)
            north_cmip6 = np.ma.masked_invalid(north_cmip6)
            south_cmip6 = np.ma.masked_invalid(south_cmip6)

            east_cmip6 = np.ma.masked_greater(east_cmip6, 700)
            west_cmip6 = np.ma.masked_greater(west_cmip6, 700)
            south_cmip6 = np.ma.masked_greater(south_cmip6, 700)
            north_cmip6 = np.ma.masked_greater(north_cmip6, 700)
            print ('    mean', east_cmip6.mean(axis=(1,2)))
            print ('  has it NaNs east_era5?', np.any(np.isnan(east_era5)))
            print ('  has it NaNs west_era5?', np.any(np.isnan(west_era5)))
            print ('  has it NaNs south_era5?', np.any(np.isnan(south_era5)))
            print ('  has it NaNs north_era5?', np.any(np.isnan(north_era5)))
            print ('  has it NaNs east_cmip6?', np.any(np.isnan(east_cmip6)))
            print ('  has it NaNs west_cmip6?', np.any(np.isnan(west_cmip6)))
            print ('  has it NaNs south_cmip6?', np.any(np.isnan(south_cmip6)))
            print ('  has it NaNs north_cmip6?', np.any(np.isnan(north_cmip6)))

            series_metrics_bound(east_era5,east_cmip6,models,'VIMF_Eastern',path_entry)
            series_metrics_bound(west_era5,west_cmip6,models,'VIMF_Western',path_entry)
            series_metrics_bound(north_era5,north_cmip6,models,'VIMF_Northern',path_entry)
            series_metrics_bound(south_era5,south_cmip6,models,'VIMF_Southern',path_entry)

            print('-----------------------------------------------------------------------------------')
            print('VIMF: series metrics OK')
            print('-----------------------------------------------------------------------------------')

            #################################################################################
            #Applying the filter to the plot 
            models, north_cmip6=filter_series_plots_bounds(north_cmip6,models,vimf_lower,vimf_upper)
            models, south_cmip6=filter_series_plots_bounds(south_cmip6,models,vimf_lower,vimf_upper)
            models, east_cmip6=filter_series_plots_bounds(east_cmip6,models,vimf_lower,vimf_upper)
            models, west_cmip6=filter_series_plots_bounds(west_cmip6,models,vimf_lower,vimf_upper)

            #Applying another filter to the plot of the models with series of one sigle value 
            models, north_cmip6=filter_series_zeros_repeated(north_cmip6,models)
            models, south_cmip6=filter_series_zeros_repeated(south_cmip6,models)
            models, east_cmip6=filter_series_zeros_repeated(east_cmip6,models)
            models, west_cmip6=filter_series_zeros_repeated(west_cmip6,models)
            """
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

            y_limits_pl=np.arange(0,550,50)

            print('-----------------------------------------------------------------------------------')
            print('VIMF: labels OK')
            print('-----------------------------------------------------------------------------------')

            #---------------------------------------------------------------------------------
            #Creating the plot 
            fig = plt.figure(figsize=(15.5,22))
            ax1 = fig.add_subplot(4, 2, 1)
            plot_boundary(ax1,'a. ', north_era5[0],models,north_cmip6[:,0,:],\
            '[Longitude]',arange_x_n,labels_plot_northern,'yes',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'No',8,y_limits_pl,['CESM2','IITM-ESM','E3SM-1-1-ECA','GISS-E2-1-H','INM-CM4-8'])

            ax2 = fig.add_subplot(4, 2, 2)
            plot_boundary(ax2,'b. ', north_era5[1],models,north_cmip6[:,1,:],\
            '[Longitude]',arange_x_n,labels_plot_northern,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'No',8,y_limits_pl,['CESM2','IITM-ESM','E3SM-1-1-ECA','GISS-E2-1-H','INM-CM4-8'])

            ax3 = fig.add_subplot(4, 2, 3)
            plot_boundary(ax3,'c. ', south_era5[0],models,south_cmip6[:,0,:],\
            '[Longitude]',arange_x_s,labels_plot_southern,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'Yes',8,y_limits_pl,['CESM2','IITM-ESM','E3SM-1-1-ECA','GISS-E2-1-H','INM-CM4-8'])

            ax4 = fig.add_subplot(4, 2, 4)
            plot_boundary(ax4,'d. ', south_era5[1],models,south_cmip6[:,1,:],\
            '[Longitude]',arange_x_s,labels_plot_southern,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'Yes',8,y_limits_pl,['CESM2','IITM-ESM','E3SM-1-1-ECA','GISS-E2-1-H','INM-CM4-8'])

            ax5 = fig.add_subplot(4, 2, 5)
            plot_boundary(ax5,'e. ', west_era5[0],models,west_cmip6[:,0,:],\
            '[Latitude]',arange_x_w,labels_plot_western,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'No',11,y_limits_pl,['CESM2','IITM-ESM','E3SM-1-1-ECA','GISS-E2-1-H','INM-CM4-8'])

            ax6 = fig.add_subplot(4, 2, 6)
            plot_boundary(ax6,'f. ', west_era5[1],models,west_cmip6[:,1,:],\
            '[Latitude]',arange_x_w,labels_plot_western,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'No',11,y_limits_pl,['CESM2','IITM-ESM','E3SM-1-1-ECA','GISS-E2-1-H','INM-CM4-8'])

            ax7 = fig.add_subplot(4, 2, 7)
            plot_boundary(ax7,'g. ', east_era5[0],models,east_cmip6[:,0,:],\
            '[Latitude]',arange_x_e,labels_plot_eastern,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'Yes',11,y_limits_pl,['CESM2','IITM-ESM','E3SM-1-1-ECA','GISS-E2-1-H','INM-CM4-8'])

            ax8 = fig.add_subplot(4, 2, 8)
            plot_boundary(ax8,'h. ', east_era5[1],models,east_cmip6[:,1,:],\
            '[Latitude]',arange_x_e,labels_plot_eastern,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'Yes',11,y_limits_pl,['CESM2','IITM-ESM','E3SM-1-1-ECA','GISS-E2-1-H','INM-CM4-8'])
            
            nrows = 20
            ncols = int(np.ceil(len(models) / float(nrows)))

            #fig.legend(bbox_to_anchor=(0.95, 0.90), ncol=2,loc='upper left', fontsize=str(legends_str))

            plt.text(0.4,1.3,'DJF', fontsize=fig_title_font,rotation='horizontal',transform=ax1.transAxes)
            plt.text(0.4,1.3,'JJA', fontsize=fig_title_font,rotation='horizontal',transform=ax2.transAxes)

            plt.text(-0.3,0.3,'Northern', fontsize=title_str_size,rotation='vertical',transform=ax1.transAxes)
            plt.text(-0.3,0.3,'Southern', fontsize=title_str_size,rotation='vertical',transform=ax3.transAxes)
            plt.text(-0.3,0.3,'Western', fontsize=title_str_size,rotation='vertical',transform=ax5.transAxes)
            plt.text(-0.3,0.3,'Eastern', fontsize=title_str_size,rotation='vertical',transform=ax7.transAxes)

            fig.savefig(path_save+'VIMF_Boundaries_Series.png', format = 'png',\
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
                path_save+'VIMF_Boundaries_Series_legend.png', format = 'png',\
            bbox_inches='tight',bbox_extra_artists=[legend_squared],
            )
            plt.close()
        
        except Exception as e:
            print('Error plot VIMF')
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
    
    elif list_calculation[i]=='MSE':

        try:

            #input
            models=np.load(path_entry+'MSE_models_N.npz',allow_pickle=True)['arr_0']

            #reference data
            east_era5=np.load(path_entry+'MSE_ERA5_eastern_Boundary.npz',\
            allow_pickle=True)['arr_0']*(1e-10)
            west_era5=np.load(path_entry+'MSE_ERA5_western_Boundary.npz',\
            allow_pickle=True)['arr_0']*(1e-10)
            north_era5=np.load(path_entry+'MSE_ERA5_northern_Boundary.npz',\
            allow_pickle=True)['arr_0']*(1e-10)
            south_era5=np.load(path_entry+'MSE_ERA5_southern_Boundary.npz',\
            allow_pickle=True)['arr_0']*(1e-10)

            #models
            north_cmip6=np.load(path_entry+'MSE_CMIP6_northern_Boundary.npz',\
            allow_pickle=True)['arr_0']*(1e-10)
            south_cmip6=np.load(path_entry+'MSE_CMIP6_southern_Boundary.npz',\
            allow_pickle=True)['arr_0']*(1e-10)
            west_cmip6=np.load(path_entry+'MSE_CMIP6_western_Boundary.npz',\
            allow_pickle=True)['arr_0']*(1e-10)
            east_cmip6=np.load(path_entry+'MSE_CMIP6_eastern_Boundary.npz',\
            allow_pickle=True)['arr_0']*(1e-10)

            print('-----------------------------------------------------------------------------------')
            print('MSE: files read OK')
            print('-----------------------------------------------------------------------------------')

            #Generating the individual plots for each model 
            #individual_plots_boundary('MSE',models,path_entry,path_save_ind,dx_common,dy_common)
        
            """
            east_era5 = np.ma.masked_invalid(east_era5)
            west_era5 = np.ma.masked_invalid(west_era5)
            north_era5 = np.ma.masked_invalid(north_era5)
            south_era5 = np.ma.masked_invalid(south_era5)

            east_cmip6 = np.ma.masked_invalid(east_cmip6)
            west_cmip6 = np.ma.masked_invalid(west_cmip6)
            north_cmip6 = np.ma.masked_invalid(north_cmip6)
            south_cmip6 = np.ma.masked_invalid(south_cmip6)

            east_cmip6 = np.ma.masked_greater(east_cmip6, 10.e20)
            west_cmip6 = np.ma.masked_greater(west_cmip6, 10.e20)
            south_cmip6 = np.ma.masked_greater(south_cmip6, 10.e20)
            north_cmip6 = np.ma.masked_greater(north_cmip6, 10.e20)
            print ('    mean', east_cmip6.mean(axis=(1,2)))
            print ('  has it NaNs east_era5?', np.any(np.isnan(east_era5)))
            print ('  has it NaNs west_era5?', np.any(np.isnan(west_era5)))
            print ('  has it NaNs south_era5?', np.any(np.isnan(south_era5)))
            print ('  has it NaNs north_era5?', np.any(np.isnan(north_era5)))
            print ('  has it NaNs east_cmip6?', np.any(np.isnan(east_cmip6)))
            print ('  has it NaNs west_cmip6?', np.any(np.isnan(west_cmip6)))
            print ('  has it NaNs south_cmip6?', np.any(np.isnan(south_cmip6)))
            print ('  has it NaNs north_cmip6?', np.any(np.isnan(north_cmip6)))

            series_metrics_bound(east_era5,east_cmip6,models,'VIHF_Eastern',path_entry)
            series_metrics_bound(west_era5,west_cmip6,models,'VIHF_Western',path_entry)
            series_metrics_bound(north_era5,north_cmip6,models,'VIHF_Northern',path_entry)
            series_metrics_bound(south_era5,south_cmip6,models,'VIHF_Southern',path_entry)

            print('-----------------------------------------------------------------------------------')
            print('MSE: series metrics OK')
            print('-----------------------------------------------------------------------------------')

            #################################################################################
            #Applying the filter to the plot 
            models, north_cmip6=filter_series_plots_bounds(north_cmip6,models,vihf_lower,vihf_upper)
            models, south_cmip6=filter_series_plots_bounds(south_cmip6,models,vihf_lower,vihf_upper)
            models, east_cmip6=filter_series_plots_bounds(east_cmip6,models,vihf_lower,vihf_upper)
            models, west_cmip6=filter_series_plots_bounds(west_cmip6,models,vihf_lower,vihf_upper)

            #Applying another filter to the plot of the models with series of one sigle value 
            models, north_cmip6=filter_series_zeros_repeated(north_cmip6,models)
            models, south_cmip6=filter_series_zeros_repeated(south_cmip6,models)
            models, east_cmip6=filter_series_zeros_repeated(east_cmip6,models)
            models, west_cmip6=filter_series_zeros_repeated(west_cmip6,models)
            """
            print('-----------------------------------------------------------------------------------')
            print('MSE: filter OK')
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

            y_limits_pl=np.arange(-15,25,5)

            print('-----------------------------------------------------------------------------------')
            print('MSE: labels OK')
            print('-----------------------------------------------------------------------------------')

            #---------------------------------------------------------------------------------
            #Creating the plot 
            fig = plt.figure(figsize=(15.5,22))
            ax1 = fig.add_subplot(4, 2, 1)
            plot_boundary(ax1,'a. ', north_era5[0],models,north_cmip6[:,0,:],\
            '[Longitude]',arange_x_n,labels_plot_northern,'yes',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'No',8,y_limits_pl,['CESM2','IITM-ESM','E3SM-1-1-ECA','INM-CM4-8','GISS-E2-1-H'])

            ax2 = fig.add_subplot(4, 2, 2)
            plot_boundary(ax2,'b. ', north_era5[1],models,north_cmip6[:,1,:],\
            '[Longitude]',arange_x_n,labels_plot_northern,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'No',8,y_limits_pl,['CESM2','IITM-ESM','E3SM-1-1-ECA','INM-CM4-8','GISS-E2-1-H'])

            ax3 = fig.add_subplot(4, 2, 3)
            plot_boundary(ax3,'c. ', south_era5[0],models,south_cmip6[:,0,:],\
            '[Longitude]',arange_x_s,labels_plot_southern,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'Yes',8,y_limits_pl,['CESM2','IITM-ESM','E3SM-1-1-ECA','INM-CM4-8','GISS-E2-1-H'])

            ax4 = fig.add_subplot(4, 2, 4)
            plot_boundary(ax4,'d. ', south_era5[1],models,south_cmip6[:,1,:],\
            '[Longitude]',arange_x_s,labels_plot_southern,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'Yes',8,y_limits_pl,['CESM2','IITM-ESM','E3SM-1-1-ECA','INM-CM4-8','GISS-E2-1-H'])

            ax5 = fig.add_subplot(4, 2, 5)
            plot_boundary(ax5,'e. ', west_era5[0],models,west_cmip6[:,0,:],\
            '[Latitude]',arange_x_w,labels_plot_western,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'No',11,y_limits_pl,['CESM2','IITM-ESM','E3SM-1-1-ECA','INM-CM4-8','GISS-E2-1-H'])

            ax6 = fig.add_subplot(4, 2, 6)
            plot_boundary(ax6,'f. ', west_era5[1],models,west_cmip6[:,1,:],\
            '[Latitude]',arange_x_w,labels_plot_western,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'No',11,y_limits_pl,['CESM2','IITM-ESM','E3SM-1-1-ECA','INM-CM4-8','GISS-E2-1-H'])

            ax7 = fig.add_subplot(4, 2, 7)
            plot_boundary(ax7,'g. ', east_era5[0],models,east_cmip6[:,0,:],\
            '[Latitude]',arange_x_e,labels_plot_eastern,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'Yes',11,y_limits_pl,['CESM2','IITM-ESM','E3SM-1-1-ECA','INM-CM4-8','GISS-E2-1-H'])

            ax8 = fig.add_subplot(4, 2, 8)
            plot_boundary(ax8,'h. ', east_era5[1],models,east_cmip6[:,1,:],\
            '[Latitude]',arange_x_e,labels_plot_eastern,'no',\
                y_label_str,title_str_size,xy_label_str, tick_labels_str,'Yes',11,y_limits_pl,['CESM2','IITM-ESM','E3SM-1-1-ECA','INM-CM4-8','GISS-E2-1-H'])
            
            nrows = 20
            ncols = int(np.ceil(len(models) / float(nrows)))

            #fig.legend(bbox_to_anchor=(0.95, 0.90), ncol=2,loc='upper left', fontsize=str(legends_str))

            plt.text(0.4,1.3,'DJF', fontsize=fig_title_font,rotation='horizontal',transform=ax1.transAxes)
            plt.text(0.4,1.3,'JJA', fontsize=fig_title_font,rotation='horizontal',transform=ax2.transAxes)

            plt.text(-0.3,0.3,'Northern', fontsize=title_str_size,rotation='vertical',transform=ax1.transAxes)
            plt.text(-0.3,0.3,'Southern', fontsize=title_str_size,rotation='vertical',transform=ax3.transAxes)
            plt.text(-0.3,0.3,'Western', fontsize=title_str_size,rotation='vertical',transform=ax5.transAxes)
            plt.text(-0.3,0.3,'Eastern', fontsize=title_str_size,rotation='vertical',transform=ax7.transAxes)

            fig.savefig(path_save+'VIHF_Boundaries_Series.png', format = 'png',\
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
                path_save+'VIHF_Boundaries_Series_legend.png', format = 'png',\
            bbox_inches='tight',bbox_extra_artists=[legend_squared],
            )
            plt.close()
        
        except Exception as e:
            print('Error plot VIHF')
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
    
    elif list_calculation[i]=='qu_qv':

        try:

            #input
            models=np.load(path_entry+'qu_qv_models_N.npz',allow_pickle=True)['arr_0']

            var_mmm_north=seasonal_ensamble(models,path_entry,'qu_qv_north',len(p_level_common),len(Lon_common_fl))

            var_mmm_south=seasonal_ensamble(models,path_entry,'qu_qv_south',len(p_level_common),len(Lon_common_fl))

            var_mmm_east=seasonal_ensamble(models,path_entry,'qu_qv_east',len(p_level_common),len(Lat_common_fl))

            var_mmm_west=seasonal_ensamble(models,path_entry,'qu_qv_west',len(p_level_common),len(Lat_common_fl))

            var_mmm_bias_north=seasonal_ensamble(models,path_entry,'qu_qv_north_bias',len(p_level_common),len(Lon_common_fl))

            var_mmm_bias_south=seasonal_ensamble(models,path_entry,'qu_qv_south_bias',len(p_level_common),len(Lon_common_fl))

            var_mmm_bias_east=seasonal_ensamble(models,path_entry,'qu_qv_east_bias',len(p_level_common),len(Lat_common_fl))

            var_mmm_bias_west=seasonal_ensamble(models,path_entry,'qu_qv_west_bias',len(p_level_common),len(Lat_common_fl))

            bias_mmm_agreement_north=agreement_sign(models,path_entry,'qu_qv_north_bias',len(p_level_common),len(Lon_common_fl))

            bias_mmm_agreement_south=agreement_sign(models,path_entry,'qu_qv_south_bias',len(p_level_common),len(Lon_common_fl))

            bias_mmm_agreement_east=agreement_sign(models,path_entry,'qu_qv_east_bias',len(p_level_common),len(Lat_common_fl))

            bias_mmm_agreement_west=agreement_sign(models,path_entry,'qu_qv_west_bias',len(p_level_common),len(Lat_common_fl))

            ################################################################################
            #PLOT
            models_metrics=pd.read_csv(path_entry+'taylorDiagram_metrics_qu_qv.csv', index_col=[0])

            ref_std=pd.read_csv(path_entry+'reference_std_original.csv',index_col=[0])

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

            plot_label='[m g/ Kg s]'

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
                csb=plotCells(ax9,var_mmm_bias_north[index_season],Lon_common_fl,p_level_plot,cmap_bias,levels_bias,'Longitude',lon_plot,10,'i.','yes',\
                            title_str_size,xy_label_str, tick_labels_str,'no',bias_mmm_agreement_north[index_season])


                ax10 = fig.add_subplot(3, 4, 10)
                csb=plotCells(ax10,var_mmm_bias_south[index_season],Lon_common_fl,p_level_plot,cmap_bias,levels_bias,'Longitude',lon_plot,10,'j.','no',\
                            title_str_size,xy_label_str, tick_labels_str,'no',bias_mmm_agreement_south[index_season])


                ax11 = fig.add_subplot(3, 4, 11)
                csb=plotCells(ax11,var_mmm_bias_east[index_season],Lat_common_fl,p_level_plot,cmap_bias,levels_bias,'Latitude',lat_plot,13,'k.','no',\
                            title_str_size,xy_label_str, tick_labels_str,'no',bias_mmm_agreement_east[index_season])

                ax12 = fig.add_subplot(3, 4, 12)
                csb=plotCells(ax12,var_mmm_bias_west[index_season],Lat_common_fl,p_level_plot,cmap_bias,levels_bias,'Latitude',lat_plot,13,'l.','no',\
                            title_str_size,xy_label_str, tick_labels_str,'no',bias_mmm_agreement_west[index_season])

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
                plt.savefig(path_save+'qu_qv_'+seasons_labels+'.png', \
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
                path_save+'qu_qv_'+seasons_labels+'_legend.png', format = 'png',\
                bbox_inches='tight',bbox_extra_artists=[legend_squared],
                ) 
                plt.close()

        except Exception as e:
            print('Error plot qu_qv')
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
    
    elif list_calculation[i]=='tu_tv':

        try:

            #input
            models=np.load(path_entry+'tu_tv_models_N.npz',allow_pickle=True)['arr_0']

            var_mmm_north=seasonal_ensamble(models,path_entry,'tu_tv_north',len(p_level_common),len(Lon_common_fl))

            var_mmm_south=seasonal_ensamble(models,path_entry,'tu_tv_south',len(p_level_common),len(Lon_common_fl))

            var_mmm_east=seasonal_ensamble(models,path_entry,'tu_tv_east',len(p_level_common),len(Lat_common_fl))

            var_mmm_west=seasonal_ensamble(models,path_entry,'tu_tv_west',len(p_level_common),len(Lat_common_fl))

            var_mmm_bias_north=seasonal_ensamble(models,path_entry,'tu_tv_north_bias',len(p_level_common),len(Lon_common_fl))

            var_mmm_bias_south=seasonal_ensamble(models,path_entry,'tu_tv_south_bias',len(p_level_common),len(Lon_common_fl))

            var_mmm_bias_east=seasonal_ensamble(models,path_entry,'tu_tv_east_bias',len(p_level_common),len(Lat_common_fl))

            var_mmm_bias_west=seasonal_ensamble(models,path_entry,'tu_tv_west_bias',len(p_level_common),len(Lat_common_fl))

            bias_mmm_agreement_north=agreement_sign(models,path_entry,'tu_tv_north_bias',len(p_level_common),len(Lon_common_fl))

            bias_mmm_agreement_south=agreement_sign(models,path_entry,'tu_tv_south_bias',len(p_level_common),len(Lon_common_fl))

            bias_mmm_agreement_east=agreement_sign(models,path_entry,'tu_tv_east_bias',len(p_level_common),len(Lat_common_fl))

            bias_mmm_agreement_west=agreement_sign(models,path_entry,'tu_tv_west_bias',len(p_level_common),len(Lat_common_fl))

            ################################################################################
            #PLOT
            models_metrics=pd.read_csv(path_entry+'taylorDiagram_metrics_tu_tv.csv', index_col=[0])

            ref_std=pd.read_csv(path_entry+'reference_std_original.csv',index_col=[0])

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

            plot_label='[ x 10³ K m/ s]'

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
                csb=plotCells(ax9,var_mmm_bias_north[index_season]/1000,Lon_common_fl,p_level_plot,cmap_bias,levels_bias,'Longitude',lon_plot,10,'i.','yes',\
                            title_str_size,xy_label_str, tick_labels_str,'no',bias_mmm_agreement_north[index_season])


                ax10 = fig.add_subplot(3, 4, 10)
                csb=plotCells(ax10,var_mmm_bias_south[index_season]/1000,Lon_common_fl,p_level_plot,cmap_bias,levels_bias,'Longitude',lon_plot,10,'j.','no',\
                            title_str_size,xy_label_str, tick_labels_str,'no',bias_mmm_agreement_south[index_season])


                ax11 = fig.add_subplot(3, 4, 11)
                csb=plotCells(ax11,var_mmm_bias_east[index_season]/1000,Lat_common_fl,p_level_plot,cmap_bias,levels_bias,'Latitude',lat_plot,13,'k.','no',\
                            title_str_size,xy_label_str, tick_labels_str,'no',bias_mmm_agreement_east[index_season])

                ax12 = fig.add_subplot(3, 4, 12)
                csb=plotCells(ax12,var_mmm_bias_west[index_season]/1000,Lat_common_fl,p_level_plot,cmap_bias,levels_bias,'Latitude',lat_plot,13,'l.','no',\
                            title_str_size,xy_label_str, tick_labels_str,'no',bias_mmm_agreement_west[index_season])

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
                plt.savefig(path_save+'tu_tv_'+seasons_labels+'.png', \
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
                path_save+'tu_tv_'+seasons_labels+'_legend.png', format = 'png',\
                bbox_inches='tight',bbox_extra_artists=[legend_squared],
                ) 
                plt.close()
        
        except Exception as e:
            print('Error plot tu_tv')
            print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")


print('############################################')
print('Plots Finished')
print('############################################')












        







    

    
    

    
