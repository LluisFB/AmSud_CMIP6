"""
This file contains all the functions to perform the calculations 
from CMIP6 models

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
import pickle
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
sns.set(style="white")
sns.set_context('notebook', font_scale=1.5)


def lat_lon_bds(lon_domain,lat_domain,var_file):
    """
    This function returns the latitude and longitude limits of the spatial
    Domain

    lat_domain=the latitudes of the domain [lower border,upper border]
    lon_domain=the longitudes of the domain [left border,rigth border]

    var_file: The file of the variable read by using xarray

    Returns:
    lat_bnds, lon_bnds = limits of the spatial Domain
    lat_name_slice,lon_name_slice = if the coordinates of the file are latitude-
    longitude or lat-lon
    """

    dims_names=list(var_file.dims)
    lat_lon_names=['lat','lon','latitude','longitude']
    if (lat_lon_names[0] in dims_names)==True:
        lat_name_slice=lat_lon_names[0]
        lon_name_slice=lat_lon_names[1]

        #for latitudes
        lat_limits=list(np.array(var_file.lat))
        upper_limit=lat_limits[0]
        if upper_limit<0.0:
            lat_bnds=[lat_domain[0],lat_domain[1]]
        else:
            lat_bnds=[lat_domain[1],lat_domain[0]]
        #for longitudes
        lon_limits=list(np.array(var_file.lon))
        left_limit=lon_limits[0]
        rigth_limit=lon_limits[-1:]
        if (left_limit>=0.0) and (rigth_limit[0]>=0.0):
            if (lon_domain[0]>=0.0) and (lon_domain[1]>0.0):
                lon_bnds=[lon_domain[0],lon_domain[1]]
            else:
                lon_domain_new_left=360+lon_domain[0]
                lon_domain_new_right=360+lon_domain[1]
                lon_bnds=[int(lon_domain_new_left),int(lon_domain_new_right)]
        else:
            if (lon_domain[0]>=0.0) and (lon_domain[1]>0.0):
                lon_domain_new_left=360-lon_domain[0]
                lon_domain_new_right=360-lon_domain[1]
                lon_bnds=[int(-lon_domain_new_left),int(-lon_domain_new_right)]
            else:
                lon_bnds=[lon_domain[0],lon_domain[1]]

    elif (lat_lon_names[2] in dims_names)==True:
        lat_name_slice=lat_lon_names[2]
        lon_name_slice=lat_lon_names[3]
        #for latitudes
        lat_limits=list(np.array(var_file.latitude))
        upper_limit=lat_limits[0]
        if upper_limit<0.0:
            lat_bnds=[lat_domain[0],lat_domain[1]]
        else:
            lat_bnds=[lat_domain[1],lat_domain[0]]
        #for longitudes
        lon_limits=list(np.array(var_file.longitude))
        left_limit=lon_limits[0]
        rigth_limit=lon_limits[-1:]
        if (left_limit>=0.0) and (rigth_limit[0]>0.0):
            if (lon_domain[0]>=0.0) and (lon_domain[1]>0.0):
                lon_bnds=[lon_domain[0],lon_domain[1]]
            else:
                lon_domain_new_left=360+lon_domain[0]
                lon_domain_new_right=360+lon_domain[1]
                lon_bnds=[int(lon_domain_new_left),int(lon_domain_new_right)]
        else:
            if (lon_domain[0]>=0.0) and (lon_domain[1]>0.0):
                lon_domain_new_left=360-lon_domain[0]
                lon_domain_new_right=360-lon_domain[1]
                lon_bnds=[int(-lon_domain_new_left),int(-lon_domain_new_right)]
            else:
                lon_bnds=[lon_domain[0],lon_domain[1]]

    else:
        print('It was not posible to select the domain limits')

    return lat_bnds,lon_bnds,lat_name_slice,lon_name_slice

def time_lat_lon_positions(first_date,last_date,lat_limits,lon_limits,lat_name,variable_file,type,level_status):
    """
    function to select the time range and the latitude and longitude domains

    returns:
    file with the spatial domain and the time range of interest
    """

    #for the spatial domain

    #for the spatial domain

    if lat_name=='lat':

        if lat_limits[0]<lat_limits[1]:
            lat_idx = np.where((variable_file.lat>=lat_limits[0])&(variable_file.lat<= lat_limits[1]))[0]
        else:
            lat_idx = np.where((variable_file.lat<=lat_limits[0])&(variable_file.lat>= lat_limits[1]))[0]
        lon_idx = np.where((variable_file.lon>=lon_limits[0])&(variable_file.lon<= lon_limits[1]))[0]

    else:
        if lat_limits[0]<lat_limits[1]:
            lat_idx = np.where((variable_file.latitude>=lat_limits[0])&(variable_file.latitude<= lat_limits[1]))[0]
        else:
            lat_idx = np.where((variable_file.latitude<=lat_limits[0])&(variable_file.latitude>= lat_limits[1]))[0]
        lon_idx = np.where((variable_file.longitude>=lon_limits[0])&(variable_file.longitude<= lon_limits[1]))[0]

    if level_status=='Yes':
        #for the time range
        if type=='ERA5':
            variable_file_delimited=variable_file[:,:,\
            lat_idx,lon_idx]
        else:
            ini_time_loc=np.where(variable_file.time.dt.strftime("%Y%m%d")==first_date)[0][0]
            fin_time_loc=np.where(variable_file.time.dt.strftime("%Y%m%d")==last_date)[0][0]
            fin_time_loc=fin_time_loc+1
            #delimiting the file
            variable_file_delimited=variable_file[ini_time_loc:fin_time_loc,:,\
            lat_idx,lon_idx]
    else:
        #for the time range
        if type=='ERA5':
            variable_file_delimited=variable_file[:,\
            lat_idx,lon_idx]
        else:
            ini_time_loc=np.where(variable_file.time.dt.strftime("%Y%m%d")==first_date)[0][0]
            fin_time_loc=np.where(variable_file.time.dt.strftime("%Y%m%d")==last_date)[0][0]
            fin_time_loc=fin_time_loc+1
            #delimiting the file
            variable_file_delimited=variable_file[ini_time_loc:fin_time_loc,\
            lat_idx,lon_idx]

    return variable_file_delimited

def data_interpolation(data_file,num_dims,Lon_old,Lat_old,Lon_new,Lat_new):
    """
    This function interpolates the entry matrix (3D or 2D) to a grid size
    previosuly delimited

    Parametros:
    data_file: array.

    num_dims: str.  '2D' o '3D'.

    Lon_old, Lat_old: arrays.

    Lon_new,Lat_new: arrays.
    """

    if num_dims=='3D':

        data_regridded=[]
        for i in range(data_file.shape[0]):
            f = interpolate.interp2d(Lon_old, Lat_old, data_file[i,:,:],kind='linear')
            znew = f(Lon_new, Lat_new)
            data_regridded.append(znew)
        data_regridded=np.array(data_regridded)

    else:
        data_regridded=[]
        f = interpolate.interp2d(Lon_old, Lat_old, data_file,kind='linear')
        znew = f(Lon_new, Lat_new)
        data_regridded.append(znew)
        data_regridded=np.array(data_regridded)

    return data_regridded

def lat_lon_lengthComparison(Lat_model,Lon_model,LatbdRef,LonbdRef,lat_refList,lon_refList,dx_model,dy_model):
    """
    This function compares the length of the list of latitude and longitude from
    the model and the reference data after changing its grid gridsize
    Lat_model, Lon_model: List of latitude and longitude of model
    LatbdRef,LonbdRef: list of latitude and longitude of the reference data
    after changing its grid size
    lat_refList,lon_reflist: ORIGINAL list of latitude and longitude of the
    reference data
    dx_model, dy_model= Grid size of the model

    Returns: list of latitudes and longitudes to use in the interpolation of
    the reference data
    """
    if len(LonbdRef)==len(Lon_model) and len(LatbdRef)==len(Lat_model):
        print('The length of the coordinates lists is the same')
    else:
        if len(LonbdRef)!=len(Lon_model):
            if len(LonbdRef)>len(Lon_model):
                diff=len(LonbdRef)-len(Lon_model)
                LonbdRef = np.arange(lon_refList[0],lon_refList[-1:][0]-\
                (diff*dx_model), dx_model)
            else:
                diff=len(Lon_model)-len(LonbdRef)
                LonbdRef = np.arange(lon_refList[0],lon_refList[-1:][0]+\
                (diff*dx_model), dx_model)
        if len(LatbdRef)!=len(Lat_model):
            if len(LatbdRef)>len(Lat_model):
                diff=len(LatbdRef)-len(Lat_model)
                LatbdRef = np.arange(lat_refList[0], lat_refList[-1:][0]-\
                (diff*dy_model), dy_model)
            else:
                diff=len(Lat_model)-len(LatbdRef)
                LatbdRef = np.arange(lat_refList[0], lat_refList[-1:][0]+\
                (diff*dy_model), dy_model)

    return LonbdRef,LatbdRef

def levels_limit(var_file,level_0,level_1):

    """
    This function return the location of the pressure levels
    level_0 and level_1 are integer of the levels in Pa
    """
    if 'plev' in list(var_file.dims):
        units=var_file.plev.units
        list_levels=np.round(np.array(var_file.plev),1)
    else:
        units=var_file.level.units
        list_levels=np.round(np.array(var_file.level),1)

    if units=='hPa' or units=='millibars':
        level_0=level_0/100
        level_1=level_1/100

    else:
        pass

    level0_position=np.where(list_levels==level_0)[0][0]
    level1_position=np.where(list_levels==level_1)[0][0]

    if level0_position<level1_position:
        lower_level=level0_position
        upper_level=level1_position
    else:
        lower_level=level1_position
        upper_level=level0_position

    return lower_level,upper_level

def taylor_diagram_metrics_def(obs_matrix,modeled_matrix):
    std_arr=np.empty((4))
    corr_arr=np.empty((4))

    if np.isnan(modeled_matrix).any()==True:
        for i in range(4):
        
            correlation=ma.corrcoef(ma.masked_invalid(obs_matrix[i].flatten()), \
            ma.masked_invalid(modeled_matrix[i].flatten()))[0,1]
            std_model=np.nanstd(modeled_matrix[i])

            corr_arr[i]=correlation
            std_arr[i]=std_model
    
    else:

        for i in range(4):
            correlation=np.corrcoef(obs_matrix[i].flatten(), modeled_matrix[i].flatten())[0,1]
            std_model=np.nanstd(modeled_matrix[i])

            corr_arr[i]=correlation
            std_arr[i]=std_model

    return corr_arr,std_arr

def seasonal_ensamble(list_models,path_entry_npz,file_name,len_lats, len_lons):
    var_seasonal_ensamble=np.empty((4,len_lats,len_lons))

    for i in range(4):
        var_ensamble=np.empty((len(list_models),len_lats,len_lons))

        for p in range(len(list_models)):
            var_model_season=np.load(path_entry_npz+list_models[p]+'_'+\
            file_name+'.npz')['arr_0']

            #selecting the specific season

            var_ensamble[p]=var_model_season[i]

        #averaging the models
        var_mean=np.mean(var_ensamble,axis=0)

        #saving in the initial arrays +
        var_seasonal_ensamble[i]=var_mean

    return var_seasonal_ensamble

class TaylorDiagram(object):
    """Taylor diagram: plot model standard deviation and correlation
    to reference (data) sample in a single-quadrant polar plot, with
    r=stddev and theta=arccos(correlation).
    source: https://gist.github.com/ycopin/3342888
    """

    def __init__(self, refstd, fig=None, rect=111, label='_'):
        """Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using mpl_toolkits.axisartist.floating_axes. refstd is
        the reference standard deviation to be compared to.
        """

        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        self.refstd = refstd          # Reference standard deviation

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = np.concatenate((np.arange(10)/10.,[0.95,0.99]))
        tlocs = np.arccos(rlocs)        # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)    # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str,rlocs))))

        # Standard deviation axis extent
        self.smin = 0
        #self.smax = 1.5*self.refstd
        self.smax = 1.7

        ghelper = FA.GridHelperCurveLinear(tr,
                                           extremes=(0,np.pi/2, # 1st quadrant
                                                     self.smin,self.smax),
                                           grid_locator1=gl1,
                                           tick_formatter1=tf1,
                                           )

        if fig is None:
            fig = plt.figure()

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)
        #ax = fig.add_subplot(rect)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")  # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom") # "X axis"
        #ax.axis["left"].label.set_text("Standard deviation (Normalized)")
        ax.axis["left"].label.set_text("Std (Normalized)")

        ax.axis["right"].set_axis_direction("top")   # "Y axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")

        ax.axis["bottom"].set_visible(False)         # Useless

        # Contours along standard deviations
        ax.grid(False)

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates

        # Add reference point and stddev contour
        print("Reference std:", self.refstd)
        l, = self.ax.plot([0], self.refstd, 'k*',
                          ls='', ms=10, label=label)
        t = np.linspace(0, np.pi/2)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t,r, 'k--', label='_')

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """Add sample (stddev,corrcoeff) to the Taylor diagram. args
        and kwargs are directly propagated to the Figure.plot
        command."""

        l, = self.ax.plot(np.arccos(corrcoef), stddev,
                          *args, **kwargs) # (theta,radius)
        self.samplePoints.append(l)

        return l

    def add_contours(self, levels=5, **kwargs):
        """Add constant centered RMS difference contours."""

        rs,ts = np.meshgrid(np.linspace(self.smin,self.smax),
                            np.linspace(0,np.pi/2))
        # Compute centered RMS difference
        rms = np.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*np.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours

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
    list_markers=['o','P','X','D']
    n_repeat=math.ceil(number_models/len(list_markers))
    list_markers_repeated=list_markers*n_repeat
    markers=list_markers_repeated[0:number_models]

    colors = plt.matplotlib.cm.Set1(np.linspace(0,1,number_models))

    dia = TaylorDiagram(ref_standar/ref_standar, fig=fig,rect=rects,label='Reference')


    # Add samples to Taylor diagram
    for i,(stddev,corrcoef,name) in enumerate(sample_models):
        dia.add_sample(stddev, corrcoef,
                       marker=markers[i] , ms=10, ls='',
                       #mfc='k', mec='k', # B&W
                       mfc=colors[i], mec=colors[i], # Colors
                       label=name)

    # Add RMS contours, and label them
    contours = dia.add_contours(levels=5, colors='0.5') # 5 levels
    dia.ax.clabel(contours, inline=1, fontsize=10, fmt='%.1f')
    # Tricky: ax is the polar ax (used for plots), _ax is the
    # container (used for layout)
    dia._ax.set_title(title_label,fontsize=title_size,loc='left', pad=20)

    nrows = 8
    ncols = int(np.ceil(len(dia.samplePoints) / float(nrows)))


    fig.legend(dia.samplePoints,
               [ p.get_label() for p in dia.samplePoints ],
               numpoints=1, prop=dict(size='medium'),bbox_to_anchor=(1.03, 0.85) \
               ,ncol=ncols,loc='right')

    #fig.tight_layout()


    return dia

def NaNs_interp(array, dims, interp_type):

    if dims=='2D':

        x = np.arange(0, array.shape[1])
        y = np.arange(0, array.shape[0])
        #mask invalid values
        array_masked = np.ma.masked_invalid(array)
        xx, yy = np.meshgrid(x, y)
        #get only the valid values
        x1 = xx[~array_masked.mask]
        y1 = yy[~array_masked.mask]
        newarr = array_masked[~array_masked.mask]

        GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                                  (xx, yy),
                                     method=interp_type)
        noNaN_arr=GD1

    else:
        noNaN_arr=np.empty((array.shape))

        for i in range(array.shape[0]):
            x = np.arange(0, array[i].shape[1])
            y = np.arange(0, array[i].shape[0])
            #mask invalid values
            array_masked = np.ma.masked_invalid(array[i])
            xx, yy = np.meshgrid(x, y)
            #get only the valid values
            x1 = xx[~array_masked.mask]
            y1 = yy[~array_masked.mask]
            newarr = array_masked[~array_masked.mask]

            GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                                      (xx, yy),
                                         method=interp_type)
            noNaN_arr[i]=GD1

    return noNaN_arr

def plotMap(axs,var_data,lonPlot,latPlot,colorMap,limits,title_label,extent, projection,title_font2,scatter_status,points_scatter,land_cov):
    """
    This function creates the maps of wind circulation for seasons and months of
    the year
    """
    #niveles=np.arange(-20,21,1)
    widths = np.linspace(0, 2, lonPlot.size)
    axs.set_title(title_label,fontsize=title_font2,loc='left')
    axs.set_extent(extent, projection)
    axs.add_feature(cfeature.COASTLINE)
    axs.add_feature(cfeature.BORDERS, linestyle=':')
    if land_cov=='yes':
        axs.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
    cs=axs.contourf(lonPlot, latPlot,var_data,limits,cmap=colorMap,extend='both')
    if scatter_status=='yes':
        sc=axs.scatter(lonPlot,latPlot,points_scatter/3,transform=projection,zorder=2, c='grey')
    # regrid_shape=40,scale=500,scale_units='width', linewidths=widths
    gl=axs.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    gl.ylines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 13.5}
    gl.ylabel_style = {'size': 13.5}

    return cs

def plotMap_vector(axs,wind_data,u_data,v_data,lonPlot,latPlot,colorMap,limits,title_label,extent, projection,title_font2):
    """
    This function creates the maps of wind circulation for seasons and months of
    the year
    """
    
    widths = np.linspace(0, 2, lonPlot.size)
    axs.set_title(title_label,fontsize=title_font2,loc='left')
    axs.set_extent(extent, projection)
    axs.add_feature(cfeature.COASTLINE)
    axs.add_feature(cfeature.BORDERS, linestyle=':')
    cs=axs.contourf(lonPlot, latPlot,wind_data,limits,cmap=colorMap,extend='both')
    axs.quiver(lonPlot[::3,::3], latPlot[::3,::3],u_data[::3,::3],v_data[::3,::3],transform=ccrs.PlateCarree())
    # regrid_shape=40,scale=500,scale_units='width', linewidths=widths
    gl=axs.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    gl.ylines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 13.5}
    gl.ylabel_style = {'size': 13.5}

    return cs

def var_field_calc(path_entry,var_sp,model_name,lat_d,lon_d,time_0,time_1,level_lower,level_upper,type_data,level_status):

    var_data=xr.open_mfdataset(path_entry+model_name+'_'+var_sp+'_original_seasonal_mean.nc')

    var_field=var_data[var_sp]

    print('##############################')
    print("var_field_calc: read file OK")
    print('##############################')

    #obtaining the domain bnds
    lat_bnd,lon_bnd,lat_slice,lon_slice=lat_lon_bds(lon_d,lat_d,var_field)
    #delimiting the spatial domain
    var_delimited=time_lat_lon_positions(time_0,time_1,lat_bnd,\
    lon_bnd,lat_slice,var_field,type_data,level_status)

    print('##############################')
    print("var_field_calc: var delimited OK")
    print('##############################')


    if level_status=='No':
        var_levels=var_delimited
    else:
        #selecting the pressure level
        ini_level,fin_level=levels_limit(var_delimited,level_lower,level_upper)
        var_levels=var_delimited[:,int(ini_level):int(fin_level+1),:,:][:,0,:,:]

    #converting into array
    var_array=np.array(var_levels)
    print('##############################')
    print("var_field_calc: var_array OK")
    print('##############################')

    #defining Lat and Lon lists
    if lat_slice=='lat':
        Lat_list=list(np.array(var_delimited.lat))
        Lon_list=list(np.array(var_delimited.lon))
    else:
        Lat_list=list(np.array(var_delimited.latitude))
        Lon_list=list(np.array(var_delimited.longitude))

    #defining the grid size
    dx_data=np.round(abs(Lon_list[1])-abs(Lon_list[0]),2)
    dy_data=np.round(abs(Lat_list[-1:][0])-  abs(Lat_list[-2:][0]),2)

    return var_array,Lat_list,Lon_list,dx_data, dy_data

def wind_field_calc(path_entry,var_sp1,var_sp2,model_name,lat_d,lon_d,time_0,time_1,level_lower,level_upper,type_data):

    var_data1=xr.open_mfdataset(path_entry+model_name+'_'+var_sp1+'_original_seasonal_mean.nc')

    u_field=var_data1[var_sp1]

    var_data2=xr.open_mfdataset(path_entry+model_name+'_'+var_sp2+'_original_seasonal_mean.nc')

    v_field=var_data2[var_sp2]

    print('##############################')
    print("wind_field_calc: read file OK")
    print('##############################')

    #obtaining the domain bnds
    lat_bnd,lon_bnd,lat_slice,lon_slice=lat_lon_bds(lon_d,lat_d,u_field)
    #delimiting the spatial domain
    u_delimited=time_lat_lon_positions(time_0,time_1,lat_bnd,\
    lon_bnd,lat_slice,u_field,type_data,'Yes')

    v_delimited=time_lat_lon_positions(time_0,time_1,lat_bnd,\
    lon_bnd,lat_slice,v_field,type_data,'Yes')

    #selecting the pressure level
    ini_level,fin_level=levels_limit(u_delimited,level_lower,level_upper)
    u_levels=u_delimited[:,int(ini_level):int(fin_level+1),:,:]
    v_levels=v_delimited[:,int(ini_level):int(fin_level+1),:,:]

    print('##############################')
    print("wind_field_calc: var_levels OK")
    print('##############################')

    #converting into array
    u_array=np.array(u_levels)[:,0,:,:]
    v_array=np.array(v_levels)[:,0,:,:]

    mag_arr=np.sqrt(u_array**2+v_array**2)

    print('##############################')
    print("wind_field_calc: mag arr OK")
    print('##############################')

    #defining Lat and Lon lists
    if lat_slice=='lat':
        Lat_list=list(np.array(u_delimited.lat))
        Lon_list=list(np.array(u_delimited.lon))
    else:
        Lat_list=list(np.array(u_delimited.latitude))
        Lon_list=list(np.array(u_delimited.longitude))

    #defining the grid size
    dx_data=np.round(abs(Lon_list[1])-abs(Lon_list[0]),2)
    dy_data=np.round(abs(Lat_list[-1:][0])-  abs(Lat_list[-2:][0]),2)


    return u_array,v_array,mag_arr,Lat_list,Lon_list,dx_data, dy_data

def netcdf_creation_original(path_entry_files,var_name,ft,lat_d,lon_d,time_0,time_1,level_lower,level_upper,type_data,level_status,path_save,model_name):

    if var_name=='tos':
        var_data=xr.open_mfdataset(path_entry_files+'tos_cmip6_'+model_name+'_historical*.nc')
    else:

        path_files_old=path_entry_files+'historical/r1i1p1f1/'+ft+'/'+var_name+'/'

        grid_list=os.listdir(path_files_old)

        if len(grid_list)>1:
            if 'gn' in grid_list:
                grid_type='gn'
        else:
            grid_type=grid_list[0]
        
        path_files=path_files_old+grid_type+'/latest/'
        
        path_to_print=path_files+var_name+'_'+ft+'_'+model_name+'_historical_*.nc'

        var_data=xr.open_mfdataset(path_files+var_name+'_'+ft+'_'+model_name+'_historical_*.nc')
        
    var_field=var_data[var_name]

    print('========================================================')
    print('The data was read successfully')
    print('========================================================')

    #obtaining the domain bnds
    lat_bnd,lon_bnd,lat_slice,lon_slice=lat_lon_bds(lon_d,lat_d,var_field)
    #Checking the time bands
    t_0_new='197901'+str(var_field[0].time.dt.strftime("%Y%m%d").values)[-2::]
    t_1_new='2014'+str(var_field[-1].time.dt.strftime("%Y%m%d").values)[4:6]+\
    str(var_field[-1].time.dt.strftime("%Y%m%d").values)[-2::]
    #delimiting the spatial domain
    var_delimited=time_lat_lon_positions(t_0_new,t_1_new,lat_bnd,\
    lon_bnd,lat_slice,var_field,type_data,level_status)

    print('========================================================')
    print('var delimited done succesfully')
    print('========================================================')

    #-----------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------
    #Assessing the order of the
    #latitudes and pressure levels
    #Pressure level: from top to surface
    if level_status=='No':
        pass 
    else:
        if np.array(var_delimited.plev)[0]>np.array(var_delimited.plev)[-1]:
            var_delimited=var_delimited.reindex(plev=list(reversed(var_delimited.plev)))
        else:
            pass
    #Latitude: From south to north
    if lat_slice=='latitude':
        if np.array(var_delimited.latitude)[0]>np.array(var_delimited.latitude)[-1]:
            var_delimited=var_delimited.reindex(latitude=list(reversed(var_delimited.latitude)))
        else:
            pass
    else:
        if np.array(var_delimited.lat)[0]>np.array(var_delimited.lat)[-1]:
            var_delimited=var_delimited.reindex(lat=list(reversed(var_delimited.lat)))
        else:
            pass

    print('========================================================')
    print('Changes in level and latitude order done succesfully')
    print('========================================================')
    
    #----------------------------------------------------------------------------------

    if level_status=='No':
        var_levels=var_delimited
    else:
        #selecting the pressure level
        ini_level,fin_level=levels_limit(var_delimited,level_lower,level_upper)
        var_levels=var_delimited[:,int(ini_level):int(fin_level+1),:,:]

    #grouping by season
    var_seasonal=var_levels.groupby('time.season').mean('time')

    print('========================================================')
    print('Seasonal mean done succesfully')
    print('========================================================')

    #--------------------------------------------------------------------------------------
    #1. Saving the netcdf of the climatology of the variable 
    var_levels.to_netcdf(path_save+model_name+'_'+var_name+'_original_mon_clim_LT.nc')
    #2. Saving the netcdf of the seasonal mean of the variable
    var_seasonal.to_netcdf(path_save+model_name+'_'+var_name+'_original_seasonal_mean.nc')

    print('========================================================')
    print('Files (seasonal and long-term) saved succesfully')
    print('========================================================')

    #defining Lat and Lon lists
    if lat_slice=='lat':
        Lat_list=list(np.array(var_delimited.lat))
        Lon_list=list(np.array(var_delimited.lon))
    else:
        Lat_list=list(np.array(var_delimited.latitude))
        Lon_list=list(np.array(var_delimited.longitude))

    #defining the grid size
    dx_data=np.round(abs(Lon_list[1])-abs(Lon_list[0]),2)
    dy_data=np.round(abs(Lat_list[-1:][0])-  abs(Lat_list[-2:][0]),2)

    print('========================================================')
    print('Lat and Lon list and grid size defined succesfully')
    print('========================================================')

    return  dx_data, dy_data, path_to_print

def netcdf_creation_original_ERA5(path_entry_files,var_name,lat_d,lon_d,level_lower,level_upper,type_data,level_status,path_save):

    range_years=np.arange(1979,2015,1)

    file_con=[]

    if level_status=='No':
        nm='asme5'
    else:
        nm='apme5'

    for b in range(len(range_years)):

        var_data_year=xr.open_dataset(path_entry_files+str(range_years[b])+'/'+var_name+'.'+str(range_years[b])+'.'+nm+'.GLOBAL_025.nc')
        var_data=var_data_year[var_name]
        file_con.append(var_data)

    var_field=xr.concat(file_con, dim='time')
    
    #obtaining the domain bnds
    lat_bnd,lon_bnd,lat_slice,lon_slice=lat_lon_bds(lon_d,lat_d,var_field)

    
    #delimiting the spatial domain
    if lat_slice=='lat':
        var_delimited=var_field.sel(lat=slice(*lat_bnd),lon=slice(*lon_bnd))
    else:
        var_delimited=var_field.sel(latitude=slice(*lat_bnd),longitude=slice(*lon_bnd))

    #-----------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------
    #Assessing the order of the
    #latitudes and pressure levels
    #Pressure level: from top to surface
    if level_status=='No':
        pass 
    else:
        if np.array(var_delimited.level)[0]>np.array(var_delimited.level)[-1]:
            var_delimited=var_delimited.reindex(level=list(reversed(var_delimited.level)))
        else:
            pass
    #Latitude: From south to north
    if lat_slice=='latitude':
        if np.array(var_delimited.latitude)[0]>np.array(var_delimited.latitude)[-1]:
            var_delimited=var_delimited.reindex(latitude=list(reversed(var_delimited.latitude)))
        else:
            pass
    else:
        if np.array(var_delimited.lat)[0]>np.array(var_delimited.lat)[-1]:
            var_delimited=var_delimited.reindex(lat=list(reversed(var_delimited.lat)))
        else:
            pass
    
    #----------------------------------------------------------------------------------

    if level_status=='No':
        var_levels=var_delimited
    else:
        #selecting the pressure level
        ini_level,fin_level=levels_limit(var_delimited,level_lower,level_upper)
        var_levels=var_delimited[:,int(ini_level):int(fin_level+1),:,:]

    #grouping by season
    var_seasonal=var_levels.groupby('time.season').mean('time')

    #--------------------------------------------------------------------------------------
    #1. Saving the netcdf of the climatology of the variable 
    var_levels.to_netcdf(path_save+'ERA5_'+var_name+'_original_mon_clim_LT.nc')
    #2. Saving the netcdf of the seasonal mean of the variable
    var_seasonal.to_netcdf(path_save+'ERA5_'+var_name+'_original_seasonal_mean.nc')

     #defining Lat and Lon lists
    if lat_slice=='lat':
        Lat_list=list(np.array(var_delimited.lat))
        Lon_list=list(np.array(var_delimited.lon))
    else:
        Lat_list=list(np.array(var_delimited.latitude))
        Lon_list=list(np.array(var_delimited.longitude))

    #defining the grid size
    dx_data=np.round(abs(Lon_list[1])-abs(Lon_list[0]),2)
    dy_data=np.round(abs(Lat_list[-1:][0])-  abs(Lat_list[-2:][0]),2)

    return  dx_data, dy_data

def variables_availability(path_models,var_str,domain):

    models_name=[]

    #Obtaining the list of files in the path
    lists_path=os.listdir(path_models)

    for r in range(len(lists_path)):
        model_fol=lists_path[r]
        path_mod=path_models+model_fol+'/'

        #Obtaining the version of the model 
        list_ver=os.listdir(path_mod)

        for g in range(len(list_ver)):

            try:
                mod_ver=list_ver[g]
                path_ver=path_mod+mod_ver+'/historical/r1i1p1f1/'+domain+'/'

                list_var=os.listdir(path_ver)

                #Checking if the varable is in the list
                if var_str in list_var:
                    models_name.append(mod_ver)

            except:
                print('Path not found')
    #---------------------------------------------------------
    #---------------------------------------------------------
    models_list=list(set(models_name))

    return models_list
    
def cdo_remapbill(path_entry_files,model_name,path_save_files):

    path_files_old=path_entry_files+'historical/r1i1p1f1/Omon/tos/'

    grid_list=os.listdir(path_files_old)

    if len(grid_list)>1:
        if 'gn' in grid_list:
            grid_type='gn'
    else:
        grid_type=grid_list[0]
    
    path_files=path_files_old+grid_type+'/latest/'

    print('###################################################')
    print(path_files)
    print('###################################################')

    remapbill_info=xr.open_mfdataset(path_files+'tos_Omon_'+model_name+'_historical_*.nc')
    m_sst_grid_var=remapbill_info['tos']

    print('###################################################')
    print('cdo_remapbill: read files OK')
    print('###################################################')

    m_grid_lat=m_sst_grid_var[m_sst_grid_var.dims[1]]
    m_grid_lon=m_sst_grid_var[m_sst_grid_var.dims[2]]
    m_grid_units=m_sst_grid_var.units

    xsize=np.array(m_grid_lon).shape[0]
    ysize=np.array(m_grid_lat).shape[0]

    print('###################################################')
    print('cdo_remapbill: xsize,ysize OK')
    print('###################################################')

    #Generating the list of the files 
    list_path=os.listdir(path_entry_files)

    files_var=[]

    for ñ in range(len(list_path)):
        if 'tos_Omon_'+model_name in list_path[ñ]:
            files_var.append(list_path[ñ])
    
    files_var=sorted(files_var)

    print('###################################################')
    print('cdo_remapbill: files var OK')
    print('###################################################')
    
    #remaping

    if len(files_var)>1:
        for ñ in range(len(files_var)):
            oras_i=files_var[ñ]
            oras_out='tos_cmip6_'+model_name+'_historical_'+str(ñ)+'.nc'

            cdo_remap='cdo -remapbil,r'+str(xsize)+'x'+str(ysize)+' '+path_files+oras_i+' '+path_save_files+oras_out

            print('###################################################')
            print(cdo_remap)
            print('###################################################') 

            os.system(cdo_remap) 

        print('###################################################')
        print('cdo_remapbill: remapbill os some files OK')
        print('###################################################') 

    else:
        #oras_i='tos_Omon_'+model_name+'_historical_r1i1p1f1_gn_185001-201412.nc'
        oras_i=files_var[0]
        oras_out='tos_cmip6_'+model_name+'_historical.nc'

        cdo_remap='cdo -remapbil,r'+str(xsize)+'x'+str(ysize)+' '+path_files+oras_i+' '+path_save_files+oras_out

        print('###################################################')
        print(cdo_remap)
        print('###################################################') 

        os.system(cdo_remap)   

        print('###################################################')
        print('cdo_remapbill: remapbill os one file OK')
        print('###################################################') 

def subtropical_jet(path_entry,var_sp,model_name, type_dataset,lon_limits,lat_limits, pressure_0,pressure_1):

    var_data=xr.open_mfdataset(path_entry+model_name+'_'+var_sp+'_original_mon_clim_LT.nc')

    dataset=var_data[var_sp]

    #delimiting the temporal range and spatial domain
    lat_bnd,lon_bnd,lat_slice,lon_slice=lat_lon_bds(lon_limits,\
    lat_limits, dataset)

    var_delimited=time_lat_lon_positions(None,None,lat_bnd,\
    lon_bnd,lat_slice,dataset,'ERA5','Yes')

    #grouping by month
    var_monthly=var_delimited.groupby('time.month').mean('time')
    ################################################################################
    #Selecting the pressure level
    ini_level,fin_level=levels_limit(var_monthly,pressure_0,pressure_1)

    var_levels=var_monthly[:,int(ini_level):int(fin_level)+1,:,:]

    #Generating the lists of latitudes and longitudes
    if lat_slice=='lat':
        Lat=list(np.array(var_levels.lat))
        Lon=list(np.array(var_levels.lon))
    else:
        Lat=list(np.array(var_levels.latitude))
        Lon=list(np.array(var_levels.longitude))

    ################################################################################
    var_array=np.array(var_levels)[:,0,:,:]

    #Creating the empty arrays to save the information
    jet_strength=np.empty((var_array.shape[0],var_array.shape[2]))
    jet_latitude=np.empty((var_array.shape[0],var_array.shape[2]))

    #Creating loop to iterate in time and longitudes

    for i in range(var_array.shape[0]):

        for j in range(var_array.shape[2]):

            u_lon=var_array[i,:,j]

            max_wind=u_lon.max()
            lat_max=Lat[np.where(u_lon==max_wind)[0][0]]

            jet_strength[i,j]=max_wind
            jet_latitude[i,j]=lat_max

    ################################################################################
    #Averaging to obtain the mean values for each month
    jet_strength_month=np.mean(jet_strength,axis=1)
    jet_latitude_month=np.mean(jet_latitude,axis=1)

    return jet_strength_month, jet_latitude_month

def westerlies(path_entry,var_sp,model_name, type_dataset,lon_limits,lat_limits, pressure_0,pressure_1):

    var_data=xr.open_mfdataset(path_entry+model_name+'_'+var_sp+'_original_mon_clim_LT.nc')

    dataset=var_data[var_sp]

    #delimiting the temporal range and spatial domain
    lat_bnd,lon_bnd,lat_slice,lon_slice=lat_lon_bds(lon_limits,\
    lat_limits, dataset)

    var_delimited=time_lat_lon_positions(None,None,lat_bnd,\
    lon_bnd,lat_slice,dataset,'ERA5','Yes')

    #grouping by month
    var_monthly=var_delimited.groupby('time.month').mean('time')

    ################################################################################
    #Selecting the pressure level
    ini_level,fin_level=levels_limit(var_monthly,pressure_0,pressure_1)

    var_levels=var_monthly[:,int(ini_level):int(fin_level)+1,:,:]

    #Generating the lists of latitudes and longitudes
    if lat_slice=='lat':
        #averaging in the longitudes
        var_lon_mean=var_levels.mean(dim='lon')

        Lat=list(np.array(var_levels.lat))
    else:
        #averaging in the longitudes
        var_lon_mean=var_levels.mean(dim='longitude')

        Lat=list(np.array(var_levels.latitude))

    ################################################################################
    #Converting into array
    var_array=np.array(var_lon_mean)[:,0,:]

    ################################################################################
    #Calculating the maximum and the latitude
    westerly_strength=np.empty((12))
    westerly_latitude=np.empty((12))

    for i in range(var_array.shape[0]):
        month_series=var_array[i,:]

        #jet strenght
        u_max=month_series.max()

        #jet latitude
        lat_max=Lat[np.where(month_series==u_max)[0][0]]

        #saving the information in the arrays
        westerly_strength[i]=u_max
        westerly_latitude[i]=lat_max

    return westerly_strength, westerly_latitude   

def tradeWind(path_entry,var_sp,model_name, type_dataset,lon_limits,lat_limits, pressure_0,pressure_1):
    
    var_data=xr.open_mfdataset(path_entry+model_name+'_'+var_sp+'_original_mon_clim_LT.nc')

    dataset=var_data[var_sp]

    #delimiting the temporal range and spatial domain
    lat_bnd,lon_bnd,lat_slice,lon_slice=lat_lon_bds(lon_limits,\
    lat_limits, dataset)

    var_delimited=time_lat_lon_positions(None,None,lat_bnd,\
    lon_bnd,lat_slice,dataset,'ERA5','Yes')

    #grouping by month
    var_monthly=var_delimited.groupby('time.month').mean('time')

    ################################################################################
    #Selecting the pressure level
    ini_level,fin_level=levels_limit(var_monthly,pressure_0,pressure_1)

    var_levels=var_monthly[:,int(ini_level):int(fin_level)+1,:,:]

    ############################################################################
    ############################################################################
    #averaging in the regions of interest to calculate the indices
    #spatial average
    if lat_slice=='lat':
        levels_lon_var=var_levels.mean(dim='lon')
        levels_spatial_var=levels_lon_var.mean(dim='lat')

    else:
        levels_lon_var=var_levels.mean(dim='longitude')
        levels_spatial_var=levels_lon_var.mean(dim='latitude')

    #converting into array
    index_array_model=np.array(levels_spatial_var)*(-1)

    return index_array_model

def subtropicalHighs(path_entry,var_sp,model_name, type_dataset, lon_limits, lat_limits):

    var_data=xr.open_mfdataset(path_entry+model_name+'_'+var_sp+'_original_mon_clim_LT.nc')

    dataset=var_data[var_sp]

    #delimiting the temporal range and spatial domain
    lat_bnd,lon_bnd,lat_slice,lon_slice=lat_lon_bds(lon_limits,\
    lat_limits, dataset)

    data_delimited=time_lat_lon_positions(None,None,lat_bnd,\
    lon_bnd,lat_slice,dataset,'ERA5','No')

    #grouping by month
    data_monthly=data_delimited.groupby('time.month').mean('time')

    #Generating the lists of latitudes and longitudes
    if lat_slice=='lat':
        Lat=list(np.array(data_monthly.lat))
        Lon=list(np.array(data_monthly.lon))
    else:
        Lat=list(np.array(data_monthly.latitude))
        Lon=list(np.array(data_monthly.longitude))

    ############################################################################
    if type_dataset=='reference':
        data_array=np.array(data_monthly)[:,:,:]/100
    else:
        data_array=np.array(data_monthly)[:,:,:]/100

    ############################################################################
    #Calculating the index
    #Calculing the index for the south atlantic subtropical high
    subtropical_strength=np.empty((12))
    subtropical_lat=np.empty((12))
    subtropical_lon=np.empty((12))

    for p in range(data_array.shape[0]):

        #comparing the maximum values
        max=np.nanmax(data_array[p])

        subtropical_strength[p]=max

        lat_loc=np.where(data_array[p]==max)[0]
        lon_loc=np.where(data_array[p]==max)[1]

        #evaluating the size of the location arrays
        if lat_loc.shape[0]>1:
            lat_loc_mean=np.mean(lat_loc)
        else:
            lat_loc_mean=lat_loc[0]

        if lon_loc.shape[0]>1:
            lon_loc_mean=np.mean(lon_loc)
        else:
            lon_loc_mean=lon_loc[0]

        subtropical_lat[p]=Lat[int(np.round(lat_loc_mean,0))]
        subtropical_lon[p]=Lon[int(np.round(lon_loc_mean,0))]

    return subtropical_strength, subtropical_lat, subtropical_lon

def subtropicalHighs_core_Seasonal(path_entry,var_sp,model_name, type_dataset, lon_limits, lat_limits):

    var_data=xr.open_mfdataset(path_entry+model_name+'_'+var_sp+'_original_seasonal_mean.nc')

    dataset=var_data[var_sp]

    #delimiting the temporal range and spatial domain
    lat_bnd,lon_bnd,lat_slice,lon_slice=lat_lon_bds(lon_limits,\
    lat_limits, dataset)

    data_delimited=time_lat_lon_positions(None,None,lat_bnd,\
    lon_bnd,lat_slice,dataset,'ERA5','No')

    #Generating the lists of latitudes and longitudes
    if lat_slice=='lat':
        Lat=list(np.array(data_delimited.lat))
        Lon=list(np.array(data_delimited.lon))
    else:
        Lat=list(np.array(data_delimited.latitude))
        Lon=list(np.array(data_delimited.longitude))

    ############################################################################
    if type_dataset=='reference':
        data_array=np.array(data_delimited)[:,:,:]/100
    else:
        data_array=np.array(data_delimited)[:,:,:]/100

    ############################################################################
    #Calculating the index
    #Calculing the index for the south atlantic subtropical high
    subtropical_strength=np.empty((4))
    subtropical_lat=np.empty((4))
    subtropical_lon=np.empty((4))

    for p in range(data_array.shape[0]):

        #comparing the maximum values
        max=np.nanmax(data_array[p])

        subtropical_strength[p]=max

        lat_loc=np.where(data_array[p]==max)[0]
        lon_loc=np.where(data_array[p]==max)[1]

        #evaluating the size of the location arrays
        if lat_loc.shape[0]>1:
            lat_loc_mean=np.mean(lat_loc)
        else:
            lat_loc_mean=lat_loc[0]

        if lon_loc.shape[0]>1:
            lon_loc_mean=np.mean(lon_loc)
        else:
            lon_loc_mean=lon_loc[0]

        subtropical_lat[p]=Lat[int(np.round(lat_loc_mean,0))]
        subtropical_lon[p]=Lon[int(np.round(lon_loc_mean,0))]

    return subtropical_strength, subtropical_lat, subtropical_lon

def Bolivian_High(path_entry,var_sp,model_name, type_dataset,lon_limits,lat_limits, pressure_0,pressure_1):

    var_data=xr.open_mfdataset(path_entry+model_name+'_'+var_sp+'_original_mon_clim_LT.nc')

    dataset=var_data[var_sp]

    #delimiting the temporal range and spatial domain
    lat_bnd,lon_bnd,lat_slice,lon_slice=lat_lon_bds(lon_limits,\
    lat_limits, dataset)

    var_delimited=time_lat_lon_positions(None,None,lat_bnd,\
    lon_bnd,lat_slice,dataset,'ERA5','Yes')

    #grouping by month
    var_monthly=var_delimited.groupby('time.month').mean('time')
    ################################################################################
    #Selecting the pressure level
    ini_level,fin_level=levels_limit(var_monthly,pressure_0,pressure_1)

    var_levels=var_monthly[:,int(ini_level):int(fin_level)+1,:,:]

    if lat_slice=='lat':
        var_lon_index=var_levels.mean(dim='lon')
        time_serie_var=var_lon_index.mean(dim='lat')

    else:
        var_lon_index=var_levels.mean(dim='longitude')
        time_serie_var=var_lon_index.mean(dim='latitude')


    ################################################################################
    var_array=np.array(time_serie_var)

    if dataset.units=='m':
        var_array=var_array/1000

    else:
        pass

    return var_array

def plot_series(axs,index_strength_ref,index_strength_m,title_subplot_0,list_models,units_index,legend_status,title_font2, label_font, ticks_font,bt_st):
    labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    reference_data=index_strength_ref
    models_data=index_strength_m
    y_label_plot=units_index
    title_subplot=title_subplot_0
    if legend_status=='Yes':
        labels_ref_plot='Reference [ERA5]'
    else:
        labels_ref_plot=None

    axs.set_title(title_subplot,fontsize=title_font2,loc='left')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    axs.plot(reference_data, color = 'k', linewidth=2.2,label=labels_ref_plot)
    #iterating in the models to obtain the serie
    colors=iter(cm.rainbow(np.linspace(0,1,len(list_models))))

    for m in range(len(list_models)):
        if legend_status=='Yes':
            labels_plots=list_models[m]
        else:
            labels_plots=None
        c=next(colors)

        axs.plot(models_data[m] ,color =c ,linewidth=1.5,label=labels_plots)
    axs.set_xticks(np.arange(0,12,1))
    axs.set_xticklabels(labels,fontsize=ticks_font)
    axs.set_ylabel(y_label_plot,fontsize=label_font)

    if bt_st=='yes':
        axs.set_xlabel('Month',fontsize=label_font)

def plot_series_int_loc(title_plot,index_strength_ref,index_strength_m,index_latitude_ref,index_longitude_ref,index_latitude_m,index_longitude_m,title_subplot_0,title_subplot_1,title_subplot_2,list_models,save_str,units_index,degree_str, path_save_plots,nrow,ncol,fzx,fzy,title_font1,title_font2, label_font, ticks_font, legend_font):
    labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    row=nrow
    column=ncol
    fig=plt.figure(figsize=(fzx,fzy))
    fig.suptitle(title_plot,fontsize=title_font1,fontweight='bold')

    for k in range(nrow):
        axs = fig.add_subplot(row, column, k+1)
        if k==0:
            reference_data=index_strength_ref
            models_data=index_strength_m
            y_label_plot=units_index
            title_subplot=title_subplot_0
            labels_ref_plot='Reference [ERA5]'


        elif k==1:
            reference_data=index_latitude_ref
            models_data=index_latitude_m
            y_label_plot='Latitude '+degree_str
            title_subplot=title_subplot_1
            labels_ref_plot=None


        else:
            reference_data=index_longitude_ref
            models_data=index_longitude_m
            y_label_plot='Longitude [°W]'
            title_subplot=title_subplot_2
            labels_ref_plot=None


        axs.set_title(title_subplot,fontsize=title_font2,loc='left')
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)

        axs.plot(reference_data, color = 'k', linewidth=2.2,label=labels_ref_plot)
        #iterating in the models to obtain the serie
        colors=iter(cm.rainbow(np.linspace(0,1,len(list_models))))

        for m in range(len(list_models)):
            if k==0:
                labels_plots=list_models[m]
            else:
                labels_plots=None
            c=next(colors)

            axs.plot(models_data[m] ,color =c ,linewidth=1.5,label=labels_plots)
        axs.set_xticks(np.arange(0,12,1))
        axs.set_xticklabels(labels,fontsize=ticks_font)
        axs.set_ylabel(y_label_plot,fontsize=label_font)
        if k!=0 and k!=1:
            axs.set_xlabel('Month',fontsize=label_font)

    fig.legend( bbox_to_anchor=(0.92, 0.9), loc='upper left', fontsize=str(legend_font))

    fig.savefig(path_save_plots+save_str+'.png', \
    format = 'png', bbox_inches='tight')
    plt.close()

def wind_indices(title_plot,index_strength_ref,index_strength_m,index_latitude_ref,index_latitude_m,title_subplot_0,title_subplot_1,y_ticks_0,y_ticks_1,list_models,save_str, path_save_plots,nrow,ncol,fzx,fzy,title_font1,title_font2, label_font, ticks_font, legend_font):
    labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug',\
    'Sep','Oct','Nov','Dec']
    row=nrow
    column=ncol
    fig=plt.figure(figsize=(fzx, fzy))
    fig.suptitle(title_plot,fontsize=title_font1,\
    fontweight='bold')

    for k in range(2):
        axs = fig.add_subplot(row, column, k+1)
        if k==0:
            reference_data=index_strength_ref
            models_data=index_strength_m
            y_label_plot='Wind [m/s]'
            title_subplot=title_subplot_0
            labels_ref_plot='Reference [ERA5]'
            y_ticks_plot=y_ticks_0

        else:
            reference_data=index_latitude_ref
            models_data=index_latitude_m
            y_label_plot='Latitude [°S]'
            title_subplot=title_subplot_1
            labels_ref_plot=None
            y_ticks_plot=y_ticks_1

        axs.set_title(title_subplot,fontsize=title_font2,loc='left')
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)

        axs.plot(reference_data, color = 'k', linewidth=2.2,label=labels_ref_plot)
        #iterating in the models to obtain the serie
        colors=iter(cm.rainbow(np.linspace(0,1,len(list_models))))

        for m in range(len(list_models)):
            if k==0:
                labels_plots=list_models[m]
            else:
                labels_plots=None
            c=next(colors)

            axs.plot(models_data[m] ,color =c ,linewidth=1.5,label=labels_plots)
        axs.set_xticks(np.arange(0,12,1))
        axs.set_xticklabels(labels,fontsize=ticks_font)
        axs.set_yticks(y_ticks_plot)
        axs.set_yticklabels(y_ticks_plot,fontsize=ticks_font)
        axs.set_ylabel(y_label_plot,fontsize=label_font)

        if k==1:
            axs.set_xlabel('Month',fontsize=label_font)

    fig.legend( bbox_to_anchor=(0.92, 0.9), loc='upper left', fontsize=str(legend_font))

    fig.savefig(path_save_plots+save_str+'.png', \
    format = 'png', bbox_inches='tight')
    plt.close()

def plot_one_plot(models_n,index_name,path_save_plots,Index_model,wind_ref_arr_index,pressure_levels_index,ylabel_str,ylim_low,ylim_up,title_str,title_font1, label_font, ticks_font, legend_font):
    color=iter(cm.rainbow(np.linspace(0,1,len(models_n))))
    labels_x=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    fig,ax = plt.subplots(figsize=(10.5,7))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylabel(ylabel_str,fontsize=label_font)
    plt.xlabel('Month',fontsize=label_font)
    for i in range(len(models_n)):
        c=next(color)
        plt.plot(Index_model[i,:],color =c ,linewidth=1.5,label=models_n[i])
    plt.plot(wind_ref_arr_index, color = 'k', linewidth=2.2,label='Reference [ERA5]')
    plt.title(title_str,\
    fontsize=title_font1,loc='left')
    plt.xticks(np.arange(0,12,1),labels_x,fontsize=ticks_font)
    plt.yticks(fontsize=ticks_font)
    plt.ylim(ylim_low,ylim_up)
    #plt.legend(fontsize=12)
    plt.legend( bbox_to_anchor=(1.05, 1.1), loc='upper left', fontsize=str(legend_font))
    fig.savefig(path_save_plots+index_name+'.png', format = 'png',\
    bbox_inches='tight')
    plt.close()

def interpolation_fields(var_array_ref,lat_refe,lon_refe,lat_mo,lon_mo,dx_mo,dy_mo):
    xnew_ref=np.arange(lon_refe[0],lon_refe[-1:][0], dx_mo)
   
    ynew_ref=np.arange(lat_refe[0],lat_refe[-1:][0], dy_mo)
    

    #Comparing the lists
    lon_interpolation,lat_interpolation=lat_lon_lengthComparison(lat_mo,\
    lon_mo,ynew_ref,xnew_ref,lat_refe,lon_refe,dx_mo,dy_mo)

    #Performing the interpolation of the reference data to the grid size of the
    #model
    var_ref_interpolated=data_interpolation(var_array_ref,'3D',lon_refe,\
    lat_refe,lon_interpolation,lat_interpolation)

    return var_ref_interpolated

def TropD_Calculate_StreamFunction_hadley(V, lat, lev):
    ''' Calculate streamfunction by integratingdivergent meridional wind from top
    of the atmosphere to surface
    Args:
    V: array of divergen meridional wind with dimensions (lat, lev)

    lat: equally spaced latitude array
    lev: vertical level array in hPa
    Returns:

    ndarray: the streamfunction psi(lat,lev)
    '''


    EarthRadius = 6371220.0
    EarthGrav = 9.80616
    B = np.ones(np.shape(V))
    # B = 0 for subsurface data
    B[np.isnan(V)]=0
    psi = np.zeros(np.shape(V))

    COS = np.repeat(np.cos(lat*np.pi/180), len(lev), axis=0).reshape(len(lat),len(lev))

    psi = (EarthRadius/EarthGrav) * 2 * np.pi \
    * integrate.cumtrapz(B * V * COS, lev*100, axis=1, initial=0)

    return psi

def TropD_Calculate_StreamFunction_walker(U, lon, lev):
    ''' Calculate streamfunction by integrating divergent zonal wind from top
    of the atmosphere to surface
    Args:
    U: array of divergent zonal wind with dimensions (lon, lev)

    lon: equally spaced latitude array
    lev: vertical level array in hPa
    Returns:

    ndarray: the streamfunction psi(lon,lev)
    '''


    EarthRadius = 6371220.0
    EarthGrav = 9.80616
    B = np.ones(np.shape(U))
    # B = 0 for subsurface data
    B[np.isnan(U)]=0
    psi = np.zeros(np.shape(U))

    psi = (EarthRadius/EarthGrav) * 2 * np.pi \
    * integrate.cumtrapz(B * U , lev*100, axis=1, initial=0)

    return psi

def hadley_cell_calc(div_compo,lon_arr,lat_arr,level_arr,lat_bnds_had,lon_bnds_had):
    lat_idx_h = np.where((lat_arr<=lat_bnds_had[0])&(lat_arr>=lat_bnds_had[1]))[0]
    lon_idx_h = np.where((lon_arr>=lon_bnds_had[0])&(lon_arr<=lon_bnds_had[1]))[0]

    lon_hadley=lon_arr[lon_idx_h]
    lat_hadley=lat_arr[lat_idx_h]

    print('##############################')
    print("hadley_cell_calc: lat_lon idx h OK")
    print('##############################')

    hadley_stream=np.empty((4,lat_hadley.shape[0],level_arr.shape[0]))
    std_h=np.empty((4))

    #Iterating in the seasons
    for m in range(4):

        div_delimited=div_compo[m,lat_idx_h[0]:lat_idx_h[-1]+1,lon_idx_h[0]:lon_idx_h[-1]+1,:]

        div_lat=np.mean(div_delimited,axis=1)
        div_input=div_lat

        had_cell_season=TropD_Calculate_StreamFunction_hadley(div_input, lat_hadley, level_arr)

        print('##############################')
        print("hadley_cell_calc: TropD_Calculate_StreamFunction_hadley OK")
        print('##############################')

        hadley_stream[m,:,:]=had_cell_season

        #Estimating the reference standar deviation
        std_season=np.nanstd(hadley_stream[m])
        std_h[m]=std_season

    return hadley_stream,std_h,lat_hadley

def walker_cell_calc(div_compo,lon_arr,lat_arr,level_arr,lat_bnds_wal,lon_bnds_wal):
    lat_idx_w = np.where((lat_arr<=lat_bnds_wal[0])&(lat_arr>=lat_bnds_wal[1]))[0]
    lon_idx_w = np.where((lon_arr>=lon_bnds_wal[0])&(lon_arr<=lon_bnds_wal[1]))[0]

    lon_walker=lon_arr[lon_idx_w]
    lat_walker=lat_arr[lat_idx_w]

    walker_stream=np.empty((4,lon_walker.shape[0],level_arr.shape[0]))
    std_w=np.empty((4))

    #Iterating in the seasons
    for m in range(4):

        div_delimited=div_compo[m,lat_idx_w[0]:lat_idx_w[-1]+1,lon_idx_w[0]:lon_idx_w[-1]+1,:]

        div_lon=np.mean(div_delimited,axis=0)
        div_input=div_lon

        wal_cell_season=TropD_Calculate_StreamFunction_walker(div_input, lon_walker, level_arr)

        print('##############################')
        print("hadley_cell_calc: TropD_Calculate_StreamFunction_walker OK")
        print('##############################')

        walker_stream[m,:,:]=wal_cell_season

        #Estimating the reference standar deviation
        std_season=np.nanstd(walker_stream[m])
        std_w[m]=std_season

    return walker_stream,std_w,lon_walker

def regional_cells(path_entry,var_sp1,var_sp2,model_name,lat_h,lon_h,lat_w,lon_w,time_0,time_1,level_lower,level_upper,type_data):

    var_data1=xr.open_mfdataset(path_entry+model_name+'_'+var_sp1+'_original_seasonal_mean.nc')

    u_field=var_data1[var_sp1]

    var_data2=xr.open_mfdataset(path_entry+model_name+'_'+var_sp2+'_original_seasonal_mean.nc')

    v_field=var_data2[var_sp2]

    print('##############################')
    print("regional_cells: read files OK")
    print('##############################')

    #obtaining the domain bnds
    lat_slice,lon_slice=lat_lon_bds(lon_h,lat_h,u_field)[2:4]

    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #Verifying the order of the pressure levels and latitudes 
    if type_data=='reference':
        if np.array(v_field.level)[0]>np.array(v_field.level)[-1]:
            v_field=v_field.reindex(level=list(reversed(v_field.level)))
            u_field=u_field.reindex(level=list(reversed(u_field.level)))
        else:
            pass
    else:
        if np.array(v_field.plev)[0]>np.array(v_field.plev)[-1]:
            v_field=v_field.reindex(plev=list(reversed(v_field.plev)))
            u_field=u_field.reindex(plev=list(reversed(u_field.plev)))
        else:
            pass

    #Latitude: From north to south
    if lat_slice=='lat':
        if np.array(v_field.lat)[-1]>np.array(v_field.lat)[0]:
            v_field=v_field.reindex(lat=list(reversed(v_field.lat)))
            u_field=u_field.reindex(lat=list(reversed(u_field.lat)))
        else:
            pass
    else:
        if np.array(v_field.latitude)[-1]>np.array(v_field.latitude)[0]:
            v_field=v_field.reindex(latitude=list(reversed(v_field.latitude)))
            u_field=u_field.reindex(latitude=list(reversed(u_field.latitude)))
        else:
            pass
    
    print('##############################')
    print("regional_cells: reindex OK")
    print('##############################')

    lat_bnd_h,lon_bnd_h=lat_lon_bds(lon_h,lat_h,u_field)[0:2]
    lat_bnd_w,lon_bnd_w=lat_lon_bds(lon_w,lat_w,u_field)[0:2]

    #selecting the pressure level
    ini_level,fin_level=levels_limit(u_field,level_lower,level_upper)
    u_levels=u_field[:,int(ini_level):int(fin_level+1),:,:]
    v_levels=v_field[:,int(ini_level):int(fin_level+1),:,:]

    print('##############################')
    print("regional_cells: levels OK")
    print('##############################')

    #obtaining the list of the pressure levels and the latitudes
    if lat_slice=='latitude':
        lat_list=np.array(u_levels.latitude)
        lon_list=np.array(u_levels.longitude)
    else:
        lat_list=np.array(u_levels.lat)
        lon_list=np.array(u_levels.lon)
    
    if type_data=='reference':

        p_level=np.array(u_levels.level)

        if u_levels.level.units=='Pa':
            p_level=p_level/100
        else:
            pass
    else:
        p_level=np.array(u_levels.plev)

        if u_levels.plev.units=='Pa':
            p_level=p_level/100
        else:
            pass

    #converting into array
    u_array=np.array(u_levels)
    v_array=np.array(v_levels)

    #-----------------------------------------------------------------------------------------------
    ############################################################################
    #Iterating in the season, handling the NaNs and transposint to obtain
    #a matrix of (lat,lon,level)

    uchi=np.empty((4,len(lat_list),len(lon_list),len(p_level)))
    vchi=np.empty((4,len(lat_list),len(lon_list),len(p_level)))

    for s in range(4):
        u_month=u_array[s]
        v_month=v_array[s]

        if np.isnan(np.sum(u_month))==True:
            #interpolating to treat the nans
            u_noNans=NaNs_interp(u_month, '3D', 'nearest')
            v_noNans=NaNs_interp(v_month, '3D', 'nearest')
        else:
            u_noNans=u_month
            v_noNans=v_month

        u_input=u_noNans.transpose(1,2,0)
        v_input=v_noNans.transpose(1,2,0)

        vw_season = VectorWind(u_input,v_input)
        uchi_season, vchi_season = vw_season.irrotationalcomponent()

        uchi[s,:,:,:]=uchi_season
        vchi[s,:,:,:]=vchi_season

    print('##############################')
    print("regional_cells: div component OK")
    print('##############################')
    
    #-----------------------------------------------------------------------------------------
    ############################################################################
    #Calculating the hadley and walker circulations
    #HADLEY CIRCULATION

    hadley_stream,std_h,lat_had=hadley_cell_calc(vchi,lon_list,lat_list,\
    p_level,lat_bnd_h,lon_bnd_h)

    #WALKER CIRCULATION
    walker_stream,std_w,lon_wal=walker_cell_calc(uchi,lon_list,lat_list,\
    p_level,lat_bnd_w,lon_bnd_w)

    #obtainig the grid size of the model 
    dx_h=np.round(abs(lat_had[0])-abs(lat_had[1]),2)
    dx_w=np.round(abs(lon_wal[1])-abs(lon_wal[0]),2)

    print('##############################')
    print("regional_cells: hadley_stream-walker_stream OK")
    print('##############################')


    return hadley_stream, std_h, lat_had,  walker_stream,std_w,lon_wal, p_level, dx_h, dx_w

def interpolation_cells(matrix_h, matrix_w, dx_h_m_ini, dx_w_m_ini, p_level_m_ini, lat_list_had_m,lon_list_wal_m,p_level_ref,lat_had_ini, lon_wal_ini, shape_h, shape_w):

    #Hadley
    dx_h=dx_h_m_ini
    dy_h=None
    #Here we create the new lists of pressure and longitude of the reference data
    xnew_h=lat_list_had_m
    ynew_h=p_level_m_ini

    #comparing lenghts of the list from reference and model lat and lon
    lat_interpolation_h,press_interpolation_h=lat_lon_lengthComparison(p_level_m_ini,lat_list_had_m,ynew_h\
    ,xnew_h,p_level_ref,lat_had_ini,dx_h,dy_h)

    matrix_interpolated_h=np.empty(shape_h.shape)
    for r in range(4):

        hadley_interpolated=data_interpolation(matrix_h[r].T,'2D',lat_had_ini,p_level_ref,\
        lat_interpolation_h,press_interpolation_h)

        matrix_interpolated_h[r,:,:]=np.flipud(hadley_interpolated[0].T)

    #Walker
    dx_w=dx_w_m_ini
    dy_w=None
    #Here whe create the new lists of pressure and longitude of the reference data
    xnew_w=np.arange(lon_wal_ini[0],lon_wal_ini[-1], dx_w_m_ini)
    ynew_w=p_level_m_ini

    #comparing lenghts of the list from reference and model lat and lon
    lon_interpolation_w,press_interpolation_w=lat_lon_lengthComparison(p_level_m_ini,lon_list_wal_m,ynew_w\
    ,xnew_w,p_level_ref,lon_wal_ini,dx_w,dy_w)

    matrix_interpolated_w=np.empty(shape_w.shape)
    for r in range(4):

        walker_interpolated=data_interpolation(matrix_w[r].T,'2D',lon_wal_ini,p_level_ref,\
        lon_interpolation_w,press_interpolation_w)

        matrix_interpolated_w[r,:,:]=walker_interpolated[0].T
    
    return matrix_interpolated_h, matrix_interpolated_w

def domain_comparison(lat_slice_var,var_1,var_2, lon_list_var2,time_zero,time_end,lat_list_var2,var_2_field,type_str, coordinate):
    if coordinate=='longitude':
    
        if lat_slice_var=='lat':
            if len(var_2.lon)==len(var_1.lon):
                lon_boundaries_var2=None
            else:
                if len(var_2.lon)>len(var_1.lon):
                    diff=np.round(len(var_2.lon)-len(var_1.lon),0)
                    if lon_list_var2[0]<0 and lon_list_var2[1]<0:
                        lon_boundaries_var2=[lon_list_var2[0],lon_list_var2[1]+diff]
                    else:
                        lon_boundaries_var2=[lon_list_var2[0],lon_list_var2[1]-diff]
                else:
                    diff=np.round(len(var_1.lon)-len(var_2.lon),0)
                    if lon_list_var2[0]<0 and lon_list_var2[1]<0:
                        lon_boundaries_var2=[lon_list_var2[0],lon_list_var2[1]-diff]
                    else:
                        lon_boundaries_var2=[lon_list_var2[0],lon_list_var2[1]+diff]

                var_2=time_lat_lon_positions(time_zero,time_end,lat_list_var2,\
                lon_boundaries_var2,lat_slice_var,var_2_field,type_str,'Yes')
        else:
            if len(var_2.longitude)==len(var_1.longitude):
                lon_boundaries_var2=None
            else:
                if len(var_2.longitude)>len(var_1.longitude):
                    diff=np.round(len(var_2.longitude)-len(var_1.longitude),0)
                    if lon_list_var2[0]<0 and lon_list_var2[1]<0:
                        lon_boundaries_var2=[lon_list_var2[0],lon_list_var2[1]+diff]
                    else:
                        lon_boundaries_var2=[lon_list_var2[0],lon_list_var2[1]-diff]
                else:
                    diff=np.round(len(var_1.longitude)-len(var_2.longitude),0)
                    if lon_list_var2[0]<0 and lon_list_var2[1]<0:
                        lon_boundaries_var2=[lon_list_var2[0],lon_list_var2[1]-diff]
                    else:
                        lon_boundaries_var2=[lon_list_var2[0],lon_list_var2[1]+diff]

                var_2=time_lat_lon_positions(time_zero,time_end,lat_list_var2,\
                lon_boundaries_var2,lat_slice_var,var_2_field,type_str,'Yes')
    
    else:
        if lat_slice_var=='lat':
            if len(var_2.lat)==len(var_1.lat):
                lat_boundaries_var2=None
            else:
                if len(var_2.lat)>len(var_1.lat):
                    diff=np.round(len(var_2.lat)-len(var_1.lat),0)
                    lat_boundaries_var2=[lat_list_var2[0],lat_list_var2[1]-diff]
                else:
                    diff=np.round(len(var_1.lat)-len(var_2.lat),0)
                    lat_boundaries_var2=[lat_list_var2[0],lat_list_var2[1]+diff]

                var_2=time_lat_lon_positions(time_zero,time_end,lat_boundaries_var2,\
                lon_list_var2,lat_slice_var,var_2_field,type_str,'Yes')
        else:
            if len(var_2.latitude)==len(var_1.latitude):
                lat_boundaries_var2=None
            else:
                if len(var_2.latitude)>len(var_1.latitude):
                    diff=np.round(len(var_2.latitude)-len(var_1.latitude),0)          
                    lat_boundaries_var2=[lat_list_var2[0],lat_list_var2[1]-diff]
                    
                else:
                    diff=np.round(len(var_1.latitude)-len(var_2.latitude),0)  
                    lat_boundaries_var2=[lat_list_var2[0],lat_list_var2[1]+diff]
                
                var_2=time_lat_lon_positions(time_zero,time_end,lat_boundaries_var2,\
                lon_list_var2,lat_slice_var,var_2_field,type_str,'yes')
        
    if coordinate=='longitude':
        band_rt=lon_boundaries_var2
    else:
        band_rt=lat_boundaries_var2
    
    return var_2, band_rt

def NaNs_levels(var_array,dims, interp_type):

    No_NaNs_matrix=np.empty((var_array.shape))

    #Iterating in the pressure levels
    for i in range(var_array.shape[1]):

        if np.isnan(np.sum(var_array[:,i,:,:]))==True:
            var_noNaN=NaNs_interp(var_array[:,i,:,:], dims, interp_type)

            print('#####################################')
            print('NaNs_levels: var noNaN OK')
            print('#################################')

        else:
            var_noNaN=var_array[:,i,:,:]

        #Saving the new matrices
        No_NaNs_matrix[:,i,:,:]=var_noNaN

    return No_NaNs_matrix

def VIMF_calc(path_entry, var_sp1, var_sp2, var_sp3, model_name,lat_d,lon_d,time_0,time_1,level_lower,level_upper,type_data):

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
    qu_integral=np.nansum(qu_column,axis=1)/9.8
    qv_integral=np.nansum(qv_column,axis=1)/9.8

    print('#########################')
    print('VIMF_calc: integrals OK')
    print('#########################')

    #defining the grid size
    dx_data=np.round(abs(Lon_list[1])-abs(Lon_list[0]),2)
    dy_data=np.round(abs(Lat_list[-1:][0])-  abs(Lat_list[-2:][0]),2)

    VIMF_estimation=np.sqrt(qu_integral**2+qv_integral**2)

    return VIMF_estimation,Lat_list,Lon_list,dx_data, dy_data

def MSE_calc(path_entry, var_sp1, var_sp2, var_sp3, var_sp4, model_name,lat_d,lon_d,time_0,time_1,level_lower,level_upper,type_data):
    
    var_data1=xr.open_mfdataset(path_entry+model_name+'_'+var_sp1+'_original_seasonal_mean.nc')

    v_field=var_data1[var_sp1]

    var_data2=xr.open_mfdataset(path_entry+model_name+'_'+var_sp2+'_original_seasonal_mean.nc')

    q_field=var_data2[var_sp2]

    var_data3=xr.open_mfdataset(path_entry+model_name+'_'+var_sp3+'_original_seasonal_mean.nc')

    t_field=var_data3[var_sp3]

    var_data4=xr.open_mfdataset(path_entry+model_name+'_'+var_sp4+'_original_seasonal_mean.nc')

    z_field=var_data4[var_sp4]

    print('#########################')
    print('MSE_calc: read files OK')
    print('#########################')

    #obtaining the domain bnds
    lat_bnd,lon_bnd,lat_slice,lon_slice=lat_lon_bds(lon_d,lat_d,v_field)
    lat_bnd_q,lon_bnd_q=lat_lon_bds(lon_d,lat_d,q_field)[0:2]
    lat_bnd_t,lon_bnd_t=lat_lon_bds(lon_d,lat_d,t_field)[0:2]
    lat_bnd_z,lon_bnd_z=lat_lon_bds(lon_d,lat_d,z_field)[0:2]
    #delimiting the spatial domain

    v_delimited=time_lat_lon_positions(time_0,time_1,lat_bnd,\
    lon_bnd,lat_slice,v_field,'ERA5','Yes')

    q_delimited=time_lat_lon_positions(time_0,time_1,lat_bnd_q,\
    lon_bnd_q,lat_slice,q_field,'ERA5','Yes')

    t_delimited=time_lat_lon_positions(time_0,time_1,lat_bnd_q,\
    lon_bnd_q,lat_slice,t_field,'ERA5','Yes')

    z_delimited=time_lat_lon_positions(time_0,time_1,lat_bnd_q,\
    lon_bnd_q,lat_slice,z_field,'ERA5','Yes')

    print('#########################')
    print('MSE_calc: var_delimited OK')
    print('#########################')

    #Comparing the delimited fields of all variables 
    q_delimited_1, lon_q_new=domain_comparison(lat_slice,v_delimited,q_delimited, lon_bnd_q,time_0,time_1,lat_bnd_q,q_field,'ERA5','longitude')
    t_delimited_1, lon_t_new=domain_comparison(lat_slice,v_delimited,t_delimited, lon_bnd_t,time_0,time_1,lat_bnd_t,t_field,'ERA5','longitude')
    z_delimited_1, lon_z_new=domain_comparison(lat_slice,v_delimited,z_delimited, lon_bnd_z,time_0,time_1,lat_bnd_z,z_field,'ERA5','longitude')

    q_delimited_f, lat_q_new=domain_comparison(lat_slice,v_delimited,q_delimited_1, lon_q_new,time_0,time_1,lat_bnd_q,q_field,'ERA5','latitude')
    t_delimited_f, lat_t_new=domain_comparison(lat_slice,v_delimited,t_delimited_1, lon_t_new,time_0,time_1,lat_bnd_t,t_field,'ERA5','latitude')
    z_delimited_f, lat_z_new=domain_comparison(lat_slice,v_delimited,z_delimited_1, lon_z_new,time_0,time_1,lat_bnd_z,z_field,'ERA5','latitude')

    print('#########################')
    print('MSE_calc: rdomain_comparison OK')
    print('#########################')

    #selecting the pressure level
    ini_level,fin_level=levels_limit(v_delimited,level_lower,level_upper)
    v_levels=v_delimited[:,int(ini_level):int(fin_level+1),:,:]
    q_levels=q_delimited_f[:,int(ini_level):int(fin_level+1),:,:]
    t_levels=t_delimited_f[:,int(ini_level):int(fin_level+1),:,:]
    z_levels=z_delimited_f[:,int(ini_level):int(fin_level+1),:,:]

    #converting into array
    v_array_t=np.array(v_levels)
    q_array_t=np.array(q_levels)
    t_array_t=np.array(t_levels)

    if type_data=='ERA5':
        z_array_t=np.array(z_levels)
    else:
        z_array_t=np.array(z_levels)*9.8

    #Evaluating the missing values in the pressure levels
    v_array=NaNs_levels(v_array_t,'3D', 'cubic')
    q_array=NaNs_levels(q_array_t,'3D', 'cubic')
    t_array=NaNs_levels(t_array_t,'3D', 'cubic')
    z_array=NaNs_levels(z_array_t,'3D', 'cubic')

    print('#########################')
    print('MSE_calc: NaNs_levels OK')
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
        if v_delimited.level.units=='Pa':
            p_levels=list(np.array(v_levels.level))
        else:
            p_levels=list(np.array(v_levels.level)*100)
    else:
        if v_delimited.plev.units=='Pa':
            p_levels=list(np.round(np.array(v_levels.plev),1))
        else:
            p_levels=list(np.round(np.array(v_levels.plev),1)*100)

    ############################################################################
    #Performing the VIMF calculation

    #geneating the integral
    MSE_column=np.empty((4,len(p_levels)-1,len(Lat_list),len(Lon_list)))

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
                    v_level=(v_array[p,i,j,k]+v_array[p,i+1,j,k])/2
                    q_level=(q_array[p,i,j,k]+q_array[p,i+1,j,k])/2
                    t_level=(t_array[p,i,j,k]+t_array[p,i+1,j,k])/2
                    z_level=(t_array[p,i,j,k]+z_array[p,i+1,j,k])/2

                    m=lv*q_level + cp*t_level + z_level

                    dp=(p_levels[i]-p_levels[i+1])

                    mse_level_grid=m *v_level*dp

                    MSE_column[p,i,j,k]=mse_level_grid

    # adding up in the pressure levels
    MSE_integral=np.nansum(MSE_column,axis=1)

    print('#########################')
    print('MSE_calc: MSE_integral OK')
    print('#########################')

    #defining the grid size
    dx_data=np.round(abs(Lon_list[1])-abs(Lon_list[0]),2)
    dy_data=np.round(abs(Lat_list[-1:][0])-  abs(Lat_list[-2:][0]),2)
    

    return MSE_integral,Lat_list,Lon_list,dx_data, dy_data

def boundaries_fluxes(path_entry, var_sp1, var_sp2, model_name,lat_d,lon_d,time_0,time_1,level_lower,level_upper,type_data):

    var_data1=xr.open_mfdataset(path_entry+model_name+'_'+var_sp1+'_original_seasonal_mean.nc')

    wind_field=var_data1[var_sp1]

    var_data2=xr.open_mfdataset(path_entry+model_name+'_'+var_sp2+'_original_seasonal_mean.nc')

    q_field=var_data2[var_sp2]

    print('#########################')
    print('boundaries_fluxes: read files OK')
    print('#########################')

    lat_bnd,lon_bnd,lat_slice,lon_slice=lat_lon_bds(lon_d,lat_d,wind_field)
    lat_bnd_q,lon_bnd_q=lat_lon_bds(lon_d,lat_d,q_field)[0:2]
    #delimiting the spatial domain
    wind_delimited=time_lat_lon_positions(time_0,time_1,lat_bnd,\
    lon_bnd,lat_slice,wind_field,'ERA5','Yes')
    q_delimited=time_lat_lon_positions(time_0,time_1,lat_bnd_q,\
    lon_bnd_q,lat_slice,q_field,'ERA5','Yes')

    print('#########################')
    print('boundaries_fluxes: var_delimited OK')
    print('#########################')

    #Applying the function to compare the length of the dimensions

    if lat_slice=='lat':
        if (len(q_delimited.lon)==len(wind_delimited.lon)) & (len(q_delimited.lat)==len(wind_delimited.lat)):
            q_delimited_f=q_delimited
        else:
            if (len(q_delimited.lon)!=len(wind_delimited.lon)) & (len(q_delimited.lat)!=len(wind_delimited.lat)):

                q_delimited_1, lon_q_new=domain_comparison(lat_slice,wind_delimited,q_delimited, lon_bnd_q,time_0,time_1,lat_bnd_q,q_field,'ERA5','longitude')
                q_delimited_f, lat_q_new=domain_comparison(lat_slice,wind_delimited,q_delimited_1, lon_q_new,time_0,time_1,lat_bnd_q,q_field,'ERA5','latitude')

            elif (len(q_delimited.lon)!=len(wind_delimited.lon)) & (len(q_delimited.lat)==len(wind_delimited.lat)):
                q_delimited_f, lon_q_new=domain_comparison(lat_slice,wind_delimited,q_delimited, lon_bnd_q,time_0,time_1,lat_bnd_q,q_field,'ERA5','longitude')
            else:
                q_delimited_f ,lat_q_new=domain_comparison(lat_slice,wind_delimited,q_delimited, lon_bnd_q,time_0,time_1,lat_bnd_q,q_field,'ERA5','latitude')
    else:
        if (len(q_delimited.longitude)==len(wind_delimited.longitude)) & (len(q_delimited.latitude)==len(wind_delimited.latitude)):
            q_delimited_f=q_delimited
        else:
            if (len(q_delimited.longitude)!=len(wind_delimited.longitude)) & (len(q_delimited.latitude)!=len(wind_delimited.latitude)):

                q_delimited_1, lon_q_new=domain_comparison(lat_slice,wind_delimited,q_delimited, lon_bnd_q,time_0,time_1,lat_bnd_q,q_field,'ERA5','longitude')
                q_delimited_f, lat_q_new=domain_comparison(lat_slice,wind_delimited,q_delimited_1, lon_q_new,time_0,time_1,lat_bnd_q,q_field,'ERA5','latitude')

            elif (len(q_delimited.longitude)!=len(wind_delimited.longitude)) & (len(q_delimited.latitude)==len(wind_delimited.latitude)):
                q_delimited_f, lon_q_new=domain_comparison(lat_slice,wind_delimited,q_delimited, lon_bnd_q,time_0,time_1,lat_bnd_q,q_field,'ERA5','longitude')
            else:
                q_delimited_f, lat_q_new=domain_comparison(lat_slice,wind_delimited,q_delimited, lon_bnd_q,time_0,time_1,lat_bnd_q,q_field,'ERA5','latitude')
    
    print('#########################')
    print('boundaries_fluxes: domain comparison OK')
    print('#########################')

    #selecting the pressure level
    ini_level,fin_level=levels_limit(wind_delimited,level_lower,level_upper)
    wind_levels=wind_delimited[:,int(ini_level):int(fin_level+1),:,:]
    q_levels=q_delimited_f[:,int(ini_level):int(fin_level+1),:,:]

    #converting into array
    wind_array_t=np.array(wind_levels)
    q_array_t=np.array(q_levels)

    #Evaluating the missing values in the pressure levels
    wind_array=NaNs_levels(wind_array_t,'3D', 'cubic')
    q_array=NaNs_levels(q_array_t,'3D', 'cubic')

    print('#########################')
    print('boundaries_fluxes: NaNs levels OK')
    print('#########################')

    #defining Lat and Lon lists
    if lat_slice=='lat':
        Lat_list=list(np.array(wind_delimited.lat))
        Lon_list=list(np.array(wind_delimited.lon))
    else:
        Lat_list=list(np.array(wind_delimited.latitude))
        Lon_list=list(np.array(wind_delimited.longitude))

    #For the list of pressure levels (They have to be in Pa)
    if type_data=='reference':
        if wind_delimited.level.units=='Pa':
            p_levels=list(np.array(wind_levels.level))
        else:
            p_levels=list(np.array(wind_levels.level)*100)
    else:
        if wind_delimited.plev.units=='Pa':
            p_levels=list(np.round(np.array(wind_levels.plev),1))
        else:
            p_levels=list(np.round(np.array(wind_levels.plev),1)*100)

    ############################################################################
    #Obtaining the product qv
    q_wind_column=wind_array*q_array

    #defining the grid size
    dx_data=np.round(abs(Lon_list[1])-abs(Lon_list[0]),2)
    dy_data=np.round(abs(Lat_list[-1:][0])-  abs(Lat_list[-2:][0]),2)

    return q_wind_column,Lat_list,Lon_list,dx_data, dy_data,p_levels

def NaNs_land(var_array):
    if np.isnan(var_array).any()==True:

        inds = np.where(np.isnan(var_array))

        var_array_r=np.empty(var_array.shape)

        for r in range(4):

            for m in range(var_array.shape[2]):
                column=var_array[r,:,m]

                if np.isnan(column).any()==True:

                    column_1=pd.Series(column).fillna(method="ffill").values

                    var_array_r[r,:,m]=column_1
                else:

                    var_array_r[r,:,m]=column

        text_r='Matrix with NaNs'

        print('################################')
        print('NaNs_lanc: ffil OK')
        print('################################')

    else:
        inds=None

        text_r='Matrix without NaNs'

        var_array_r=var_array

    return var_array_r, inds, text_r

def interpolation_boundaries(matrix_ori, p_level_ini,p_level_fin, sp_ini,sp_fin, dx_fin, dy_fin,ynew, xnew):

    sp_interpolation,press_interpolation=lat_lon_lengthComparison(p_level_fin,sp_fin,ynew\
    ,xnew,p_level_ini,sp_ini,dx_fin, dy_fin)

    matrix_interpolated=data_interpolation(matrix_ori,'3D',\
    sp_ini,p_level_ini,sp_interpolation,press_interpolation)

    print('############################################')
    print('interpolation_boundaries: matrix interpolated OK')
    print('############################################')

    return matrix_interpolated

def plotCells(axs,cell_data,horPlot,pressPlot,colorMap,limits,xlabel,labels_x_cross,step_hor, title_label,y_label_status,title_font2,label_font, ticks_font,scatter_status,points_scatter):
    """
    This function creates the maps of the seasonal regional circulations
    """
    axs.invert_yaxis()
    axs.set_title(title_label,fontsize=title_font2,loc='left')
    cs=axs.contourf(horPlot,pressPlot,cell_data,limits,cmap=colorMap,extend='both')

    if scatter_status=='yes':
        x, y = np.meshgrid(np.arange(0,horPlot.shape[0],1),np.arange(0, len(pressPlot),1))
        axs.scatter(x, y ,points_scatter*3,zorder=2, c='grey')

    axs.set_xlabel(xlabel,fontsize=label_font)
    if y_label_status=='yes':
        axs.set_ylabel('Pressure [hPa]',fontsize=label_font)
    axs.set_xticks(horPlot[::step_hor])
    axs.set_yticks(pressPlot)
    axs.set_yticklabels(pressPlot,fontsize=ticks_font)
    axs.set_xticklabels(labels_x_cross[::step_hor],fontsize=ticks_font)

    return cs

def plot_boundary(axs,title_str, ref_arr,models_list,models_arr,x_label_str,arange_x,labels_x,legend_status,ylabel_str,title_font2,label_font, ticks_font,lower_st,sep,y_l):
    axs.set_title(title_str,fontsize=title_font2,loc='left')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    #iterating in the models to obtain the serie
    colors=iter(cm.rainbow(np.linspace(0,1,len(models_list))))
    for m in range(len(models_list)):
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
    if lower_st=='Yes':
        axs.set_xlabel(x_label_str,fontsize=label_font)

def labels_str(labels_input,boundary):
    if boundary=='northern' or boundary=='southern':
        labels_x_plot=[]
        absolute=np.abs(labels_input)
        for t in range(len(labels_input)):
            new=str(absolute[t])+'°W'
            labels_x_plot.append(new)
    else:
        labels_x_plot=[]
        for t in range(len(labels_input)):
            if labels_input[t]<0.0:
                abs_coor=np.round(np.abs(labels_input[t]),0)
                new=str(abs_coor)+'°S'
            elif labels_input[t]>0.0:
                new=str(np.round(labels_input[t],0))+'°N'
            else:
                new=str(np.round(labels_input[t],0))
            labels_x_plot.append(new)

    return labels_x_plot

def std_ref(array, path_save_df, str_feature):
    std_ref=np.empty((4))
    for m in range(4):
        std_ref_season=np.nanstd(array[m])
        std_ref[m]=std_ref_season

    new_row_reference={'Characteristic': str_feature,\
    'std_DJF':std_ref[0], 'std_JJA':std_ref[1], 'std_MAM':std_ref[2], 'std_SON':std_ref[3]}

    reference_std_DT=pd.read_csv(path_save_df+'reference_std_original.csv',\
    index_col=[0])
    reference_std_DT=reference_std_DT.append(new_row_reference, ignore_index=True)
    reference_std_DT.to_csv(path_save_df+'reference_std_original.csv')

def agreement_sign(list_models,path_entry_npz,file_name,len_lats, len_lons):

    n_models_group=len(list_models)
    n_models_80_percent=np.round(((n_models_group*80)/100),0)

    var_seasonal_agreement=np.empty((4,len_lats,len_lons))

    for i in range(4):
        var_ensamble=np.empty((len(list_models),len_lats,len_lons))
        var_agree=np.empty((len(list_models),len_lats,len_lons))

        for p in range(len(list_models)):
            var_model_season=np.load(path_entry_npz+list_models[p]+'_'+\
            file_name+'.npz')['arr_0']

            #selecting the specific season

            var_ensamble[p]=var_model_season[i]

        #averaging the models
        var_mean=np.mean(var_ensamble,axis=0)

        for p in range(len(list_models)):
            var_model_season=np.load(path_entry_npz+list_models[p]+'_'+\
            file_name+'.npz')['arr_0'][i]

            positive_change_mag=np.where(var_model_season>=0.0)

            var_agree[p][positive_change_mag]=1

            negative_change_mag=np.where(var_model_season<0.0)

            var_agree[p][negative_change_mag]=-1
        
        #-----------------------------------------------------------------
        for m in range(len_lons):
            for n in range(len_lats):
                cell_mmm_mag=var_mean[n,m]
                models_mag_agree=var_agree[:,n,m]
                if cell_mmm_mag>=0:
                    count_models_agree_mag=np.where(models_mag_agree>=0.0)[0].shape
                else:
                    count_models_agree_mag=np.where(models_mag_agree<0.0)[0].shape

                if  count_models_agree_mag>=n_models_80_percent:
                    var_seasonal_agreement[i,n,m]=1
                else:
        
                    var_seasonal_agreement[i,n,m]=0

    return var_seasonal_agreement

def series_metrics(series_ref,series_models,list_models,feature,path_save_df):

    series_metrics=pd.DataFrame(columns=['Model','Corr','RMSE'])
    series_metrics.to_csv(path_save_df+feature+'_series_metrics_corr_rmse.csv')

    for r in range(len(list_models)):

        model_nm=list_models[r]

        corr_coef=np.corrcoef(series_ref, series_models[r])[0,1]

        MSE = mean_squared_error(series_ref,series_models[r])
        RMSE = math.sqrt(MSE)

        #Appending the dataframe 
        dt_row={'Model':model_nm, 'Corr':corr_coef, 'RMSE':RMSE}

        series_metrics_r=pd.read_csv(path_save_df+feature+'_series_metrics_corr_rmse.csv', index_col=[0])

        series_metrics_r=series_metrics_r.append(dt_row,ignore_index=True)

        series_metrics_r.to_csv(path_save_df+feature+'_series_metrics_corr_rmse.csv')

def series_metrics_bound(series_ref,series_models,list_models,feature,path_save_df):

    series_metrics=pd.DataFrame(columns=['Model','Season','Corr','RMSE'])
    series_metrics.to_csv(path_save_df+feature+'_series_metrics_corr_rmse.csv')

    season_lb=['DJF','JJA','MAM','SON']

    for r in range(len(list_models)):

        model_nm=list_models[r]

        for y in range(4):

            corr_coef=np.corrcoef(series_ref[y], series_models[r,y])[0,1]

            MSE = mean_squared_error(series_ref[y],series_models[r,y])
            RMSE = math.sqrt(MSE)

            #Appending the dataframe 
            dt_row={'Model':model_nm, 'Season':season_lb[y], 'Corr':corr_coef, 'RMSE':RMSE}

            series_metrics_r=pd.read_csv(path_save_df+feature+'_series_metrics_corr_rmse.csv', index_col=[0])

            series_metrics_r=series_metrics_r.append(dt_row,ignore_index=True)

            series_metrics_r.to_csv(path_save_df+feature+'_series_metrics_corr_rmse.csv')



    
    
