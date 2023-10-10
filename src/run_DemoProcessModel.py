import numpy as np
import pandas as pd
import os
from math import pi, exp, sqrt
from scipy.interpolate import interp1d
from copy import deepcopy
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit

os.chdir("/home/robert/Projects/WQ_MCL/src")
from processBased_lakeModel_functions import get_hypsography, provide_meteorology, initial_profile, run_wq_model, wq_initial_profile, provide_phosphorus #, heating_module, diffusion_module, mixing_module, convection_module, ice_module


## lake configurations
zmax = 25 # maximum lake depth
nx = 25 * 2 # number of layers we will have
dt = 3600 # 24 hours times 60 min/hour times 60 seconds/min
dx = zmax/nx # spatial step

## area and depth values of our lake 
hyps_all = get_hypsography(hypsofile = '../input/bathymetry.csv',
                            dx = dx, nx = nx)
                            
## atmospheric boundary conditions
meteo_all = provide_meteorology(meteofile = '../input/Mendota_2002.csv',
                    secchifile = None, 
                    windfactor = 1.0)
                     
## time step discretization                      
hydrodynamic_timestep = 24 * dt
total_runtime =  (365) * hydrodynamic_timestep/dt  #365 *1 # 14 * 365
startTime =   (120 + 365*12) * hydrodynamic_timestep/dt #150 * 24 * 3600
endTime =  (startTime + total_runtime) # * hydrodynamic_timestep/dt) - 1

startingDate = meteo_all[0]['date'][startTime] #* hydrodynamic_timestep/dt]
endingDate = meteo_all[0]['date'][(endTime-1)]#meteo_all[0]['date'][(startTime + total_runtime)]# * hydrodynamic_timestep/dt -1]

times = pd.date_range(startingDate, endingDate, freq='H')

nTotalSteps = int(total_runtime)

## here we define our initial profile
u_ini = initial_profile(initfile = '../input/observedTemp.txt', nx = nx, dx = dx,
                     depth = hyps_all[1],
                     startDate = startingDate)

wq_ini = wq_initial_profile(initfile = '../input/mendota_driver_data_v2.csv', nx = nx, dx = dx,
                     depth = hyps_all[1], 
                     volume = hyps_all[2][:-1],
                     startDate = startingDate)

tp_boundary = provide_phosphorus(tpfile =  '../input/Mendota_observations_tp.csv', 
                                 startingDate = startingDate)

tp_boundary = tp_boundary.dropna(subset=['tp'])

Start = datetime.datetime.now()

    
res = run_wq_model(  
    u = deepcopy(u_ini),
    o2 = deepcopy(wq_ini[0]),
    docr = deepcopy(wq_ini[1]),
    docl = deepcopy(wq_ini[1]),
    pocr = 1.27 * hyps_all[2][:-1],
    pocl = 1.27 * hyps_all[2][:-1],
    startTime = startTime, 
    endTime = endTime, 
    area = hyps_all[0][:-1],
    volume = hyps_all[2][:-1],
    depth = hyps_all[1][:-1],
    zmax = zmax,
    nx = nx,
    dt = dt,
    dx = dx,
    daily_meteo = meteo_all[0],
    secview = meteo_all[1],
    phosphorus_data = tp_boundary,
    ice = False,
    Hi = 0,
    Hs = 0,
    Hsi = 0,
    iceT = 6,
    supercooled = 0,
    diffusion_method = 'pacanowskiPhilander',#'pacanowskiPhilander',# 'hendersonSellers', 'munkAnderson' 'hondzoStefan'
    scheme ='implicit',
    km = 1.4 * 10**(-7), # 4 * 10**(-6), 
    k0 = 1 * 10**(-2),
    weight_kz = 0.5,
    kd_light = 0.5, 
    denThresh = 1e-2,
    albedo = 0.01,
    eps = 0.97,
    emissivity = 0.97,
    sigma = 5.67e-8,
    sw_factor = 1.0,
    wind_factor = 1.0,
    at_factor = 1.0,
    turb_factor = 1.0,
    p2 = 1,
    B = 0.61,
    g = 9.81,
    Cd = 0.0013, # momentum coeff (wind)
    meltP = 1,
    dt_iceon_avg = 0.8,
    Hgeo = 0.1, # geothermal heat 
    KEice = 0,
    Ice_min = 0.1,
    pgdl_mode = 'on',
    rho_snow = 250,
    IP = 0.1,
    delta= 1.08,
    conversion_constant = 0.1,
    sed_sink = -1.0 / 86400,
    k_half = 0.5,
    resp_docr = -0.001,
    resp_docl = -0.01,
    resp_poc = -0.1,
    settling_rate = 0.0)

temp=  res['temp']
o2=  res['o2']
docr=  res['docr']
docl =  res['docl']
pocr=  res['pocr']
pocl=  res['pocl']
diff =  res['diff']
avgtemp = res['average'].values
temp_initial =  res['temp_initial']
temp_heat=  res['temp_heat']
temp_diff=  res['temp_diff']
temp_mix =  res['temp_mix']
temp_conv =  res['temp_conv']
temp_ice=  res['temp_ice']
meteo=  res['meteo_input']
buoyancy = res['buoyancy']
icethickness= res['icethickness']
snowthickness= res['snowthickness']
snowicethickness= res['snowicethickness']

heat_flux_total = meteo[1,] + meteo[2,] + meteo[3,] + meteo[4,]
plt.plot(meteo[1,], color = 'blue') # longwave
plt.plot(meteo[2,], color = 'red') # latent
plt.plot(meteo[3,], color = 'orange') # sensible
plt.plot(meteo[4,], color = 'green') # shortwave
plt.show()

plt.plot(heat_flux_total)

# convert averages from array to data frame
avgtemp_df = pd.DataFrame(avgtemp, columns=["time", "thermoclineDep", "epi", "hypo", "tot", "stratFlag"])
avgtemp_df.insert(2, "icethickness", icethickness[0,], True)
avgtemp_df.insert(2, "snowthickness", snowthickness[0,], True)
avgtemp_df.insert(2, "snowicethickness", snowicethickness[0,], True)

End = datetime.datetime.now()
print(End - Start)

    
# epi/hypo/total
colors = ['#F8766D', '#00BA38', '#619CFF']
avgtemp_df.plot(x='time', y=['epi', 'hypo', 'tot'], color=colors, kind='line')
plt.show()

# stratflag
avgtemp_df.plot(x='time', y=['stratFlag'], kind='line', color="black")
plt.show()

# thermocline depth
avgtemp_df.plot(x='time', y=['thermoclineDep'], color="black")
plt.gca().invert_yaxis()
plt.scatter(avgtemp_df.time, avgtemp_df.stratFlag, c=avgtemp_df.stratFlag)
plt.show()

# ice thickness
avgtemp_df.plot(x='time', y=['icethickness'], color="black")
plt.show()

# snowice thickness
avgtemp_df.plot(x='time', y=['snowicethickness'], color="black")
plt.show()

# snow thickness
avgtemp_df.plot(x='time', y=['snowthickness'], color="black")
plt.show()

# heatmap of temps  
plt.subplots(figsize=(140,80))
sns.heatmap(temp, cmap=plt.cm.get_cmap('Spectral_r'), xticklabels=1000, yticklabels=2)
plt.show()

# heatmap of diffusivities  
plt.subplots(figsize=(140,80))
sns.heatmap(diff, cmap=plt.cm.get_cmap('Spectral_r'), xticklabels=1000, yticklabels=2)
plt.show()

# heatmap of oxygen  
plt.subplots(figsize=(140,80))
sns.heatmap(o2, cmap=plt.cm.get_cmap('Spectral_r'), xticklabels=1000, yticklabels=2)
plt.show()

# heatmap of docr  
plt.subplots(figsize=(140,80))
sns.heatmap(docr, cmap=plt.cm.get_cmap('Spectral_r'), xticklabels=1000, yticklabels=2)
plt.show()

# heatmap of docl 
plt.subplots(figsize=(140,80))
sns.heatmap(docl, cmap=plt.cm.get_cmap('Spectral_r'), xticklabels=1000, yticklabels=2)
plt.show()



time_step = 210 * 24 
depth_plot = hyps_all[1][:-1]
fig=plt.figure()
plt.plot(temp_initial[:,time_step], depth_plot, color="black")
plt.plot(temp_heat[:,time_step], depth_plot,color="red")
plt.plot(temp_diff[:,time_step], depth_plot,color="yellow")
plt.plot(temp_conv[:,time_step], depth_plot,color="green")
plt.plot(temp_ice[:,time_step], depth_plot,color="blue")
plt.gca().invert_yaxis()
plt.show()

fig=plt.figure()
plt.plot(diff[:,time_step], depth_plot, color="black")
plt.gca().invert_yaxis()
plt.show()



# heatmap of temps  
N_pts = 6

## function to calculate density from temperature

def calc_dens(wtemp):
    dens = (999.842594 + (6.793952 * 1e-2 * wtemp) - (9.095290 * 1e-3 *wtemp**2) +
      (1.001685 * 1e-4 * wtemp**3) - (1.120083 * 1e-6* wtemp**4) + 
      (6.536336 * 1e-9 * wtemp**5))
    return dens

fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(temp, cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 35)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Water Temperature  ($^\circ$C)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
ax.set_xticklabels(time_label, rotation=0)
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()
