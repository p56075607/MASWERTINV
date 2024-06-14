# %%
# Importing libraries
import numpy as np
import pygimli as pg
import os
import subprocess
from numpy import newaxis
from run_five_steady import run_five_steady
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Setting nodes
yTHMC = -np.round(
                pg.utils.grange(0, 2, n=41)
                
                ,2)[::-1]
xTHMC = np.round(
                pg.utils.grange(0, 0.5, n=2)
                ,1)
hydroDomain = pg.createGrid(x=xTHMC,
                                y=yTHMC)

pg.show(hydroDomain)

# %%
dat_file_name = r'C:\Users\Git\MASWERTINV\THMC\EX01_INPUT\workspace\PostP\001_H_SbFlow.dat'
timedata_index = []
solution_times = []
Line = []
with open(dat_file_name, 'r') as read_obj:
    for i, line in enumerate(read_obj):
        if line.startswith('ZONE'):
            timedata_index.append(i)
            # ´£¨úSOLUTIONTIMEªº­È
            parts = line.split(',')
            for part in parts:
                if 'SOLUTIONTIME' in part:
                    time_str = part.split('=')[-1].strip()
                    solution_times.append(float(time_str))
                    break
        Line.append(line)

Var = []
Var_ind = 4 #Var_ind = Saturation:2, Porocity:3, SWC:4
for i in range(len(timedata_index)):
    for j in range(len(hydroDomain.nodes())):
        Var.append(float(Line[timedata_index[i]+1+j].split()[Var_ind]))

    Variable = np.reshape(Var,[len(yTHMC),len(xTHMC)])
    Var = []
    if i == 0:
        Variable_all = Variable
        Variable_all = Variable_all[:, :, newaxis]
    else:
        Var_reshape = Variable
        Variable_all = np.dstack((Variable_all, Var_reshape))
# %%
rainfall_rate_cmday = 5
clim = [0,0.46]
fig, ax = plt.subplots(figsize=(8,8))
for i in range(20):
    t = solution_times[i*2]
    area = np.trapz(-yTHMC, x = (Variable_all[:,1,i] - Variable_all[:,1,0*24]))
    infiltrated_water = rainfall_rate_cmday*t/100

    Variable_t = Variable_all[:,:,i]
    ax.plot(Variable_t[:,1],yTHMC,label = r'day {}: $\Delta =${:.3f}, R = {:.3f} '.format(t,area,infiltrated_water))
    ax.set_xlim(clim)


ax.set_xlabel('Soil Water Content')
ax.set_ylabel('z (m)')
ax.set_title('SWC distribution for rainfall rate {} ($cm/day$)'.format(rainfall_rate_cmday))
ax.legend()
ax.grid(linestyle='--',alpha=0.5)