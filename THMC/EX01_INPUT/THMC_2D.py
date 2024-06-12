# %%
from run_twenty_steady import run_twenty_steady
import sys
sys.path.append(r'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\WF')
from swrc_func_2 import Campbell
from datenum_to_datetime import datenum_to_datetime
from swrc_func_2 import SWRC_mod_func_2
from read_THMC_flow_result import read_THMC_flow_result
import winsound
import subprocess as subprocess
from calendar import c
import pygimli as pg
import pandas as pd
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate
plt.rcParams["font.family"] = "Book Antiqua"
%config InlineBackend.figure_format='svg'
import vtuIO
from os.path import  join
import os
import pickle
from datetime import datetime, timedelta
fig_save_ph = r'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\THMC\result'

# %%
# Setting nodes
yTHMC = -np.round(pg.cat(pg.cat(pg.utils.grange(0, 0.6, n=31),
                                pg.utils.grange(0.64, 1, n=10)),

                         pg.cat(pg.utils.grange(1.1, 2, n=10),
                                pg.utils.grange(2.2, 5, n=15)))
                 ,2)[::-1]
xTHMC = np.round(pg.cat(pg.cat(pg.utils.grange(20, 24, n=11),
                        pg.utils.grange(24.2, 26, n=10)),
                pg.utils.grange(26.4, 30, n=10))
        ,1)
hydroDomain = pg.createGrid(x=xTHMC,
                                y=yTHMC)
# %%
XTHMC,YTHMC,fill_alluvium = pickle.load(open("fill_and_alluvium_material.pckl","rb"))
fig, ax = plt.subplots(figsize=(10,10))
pcolor_plot1 = ax.pcolormesh(XTHMC,YTHMC,fill_alluvium, edgecolors='k', linewidths=0.08)
ax.set_xlabel('x (m)')
ax.set_ylabel('z (m)')
ax.set_aspect('equal')

ax.scatter(25.5,-0.1,s=10,label='10 cm')
ax.scatter(25.5,-0.2,s=10,label='20 cm')
ax.scatter(25.5,-0.3,s=10,label='30 cm')
ax.scatter(25.5,-0.5,s=10,label='50 cm')
ax.scatter(25.5,-1,s=10  ,label='100 cm')

ax.set_xlim([20,30])  
ax.set_yticks([0,-0.14,-0.4,-0.76,-1,-2.0,-5])
ax.set_ylim([-3,0]) 
ax.legend(fontsize=5, loc='lower right')
# %% DM FILE
# Writing output file .dm æ³¨æ??:THMC ç¶²æ?¼ç??é»?ç·¨è??è¦?å¾?1???å§?
data_write_path = r'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\THMC\workspace_2D\TARI.dm'
with open(data_write_path, 'w') as write_obj:
    write_obj.write('DATA SET 1 NODAL POINT COORDINATES \n')
    write_obj.write('1   %d   0  0\n'%(len(hydroDomain.nodes())))
    for _,n in enumerate(hydroDomain.nodes()):
        write_obj.write('%d   %.2f  %.2f\n'%(n.id()+1,n.x(),n.y()))

    write_obj.write('DATA SET 2 ELEMENT INCIDENCES \n')
    write_obj.write('1   %d   0\n'%(len(hydroDomain.cells())))
    for i,cell in enumerate(hydroDomain.cells()):
        write_obj.write('%d    %d    %d    %d    %d\n'%(cell.nodes()[0].id()+1,
                                                        cell.nodes()[1].id()+1,
                                                        cell.nodes()[2].id()+1,
                                                        cell.nodes()[3].id()+1,
                                                        1+fill_alluvium[int(np.floor(i/30)),i%30]))

# %% INP FILE
# Input hydrological data
hydro_path = 'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\WF'
hydro_fname = "hydrolic_one_hour.csv"

hydro_data=pd.read_csv(join(hydro_path,hydro_fname),encoding='utf-8')
hydro_data['WF_datetime'] = pd.to_datetime(hydro_data['WF_datetime'])
hydro_data['datetime2'] = hydro_data['WF_datetime']
hydro_data = hydro_data.set_index('WF_datetime')

time0 = '2022-01-25 12:00:00'
time1 = '2022-03-06 00:00:00'
hydro_data = hydro_data[time0:time1]

# DATA SET 22: Variable rainfall/evaporation-seepage B.C.
# Rainfall bounday condition as time series
rain = hydro_data['rain']/1000 # Unitï¼?[m]
rain_timeseries = []
for i in range(len(rain)):
    if i == 0:
        rain_timeseries.append([i,rain[i]])
    elif i == len(rain)-1:
        if rain[i] != rain[i-1]:
            rain_timeseries.append([i-0.001,rain[i-1]])
        rain_timeseries.append([i,rain[i]])
        rain_timeseries.append([1e+38,rain[i]])
    else:
        if rain[i] != rain[i-1]:
            rain_timeseries.append([i-0.001,rain[i-1]])
            rain_timeseries.append([i,rain[i]])


# DATA SET 6: Material properties
# Input hydrological data
hydro_path = 'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\WF'
hydro_fname = "hydrolic_one_hour.csv"

hydro_data=pd.read_csv(join(hydro_path,hydro_fname),encoding='utf-8')
hydro_data['WF_datetime'] = pd.to_datetime(hydro_data['WF_datetime'])
hydro_data['datetime2'] = hydro_data['WF_datetime']
hydro_data = hydro_data.set_index('WF_datetime')

# hydro_data = pd.concat([hydro_data['2022-02-24 12:00:00':'2022-03-04 00:00:00']
#                    ,hydro_data['2022-03-09 00:00:00':'2022-03-23 00:00:00']])
psi_target = 'suction_1'
theta_target = 'SWC_4'
psi=-np.copy(hydro_data[psi_target])  #first column stored as water potential psi
theta=np.copy(hydro_data[theta_target]) #second column stored as theta
Solver = 'trf' # select 'lm', 'trf' or 'dogbox'
SWRfunc = Campbell # select VG or BC or Campbell
init_vals = [0.3,1,0.083] # 3 or 4 parameters: (theta_r),phi,[n or lamda], [alpha or beta] 
bvals = [(0.3  ,0.1  ,0),
         (0.5,5,1)] #give the lower and upper bounds

psimodel,fitP,error,R2 = SWRC_mod_func_2(psi,theta,SWRfunc,Solver,init_vals,bvals)

# Capillary Suction vs Relative Hydraulic Conductivity. Unitï¼?[m]
phi = fitP[0]
lamda = fitP[1]
betta = fitP[2]
saturation = np.linspace(0.01,1,16)
# %%
data_write_path = r'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\THMC\workspace_2D\TARI.inp'
NMAT = 2 # No. of material types.
Ks1 = 0.0008 
Ks2 = 0.01   
 
# ??®ä??: m/hr

phi1 = 0.4404
phi2 = 0.2


Ks_i =  [Ks1,Ks2] # Unitï¼?[m/hour]
phi_i = [phi1,phi2]
rho_i = [1600,2180]
theta_ri = [0.05,0.01]
# Writing output file .inp   
with open(data_write_path, 'w') as write_obj:
    write_obj.write('100          TARI_WF hydrogeological simulation (kg-m-hour) \n')
    write_obj.write(' 1  0  0  0  0  0  1  1  1 0 0 0\n 0\n')

    write_obj.write('DATA SET 2: Coupling iteration parameters \n')
    write_obj.write(' 1   1   1  1\n')

    write_obj.write('DATA SET 3: Iteration and optional parameters \n')
    write_obj.write('-1  0\n50    25    100    1    0\n 0     0      0    1    0   0   12\n 0.002  0.002   1.0  1.0  1.0  0.5   0\n')
    
    write_obj.write('DATA SET 4: Time step and printouts as well as storage control \n')
    NTIF = len(rain) # Number of time steps or time increments for flow simulations.
    write_obj.write('{:d}  0   1  0  64.0  8000.0 \n0\n67   5\n'.format(NTIF))
    NTISTO = int(len(rain)) # No. of additional time-steps to store flow, transport and heat transfer simulations in auxiliary storage device.
    write_obj.write('{:d}\n'.format(NTISTO))
    for i in range(NTISTO):
        write_obj.write('{:d} '.format(i+1))
    write_obj.write('\n1  1.0D38\n')

    write_obj.write('DATA SET 6: Material properties \n')
    write_obj.write('{:d}\n'.format(NMAT))
    write_obj.write('11    0     16    ')
    for i in range(NMAT):
        write_obj.write('{:d}    '.format(1))
    write_obj.write('1.27e+08  0.0\n')

    for i in range(NMAT):
        write_obj.write('0.0   0.0   {:.2f}  {:.1e}  {:.1e}    0.0  1000  3.607   {:d}  {:.2f} 0.0\n'.format(
                                                                        phi_i[i],Ks_i[i],Ks_i[i],rho_i[i],theta_ri[i]))
    for i in range(NMAT-1):
        h_ins = -(1/betta)*(1/saturation)**(1/lamda)/100 # Unitï¼?[m]
        K_ins = (saturation)**(3+2/lamda)
        water_capacity = np.gradient(saturation*phi_i[i],h_ins,edge_order=1)
        for i in range(len(saturation)):
            write_obj.write('{:.3e}   {:.3e}   {:.3e}    {:.3e}\n'.format(
                                    h_ins[i],saturation[i],K_ins[i],water_capacity[i]))

    write_obj.write('-100	0.23637009	     6.13E-07	      0.000333012\n')
    write_obj.write('-90	    0.241629004	     9.68E-07	      0.000455139\n')
    write_obj.write('-80	    0.248147795	     1.61E-06	      0.000347415\n')
    write_obj.write('-70	    0.256447197	     2.87E-06	      0.000469897\n')
    write_obj.write('-60	    0.267382387	     5.58E-06	      0.000544772\n')
    write_obj.write('-50	    0.282462996	     1.22E-05	      0.000800609\n')
    write_obj.write('-40	    0.304625786	     3.19E-05	      0.001293251\n')
    write_obj.write('-30	    0.340426114	     0.000108551	  0.002437219\n')
    write_obj.write('-20	    0.407894245	     0.000591095	  0.00608502\n')
    write_obj.write('-15	    0.469763124	     0.001890583	  0.012769518\n')
    write_obj.write('-10	    0.576342365	     0.008804346	  0.015304208\n')
    write_obj.write('0	    1	             1	              0\n')
    write_obj.write('10	    1	             1	              0\n')
    write_obj.write('20	    1	             1	              0\n')
    write_obj.write('30	    1	             1	              0\n')
    write_obj.write('100	    1	             1	              0\n')

    write_obj.write('DATA SET 19: Input for initial or pre-initial conditions for flow \n')
    write_obj.write(' 1\n')
    # Initial flow ??????è§?æ¸¬æ°´ä½?æ·±åº¦ï¼????è¨­é????¨è?³ç©©???ï¼?è§?å¯????çµ?å£????æ°´é?­å??å¸?ï¼?æ­¤ç???????¶å??æ¨¡æ?¬å??å§?æ¢?ä»?
    yIC_20m = -np.round(
                    pg.cat(pg.utils.grange(0, 5, n=26),
                    pg.utils.grange(5.5, 20, n=30))
                    ,2)[::-1]
    xIC_20m = np.round(
                    pg.utils.grange(20, 30, n=21)
                    ,1)
    pg.boxprint('RUNNING TWENTY STEADY AS INITIAL CONDITION....')
    Pressure = run_twenty_steady(Ks2=0.008,Ks3=0.012,phi2=0.45,phi3=0.2)
    pressure_head = interpolate.interp1d(yIC_20m,Pressure[:,1], fill_value="extrapolate")
    pressure_head_profile = pressure_head(yTHMC)

    for _,n in enumerate(hydroDomain.nodes()):
        yCord = round(n.y(),3)
        write_obj.write('%d   %.3f\n'%(n.id()+1,pressure_head_profile[yTHMC == yCord]))

    write_obj.write('DATA SET 20: Element (distributed) source/sink for flow \n')
    write_obj.write('0   0   0\n')
    write_obj.write('DATA SET 21: Point (well) source/sink data for flow \n')
    write_obj.write('0  0    0\n')
    write_obj.write('DATA SET 22: Variable rainfall/evaporation-seepage B.C.\n')
    write_obj.write('1  30    31    1      {:d}\n'.format(len(rain_timeseries)))
    for i in range(len(rain_timeseries)):
            write_obj.write('{} '.format(rain_timeseries[i][0]))
            write_obj.write('{} '.format(rain_timeseries[i][1]))
    write_obj.write('\n')
    write_obj.write(' 1   30     1     2016     1\n')
    write_obj.write(' 0    0     0      0      0\n')
    write_obj.write(' 1   29     1      1      0\n')
    write_obj.write(' 0    0     0      0      0\n')
    write_obj.write(' 1   30     1      0.0    0.0   0.0\n')
    write_obj.write(' 0    0     0      0      0   0\n')
    write_obj.write(' 1   30     1    -90D2    0.0     0.0\n')
    write_obj.write(' 0    0     0      0      0       0 \n')
    write_obj.write(' 1   29    1921    1      2     1     1    1    1\n')
    write_obj.write('0     0     0      0      0     0     0    0    0\n')
    write_obj.write('DATA SET 23: Dirichlet BC. for flow\n')
    write_obj.write('31       1        2\n')
    H_5m = -5 + pressure_head_profile[0] 
    write_obj.write('  0.0     {:.3f}    1.0D38  {:.3f}\n'.format(H_5m,H_5m))
    write_obj.write('1   30   1   1   1\n')
    write_obj.write('0   0   0   0   0\n')
    write_obj.write('1       30        1       1      0\n')
    write_obj.write('0       0        0       0      0\n')
    write_obj.write('DATA SET 24: Cauchy B.C. for flow\n')
    write_obj.write('0       0        0       0      0\n')
    write_obj.write('DATA SET 25: Neumann B.C. for flow\n')
    write_obj.write('0       0        0       0      0\n')
    write_obj.write('DATA SET 26: River B.C. for flow\n')
    write_obj.write('0       0        0       0      0      0\n')
    write_obj.write('0    END OF JOB ------\n')



# Run THMC2D.exe
os.chdir(r'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\THMC\workspace_2D')
# os.startfile("THMC2D.exe")
pg.boxprint('Running THMC2D.exe, please wait ...')
p = subprocess.Popen("THMC2D.exe")
p_status = p.wait()
pg.boxprint('THMC COMPLETED!!')

# Writing output file PostP.txt
data_write_path = r'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\THMC\workspace_2D\PostP.txt'
with open(data_write_path, 'w') as write_obj:
    
    write_obj.write('2\n')
    write_obj.write('ENDDIMENSION\n')
    write_obj.write('\n')
    write_obj.write('13\n')
    write_obj.write('ENDFUNCTIONMOD\n')
    write_obj.write('\n')
    write_obj.write('\n')
    write_obj.write('ENDINITIALCONDITION\n')
    write_obj.write('\n')
    write_obj.write('\n')
    write_obj.write('ENDINITIALSTEP\n')
    write_obj.write('\n')
    write_obj.write('\n')
    write_obj.write('ENDSTEADYSTEP\n')
    write_obj.write('\n')
    for i in range(int(NTISTO)):
            write_obj.write('{:d}\n'.format(i+1))
    write_obj.write('\n')
    write_obj.write('ENDTRANSIENTSSTEP\n')
    write_obj.write('\n')
    write_obj.write('DNST_PRSS_PLOT: 001_H_Dnsty_Prss.dat   \n')
    write_obj.write('VELOCITY__PLOT: 001_H_FlowVel.dat      \n')
    write_obj.write('SUBFLOW___PLOT: 001_H_SbFlow.dat       \n')
    write_obj.write('HUMIDITY__PLOT: 001_HT_Humid.dat       \n')
    write_obj.write('DSPLMT_F__PLOT: 001_H_AvgDsplmt_F.dat\n')
    write_obj.write('THERMAL___PLOT: 001_T_Tmptr.dat        \n')
    write_obj.write('CONCTOTAL_PLOT: 001_C_CmConcenTotal.dat\n')
    write_obj.write('CONCDISLV_PLOT: 001_C_CmConcenDislv.dat\n')
    write_obj.write('CONCADSRB_PLOT: 001_C_CmConcenAdsrb.dat\n')
    write_obj.write('CONCPRCIP_PLOT: 001_C_CmConcenPrcip.dat\n')
    write_obj.write('CONCMINER_PLOT: 001_C_MnConcen.dat\n')
    write_obj.write('CONCSPCIE_PLOT: 001_C_SpConcen.dat    \n') 
    write_obj.write('PH_VALUE__PLOT: 001_C_pHvalue.dat  \n')
    write_obj.write('DSPLCMT___PLOT: 001_M_Dsplmt.dat \n')
    write_obj.write('STSS_EFF__PLOT: 001_M_StssEff.dat\n')
    write_obj.write('STSS_TTL__PLOT: 001_M_StssTtl.dat\n')
    write_obj.write('STSS_OTHR_PLOT: 001_M_StssOth.dat\n')
    write_obj.write('ENDPLOTNAME\n')
    write_obj.write('\n')
    write_obj.write('\n')
    write_obj.write('ENDCOMPONENTSNAME\n')
    write_obj.write('\n')
    write_obj.write('\n')
    write_obj.write('ENDMINERALNAME\n')
    write_obj.write('\n')
    write_obj.write('ENDSPECIESNAME\n')
    write_obj.write('\n')
    write_obj.write('\n')
    write_obj.write('ENDHYDROGENIONNAME\n')
    write_obj.write('\n')
    write_obj.write('\n')
    write_obj.write('OUTPUT REFERENCE  \n')
    write_obj.write('================================================================================\n')
    write_obj.write('MOD|OUTPUT FILE NAME|THMC|OUTPUT TYPE        |OUTPUT VARIABLES\n')
    write_obj.write('---+----------------+----+-------------------+----------------------------------\n')
    write_obj.write('11 |DNST_PRSS_PLOT: |H   |DENSITY/PRESSURE   |WATER DENSITY,PRESSURE,TOTAL HEAD\n')
    write_obj.write('12 |VELOCITY__PLOT: |H   |FLOW VELOCITY      |FLOW VELOCITY\n')
    write_obj.write('13 |SUBFLOW___PLOT: |H   |SUBSURFACE FLOW    |SATURATION,POROSITY\n')
    write_obj.write('14 |HUMIDITY__PLOT: |HT  |HUMIDITY           |HUMIDITY\n')
    write_obj.write('15 |DSPLMT_F__PLOT: |H   |AVG DSPLMT BY FLOW |AVG DSPLMT BY FLOW\n')
    write_obj.write('21 |THERMAL___PLOT: |T   |THERMAL            |TEMPERATURE\n')
    write_obj.write('31 |CONCTOTAL_PLOT: |C   |TOTAL CONCEN       |COMPONENT TOTAL CONCENTRATION\n')
    write_obj.write('32 |CONCDISLV_PLOT: |C   |TOTAL DISLV CONCEN |COMPONENT TOTAL DISSOLVED CONCENTRATION\n')
    write_obj.write('33 |CONCADSRB_PLOT: |C   |TOTAL ADSRB CONCEN |COMPONENT TOTAL ADSORBED CONCENTRATION\n')
    write_obj.write('34 |CONCPRCIP_PLOT: |C   |TOTAL PRCIP CONCEN |COMPONENT TOTAL PRECIPITATED CONCENTRATION\n')
    write_obj.write('35 |CONCMINER_PLOT: |C   |MINERAL CONCEN     |MINERAL CONCENTRATION\n')
    write_obj.write('36 |CONCSPCIE_PLOT: |C   |SPECIES CONCEN     |SPECIES CONCENTRATION\n')
    write_obj.write('37 |PH_VALUE__PLOT: |C   |PH VALUE           |PH VALUE\n')
    write_obj.write('41 |DSPLCMT___PLOT: |M   |DISPLACEMENT       |DISPLACEMENT\n')
    write_obj.write('42 |STSS_EFF__PLOT: |M   |EFFECTIVE STRESS   |EFFECTIVE STRESS\n')
    write_obj.write('43 |STSS_TTL__PLOT: |M   |TOTAL STRESS       |TOTAL STRESS\n')
    write_obj.write('44 |STSS_OTHR_PLOT: |M   |OTHER STRESS       |PORE WATER PRESSURE,SWELLING PRESSURE,THERMAL STRESS,CHEMICAL STRESS\n')
    write_obj.write('================================================================================\n')


# Run FUPP.exe
os.chdir(r'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\THMC\workspace_2D')
# os.startfile("FUPP.exe")
pg.boxprint('Running FUPP.exe to convert the THMC output file, please wait ...')
p = subprocess.Popen("FUPP.exe")
p_status = p.wait()
pg.boxprint('FUPP COMPLETED!!')


# Load ASCII dat file
dat_file_name = r'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\THMC\workspace_2D\PostP\001_H_SbFlow.dat'
Variable_all,var_10_cm,var_20_cm,var_30_cm,var_50_cm,var_100_cm = read_THMC_flow_result(dat_file_name,Var_ind=4)

# Creat datetime array
time_index = []
for i in range(int(len(rain))):
            time_index.append(i+1)
datetime_array = np.arange(datetime(2022,1,25,12), datetime(2022,3,6,1), timedelta(hours=1)).astype(datetime)
time_index = np.array(time_index)
datetime_THMC = datetime_array[time_index-1]

# Input hydrological data
hydro_path = 'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\WF'
hydro_fname = "hydrolic_one_hour.csv"

hydro_data=pd.read_csv(join(hydro_path,hydro_fname),encoding='utf-8')
hydro_data['WF_datetime'] = pd.to_datetime(hydro_data['WF_datetime'])
hydro_data = hydro_data.set_index('WF_datetime')

time0 = '2022-01-25 12:00:00'
time1 = '2022-03-06 00:00:00'
hydro_data = hydro_data[time0:time1]

# %%

hydro_10 = hydro_data['SWC_1']
hydro_20 = hydro_data['SWC_2']
hydro_30 = hydro_data['SWC_3']
hydro_50 = hydro_data['SWC_4']
hydro_100 = hydro_data['SWC_5']

fig, ax = plt.subplots(2,1,figsize=(14, 10))
line1, = ax[0].plot(datetime_array,hydro_10,label='SWC at 10 cm')
line2, = ax[0].plot(datetime_array,hydro_20 ,label='SWC at 20 cm')
line3, = ax[0].plot(datetime_array,hydro_30 ,label='SWC at 30 cm')
line4, = ax[0].plot(datetime_array,hydro_50 ,label='SWC at 50 cm')
line5, = ax[0].plot(datetime_array,hydro_100,label='SWC at 100 cm')
ax[0].grid()
ax[0].set_xlabel('Date')
ax[0].set_ylabel('Soil Water Content')
ax[0].set_title('Measured Soil Water Content', fontweight='bold')
# ax[0].legend(handles=[line1,line2,line3,line4,line5])
ax[0].set_ylim([0,0.45])
ax2 = ax[0].twinx()
ax2.bar(datetime_array,hydro_data['rain']
                            ,color=[0.3010, 0.7450, 0.9330]
                            ,align='edge'
                            ,width=1/24
                            # ,edgecolor='black'
                            )
ax2.set_ylim([0, 6])
ax2.set_ylabel('Precipitation (mm)',color=[0.3010, 0.7450, 0.9330])

ax[0].set_zorder(ax2.get_zorder()+1)
ax[0].set_frame_on(False)

# THMC result visualization 
RRMSE_10 = np.sqrt(np.mean(((var_10_cm-np.array(hydro_10))**2)))/(max(hydro_10)-min(hydro_10))
RRMSE_20 = np.sqrt(np.mean(((var_20_cm-np.array(hydro_20))**2)))/(max(hydro_20)-min(hydro_20))
RRMSE_30 = np.sqrt(np.mean(((var_30_cm-np.array(hydro_30))**2)))/(max(hydro_30)-min(hydro_30))
RRMSE_50 = np.sqrt(np.mean(((var_50_cm-np.array(hydro_50))**2)))/(max(hydro_50)-min(hydro_50))
RRMSE_100 = np.sqrt(np.mean(((var_100_cm-np.array(hydro_100))**2)))/(max(hydro_100)-min(hydro_100))

line1, = ax[1].plot(datetime_THMC,var_10_cm,label='10 cm, RRMSE={:.3f}'.format(RRMSE_10))
line2, = ax[1].plot(datetime_THMC,var_20_cm,label='20 cm, RRMSE={:.3f}'.format(RRMSE_20))
line3, = ax[1].plot(datetime_THMC,var_30_cm,label='30 cm, RRMSE={:.3f}'.format(RRMSE_30))
line4, = ax[1].plot(datetime_THMC,var_50_cm,label='50 cm, RRMSE={:.3f}'.format(RRMSE_50))
line5, = ax[1].plot(datetime_THMC,var_100_cm,label='100 cm, RRMSE={:.3f}'.format(RRMSE_100))
ax[1].grid()
ax[1].set_xlabel('Date')
ax[1].set_ylabel('Soil Water Content')
ax[1].set_title(r'Simulated Soil Water Content $K_s$={}, {} (m/hour)  $\phi$={}, {} $\rho$={}, {}'.format(
                                                                        Ks_i[0],Ks_i[1],
                                                                        phi_i[0],phi_i[1],
                                                                        rho_i[0],rho_i[1]),
                                                                        fontweight='bold')
ax[1].legend(handles=[line1,line2,line3,line4,line5], loc='center left', bbox_to_anchor=(1.05, 0.1))
ax[1].set_ylim([0,0.45])
ax2 = ax[1].twinx()
ax2.bar(datetime_array,hydro_data['rain']
                            ,color=[0.3010, 0.7450, 0.9330]
                            ,align='edge'
                            ,width=1/24
                            # ,edgecolor='black'
                            )
ax2.set_ylim([0, 6])
ax2.set_ylabel('Precipitation (mm)',color=[0.3010, 0.7450, 0.9330])
ax[1].set_zorder(ax2.get_zorder()+1)
ax[1].set_frame_on(False)
# %%
XTHMC,YTHMC = np.meshgrid(xTHMC,yTHMC)
clim = [0,0.45]
levels = 32
fig, ax = plt.subplots(figsize=(8,8))
for j in range(1):#int(len(datetime_THMC)/2)):
    i = 2*j+700
    Variable_t = Variable_all[:,:,i]
    ax.contourf(XTHMC,YTHMC,Variable_t
                ,levels = levels
                ,cmap='jet_r'
                ,vmin=clim[0],vmax=clim[1]
                )
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')

    divider = make_axes_locatable(ax)
    cbaxes = divider.append_axes("right", size="5%", pad=.15)

    # cbaxes = fig.add_axes([0.92, 0.43, 0.04, 0.4])
    m = plt.cm.ScalarMappable(cmap=plt.cm.jet_r)
    m.set_array(Variable_t)
    m.set_clim(clim[0],clim[1])
    cb = plt.colorbar(m, boundaries=np.linspace(clim[0],clim[1], levels),cax=cbaxes)
    cb_ytick = np.linspace(clim[0],clim[1],8)
    cb.ax.set_yticks(cb_ytick)
    cb.ax.set_yticklabels(['{:.2f}'.format(x) for x in cb_ytick])
    cb.ax.set_ylabel('SWC')
    ax.set_title('SWC profile on {} '.format(datetime_THMC[i]), fontsize=16, fontweight='bold')
# %%
winsound.Beep(500, 750)
winsound.Beep(500, 750)
winsound.Beep(500, 750)