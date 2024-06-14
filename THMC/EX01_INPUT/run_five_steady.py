# %%
import sys
import subprocess as subprocess
from calendar import c
import pygimli as pg
import pandas as pd
import numpy as np
from numpy import newaxis
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams["font.family"] = "Book Antiqua"
from os.path import  join
import os
from datetime import datetime, timedelta
from scipy.optimize import root

def run_five_steady():
    fig_save_ph = r'C:\Users\Git\MASWERTINV\THMC\EX01_INPUT\result'

    # Setting five-steady nodes
    yTHMC = -np.round(
                    pg.utils.grange(0, 5, n=101)
                    
                    ,2)[::-1]
    xTHMC = np.round(
                    pg.utils.grange(0, 1, n=11)
                    ,1)
    hydroDomain = pg.createGrid(x=xTHMC,
                                    y=yTHMC)

    # Writing output file .dm 注意:THMC 網格節點編號要從1開始
    data_write_path = r'C:\Users\Git\MASWERTINV\THMC\EX01_INPUT\IC\five.dm'
    with open(data_write_path, 'w') as write_obj:
        write_obj.write('DATA SET 1 NODAL POINT COORDINATES \n')
        write_obj.write('1   %d   0  0\n'%(len(hydroDomain.nodes())))
        for _,n in enumerate(hydroDomain.nodes()):
            write_obj.write('%d   %.2f  %.2f\n'%(n.id()+1,n.x(),n.y()))

        write_obj.write('DATA SET 2 ELEMENT INCIDENCES \n')
        write_obj.write('1   %d   0\n'%(len(hydroDomain.cells())))
        for cell in hydroDomain.cells():
            if cell.center().y() > -2:
                write_obj.write('%d    %d    %d    %d    %d\n'%(cell.nodes()[0].id()+1,
                                                                cell.nodes()[1].id()+1,
                                                                cell.nodes()[2].id()+1,
                                                                cell.nodes()[3].id()+1,
                                                                1))
            else:
                write_obj.write('%d    %d    %d    %d    %d\n'%(cell.nodes()[0].id()+1,
                                                                cell.nodes()[1].id()+1,
                                                                cell.nodes()[2].id()+1,
                                                                cell.nodes()[3].id()+1,
                                                                1))


    # DATA SET 6: Material properties
    saturation = np.linspace(0.1,1,16)

    # [Soil] Van Genuchten 
    theta_r = 0.034  
    theta_s = 0.46  
    alpha = 1.6     
    n = 1.37     
    Ks = 1.39e-6 #[m/s] 
    Ks = Ks*3600 # [m/hour]

    def van_genuchten_inv(theta, theta_r, theta_s, alpha, n):
        # Calculate effective saturation
        S_vg = (theta - theta_r) / (theta_s - theta_r)
        # Ensure values are within valid range
        S_vg = np.clip(S_vg, 0, 1)
        print(S_vg)
        m = 1 - 1/n
        h = np.zeros(len(S_vg))
        for i in range(len(S_vg)):
            if S_vg[i] == 1:
                h[i] = 0
            elif S_vg[i] < 1:
                # Calculate pressure head using van Genuchten equation
                try:
                    h[i] = (1/alpha) * ((S_vg[i]**(-1/m) - 1)**(1/n))
                except (ZeroDivisionError, ValueError) as e:
                    print(f"Error in calculation at index {i}: {e}")
                    h[i] = np.nan
        return h
    
    def van_genuchten_K(theta, theta_r, theta_s, n):
        S_vg = (theta - theta_r) / (theta_s - theta_r)
        
        return S_vg**(1/2)*(1- (1- S_vg**(n/(n-1)))**((n-1)/n) )**2
    

    # INP FILE
    # Writing output file .inp 
    data_write_path = r'C:\Users\Git\MASWERTINV\THMC\EX01_INPUT\IC\five.inp'
    with open(data_write_path, 'w') as write_obj:
        write_obj.write('99          TARI_WF hydrogeological simulation twenty-steady (kg-m-hour) \n')
        write_obj.write(' 1  0  0  0  0  0  1  1  1 0 0 0\n 0\n')

        write_obj.write('DATA SET 2: Coupling iteration parameters \n')
        write_obj.write(' 1   1   1  1\n')

        write_obj.write('DATA SET 3: Iteration and optional parameters \n')
        write_obj.write('-1  0\n50    25    100    1    0\n 0     0      0    0    0   0   12\n 0.002  0.002   1.0  1.0  1.0  0.5   0\n')
        
        write_obj.write('DATA SET 4: Time step and printouts as well as storage control \n')
        write_obj.write('0  0   0.25  1.0  256.0  8000.0\n1\n0  5  \n1\n0\n')
        write_obj.write('1  1.0D38\n')

        write_obj.write('DATA SET 6: Material properties \n')
        # write_obj.write('1\n')
        # write_obj.write('11    0     16    ')

        NMAT = 1 # No. of material types.
        write_obj.write('{:d}\n'.format(NMAT))
        write_obj.write('11    0     16    ')
        for i in range(NMAT):
            write_obj.write('{:d}    '.format(1))
        write_obj.write('1.27e+08  0.0\n')
        Ks_i =  [Ks] # Unit：[m/hour]
        phi_i = [theta_s]
        rho_i = [1600]
        theta_ri = [theta_r]
        for i in range(NMAT):
            write_obj.write('0.0   0.0   {:.2f}  0  {:.1e}    0.0  1000  3.607   {:d}  {:.2f} 0.0\n'.format(
                                                                            phi_i[i],Ks_i[i],rho_i[i],theta_ri[i]))
        for i in range(NMAT):
            h_ins = -van_genuchten_inv(saturation*theta_s, theta_r, theta_s, alpha, n) # Unit：[m]
            print(h_ins)
            K_ins = van_genuchten_K(saturation*theta_s, theta_r, theta_s, n)
            water_capacity = np.gradient(saturation*phi_i[i],h_ins,edge_order=1)
            for i in range(len(saturation)):
                write_obj.write('{:.3e}   {:.3e}   {:.3e}    {:.3e}\n'.format(
                                        h_ins[i],saturation[i],K_ins[i],water_capacity[i]))

        write_obj.write('DATA SET 19: Input for initial or pre-initial conditions for flow \n')
        write_obj.write(' 1\n')
        # Initial flow 
        h_0 = -1033.2
        h_5 = 0
        pressure_head = interpolate.interp1d([0,500],[h_0,h_5], fill_value="extrapolate")
        pressure_head_profile = pressure_head(yTHMC*-100)/100

        for _,n in enumerate(hydroDomain.nodes()):
            yCord = round(n.y(),1)
            write_obj.write('%d   %.3f\n'%(n.id()+1,pressure_head_profile[yTHMC == yCord]))

        write_obj.write('DATA SET 20: Element (distributed) source/sink for flow \n')
        write_obj.write('0   0   0\n')
        write_obj.write('DATA SET 21: Point (well) source/sink data for flow \n')
        write_obj.write('0  0    0\n')
        write_obj.write('DATA SET 22: Variable rainfall/evaporation-seepage B.C.\n')
        write_obj.write('1  10    11    1      2\n')
        write_obj.write('      0.0   0   1.0D38   0\n')
        write_obj.write(' 1   10     1     1101     1\n')
        write_obj.write(' 0    0     0      0      0\n')
        write_obj.write(' 1   9     1      1      0\n')
        write_obj.write(' 0    0     0      0      0\n')
        write_obj.write(' 1   10     1      0.0    0.0   0.0\n')
        write_obj.write(' 0    0     0      0      0   0\n')
        write_obj.write(' 1   10     1    -90D2    0.0     0.0\n')
        write_obj.write(' 0    0     0      0      0       0 \n')
        write_obj.write(' 1   9    991    1      2     1     1    1    1\n')
        write_obj.write('0     0     0      0      0     0     0    0    0\n')
        write_obj.write('DATA SET 23: Dirichlet BC. for flow\n')
        write_obj.write('11       1        2\n')
        write_obj.write('  0.0     -5    1.0D38  -5\n')
        write_obj.write('1   10   1   1   1\n')
        write_obj.write('0   0   0   0   0\n')
        write_obj.write('1       10        1       1      0\n')
        write_obj.write('0       0        0       0      0\n')
        write_obj.write('DATA SET 24: Cauchy B.C. for flow\n')
        write_obj.write('0       0        0       0      0\n')
        write_obj.write('DATA SET 25: Neumann B.C. for flow\n')
        write_obj.write('0       0        0       0      0\n')
        write_obj.write('DATA SET 26: River B.C. for flow\n')
        write_obj.write('0       0        0       0      0      0\n')
        write_obj.write('0    END OF JOB ------\n')

    # Run THMC2D.exe
    os.chdir(r'C:\Users\Git\MASWERTINV\THMC\EX01_INPUT\IC')
    # os.startfile("THMC2D.exe")
    print('Running THMC2D.exe, please wait ...')
    p = subprocess.Popen("THMC2D.exe")
    p_status = p.wait()
    print('THMC COMPLETED!!')


    # Writing output file PostP.txt
    data_write_path = r'C:\Users\Git\MASWERTINV\THMC\EX01_INPUT\IC\PostP.txt'
    with open(data_write_path, 'w') as write_obj:
        write_obj.write('2\n')
        write_obj.write('ENDDIMENSION\n')
        write_obj.write('\n')
        write_obj.write('11\n')
        write_obj.write('ENDFUNCTIONMOD\n')
        write_obj.write('\n')
        write_obj.write('\n')
        write_obj.write('ENDINITIALCONDITION\n')
        write_obj.write('\n')
        write_obj.write('\n')
        write_obj.write('ENDINITIALSTEP\n')
        write_obj.write('\n')
        write_obj.write('0\n')
        write_obj.write('ENDSTEADYSTEP\n')
        write_obj.write('\n')
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
    os.chdir(r'C:\Users\Git\MASWERTINV\THMC\EX01_INPUT\IC')
    # os.startfile("FUPP.exe")
    print('Running FUPP.exe to convert the THMC output file, please wait ...')
    p = subprocess.Popen("FUPP.exe")
    p_status = p.wait()
    print('FUPP COMPLETED!!')

    # Load ASCII dat file
    dat_file_name = r'C:\Users\Git\MASWERTINV\THMC\EX01_INPUT\IC\PostP\001_H_Dnsty_Prss.dat'
    timedata_index = []
    Line = []
    with open(dat_file_name, 'r') as read_obj:
        for i,line in enumerate(read_obj):
                if line[0] == 'Z':
                    timedata_index.append(i)
                
                Line.append(line)

    Var = []
    Var_ind = 3
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

    # THMC result visualization 
    Presure = Variable_all[:,:,0]
    XTHMC,YTHMC = np.meshgrid(xTHMC,yTHMC)

    clim = [np.amin(Presure),0]
    fig, ax = plt.subplots()
    ax.contourf(XTHMC,YTHMC,Presure
                    ,levels = 100
                    ,cmap='jet_r'
                    ,vmin=clim[0],vmax=clim[1]
                    )
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax_pos = ax.get_position()
    ax2 = ax.twiny()
    ax2.set_xlim([np.floor(np.amin(Presure)),0])
    ax2.set_xlabel('Steady Pressure ($mH_2O$)')
    ax2.plot(Presure[:,1],yTHMC)
    ax2.set_position(ax_pos)

    cbaxes = fig.add_axes([0.665, 0.1, 0.03, 0.8])
    m = plt.cm.ScalarMappable(cmap=plt.cm.jet_r)
    m.set_array(Presure)
    m.set_clim(clim[0],clim[1])
    cb = plt.colorbar(m, boundaries=np.linspace(clim[0],clim[1], 32),cax=cbaxes)
    cb_ytick = np.linspace(clim[0],clim[1],8)
    cb.ax.set_yticks(cb_ytick)
    cb.ax.set_yticklabels(['{:.2f}'.format(x) for x in cb_ytick])
    cb.ax.set_ylabel('Pressure ($mH_2O$)')

    fig.savefig(join(fig_save_ph,'Steady_IC K={} phi={} rho={}.png'.format(
                                                                            Ks_i,
                                                                            phi_i,
                                                                            rho_i))
                                                                            ,dpi=400, bbox_inches='tight')

    plt.close()
    return Presure

# pressure = run_five_steady()