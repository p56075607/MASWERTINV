# %%
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
from scipy.interpolate import griddata

csv_name = 'water content.csv'
df = pd.read_csv(csv_name, comment='%', header=None)
df.columns = ['X', 'Y', 'theta']
n = 2
cFluid = 0.03
df['resistivity'] = 1/(cFluid*df['theta']**n)

geo = pd.read_csv('geometry.csv', header=None)
geo.columns = ['x', 'y']
# %%
# slope = mt.createPolygon(np.array(geo),isClosed=True, marker = 2,boundaryMarker=-1)
electrode_x = np.linspace(start=6, stop=24, num=25)
electrode_y = np.linspace(start=20, stop=10, num=25)
scheme = pg.physics.ert.createData(elecs=np.column_stack((electrode_x, electrode_y)),
                           schemeName='dd')
plc = mt.createParaMeshPLC(scheme,paraDepth=5,paraMaxCellSize=0.1)
# for p in scheme.sensors():
#     slope.createNode(p)
#     slope.createNode(p - [0, 0.1])
   
mesh = mt.createMesh(plc)
pg.show(mesh,markers=True,showMesh=True)

# %%
grid_resistivity = griddata((df[['X', 'Y']].to_numpy()), df['resistivity'].to_numpy(), 
                  (np.array(mesh.cellCenters())[:, :2]), method='linear', fill_value=np.nan)
fill_value = np.nanmean(grid_resistivity)
grid_resistivity = np.nan_to_num(grid_resistivity, nan=fill_value)
# %%
pg.show(mesh, data=grid_resistivity, label='resistivity',cMap='jet',cMin=272,cMax=1350)

# %%
data = ert.simulate(mesh = mesh, scheme=scheme, res=grid_resistivity, noiseLevel=1,
                    noiseAbs=1e-6, 
                    seed=1337)
# %%
ert.showData(data)

# %%
mgr = ert.ERTManager()
mgr.invert(data=data, mesh=mesh, lam=100,verbose=True)

# %%
mgr.showResultAndFit()