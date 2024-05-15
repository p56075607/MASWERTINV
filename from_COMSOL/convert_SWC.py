# %%
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
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
slope = mt.createPolygon(np.array(geo),isClosed=True, marker = 2,boundaryMarker=-1)
mesh = mt.createMesh(slope,area=0.1)
pg.show(mesh,markers=True,showMesh=True)

# %%
grid_z = griddata((df[['X', 'Y']].to_numpy()), df['resistivity'].to_numpy(), 
                  (np.array(mesh.cellCenters())[:, :2]), method='linear')

# %%
pg.show(mesh, data=grid_z, label='resistivity')

# %%
# %%
