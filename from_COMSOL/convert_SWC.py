# %%
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt

csv_name = 'water content.csv'
df = pd.read_csv(csv_name, comment='%', header=None)
df.columns = ['X', 'Y', 'theta']

print(df)
# plt.scatter(df['X'], df['Y'], c=df['theta'], cmap='viridis')

geo = pd.read_csv('geometry.csv', header=None)
geo.columns = ['x', 'y']
np.array(geo)
# %%
slope = mt.createPolygon(
    np.array(geo),
    isClosed=True, marker = 2,boundaryMarker=-1)
mesh = mt.createMesh(slope, area=0.1)
pg.show(mesh,markers=True,showMesh=True)