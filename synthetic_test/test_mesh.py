# %%
# Build a two-layer model for electrical resistivity tomography synthetic test using pygimli package
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from os.path import join

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
plt.rcParams["font.family"] = 'Times New Roman'#"Microsoft Sans Serif"
# %%
data = pg.load('simple.dat')

# %%
left = 0
right = 100
depth = 30
world2 = mt.createWorld(start=[left, 0], end=[right, -depth], worldMarker=True)
mesh2 = mt.createMesh(world2, 
                     area=1,
                     quality=33)    # Quality mesh generation with no angles smaller than X degrees 
ax,_ = pg.show(mesh2,markers=True)
ax.set_xlabel('Distance (m)',fontsize=13)
ax.set_ylabel('Depth (m)',fontsize=13)
# %%
mgr2 = ert.ERTManager(data)
# Run the inversion with the preset data. The Inversion mesh will be created
# with default settings.
inv2 = mgr2.invert(mesh=mesh2, lam=100, verbose=True)
# %%
world = mt.createWorld(start=[left, 0], end=[right, -depth], worldMarker=True,marker=2)
mesh = mt.createMesh(world, 
                     area=1,
                     quality=33)    # Quality mesh generation with no angles smaller than X degrees 
mesh_appened = mt.appendTriangleBoundary(mesh,xbound=100,ybound=100,marker=1)
ax,_ = pg.show(mesh_appened,markers=True)
# %%
mgr = ert.ERTManager(data)
# Run the inversion with the preset data. The Inversion mesh will be created
# with default settings.
inv = mgr.invert(mesh=mesh_appened, lam=100, verbose=True)
# %%
# Create a block
plc = mt.createParaMeshPLC(data, paraDepth=depth, boundary=-2)
meshplc = mt.createMesh(plc,
                        area=1,
                        quality=33)    # Quality mesh generation with no angles smaller than X degrees
ax,_ = pg.show(meshplc,markers=True)
ax.set_xlabel('Distance (m)',fontsize=13)
ax.set_ylabel('Depth (m)',fontsize=13)
# %%
mgrplc = ert.ERTManager(data)
# Run the inversion with the preset data. The Inversion mesh will be created
# with default settings.
invplc = mgrplc.invert(mesh=meshplc, lam=100, verbose=True)
# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4))
kw = dict(cMin=50, cMax=150, logScale=True, cMap='jet',
          xlabel='Distance (m)', ylabel='Depth (m)', 
          label=pg.unit('res'), orientation='vertical')
pg.show(mesh2, mgr2.model, ax=ax1, **kw)
# pg.show(mgrplc.paraDomain, mgrplc.model, ax=ax2, **kw)
pg.show(mgr.paraDomain, mgr.model, ax=ax2, **kw)
ax2.set_xlim(left, right)