# %%
# Build a two-layer model with 2 artificial structures for ERT synthetic test using pygimli package
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from os.path import join

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
plt.rcParams["font.family"] = 'Times New Roman'#"Microsoft Sans Serif"
# %% Creat dry time model
# Geometry definition
# Create geometry definition for the modelling domain. 

left = 0
right = 128
depth = 30

world = mt.createWorld(start=[left, 0], end=[right, -depth])

artif1 = mt.createRectangle(start=[37.5, 0], end=[42.5, -10],marker=2)
artif2 = mt.createRectangle(start=[77.5, 0], end=[82.5, -10],marker=3)
geom = world + artif1 + artif2
pg.show(geom, markers=True)

# Synthetic data generation
# Create a Dipole Dipole ('dd') measuring scheme with 21 electrodes.
scheme = ert.createData(elecs=np.linspace(start=0, stop=128, num=65),
                           schemeName='dd')

# Create a mesh for the finite element modelling with appropriate mesh quality.
mesh = mt.createMesh(geom, 
                     area=1,
                     quality=33)    # Quality mesh generation with no angles smaller than X degrees 
pg.show(mesh,markers=False)

# Create a map to set resistivity values in the appropriate regions
# [[regionNumber, resistivity], [regionNumber, resistivity], [...]
rhomap = [[1, 300.],
          [2, 1500.],
          [3, 1500.]]

# Take a look at the mesh and the resistivity distribution
kw = dict(cMin=100, cMax=1500, logScale=True, cMap='jet',
          xlabel='Distance (m)', ylabel='Depth (m)', 
          label=pg.unit('res'), orientation='vertical')
ax, cb = pg.show(mesh, 
        data=rhomap, 
        showMesh=True,**kw)
ax.set_xlabel(ax.get_xlabel(),fontsize=13)
ax.set_ylabel(ax.get_ylabel(),fontsize=13)
cb.ax.set_ylabel(cb.ax.get_ylabel(), fontsize=13)
fig = ax.figure
# fig.savefig(join('results','layer.png'), dpi=300, bbox_inches='tight')
# # save the mesh to binary file
# mesh.save("mesh.bms") # can be load by pg.load()

# Perform the modelling with the mesh and the measuring scheme itself
# and return a data container with apparent resistivity values,
# geometric factors and estimated data errors specified by the noise setting.
# The noise is also added to the data. Here 1% plus 1µV.
# Note, we force a specific noise seed as we want reproducable results for
# testing purposes.
data = ert.simulate(mesh, scheme=scheme, res=rhomap, noiseLevel=1,
                    noiseAbs=1e-6, 
                    seed=1337) #seed : numpy.random seed for repeatable noise in synthetic experiments 

pg.info(np.linalg.norm(data['err']), np.linalg.norm(data['rhoa']))
pg.info('Simulated data', data)
pg.info('The data contains:', data.dataMap().keys())
pg.info('Simulated rhoa (min/max)', min(data['rhoa']), max(data['rhoa']))
pg.info('Selected data noise %(min/max)', min(data['err'])*100, max(data['err'])*100)

ax, _ = pg.show(data,cMap='jet')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Array type and \nElectrode separation (m)')
fig = ax.figure
# fig.savefig(join('results','data.png'), dpi=300, bbox_inches='tight')

# # save the data for further use
# data.save('simple.dat')

# Plot the eletrode position
fig, ax = plt.subplots()
ax.plot(np.array(pg.x(data)), np.array(pg.z(data)),'kv',label='Electrode')

ax.set_ylim([-10,0.5])
ax.set_xlabel('Distance (m)',fontsize=13)
ax.set_ylabel('Depth (m)',fontsize=13)
ax.legend(loc='right')
ax.set_yticks([-10,-5,0])
ax.set_xticks([0, 10, 20, 30, 40, 50 ,60, 70, 80,90,100,110,120,130])
ax.set_aspect('equal')
fig = ax.figure
# fig.savefig(join('results','electrode.png'), dpi=300, bbox_inches='tight')

# %% Inversion using normal mesh (no prior layer scheme)
plc = mt.createParaMeshPLC(data, paraDepth=30, boundary=0.5)

mesh3 = mt.createMesh(plc,
                      area=10,
                      quality=33)    # Quality mesh generation with no angles smaller than X degrees
ax,_ = pg.show(mesh3,markers=True)

# %% Inversion with the ERTManager
# Creat the ERT Manager
mgr2 = ert.ERTManager(data)
# Run the inversion with the preset data. The Inversion mesh will be created
# with default settings.
mgr2.invert(mesh=mesh3, lam=100, verbose=True)
mgr2.showResultAndFit(cMap='jet')

# %%  Inversion using structural constrain mesh
plc = mt.createParaMeshPLC(data, paraDepth=30, boundary=0.1)
artif1 = mt.createRectangle(start=[37.5, 0], end=[42.5, -10],isClosed=False,marker=2,boundaryMarker=2 )
artif2 = mt.createRectangle(start=[77.5, 0], end=[82.5, -10],isClosed=False,marker=2,boundaryMarker=2)
plc = plc + artif1 + artif2
mesh4 = mt.createMesh(plc,
                      area=10,
                      quality=33)    # Quality mesh generation with no angles smaller than X degrees
ax,_ = pg.show(mesh4,markers=True)
# %% Inversion with the ERTManager
# Creat the ERT Manager
mgr3 = ert.ERTManager(data)
# Run the inversion with the preset data. The Inversion mesh will be created
# with default settings.
mgr3.invert(mesh=mesh4, lam=100, verbose=True)
mgr3.showResultAndFit(cMap='jet')

# %% Creat wet time model
# Geometry definition
# Create geometry definition for the modelling domain. 

left = 0
right = 128
depth = 30

world = mt.createWorld(start=[left, 0], end=[right, -depth])

artif1 = mt.createRectangle(start=[37.5, 0], end=[42.5, -10],marker=2)
artif2 = mt.createRectangle(start=[77.5, 0], end=[82.5, -10],marker=3)
wet1 = mt.createRectangle(start=[0, 0], end=[37.5, -5],marker=4)
wet2 = mt.createRectangle(start=[42.5, 0], end=[77.5, -5],marker=4)
wet3 = mt.createRectangle(start=[82.5, 0], end=[128, -5],marker=4)

geom = world + artif1 + artif2 + wet1 + wet2 + wet3
pg.show(geom, markers=False)
# %%
# Synthetic data generation
# Create a Dipole Dipole ('dd') measuring scheme with 21 electrodes.
scheme = ert.createData(elecs=np.linspace(start=0, stop=128, num=65),
                           schemeName='dd')

# Create a mesh for the finite element modelling with appropriate mesh quality.
mesh = mt.createMesh(geom, 
                     area=1,
                     quality=33)    # Quality mesh generation with no angles smaller than X degrees 
pg.show(mesh,markers=False)

# Create a map to set resistivity values in the appropriate regions
# [[regionNumber, resistivity], [regionNumber, resistivity], [...]
rhomap = [[1, 300.],
          [2, 1500.],
          [3, 1500.],
          [4, 50.]]

# Take a look at the mesh and the resistivity distribution
kw = dict(cMin=50, cMax=1500, logScale=True, cMap='jet',
          xlabel='Distance (m)', ylabel='Depth (m)', 
          label=pg.unit('res'), orientation='vertical')
ax, cb = pg.show(mesh, 
        data=rhomap, 
        showMesh=True,**kw)
ax.set_xlabel(ax.get_xlabel(),fontsize=13)
ax.set_ylabel(ax.get_ylabel(),fontsize=13)
cb.ax.set_ylabel(cb.ax.get_ylabel(), fontsize=13)
fig = ax.figure
# %%
# fig.savefig(join('results','layer.png'), dpi=300, bbox_inches='tight')
# # save the mesh to binary file
# mesh.save("mesh.bms") # can be load by pg.load()

# Perform the modelling with the mesh and the measuring scheme itself
# and return a data container with apparent resistivity values,
# geometric factors and estimated data errors specified by the noise setting.
# The noise is also added to the data. Here 1% plus 1µV.
# Note, we force a specific noise seed as we want reproducable results for
# testing purposes.
data = ert.simulate(mesh, scheme=scheme, res=rhomap, noiseLevel=1,
                    noiseAbs=1e-6, 
                    seed=1337) #seed : numpy.random seed for repeatable noise in synthetic experiments 

pg.info(np.linalg.norm(data['err']), np.linalg.norm(data['rhoa']))
pg.info('Simulated data', data)
pg.info('The data contains:', data.dataMap().keys())
pg.info('Simulated rhoa (min/max)', min(data['rhoa']), max(data['rhoa']))
pg.info('Selected data noise %(min/max)', min(data['err'])*100, max(data['err'])*100)

ax, _ = pg.show(data,cMap='jet')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Array type and \nElectrode separation (m)')
fig = ax.figure
# fig.savefig(join('results','data.png'), dpi=300, bbox_inches='tight')

# # save the data for further use
# data.save('simple.dat')
# Inversion using normal mesh (no prior layer scheme)


# %%
# Creat the ERT Manager
mgr4 = ert.ERTManager(data)
# Run the inversion with the preset data. The Inversion mesh will be created
# with default settings.
mgr4.invert(mesh=mesh3, lam=100, verbose=True)
mgr4.showResultAndFit(cMap='jet')

#%% Creat the ERT Manager
mgr5 = ert.ERTManager(data)
# Run the inversion with the preset data. The Inversion mesh will be created
# with default settings.
mgr5.invert(mesh=mesh4, lam=100, verbose=True)
mgr5.showResultAndFit(cMap='jet')


# %%
# Comparesion of the results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,constrained_layout=True)
mgr2.showResult(ax=ax1,coverage=None,**kw)
mgr3.showResult(ax=ax2,coverage=None,**kw)
mgr4.showResult(ax=ax3,coverage=None,**kw)
mgr5.showResult(ax=ax4,coverage=None,**kw)


# %%
# Comparesion of the results by the residual profile
# Re-interpolate the grid
mesh_x = np.linspace(0,100,100)
mesh_y = np.linspace(-30,0,60)
grid = pg.createGrid(x=mesh_x,y=mesh_y )

# Creat a pg RVector with the length of the cell of mesh and the value of rhomap
rho = pg.Vector(np.array([row[1] for row in rhomap])[mesh.cellMarkers() - 1] )
# Distinguish the region of the mesh and insert the value of rhomap
rho_grid = pg.interpolate(mesh, rho, grid.cellCenters())

