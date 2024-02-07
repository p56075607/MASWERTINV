# %%
# Build a three-layer model for electrical resistivity tomography synthetic test using pygimli package
import matplotlib.pyplot as plt
import numpy as np

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert

# %%
# Geometry definition
# Create geometry definition for the modelling domain. 
# ``worldMarker=True`` indicates the default boundary conditions for the ERT
# dimensions of the world
left = 0
right = 100
depth = 50

world = mt.createWorld(start=[left, 0], end=[right, -depth],
                       layers=[-5, -20], worldMarker=True)
pg.show(world)

# %%
# Synthetic data generation
# Create a Dipole Dipole ('dd') measuring scheme with 21 electrodes.
scheme = ert.createData(elecs=np.linspace(start=0, stop=100, num=21),
                           schemeName='dd')

# Put all electrode (aka sensors) positions into the PLC to enforce mesh
# refinement. Due to experience, its convenient to add further refinement
# nodes in a distance of 10% of electrode spacing to achieve sufficient
# numerical accuracy.
for p in scheme.sensors():
    world.createNode(p)
    world.createNode(p - [0, 0.1])
# %%
# Create a mesh for the finite element modelling with appropriate mesh quality.
mesh = mt.createMesh(world, 
                    #  area=1.0,
                     quality=33)    # Quality mesh generation with no angles smaller than X degrees 
pg.show(mesh)
# %%
# Create a map to set resistivity values in the appropriate regions
# [[regionNumber, resistivity], [regionNumber, resistivity], [...]
rhomap = [[1, 50.],
          [2, 100.],
          [3, 150.]]

# Take a look at the mesh and the resistivity distribution
pg.show(mesh, data=rhomap, label=pg.unit('res'), showMesh=True)
# save the mesh to binary file
mesh.save("mesh.bms"); # can be load by pg.load()
# %%
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

pg.show(data)
# save the data for further use
data.save('simple.dat')

# %%
# Add triangleboundary as inversiondomain
grid = pg.meshtools.appendTriangleBoundary(mesh, marker=1,
                                           xbound=50, ybound=50)
pg.show(grid,markers=True)
# %% Inversion with the ERTManager
# Creat the ERT Manager
mgr = ert.ERTManager(data)
# Run the inversion with the preset data. The Inversion mesh will be created
# with default settings.
inv = mgr.invert(area=1, lam=20, verbose=True)

mgr.showResultAndFit()