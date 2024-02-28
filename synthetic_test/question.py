# Build a three-layer model for electrical resistivity tomography synthetic test using pygimli package
import matplotlib.pyplot as plt
import numpy as np
from os.path import join

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert

# Model setup
c1 = mt.createCircle(pos=(0, 310),radius=200, start=1.5*np.pi, end=1.7*np.pi,isClosed=True, marker = 2, area=1)
slope = mt.createPolygon([[0.0, 80], [0.0, 110], 
                          [c1.node(12).pos()[0], c1.node(12).pos()[1]], 
                          [c1.node(12).pos()[0], 80]],
                          isClosed=True)
geom = slope + c1
mesh = mt.createMesh(geom,area=1, quality=33)
pg.show(mesh, markers=True, showMesh=True)

# Synthetic data generation
electrode_x = np.linspace(start=0, stop=c1.node(12).pos()[0], num=25)
electrode_y = np.linspace(start=110, stop=c1.node(12).pos()[1], num=25)
# Plot the eletrode position
ax, _ = pg.show(slope, markers=False, showMesh=False)
ax.plot(electrode_x, electrode_y,'kv',label='Electrode')

scheme = ert.createData(elecs=np.column_stack((electrode_x, electrode_y)),
                           schemeName='dd')

# Create a map to set resistivity values in the appropriate regions
# [[regionNumber, resistivity], [regionNumber, resist３ivity], [...]
rhomap = [[1, 150.],
          [2, 50.]]

# Forward modelling
data = ert.simulate(mesh, scheme=scheme, res=rhomap, noiseLevel=1,
                    noiseAbs=1e-6, 
                    seed=1337) #seed : numpy.random seed for repeatable noise in synthetic experiments 

# Inversion using structural constrain mesh
c2 = mt.createCircle(pos=(0, 310),radius=200, start=1.53*np.pi, end=1.67*np.pi,isClosed=False, marker = 1)

plc = slope + c2
mesh3 = mt.createMesh(plc,
                      area=1,
                      quality=33)    # Quality mesh generation with no angles smaller than X degrees
ax,_ = pg.show(mesh3)

# Creat the ERT Manager
mgr3 = ert.ERTManager(data)
inv3 = mgr3.invert(mesh=mesh3, lam=100, verbose=True)
mgr3.showResultAndFit(cMap='jet')