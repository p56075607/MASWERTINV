# %%
# Build a two-layer slope slip model for electrical resistivity tomography synthetic test using pygimli package
import matplotlib.pyplot as plt
import numpy as np
from os.path import join

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert

c1 = mt.createCircle(pos=(0, 310),radius=200, start=1.5*np.pi, end=1.7*np.pi,
                     isClosed=True, marker = 3, area=1)

electrode_x = np.linspace(start=0, stop=c1.node(12).pos()[0], num=25)
electrode_y = np.linspace(start=110, stop=c1.node(12).pos()[1], num=25)

scheme = ert.createData(elecs=np.column_stack((electrode_x, electrode_y)),
                           schemeName='dd')

# %%
# Use createParaMeshPLC 
plc = mt.createParaMeshPLC(scheme, paraDepth=30,xbound=100,ybound=100
                        #    , boundary=1
                           )
c3 = mt.createCircle(pos=(0, 310),radius=200, start=1.5*np.pi, end=1.7*np.pi,
                     isClosed=False, marker = 3, area=1)
plc = plc + c3
# plc.regionMarker(2).setPos([60, 125])
# plc.regionMarker(1).setPos([60, 100])
meshI = mt.createMesh(plc, quality=33.5)
pg.show(meshI,markers=True)
# %%
rhomap = [[0, 50.],
          [1, 150.],
          [2, 150.]]

# Forward modelling
data_plc2 = ert.simulate(meshI, scheme=scheme, res=rhomap, noiseLevel=1,
                    noiseAbs=1e-6, 
                    seed=1337)
ert.showData(data_plc2)
# %%
surface = mt.createLine(start=[0.0, 110], end=[c1.node(12).pos()[0], c1.node(12).pos()[1]],boundaryMarker=-1,marker=2)
slope = mt.createPolygon([[0.0, 110],[0.0, 80], [c1.node(12).pos()[0], 80],
                          [c1.node(12).pos()[0], c1.node(12).pos()[1]]],
                          isClosed=False, marker = 2,boundaryMarker=1)
c2 = mt.createCircle(pos=(0, 310),radius=200, start=1.5*np.pi, end=1.7*np.pi,isClosed=False, 
                     boundaryMarker=2, marker = 2)
world = surface + slope + c2
world.addRegionMarker(pos=[60, 100],marker=2)

mesh_world = mt.createMesh(world, area=2)
pg.show(mesh_world,markers=True)
# %%
appended_world = mt.appendTriangleBoundary(mesh_world,xbound=100,ybound=100,marker=1)#, quality=33)
rhomap = [[0, 50.],
          [1, 150.],
          [2, 150.]]
# pg.show(appended_world,rhomap,markers=False,cMap='jet')
pg.show(appended_world,showMesh=True)
# %%
# Forward modelling
data = ert.simulate(appended_world, scheme=scheme, res=rhomap, noiseLevel=1,
                    noiseAbs=1e-6, 
                    seed=1337)

# %%
geom = surface + slope
geom.addRegionMarker(pos=[60, 100],marker=2)
mesh3 = mt.createMesh(geom, area=2,marker=2)

appended_mesh = mt.appendTriangleBoundary(mesh3,xbound=100,ybound=100,marker=1)#, quality=33)

ax,_ = pg.show(appended_mesh,markers=True)
# %%
# Creat the ERT Manager
mgr3 = ert.ERTManager(data)
inv3 = mgr3.invert(mesh=appended_mesh, lam=100, verbose=True)
mgr3.showResultAndFit(cMap='jet')
