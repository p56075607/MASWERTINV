# %%
# Build a three-layer model for electrical resistivity tomography synthetic test using pygimli package
import matplotlib.pyplot as plt
import numpy as np

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
plt.rcParams["font.family"] = "Book Antiqua"
# %%
# Geometry definition
# Create geometry definition for the modelling domain. 
# ``worldMarker=True`` indicates the default boundary conditions for the ERT
# dimensions of the world
left = 0
right = 100
depth = 30

world = mt.createWorld(start=[left, 0], end=[right, -depth],
                       layers=[-5, -15], 
                       worldMarker=True)
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
                     area=10,
                     quality=33)    # Quality mesh generation with no angles smaller than X degrees 
pg.show(mesh,markers=True)
# %%
# Create a map to set resistivity values in the appropriate regions
# [[regionNumber, resistivity], [regionNumber, resistivity], [...]
rhomap = [[1, 50.],
          [2, 100.],
          [3, 150.]]

# Take a look at the mesh and the resistivity distribution
pg.show(mesh, data=rhomap, cMap='jet', label=pg.unit('res'), showMesh=True)
# save the mesh to binary file
mesh.save("mesh.bms") # can be load by pg.load()
# %%
# Add triangleboundary as inversiondomain
# grid = pg.meshtools.appendTriangleBoundary(mesh, marker=4,
#                                            xbound=50, ybound=50)
# pg.show(grid,markers=True)
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

pg.show(data,cMap='jet')
# save the data for further use
data.save('simple.dat')


# %%
world2 = mt.createWorld(start=[left, 0], end=[right, -depth], worldMarker=True)
mesh2 = mt.createMesh(world2, 
                     area=1,
                     quality=33)    # Quality mesh generation with no angles smaller than X degrees 
# Add triangleboundary as inversiondomain
# grid2 = pg.meshtools.appendTriangleBoundary(mesh2, marker=2,
#                                            xbound=50, ybound=50)
# pg.show(grid2,markers=True)
# %% Inversion with the ERTManager
# Creat the ERT Manager
mgr2 = ert.ERTManager(data)
# Run the inversion with the preset data. The Inversion mesh will be created
# with default settings.
inv2 = mgr2.invert(mesh=mesh2, lam=100, verbose=True)
mgr2.showResultAndFit(cMap='jet')

# %%
# Inversion using three-layer based mesh
world3 = mt.createWorld(start=[left, 0], end=[right, -depth], 
                        # layers=[-5, -15],
                        worldMarker=True)
body = mt.createPolygon([(0,-5),(100,-5),(100,-15),(0,-15)],isClosed=True, marker=2)
geo = world3 + body
mesh3 = mt.createMesh(geo,
                      area=1,
                      quality=33)    # Quality mesh generation with no angles smaller than X degrees
pg.show(mesh3, markers=True)
# %% Inversion with the ERTManager
# Creat the ERT Manager
mgr3 = ert.ERTManager(data)
# Run the inversion with the preset data. The Inversion mesh will be created
# with default settings.
inv3 = mgr3.invert(mesh=mesh3, lam=100, verbose=True)
mgr3.showResultAndFit(cMap='jet')


# %%
# Comparesion of the results
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,7))
pg.show(mesh, rhomap, ax=ax1, hold=True, cMap="jet", logScale=True, label='Resistivity ($\Omega$m)',
        orientation="vertical", cMin=50, cMax=150)
mgr2.showResult(ax=ax2, cMap="jet", cMin=50, cMax=150, orientation="vertical",coverage=None)
mgr3.showResult(ax=ax3, cMap="jet", cMin=50, cMax=150, orientation="vertical",coverage=None)

# %%
# # Inversion using structured grid
# # You can also provide your own mesh (e.g., a structured grid if you like them)
# # Note, that x and y coordinates needs to be in ascending order to ensure that
# # all the cells in the grid have the correct orientation, i.e., all cells need
# # to be numbered counter-clockwise and the boundary normal directions need to
# # point outside.
# yDevide = 1.0 - np.logspace(np.log10(1.0), np.log10(depth),31 )
# xDevide = np.linspace(start=left, stop=right, num=100)
# inversionDomain = pg.createGrid(x=xDevide,
#                                 y=yDevide[::-1],
#                                 marker=2)
# pg.show(inversionDomain)
# # Inversion with custom mesh
# # --------------------------
# # The inversion domain for ERT problems needs a boundary that represents the
# # far regions in the subsurface of the halfspace.
# # Give a cell marker lower than the marker for the inversion region, the lowest
# # cell marker in the mesh will be the inversion boundary region by default.
# grid = pg.meshtools.appendTriangleBoundary(inversionDomain, marker=1,
#                                            xbound=50, ybound=50)
# pg.show(grid, markers=True)
# # %%
# # Creat the ERT Manager
# mgr3 = ert.ERTManager(data)
# inv3 = mgr3.invert(mesh=grid, lam=100, verbose=True)
# mgr3.showResultAndFit(cMap='jet')