# %%
# Build a three-layer model for electrical resistivity tomography synthetic test using pygimli package
import matplotlib.pyplot as plt
import numpy as np

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
plt.rcParams["font.family"] = 'Times New Roman'#"Microsoft Sans Serif"

# %% Model setup
c1 = mt.createCircle(pos=(0, 310),radius=200, start=1.5*np.pi, end=1.7*np.pi,isClosed=True, marker = 2, area=1)
ax,_ = pg.show(c1)


# We start by creating a three-layered slope (The model is taken from the BSc
# thesis of Constanze Reinken conducted at the University of Bonn).

slope = mt.createPolygon([[0.0, 80], [0.0, 110], 
                          [c1.node(12).pos()[0], c1.node(12).pos()[1]], 
                          [c1.node(12).pos()[0], 80]],
                          isClosed=True)

# orig_x = 20
# orig_y = 210
# radius = 90
# theta = np.linspace(1.5*np.pi, 1.75*np.pi, 100)  # Angle values from 0 to 2*pi
# circle_x = orig_x + radius * np.cos(theta)
# circle_y = orig_y + radius * np.sin(theta)

# # concate x_rotated, y_rotated horizontally
# slip_surface = np.column_stack((circle_x, circle_y))

# interface = mt.createPolygon(slip_surface )
geom = slope + c1
# ax, _ = pg.show(geom)

mesh = mt.createMesh(geom,area=5, quality=33)
pg.show(mesh, markers=True, showMesh=True)

# %%
# Synthetic data generation
# Create a Dipole Dipole ('dd') measuring scheme with 25 electrodes.
electrode_x = np.linspace(start=0, stop=c1.node(12).pos()[0], num=25)
electrode_y = np.linspace(start=110, stop=c1.node(12).pos()[1], num=25)
plt.scatter(electrode_x, electrode_y)
# %%
scheme = ert.createData(elecs=np.column_stack((electrode_x, electrode_y)),
                           schemeName='dd')

# %%
# Create a map to set resistivity values in the appropriate regions
# [[regionNumber, resistivity], [regionNumber, resistivity], [...]
rhomap = [[1, 150.],
          [2, 50.]]

# Take a look at the mesh and the resistivity distribution
pg.show(mesh, data=rhomap, cMap='jet', label=pg.unit('res'), showMesh=True)
# save the mesh to binary file
mesh.save("mesh_slope.bms") # can be load by pg.load()
# %%
# Add triangleboundary as inversiondomain
# grid = pg.meshtools.appendTriangleBoundary(mesh, marker=4,
#                                            xbound=50, ybound=50)
# pg.show(grid,markers=True)

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
data.save('slope.dat')


# %% Inversion using normal mesh (no prior layer scheme)
# slope2 = mt.createPolygon([[30, 80], [0.0, 110], 
#                           [c1.node(12).pos()[0], c1.node(12).pos()[1]], 
#                           [c1.node(12).pos()[0]-20, 100]],
#                           isClosed=True)
mesh2 = mt.createMesh(slope, 
                     area=5,
                     quality=33)    # Quality mesh generation with no angles smaller than X degrees 
pg.show(mesh2,markers=True)
# # Add triangleboundary as inversiondomain
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

# %% Inversion using two-layer based mesh
c2 = mt.createCircle(pos=(0, 310),radius=200, start=1.53*np.pi, end=1.67*np.pi,isClosed=False, marker = 1)

plc = slope + c2
mesh3 = mt.createMesh(plc,
                      area=5,
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
