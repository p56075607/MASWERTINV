# %%
# Build a three-layer model for electrical resistivity tomography synthetic test using pygimli package
import matplotlib.pyplot as plt
import numpy as np

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
plt.rcParams["font.family"] = 'Times New Roman'#"Microsoft Sans Serif"
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

# Synthetic data generation
# Create a Dipole Dipole ('dd') measuring scheme with 21 electrodes.
scheme = ert.createData(elecs=np.linspace(start=0, stop=100, num=21),
                           schemeName='dd')

# Put all electrode (aka sensors) positions into the PLC to enforce mesh
# refinement. Due to experience, its convenient to add further refinement
# nodes in a distance of 10% of electrode spacing to achieve sufficient
# numerical accuracy.
# for p in scheme.sensors():
#     world.createNode(p)
#     world.createNode(p - [0, 0.1])

# Create a mesh for the finite element modelling with appropriate mesh quality.
mesh = mt.createMesh(world, 
                     area=1,
                     quality=33)    # Quality mesh generation with no angles smaller than X degrees 
pg.show(mesh,markers=True)

# Create a map to set resistivity values in the appropriate regions
# [[regionNumber, resistivity], [regionNumber, resistivity], [...]
rhomap = [[1, 50.],
          [2, 100.],
          [3, 150.]]

# Take a look at the mesh and the resistivity distribution
pg.show(mesh, 
        # data=rhomap, cMap='jet', label=pg.unit('res'), 
        showMesh=True)
# save the mesh to binary file
mesh.save("mesh.bms") # can be load by pg.load()
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
data.save('simple.dat')


# %% Inversion using normal mesh (no prior layer scheme)
world2 = mt.createWorld(start=[left, 0], end=[right, -depth], worldMarker=True)
mesh2 = mt.createMesh(world2, 
                     area=5,
                     quality=33)    # Quality mesh generation with no angles smaller than X degrees 
pg.show(mesh2)
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

# %% Inversion using three-layer based mesh
plc = mt.createParaMeshPLC(data, paraDepth=30, boundary=0.5)
left_edge = 30
right_edge = 70
interface1 = mt.createLine(start=[left_edge, -5 ], end=[right_edge, -5] )
interface2 = mt.createLine(start=[left_edge, -15], end=[right_edge, -15])
plc = interface1 + interface2 + plc
pg.show(plc, markers=True)

mesh3 = mt.createMesh(plc,
                      area=5,
                      quality=33)    # Quality mesh generation with no angles smaller than X degrees
ax,_ = pg.show(mesh3)
ax.set_xlim(0, 100)
ax.set_ylim(-30, 0)
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
ax3.set_xlim(0, 100)

# %%
# Comparesion of the results by the residual profile
# Re-interpolate the grid
mesh_x = np.linspace(0,100,100)
mesh_y = np.linspace(-30,0,60)
grid = pg.createGrid(x=mesh_x,y=mesh_y )

# Creat a pg RVector with the length of the cell of mesh and the value of rhomap
rho = pg.Vector(np.array([row[1] for row in rhomap])[mesh.cellMarkers() - 1] )#pg.Vector(mesh.cellCount())
# Distinguish the region of the mesh and insert the value of rhomap
rho_grid = pg.interpolate(mesh, rho, grid.cellCenters())

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, figsize=(16,8),constrained_layout=True)
ax2.axis('off')
pg.viewer.showMesh(grid,data=rho_grid,ax=ax1,
                    label='Resistivity $\Omega m$',
                    logScale=True,cMap='jet',cMin=50,cMax=150,
                    xlabel="x (m)", ylabel="z (m)",orientation = 'vertical')
ax1.set_title('Original resistivity model profile')

# normal grid 
rho_normal_grid = pg.interpolate(mesh2, mgr2.model, grid.cellCenters())
pg.viewer.showMesh(grid,data=rho_normal_grid,ax=ax3,
                    label='Resistivity $\Omega m$',
                    logScale=True,cMap='jet',cMin=50,cMax=150,
                    xlabel="x (m)", ylabel="z (m)",orientation = 'vertical')
ax3.set_title('Normal mesh inverted resistivity profile')
# plot a triangle
triangle_left = np.array([[left, -depth], [depth, -depth], [left,0], [left, -depth]])
triangle_right = np.array([[right, -depth], [right-depth, -depth], [right,0], [right, depth]])
ax3.add_patch(plt.Polygon(triangle_left,color='white'))
ax3.add_patch(plt.Polygon(triangle_right,color='white'))

# structured constrained grid 
rho_layer_grid = pg.interpolate(mgr3.paraDomain, mgr3.model, grid.cellCenters())
pg.viewer.showMesh(grid,data=rho_layer_grid,ax=ax5,
                            label='Resistivity $\Omega m$',
                            logScale=True,cMap='jet',cMin=50,cMax=150,
                            xlabel="x (m)", ylabel="z (m)",orientation = 'vertical')
ax5.set_title('Structured constrained inverted resistivity profile')
ax5.add_patch(plt.Polygon(triangle_left,color='white'))
ax5.add_patch(plt.Polygon(triangle_right,color='white'))

# Calculate the resistivity relative difference
residual_normal_grid = ((rho_normal_grid - rho_grid)/rho_grid)*100
pg.viewer.showMesh(grid,data=residual_normal_grid,ax=ax4,
                    label='Resistivity residual $\Omega m$',
                #     logScale=True, 
                    cMap='RdBu', 
                    # cMin=-0.3,cMax=0.3,
                    xlabel="x (m)", ylabel="z (m)",orientation = 'vertical')
ax4.set_title('Normal mesh resistivity residual profile')
ax4.add_patch(plt.Polygon(triangle_left,color='white'))
ax4.add_patch(plt.Polygon(triangle_right,color='white'))

residual_layer_grid = ((rho_layer_grid - rho_grid)/rho_grid)*100
pg.viewer.showMesh(grid,data=residual_layer_grid,ax=ax6,
                    label='Resistivity residual $\Omega m$',
                #     logScale=True, 
                    cMap='RdBu', 
                    # cMin=-0.3,cMax=0.3,
                    xlabel="x (m)", ylabel="z (m)",orientation = 'vertical')
ax6.set_title('Layered mesh resistivity residual profile')
ax6.add_patch(plt.Polygon(triangle_left,color='white'))
ax6.add_patch(plt.Polygon(triangle_right,color='white'))
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