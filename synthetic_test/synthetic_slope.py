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
mesh2 = mt.createMesh(slope, 
                     area=5,
                     quality=33)    # Quality mesh generation with no angles smaller than X degrees 
pg.show(mesh2,markers=True)
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
# Comparesion of the results by the residual profile
# Re-interpolate the grid
mesh_x = np.linspace(0,c1.node(12).pos()[0],500)
mesh_y = np.linspace(80,c1.node(12).pos()[1],100)
grid = pg.createGrid(x=mesh_x,y=mesh_y )

# Creat a pg RVector with the length of the cell of mesh and the value of rhomap
rho = pg.Vector(np.array([row[1] for row in rhomap])[mesh.cellMarkers() - 1] )
# Distinguish the region of the mesh and insert the value of rhomap
rho_grid = pg.interpolate(mesh, rho, grid.cellCenters())

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, figsize=(12,8),constrained_layout=True)
ax2.axis('off')
# Subplot 1:Original resistivity model
pg.viewer.showMesh(mesh, rhomap,ax=ax1,
                    label='Resistivity ($\Omega m$)',
                    logScale=True,cMap='jet',cMin=50,cMax=150,
                    xlabel="x (m)", ylabel="z (m)",orientation = 'vertical')
ax1.set_title('Original resistivity model profile',fontweight="bold", size=16)

# Subplot 3:normal grid 
rho_normal_grid = pg.interpolate(mesh2, mgr2.model, grid.cellCenters())
pg.viewer.showMesh(grid,data=rho_normal_grid,ax=ax3,
                    label='Resistivity ($\Omega m$)',
                    logScale=True,cMap='jet',cMin=50,cMax=150,
                    xlabel="x (m)", ylabel="z (m)",orientation = 'vertical')
ax3.plot([0.0, c1.node(12).pos()[0]],[110, c1.node(12).pos()[1]],linewidth=1,color='k')
ax3.set_title('Normal mesh inverted resistivity profile',fontweight="bold", size=16)
# cut inversion domain and turn white outside
white_polygon = np.array([[0, 80], [0.0, 110], [30, 80], [100, 100], 
                          [c1.node(12).pos()[0], c1.node(12).pos()[1]], 
                          [c1.node(12).pos()[0], 80],[0, 80]])
ax3.add_patch(plt.Polygon(white_polygon,color='white'))
ax3.plot(electrode_x, electrode_y, 'ko', markersize=2)

# Subplot 5:structured constrained grid 
rho_layer_grid = pg.interpolate(mgr3.paraDomain, mgr3.model, grid.cellCenters())
pg.viewer.showMesh(grid,data=rho_layer_grid,ax=ax5,
                    label='Resistivity ($\Omega m$)',
                    logScale=True,cMap='jet',cMin=50,cMax=150,
                    xlabel="x (m)", ylabel="z (m)",orientation = 'vertical')
ax5.set_title('Structured constrained inverted resistivity profile',fontweight="bold", size=16)
ax5.plot([0.0, c1.node(12).pos()[0]],[110, c1.node(12).pos()[1]],linewidth=1,color='k')
# cut inversion domain and turn white outside
ax5.add_patch(plt.Polygon(white_polygon,color='white'))
ax5.plot(electrode_x, electrode_y, 'ko', markersize=2)
pg.show(c2,ax=ax5,linewidth=3)

# Calculate the resistivity relative difference
# Subplot 4:Normal mesh resistivity residual
residual_normal_grid = ((rho_normal_grid - rho_grid)/rho_grid)*100
pg.viewer.showMesh(grid,data=residual_normal_grid,ax=ax4,
                    label='Relative resistivity difference (%)',
                #     logScale=True, 
                    cMap='RdBu_r', 
                     cMin=-50,cMax=50,
                    xlabel="x (m)", ylabel="z (m)",orientation = 'vertical')
ax4.set_title('Normal mesh resistivity difference profile',fontweight="bold", size=16)
ax4.add_patch(plt.Polygon(white_polygon,color='white'))
ax4.plot([0.0, c1.node(12).pos()[0]],[110, c1.node(12).pos()[1]],linewidth=1,color='k')
ax4.plot(electrode_x, electrode_y,'ko',markersize=2)

# Subplot 6:Layered mesh resistivity residual
residual_layer_grid = ((rho_layer_grid - rho_grid)/rho_grid)*100
pg.viewer.showMesh(grid,data=residual_layer_grid,ax=ax6,
                    label='Relative resistivity difference (%)',
                #     logScale=True, 
                    cMap='RdBu_r', 
                    cMin=-50,cMax=50,
                    xlabel="x (m)", ylabel="z (m)",orientation = 'vertical',
                    )
ax6.set_title('Structured constrained resistivity difference profile',fontweight="bold", size=16)
ax6.add_patch(plt.Polygon(white_polygon,color='white'))
ax6.plot([0.0, c1.node(12).pos()[0]],[110, c1.node(12).pos()[1]],linewidth=1,color='k')
ax6.plot(electrode_x, electrode_y,'ko',markersize=2)
pg.show(c2,ax=ax6)

# %%
fig.savefig('slope_compare.png', dpi=300, bbox_inches='tight')
