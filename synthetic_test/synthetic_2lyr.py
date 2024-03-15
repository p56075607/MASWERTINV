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
# Geometry definition
# Create geometry definition for the modelling domain. 
# ``worldMarker=True`` indicates the default boundary conditions for the ERT
# dimensions of the world
left = 0
right = 100
depth = 30

world = mt.createWorld(start=[left, 0], end=[right, -depth],
                       layers=[-5], 
                       worldMarker=True)
block = mt.createRectangle(start=[30, -10], end=[70, -20], marker=3)
geom = block + world
pg.show(geom,markers=True)
# %%
# Synthetic data generation
# Create a Dipole Dipole ('dd') measuring scheme with 21 electrodes.
scheme = ert.createData(elecs=np.linspace(start=0, stop=100, num=51),
                           schemeName='dd')

# Put all electrode (aka sensors) positions into the PLC to enforce mesh
# refinement. Due to experience, its convenient to add further refinement
# nodes in a distance of 10% of electrode spacing to achieve sufficient
# numerical accuracy.
# for p in scheme.sensors():
#     world.createNode(p)
#     world.createNode(p - [0, 0.1])

# Create a mesh for the finite element modelling with appropriate mesh quality.
mesh = mt.createMesh(geom, 
                     area=1,
                     quality=33)    # Quality mesh generation with no angles smaller than X degrees 
pg.show(mesh,markers=True)

# Create a map to set resistivity values in the appropriate regions
# [[regionNumber, resistivity], [regionNumber, resistivity], [...]
rhomap = [[1, 100.],
          [2, 500.],
          [3, 1500.]]

# Take a look at the mesh and the resistivity distribution
kw = dict(cMin=100, cMax=1500, logScale=True, cMap='jet',
          xlabel='Distance (m)', ylabel='Depth (m)', 
          label=pg.unit('res'), orientation='vertical')
ax,_ = pg.show(mesh, 
        data=rhomap, **kw, showMesh=True) 
ax.set_xlabel('Distance (m)',fontsize=13)
ax.set_ylabel('Depth (m)',fontsize=13)
fig = ax.figure
# fig.savefig(join('results','layer.png'), dpi=300, bbox_inches='tight')
# save the mesh to binary file
mesh.save("mesh.bms") # can be load by pg.load()
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

ax, _ = pg.show(data,cMap='jet')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Array type and \nElectrode separation (m)')
fig = ax.figure
# fig.savefig(join('results','data.png'), dpi=300, bbox_inches='tight')

# save the data for further use
data.save('simple.dat')

# %%
# Plot the eletrode position
fig, ax = plt.subplots()
ax.plot(np.array(pg.x(data)), np.array(pg.z(data)),'kv',label='Electrode')

ax.set_ylim([-10,0.5])
ax.set_xlabel('Distance (m)',fontsize=13)
ax.set_ylabel('Depth (m)',fontsize=13)
ax.legend(loc='right')
ax.set_yticks([-10,-5,0])
ax.set_xticks([0, 10, 20, 30, 40, 50 ,60, 70, 80,90,100])
ax.set_aspect('equal')
fig = ax.figure
# fig.savefig(join('results','electrode.png'), dpi=300, bbox_inches='tight')

# %% Inversion using normal mesh (no prior layer scheme)
world2 = mt.createWorld(start=[left, 0], end=[right, -depth], worldMarker=True)
mesh2 = mt.createMesh(world2, 
                     area=0.5,
                     quality=33)    # Quality mesh generation with no angles smaller than X degrees 
ax,_ = pg.show(mesh2)
ax.set_xlabel('Distance (m)',fontsize=13)
ax.set_ylabel('Depth (m)',fontsize=13)
fig = ax.figure
# fig.savefig(join('results','normal_mesh.png'), dpi=300, bbox_inches='tight')

# %% Inversion with the ERTManager
# Creat the ERT Manager
mgr2 = ert.ERTManager(data)
# Run the inversion with the preset data. The Inversion mesh will be created
# with default settings.
inv2 = mgr2.invert(mesh=mesh2, lam=100, verbose=True)
mgr2.showResultAndFit(cMap='jet')

# %% Inversion using three-layer based mesh
plc = mt.createParaMeshPLC(data, paraDepth=30, boundary=0.5)
left_edge = left
right_edge = right
interface2 = mt.createLine(start=[left_edge, -5], end=[right_edge, -5])
plc = interface2 + plc
pg.show(plc, markers=True)
# %%
mesh3 = mt.createMesh(plc,
                      area=0.5,
                      quality=33)    # Quality mesh generation with no angles smaller than X degrees
ax,_ = pg.show(mesh3)
ax.set_xlim(0, 100)
ax.set_ylim(-30, 0)
ax.set_xlabel('Distance (m)',fontsize=13)
ax.set_ylabel('Depth (m)',fontsize=13)
fig = ax.figure
# fig.savefig(join('results','layered_mesh.png'), dpi=300, bbox_inches='tight')
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
pg.show(mesh, rhomap, ax=ax1, **kw)
mgr2.showResult(ax=ax2, **kw)
mgr3.showResult(ax=ax3, **kw)
ax3.set_xlim(0, 100)

# %%
# Comparesion of the results by the residual profile
# Re-interpolate the grid
mesh_x = np.linspace(0,100,300)
mesh_y = np.linspace(-30,0,180)
grid = pg.createGrid(x=mesh_x,y=mesh_y )

# Creat a pg RVector with the length of the cell of mesh and the value of rhomap
rho = pg.Vector(np.array([row[1] for row in rhomap])[mesh.cellMarkers() - 1] )
# Distinguish the region of the mesh and insert the value of rhomap
rho_grid = pg.interpolate(mesh, rho, grid.cellCenters())

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, figsize=(16,8),constrained_layout=True)
ax2.axis('off')
# Subplot 1:Original resistivity model
pg.viewer.showMesh(mesh, rhomap,ax=ax1, **kw)
ax1.set_title('Original resistivity model profile',fontweight="bold", size=16)

def plot_resistivity(ax, mesh, mgr, data, title, **kw):
    pg.viewer.showMesh(mesh, mgr.model,#grid,data=rho_normal_grid,
                    ax=ax, **kw)
    # plot a triangle
    triangle_left = np.array([[left, -depth], [depth, -depth], [left,0], [left, -depth]])
    triangle_right = np.array([[right, -depth], [right-depth, -depth], [right,0], [right, depth]])
    ax.add_patch(plt.Polygon(triangle_left,color='white'))
    ax.add_patch(plt.Polygon(triangle_right,color='white'))
    ax.plot(np.array(pg.x(data)), np.array(pg.z(data)),'ko')
    ax.set_title(title,fontweight="bold", size=16)
    ax.plot(pg.x(interface2.nodes()),pg.y(interface2.nodes()),'--k')
    ax.set_ylim(-30, 0)
    ax.set_xlim(0,100)
    ax.text(5,-27.5,'RRMS: {:.2f}%'.format(
            mgr.inv.relrms())
                ,fontweight="bold", size=12)

# Subplot 3:normal grid 
rho_normal_grid = pg.interpolate(mesh2, mgr2.model, grid.cellCenters())
plot_resistivity(ax=ax3, mesh=mesh2, mgr=mgr2, data=data, title='Normal mesh inverted resistivity profile', **kw)
# Subplot 5:structured constrained grid 
rho_layer_grid = pg.interpolate(mgr3.paraDomain, mgr3.model, grid.cellCenters())
plot_resistivity(ax=ax5, mesh=mgr3.paraDomain, mgr=mgr3, data=data, title='Structured constrained inverted resistivity profile', **kw)
# Plot the residual profile
# Calculate the resistivity relative difference
kw_compare = dict(cMin=-50, cMax=50, cMap='bwr',
                  label='Relative resistivity difference \n(%)',
                  xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical')
# Subplot 4:Normal mesh resistivity residual

def plot_residual(ax, grid, data, title, **kw_compare):
    pg.viewer.showMesh(grid,data,ax=ax, **kw_compare)
    ax.set_title(title,fontweight="bold", size=16)
    # plot a triangle
    triangle_left = np.array([[left, -depth], [depth, -depth], [left,0], [left, -depth]])
    triangle_right = np.array([[right, -depth], [right-depth, -depth], [right,0], [right, depth]])
    ax.add_patch(plt.Polygon(triangle_left,color='white'))
    ax.add_patch(plt.Polygon(triangle_right,color='white'))
    ax.plot(pg.x(interface2.nodes()),pg.y(interface2.nodes()),'--k')
    ax.set_ylim(-30, 0)
    ax.set_xlim(0,100)

residual_normal_grid = ((rho_normal_grid - rho_grid)/rho_grid)*100
plot_residual(ax=ax4, grid=grid, data=residual_normal_grid, title='Normal mesh resistivity difference profile', **kw_compare)
# Subplot 6:Layered mesh resistivity residual
residual_layer_grid = ((rho_layer_grid - rho_grid)/rho_grid)*100
plot_residual(ax=ax6, grid=grid, data=residual_layer_grid, title='Structured constrained resistivity difference profile', **kw_compare)

# fig.savefig(join('results','synthetic_compare.png'), dpi=300, bbox_inches='tight', transparent=True)
# %%
# # Plot profile using contour
# class StretchOutNormalize(plt.Normalize):
#     def __init__(self, vmin=None, vmax=None, low=None, up=None, clip=False):
#         self.low = low
#         self.up = up
#         plt.Normalize.__init__(self, vmin, vmax, clip)

#     def __call__(self, value, clip=None):
#         x, y = [self.vmin, self.low, self.up, self.vmax], [0, 0.5-1e-9, 0.5+1e-9, 1]
#         return np.ma.masked_array(np.interp(value, x, y))

# clim = [-50, 50]
# midnorm=StretchOutNormalize(vmin=clim[0], vmax=clim[1], low=-5, up=5)

# X,Y = np.meshgrid(mesh_x,mesh_y)
# dry_normal_diff = ((dry_normal_grid - rho_grid)/rho_grid)*100
# dry_normal_diff_pos = pg.interpolate(grid, dry_normal_diff, grid.positions())
# dry_normal_mesh = np.reshape(dry_normal_diff_pos,(len(mesh_y),len(mesh_x)))
# ax5.contourf(X,Y,dry_normal_mesh,
#             levels = 128,
#             cmap='bwr',
#             norm=midnorm)
# ax5.set_title('Wet resistivity model with normal mesh compared to TRUE', fontweight="bold", size=16)
# ax5.set_xlabel('Distance (m)')
# ax5.set_ylabel('Depth (m)')
# ax5.add_patch(plt.Polygon(triangle_left,color='white'))
# ax5.add_patch(plt.Polygon(triangle_right,color='white'))
# ax5.plot(pg.x(artif1.nodes()),pg.y(artif1.nodes()),'--k')
# ax5.plot(pg.x(artif2.nodes()),pg.y(artif2.nodes()),'--k')
# ax5.set_xlim([0, 128])
# ax5.set_ylim([-30, 0])
# ax5.set_aspect('equal')

# divider = make_axes_locatable(ax5)
# cbaxes = divider.append_axes("right", size="4%", pad=.15)
# m = plt.cm.ScalarMappable(cmap=plt.cm.bwr,norm=midnorm)
# m.set_array(dry_normal_mesh)
# m.set_clim(clim[0],clim[1])
# cb = plt.colorbar(m,
#                 boundaries=np.linspace(clim[0],clim[1], 128),
#                 cax=cbaxes)
# cb_ytick = np.linspace(clim[0],clim[1],5)
# cb.ax.set_yticks(cb_ytick)
# cb.ax.set_yticklabels(['{:.0f}'.format(x) for x in cb_ytick])
# cb.ax.set_ylabel('Relative resistivity difference\n(%)')