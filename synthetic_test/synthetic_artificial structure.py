# %%
# Build a two-layer model with 2 artificial structures for ERT synthetic test using pygimli package
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from os.path import join

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
from os.path import join
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
rhomap = [[1, 500.],
          [2, 1500.],
          [3, 1500.]]

# Take a look at the mesh and the resistivity distribution
kw = dict(cMin=250, cMax=1500, logScale=True, cMap='jet',
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
plc = mt.createParaMeshPLC(data, paraDepth=30, boundary=0.2)

mesh3 = mt.createMesh(plc,
                      area=1,
                #       quality=33
                      )    # Quality mesh generation with no angles smaller than X degrees
ax,_ = pg.show(mesh3,markers=True)

# %% Inversion with the ERTManager
# Creat the ERT Manager
mgr_dry_normal = ert.ERTManager(data)
# Run the inversion with the preset data. The Inversion mesh will be created
# with default settings.
mgr_dry_normal.invert(mesh=mesh3, lam=100, verbose=True)
mgr_dry_normal.showResultAndFit(cMap='jet')

# %%  Inversion using structural constrain mesh
plc = mt.createParaMeshPLC(data, paraDepth=30, boundary=0.2)
artif1 = mt.createRectangle(start=[37.5, 0], end=[42.5, -10],isClosed=False,marker=2,boundaryMarker=2 )
artif2 = mt.createRectangle(start=[77.5, 0], end=[82.5, -10],isClosed=False,marker=2,boundaryMarker=2)
plc = plc + artif1 + artif2
mesh4 = mt.createMesh(plc,
                      area=1,
                #       quality=33
                      )    # Quality mesh generation with no angles smaller than X degrees
ax,_ = pg.show(mesh4,markers=False)
# %% Inversion with the ERTManager
# Creat the ERT Manager
mgr_dry_struc = ert.ERTManager(data)
# Run the inversion with the preset data. The Inversion mesh will be created
# with default settings.
mgr_dry_struc.invert(mesh=mesh4, lam=100, verbose=True)
mgr_dry_struc.showResultAndFit(cMap='jet')

# %% Creat wet time model
# Geometry definition
# Create geometry definition for the modelling domain. 

left = 0
right = 128
depth = 30

world = mt.createWorld(start=[left, 0], end=[right, -depth])

artif1 = mt.createRectangle(start=[37.5, 0.1], end=[42.5, -10],marker=2)
artif2 = mt.createRectangle(start=[77.5, 0.1], end=[82.5, -10],marker=3)
wet1 = mt.createLine(start=[0,    -5], end=[37.5, -5])
wet2 = mt.createLine(start=[42.5, -5], end=[77.5, -5])
wet3 = mt.createLine(start=[82.5, -5], end=[128,  -5])

geom = world + artif1 + artif2 + wet1 + wet2 + wet3
pg.show(geom, markers=True)
# %%
# Synthetic data generation
# Create a Dipole Dipole ('dd') measuring scheme with 21 electrodes.
scheme = ert.createData(elecs=np.linspace(start=0, stop=128, num=65),
                           schemeName='dd')

# Create a mesh for the finite element modelling with appropriate mesh quality.
mesh2 = mt.createMesh(geom, 
                     area=1,
                     quality=33)    # Quality mesh generation with no angles smaller than X degrees 
pg.show(mesh2,markers=False)

# Create a map to set resistivity values in the appropriate regions
# [[regionNumber, resistivity], [regionNumber, resistivity], [...]
rhomap2 = [[1, 500.],
          [2, 1500.],
          [3, 1500.],
          [0, 250.]]

# Take a look at the mesh and the resistivity distribution
ax, cb = pg.show(mesh2, 
        data=rhomap2, 
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
# and return a data2 container with apparent resistivity values,
# geometric factors and estimated data2 errors specified by the noise setting.
# The noise is also added to the data2. Here 1% plus 1µV.
# Note, we force a specific noise seed as we want reproducable results for
# testing purposes.
data2 = ert.simulate(mesh2, scheme=scheme, res=rhomap2, noiseLevel=1,
                    noiseAbs=1e-6, 
                    seed=1337) #seed : numpy.random seed for repeatable noise in synthetic experiments 

pg.info(np.linalg.norm(data2['err']), np.linalg.norm(data2['rhoa']))
pg.info('Simulated data2', data2)
pg.info('The data2 contains:', data2.dataMap().keys())
pg.info('Simulated rhoa (min/max)', min(data2['rhoa']), max(data2['rhoa']))
pg.info('Selected data2 noise %(min/max)', min(data2['err'])*100, max(data2['err'])*100)

ax, _ = pg.show(data2,cMap='jet')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Array type and \nElectrode separation (m)')
fig = ax.figure
# fig.savefig(join('results','data2.png'), dpi=300, bbox_inches='tight')

# # save the data2 for further use
# data2.save('simple.dat')
# Inversion using normal mesh (no prior layer scheme)


# %%
# Creat the ERT Manager
mgr_wet_normal = ert.ERTManager(data2)
# Run the inversion with the preset data2. The Inversion mesh will be created
# with default settings.
mgr_wet_normal.invert(mesh=mesh3, lam=100, verbose=True)
mgr_wet_normal.showResultAndFit(cMap='jet')

#%% Creat the ERT Manager
mgr_wet_struc = ert.ERTManager(data2)
# Run the inversion with the preset data2. The Inversion mesh will be created
# with default settings.
mgr_wet_struc.invert(mesh=mesh4, lam=100, verbose=True)
mgr_wet_struc.showResultAndFit(cMap='jet')


# %%
# Comparesion of the results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(16,4),constrained_layout=True)
mgr_dry_normal.showResult(ax=ax1,coverage=None,**kw)
mgr_dry_struc.showResult(ax=ax2,coverage=None,**kw)
mgr_wet_normal.showResult(ax=ax3,coverage=None,**kw)
mgr_wet_struc.showResult(ax=ax4,coverage=None,**kw)


# %%
mesh_x = np.linspace(left,right,300)
mesh_y = np.linspace(-depth,0,100)
grid = pg.createGrid(x=mesh_x,y=mesh_y )
# Comparesion of the results by the residual profile
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, 
                                                         figsize=(16,6), 
                                                         constrained_layout=True)
# Subplot 1: Drytime normal mesh
rho_dry_grid = pg.interpolate(mgr_dry_normal.paraDomain, mgr_dry_normal.model, grid.cellCenters())
pg.viewer.showMesh(mgr_dry_normal.paraDomain, mgr_dry_normal.model, ax=ax1,**kw)
ax1.set_title('Dry time normal mesh inverted resistivity profile',fontweight="bold", size=16)
# plot a triangle
triangle_left = np.array([[left, -depth], [depth, -depth], [left,0], [left, -depth]])
triangle_right = np.array([[right, -depth], [right-depth, -depth], [right,0], [right, depth]])
ax1.add_patch(plt.Polygon(triangle_left,color='white'))
ax1.add_patch(plt.Polygon(triangle_right,color='white'))
ax1.plot(pg.x(artif1.nodes()),pg.y(artif1.nodes()),'--k')
ax1.plot(pg.x(artif2.nodes()),pg.y(artif2.nodes()),'--k')
ax1.set_xlim([0, 128])
ax1.set_ylim([-30, 0])
ax1.text(5,-27.5,'RRMS: {:.2f}%'.format(
         mgr_dry_normal.inv.relrms())
            ,fontweight="bold", size=12)

# Subplot 2: Drytime structural mesh
pg.viewer.showMesh(mgr_dry_struc.paraDomain, mgr_dry_struc.model, ax=ax2,**kw)
ax2.set_title('Dry time structural mesh inverted resistivity profile',fontweight="bold", size=16)
ax2.set_xlim([0, 128])
ax2.add_patch(plt.Polygon(triangle_left,color='white'))
ax2.add_patch(plt.Polygon(triangle_right,color='white'))
ax2.text(5,-27.5,'RRMS: {:.2f}%'.format(
         mgr_dry_struc.inv.relrms())
            ,fontweight="bold", size=12)

# Subplot 3: Wettime normal mesh
pg.viewer.showMesh(mgr_wet_normal.paraDomain, mgr_wet_normal.model, ax=ax3,**kw)
ax3.set_title('Wet time normal mesh inverted resistivity profile',fontweight="bold", size=16)
ax3.add_patch(plt.Polygon(triangle_left,color='white'))
ax3.add_patch(plt.Polygon(triangle_right,color='white'))
ax3.plot(pg.x(artif1.nodes()),pg.y(artif1.nodes()),'--k')
ax3.plot(pg.x(artif2.nodes()),pg.y(artif2.nodes()),'--k')
ax3.set_xlim([0, 128])
ax3.set_ylim([-30, 0])
ax3.text(5,-27.5,'RRMS: {:.2f}%'.format(
         mgr_wet_normal.inv.relrms())
            ,fontweight="bold", size=12)

# Subplot 4: Wettime structural mesh
pg.viewer.showMesh(mgr_wet_struc.paraDomain, mgr_wet_struc.model, ax=ax4,**kw)
ax4.set_title('Wet time structural mesh inverted resistivity profile',fontweight="bold", size=16)
ax4.set_xlim([0, 128])
ax4.add_patch(plt.Polygon(triangle_left,color='white'))
ax4.add_patch(plt.Polygon(triangle_right,color='white'))
ax4.text(5,-27.5,'RRMS: {:.2f}%'.format(
         mgr_wet_struc.inv.relrms())
            ,fontweight="bold", size=12)

# Calculate the resistivity relative difference
kw_compare = dict(cMin=-50, cMax=50, cMap='bwr',
                  label='Relative resistivity difference \n(%)',
                  xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical')

# Subplot 5:Normal mesh resistivity dry/wet residual
residual_normal_grid =  ((mgr_wet_normal.model - mgr_dry_normal.model)/mgr_dry_normal.model)*100
pg.viewer.showMesh(mgr_dry_normal.paraDomain,data=residual_normal_grid,ax=ax5, **kw_compare)
ax5.add_patch(plt.Polygon(triangle_left,color='white'))
ax5.add_patch(plt.Polygon(triangle_right,color='white'))
ax5.plot(pg.x(artif1.nodes()),pg.y(artif1.nodes()),'--k')
ax5.plot(pg.x(artif2.nodes()),pg.y(artif2.nodes()),'--k')
ax5.set_xlim([0, 128])
ax5.set_ylim([-30, 0])
ax5.set_title('Wet vs dry normal mesh resistivity difference profile',fontweight="bold", size=16)


# Subplot 6:Structural mesh resistivity dry/wet residual
residual_struct_grid = ((mgr_wet_struc.model - mgr_dry_struc.model)/mgr_dry_struc.model)*100
pg.viewer.showMesh(mgr_dry_struc.paraDomain,data=residual_struct_grid,ax=ax6, **kw_compare)
ax6.set_xlim([0, 128])
ax6.add_patch(plt.Polygon(triangle_left,color='white'))
ax6.add_patch(plt.Polygon(triangle_right,color='white'))
ax6.set_title('Wet vs dry structural mesh resistivity difference profile',fontweight="bold", size=16)

fig.savefig(join('results', 'artificial_wet&dry.png'), dpi=300, bbox_inches='tight')

# %% Plot TRUE model
# Re-interpolate the grid


# Creat a pg RVector with the length of the cell of mesh and the value of rhomap
rho = pg.Vector(np.array([row[1] for row in rhomap])[mesh.cellMarkers() - 1] )
# Distinguish the region of the mesh and insert the value of rhomap
rho_grid = pg.interpolate(mesh, rho, grid.cellCenters())

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(16,4), 
                                                         constrained_layout=True)
# Subplot1: Dry true model
pg.viewer.showMesh(mesh, rhomap, ax=ax1, **kw)
ax1.set_title('Dry true resistivity model',fontweight="bold", size=16)

# Subplot2: Wet true model
rho2 = pg.Vector(np.array([row[1] for row in rhomap2])[mesh2.cellMarkers() - 1] )
rho_wet_grid = pg.interpolate(mesh2, rho2, grid.cellCenters())
pg.viewer.showMesh(mesh2, rhomap2, ax=ax2, **kw)
ax2.set_title('Wet true resistivity model',fontweight="bold", size=16)


residual_dw_grid = ((rho_wet_grid - rho_grid)/rho_grid)*100
# Subplot3: true model residual
pg.viewer.showMesh(grid, residual_dw_grid, ax=ax3, **kw_compare)
ax3.set_title('True resistivity model difference',fontweight="bold", size=16)
ax3.plot(pg.x(artif1.nodes()),pg.y(artif1.nodes()),'--k')
ax3.plot(pg.x(artif2.nodes()),pg.y(artif2.nodes()),'--k')
ax4.axis('off')

fig.savefig(join('results', 'artificial_true.png'), dpi=300, bbox_inches='tight')

# %% Dry mesh compare
dry_normal_grid = pg.interpolate(mgr_dry_normal.paraDomain, mgr_dry_normal.model, grid.cellCenters())

dry_struc_grid = pg.interpolate(mgr_dry_struc.paraDomain, mgr_dry_struc.model, grid.cellCenters())
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, 
                                                         figsize=(22,4), 
                                                         constrained_layout=True)
# Subplot1: Dry true model
pg.viewer.showMesh(mesh, rhomap, ax=ax1, **kw)
ax1.set_title('Dry true resistivity model',fontweight="bold", size=16)

# Subplot 2: Drytime normal mesh
pg.viewer.showMesh(mgr_dry_normal.paraDomain, mgr_dry_normal.model, ax=ax2, **kw)
ax2.add_patch(plt.Polygon(triangle_left,color='white'))
ax2.add_patch(plt.Polygon(triangle_right,color='white'))
ax2.set_title('Dry resistivity model with normal mesh',fontweight="bold", size=16)
ax2.plot(pg.x(artif1.nodes()),pg.y(artif1.nodes()),'--k')
ax2.plot(pg.x(artif2.nodes()),pg.y(artif2.nodes()),'--k')
ax2.set_xlim([0, 128])
ax2.set_ylim([-30, 0])

# Subplot 3: Drytime structural mesh
pg.viewer.showMesh(mgr_dry_struc.paraDomain, mgr_dry_struc.model, ax=ax3, **kw)
ax3.add_patch(plt.Polygon(triangle_left,color='white'))
ax3.add_patch(plt.Polygon(triangle_right,color='white'))
ax3.set_xlim([0, 128])
ax3.set_title('Dry resistivity model with structural mesh',fontweight="bold", size=16)


class StretchOutNormalize(plt.Normalize):
    def __init__(self, vmin=None, vmax=None, low=None, up=None, clip=False):
        self.low = low
        self.up = up
        plt.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.low, self.up, self.vmax], [0, 0.5-1e-9, 0.5+1e-9, 1]
        return np.ma.masked_array(np.interp(value, x, y))

clim = [-50, 50]
midnorm=StretchOutNormalize(vmin=clim[0], vmax=clim[1], low=-5, up=5)

X,Y = np.meshgrid(mesh_x,mesh_y)
dry_normal_diff = ((dry_normal_grid - rho_grid)/rho_grid)*100
dry_normal_diff_pos = pg.interpolate(grid, dry_normal_diff, grid.positions())
dry_normal_mesh = np.reshape(dry_normal_diff_pos,(len(mesh_y),len(mesh_x)))
ax5.contourf(X,Y,dry_normal_mesh,
            levels = 128,
            cmap='bwr',
            norm=midnorm)
ax5.set_title('Wet resistivity model with normal mesh compared to TRUE', fontweight="bold", size=16)
ax5.set_xlabel('Distance (m)')
ax5.set_ylabel('Depth (m)')
ax5.add_patch(plt.Polygon(triangle_left,color='white'))
ax5.add_patch(plt.Polygon(triangle_right,color='white'))
ax5.plot(pg.x(artif1.nodes()),pg.y(artif1.nodes()),'--k')
ax5.plot(pg.x(artif2.nodes()),pg.y(artif2.nodes()),'--k')
ax5.set_xlim([0, 128])
ax5.set_ylim([-30, 0])
ax5.set_aspect('equal')

divider = make_axes_locatable(ax5)
cbaxes = divider.append_axes("right", size="4%", pad=.15)
m = plt.cm.ScalarMappable(cmap=plt.cm.bwr,norm=midnorm)
m.set_array(dry_normal_mesh)
m.set_clim(clim[0],clim[1])
cb = plt.colorbar(m,
                boundaries=np.linspace(clim[0],clim[1], 128),
                cax=cbaxes)
cb_ytick = np.linspace(clim[0],clim[1],5)
cb.ax.set_yticks(cb_ytick)
cb.ax.set_yticklabels(['{:.0f}'.format(x) for x in cb_ytick])
cb.ax.set_ylabel('Relative resistivity difference\n(%)')


dry_struct_diff = ((dry_struc_grid - rho_grid)/rho_grid)*100
dry_sctruc_diff_pos = pg.interpolate(grid, dry_struct_diff, grid.positions())
dry_sctruc_mesh = np.reshape(dry_sctruc_diff_pos,(len(mesh_y),len(mesh_x)))
ax6.contourf(X,Y,dry_sctruc_mesh,
            levels = 128,
            cmap='bwr',
            norm=midnorm)
ax6.set_title('Wet resistivity model with sctructural mesh compared to TRUE', fontweight="bold", size=16)
ax6.set_xlabel('Distance (m)')
ax6.set_ylabel('Depth (m)')
ax6.add_patch(plt.Polygon(triangle_left,color='white'))
ax6.add_patch(plt.Polygon(triangle_right,color='white'))
ax6.plot(pg.x(artif1.nodes()),pg.y(artif1.nodes()),'-k')
ax6.plot(pg.x(artif2.nodes()),pg.y(artif2.nodes()),'-k')
ax6.set_xlim([0, 128])
ax6.set_ylim([-30, 0])
ax6.set_aspect('equal')

divider = make_axes_locatable(ax6)
cbaxes = divider.append_axes("right", size="4%", pad=.15)
m = plt.cm.ScalarMappable(cmap=plt.cm.bwr,norm=midnorm)
m.set_array(dry_sctruc_mesh)
m.set_clim(clim[0],clim[1])
cb = plt.colorbar(m,
                boundaries=np.linspace(clim[0],clim[1], 128),
                cax=cbaxes)
cb_ytick = np.linspace(clim[0],clim[1],5)
cb.ax.set_yticks(cb_ytick)
cb.ax.set_yticklabels(['{:.0f}'.format(x) for x in cb_ytick])
cb.ax.set_ylabel('Relative resistivity difference\n(%)')

pg.viewer.showMesh(grid, ((rho_grid - rho_grid)/rho_grid)*100
                   , ax=ax4, **kw_compare)
ax4.set_title('Ideality', fontweight="bold", size=16)
ax4.set_aspect('equal')
ax4.plot(pg.x(artif1.nodes()),pg.y(artif1.nodes()),'--k')
ax4.plot(pg.x(artif2.nodes()),pg.y(artif2.nodes()),'--k')
ax4.set_xlim([0, 128])
ax4.set_ylim([-30, 0])

fig.savefig(join('results', 'artificial_dry.png'), dpi=300, bbox_inches='tight')

# %% Wet mesh compare
wet_normal_grid = pg.interpolate(mgr_wet_normal.paraDomain, mgr_wet_normal.model, grid.cellCenters())

wet_struc_grid = pg.interpolate(mgr_wet_struc.paraDomain, mgr_wet_struc.model, grid.cellCenters())

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, 
                                                         figsize=(22,4), 
                                                         constrained_layout=True)
# Subplot1: Wet true model
pg.viewer.showMesh(mesh2, rhomap2, ax=ax1, **kw)
ax1.set_title('Wet true resistivity model',fontweight="bold", size=16)

# Subplot 2: wettime normal mesh
pg.viewer.showMesh(mgr_wet_normal.paraDomain, mgr_wet_normal.model, ax=ax2, **kw)
ax2.add_patch(plt.Polygon(triangle_left,color='white'))
ax2.add_patch(plt.Polygon(triangle_right,color='white'))
ax2.set_title('wet resistivity model with normal mesh',fontweight="bold", size=16)
ax2.plot(pg.x(artif1.nodes()),pg.y(artif1.nodes()),'--k')
ax2.plot(pg.x(artif2.nodes()),pg.y(artif2.nodes()),'--k')
ax2.set_xlim([0, 128])
ax2.set_ylim([-30, 0])

# Subplot 3: wettime structural mesh
pg.viewer.showMesh(mgr_wet_struc.paraDomain, mgr_wet_struc.model, ax=ax3, **kw)
ax3.add_patch(plt.Polygon(triangle_left,color='white'))
ax3.add_patch(plt.Polygon(triangle_right,color='white'))
ax3.set_xlim([0, 128])
ax3.set_title('wet resistivity model with structural mesh',fontweight="bold", size=16)


class StretchOutNormalize(plt.Normalize):
    def __init__(self, vmin=None, vmax=None, low=None, up=None, clip=False):
        self.low = low
        self.up = up
        plt.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.low, self.up, self.vmax], [0, 0.5-1e-9, 0.5+1e-9, 1]
        return np.ma.masked_array(np.interp(value, x, y))

clim = [-50, 50]
midnorm=StretchOutNormalize(vmin=clim[0], vmax=clim[1], low=-5, up=5)

X,Y = np.meshgrid(mesh_x,mesh_y)
wet_normal_diff = ((wet_normal_grid - rho_grid)/rho_grid)*100
wet_normal_diff_pos = pg.interpolate(grid, wet_normal_diff, grid.positions())
wet_normal_mesh = np.reshape(wet_normal_diff_pos,(len(mesh_y),len(mesh_x)))
ax5.contourf(X,Y,wet_normal_mesh,
            levels = 128,
            cmap='bwr',
            norm=midnorm)
ax5.set_title('Wet resistivity model with normal mesh compared to TRUE', fontweight="bold", size=16)
ax5.set_xlabel('Distance (m)')
ax5.set_ylabel('Depth (m)')
ax5.add_patch(plt.Polygon(triangle_left,color='white'))
ax5.add_patch(plt.Polygon(triangle_right,color='white'))
ax5.plot(pg.x(artif1.nodes()),pg.y(artif1.nodes()),'--k')
ax5.plot(pg.x(artif2.nodes()),pg.y(artif2.nodes()),'--k')
ax5.set_xlim([0, 128])
ax5.set_ylim([-30, 0])
ax5.set_aspect('equal')

divider = make_axes_locatable(ax5)
cbaxes = divider.append_axes("right", size="4%", pad=.15)
m = plt.cm.ScalarMappable(cmap=plt.cm.bwr,norm=midnorm)
m.set_array(wet_normal_mesh)
m.set_clim(clim[0],clim[1])
cb = plt.colorbar(m,
                boundaries=np.linspace(clim[0],clim[1], 128),
                cax=cbaxes)
cb_ytick = np.linspace(clim[0],clim[1],5)
cb.ax.set_yticks(cb_ytick)
cb.ax.set_yticklabels(['{:.0f}'.format(x) for x in cb_ytick])
cb.ax.set_ylabel('Relative resistivity difference\n(%)')


wet_struct_diff = ((wet_struc_grid - rho_grid)/rho_grid)*100
wet_sctruc_diff_pos = pg.interpolate(grid, wet_struct_diff, grid.positions())
wet_sctruc_mesh = np.reshape(wet_sctruc_diff_pos,(len(mesh_y),len(mesh_x)))
ax6.contourf(X,Y,wet_sctruc_mesh,
            levels = 128,
            cmap='bwr',
            norm=midnorm)
ax6.set_title('Wet resistivity model with sctructural mesh compared to TRUE', fontweight="bold", size=16)
ax6.set_xlabel('Distance (m)')
ax6.set_ylabel('Depth (m)')
ax6.add_patch(plt.Polygon(triangle_left,color='white'))
ax6.add_patch(plt.Polygon(triangle_right,color='white'))
ax6.plot(pg.x(artif1.nodes()),pg.y(artif1.nodes()),'-k')
ax6.plot(pg.x(artif2.nodes()),pg.y(artif2.nodes()),'-k')
ax6.set_xlim([0, 128])
ax6.set_ylim([-30, 0])
ax6.set_aspect('equal')

divider = make_axes_locatable(ax6)
cbaxes = divider.append_axes("right", size="4%", pad=.15)
m = plt.cm.ScalarMappable(cmap=plt.cm.bwr,norm=midnorm)
m.set_array(wet_sctruc_mesh)
m.set_clim(clim[0],clim[1])
cb = plt.colorbar(m,
                boundaries=np.linspace(clim[0],clim[1], 128),
                cax=cbaxes)
cb_ytick = np.linspace(clim[0],clim[1],5)
cb.ax.set_yticks(cb_ytick)
cb.ax.set_yticklabels(['{:.0f}'.format(x) for x in cb_ytick])
cb.ax.set_ylabel('Relative resistivity difference\n(%)')

pg.viewer.showMesh(grid, ((rho_grid - rho_grid)/rho_grid)*100
                   , ax=ax4, **kw_compare)
ax4.set_title('Ideality', fontweight="bold", size=16)
ax4.set_aspect('equal')
ax4.plot(pg.x(artif1.nodes()),pg.y(artif1.nodes()),'--k')
ax4.plot(pg.x(artif2.nodes()),pg.y(artif2.nodes()),'--k')
ax4.set_xlim([0, 128])
ax4.set_ylim([-30, 0])

fig.savefig(join('results', 'artificial_wet.png'), dpi=300, bbox_inches='tight')