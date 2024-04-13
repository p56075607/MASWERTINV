# %%
# Build a two-layer slope slip model for electrical resistivity tomography synthetic test using pygimli package
import matplotlib.pyplot as plt
import numpy as np
from os.path import join

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
plt.rcParams['font.family'] = 'Times New Roman'#'Microsoft Sans Serif'
from pygimli.frameworks import PriorModelling
from pygimli.viewer.mpl import draw1DColumn

# %% Model setup
c1 = mt.createCircle(pos=(0, 310),radius=200, start=1.5*np.pi, end=1.7*np.pi,isClosed=True, marker = 3, area=1)
ax,_ = pg.show(c1)

# We start by creating a three-layered slope (The model is taken from the BSc
# thesis of Constanze Reinken conducted at the University of Bonn).

slope = mt.createPolygon([[0.0, 80], [0.0, 110], 
                          [c1.node(12).pos()[0], c1.node(12).pos()[1]], 
                          [c1.node(12).pos()[0], 80]],
                          isClosed=True, marker = 2,boundaryMarker=-1)
geometry = c1 + slope
mesh = mt.createMesh(geometry, quality=33)
pg.show(mesh,markers=True)
#%%
# Synthetic data generation
# Create a Dipole Dipole ('dd') measuring scheme with 25 electrodes.
electrode_x = np.linspace(start=0, stop=c1.node(12).pos()[0], num=25)
electrode_y = np.linspace(start=110, stop=c1.node(12).pos()[1], num=25)
# Plot the eletrode position
ax, _ = pg.show(slope, markers=False, showMesh=False)
ax.plot(electrode_x, electrode_y,'kv',label='Electrode')

ax.set_xlim([0-3,c1.node(12).pos()[0]+3])
ax.set_ylim([80,c1.node(12).pos()[1]+3])
ax.set_xlabel('Distance (m)',fontsize=13)
ax.set_ylabel('Depth (m)',fontsize=13)
ax.legend(loc='right')
ax.set_aspect('equal')
fig = ax.figure
# fig.savefig(join('results','slope_electrode.png'), dpi=300, bbox_inches='tight')

scheme = ert.createData(elecs=np.column_stack((electrode_x, electrode_y)),
                           schemeName='dd')
# Create a mesh for the finite element modelling with appropriate mesh quality.
plc = mt.createParaMeshPLC(scheme, paraDepth=30, boundary=1)
c3 = mt.createCircle(pos=(0, 310),radius=200, start=1.5*np.pi, end=1.7*np.pi,isClosed=False, marker = 3, area=1)
plc = plc + c3
plc.regionMarker(2).setPos([60, 125])
plc.regionMarker(1).setPos([60, 100])
meshI = mt.createMesh(plc, quality=33.5)
pg.show(meshI,markers=True)
# %%
# Create a map to set resistivity values in the appropriate regions
# [[regionNumber, resistivity], [regionNumber, resistivity], [...]
rhomap = [[0, 150.],
          [1, 150.],
          [2, 50.]]

# Take a look at the mesh and the resistivity distribution
kw = dict(cMin=50, cMax=150, logScale=True, cMap='jet',
          xlabel='Distance (m)', ylabel='Depth (m)', 
          label=pg.unit('res'), orientation='vertical')
ax, cb = pg.show(meshI, 
        data=rhomap, 
        showMesh=True,**kw)
ax.set_xlabel(ax.get_xlabel(),fontsize=13)
ax.set_ylabel(ax.get_ylabel(),fontsize=13)
cb.ax.set_ylabel(cb.ax.get_ylabel(), fontsize=13)
fig = ax.figure
# fig.savefig(join('results','slope.png'), dpi=300, bbox_inches='tight')

# %%
# Perform the modelling with the mesh and the measuring scheme itself
# and return a data container with apparent resistivity values,
# geometric factors and estimated data errors specified by the noise setting.
# The noise is also added to the data. Here 1% plus 1µV.
# Note, we force a specific noise seed as we want reproducable results for
# testing purposes.
data = ert.simulate(meshI, scheme=scheme, res=rhomap, noiseLevel=1,
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
# fig.savefig(join('results','slope_data.png'), dpi=300, bbox_inches='tight',transparent=True)
# # save the data for further use
# data.save('slope.dat')


# %% Inversion using normal mesh (no prior layer scheme)
surface = mt.createLine(start=[0.0, 110], end=[c1.node(12).pos()[0], c1.node(12).pos()[1]],boundaryMarker=-1,marker=2)
slope = mt.createPolygon([[0.0, 110],[0.0, 80], [c1.node(12).pos()[0], 80],
                          [c1.node(12).pos()[0], c1.node(12).pos()[1]]],
                          isClosed=False, marker = 2,boundaryMarker=1)
world = surface + slope
world.addRegionMarker(pos=[60, 100],marker=2)
# world.regionMarker(2).setPos([60, 100])
# ax,_ = pg.show(world,markers=True,showMesh=True)

mesh_world = mt.createMesh(world, area=1)

mesh2 = mt.appendTriangleBoundary(mesh_world,xbound=100,ybound=100,marker=1)#, quality=33)
ax,_ = pg.show(mesh_world,markers=False,showMesh=True)

ax.set_xlabel('Distance (m)',fontsize=13)
ax.set_ylabel('Depth (m)',fontsize=13)
fig = ax.figure
# fig.savefig(join('results','slope_normal_mesh.png'), dpi=300, bbox_inches='tight',transparent=True)
# %% Inversion with the ERTManager
# Creat the ERT Manager
mgr2 = ert.ERTManager(data)
# Run the inversion with the preset data. The Inversion mesh will be created
# with default settings.
inv2 = mgr2.invert(mesh=mesh2, lam=100, verbose=True)
mgr2.showResultAndFit(cMap='jet')

# %% Inversion using two-layer based mesh
surface = mt.createLine(start=[0.0, 110], end=[c1.node(12).pos()[0], c1.node(12).pos()[1]],boundaryMarker=-1,marker=2)
slope = mt.createPolygon([[0.0, 110],[0.0, 80], [c1.node(12).pos()[0], 80],
                          [c1.node(12).pos()[0], c1.node(12).pos()[1]]],
                          isClosed=False, marker = 2,boundaryMarker=1)
c2 = mt.createCircle(pos=(0, 310),radius=200, start=1.53*np.pi, end=1.67*np.pi,isClosed=False, 
                     boundaryMarker=2, marker = 2)
c4 = mt.createCircle(pos=(0, 312),radius=200, start=1.53*np.pi, end=1.67*np.pi,isClosed=False, 
                     boundaryMarker=2, marker = 2)
c5 = mt.createCircle(pos=(0, 308),radius=200, start=1.53*np.pi, end=1.67*np.pi,isClosed=False, 
                     boundaryMarker=2, marker = 2)
world = surface + slope + c2+c4+c5
world.addRegionMarker(pos=[60, 100],marker=2)
# world.regionMarker(2).setPos([60, 100])
# ax,_ = pg.show(world,markers=True,showMesh=True)

mesh_world = mt.createMesh(world, area=1)

appended_world = mt.appendTriangleBoundary(mesh_world,xbound=100,ybound=100,marker=1)#, quality=33)
ax,_ = pg.show(mesh_world,markers=False,showMesh=True)
ax.set_xlabel('Distance (m)',fontsize=13)
ax.set_ylabel('Depth (m)',fontsize=13)
fig = ax.figure
# fig.savefig(join('results','slope_layered_mesh.png'), dpi=300, bbox_inches='tight',transparent=True)

# %%
layer_depth = np.linspace(308, 312, 3)

mgrs = []
for n, depth_i in enumerate(layer_depth):
    surface = mt.createLine(start=[0.0, 110], end=[c1.node(12).pos()[0], c1.node(12).pos()[1]],boundaryMarker=-1,marker=2)
    slope = mt.createPolygon([[0.0, 110],[0.0, 80], [c1.node(12).pos()[0], 80],
                        [c1.node(12).pos()[0], c1.node(12).pos()[1]]],
                        isClosed=False, marker = 2,boundaryMarker=1)
    c2 = mt.createCircle(pos=(0, depth_i),radius=200, start=1.53*np.pi, end=1.67*np.pi,isClosed=False, 
                     boundaryMarker=2, marker = 2)
    world = surface + slope + c2
    world.addRegionMarker(pos=[60, 100],marker=2)
    mesh_world = mt.createMesh(world, area=1)
    appended_world = mt.appendTriangleBoundary(mesh_world,xbound=100,ybound=100,marker=1)
    # Creat the ERT Manager
    mgr3 = ert.ERTManager(data)
    # Run the inversion with the preset data. The Inversion mesh will be created
    # with default settings.
    inv3 = mgr3.invert(mesh=appended_world, lam=100, verbose=True)
    mgrs.append(mgr3)
# %% Plot the results by resistivity and the residual_struc profile  
white_polygon = np.array([[0, 80], [0.0, 110], [30, 80], [100, 100], 
                        [c1.node(12).pos()[0], c1.node(12).pos()[1]], 
                        [c1.node(12).pos()[0], 80],[0, 80]])
# Calculate the resistivity relative difference
kw_compare = dict(cMin=-50, cMax=50, cMap='bwr',
                  label='Relative resistivity difference (%)',
                  xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical')
# Comparesion of the results by the residual_struc profile
# Re-interpolate the grid
mesh_x = np.linspace(0,c1.node(12).pos()[0],500)
mesh_y = np.linspace(80,c1.node(12).pos()[1],100)
grid = pg.createGrid(x=mesh_x,y=mesh_y )
# Creat a pg RVector with the length of the cell of mesh and the value of rhomap
rho = pg.Vector(np.array([row[1] for row in rhomap[1:3]])[mesh.cellMarkers() - 2] )
# Distinguish the region of the mesh and insert the value of rhomap
rho_grid = pg.interpolate(mesh, rho, grid.cellCenters())

fig = plt.figure(figsize=(20, 15),constrained_layout=True)
for i, depth_i in enumerate(layer_depth):
        c2 = mt.createCircle(pos=(0, depth_i),radius=200, start=1.53*np.pi, end=1.67*np.pi,isClosed=False, 
                boundaryMarker=2, marker = 2)
        ax = plt.subplot(3, 2, 2*i+1)
        pg.show(mgrs[i].paraDomain, mgrs[i].model, ax=ax, **kw)
        ax.set_xlabel(ax.get_xlabel(),fontsize=13)
        ax.set_ylabel(ax.get_ylabel(),fontsize=13)
        ax.set_title('Inverted resistivity profile depth=ori+({:.2f})m'.format((i-1)*2),fontweight='bold', size=16)
        cb.ax.set_ylabel(cb.ax.get_ylabel(), fontsize=13)
        ax.add_patch(plt.Polygon(white_polygon,color='white'))
        ax.plot(electrode_x, electrode_y, 'ko', markersize=2)
        ax.plot(pg.x(c2.nodes()),pg.y(c2.nodes()),'-k')
        ax.text(60,85,'RRMS: {:.2f}%, $\chi^2$: {:.2f}'.format(
                mgrs[i].inv.relrms(), mgrs[i].inv.chi2())
                ,fontweight='bold', size=16)
        ax.set_xlim(0,c1.node(12).pos()[0])
        ax.set_ylim(80,c1.node(12).pos()[1])
        
        # Compare profiles
        rho_constrain = pg.interpolate(mgrs[i].paraDomain, mgrs[i].model, grid.cellCenters())
        ax = plt.subplot(3, 2, 2*i+2)
        residual_struc = ((rho_constrain-rho_grid)/rho_grid)*100
        pg.show(grid, residual_struc, ax=ax, **kw_compare)
        ax.set_xlabel(ax.get_xlabel(),fontsize=13)
        ax.set_ylabel(ax.get_ylabel(),fontsize=13)
        ax.set_title('Resistivity difference profile depth=ori+({:.2f})m'.format((i-1)*2),fontweight='bold', size=16)
        cb.ax.set_ylabel(cb.ax.get_ylabel(), fontsize=13)
        ax.add_patch(plt.Polygon(white_polygon,color='white'))
        ax.plot(electrode_x, electrode_y, 'ko', markersize=2)
        ax.plot(pg.x(c2.nodes()),pg.y(c2.nodes()),'-k')
        ax.text(60,85,'Avg. difference: {:.2f}%'.format(
            np.nansum(abs(residual_struc))/len(residual_struc))
            ,fontweight="bold", size=16)  


fig.savefig(join('results','slope_synthetic_depth.png'), dpi=300, bbox_inches='tight', transparent=True)

# %%
# Plot the results of normal mesh
fig = plt.figure(figsize=(16, 4),constrained_layout=True)
ax = plt.subplot(1, 2, 1)
pg.show(mgr2.paraDomain, mgr2.model, ax=ax, **kw)
ax.set_xlabel(ax.get_xlabel(),fontsize=13)
ax.set_ylabel(ax.get_ylabel(),fontsize=13)
ax.set_title('Inverted resistivity profile of normal mesh',fontweight='bold', size=16)
cb.ax.set_ylabel(cb.ax.get_ylabel(), fontsize=13)
ax.add_patch(plt.Polygon(white_polygon,color='white'))
ax.plot(electrode_x, electrode_y, 'ko', markersize=2)
ax.plot(pg.x(c2.nodes()),pg.y(c2.nodes()),'--k')
ax.text(60,85,'RRMS: {:.2f}%, $\chi^2$: {:.2f}'.format(
        mgr2.inv.relrms(), mgr2.inv.chi2())
        ,fontweight='bold', size=16)
ax.set_xlim(0,c1.node(12).pos()[0])
ax.set_ylim(80,c1.node(12).pos()[1])
# Compare profiles
ax = plt.subplot(1, 2, 2)
rho_constrain = pg.interpolate(mgr2.paraDomain, mgr2.model, grid.cellCenters())
residual_struc = ((rho_constrain-rho_grid)/rho_grid)*100
pg.show(grid, residual_struc, ax=ax, **kw_compare)
ax.set_xlabel(ax.get_xlabel(),fontsize=13)
ax.set_ylabel(ax.get_ylabel(),fontsize=13)
ax.set_title('Resistivity difference profile of normal mesh',fontweight='bold', size=16)
cb.ax.set_ylabel(cb.ax.get_ylabel(), fontsize=13)
ax.add_patch(plt.Polygon(white_polygon,color='white'))
ax.plot(electrode_x, electrode_y, 'ko', markersize=2)
ax.plot(pg.x(c2.nodes()),pg.y(c2.nodes()),'--k')
ax.text(60,85,'Avg. difference: {:.2f}%'.format(
    np.nansum(abs(residual_struc))/len(residual_struc)
    ),fontweight="bold", size=16)
fig.savefig(join('results', 'slope_synthetic_normal.png'), dpi=300, bbox_inches='tight', transparent=True)
# %%
def calculate_y_from_line(x1, y1, x2, y2, x):
        return y1 + (y2 - y1) / (x2 - x1) * (x - x1)
ymax = calculate_y_from_line(0, 110, c1.node(12).pos()[0], c1.node(12).pos()[1], 60)
y = np.linspace(90,ymax,100)
x = 60*np.ones(len(y))
posVec = [pg.Pos(pos) for pos in zip(x, y)]
para = pg.Mesh(mgr2.paraDomain)  # make a copy
para.setCellMarkers(pg.IVector(para.cellCount()))
fopDP = PriorModelling(para, posVec)
resSmooth = fopDP(mgr2.model)

fig, ax = plt.subplots()
# Find the index of mesh_x that is closest to specific_x_value
index = np.abs(pg.x(c1.nodes()) - 60).argmin()
# Extract the corresponding values of the entire y column
y_interface = pg.y(c1.nodes())[index]
index2 = np.abs(y - y_interface).argmin()
true_rho = np.ones(len(y))
for i in range(len(true_rho)):
        if i < index2:
                true_rho[i] = 150
        else:
                true_rho[i] = 50
# ax.semilogx(list(resSmooth), y,'--k', label="Normal mesh")
ax.semilogx(list(true_rho), y,'-k', label="True value")
for i, depth_i in enumerate(layer_depth):
        posVec = [pg.Pos(pos) for pos in zip(x, y)]
        para = pg.Mesh(mgrs[i].paraDomain)  # make a copy
        para.setCellMarkers(pg.IVector(para.cellCount()))
        fopDP = PriorModelling(para, posVec)
        ax.semilogx(fopDP(mgrs[i].model),y,label='depth = ori+{:.0f}m'.format((i-1)*2))
ax.set_xlabel(r"$\rho$ ($\Omega$m)")
ax.set_ylabel("depth (m)")
ax.grid(which='both',linestyle='--',linewidth=0.5)
ax.legend()
fig.savefig(join('results','slope_synthetic_depth_1D.png'), dpi=300, bbox_inches='tight', transparent=True)
