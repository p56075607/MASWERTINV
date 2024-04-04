# %%
# Build a two-layer model for electrical resistivity tomography synthetic and sensitivity test using pygimli package
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from os.path import join

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
plt.rcParams["font.family"] = 'Times New Roman'#"Microsoft Sans Serif"

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
ax,_ = pg.show(mesh, 
        data=rhomap, cMap='jet', logScale=True,
        label=pg.unit('res'), 
        showMesh=True)
ax.set_xlabel('Distance (m)',fontsize=13)
ax.set_ylabel('Depth (m)',fontsize=13)
fig = ax.figure
# fig.savefig(join('results','layer.png'), dpi=300, bbox_inches='tight')
# save the mesh to binary file
mesh.save("mesh.bms") # can be load by pg.load()

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

# Inversion using normal mesh (no prior layer scheme)
world2 = mt.createWorld(start=[left, 0], end=[right, -depth], worldMarker=True)
mesh2 = mt.createMesh(world2, 
                     area=1,
                     quality=33)    # Quality mesh generation with no angles smaller than X degrees 
ax,_ = pg.show(mesh2)
ax.set_xlabel('Distance (m)',fontsize=13)
ax.set_ylabel('Depth (m)',fontsize=13)
fig = ax.figure
# fig.savefig(join('results','normal_mesh.png'), dpi=300, bbox_inches='tight')

# Inversion with the ERTManager
# Creat the ERT Manager
mgr2 = ert.ERTManager(data)
# Run the inversion with the preset data. The Inversion mesh will be created
# with default settings.
inv2 = mgr2.invert(mesh=mesh2, lam=100, verbose=True)
mgr2.showResultAndFit(cMap='jet')

# %% Comparesion of the results by the residual profile
# Re-interpolate the grid
mesh_x = np.linspace(0,100,100)
mesh_y = np.linspace(-30,0,60)
grid = pg.createGrid(x=mesh_x,y=mesh_y )

# Creat a pg RVector with the length of the cell of mesh and the value of rhomap
rho = pg.Vector(np.array([row[1] for row in rhomap])[mesh.cellMarkers() - 1] )
# Distinguish the region of the mesh and insert the value of rhomap
rho_grid = pg.interpolate(mesh, rho, grid.cellCenters())

# %% Inversion using layer interface based mesh
left_edge = 30
right_edge = 70
layer_depth = np.linspace(-3, -7, 5)

mgrs = []
for n, depth_i in enumerate(layer_depth):
    plc = mt.createParaMeshPLC(data, paraDepth=30, boundary=0.5)
    interface2 = mt.createLine(start=[left_edge, depth_i], end=[right_edge, depth_i])
    plc += interface2
    mesh3 = mt.createMesh(plc,
                        area=1,
                        quality=33)    # Quality mesh generation with no angles smaller than X degrees
    # Creat the ERT Manager
    mgr3 = ert.ERTManager(data)
    # Run the inversion with the preset data. The Inversion mesh will be created
    # with default settings.
    inv3 = mgr3.invert(mesh=mesh3, lam=100, verbose=True)
    mgrs.append(mgr3)
# %% Plot the results by resistivity and the residual profile    
fig = plt.figure(figsize=(18, 20),constrained_layout=False)
for n, depth_i in enumerate(layer_depth):
    interface2 = mt.createLine(start=[left_edge, depth_i], end=[right_edge, depth_i])
    ax = plt.subplot(len(layer_depth), 2, 2*n+1)
    rho_layer_grid = pg.interpolate(mgrs[n].paraDomain, mgrs[n].model, grid.cellCenters())
    # Subplots:structured constrained grid
    pg.viewer.showMesh(mgrs[n].paraDomain, mgrs[n].model,#grid,data=rho_layer_grid,
                        ax=ax,
                        label='Resistivity ($\Omega m$)',
                        logScale=True,cMap='jet',cMin=50,cMax=150,
                        xlabel='Distance (m)', ylabel='Depth (m)',orientation = 'vertical')
    ax.set_title('Structured constrained inverted resistivity profile\nInterface at {:.0f} m'.format(depth_i),fontweight="bold", size=16)
    triangle_left = np.array([[left, -depth], [depth, -depth], [left,0], [left, -depth]])
    triangle_right = np.array([[right, -depth], [right-depth, -depth], [right,0], [right, depth]])
    ax.add_patch(plt.Polygon(triangle_left,color='white'))
    ax.add_patch(plt.Polygon(triangle_right,color='white'))
    pg.show(interface2,ax=ax)
    ax.plot(np.array(pg.x(data)), np.array(pg.z(data)),'ko')
    ax.set_ylim(-30, 0)
    ax.set_xlim(0,100)
    ax.text(5,-25,'RRMS: {:.2f}%, $\chi^2$: {:.2f}'.format(
        mgrs[n].inv.relrms(), mgrs[n].inv.chi2())
            ,fontweight="bold", size=16)
    # Calculate the resistivity relative difference
    # Subplot:Layered mesh resistivity residual
    ax = plt.subplot(len(layer_depth), 2, 2*n+2)
    residual_layer_grid = ((rho_layer_grid - rho_grid)/rho_grid)*100
    pg.viewer.showMesh(grid,data=residual_layer_grid,ax=ax,
                        label='Relative resistivity difference (%)',
                    #     logScale=True, 
                        cMap='bwr', 
                        cMin=-35,cMax=35,
                        xlabel="x (m)", ylabel="z (m)",orientation = 'vertical',
                        )
    ax.set_title('Structured constrained resistivity difference profile\nInterface at {:.0f} m'.format(depth_i),fontweight="bold", size=16)
    ax.add_patch(plt.Polygon(triangle_left,color='white'))
    ax.add_patch(plt.Polygon(triangle_right,color='white'))
    pg.show(interface2,ax=ax)
    ax.plot(np.array(pg.x(data)), np.array(pg.z(data)),'ko')
    ax.set_ylim(-30, 0)    
    ax.set_xlim(0,100)
    ax.text(5,-25,'Avg. difference: {:.2f}%'.format(
            sum(abs(residual_layer_grid))/len(residual_layer_grid))
            ,fontweight="bold", size=16)
fig.savefig(join('results','layer_sensitivity.png'), dpi=300, bbox_inches='tight')
# %%
# Plot the results of normal mesh
kw = dict(cMin=50, cMax=150, logScale=True, cMap='jet',
          xlabel='Distance (m)', ylabel='Depth (m)', 
          label=pg.unit('res'), orientation='vertical')
fig = plt.figure(figsize=(16, 4),constrained_layout=True)
ax = plt.subplot(1, 2, 1)
_,cb = pg.show(mgr2.paraDomain, mgr2.model, ax=ax, **kw)
ax.set_xlabel(ax.get_xlabel(),fontsize=13)
ax.set_ylabel(ax.get_ylabel(),fontsize=13)
ax.set_title('Inverted resistivity profile of normal mesh',fontweight='bold', size=16)
cb.ax.set_ylabel(cb.ax.get_ylabel(), fontsize=13)
ax.add_patch(plt.Polygon(triangle_left,color='white'))
ax.add_patch(plt.Polygon(triangle_right,color='white'))
ax.plot(np.array(pg.x(data)), np.array(pg.z(data)),'ko')
ax.plot(pg.x(interface2.nodes()),pg.y(interface2.nodes()),'--k')
ax.text(5,-25,'RRMS: {:.2f}%, $\chi^2$: {:.2f}'.format(
        mgr2.inv.relrms(), mgr2.inv.chi2())
        ,fontweight='bold', size=16)
ax.set_ylim(-30, 0)
ax.set_xlim(0,100)
# Compare profiles
kw_compare = dict(cMin=-50, cMax=50, cMap='bwr',
                  label='Relative resistivity difference (%)',
                  xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical')
ax = plt.subplot(1, 2, 2)
rho_constrain = pg.interpolate(mgr2.paraDomain, mgr2.model, grid.cellCenters())
residual_struc = ((rho_constrain-rho_grid)/rho_grid)*100
pg.show(grid, residual_struc, ax=ax, **kw_compare)
ax.set_xlabel(ax.get_xlabel(),fontsize=13)
ax.set_ylabel(ax.get_ylabel(),fontsize=13)
ax.set_title('Resistivity difference profile of normal mesh',fontweight='bold', size=16)
cb.ax.set_ylabel(cb.ax.get_ylabel(), fontsize=13)
ax.add_patch(plt.Polygon(triangle_left,color='white'))
ax.add_patch(plt.Polygon(triangle_right,color='white'))
ax.plot(pg.x(interface2.nodes()),pg.y(interface2.nodes()),'--k')
ax.plot(np.array(pg.x(data)), np.array(pg.z(data)),'ko')
ax.text(5,-25,'Avg. difference: {:.2f}%'.format(
    np.nansum(abs(residual_struc))/len(residual_struc)
    ),fontweight="bold", size=16)
ax.set_ylim(-30, 0)
ax.set_xlim(0,100)
# %%
from pygimli.frameworks import PriorModelling
y = np.linspace(-30,0,100)
x = 50*np.ones(len(y))
posVec = [pg.Pos(pos) for pos in zip(x, y)]
para = pg.Mesh(mgr2.paraDomain)  # make a copy
para.setCellMarkers(pg.IVector(para.cellCount()))
fopDP = PriorModelling(para, posVec)
resSmooth = fopDP(mgr2.model)

fig, ax = plt.subplots()
interface2 = mt.createLine(start=[left_edge, -5], end=[right_edge, -5])
# Find the index of mesh_x that is closest to specific_x_value
index = np.abs(pg.x(interface2.nodes()) - 50).argmin()
# Extract the corresponding values of the entire y column
y_interface = pg.y(interface2.nodes())[index]
index2 = np.abs(y - y_interface).argmin()

interface3 = mt.createLine(start=[left_edge, -15], end=[right_edge, -15])
y_interface3 = pg.y(interface3.nodes())[index]
index3 = np.abs(y - y_interface3).argmin()

true_rho = np.ones(len(y))
for i in range(len(true_rho)):
        if i > index2:
            true_rho[i] = 50
        elif i < index3:
            true_rho[i] = 150
        else:
            true_rho[i] = 100
ax.semilogx(list(resSmooth), y,'--k', label="Normal mesh")
ax.semilogx(list(true_rho), y,'-k',linewidth=3, label="True value")
for i, w_s in enumerate(np.linspace(0,1,5)):
        posVec = [pg.Pos(pos) for pos in zip(x, y)]
        para = pg.Mesh(mgrs[i].paraDomain)  # make a copy
        para.setCellMarkers(pg.IVector(para.cellCount()))
        fopDP = PriorModelling(para, posVec)
        ax.semilogx(fopDP(mgrs[i].model),y,label='$W_s = {:.2f}$'.format(w_s))
ax.set_xlabel(r"$\rho$ ($\Omega$m)")
ax.set_ylabel("depth (m)")
ax.grid(which='both',linestyle='--',linewidth=0.5)
ax.legend()
fig.savefig(join('results','synthetic_depth_sensitivity_1D.png'), dpi=300, bbox_inches='tight', transparent=True)

#%%
# Show all the mesh and the interface
plc = mt.createParaMeshPLC(data, paraDepth=30, boundary=0.5)
for n, depth_i in enumerate(layer_depth):
    interface2 = mt.createLine(start=[left_edge, depth_i], end=[right_edge, depth_i])
    plc += interface2
mesh_all = mt.createMesh(plc,
                    area=1,
                    quality=33)   
ax,_ = pg.show(mesh_all,label='Mesh interface at different depth',showMesh=True)
interface_long = mt.createLine(start=[left, -5], end=[right, -5])
ax.plot(pg.x(interface_long.nodes()),pg.y(interface_long.nodes()),'--b',label='True model interface')
ax.set_xlim([0,100])
ax.set_ylim([-20,0])
ax.set_xlabel('Distance (m)',fontsize=13)
ax.set_ylabel('Depth (m)',fontsize=13)
ax.set_yticks(np.linspace(-20,0,5))
ax.legend()
fig = ax.figure
fig.savefig(join('results','layer_sensitivity_interface.png'), dpi=300, bbox_inches='tight')