# %%
# Build a two-layer slope slip model for electrical resistivity tomography synthetic test using pygimli package
import matplotlib.pyplot as plt
import numpy as np
from os.path import join

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
plt.rcParams['font.family'] = 'Times New Roman'#'Microsoft Sans Serif'

# %% Model setup
c1 = mt.createCircle(pos=(0, 310),radius=200, start=1.5*np.pi, end=1.7*np.pi,isClosed=True, marker = 3, area=1)
ax,_ = pg.show(c1)

# We start by creating a three-layered slope (The model is taken from the BSc
# thesis of Constanze Reinken conducted at the University of Bonn).

slope = mt.createPolygon([[0.0, 80], [0.0, 110], 
                          [c1.node(12).pos()[0], c1.node(12).pos()[1]], 
                          [c1.node(12).pos()[0], 80]],
                          isClosed=True, marker = 2)
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
fig.savefig(join('results','slope_electrode.png'), dpi=300, bbox_inches='tight')

scheme = ert.createData(elecs=np.column_stack((electrode_x, electrode_y)),
                           schemeName='dd')
# Create a mesh for the finite element modelling with appropriate mesh quality.
plc = mt.createParaMeshPLC(scheme, paraDepth=30, boundary=1)
c3 = mt.createCircle(pos=(0, 310),radius=200, start=1.5*np.pi, end=1.7*np.pi,isClosed=False, marker = 3, area=1)
plc = plc + c3
plc.regionMarker(2).setPos([60, 125])
plc.regionMarker(1).setPos([60, 100])
meshI = mt.createMesh(plc, quality=33)
pg.show(meshI,markers=True)
# %%
# Create a map to set resistivity values in the appropriate regions
# [[regionNumber, resistivity], [regionNumber, resist３ivity], [...]
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
fig.savefig(join('results','slope.png'), dpi=300, bbox_inches='tight')

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
fig.savefig(join('results','slope_data.png'), dpi=300, bbox_inches='tight',transparent=True)
# save the data for further use
data.save('slope.dat')


# %% Inversion using normal mesh (no prior layer scheme)
# mesh2 = mt.createMesh(slope, 
#                      area=10,
#                      quality=33)    # Quality mesh generation with no angles smaller than X degrees 
plc = mt.createParaMeshPLC(scheme, paraDepth=30, boundary=1)
mesh2 = mt.createMesh(plc, quality=33)
ax,_ = pg.show(mesh2,markers=True)
ax.set_xlabel('Distance (m)',fontsize=13)
ax.set_ylabel('Depth (m)',fontsize=13)
fig = ax.figure
fig.savefig(join('results','slope_normal_mesh.png'), dpi=300, bbox_inches='tight',transparent=True)
# %% Inversion with the ERTManager
# Creat the ERT Manager
mgr2 = ert.ERTManager(data)
# Run the inversion with the preset data. The Inversion mesh will be created
# with default settings.
inv2 = mgr2.invert(mesh=mesh2, lam=100, verbose=True)
mgr2.showResultAndFit(cMap='jet')

# %% Inversion using two-layer based mesh
plc = mt.createParaMeshPLC(scheme, paraDepth=30, boundary=1)
c2 = mt.createCircle(pos=(0, 310),radius=200, start=1.53*np.pi, end=1.67*np.pi,isClosed=False, 
                     boundaryMarker=2) # The marker set to 2 from 1 to be distinguished from the outer boundary
plc = plc + c2
mesh3 = mt.createMesh(plc,
                #       area=1,
                      quality=33)    # Quality mesh generation with no angles smaller than X degrees
ax,_ = pg.show(mesh3,markers=True)
ax.set_xlabel('Distance (m)',fontsize=13)
ax.set_ylabel('Depth (m)',fontsize=13)
fig = ax.figure
fig.savefig(join('results','slope_layered_mesh.png'), dpi=300, bbox_inches='tight',transparent=True)
# %%
mgr3 = ert.ERTManager(data)
# Run the inversion with the preset data. The Inversion mesh will be created
# with default settings.
inv3 = mgr3.invert(mesh=mesh3, lam=100, verbose=True)
mgr3.showResultAndFit(cMap='jet')
# %%
# Plot the inverted profile
ax, cb = pg.show(mgr3.paraDomain,mgr3.model,**kw)
ax.set_xlim([0, c1.node(12).pos()[0]])
# %% Inversion with the ERTModelling and testing different structural constrain weighting values
# Creat the ERT inversion object list
invs = []
for w_s in np.linspace(0,1,5):
        
        # Set such an interface weight of a FORWARD OPERATOR
        fop = ert.ERTModelling()
        fop.setMesh(mesh3)
        fop.regionManager().setInterfaceConstraint(2, w_s)
        fop.setData(data)
        inv3 = pg.Inversion(fop=fop, verbose=True)
        transLog = pg.trans.TransLog()
        inv3.modelTrans = transLog
        inv3.dataTrans = transLog

        # Run the inversion with the preset data. The Inversion mesh will be created
        # with default settings.
        inv3.run(data['rhoa'], data['err'],lam=100)
        invs.append(inv3)
        '''
        # The same inversion setting can be done with the ERTManager
        mgr3 = ert.ERTManager(data)
        mgr3.fop.setMesh(mesh3)
        mgr3.fop.regionManager().setInterfaceConstraint(2, 0.5)
        mgr3.invert(lam=100, verbose=True)
        invs.append(mgr3)
        '''
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
rho = pg.Vector(np.array([row[1] for row in rhomap])[mesh.cellMarkers() - 1] )
# Distinguish the region of the mesh and insert the value of rhomap
rho_grid = pg.interpolate(mesh, rho, grid.cellCenters())

fig = plt.figure(figsize=(16, 20),constrained_layout=True)
for i, w_s in enumerate(np.linspace(0,1,5)):
        ax = plt.subplot(5, 2, 2*i+1)
        pg.show(mesh3, invs[i].model, ax=ax, **kw)
        ax.set_xlabel(ax.get_xlabel(),fontsize=13)
        ax.set_ylabel(ax.get_ylabel(),fontsize=13)
        ax.set_title('Inverted resistivity profile $W_s={:.2f}$'.format(w_s),fontweight='bold', size=16)
        cb.ax.set_ylabel(cb.ax.get_ylabel(), fontsize=13)
        ax.add_patch(plt.Polygon(white_polygon,color='white'))
        ax.plot(electrode_x, electrode_y, 'ko', markersize=2)
        ax.plot(pg.x(c2.nodes()),pg.y(c2.nodes()),'--k')
        ax.text(60,85,'RRMS: {:.2f}%, $\chi^2$: {:.2f}'.format(
                invs[i].relrms(), invs[i].chi2())
                ,fontweight='bold', size=16)
        
        # Compare profiles
        rho_constrain = pg.interpolate(mesh3, invs[i].model, grid.cellCenters())
        ax = plt.subplot(5, 2, 2*i+2)
        residual_struc = ((rho_constrain-rho_grid)/rho_grid)*100
        pg.show(grid, residual_struc, ax=ax, **kw_compare)
        ax.set_xlabel(ax.get_xlabel(),fontsize=13)
        ax.set_ylabel(ax.get_ylabel(),fontsize=13)
        ax.set_title('Resistivity difference profile $W_s={:.2f}$'.format(w_s),fontweight='bold', size=16)
        cb.ax.set_ylabel(cb.ax.get_ylabel(), fontsize=13)
        ax.add_patch(plt.Polygon(white_polygon,color='white'))
        ax.plot(electrode_x, electrode_y, 'ko', markersize=2)
        ax.plot(pg.x(c2.nodes()),pg.y(c2.nodes()),'--k')
        ax.text(60,85,'Avg. difference: {:.2f}%'.format(
            np.nansum(abs(residual_struc))/len(residual_struc))
            ,fontweight="bold", size=16)  


# fig.savefig(join('results','slope_synthetic_Ws.png'), dpi=300, bbox_inches='tight', transparent=True)
