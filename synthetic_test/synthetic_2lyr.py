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


def synthetic_2lyr_creatModel(rhomap):
    left = 0
    right = 100
    depth = 30
    world = mt.createWorld(start=[left, 0], end=[right, -depth],
                        layers=[-5], 
                        worldMarker=True)
    block = mt.createRectangle(start=[30, -10], end=[70, -20], marker=3)
    geom = block + world

    # Synthetic data generation
    scheme = ert.createData(elecs=np.linspace(start=0, stop=100, num=51),
                            schemeName='dd')

    # Create a mesh for the finite element modelling with appropriate mesh quality.
    mesh = mt.createMesh(geom, 
                        area=1,
                        quality=33)    # Quality mesh generation with no angles smaller than X degrees 

    data = ert.simulate(mesh, scheme=scheme, res=rhomap, noiseLevel=1,
                        noiseAbs=1e-6, 
                        seed=1337) #seed : numpy.random seed for repeatable noise in synthetic experiments 

    pg.info(np.linalg.norm(data['err']), np.linalg.norm(data['rhoa']))
    pg.info('Simulated data', data)
    pg.info('The data contains:', data.dataMap().keys())
    pg.info('Simulated rhoa (min/max)', min(data['rhoa']), max(data['rhoa']))
    pg.info('Selected data noise %(min/max)', min(data['err'])*100, max(data['err'])*100)

    # Inversion using normal mesh (no prior layer scheme)
    world2 = mt.createWorld(start=[left, 0], end=[right, -depth], worldMarker=True)
    mesh2 = mt.createMesh(world2, 
                        area=0.5,
                        quality=33)    # Quality mesh generation with no angles smaller than X degrees 

    # Inversion using layer-based mesh
    plc = mt.createParaMeshPLC(data, paraDepth=depth, boundary=0.5)
    left_edge = left
    right_edge = right
    interface2 = mt.createLine(start=[left_edge, -5], end=[right_edge, -5])
    plc = interface2 + plc

    mesh3 = mt.createMesh(plc,
                        area=0.5,
                        quality=33)    # Quality mesh generation with no angles smaller than X degrees
    
    return mesh, data, mesh2, mesh3, left, right, depth, interface2

def synthetic_2lyr_runInversion(data, left, right, depth, test_name, mesh2, mesh3, lam):
    # Inversion with the ERTManager
    # Creat the ERT Manager
    mgr2 = ert.ERTManager(data)
    # Run the inversion with the preset data. The Inversion mesh will be created
    # with default settings.
    inv2 = mgr2.invert(mesh=mesh2, lam=lam, verbose=True)

    # Inversion with the ERTManager
    # Creat the ERT Manager
    mgr3 = ert.ERTManager(data)
    # Run the inversion with the preset data. The Inversion mesh will be created
    # with default settings.
    inv3 = mgr3.invert(mesh=mesh3, lam=lam, verbose=True)
    
    return mgr2, mgr3

def synthetic_2lyr_plotResults(mgr2, mgr3, rhomap, mesh, data, mesh2, mesh3, interface2, left, right, depth, test_name, lam, plot_result, save_plot):
    # Export the information about the inversion
    def save_inv_result_and_info(output_ph, mgr, lam):
        mgr.saveResult(output_ph)
        with open(join(output_ph,'ERTManager','inv_info.txt'), 'w') as write_obj:
            write_obj.write('## Final result ##\n')
            write_obj.write('rrms:{} %\n'.format(mgr.inv.relrms()))
            write_obj.write('chi2:{}\n'.format(mgr.inv.chi2()))
            write_obj.write('## Inversion parameters ##\n')
            write_obj.write('use lam:{}\n'.format(lam))
            write_obj.write('## Iteration ##\n')
            write_obj.write('Iter.  rrms  chi2\n')
            for iter in range(len(mgr.inv.rrmsHistory)):
                write_obj.write('{:.0f},{:.2f},{:.2f}\n'.format(iter,mgr.inv.rrmsHistory[iter],mgr.inv.chi2History[iter]))

            # Export model response in this inversion 
            pg.utils.saveResult(join(output_ph,'ERTManager','model_response.txt'),
                                data=mgr.inv.response, mode='w') 
    mgr2.showResultAndFit(cMap='jet')
    output_ph_normal = join('output',test_name,'normal')
    save_inv_result_and_info(output_ph_normal, mgr2, lam)

    output_ph_layered = join('output',test_name,'layered')
    mgr3.showResultAndFit(cMap='jet')
    save_inv_result_and_info(output_ph_layered, mgr3, lam)

    # Plot the process results
    # Take a look at the mesh and the resistivity distribution
    kw = dict(cMin=100, cMax=1500, logScale=True, cMap='jet',
            xlabel='Distance (m)', ylabel='Depth (m)', 
            label=pg.unit('res'), orientation='vertical')
    if plot_result == True:
        # Plot the true model
        ax,_ = pg.show(mesh, 
                data=rhomap, **kw, showMesh=False) 
        ax.set_xlabel('Distance (m)',fontsize=13)
        ax.set_ylabel('Depth (m)',fontsize=13)
        fig = ax.figure
        if save_plot == True:
            fig.savefig(join('output',test_name,'True_model.png'), dpi=300, bbox_inches='tight')

        # Plot the simulated data
        ax, _ = pg.show(data,cMap='jet')
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Array type and \nElectrode separation (m)')
        ax.plot(np.array(pg.x(data)), np.array(pg.z(data)),'kv',label='Electrode')
        fig = ax.figure
        if save_plot == True:
            fig.savefig(join('output',test_name,'Simulated_data.png'), dpi=300, bbox_inches='tight')

        # Plot the normal mesh
        ax,_ = pg.show(mesh2)
        ax.set_xlabel('Distance (m)',fontsize=13)
        ax.set_ylabel('Depth (m)',fontsize=13)
        fig = ax.figure
        if save_plot == True:
            fig.savefig(join('output',test_name,'Normal_mesh.png'), dpi=300, bbox_inches='tight')

        # Plot the layered mesh
        ax,_ = pg.show(mesh3)
        ax.set_xlim(0, 100)
        ax.set_ylim(-30, 0)
        ax.set_xlabel('Distance (m)',fontsize=13)
        ax.set_ylabel('Depth (m)',fontsize=13)
        fig = ax.figure
        if save_plot == True:
            fig.savefig(join('output',test_name,'Layered_mesh.png'), dpi=300, bbox_inches='tight')   

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
                        ,fontweight="bold", size=16)

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
            ax.text(5,-25,'Avg. difference: {:.2f}%'.format(
                    np.nansum(abs(data))/len(data))
                    ,fontweight="bold", size=16)
            ax.set_ylim(-30, 0)
            ax.set_xlim(0,100)

        residual_normal_grid = ((rho_normal_grid - rho_grid)/rho_grid)*100
        plot_residual(ax=ax4, grid=grid, data=residual_normal_grid, title='Normal mesh resistivity difference profile', **kw_compare)
        # Subplot 6:Layered mesh resistivity residual
        residual_layer_grid = ((rho_layer_grid - rho_grid)/rho_grid)*100
        plot_residual(ax=ax6, grid=grid, data=residual_layer_grid, title='Structured constrained resistivity difference profile', **kw_compare)
        if save_plot == True:
            fig.savefig(join('output',test_name,'Compare.png'), dpi=300, bbox_inches='tight', transparent=True)

    return mgr2, mgr3

# %%
# Test the synthetic_2ly TARI model
rhomap = [[1, 100.],
          [2, 500.],
          [3, 1500.]]
test_name = 'synthetic_2lyr_tari'
lam=100
plot_result = True
save_plot = True
tari_parameter = []

mesh, data, mesh2, mesh3, left, right, depth, interface2 = synthetic_2lyr_creatModel(rhomap)
tari_parameter.append([mesh, data, mesh2, mesh3, left, right, depth, interface2])
mgr2, mgr3 = synthetic_2lyr_runInversion(data, left, right, depth, test_name, mesh2, mesh3, lam)
tari_parameter.append([mgr2, mgr3])
synthetic_2lyr_plotResults(mgr2, mgr3, rhomap, mesh, data, mesh2, mesh3, interface2, left, right, depth, test_name, lam, plot_result, save_plot)

# %%
# Test the synthetic_2ly HSR model
rhomap = [[1, 1500.],
          [2, 500.],
          [3, 100.]]
test_name = 'synthetic_2lyr_hsr'
lam=100
plot_result = True
save_plot = True
hsr_parameter = []

mesh, data, mesh2, mesh3, left, right, depth, interface2 = synthetic_2lyr_creatModel(rhomap)
hsr_parameter.append([mesh, data, mesh2, mesh3, left, right, depth, interface2])
mgr2, mgr3 = synthetic_2lyr_runInversion(data, left, right, depth, test_name, mesh2, mesh3, lam)
hsr_parameter.append([mgr2, mgr3])
synthetic_2lyr_plotResults(mgr2, mgr3, rhomap, mesh, data, mesh2, mesh3, interface2, left, right, depth, test_name, lam, plot_result, save_plot)
