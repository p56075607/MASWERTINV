# %%
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'#'Microsoft Sans Serif'
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
from scipy.interpolate import griddata, NearestNDInterpolator
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join
from os import listdir
from matplotlib.patches import Polygon
from shapely.geometry import LineString, Point
from scipy.optimize import root
import pickle
from scipy.spatial import Delaunay

def import_geometry_csv():
    geo = pd.read_csv('geometry.csv', header=None)
    geo.columns = ['x', 'y']

    return geo
geo = import_geometry_csv()
geo['y'] = geo['y']-5

# Define the function to create the electrode coordinates
def create_electrode_and_mesh(geo):
    line_coords = np.array(geo[1:-1])
    def create_electrode_coords(line_coords):
        # calculate the total distance of the line
        distances = np.sqrt(np.sum(np.diff(line_coords, axis=0)**2, axis=1))
        total_distance = np.sum(distances)

        # generate equidistant points
        num_points = int(total_distance) + 1
        electrode_distances = np.linspace(0, total_distance, num=num_points)

        # calculate the coordinates of the electrodes
        electrode_coords = []
        current_distance = 0
        for i in range(len(line_coords) - 1):
            segment_length = distances[i]
            while current_distance <= segment_length:
                t = current_distance / segment_length
                point = (1 - t) * line_coords[i] + t * line_coords[i + 1]
                electrode_coords.append(point)
                current_distance += 1
            current_distance -= segment_length

        electrode_coords = np.array(electrode_coords)

        # # plot the line and the electrodes
        # plt.plot(line_coords[:, 0], line_coords[:, 1], 'k-', label='Terrain Line')
        # plt.plot(electrode_coords[:, 0], electrode_coords[:, 1], 'ro', label='electrode')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.legend()
        # plt.show()

        # extract the x and y coordinates of the electrodes
        electrode_x, electrode_y = electrode_coords[:, 0], electrode_coords[:, 1]
        print(f"Electrode X coordinates: {electrode_x}")
        print(f"Electrode Y coordinates: {electrode_y}")
        return electrode_x, electrode_y

    electrode_x, electrode_y = create_electrode_coords(line_coords)
    scheme = ert.createData(elecs=np.column_stack((electrode_x, electrode_y)),
                            schemeName=None)

    interface_coords = np.array(geo[1:-1])
    interface_coords[:,1]= interface_coords[:,1]-2
    print(  interface_coords)
    slope = mt.createPolygon(interface_coords,
                            isClosed=False, marker = 2,boundaryMarker=2)
    # pg.show(slope, markers=True, showMesh=True)

    plc_forward = mt.createParaMeshPLC(scheme,paraDepth=5,paraDX=1/10,paraMaxCellSize=0.05,
                                        balanceDepth=True)
    plc_forward = plc_forward + slope
    plc_inverse = mt.createParaMeshPLC(scheme,paraDepth=5,paraDX=1/10,paraMaxCellSize=0.1,
                                        balanceDepth=True)
    plc_inverse_constrain = plc_inverse + slope
    mesh_inverse = mt.createMesh(plc_inverse)
    mesh_inverse_constrain = mt.createMesh(plc_inverse_constrain)
    mesh = mt.createMesh(plc_forward)
    # ax,_ = pg.show(mesh_inverse,markers=False,showMesh=True)
    # ax.set_xlim([-10, 40])
    # ax.set_ylim([5,20])

    return electrode_x, electrode_y, mesh, mesh_inverse, mesh_inverse_constrain, interface_coords, slope
electrode_x, electrode_y, mesh, mesh_inverse, mesh_inverse_constrain, interface_coords, slope = create_electrode_and_mesh(geo)

def import_COMSOL_csv(csv_name = '2Layers_water content (1a).csv',plot=True,geo=geo,style='scatter'):
    
    df = pd.read_csv(csv_name, comment='%', header=None)
    df.columns = ['X', 'Y', 'theta']
    if plot:
        fig,ax = plt.subplots(figsize=(6.4, 4.8))
        if style == 'scatter':
            plot = ax.scatter(df['X'], df['Y'], c=df['theta'], marker='o',s=1,cmap='Blues',
                                        vmin=min(df['theta']),vmax=max(df['theta']))
        else:
            x_min, x_max = df['X'].min(), df['X'].max()
            y_min, y_max = df['Y'].min(), df['Y'].max()

            grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

            grid_theta = griddata((df['X'], df['Y']), df['theta'], (grid_x, grid_y), method='linear')

            plot = ax.contourf(grid_x, grid_y, grid_theta, levels=32, cmap='Blues'
                                ,vmin=min(df['theta']),vmax=max(df['theta']))

            terrain_polygon = np.vstack((geo.to_numpy()[2:5], [max(geo['x']), max(geo['y'])], [min(geo['x']), max(geo['y'])]))
            print(terrain_polygon)
            mask_polygon = Polygon(terrain_polygon, closed=True, color='white', zorder=10)

            ax.add_patch(mask_polygon)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(r'$\theta$ Plot from COMSOL')
        ax.set_aspect('equal')
        ax.set_xlim([min(geo['x']), max(geo['x'])])
        ax.set_ylim([min(geo['y']),max(geo['y'])])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2.5%', pad=0.05)
        cbar = fig.colorbar(plot, cax=cax)
        cbar.set_label(r'$\theta$')
        fig.savefig(join('results',csv_name[-7:-4]+'SWC Plot from COMSOL.png'),dpi=300,bbox_inches='tight')

    return df

def Forward_inversion(csv_file_name,geo=geo,plot=True):
    df = import_COMSOL_csv(csv_name = csv_file_name,plot=False,style='scatter')

    # Tow layers
    def convert_SWC_to_resistivity(df, mesh, interface_coords,plot=True):
        grid_SWC = griddata((df[['X', 'Y']].to_numpy()), df['theta'].to_numpy(), 
                    (np.array(mesh.cellCenters())[:, :2]), method='linear', fill_value=np.nan)

        fill_value = np.nanmedian(grid_SWC)
        grid_SWC = np.nan_to_num(grid_SWC, nan=fill_value)
        line = LineString(interface_coords)
        resistivity = np.zeros(mesh.cellCount())
        # check each point
        for i, point in enumerate(np.array(mesh.cellCenters())[:,:-1]):
            point = Point(point)
            distance = line.distance(point)
            # check the position of the point relative to the line
            if point.y > line.interpolate(line.project(point)).y:
                n = 1.83
                cFluid = 1/(0.57*106)
                resistivity[i] = 1/(cFluid*grid_SWC[i]**n)
            else:
                n = 1.34
                cFluid = 1/(0.58*75)
                resistivity[i] = 1/(cFluid*grid_SWC[i]**n)
        if plot==True:
            kw = dict(cMin=min(resistivity), cMax=max(resistivity), logScale=True, cMap='jet',
                    xlabel='X (m)', ylabel='Y (m)', 
                    label=pg.unit('res'), orientation='vertical')
            ax,_ = pg.show(mesh,resistivity, **kw)
            ax.set_xlim([min(geo['x']), max(geo['x'])])
            ax.set_ylim([min(geo['y']),max(geo['y'])])
            ax.set_title('Resistivity Model from SWC')
            fig = ax.figure
            fig.savefig(join('results','Resistivity Model from SWC.png'),dpi=300,bbox_inches='tight')
        return resistivity
    resistivity = convert_SWC_to_resistivity(df, mesh, interface_coords,plot=True)
    # resistivity_constrain = convert_SWC_to_resistivity(df, mesh_inverse_constrain, interface_coords)

    def combine_array(schemeName,mesh, res):
        def show_simulation(data):
            pg.info(np.linalg.norm(data['err']), np.linalg.norm(data['rhoa']))
            pg.info('Simulated data', data)
            pg.info('The data contains:', data.dataMap().keys())
            pg.info('Simulated rhoa (min/max)', min(data['rhoa']), max(data['rhoa']))
            pg.info('Selected data noise %(min/max)', min(data['err'])*100, max(data['err'])*100)
            
        kw_noise = dict(noiseLevel=0.01, noiseAbs=0.001, seed=1337)
        if (len(schemeName) == 1):
            data = ert.simulate(mesh = mesh, res=res,
                        scheme=ert.createData(elecs=np.column_stack((electrode_x, electrode_y)),
                            schemeName=schemeName),
                        **kw_noise)
            
        elif(len(schemeName) > 1):
            data = ert.simulate(mesh = mesh, res=res,
                        scheme=ert.createData(elecs=np.column_stack((electrode_x, electrode_y)),
                            schemeName=schemeName[0]),
                        **kw_noise)
            for i in range(1,len(schemeName)):
                print('Simulating', schemeName[i])
                data.add(ert.simulate(mesh = mesh, res=res,
                        scheme=ert.createData(elecs=np.column_stack((electrode_x, electrode_y)),
                            schemeName=schemeName[i]),
                        **kw_noise))

        show_simulation(data)
        return data
        
    data = combine_array(schemeName=['dd','wa','wb','slm'],mesh = mesh, res=resistivity)
    if plot==True:
        ax,_ = ert.showData(data)
        fig = ax.figure
        fig.savefig(join('results','Forward_resistivity.png'),dpi=300,bbox_inches='tight')

    mgr = ert.ERTManager()
    mgr.invert(data=data, mesh=mesh_inverse_constrain, lam=100,verbose=True)
    mgr.showResultAndFit()

    mgr_normal = ert.ERTManager()
    mgr_normal.invert(data=data, mesh=mesh_inverse, lam=100,verbose=True)
    mgr_normal.showResultAndFit()

    return mgr, mgr_normal, data, resistivity,df

    # 定義 Van Genuchten 模???漱洠蝻?

def van_genuchten_inv(theta, theta_r, theta_s, alpha, n):
    if theta == theta_s:
        return 0  
    m = 1 - 1/n
    func = lambda h: theta_r + (theta_s - theta_r) / (1 + (alpha * np.abs(h))**n)**m - theta
    
    initial_guesses = [-1, -10, -100, -1000]
    for h_guess in initial_guesses:
        sol = root(func, h_guess)
        if sol.success:
            return sol.x[0]
    
    raise ValueError(f"Solution not found for theta = {theta}")

def brooks_corey_inv(theta, theta_r, theta_s, alpha, n):
    Se = (theta - theta_r) / (theta_s - theta_r)
    
    if Se == 1:
        return -(1 / alpha)/100 #[m]
    else:
        h = -(1 / alpha) * (Se ** (-1 / n))/100 #[m]
        return h

def Archie_inv(resistivity, cFluid, n_Archies):
    return (1 / (resistivity * cFluid)) ** (1 / n_Archies)

def convert_resistivity_to_Hp(df, resistivity, mesh, interface_coords):

    # Define interface line
    line = LineString(interface_coords)
    
    # Initialize SWC and Hp arrays for mesh points
    mesh_SWC = np.zeros(len(resistivity))
    mesh_Hp = np.zeros(len(resistivity))
    
    # Convert resistivity to SWC and Hp at mesh points
    for i, point in enumerate(np.array(mesh.cellCenters())[:, :2]):
        point_geom = Point(point)
        if point[1] > line.interpolate(line.project(point_geom)).y:
            # [Top layer] SWRC parameters
            theta_r = 0.041  
            theta_s = 0.412  
            alpha = 0.068    
            n = 0.322
            n_Archies = 1.83
            cFluid = 1 / (0.57 * 106)
        else:
            # [Bottom layer] SWRC parameters
            theta_r = 0.090  
            theta_s = 0.385  
            alpha = 0.027     
            n = 0.131
            n_Archies = 1.34
            cFluid = 1 / (0.58 * 75)
        
        mesh_SWC[i] = Archie_inv(resistivity[i], cFluid, n_Archies)
        
        if mesh_SWC[i] <= theta_r:
            print('SWC[i]',mesh_SWC[i],point)
            mesh_SWC[i] = theta_r + 0.0001
        
        mesh_Hp[i] = brooks_corey_inv(mesh_SWC[i], theta_r, theta_s, alpha, n)
    
    # Interpolate SWC and Hp to the target points
    df_coords = df[['X', 'Y']].to_numpy()
    grid_SWC = griddata(np.array(mesh.cellCenters())[:, :2], mesh_SWC, df_coords, method='linear', fill_value=np.nan)
    grid_Hp = griddata(np.array(mesh.cellCenters())[:, :2], mesh_Hp, df_coords, method='linear', fill_value=np.nan)
    
    fill_value_SWC = np.nanmedian(grid_SWC)
    fill_value_Hp = np.nanmedian(grid_Hp)
    
    grid_SWC = np.nan_to_num(grid_SWC, nan=fill_value_SWC)
    grid_Hp = np.nan_to_num(grid_Hp, nan=fill_value_Hp)
    
    return grid_Hp, grid_SWC


csv_path = 'water content (from COMSOL)'
csvfiles = [_ for _ in listdir(csv_path) if _.endswith('.csv')]

def extract_day_number(filename):
    # Split the filename to get the part with the number
    return int(filename.split('_')[-1].replace('d.csv', ''))

csvfiles = sorted(csvfiles, key=extract_day_number)
print(csvfiles)

for i,csv_file_name in enumerate(csvfiles[:1]):
    df = read_Layers_water_content[i]['df']

    Hp, SWC= convert_resistivity_to_Hp(df, read_Layers_water_content[i]['mgr']['model'], read_Layers_water_content[i]['mgr']['paraDomain'], interface_coords)
    Hp_normal, SWC_normal = convert_resistivity_to_Hp(df, read_Layers_water_content[i]['mgr_normal']['model'], read_Layers_water_content[i]['mgr_normal']['paraDomain'], interface_coords)

    df['pressure_head_constrain'] = Hp
    df['pressure_head_normal'] = Hp_normal

    df.to_csv(join('To COMSOL csv','inverted_pressure_head_constrain'+csv_file_name), columns=['X', 'Y', 'pressure_head_constrain'], index=False)
    df.to_csv(join('To COMSOL csv','inverted_pressure_head_normal'+csv_file_name), columns=['X', 'Y', 'pressure_head_normal'], index=False)
# %%
def save_inversion_results(mgr, save_ph):
    mgr.saveResult(save_ph)
    # Export the information about the inversion
    output_ph = join(save_ph,'ERTManager','inv_info.txt')
    with open(output_ph, 'w') as write_obj:
        write_obj.write('## Final result ##\n')
        write_obj.write('rrms:{}\n'.format(mgr.inv.relrms()))
        write_obj.write('chi2:{}\n'.format(mgr.inv.chi2()))

        write_obj.write('## Inversion parameters ##\n')
        write_obj.write('use lam:{}\n'.format(mgr.inv.lam))

        write_obj.write('## Iteration ##\n')
        write_obj.write('Iter.  rrms  chi2\n')
        for iter in range(len(mgr.inv.rrmsHistory)):
            write_obj.write('{:.0f},{:.2f},{:.2f}\n'.format(iter,mgr.inv.rrmsHistory[iter],mgr.inv.chi2History[iter]))
    # Export model response in this inversion 
    pg.utils.saveResult(join(save_ph,'ERTManager','model_response.txt'),
                        data=mgr.inv.response, mode='w')

Layers_water_content = {}
for i,csv_file_name in enumerate(csvfiles):
    print('Processing', csv_file_name)
    mgr, mgr_normal, data, resistivity,df = Forward_inversion(join(csv_path,csv_file_name),geo=geo,plot=False)
    Hp, SWC= convert_resistivity_to_Hp(df, mgr.model, mgr.paraDomain, interface_coords)
    Hp_normal, SWC_normal = convert_resistivity_to_Hp(df, mgr_normal.model, mgr_normal.paraDomain, interface_coords)

    SWC_real = df['theta'].to_numpy()

    save_ph = join('Inversion_results',csv_file_name[-7:-4]+'constrained')
    save_inversion_results(mgr, save_ph)

    save_ph = join('Inversion_results',csv_file_name[-7:-4]+'normal')
    save_inversion_results(mgr_normal, save_ph)

    Layers_water_content[i] = {
        'df': df,
        'SWC': SWC,
        'SWC_normal': SWC_normal,
        'SWC_real': SWC_real,
        'Hp': Hp,
        'Hp_normal': Hp_normal,
        'resistivity': resistivity,
        
    }

# open('rainfall.pkl', 'wb').write(pickle.dumps(Layers_water_content))
# %%
def load_inversion_results(save_ph):
    output_ph = join(save_ph,'ERTManager')
    para_domain = pg.load(join(output_ph,'resistivity-pd.bms'))
    mesh_fw = pg.load(join(output_ph,'resistivity-mesh.bms'))
    # Load data file
    data_path = join(output_ph,'data.dat')
    data = ert.load(data_path)
    investg_depth = (max(pg.x(data))-min(pg.x(data)))*0.3
    # Load model response
    resp_path = join(output_ph,'model_response.txt')
    response = np.loadtxt(resp_path)
    model = pg.load(join(output_ph,'resistivity.vector'))
    coverage = pg.load(join(output_ph,'resistivity-cov.vector'))

    inv_info_path = join(output_ph,'inv_info.txt')
    Line = []
    section_idx = 0
    with open(inv_info_path, 'r') as read_obj:
        for i,line in enumerate(read_obj):
                Line.append(line.rstrip('\n'))

    final_result = Line[Line.index('## Final result ##')+1:Line.index('## Inversion parameters ##')]
    rrms = float(final_result[0].split(':')[1])
    chi2 = float(final_result[1].split(':')[1])
    inversion_para = Line[Line.index('## Inversion parameters ##')+1:Line.index('## Iteration ##')]
    lam = int(inversion_para[0].split(':')[1])
    iteration = Line[Line.index('## Iteration ##')+2:]
    rrmsHistory = np.zeros(len(iteration))
    chi2History = np.zeros(len(iteration))
    for i in range(len(iteration)):
        rrmsHistory[i] = float(iteration[i].split(',')[1])
        chi2History[i] = float(iteration[i].split(',')[2])

    mgr_dict = {'paraDomain': para_domain, 'mesh_fw': mesh_fw, 'data': data, 
                'response': response, 'model': model, 'coverage': coverage, 
                'investg_depth': investg_depth, 'rrms': rrms, 'chi2': chi2, 'lam': lam,
                'rrmsHistory': rrmsHistory, 'chi2History': chi2History}

    return mgr_dict

read_Layers_water_content = pickle.loads(open('rainfall.pkl', 'rb').read())

for i,csv_file_name in enumerate(csvfiles):
    save_ph = join('Inversion_results',csv_file_name[-7:-4]+'constrained')
    read_Layers_water_content[i]['mgr'] = load_inversion_results(save_ph)
    save_ph = join('Inversion_results', csv_file_name[-7:-4]+'normal')
    read_Layers_water_content[i]['mgr_normal'] = load_inversion_results(save_ph)
    read_Layers_water_content[i]['data'] = read_Layers_water_content[i]['mgr']['data']

# %%
def plot_compare_result(Layers_water_content ,csv_file_name):

    resistivity = Layers_water_content['resistivity']
    print(Layers_water_content['data'])
    data = Layers_water_content['data']
    mgr_normal = Layers_water_content['mgr_normal']
    mgr = Layers_water_content['mgr']


    # Comparesion of the results by the residual profile
    # Re-interpolate the grid
    left = 0
    right = 30
    depth = 5

    mesh_x = np.linspace(left,right,300)
    mesh_y = np.linspace(-depth,20,180)
    grid = pg.createGrid(x=mesh_x,y=mesh_y )
    kw = dict(cMin=min(resistivity), cMax=max(resistivity), logScale=True, cMap='jet',
                        xlabel='X (m)', ylabel='Y (m)', 
                        label=pg.unit('res'), orientation='vertical')
    # Distinguish the region of the mesh and insert the value of rhomap
    rho_grid = pg.interpolate(mesh, resistivity, grid.cellCenters())
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, figsize=(10, 10),
                                                            constrained_layout=True,
                                                            gridspec_kw={'wspace': 0.2})
    fig.suptitle(csv_file_name,fontsize=20, fontweight="bold")
    ax2.axis('off')
    # Subplot 1:Original resistivity model
    pg.viewer.showMesh(mesh, resistivity,ax=ax1, **kw)
    ax1.set_title('Original resistivity model profile',fontweight="bold", size=16)
    ax1.set_xlim([min(geo['x']), max(geo['x'])])
    ax1.set_ylim([min(geo['y']),max(geo['y'])])
    def plot_resistivity(ax, mgr, data, title, **kw):
        pg.viewer.showMesh(mgr['paraDomain'], mgr['model'],coverage=mgr['coverage'],#grid,data=rho_normal_grid,
                        ax=ax, **kw)
        ax.plot(np.array(pg.x(data)), np.array(pg.y(data)),'kv',markersize=3)
        ax.set_title(title,fontweight="bold", size=16)
        ax.set_xlim([min(geo['x']), max(geo['x'])])
        ax.set_ylim([min(geo['y']),max(geo['y'])])
        ax.text(left+1,min(geo['y'])+1,'RRMS: {:.2f}%'.format(
                mgr['rrms'])
                    ,fontweight="bold", size=16)
    # Subplot 3:normal grid 
    rho_normal_grid = pg.interpolate(mgr_normal['paraDomain'], mgr_normal['model'], grid.cellCenters())
    plot_resistivity(ax=ax3, mgr=mgr_normal, data=data, title='Normal mesh inverted resistivity profile', **kw)
    ax3.plot(pg.x(slope.nodes()),pg.y(slope.nodes()),'--k')
    # Subplot 5:structured constrained grid 
    rho_layer_grid = pg.interpolate(mgr['paraDomain'], mgr['model'], grid.cellCenters())
    plot_resistivity(ax=ax5, mgr=mgr, data=data, title='Structured constrained inverted resistivity profile', **kw)
    ax5.plot(pg.x(slope.nodes()),pg.y(slope.nodes()),'-k')
    # Plot the residual profile
    # Calculate the resistivity relative difference
    kw_compare = dict(cMin=-10, cMax=10, cMap='bwr',
                    label='Relative resistivity difference \n(%)')
    # Subplot 4:Normal mesh resistivity residual
    def plot_residual_contour(ax, grid, data, title,mesh_x,mesh_y, **kw_compare):
        class StretchOutNormalize(plt.Normalize):
            def __init__(self, vmin=None, vmax=None, low=None, up=None, clip=False):
                self.low = low
                self.up = up
                plt.Normalize.__init__(self, vmin, vmax, clip)
            def __call__(self, value, clip=None):
                x, y = [self.vmin, self.low, self.up, self.vmax], [0, 0.5-1e-9, 0.5+1e-9, 1]
                return np.ma.masked_array(np.interp(value, x, y))
        clim = [kw_compare['cMin'], kw_compare['cMax']]
        midnorm=StretchOutNormalize(vmin=clim[0], vmax=clim[1], low=-1, up=1)
        X,Y = np.meshgrid(mesh_x,mesh_y)
        diff_pos = pg.interpolate(grid, data, grid.positions())
        mesh = np.reshape(diff_pos,(len(mesh_y),len(mesh_x)))
        ax.contourf(X,Y,mesh,
                    levels = 128,
                    cmap='bwr',
                    norm=midnorm)
        ax.set_title(title, fontweight="bold", size=16)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.plot(pg.x(slope.nodes()),pg.y(slope.nodes())+2,'-k')

        ax.set_xlim([min(geo['x']), max(geo['x'])])
        ax.set_ylim([min(geo['y']),max(geo['y'])])
        ax.set_aspect('equal')
        divider = make_axes_locatable(ax)
        cbaxes = divider.append_axes("right", size="4%", pad=.15)
        m = plt.cm.ScalarMappable(cmap=plt.cm.bwr,norm=midnorm)
        m.set_array(mesh)
        m.set_clim(clim[0],clim[1])
        cb = plt.colorbar(m,
                        boundaries=np.linspace(clim[0],clim[1], 128),
                        cax=cbaxes)
        cb_ytick = np.linspace(clim[0],clim[1],5)
        cb.ax.set_yticks(cb_ytick)
        cb.ax.set_yticklabels(['{:.0f}'.format(x) for x in cb_ytick])
        cb.ax.set_ylabel(kw_compare['label'])
    residual_normal_grid = ((np.log10(rho_normal_grid) - np.log10(rho_grid))/np.log10(rho_grid))*100
    # plot_residual(ax=ax4, grid=grid, data=residual_normal_grid, title='Normal mesh resistivity difference profile', **kw_compare)
    plot_residual_contour(ax=ax4, grid=grid, data=residual_normal_grid, title='Normal mesh resistivity difference profile',mesh_x=mesh_x,mesh_y=mesh_y, **kw_compare)
    ax4.plot(pg.x(slope.nodes()),pg.y(slope.nodes()),'--k')
    # Subplot 6:Layered mesh resistivity residual
    residual_layer_grid = ((np.log10(rho_layer_grid) - np.log10(rho_grid))/np.log10(rho_grid))*100
    # plot_residual(ax=ax6, grid=grid, data=residual_layer_grid, title='Structured constrained resistivity difference profile', **kw_compare)
    plot_residual_contour(ax=ax6, grid=grid, data=residual_layer_grid, title='Structured constrained resistivity difference profile',mesh_x=mesh_x,mesh_y=mesh_y, **kw_compare)
    ax6.plot(pg.x(slope.nodes()),pg.y(slope.nodes()),'-k')

    fig.savefig(join('COMSOL_result',csv_file_name[-7:-4]+'Compare Plot.png'), dpi=300, bbox_inches='tight', transparent=False)
    plt.close()

def plot_SWC_compare_result(Layers_water_content, csv_file_name):

    SWC = Layers_water_content['SWC']
    SWC_normal = Layers_water_content['SWC_normal']
    SWC_real = Layers_water_content['df']['theta'].to_numpy()
    df = Layers_water_content['df']

    # Comparesion of the results by the residual profile
    # Re-interpolate the grid

    x_min, x_max = df['X'].min(), df['X'].max()
    y_min, y_max = df['Y'].min(), df['Y'].max()

    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, figsize=(10, 10),
                                                                constrained_layout=True,
                                                                gridspec_kw={'wspace': 0.2})
    fig.suptitle(csv_file_name,fontsize=20, fontweight="bold")
    ax2.axis('off')

    def plot_SWC_contour(ax,  data, title, grid_x, grid_y):
        grid_theta = griddata((df['X'], df['Y']), data, (grid_x, grid_y), method='linear')
        plot = ax.contourf(grid_x, grid_y, grid_theta, levels=8, cmap='Blues'
                            ,vmin=min(SWC_real),vmax=max(SWC_real))

        terrain_polygon = np.vstack((geo.to_numpy()[2:5], [max(geo['x']), max(geo['y'])], [min(geo['x']), max(geo['y'])]))
        mask_polygon = Polygon(terrain_polygon, closed=True, color='white', zorder=10)

        ax.add_patch(mask_polygon)
        ax.plot(pg.x(slope.nodes()),pg.y(slope.nodes()),'-k',linewidth=2, zorder=11)
        ax.plot(pg.x(slope.nodes()),pg.y(slope.nodes())+2,'-k',linewidth=2, zorder=11)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title, fontweight="bold", size=16)
        ax.set_aspect('equal')
        ax.set_xlim([min(geo['x']), max(geo['x'])])
        ax.set_ylim([min(geo['y']),max(geo['y'])])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2.5%', pad=0.05)
        cbar = fig.colorbar(plot, cax=cax)
        cbar.set_label(r'$\theta$')

        return grid_theta
    # Subplot 1:Original SWC model
    grid_theta_real = plot_SWC_contour(ax=ax1, data=SWC_real, title='Original COMSOL SWC profile',grid_x=grid_x, grid_y=grid_y)
    # Subplot 3:normal grid
    grid_theta_normal = plot_SWC_contour(ax=ax3, data=SWC_normal, title='Normal mesh inverted SWC profile',grid_x=grid_x, grid_y=grid_y)
    # Subplot 5:structured constrained grid
    grid_theta_constrain = plot_SWC_contour(ax=ax5, data=SWC, title='Structured constrained inverted SWC profile',grid_x=grid_x, grid_y=grid_y)

    # Plot the residual profile
    # Calculate the SWC relative difference
    kw_compare = dict(cMin=-50, cMax=50, cMap='bwr',
                    label='SWC relative difference \n(%)')
    def plot_SWC_residual_contour(ax, data, title,grid_x,grid_y, **kw_compare):
        class StretchOutNormalize(plt.Normalize):
            def __init__(self, vmin=None, vmax=None, low=None, up=None, clip=False):
                self.low = low
                self.up = up
                plt.Normalize.__init__(self, vmin, vmax, clip)
            def __call__(self, value, clip=None):
                x, y = [self.vmin, self.low, self.up, self.vmax], [0, 0.5-1e-9, 0.5+1e-9, 1]
                return np.ma.masked_array(np.interp(value, x, y))
        clim = [kw_compare['cMin'], kw_compare['cMax']]
        midnorm=StretchOutNormalize(vmin=clim[0], vmax=clim[1], low=0, up=0)

        ax.contourf(grid_x,grid_y,data,
                    levels = 128,
                    cmap='bwr_r',
                    norm=midnorm)
        ax.set_title(title, fontweight="bold", size=16)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.plot(pg.x(slope.nodes()),pg.y(slope.nodes()),'-k',linewidth=2, zorder=11)
        ax.plot(pg.x(slope.nodes()),pg.y(slope.nodes())+2,'-k',linewidth=2, zorder=11)
        terrain_polygon = np.vstack((geo.to_numpy()[2:5], [max(geo['x']), max(geo['y'])], [min(geo['x']), max(geo['y'])]))
        mask_polygon = Polygon(terrain_polygon, closed=True, color='white', zorder=10)

        ax.add_patch(mask_polygon)

        ax.set_xlim([min(geo['x']), max(geo['x'])])
        ax.set_ylim([min(geo['y']),max(geo['y'])])
        ax.set_aspect('equal')
        divider = make_axes_locatable(ax)
        cbaxes = divider.append_axes("right", size="4%", pad=.15)
        m = plt.cm.ScalarMappable(cmap=plt.cm.bwr,norm=midnorm)
        m.set_array(data)
        m.set_clim(clim[0],clim[1])
        cb = plt.colorbar(m,
                        boundaries=np.linspace(clim[0],clim[1], 128),
                        cax=cbaxes)
        cb_ytick = np.linspace(clim[0],clim[1],5)
        cb.ax.set_yticks(cb_ytick)
        cb.ax.set_yticklabels(['{:.0f}'.format(x) for x in cb_ytick])
        cb.ax.set_ylabel(kw_compare['label'])

    # Subplot 4:Normal mesh SWC residual
    residual_normal_grid = ((grid_theta_normal - grid_theta_real)/grid_theta_real)*100
    plot_SWC_residual_contour(ax=ax4, data=residual_normal_grid, title='Normal mesh SWC difference profile',grid_x=grid_x, grid_y=grid_y, **kw_compare)
    # Subplot 6:Constrained mesh SWC residual
    residual_constrain_grid = ((grid_theta_constrain - grid_theta_real)/grid_theta_real)*100
    plot_SWC_residual_contour(ax=ax6, data=residual_constrain_grid, title='Structured constrained SWC difference profile',grid_x=grid_x, grid_y=grid_y, **kw_compare)
    
    fig.savefig(join('COMSOL_result_SWC',csv_file_name[-7:-4]+'SWC Compare Plot.png'), dpi=300, bbox_inches='tight', transparent=False)

for j,csv_file_name in enumerate(csvfiles):    
    plot_compare_result(read_Layers_water_content[j],csv_file_name)
    plot_SWC_compare_result(read_Layers_water_content[j], csv_file_name)
# %%
Layers_water_content_partial = {}
Layers_water_content = read_Layers_water_content
for j,csv_file_name in enumerate(csvfiles):
    df = import_COMSOL_csv(csv_name = join(csv_path,csv_file_name),plot=False,style='scatter')
    interface_coords = np.array(geo[1:-1])
    interface_coords[:,1]= interface_coords[:,1]-5
    line = LineString(interface_coords)
    SWC_partial = []
    SWC_normal_partial = []
    SWC_real_partial = []
    xy = []

    # 檢查每個點
    for i, point in enumerate(df[['X', 'Y']].to_numpy()):
        point = Point(point)
        distance = line.distance(point)
        # 判斷點相對於折線的位置
        if (point.y > line.interpolate(line.project(point)).y) and (point.x>6) and (point.x<24):
            SWC_partial.append(Layers_water_content[j]['SWC'][i])
            SWC_normal_partial.append(Layers_water_content[j]['SWC_normal'][i])
            SWC_real_partial.append(df['theta'].to_numpy()[i])
            xy.append(df[['X', 'Y']].to_numpy()[i])
            
        
    RRMSE = lambda x, y: np.sqrt(np.sum(((x - y)/y)**2) / len(x)) * 100
    #RRMSE = lambda x, y: 1 - np.sum((x - y)**2) / np.sum((y - np.mean(y))**2) 
    Layers_water_content_partial[j] = {
        'SWC_partial': SWC_partial,
        'SWC_normal_partial': SWC_normal_partial,
        'RRMSE': RRMSE(np.array(SWC_partial), np.array(SWC_real_partial)),
        'RRMSE_normal': RRMSE(np.array(SWC_normal_partial), np.array(SWC_real_partial))
    }

# %%
rain_rate = 3*5 # mm/day
# rain for 10 days
cumulative_rain = [rain_rate * i for i in range(0, 11)]
RRMSE_all = [Layers_water_content_partial[key]['RRMSE'] for key in Layers_water_content_partial]
RRMSE_normal_all = [Layers_water_content_partial[key]['RRMSE_normal'] for key in Layers_water_content_partial]
fig, ax = plt.subplots(figsize=(6.4, 4.8))
ax.plot(cumulative_rain,RRMSE_all,'-bo', label='Structured constrained mesh')
ax.hlines(y=np.mean(RRMSE_all), xmin=min(cumulative_rain), xmax=max(cumulative_rain), 
          colors='b', linestyles='--', label='Mean RRMSE')
ax.plot(cumulative_rain,RRMSE_normal_all,'-ro', label='Normal mesh')
ax.hlines(y=np.mean(RRMSE_normal_all), xmin=min(cumulative_rain), xmax=max(cumulative_rain), 
          colors='r', linestyles='--', label='Mean RRMSE')
ax.set_xlabel('Cumulative infiltration (mm)')
ax.set_ylabel('Soil Water Content RRMSE (%)')
ax.set_title('RRMSE between True SWC and Inverted SWC')
ax.set_xticks(cumulative_rain)
ax.legend(bbox_to_anchor=(1, 1))
ax.grid(linestyle='--', alpha=0.5)
fig.savefig(join('results', 'RRMSE_SWC.png'), dpi=300, bbox_inches='tight')
# %%
for j,csv_file_name in enumerate(csvfiles):
    df = import_COMSOL_csv(csv_name = join(csv_path,csv_file_name),plot=False,style='scatter')
    df['theta_constrain_inverted'] = read_Layers_water_content[j]['SWC']
    df['theta_normal_inverted'] = read_Layers_water_content[j]['SWC_normal']
    df['pressure_head_constrain'] = read_Layers_water_content[j]['Hp']
    df['pressure_head_normal'] = read_Layers_water_content[j]['Hp_normal']

    df.to_csv(join('To COMSOL csv','inverted_'+csv_file_name), index=False)


# # %%
# fig,ax = plt.subplots(figsize=(6.4, 4.8))
# scatter = ax.scatter(x=np.array(xy)[:,0],y=np.array(xy)[:,1], c=SWC_real_partial#Layers_water_content_partial[0]['SWC_normal_partial']
#                      , marker='o',s=1,cmap='Blues',
#                      vmin=min(df['theta']),vmax=max(df['theta'])
#                      )
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_title(r'Scatter Plot To COMSOL')
# ax.set_aspect('equal')
# ax.set_xlim([0, 30])
# ax.set_ylim([5,20])
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='2.5%', pad=0.05)
# cbar = fig.colorbar(scatter, cax=cax)
# # cbar.set_label(r'$mark$')
# # %%
# # Hp_normal = convert_resistivity_to_Hp(df, mgr_normal.model, mgr_normal.paraDomain, interface_coords)
# fig,ax = plt.subplots(figsize=(6.4, 4.8))
# scatter = ax.scatter(df['X'], df['Y'], c=Hp, marker='o',s=1,cmap='Blues',
#                      vmin=-18,vmax=0
#                      )
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_title(r'$H_p$ Scatter Plot To COMSOL')
# ax.set_aspect('equal')
# ax.set_xlim([0, 30])
# ax.set_ylim([5,20])
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='2.5%', pad=0.05)
# cbar = fig.colorbar(scatter, cax=cax)
# cbar.set_label(r'$m$')
# fig.savefig(join('results','Head Scatter Plot To COMSOL.png'),dpi=300,bbox_inches='tight')

# fig,ax = plt.subplots(figsize=(6.4, 4.8))
# scatter = ax.scatter(df['X'], df['Y'], c=SWC, marker='o',s=1,cmap='Blues',
#                      vmin=min(df['theta']),vmax=max(df['theta'])
#                      )
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_title(r'$\theta$ Scatter Plot To COMSOL')
# ax.set_aspect('equal')
# ax.set_xlim([0, 30])
# ax.set_ylim([5,20])
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='2.5%', pad=0.05)
# cbar = fig.colorbar(scatter, cax=cax)
# cbar.set_label(r'$\theta$')
# fig.savefig(join('results','SWC Scatter Plot To COMSOL.png'),dpi=300,bbox_inches='tight')
# %%
# plt.scatter(df['X'], df['Y'], c=comsol_water_content,s=1, cmap='jet_r')
# plt.colorbar()
df['water_content'] = SWC
df['pressure_head'] = Hp
df.to_csv('TO_comsol.csv', index=False,columns=['X', 'Y', 'water_content', 'pressure_head'])
# %%
