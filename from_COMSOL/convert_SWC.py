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

def import_geometry_csv():
    geo = pd.read_csv('geometry.csv', header=None)
    geo.columns = ['x', 'y']

    return geo
geo = import_geometry_csv()

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
        ax.set_xlim([0, 30])
        ax.set_ylim([5,20])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2.5%', pad=0.05)
        cbar = fig.colorbar(plot, cax=cax)
        cbar.set_label(r'$\theta$')
        fig.savefig(join('results',csv_name[-7:-4]+'SWC Plot from COMSOL.png'),dpi=300,bbox_inches='tight')

    return df

def Forward_inversion(csv_file_name,geo=geo,plot=True):
    df = import_COMSOL_csv(csv_name = csv_file_name,plot=True,style='scatter')

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
            ax.set_xlim([0, 30])
            ax.set_ylim([5,20])
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
        return 0  # 飽和??態?U?瑰ㄓO?蘉Y設置?偎s
    m = 1 - 1/n
    func = lambda h: theta_r + (theta_s - theta_r) / (1 + (alpha * np.abs(h))**n)**m - theta
    
    # 使用?h?茠鴝l猜測?�來提�?穩健性
    initial_guesses = [-1, -10, -100, -1000]
    for h_guess in initial_guesses:
        sol = root(func, h_guess)
        if sol.success:
            return sol.x[0]
    
    raise ValueError(f"Solution not found for theta = {theta}")

def convert_resistivity_to_Hp(df, resistivity, mesh, interface_coords):
    grid_resistivity = griddata((np.array(mesh.cellCenters())[:, :2]), resistivity, 
                                (df[['X', 'Y']].to_numpy()), method='linear', fill_value=np.nan)
    fill_value = np.nanmedian(grid_resistivity)
    grid_resistivity = np.nan_to_num(grid_resistivity, nan=fill_value)
    line = LineString(interface_coords)
    SWC = np.zeros(len(df))
    Hp = np.zeros(len(df))
    # 檢查每?蚋I
    for i, point in enumerate(df[['X', 'Y']].to_numpy()):
        point = Point(point)
        distance = line.distance(point)
        # ?P斷點?蛫鴭顜朣u?漲鼽m
        if point.y > line.interpolate(line.project(point)).y:
            n = 1.83
            cFluid = 1/(0.57*106)
            SWC[i] = (1/(grid_resistivity[i]*cFluid))**(1/n)

            # [Soil] Van Genuchten 模??參數
            theta_r = 0.034  # 殘餘?t?艨q
            theta_s = 0.46  # 飽和?t?艨q
            alpha = 1.6     # 經驗參數
            n = 1.37       # 經驗參數

            Hp[i] = van_genuchten_inv(SWC[i], theta_r, theta_s, alpha, n)

        else:
            n = 1.34
            cFluid = 1/(0.58*75)
            SWC[i] = (1/(grid_resistivity[i]*cFluid))**(1/n)

            # [Rock] Van Genuchten 模??參數
            theta_r = 0.031  # 殘餘?t?艨q
            theta_s = 0.467  # 飽和?t?艨q
            alpha = 3.64     # 經驗參數
            n = 1.121       # 經驗參數

            Hp[i] = van_genuchten_inv(SWC[i], theta_r, theta_s, alpha, n)


    return Hp, SWC
# %%
csv_path = 'SWC0531'
csvfiles = [_ for _ in listdir(csv_path) if _.endswith('.csv')]

Layers_water_content = {}
for i,csv_file_name in enumerate(csvfiles):
    print('Processing', csv_file_name)
    mgr, mgr_normal, data, resistivity,df = Forward_inversion(join(csv_path,csv_file_name),geo=geo,plot=True)
    Hp, SWC= convert_resistivity_to_Hp(df, mgr.model, mgr.paraDomain, interface_coords)
    Hp_normal, SWC_normal = convert_resistivity_to_Hp(df, mgr_normal.model, mgr_normal.paraDomain, interface_coords)

    SWC_real = df['theta'].to_numpy()

    RRMSE = lambda x, y: np.sqrt(np.sum(((x - y)/y)**2) / len(x)) * 100
    Layers_water_content[i] = {
        'SWC': SWC,
        'SWC_normal': SWC_normal,
        'RRMSE': RRMSE(SWC, SWC_real),
        'RRMSE_normal': RRMSE(SWC_normal, SWC_real)
    }
    print('RRMSE of SWC:', Layers_water_content[i]['RRMSE'])
    print('RRMSE of SWC normal:', Layers_water_content[i]['RRMSE_normal'])
# %%
rain_rate = 365 # mm/day
# rain for 10 days
cumulative_rain = [rain_rate * i for i in range(1, 12)]
RRMSE_all = [Layers_water_content[key]['RRMSE'] for key in Layers_water_content]
RRMSE_all.append(RRMSE_all.pop(1))
RRMSE_normal_all = [Layers_water_content[key]['RRMSE_normal'] for key in Layers_water_content]
RRMSE_normal_all.append(RRMSE_normal_all.pop(1))
fig, ax = plt.subplots(figsize=(6.4, 4.8))
ax.plot(cumulative_rain,RRMSE_all,'-bo', label='Structured constrained mesh')
ax.plot(cumulative_rain,RRMSE_normal_all,'-ro', label='Normal mesh')
ax.set_xlabel('Cumulative infiltration (mm)')
ax.set_ylabel('Soil Water Content RRMSE (%)')
ax.set_title('RRMSE between True SWC and Inverted SWC')
ax.legend()
fig.savefig(join('results', 'RRMSE_SWC.png'), dpi=300, bbox_inches='tight')
# %%
# Comparesion of the results by the residual profile
# Re-interpolate the grid
left = 0
right = 30
depth = 5

mesh_x = np.linspace(left,right,300)
mesh_y = np.linspace(-depth,20,180)
grid = pg.createGrid(x=mesh_x,y=mesh_y )

# Distinguish the region of the mesh and insert the value of rhomap
rho_grid = pg.interpolate(mesh, resistivity, grid.cellCenters())
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, figsize=(10, 10),
                                                         constrained_layout=True,
                                                         gridspec_kw={'wspace': 0.2})
ax2.axis('off')
# Subplot 1:Original resistivity model
pg.viewer.showMesh(mesh, resistivity,ax=ax1, **kw)
ax1.set_title('Original resistivity model profile',fontweight="bold", size=16)
ax1.set_xlim([0, 30])
ax1.set_ylim([5,20])
def plot_resistivity(ax, mgr, data, title, **kw):
    pg.viewer.showMesh(mgr.paraDomain, mgr.model,coverage=mgr.coverage(),#grid,data=rho_normal_grid,
                    ax=ax, **kw)
    ax.plot(np.array(pg.x(data)), np.array(pg.z(data)),'ko')
    ax.set_title(title,fontweight="bold", size=16)
    ax.set_xlim([0, 30])
    ax.set_ylim([5,20])
    ax.text(left+1,6,'RRMS: {:.2f}%'.format(
            mgr.inv.relrms())
                ,fontweight="bold", size=16)
# Subplot 3:normal grid 
rho_normal_grid = pg.interpolate(mgr_normal.paraDomain, mgr_normal.model, grid.cellCenters())
plot_resistivity(ax=ax3, mgr=mgr_normal, data=data, title='Normal mesh inverted resistivity profile', **kw)
ax3.plot(pg.x(slope.nodes()),pg.y(slope.nodes()),'--k')
# Subplot 5:structured constrained grid 
rho_layer_grid = pg.interpolate(mgr.paraDomain, mgr.model, grid.cellCenters())
plot_resistivity(ax=ax5, mgr=mgr, data=data, title='Structured constrained inverted resistivity profile', **kw)
ax5.plot(pg.x(slope.nodes()),pg.y(slope.nodes()),'-k')
# Plot the residual profile
# Calculate the resistivity relative difference
kw_compare = dict(cMin=-50, cMax=50, cMap='bwr',
                label='Relative resistivity difference \n(%)',
                xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical')
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
    clim = [-50, 50]
    midnorm=StretchOutNormalize(vmin=clim[0], vmax=clim[1], low=-10, up=10)
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

    ax.set_xlim([0, 30])
    ax.set_ylim([5,20])
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
    cb.ax.set_ylabel('Relative resistivity difference\n(%)')
residual_normal_grid = ((rho_normal_grid - rho_grid)/rho_grid)*100
# plot_residual(ax=ax4, grid=grid, data=residual_normal_grid, title='Normal mesh resistivity difference profile', **kw_compare)
plot_residual_contour(ax=ax4, grid=grid, data=residual_normal_grid, title='Normal mesh resistivity difference profile',mesh_x=mesh_x,mesh_y=mesh_y, **kw_compare)
ax4.plot(pg.x(slope.nodes()),pg.y(slope.nodes()),'--k')
# Subplot 6:Layered mesh resistivity residual
residual_layer_grid = ((rho_layer_grid - rho_grid)/rho_grid)*100
# plot_residual(ax=ax6, grid=grid, data=residual_layer_grid, title='Structured constrained resistivity difference profile', **kw_compare)
plot_residual_contour(ax=ax6, grid=grid, data=residual_layer_grid, title='Structured constrained resistivity difference profile',mesh_x=mesh_x,mesh_y=mesh_y, **kw_compare)
ax6.plot(pg.x(slope.nodes()),pg.y(slope.nodes()),'-k')

fig.savefig(join('results','Compare.png'), dpi=300, bbox_inches='tight', transparent=False)

# %%
fig,ax = plt.subplots(figsize=(6.4, 4.8))
scatter = ax.scatter(df['X'], df['Y'], c=mark, marker='o',s=1,cmap='Blues',
                     vmin=-1,vmax=1
                     )
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(r'$H_p$ Scatter Plot To COMSOL')
ax.set_aspect('equal')
ax.set_xlim([0, 30])
ax.set_ylim([5,20])
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2.5%', pad=0.05)
cbar = fig.colorbar(scatter, cax=cax)
cbar.set_label(r'$mark$')
# %%
# Hp_normal = convert_resistivity_to_Hp(df, mgr_normal.model, mgr_normal.paraDomain, interface_coords)
fig,ax = plt.subplots(figsize=(6.4, 4.8))
scatter = ax.scatter(df['X'], df['Y'], c=Hp, marker='o',s=1,cmap='Blues',
                     vmin=-18,vmax=0
                     )
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(r'$H_p$ Scatter Plot To COMSOL')
ax.set_aspect('equal')
ax.set_xlim([0, 30])
ax.set_ylim([5,20])
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2.5%', pad=0.05)
cbar = fig.colorbar(scatter, cax=cax)
cbar.set_label(r'$m$')
fig.savefig(join('results','Head Scatter Plot To COMSOL.png'),dpi=300,bbox_inches='tight')

fig,ax = plt.subplots(figsize=(6.4, 4.8))
scatter = ax.scatter(df['X'], df['Y'], c=SWC, marker='o',s=1,cmap='Blues',
                     vmin=min(df['theta']),vmax=max(df['theta'])
                     )
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(r'$\theta$ Scatter Plot To COMSOL')
ax.set_aspect('equal')
ax.set_xlim([0, 30])
ax.set_ylim([5,20])
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='2.5%', pad=0.05)
cbar = fig.colorbar(scatter, cax=cax)
cbar.set_label(r'$\theta$')
fig.savefig(join('results','SWC Scatter Plot To COMSOL.png'),dpi=300,bbox_inches='tight')
# %%
# plt.scatter(df['X'], df['Y'], c=comsol_water_content,s=1, cmap='jet_r')
# plt.colorbar()
df['water_content'] = SWC
df['pressure_head'] = Hp
df.to_csv('TO_comsol.csv', index=False,columns=['X', 'Y', 'water_content', 'pressure_head'])
# %%
