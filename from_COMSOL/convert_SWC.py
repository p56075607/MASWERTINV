# %%
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
from scipy.interpolate import griddata, NearestNDInterpolator
from matplotlib.gridspec import GridSpec

csv_name = 'water content.csv'
df = pd.read_csv(csv_name, comment='%', header=None)
df.columns = ['X', 'Y', 'theta']
n = 2
cFluid = 0.03
resistivity = 1/(cFluid*df['theta']**n)

geo = pd.read_csv('geometry.csv', header=None)
geo.columns = ['x', 'y']
# %%
# electrode_x = np.linspace(start=geo['x'].loc[2], stop=geo['x'].loc[3], num=25)
# electrode_y = np.linspace(start=geo['y'].loc[2], stop=geo['y'].loc[3], num=25)
# 定義折線座標
line_coords = np.array(geo[1:-1])
def create_electrode_coords(line_coords):
    # 計算每段的距離和總距離
    distances = np.sqrt(np.sum(np.diff(line_coords, axis=0)**2, axis=1))
    total_distance = np.sum(distances)

    # 生成等距點
    num_points = int(total_distance) + 1
    electrode_distances = np.linspace(0, total_distance, num=num_points)

    # 計算等距點的座標
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

    # 繪製折線和電極座標
    plt.plot(line_coords[:, 0], line_coords[:, 1], 'k-', label='Terrain Line')
    plt.plot(electrode_coords[:, 0], electrode_coords[:, 1], 'ro', label='electrode')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    # 輸出電極座標
    electrode_x, electrode_y = electrode_coords[:, 0], electrode_coords[:, 1]
    print(f"Electrode X coordinates: {electrode_x}")
    print(f"Electrode Y coordinates: {electrode_y}")
    return electrode_x, electrode_y

electrode_x, electrode_y = create_electrode_coords(line_coords)
scheme = ert.createData(elecs=np.column_stack((electrode_x, electrode_y)),
                           schemeName=None)
# %%
plc = mt.createParaMeshPLC(scheme,paraDepth=5,paraMaxCellSize=0.1
                           )
   
mesh = mt.createMesh(plc)
ax,_ = pg.show(mesh,markers=True,showMesh=True)
# ax.set_xlim([0, 30])
# ax.set_ylim([5,20])
# %%
grid_resistivity = griddata((df[['X', 'Y']].to_numpy()), resistivity, 
                  (np.array(mesh.cellCenters())[:, :2]), method='linear', fill_value=np.nan)

fill_value = np.nanmedian(grid_resistivity)
grid_resistivity = np.nan_to_num(grid_resistivity, nan=fill_value)
# %%
pg.show(mesh, data=grid_resistivity, label='resistivity',cMap='jet',cMin=272,cMax=1350)

# %%
def show_simulation(data):
    pg.info(np.linalg.norm(data['err']), np.linalg.norm(data['rhoa']))
    pg.info('Simulated data', data)
    pg.info('The data contains:', data.dataMap().keys())
    pg.info('Simulated rhoa (min/max)', min(data['rhoa']), max(data['rhoa']))
    pg.info('Selected data noise %(min/max)', min(data['err'])*100, max(data['err'])*100)

def combine_array(schemeName='dd'):
    if (len(schemeName) == 1):
        data = ert.simulate(mesh = mesh, res=grid_resistivity,
                    scheme=ert.createData(elecs=np.column_stack((electrode_x, electrode_y)),
                           schemeName=schemeName),
                    noiseLevel=1, noiseAbs=1e-6, seed=1337)
        
    elif(len(schemeName) > 1):
        data = data = ert.simulate(mesh = mesh, res=grid_resistivity,
                    scheme=ert.createData(elecs=np.column_stack((electrode_x, electrode_y)),
                           schemeName=schemeName[0]),
                    noiseLevel=1, noiseAbs=1e-6, seed=1337)
        for i in range(1,len(schemeName)):
            print('Simulating', schemeName[i])
            data.add(ert.simulate(mesh = mesh, res=grid_resistivity,
                    scheme=ert.createData(elecs=np.column_stack((electrode_x, electrode_y)),
                           schemeName=schemeName[i]),
                    noiseLevel=1, noiseAbs=1e-6, seed=1337))

    show_simulation(data)
    return data
    
data = combine_array(schemeName=['dd','wa','wb','slm'])

ert.showData(data)
# %%
mgr = ert.ERTManager()
mgr.invert(data=data, mesh=mesh, lam=100,verbose=True)
mgr.showResultAndFit()

# %%
# Plot resistivity model results
kw = dict(cMin=272, cMax=1350, logScale=True, cMap='jet',
          xlabel='Distance (m)', ylabel='Depth (m)', 
          label=pg.unit('res'), orientation='vertical')
# 創建圖形和子圖，使用 GridSpec 進行布局管理
fig = plt.figure(figsize=(20, 5), constrained_layout=True)
gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.1)

# 子圖1：展示 mgr.paraDomain 和 mgr.model
ax1 = fig.add_subplot(gs[0, 1])
# 替換這行為正確的 mgr.paraDomain 和 mgr.model 數據
pg.show(mgr.paraDomain, mgr.model, ax=ax1, **kw)
ax1.plot(np.array(pg.x(data)), np.array(pg.y(data)), 'wv', markersize=5)
ax1.set_xlim([0, 30])

# 子圖2：展示 mesh 和 grid_resistivity
ax2 = fig.add_subplot(gs[0, 0])
# 替換這行為正確的 mesh 和 grid_resistivity 數據
pg.show(mesh, data=grid_resistivity, ax=ax2, **kw)
ax2.set_xlim(ax1.get_xlim())
ax2.set_ylim(ax1.get_ylim())

# %%
# Plot resistivity model difference results
kw_compare = dict(cMin=-50, cMax=50, cMap='bwr',
                  label='Relative resistivity difference (%)',
                  xlabel='Distance (m)', ylabel='Depth (m)', orientation='vertical')
inverison_mesh = mgr.paraDomain
inverison_values = np.array(mgr.model)
selected_res = []
for i in range(mesh.cellCount()):
    if mesh.cellMarkers()[i] == 2:
        selected_res.append(grid_resistivity[i])

diff = ((inverison_values-selected_res)/selected_res)*100
fig = plt.figure(figsize=(10, 5), constrained_layout=True)
ax,_ = pg.show(inverison_mesh, data=diff, **kw_compare)
ax.set_xlim([0, 30])
# %%


water_content = (1/(inverison_values*cFluid))**(1/n)
# %%
comsol_water_content = griddata((np.array(inverison_mesh.cellCenters())[:, :2]), water_content, 
                            (df[['X', 'Y']].to_numpy()), method='linear')

# 處理插值範圍外的值：使用 NearestNDInterpolator
nan_indices = np.isnan(comsol_water_content)
if nan_indices.any():
    nearest_interp = NearestNDInterpolator((np.array(inverison_mesh.cellCenters())[:, :2]), water_content)
    comsol_water_content[nan_indices] = nearest_interp(df['X'].to_numpy()[nan_indices], df['Y'].to_numpy()[nan_indices])

plt.scatter(df['X'], df['Y'], c=comsol_water_content,s=1, cmap='jet_r')
plt.colorbar()
df['water_content'] = comsol_water_content
df.to_csv('water_content_TO_comsol.csv', index=False,columns=['X', 'Y', 'water_content'])