# %%
import pygimli as pg
import pygimli.meshtools as mt
import numpy as np
import matplotlib.pyplot as plt
# Create a mesh with 3 layers and an outer region for extrapolation
layers = mt.createWorld([0,-50],[100,0], layers=[-15,-35])
inner = mt.createMesh(layers, area=3)
mesh = mt.appendTriangleBoundary(inner, xbound=120, ybound=50,
                                 area=20, marker=0)
# Create data for the inner region only
layer_vals = [20,30,50]
data = np.array(layer_vals)[inner.cellMarkers() - 1]
# The following fails since len(data) != mesh.cellCount(), extrapolate
pg.show(mesh, data)
# %%
# Create data vector, where zeros fill the outer region
data_with_outer = np.array([0] + layer_vals)[mesh.cellMarkers()]
# Actual extrapolation
extrapolated_data = mt.fillEmptyToCellArray(mesh,
                                 data_with_outer, slope=False)
extrapolated_data_with_slope = mt.fillEmptyToCellArray(mesh,
                                data_with_outer, slope=True)
# Visualization
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,8), sharey=True)
_ = pg.show(mesh, data_with_outer, ax=ax1, cMin=0)
_ = pg.show(mesh, extrapolated_data, ax=ax2, cMin=0)
_ = pg.show(mesh, extrapolated_data_with_slope, ax=ax3, cMin=0)
_ = ax1.set_title("Original data")
_ = ax2.set_title("Extrapolated with slope=False")
_ = ax3.set_title("Extrapolated with slope=True")