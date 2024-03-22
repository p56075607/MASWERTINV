# %%
import matplotlib.pyplot as plt
import numpy as np

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
world = mt.createWorld(start=[-50, 0], end=[50, -50], 
                       worldMarker=True)
geom = world
pg.show(geom)
scheme = ert.createData(elecs=np.linspace(start=-15, stop=15, num=21),
                           schemeName='wa')
# Create a mesh for the finite element modelling with appropriate mesh quality.
mesh = mt.createMesh(geom, quality=34,markers=True, area=0.2)
rhomap = [[1, 100.]]
pg.show(mesh, data=rhomap, label=pg.unit('res'), showMesh=True)
data = ert.simulate(mesh, scheme=scheme, res=rhomap, noiseLevel=0.01, noiseAbs=1e-6, seed=1337)
pg.info(np.linalg.norm(data['err']), np.linalg.norm(data['rhoa']))
pg.info('Simulated data', data)
pg.info('The data contains:', data.dataMap().keys())

pg.info('Simulated rhoa (min/max)', min(data['rhoa']), max(data['rhoa']))
pg.info('Selected data noise %(min/max)', min(data['err'])*100, max(data['err'])*100)
# %%
plt.scatter(np.arange(len(data['rhoa'])),data['rhoa'],s=1)
data.save('data.dat')
# %%
mgr = ert.ERTManager(data)
inv = mgr.invert(lam=20, verbose=True)
mgr.showResultAndFit()