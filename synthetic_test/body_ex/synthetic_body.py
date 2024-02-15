# %%
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:00:41 2023

@author: Taufiq Rafie
"""

import matplotlib.pyplot as plt
import numpy as np

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.viewer import showMesh
from pygimli.physics import ert


# Geometry Definition

world = mt.createWorld(start=[0,-70], end=[350,0],
                      worldMarker=True)

#pg.show(world)

#dip = mt.createLine(start=[0,-40], end=[100,-25])

#pg.show(dip)

#geom = geom + dip

body = mt.createPolygon([(0,-20),(100,-20),(125,-10),(225,-10),(250,-20),(350,-20),
                         (350,-30),(250,-30),(225,-40),(125,-40),(100,-30),(0,-30)], 
                        isClosed=True, marker=2)
#pg.show(body, showNodes=True)

geo = world + body
pg.show(geo)

#mesh = mt.createMesh(geo, area=1.0, quality=33)
#showMesh(mesh, markers=True, showMesh=True)
#pg.show(mesh)

#%%

# Synthetic data generation

scheme = ert.createData(elecs=np.linspace(start=5, stop=345, num=69), 
                        schemeName='gr')

for p in scheme.sensors():
    geo.createNode(p)
    geo.createNode(p - [0, 0.1])

mesh = mt.createMesh(geo, area=10, quality=34)

rhomap =[[0, 100],[1,1000],[2,10]]
pg.show(mesh, data=rhomap, label=pg.unit('res'), showMesh=True)

data = ert.simulate(mesh, scheme=scheme, res=rhomap, noiseLevel=0.1,
                    seed=42)

pg.info(np.linalg.norm(data['err']), np.linalg.norm(data['rhoa']))
pg.info('Simulated data', data)
pg.info('The data contains:', data.dataMap().keys())

pg.info('Simulated rhoa (min/max)', min(data['rhoa']), max(data['rhoa']))
pg.info('Selected data noise %(min/max)', min(data['err'])*100, max(data['err'])*100)

data.remove(data['rhoa']<0)
pg.info('Filtered rhoa(min/max)', min(data['rhoa']), max(data['rhoa']))

data.save('synthetic_body.dat')
ert.show(data)

#%%
# DC Inversion

mgr = ert.Manager(data)

inv = mgr.invert(mesh=mesh, lam=20, verbose=True, paraDepth=70, paraMaxCellSize=5)
#np.testing.assert_approx_equal(mgr.inv.chi2(), 0.7, significant=1)
mgr.showMisfit()

mgr.showResultAndFit(cMin=10, cMax=1000)
mgr.showResult(cMin=10, cMax=1000)

# %%
# Comparesion of the results
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,7))
pg.show(mesh, rhomap, ax=ax1, hold=True, cMap="jet", logScale=True, label='Resistivity ($\Omega$m)',
        orientation="vertical", cMin=10, cMax=1000)
mgr.showResult(ax=ax2, cMap="jet", cMin=10, cMax=1000, orientation="vertical",coverage=None)