# %%
import numpy as np

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics.petro import resistivityArchie

m1 = pg.load('datamesh_pointcloud_border.bms')
print(m1)
m2 = pg.load('meshERT.bms')
print(m2)
r1 = pg.load('rFluid.vector')
print(r1)

pg.show(m1, r1, label='r1')

r2 = pg.interpolate(m1, r1, m2.cellCenters())
pg.show(m2, r2, label='r2')

a1 = resistivityArchie(rFluid=np.array(r1), porosity=0.3, m=1.8, mesh=m1)
pg.show(m1, a1, label='a1')

a2 = pg.interpolate(m1, a1, m2.cellCenters())
pg.show(m2, a2, label='a2')

# this is a very very simple extrapolation using simple value prolongation
a3 = mt.fillEmptyToCellArray(m2, a2, slope=True)
pg.show(m2, a3, label='a3')

# %%
resBulk = resistivityArchie(r1, porosity=0.3, m=1.8, mesh = m1, meshI = m2, fill=1) 
print(resBulk)
pg.show(m2,resBulk,colorBar=True,showMesh = True)