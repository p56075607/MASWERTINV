# %%
import numpy as np
import pygimli as pg
from pygimli.physics import ert

data=ert.load("data/NWG-1M.txt")
print(data)

data['k'] = ert.createGeometricFactors(data)
print(data)

k0 = ert.createGeometricFactors(data)

ert.showData(data, vals=k0/data['k'], label='Topography effect')

mgr = ert.ERTManager(sr=False)

mgr.checkData(data)
data.remove(data['rhoa'] < 0)
ert.showData(data)
print(data)

data['err'] = ert.estimateError(data, absoluteUError=5e-5, relativeError=0.03)
print(data)
ert.show(data, data['err']*100)

inv = mgr.invert(data, lam=20, verbose=True, quality=33.6)
#mgr.showResult()
mgr.showResult(cMin=6, cMax=10000, logScale=True, coverage=mgr.standardizedCoverage(-2))