# %%
#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Incorporating prior data into ERT inversion
===========================================
Prior data can often help to overcome ambiguity in the inversion process.
Here we demonstrate the use of direct push (DP) data in an ERT inversion of
data collected at the surface.
"""
# sphinx_gallery_thumbnail_number = 5

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pygimli as pg
from pygimli.physics import ert
from pygimli.frameworks import PriorModelling, JointModelling
from pygimli.viewer.mpl import draw1DColumn
plt.rcParams["font.family"] = "Microsoft Sans Serif"
# %%%
# The prior data
# --------------
#
# This field data is from a site with layered sands and clays over a
# resistive bedrock. We load it from the example repository.
#
# As a position of x=155m (center of the profile) we have a
# borehole/direct push with known in-situ-data. We load the three-column
# file using numpy.
#

x, z, r = pg.getExampleData("ert/bedrock.txt").T
fig, ax = plt.subplots()
ax.semilogx(r, z, "*-")
ax.set_xlabel(r"$\rho$ ($\Omega$m)")
ax.set_ylabel("depth (m)")
ax.grid(True)

# %%%
# We mainly see four layers: 1. a conductive (clayey) overburden of about
# 17m thickness, 2. a medium resistivity interbedding of silt and sand,
# about 7m thick 3. again clay with 8m thickness 4. the resistive bedrock
# with a few hundred :math:`\Omega`m
#

# %%%
# The ERT data
# ------------
# We load the ERT data from the example repository and plot the pseudosection.
#

data = pg.getExampleData("ert/bedrock.dat")
print(data)
ax, cb = ert.show(data)

# %%%
# The apparent resistivities show increasing values with larger spacings
# with no observable noise. We first compute geometric factors and
# estimate an error model using rather low values for both error parts.
#

data["k"] = ert.geometricFactors(data)
data["err"] = ert.estimateError(data, relativeError=0.025, absoluteUError=100e-6)

# %%%
# We create an ERT manager and invert the data, already using a rather low value for
# the vertical smoothness to account for the layered model.
#
mgr = ert.ERTManager(data, verbose=True)
mgr.invert(paraDepth=70, quality=34.4, paraMaxCellSize=500, zWeight=0.1, lam=30)

# %%%
# For reasons of comparability, we define a unique colormap and store all
# options in a dictionary to be used in subsequent show commands.
#
# We plot the result with these and plot the DP points onto the mesh.
#

kw = dict(cMin=10, cMax=500, logScale=True, cMap="jet",
          xlabel="x (m)", ylabel="y (m)")
ax, cb = mgr.showResult(coverage = None,**kw)
zz = np.abs(z)
iz = np.argsort(z)
dz = np.diff(zz[iz])
thk = np.hstack([dz, dz[-1]])
ztop = -zz[iz[0]]-dz[0]/2
colkw = dict(x=x[0], val=r[iz], thk=thk, width=4, ztopo=ztop)
draw1DColumn(ax, **colkw, **kw)
ax.grid(linestyle='--',linewidth=0.5)


# %%%
# We want to extract the resistivity from the mesh at the positions where
# the prior data are available. To this end, we create a list of positions
# (``pg.Pos`` class) and use a forward operator that picks the values from a
# model vector according to the cell where the position is in. See the
# regularization tutorial for details about that.
#

posVec = [pg.Pos(pos) for pos in zip(x, z)]
para = pg.Mesh(mgr.paraDomain)  # make a copy
para.setCellMarkers(pg.IVector(para.cellCount()))
fopDP = PriorModelling(para, posVec)

# %%%
# We can now use it to retrieve the model values, store it and plot it along
# with the measured values.
#

fig, ax = plt.subplots()
ax.semilogx(r, z, label="borehole")
resSmooth = fopDP(mgr.model)
ax.semilogx(list(resSmooth), z, label="ERT")
ax.set_xlabel(r"$\rho$ ($\Omega$m)")
ax.set_ylabel("depth (m)")
ax.grid(linestyle='--',linewidth=0.5)
ax.legend()

# %%%
# The anisotropic regularization starts to see the good conductor, but only
# the geostatistical regularization operator is able to retrieve values that
# are close to the direct push. Both show the conductor too deep.
#
# One alternative could be to use the interfaces as structural constraints in
# the neighborhood of the borehole. See ERT with structural constraints example
#
# As the DP data is not only good for comparison, we want to use its values as
# data in inversion. This is easily accomplished by taking the mapping operator
# that we already use for interpolation as a forward operator.
#
# We set up an inversion with this mesh, logarithmic transformations and
# invert the model.
#

inv = pg.Inversion(fop=fopDP, verbose=True)
inv.mesh = para
tLog = pg.trans.TransLog()
inv.modelTrans = tLog
inv.dataTrans = tLog
inv.setRegularization(correlationLengths=[40, 4])

model_layer = inv.run(r, errorVals = 0.03*np.ones(len(r)))
ax, cb = pg.show(para, model_layer, **kw)
draw1DColumn(ax, **colkw,**kw)

# %%%
# Apparently, the geostatistical operator can be used to extrapolate
# values with given assumptions.
#

# %%%
# Joint inversion of ERT and DP data
# ----------------------------------
#
# We now use the framework ``JointModelling`` to combine the ERT and the
# DP forward operators. So we set up a new ERT modelling operator and join
# it with ``fopDP``.
#

# fopERT = ert.ERTModelling()
# fopERT.setMesh(mesh)
# fopERT.setData(data) # not necessary as done by JointModelling
# fopJoint = JointModelling([fopERT, fopDP])
# fopJoint.setMesh(para)


fopJoint = JointModelling([mgr.fop, fopDP])
fopJoint.setData([data, pg.Vector(r)])  # needs to have .size() attribute!





# %%%
# We first test the joint forward operator. We create a modelling vector
# of constant resistivity and distribute the model response into the two
# parts that can be looked at individually.
#

model = pg.Vector(para.cellCount(), 100)
response = fopJoint(model)
respERT = response[:data.size()]
respDP = response[data.size():]
print(respDP)

# %%%
# The jacobian can be created and looked up by
#

fopJoint.createJacobian(model_layer)  # works
J = fopJoint.jacobian()
# %%
print(type(J))  # wrong type
ax, cb = pg.show(J,orientation = 'vertical')
ax.set_ylabel('Number of data',fontsize=12)
ax.set_title('Number of model')
cb.ax.set_xlabel('Matrix type',fontsize=12)
# %%
print(J.mat(0))
fig, ax = plt.subplots(figsize=(12,12))
_, cb = pg.show( J.mat(1),ax = ax, markersize=5)
ax.set_ylabel('Number of data',fontsize=12)
ax.set_title('Number of model')
# %%%
# For the joint inversion, concatenate the data and error vectors, create a new
# inversion instance, set logarithmic transformations and run the inversion.
#

dataVec = np.concatenate((data["rhoa"], r))
errorVec = np.concatenate((data["err"], np.ones_like(r)*0.2))
inv = pg.Inversion(fop=fopJoint, verbose=True)
transLog = pg.trans.TransLog()
inv.modelTrans = transLog
inv.dataTrans = transLog
inv.run(dataVec, errorVec, startModel=model_layer,lam=50#, zWeight = 0.05
        )

# Plot the result
ax, cb = pg.show(para, inv.model #,coverage=mgr.coverage()
, **kw)
draw1DColumn(ax, **colkw,**kw)
ax.grid(linestyle='--',linewidth=0.5)


# %%
# Plot subplots
fig, ax = plt.subplot_mosaic([['(a)'], ['(b)']],
                              constrained_layout=True
                              ,figsize=(9, 6))
for label, axx in ax.items():
    axx.set_title(label, fontfamily='serif', loc='left', fontsize=18)

kw = dict(cMin=10, cMax=500, logScale=True, cMap="jet",
          xlabel="Distance (m)", ylabel="Depth (m)",orientation = 'vertical')
# mgr.showResult(ax=ax['(a)'], **kw,coverage = None)
_,cb = pg.show(para, mgr.model, **kw,ax=ax['(a)'])
cb.ax.set_ylabel('Resistivity ($\Omega m$)')
zz = np.abs(z)
iz = np.argsort(z)
dz = np.diff(zz[iz])
thk = np.hstack([dz, dz[-1]])
ztop = -zz[iz[0]]-dz[0]/2
colkw = dict(x=x[0], val=r[iz], thk=thk, width=4, ztopo=ztop)
draw1DColumn(ax=ax['(a)'], **colkw, **kw)
ax['(a)'].grid(linestyle='--',linewidth=0.5)
ax['(a)'].set_title('Normal Inversion of ERT Profile', fontsize=18, fontweight='bold')
ax['(a)'].plot(np.array(pg.x(data)), np.array(pg.z(data)), 'kd',markersize = 3)
ax['(a)'].add_patch(Rectangle((x[0]-2, z[-1]+1), 4, z[0]-z[-1]-1,
             edgecolor = 'white',
             facecolor = 'none',
             lw=1))

# Subplot b
_,cb = pg.show(para, inv.model, **kw,ax=ax['(b)'])
draw1DColumn(ax=ax['(b)'], **colkw,**kw)
cb.ax.set_ylabel('Resistivity ($\Omega m$)')
ax['(b)'].grid(linestyle='--',linewidth=0.5)
ax['(b)'].set_title('Joint Inversion of Logging & ERT Profile', fontsize=18, fontweight='bold')
ax['(b)'].plot(np.array(pg.x(data)), np.array(pg.z(data)), 'kd',markersize = 3)
ax['(b)'].add_patch(Rectangle((x[0]-2, z[-1]+1), 4, z[0]-z[-1]-1,
             edgecolor = 'white',
             facecolor = 'none',
             lw=1))

fig.savefig('Compare.png',dpi=300, bbox_inches='tight')
# %%
respDP = inv.response[data.size():]
fig, ax = plt.subplots(figsize=(4, 6))
ax.semilogx(r, z,"o-",color='black',markersize = 2.5, label="Well logging")
ax.semilogx(list(resSmooth), z,color='blue', label="(a) ERT")
ax.set_ylabel('Depth (m)')
ax.set_xlabel('Resistivity ($\Omega m$)')
# resMesh = pg.interpolate(srcMesh=para, inVec=inv.model, destPos=posVec)
# ax.semilogx(resMesh, z, label="ERT+DP")
ax.semilogx(respDP, z,color='red', label="(b) ERT + logging")
ax.grid(linestyle='--',linewidth=0.5)
ax.legend()
fig.savefig('1D.png',dpi=300, bbox_inches='tight')
# %%
# Plot ERT subplots
fig, ax = plt.subplot_mosaic([['(a)'], ['(b)']],
                              constrained_layout=True
                              ,figsize=(9, 6))
for label, axx in ax.items():
    axx.set_title(label, fontfamily='serif', loc='left', fontsize=18)

kw = dict(cMin=10, cMax=500, logScale=True, cMap="jet",
          xlabel="Distance (m)", ylabel="Depth (m)",orientation = 'vertical')
# mgr.showResult(ax=ax['(a)'], **kw,coverage = None)
_,cb = pg.show(para, mgr.model, **kw,ax=ax['(a)'])
cb.ax.set_ylabel('Resistivity ($\Omega m$)')
zz = np.abs(z)
iz = np.argsort(z)
dz = np.diff(zz[iz])
thk = np.hstack([dz, dz[-1]])
ztop = -zz[iz[0]]-dz[0]/2
colkw = dict(x=x[0], val=r[iz], thk=thk, width=4, ztopo=ztop)
# draw1DColumn(ax=ax['(a)'], **colkw, **kw)
ax['(a)'].grid(linestyle='--',linewidth=0.5)
ax['(a)'].set_title('Normal Inversion of ERT Profile', fontsize=18, fontweight='bold')
ax['(a)'].plot(np.array(pg.x(data)), np.array(pg.z(data)), 'kd',markersize = 3)

# Subplot b
_,cb = pg.show(para, inv.model, **kw,ax=ax['(b)'])
draw1DColumn(ax=ax['(b)'], **colkw,**kw)
cb.ax.set_ylabel('Resistivity ($\Omega m$)')
ax['(b)'].grid(linestyle='--',linewidth=0.5)
ax['(b)'].set_title('Joint Inversion of Logging & ERT Profile', fontsize=18, fontweight='bold')
ax['(b)'].plot(np.array(pg.x(data)), np.array(pg.z(data)), 'kd',markersize = 3)
ax['(b)'].add_patch(Rectangle((x[0]-2, z[-1]+1), 4, z[0]-z[-1]-1,
             edgecolor = 'white',
             facecolor = 'none',
             lw=1))
fig.savefig('ERT_only.png',dpi=300, bbox_inches='tight')

# %%
# 1D ERT logging
respDP = inv.response[data.size():]
fig, ax = plt.subplots(figsize=(4, 6))
ax.semilogx(r, z,"o-",color='black',markersize = 2.5, label="Well logging")
ax.semilogx(list(resSmooth), z,color='blue', label="(a) ERT")
ax.set_ylabel('Depth (m)')
ax.set_xlabel('Resistivity ($\Omega m$)')
# resMesh = pg.interpolate(srcMesh=para, inVec=inv.model, destPos=posVec)
# ax.semilogx(resMesh, z, label="ERT+DP")
ax.grid(linestyle='--',linewidth=0.5)
ax.legend()
fig.savefig('1D ERT logging.png',dpi=300, bbox_inches='tight')