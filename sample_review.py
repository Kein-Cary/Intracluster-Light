import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse

import h5py
import random
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
from astropy import cosmology as apcy

# constant
vc = C.c.to(U.km/U.s).value
G = C.G.value # gravitation constant
Ms = C.M_sun.value # solar mass
kpc2m = U.kpc.to(U.m)
Msun2kg = U.M_sun.to(U.kg)

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
Lsun2erg_s = U.L_sun.to(U.erg/U.s)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Lsun = C.L_sun.value*10**7

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
DH = vc/H0

with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5', 'r') as f:
	catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]

# read the catalog
goal_data = fits.getdata('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
redshift = np.array(goal_data.Z_SPEC)
Mag_bcgs = np.array(goal_data.MODEL_MAG_R)
Mag_err = np.array(goal_data.MODEL_MAGERR_R)
lamda = np.array(goal_data.LAMBDA)

com_Lamb = lamda[(redshift >= 0.2) & (redshift <= 0.3)]
com_Mag = Mag_bcgs[(redshift >= 0.2) & (redshift <= 0.3)]
com_Mag_err = Mag_err[(redshift >= 0.2) & (redshift <= 0.3)]

plt.figure()
ax = plt.subplot(111)
ax.set_title('$\lambda -- z $')
ax.scatter(z, np.log10(com_Lamb), s = 5, color = 'b')
ax.set_xlabel('Z_SPEC')
ax.set_ylabel('$log \lambda $')
plt.savefig('lambd_z.png', dpi = 300)
plt.close()

plt.figure()
ax = plt.subplot(111)
ax.set_title('$ M_{r} -- z $')
ax.scatter(z, com_Mag, s = 5, color = 'b', alpha = 0.5)
ax.set_xlabel('Z_SPEC')
ax.set_ylabel('$M_{r}[mag]$')
ax.invert_yaxis()
plt.savefig('r_Mag_z.png', dpi = 300)
plt.close()

plt.figure()
ax = plt.subplot(111)
ax.set_title('$ \lambda -- M_{r}$')
ax.scatter( com_Mag, np.log10(com_Lamb), s = 5, color = 'b', alpha = 0.5)
ax.set_xlabel('$M_{r}[mag]$')
ax.set_ylabel('$ log \ambda $')
plt.savefig('r_Mag_lambd.png', dpi = 300)
plt.close()
