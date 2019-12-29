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

'''
############ version 1 sample (z selection, but no selection for images, sky images)
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
'''
'''
########### version 2 (2019.12.20, selected by sky image)
load = '/home/xkchen/mywork/ICL/data/redmapper/'
dat = pds.read_csv(load + 'r_band_sky_catalog.csv')
r_Mag = np.array( dat['r_Mag'] )
z = np.array( dat['z'] )
Dl_0 = Test_model.luminosity_distance(z).value
Dl_ref = Test_model.luminosity_distance(0.25).value
Mag_25 = r_Mag + 5 * np.log10(Dl_ref / Dl_0)
std_me = np.std(Mag_25)
'''
##### more closer to frame center sample
with h5py.File('/home/xkchen/jupyter/r_band_sky_0.8Mpc_select.h5', 'r') as f:
	dat = np.array(f['a'])
z = dat[-2,:]
r_Mag = dat[-1,:]
Dl_0 = Test_model.luminosity_distance(z).value
Dl_ref = Test_model.luminosity_distance(0.25).value
Mag_25 = r_Mag + 5 * np.log10(Dl_ref / Dl_0)
std_me = np.std(Mag_25)

Mag_Z05 = pds.read_csv('/home/xkchen/mywork/ICL/Zibetti_SB/Mag_Z05.csv')
mag_ref = np.array(Mag_Z05['Mag'])
N_tot = ['%.0f' % ll for ll in Mag_Z05['Number']]
N_tot = [np.int(ll) for ll in N_tot]
N_tot = np.array(N_tot)
## estimate the standard deviation
d_mag = mag_ref[1:] - mag_ref[:-1]
d_mag = np.mean(d_mag)
std_Z05 = np.sum( (mag_ref - np.mean(mag_ref))**2 * N_tot * d_mag ) / np.sum(N_tot * d_mag)

bins = Mag_Z05['Mag']

plt.figure()
ax = plt.subplot(111)
#ax.hist(Mag_25, bins = bins, histtype = 'bar', color = 'r', label = 'Mine', alpha = 0.5)
#ax.bar(Mag_Z05['Mag'], Mag_Z05['Number'], width = 0.1, align = 'center', color = 'b', label = 'Z05', alpha = 0.5)

ax.hist(Mag_25, bins = bins, histtype = 'bar', color = 'r', density = True, label = 'Mine', alpha = 0.5)
ax.bar(Mag_Z05['Mag'], Mag_Z05['Number'] / np.sum(N_tot * d_mag), width = 0.1, align = 'center', color = 'b', label = 'Z05', alpha = 0.5)

ax.axvline(x = np.mean(mag_ref), linestyle = '--', color = 'r', alpha = 0.5)
ax.axvline(x = np.median(mag_ref), linestyle = '-', color = 'r', alpha = 0.5)
ax.axvline(x = np.mean(Mag_25), linestyle = '--', color = 'b', alpha = 0.5)
ax.axvline(x = np.median(Mag_25), linestyle = '-', color = 'b', alpha = 0.5)
#ax.text(18.5, 200, s = '$ \sigma_{Mine} = %.3f $' % std_me + '\n' + '$ \sigma_{Z05} = %.3f $' % std_Z05,)
ax.text(18.5, 0.8, s = '$ \sigma_{Mine} = %.3f $' % std_me + '\n' + '$ \sigma_{Z05} = %.3f $' % std_Z05,)

ax.set_xlabel('$ M_{r,0.25} $')
#ax.set_ylabel(' # of clusters')
ax.set_ylabel('PDF')
ax.legend(loc = 1)

#plt.savefig('Mag_hist.png', dpi = 300)
plt.savefig('Mag_pdf_sky_select.png', dpi = 300)
plt.show()
