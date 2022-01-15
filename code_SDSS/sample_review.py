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

Dl_0 = Test_model.luminosity_distance(z).value
Dl_ref = Test_model.luminosity_distance(0.25).value
Mag_25 = com_Mag + 5 * np.log10(Dl_ref / Dl_0)

plt.figure()
bins = np.linspace(15.5, 19.5, 25)
ax = plt.subplot(111)
ax.set_title('BCG apparent magnitude')
ax.hist(com_Mag, bins = bins, density = True, color = 'r', alpha = 0.5, label = '$M_{r, z}$')
ax.hist(Mag_25, bins = bins, density = True, color = 'b', alpha = 0.5, label = '$M_{r, 0.25}$')
ax.set_xlabel('$M_{r}$')
ax.set_ylabel('PDF')
ax.legend(loc = 1)
plt.savefig('r_Mag_025.png', dpi = 300)
plt.close()

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
########### version 2: (2019.12.20, selected by sky image)
load = '/home/xkchen/mywork/ICL/data/redmapper/'
dat = pds.read_csv(load + 'r_band_sky_catalog.csv')
r_Mag = np.array( dat['r_Mag'] )
z = np.array( dat['z'] )
Dl_0 = Test_model.luminosity_distance(z).value
Dl_ref = Test_model.luminosity_distance(0.25).value
Mag_25 = r_Mag + 5 * np.log10(Dl_ref / Dl_0)
std_me = np.std(Mag_25)

bins_z = np.linspace(0.2, 0.3, 26)
plt.figure()
ax = plt.subplot(111)
ax.hist(z, bins = bins_z, color = 'b', alpha = 0.5)
ax.axvline(x = 0.25, linestyle = '--', color = 'r', alpha = 0.5, label = '$z_{ref} = 0.25$')
ax.set_xlabel('z')
ax.set_ylabel('Number of cluster')
ax.legend(loc = 2)
plt.savefig('r_z_hist_sky_select.png', dpi = 300)
plt.close()

bins_M = np.linspace(15.5, 19.5, 26)
plt.figure()
ax = plt.subplot(111)
ax.set_title('BCG r band apparent magnitude histogram')
ax.hist(r_Mag, bins = bins_M, color = 'b', alpha = 0.5, label = '$ M_{z,i} $')
ax.hist(Mag_25, bins = bins_M, color = 'r', alpha = 0.5, label = '$ M_{z = 0.25} $')
ax.set_xlabel('$ m_{r} [mag] $')
ax.set_ylabel('Number of cluster')
ax.legend(loc = 2)
plt.savefig('r_Mag_hist_sky_select.png', dpi = 300)
plt.close()

raise
##### version 3 : more closer to frame center sample
with h5py.File('/home/xkchen/jupyter/r_band_sky_0.65Mpc_select.h5', 'r') as f:
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
'''
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
'''
## check the richness
with h5py.File('/home/xkchen/jupyter/r_band_2013_imgs_sky_select.h5', 'r') as f:
	dat0 = np.array(f['a'])
ra0 = dat0[0, :]
dec0 = dat0[1,:]
z0 = dat0[2,:]

with h5py.File('/home/xkchen/jupyter/r_band_sky_0.8Mpc_select.h5', 'r') as f:
	dat1 = np.array(f['a'])
ra1 = dat1[0, :]
dec1 = dat1[1,:]
z1 = dat1[2,:]

with h5py.File('/home/xkchen/jupyter/r_band_sky_0.65Mpc_select.h5', 'r') as f:
	dat2 = np.array(f['a'])
ra2 = dat2[0, :]
dec2 = dat2[1,:]
z2 = dat2[2,:]

# read the catalog
goal_data = fits.getdata('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
redshift = np.array(goal_data.Z_SPEC)
Mag_bcgs = np.array(goal_data.MODEL_MAG_R)
Mag_err = np.array(goal_data.MODEL_MAGERR_R)
lamda = np.array(goal_data.LAMBDA)

with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5', 'r') as f:
	catalogue = np.array(f['a'])
com_z = catalogue[0]
com_ra = catalogue[1]
com_dec = catalogue[2]

com_Lamb = lamda[(redshift >= 0.2) & (redshift <= 0.3)]
com_Mag = Mag_bcgs[(redshift >= 0.2) & (redshift <= 0.3)]
com_Mag_err = Mag_err[(redshift >= 0.2) & (redshift <= 0.3)]

rich0 = np.zeros(len(z0), dtype = np.float)
rich1 = np.zeros(len(z1), dtype = np.float)
rich2 = np.zeros(len(z2), dtype = np.float)

for kk in range(len(z0)):
	ra_g = np.float('%.3f' % ra0[kk])
	dec_g = np.float('%.3f' % dec0[kk])
	z_g = np.float('%.3f' % z0[kk])

	for jj in range(len(com_z)):
		ra_c = np.float('%.3f' % com_ra[jj])
		dec_c = np.float('%.3f' % com_dec[jj])
		z_c = np.float('%.3f' % com_z[jj])
		identy = (ra_c == ra_g) & (dec_c == dec_g) & (z_c == z_g)

		if identy == True:
			rich0[kk] = com_Lamb[jj]
		else:
			continue

for kk in range(len(z1)):
	ra_g = np.float('%.3f' % ra1[kk])
	dec_g = np.float('%.3f' % dec1[kk])
	z_g = np.float('%.3f' % z1[kk])

	for jj in range(len(com_z)):
		ra_c = np.float('%.3f' % com_ra[jj])
		dec_c = np.float('%.3f' % com_dec[jj])
		z_c = np.float('%.3f' % com_z[jj])
		identy = (ra_c == ra_g) & (dec_c == dec_g) & (z_c == z_g)

		if identy == True:
			rich1[kk] = com_Lamb[jj]
		else:
			continue

for kk in range(len(z2)):
	ra_g = np.float('%.3f' % ra2[kk])
	dec_g = np.float('%.3f' % dec2[kk])
	z_g = np.float('%.3f' % z2[kk])

	for jj in range(len(com_z)):
		ra_c = np.float('%.3f' % com_ra[jj])
		dec_c = np.float('%.3f' % com_dec[jj])
		z_c = np.float('%.3f' % com_z[jj])
		identy = (ra_c == ra_g) & (dec_c == dec_g) & (z_c == z_g)

		if identy == True:
			rich2[kk] = com_Lamb[jj]
		else:
			continue

bins = np.linspace(19, 190, 21)
plt.figure()
ax = plt.subplot(111)
ax.hist(rich0, bins = bins, density = True, histtype = 'step', color = 'r', label = '1Mpc', alpha = 0.5)
ax.hist(rich1, bins = bins, density = True, histtype = 'step', color = 'g', label = '0.8Mpc', alpha = 0.5)
ax.hist(rich2, bins = bins, density = True, histtype = 'step', color = 'b', label = '0.65Mpc', alpha = 0.5)
ax.set_xlabel('$ \lambda $')
ax.set_ylabel('PDF')
ax.set_title('richness in sub-samples')
ax.legend(loc = 1)
plt.savefig('richness_compare.png', dpi = 300)
plt.close()
