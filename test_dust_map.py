import matplotlib as mpl
import handy.scatter as hsc
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pd
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

from resamp import gen
import subprocess as subpro
from dustmaps.sfd import SFDQuery
from extinction_redden import A_wave
from astropy.coordinates import SkyCoord
from light_measure import light_measure, flux_recal

import time
import sfdmap
m = sfdmap.SFDMap('/mnt/ddnfs/data_users/cxkttwl/ICL/data/dust_map/sfddata_maskin', scaling = 0.86)
# constant
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
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

pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.250
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2
# total catalogue with redshift
with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sample_catalog.h5') as f:
	catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]
# dust correct
Rv = 3.1
sfd = SFDQuery()
band = ['u', 'g', 'r', 'i', 'z']
l_wave = np.array([3551, 4686, 6166, 7480, 8932])
mag_add = np.array([-0.04, 0, 0, 0, 0.02])
zop = np.array([22.46, 22.5, 22.5, 22.5, 22.52])
sb_lim = np.array([24.35, 25, 24.5, 24, 22.9])

def map_stack():

	x0 = np.linspace(0, 2047, 2048)
	y0 = np.linspace(0, 1488, 1489)
	img_grid = np.array(np.meshgrid(x0, y0))

	load = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/'
	mapN = np.int(100)
	for q in range(len(band)):
		dust_map_11 = np.zeros((1489, 2048), dtype = np.float)
		dust_map_98 = np.zeros((1489, 2048), dtype = np.float)

		for k in range(mapN):
			file = 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[q], ra[k], dec[k], z[k])
			data_f = fits.open(load+file)
			img = data_f[0].data
			head_inf = data_f[0].header
			wcs = awc.WCS(head_inf)
			x_side = data_f[0].data.shape[1]
			y_side = data_f[0].data.shape[0]

			t0 = time.time()
			ra_img, dec_img = wcs.all_pix2world(img_grid[0,:], img_grid[1,:], 1)
			pos = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
			BEV = sfd(pos)
			bev = m.ebv(pos)

			dust_map_11 = dust_map_11 + bev
			dust_map_98 = dust_map_98 + BEV * 0.86
			print(k)
		map_11 = dust_map_11 / mapN
		map_98 = dust_map_98 / mapN

		plt.figure()
		gf1 = plt.imshow(map_11, cmap = 'rainbow', origin = 'lower')
		plt.title('$map_{2011} \; stack \; %.0f \; in \; %s \; band$' % (mapN, band[q]))
		plt.colorbar(gf1, fraction = 0.035, pad = 0.01, label = '$f[nmagy]$')
		plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/map_11_stack_%s_band.png' % band[q], dpi = 600)
		plt.close()

		plt.figure()
		gf2 = plt.imshow(map_98, cmap = 'rainbow', origin = 'lower')
		plt.title('$map_{1998} \; stack \; %.0f \; in \; %s \; band$' % (mapN, band[q]))
		plt.colorbar(gf1, fraction = 0.035, pad = 0.01, label = '$f[nmagy]$')
		plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/map_98_stack_%s_band.png' % band[q], dpi = 600)
		plt.close()

		print(q)

	return
def main():
	map_stack()

if __name__ == "__main__":
	main()