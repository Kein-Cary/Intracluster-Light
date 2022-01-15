import matplotlib as mpl
mpl.use('Agg')
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

import subprocess as subpro
from dustmaps.sfd import SFDQuery
from extinction_redden import A_wave
from astropy.coordinates import SkyCoord
from light_measure import light_measure

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

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
f0 = 3631*Jy # zero point in unit (erg/s)/cm^-2

# dust correct
Rv = 3.1
sfd = SFDQuery()

band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])

home = '/mnt/ddnfs/data_users/cxkttwl/ICL/'
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
def map_stack(band_id, set_ra, set_dec, set_z):

	kk = band_id

	x0 = np.linspace(0, 2047, 2048)
	y0 = np.linspace(0, 1488, 1489)
	img_grid = np.array(np.meshgrid(x0, y0))

	load = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/'
	zN = len(set_z)

	dust_map_11 = np.zeros((1489, 2048), dtype = np.float)

	for k in range(zN):
		#file = 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[kk], set_ra[k], set_dec[k], set_z[k]) ## real cluster
		file = 'redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[kk], set_ra[k], set_dec[k], set_z[k]) ## random cluster

		data_f = fits.open(home + file)
		img = data_f[0].data
		head_inf = data_f[0].header
		wcs = awc.WCS(head_inf)

		ra_img, dec_img = wcs.all_pix2world(img_grid[0,:], img_grid[1,:], 1)
		pos = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
		BEV = sfd(pos)
		bev = m.ebv(pos)

		dust_map_11 = dust_map_11 + bev

	map_11 = dust_map_11

	with h5py.File(tmp + 'stack_dust-map-11_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(map_11)

	return

def main():

	x0 = np.linspace(0, 2047, 2048)
	y0 = np.linspace(0, 1488, 1489)
	"""
	# total catalogue with spec-redshift
	with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/mpi_h5/sample_catalog.h5', 'r') as f:
		catalogue = np.array(f['a'])
	z = catalogue[0]
	ra = catalogue[1]
	dec = catalogue[2]
	"""
	for kk in range(3):

		with h5py.File(load + 'random_cat/cat_select/rand_%s_band_catalog.h5' % (band[kk]), 'r') as f:
			tmp_array = np.array(f['a'])
		ra, dec, z, rich = np.array(tmp_array[0]), np.array(tmp_array[1]), np.array(tmp_array[2]), np.array(tmp_array[3])
		zN = len(z)

		set_z, set_ra, set_dec = z, ra, dec

		DN = len(set_z)
		m, n = divmod(DN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		map_stack(kk, set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1], set_z[N_sub0 :N_sub1])
		commd.Barrier()

		if rank == 0:

			mean_map11 = np.zeros((len(y0), len(x0)), dtype = np.float)

			for pp in range(cpus):

				with h5py.File(tmp + 'stack_dust-map-11_%d_in_%s_band.h5' % (pp, band[kk]), 'r') as f:
					map_11 = np.array(f['a'])
				mean_map11 = mean_map11 + map_11

			mean_map11 = mean_map11 / DN

			with h5py.File(load + 'random_cat/angle_stack/map_11_stack_%s_band.h5' % (band[kk],), 'w') as f:
				f['a'] = np.array(mean_map11)

			plt.figure(figsize = (7.5, 4.5))
			gf1 = plt.imshow(mean_map11, cmap = 'rainbow', origin = 'lower')
			plt.title('$map_{2011} \; stack \; in \; %s \; band$' % (band[kk],) )
			plt.colorbar(gf1, fraction = 0.035, pad = 0.01, label = '$ E_{B-V} $')
			plt.subplots_adjust(left = 0.01, right = 0.85)
			plt.savefig(load + 'random_cat/angle_stack/map_11_stack_%s_band.png' % band[kk], dpi = 600)
			plt.close()

			Av = Rv * mean_map11
			Al = A_wave(l_wave[kk], Rv) * Av
			devi_flux = 10**(Al / 2.5) - 1

			plt.figure(figsize = (7.5, 4.5))
			plt.title('$ A_{\\lambda} \,[based \; on \; map_{2011} \; %s \; band]$' % (band[kk],) )
			gf1 = plt.imshow(Al, cmap = 'rainbow', origin = 'lower')
			plt.colorbar(gf1, fraction = 0.035, pad = 0.01,)
			plt.subplots_adjust(left = 0.01, right = 0.85)
			plt.savefig(load + 'random_cat/angle_stack/Al_map-11_stack_%s_band.png' % band[kk], dpi = 300)
			plt.close()

			plt.figure(figsize = (7.5, 4.5))
			plt.title('$ 10^{ A_{\\lambda} / 2.5} - 1 \,[based \; on \; map_{2011} \; %s \; band]$' % (band[kk],) )
			gf1 = plt.imshow(devi_flux, cmap = 'rainbow', origin = 'lower')
			plt.colorbar(gf1, fraction = 0.035, pad = 0.01,)
			plt.subplots_adjust(left = 0.01, right = 0.85)
			plt.savefig(load + 'random_cat/angle_stack/flux_devi_map-11_stack_%s_band.png' % band[kk], dpi = 300)
			plt.close()

		commd.Barrier()

if __name__ == "__main__":
	main()
