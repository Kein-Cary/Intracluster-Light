import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

from dustmaps.sfd import SFDQuery
from extinction_redden import A_wave
from astropy.coordinates import SkyCoord
from resample_modelu import sum_samp, down_samp
from light_measure import light_measure, flux_recal

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.ascii as asc
import astropy.io.fits as fits

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()
import time

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

pixel = 0.396
z_ref = 0.250
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # zero point in unit (erg/s)/cm^-2
R0 = 1 # Mpc
Angu_ref = (R0 / Da_ref) * rad2asec
Rpp = Angu_ref / pixel

d_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])
mag_add = np.array([0, 0, 0, -0.04, 0.02])

Rv = 3.1
sfd = SFDQuery()

def BCG_hist():

	for tt in range(len(band)):
		with h5py.File(load + 'mpi_h5/%s_band_sample_catalog.h5' % band[tt], 'r') as f:
			sub_array = np.array(f['a'])
		ra, dec, z, rich, r_mag = sub_array[0,:], sub_array[1,:], sub_array[2,:], sub_array[3,:], sub_array[4,:]
		zN = len(z)

		cenx = np.zeros(zN, dtype = np.float)
		ceny = np.zeros(zN, dtype = np.float)

		for kk in range(zN):
			ra_g, dec_g, z_g = ra[kk], dec[kk], z[kk]
			Da_g = Test_model.angular_diameter_distance(z_g).value

			pro_f = d_file + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[tt], ra_g, dec_g, z_g)
			data_f = fits.open(pro_f)
			img = data_f[0].data
			head_inf = data_f[0].header
			wcs = awc.WCS(head_inf)
			cx_BCG, cy_BCG = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

			cenx[kk] = cx_BCG * 1
			ceny[kk] = cy_BCG * 1

		BCG_pos = np.array([cenx, ceny])
		with h5py.File(load + 'mpi_h5/%s_band_%d_BCG_position.h5' % (band[tt], zN), 'w') as f:
			f['a'] = np.array(BCG_pos)
		with h5py.File(load + 'mpi_h5/%s_band_%d_BCG_position.h5' % (band[tt], zN) ) as f:
			for pp in range(len(BCG_pos)):
				f['a'][pp,:] = BCG_pos[pp,:]

	return

def hist_BCG():
	x0, y0 = 1024, 745

	cen_Ds = []
	for tt in range(len(band)):

		with h5py.File(load + 'mpi_h5/%s_band_sample_catalog.h5' % band[tt], 'r') as f:
			sub_array = np.array(f['a'])
		ra, dec, z, rich, r_mag = sub_array[0,:], sub_array[1,:], sub_array[2,:], sub_array[3,:], sub_array[4,:]
		zN = len(z)

		with h5py.File(load + 'mpi_h5/%s_band_%d_BCG_position.h5' % (band[tt], zN) ) as f:
			BCG_pos = np.array(f['a'])
		cen_x = BCG_pos[0,:]
		cen_y = BCG_pos[1,:]

		DR = np.sqrt((cen_x - x0)**2 + (cen_y - y0)**2) 
		cen_Ds.append(DR)
		'''
		## sparial view
		plt.figure()
		plt.title('BCG position [%s band %d imgs]' % (band[tt], zN) )
		plt.plot(1024, 745, 'b^', label = 'frame center')
		plt.hist2d(cen_x, cen_y, bins = [25, 25], cmap = 'plasma', cmin = 1)
		plt.colorbar(fraction = 0.035, pad = 0.01, label = 'cluster Number')
		plt.legend(loc = 1)
		plt.savefig(load + 'BCG_pos_hist_%s_band_%d_imgs.png' % (band[tt], zN), dpi = 300)
		plt.close()

		plt.figure()
		plt.title('BCG position [%s band %d imgs]' % (band[tt], zN) )
		plt.scatter(cen_x, cen_y, s = 5, marker = 'o', color = 'r')
		plt.plot(x0, y0, 'b^', label = 'frame center')
		plt.savefig(load + 'BCG_pos_dot_%s_band_%d_imgs.png' % (band[tt], zN), dpi = 300)
		plt.close()
		'''
	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('Distance between BCG and frame center')

	for kk in range(len(cen_Ds)):
		ax.hist(cen_Ds[kk], bins = 20, histtype = 'step', color = mpl.cm.rainbow(kk / len(band)), 
			density = True, alpha = 0.5, label = '%s band' % band[kk])
	ax.set_xlabel('R[pixels]')
	ax.set_ylabel('PDF')
	ax.legend(loc = 1)
	plt.savefig(load + 'BCG_cen_DS_hist.png', dpi = 300)
	plt.close()

	return

def main():
	#BCG_hist()
	hist_BCG()

if __name__ == "__main__":
	main()
