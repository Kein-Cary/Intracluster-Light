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
h = H0 / 100
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

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'
band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])
mag_add = np.array([0, 0, 0, -0.04, 0.02])

Rv = 3.1
sfd = SFDQuery()

def selection(band_id, sub_z, sub_ra, sub_dec):

	ii = np.int(band_id)
	zn = len(sub_z)

	ra_fit = np.zeros(zn, dtype = np.float)
	dec_fit = np.zeros(zn, dtype = np.float)
	z_fit = np.zeros(zn, dtype = np.float)

	for k in range(zn):
		ra_g = sub_ra[k]
		dec_g = sub_dec[k]
		z_g = sub_z[k]

		data_A = fits.getdata(load + 'resample/1_5sigma/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
		img_A = data_A[0]
		xn = data_A[1]['CENTER_X']
		yn = data_A[1]['CENTER_Y']

		cx = np.int(img_A.shape[1] / 2)
		cy = np.int(img_A.shape[0] / 2)
		## select clusters (BCG is located in the center region)
		x_side = np.array([cx - Rpp, cx + Rpp])
		y_side = np.array([cy - Rpp, cy + Rpp])
		idx = (x_side[0] < xn) & (xn < x_side[1])
		idy = (y_side[0] < yn) & (yn < y_side[1])
		idz = idx & idy

		if idz == True:
			ra_fit[k], dec_fit[k], z_fit[k] = ra_g, dec_g, z_g
		else:
			continue
	ra_fit = ra_fit[ra_fit != 0]
	dec_fit = dec_fit[dec_fit != 0]
	z_fit = z_fit[z_fit != 0]
	sel_arr = np.array([ra_fit, dec_fit, z_fit])

	with h5py.File(tmp + 'sky_select_%d_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(sel_arr)
	with h5py.File(tmp + 'sky_select_%d_%s_band.h5' % (rank, band[ii]) ) as f:
		for ll in range(len(sel_arr)):
			f['a'][ll,:] = sel_arr[ll,:]

def imgs_cut(band_id, sub_z, sub_ra, sub_dec):

	ii = np.int(band_id)
	zn = len(sub_z)
	for k in range(zn):
		ra_g = sub_ra[k]
		dec_g = sub_dec[k]
		z_g = sub_z[k]

		data = fits.getdata(load + 'resample/1_5sigma/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
		img = data[0]
		BCGx, BCGy = data[1]['CENTER_X'], data[1]['CENTER_Y']
		RA0, DEC0 = data[1]['CRVAL1'], data[1]['CRVAL2']

		xc, yc = np.int(img.shape[1] / 2), np.int(img.shape[0] / 2)
		re_img = img[yc - np.int(Rpp): yc + np.int(Rpp), xc - np.int(1.3 * Rpp): xc + np.int(1.3 * Rpp)]

		New_bcgx = BCGx - (xc - np.int(1.3 * Rpp))
		New_bcgy = BCGy - (yc - np.int(Rpp))

		Lx = re_img.shape[1]
		Ly = re_img.shape[0]
		Crx = np.int(1.3 * Rpp)
		Cry = np.int(Rpp)

		keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2','CENTER_X','CENTER_Y',
				'CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z', 'P_SCALE']
		value = ['T', 32, 2, Lx, Ly, Crx, Cry, New_bcgx, New_bcgy, RA0, DEC0, ra_g, dec_g, z_g, pixel]
		ff = dict(zip(keys,value))
		fil = fits.Header(ff)
		fits.writeto(load + 
		'sky_select_img/imgs/cut_edge-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), re_img, header = fil, overwrite=True)

	return

def main():
	'''
	###3 build the catalogue
	for tt in range( len(band) ):
		with h5py.File(load + 'mpi_h5/%s_band_sample_catalog.h5' % band[tt], 'r') as f:
			sub_array = np.array(f['a'])
		ra, dec, z, rich, r_mag = sub_array[0,:], sub_array[1,:], sub_array[2,:], sub_array[3,:], sub_array[4,:]
		zN = len(z)

		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m

		selection(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
		commd.Barrier()

	if rank == 0:
		for tt in range( len(band) ):

			set_ra = np.array([0.])
			set_dec = np.array([0.])
			set_z = np.array([0.])

			for pp in range(cpus):
				with h5py.File(tmp + 'sky_select_%d_%s_band.h5' % (pp, band[tt]), 'r') as f:
					sel_arr = np.array(f['a'])
				set_ra = np.r_[ set_ra, sel_arr[0,:] ]
				set_dec = np.r_[ set_dec, sel_arr[1,:] ]
				set_z = np.r_[ set_z, sel_arr[2,:] ]

			set_ra = set_ra[1:]
			set_dec = set_dec[1:]
			set_z = set_z[1:]
			n_sum = len(set_z)
			set_array = np.array([set_ra, set_dec, set_z])
			with h5py.File(load + 'sky_select_img/%s_band_%d_imgs_sky_select.h5' % (band[tt], n_sum), 'w') as f:
				f['a'] = np.array(set_array)
			with h5py.File(load + 'sky_select_img/%s_band_%d_imgs_sky_select.h5' % (band[tt], n_sum) ) as f:
				for ll in range( len(set_array) ):
					f['a'][ll,:] = set_array[ll,:]
	commd.Barrier()
	'''
	N_sum = np.array([2055, 2048, 2049, 2050, 2050])
	for kk in range(3):

		with h5py.File(load + 'sky_select_img/%s_band_%d_imgs_sky_select.h5' % (band[kk], N_sum[kk]), 'r') as f:
			set_array = np.array(f['a'])
		set_ra, set_dec, set_z = set_array[0,:], set_array[1,:], set_array[2,:]

		zN = len(set_z)
		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m

		imgs_cut(kk, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])
		commd.Barrier()

if __name__ == "__main__":
	main()
