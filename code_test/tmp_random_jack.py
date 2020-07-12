import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

import astropy.units as U
import astropy.constants as C

import h5py
import time
import numpy as np
import pandas as pds
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.fits as fits

from scipy import ndimage
from astropy import cosmology as apcy

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
kpc2m = U.kpc.to(U.m)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Msun2kg = U.M_sun.to(U.kg)
Lsun = C.L_sun.value*10**7
G = C.G.value

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel, z_ref = 0.396, 0.250
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # (erg/s)/cm^-2
R0 = 1 # Mpc
Angu_ref = (R0 / Da_ref) * rad2asec
Rpp = Angu_ref / pixel

home = '/media/xkchen/My Passport/data/SDSS/'
load = '/media/xkchen/My Passport/data/SDSS/'
band = ['r', 'g', 'i', 'u', 'z']

def rand_sub_samp(band_id, sub_z, sub_ra, sub_dec, img_x, img_y, mix_id):

	stack_N = len(sub_z)
	kk = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	for jj in range(stack_N):

		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]
		xn, yn = img_x[jj], img_y[jj]

		#data_A = fits.open(load + 'random_cat/mask_no_dust/random_mask_%s_ra%.3f_dec%.3f_z%.3f.fits' % (band[kk], ra_g, dec_g, z_g))
		data_A = fits.open(load + 're_mask/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f.fits' % (band[kk], ra_g, dec_g, z_g) )

		img_A = data_A[0].data
		head = data_A[0].header
		#wcs_lis = awc.WCS(head)
		#xn, yn = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

		# centered on cat.(ra, dec)
		la0 = np.int(y0 - yn)
		la1 = np.int(y0 - yn + img_A.shape[0])
		lb0 = np.int(x0 - xn)
		lb1 = np.int(x0 - xn + img_A.shape[1])

		idx = np.isnan(img_A)
		idv = np.where(idx == False)

		sum_array_A[la0: la1, lb0: lb1][idv] = sum_array_A[la0: la1, lb0: lb1][idv] + img_A[idv]
		count_array_A[la0: la1, lb0: lb1][idv] = img_A[idv]
		id_nan = np.isnan(count_array_A)
		id_fals = np.where(id_nan == False)
		p_count_A[id_fals] = p_count_A[id_fals] + 1.
		count_array_A[la0: la1, lb0: lb1][idv] = np.nan

	id_zero = p_count_A == 0
	p_count_A[id_zero] = np.nan
	sum_array_A[id_zero] = np.nan

	stack_img = sum_array_A / p_count_A
	where_are_inf = np.isinf(stack_img)
	stack_img[where_are_inf] = np.nan

	## sub-sample for jackknife
	#with h5py.File(home + 'tmp_stack/jack_random/rand_cat/%s_band_cat-stack_random-img_%d-sub-smp.h5' % (band[kk], mix_id), 'w') as f:
	#with h5py.File(home + 'tmp_stack/jack_random/apply_bcgs/%s_band_bcgs-stack_random-img_%d-sub-smp.h5' % (band[kk], mix_id), 'w') as f:
	#	f['a'] = np.array(stack_img)

	## remask test
	#with h5py.File(home + 're_mask/jack/%s_band_cat-stack_random-img_%d-sub-smp.h5' % (band[kk], mix_id), 'w') as f:
	with h5py.File(home + 're_mask/jack/%s_band_bcgs-stack_random-img_%d-sub-smp.h5' % (band[kk], mix_id), 'w') as f:
		f['a'] = np.array(stack_img)

	return

def rand_jack_samp(band_id, id_set, mix_id):

	kk = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	for jj in id_set:

		#with h5py.File(home + 'tmp_stack/jack_random/rand_cat/%s_band_cat-stack_random-img_%d-sub-smp.h5' % (band[kk], jj), 'r') as f:
		#with h5py.File(home + 'tmp_stack/jack_random/apply_bcgs/%s_band_bcgs-stack_random-img_%d-sub-smp.h5' % (band[kk], jj), 'r') as f:
		## remask test
		#with h5py.File(home + 're_mask/jack/%s_band_cat-stack_random-img_%d-sub-smp.h5' % (band[kk], jj), 'r') as f:
		with h5py.File(home + 're_mask/jack/%s_band_bcgs-stack_random-img_%d-sub-smp.h5' % (band[kk], jj), 'r') as f:
			sub_img = np.array(f['a'])

		id_nn = np.isnan(sub_img)
		idv = id_nn == False
		sum_array_A[idv] = sum_array_A[idv] + sub_img[idv]
		count_array_A[idv] = sub_img[idv]
		id_nan = np.isnan(count_array_A)
		id_fals = id_nan == False
		p_count_A[id_fals] = p_count_A[id_fals] + 1.
		count_array_A[idv] = np.nan

	id_zero = p_count_A == 0
	p_count_A[id_zero] = np.nan
	sum_array_A[id_zero] = np.nan

	stack_img = sum_array_A / p_count_A
	where_are_inf = np.isinf(stack_img)
	stack_img[where_are_inf] = np.nan

	#with h5py.File(home + 'tmp_stack/jack_random/rand_cat/%s_band_cat-stack_random-img_%d-sub-jack.h5' % (band[kk], mix_id), 'w') as f:
	#with h5py.File(home + 'tmp_stack/jack_random/apply_bcgs/%s_band_bcgs-stack_random-img_%d-sub-jack.h5' % (band[kk], mix_id), 'w') as f:
	#	f['a'] = np.array(stack_img)

	## remask test
	#with h5py.File(home + 're_mask/jack/%s_band_cat-stack_random-img_%d-sub-jack.h5' % (band[kk], mix_id), 'w') as f:
	with h5py.File(home + 're_mask/jack/%s_band_bcgs-stack_random-img_%d-sub-jack.h5' % (band[kk], mix_id), 'w') as f:
		f['a'] = np.array(stack_img)

	return

def clus_sub_samp(band_id, sub_z, sub_ra, sub_dec, mix_id):

	stack_N = len(sub_z)
	kk = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	for jj in range(stack_N):

		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]

		#data_A = fits.open(home + 'tmp_stack/real_cluster/A_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits' % (band[kk], ra_g, dec_g, z_g))
		data_A = fits.open(home + 're_mask/cluster/mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits' % (band[kk], ra_g, dec_g, z_g) )
		img_A = data_A[0].data
		head = data_A[0].header
		wcs_lis = awc.WCS(head)
		xn, yn = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

		la0 = np.int(y0 - yn)
		la1 = np.int(y0 - yn + img_A.shape[0])
		lb0 = np.int(x0 - xn)
		lb1 = np.int(x0 - xn + img_A.shape[1])

		idx = np.isnan(img_A)
		idv = np.where(idx == False)

		sum_array_A[la0: la1, lb0: lb1][idv] = sum_array_A[la0: la1, lb0: lb1][idv] + img_A[idv]
		count_array_A[la0: la1, lb0: lb1][idv] = img_A[idv]
		id_nan = np.isnan(count_array_A)
		id_fals = np.where(id_nan == False)
		p_count_A[id_fals] = p_count_A[id_fals] + 1.
		count_array_A[la0: la1, lb0: lb1][idv] = np.nan

	id_zero = p_count_A == 0
	p_count_A[id_zero] = np.nan
	sum_array_A[id_zero] = np.nan

	stack_img = sum_array_A / p_count_A
	where_are_inf = np.isinf(stack_img)
	stack_img[where_are_inf] = np.nan

	#with h5py.File(home + 'tmp_stack/jack_random/apply_bcgs/%s_band_bcgs-stack_clust-img_%d-sub-smp.h5' % (band[kk], mix_id), 'w') as f:
	#	f['a'] = np.array(stack_img)

	with h5py.File(home + 're_mask/jack/%s_band_bcgs-stack_clust-img_%d-sub-smp.h5' % (band[kk], mix_id), 'w') as f:
		f['a'] = np.array(stack_img)

	return

def clus_jack_samp(band_id, id_set, mix_id):

	kk = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	for jj in id_set:

		#with h5py.File(home + 'tmp_stack/jack_random/apply_bcgs/%s_band_bcgs-stack_clust-img_%d-sub-smp.h5' % (band[kk], jj), 'r') as f:
		with h5py.File(home + 're_mask/jack/%s_band_bcgs-stack_clust-img_%d-sub-smp.h5' % (band[kk], jj), 'r') as f:
			sub_img = np.array(f['a'])

		id_nn = np.isnan(sub_img)
		idv = id_nn == False
		sum_array_A[idv] = sum_array_A[idv] + sub_img[idv]
		count_array_A[idv] = sub_img[idv]
		id_nan = np.isnan(count_array_A)
		id_fals = id_nan == False
		p_count_A[id_fals] = p_count_A[id_fals] + 1.
		count_array_A[idv] = np.nan

	id_zero = p_count_A == 0
	p_count_A[id_zero] = np.nan
	sum_array_A[id_zero] = np.nan

	stack_img = sum_array_A / p_count_A
	where_are_inf = np.isinf(stack_img)
	stack_img[where_are_inf] = np.nan

	#with h5py.File(home + 'tmp_stack/jack_random/apply_bcgs/%s_band_bcgs-stack_clust-img_%d-sub-jack.h5' % (band[kk], mix_id), 'w') as f:
	with h5py.File(home + 're_mask/jack/%s_band_bcgs-stack_clust-img_%d-sub-jack.h5' % (band[kk], mix_id), 'w') as f:
		f['a'] = np.array(stack_img)

	return

def main():

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	N_bin = 30

	### random imgs
	for kk in range( 1 ):

		with h5py.File(load + 'random_cat/cat_select/rand_%s_band_catalog.h5' % (band[kk]), 'r') as f:
			tmp_array = np.array(f['a'])
		ra, dec, z = np.array(tmp_array[0]), np.array(tmp_array[1]), np.array(tmp_array[2])

		## use random img cat.
		#with h5py.File(home + 'tmp_stack/r_band_random_img-position.h5', 'r') as f:
		#	clus_x = np.array(f['x'])
		#	clus_y = np.array(f['y'])

		## use the BCGs img position
		with h5py.File(home + 'tmp_stack/jack_random/copy_r_band_cluster_img-coord.h5', 'r') as f:
			clus_x = np.array(f['x'])
			clus_y = np.array(f['y'])

		DN = len(z)
		lis_z = z[:DN]
		lis_ra = ra[:DN]
		lis_dec = dec[:DN]

		zN = len(lis_z)
		n_step = zN // N_bin
		id_arr = np.linspace(0, zN - 1, zN)
		id_arr = id_arr.astype(int)

		for nn in range(N_bin):

			if nn == N_bin - 1:
				dot = id_arr[nn * n_step:]
			else:
				dot = id_arr[nn * n_step: (nn + 1) * n_step]

			set_z = lis_z[dot]
			set_ra = lis_ra[dot]
			set_dec = lis_dec[dot]

			set_x = clus_x[dot]
			set_y = clus_y[dot]

			rand_sub_samp(kk, set_z, set_ra, set_dec, set_x, set_y, nn)

		for nn in range(N_bin):
			id_arry = np.linspace(0, N_bin -1, N_bin)
			id_arry = id_arry.astype(int)
			jack_id = list(id_arry)
			jack_id.remove(jack_id[nn])
			jack_id = np.array(jack_id)

			rand_jack_samp(kk, jack_id, nn)
	raise
	### cluster imgs
	for kk in range( 1 ):

		with h5py.File(load + 'mpi_h5/%s_band_sky_catalog.h5' % band[kk], 'r') as f:
			set_array = np.array(f['a'])
		ra, dec, z = set_array[0,:], set_array[1,:], set_array[2,:]

		DN = len(z)
		lis_z = z[:DN]
		lis_ra = ra[:DN]
		lis_dec = dec[:DN]

		zN = len(lis_z)
		n_step = zN // N_bin
		id_arr = np.linspace(0, zN - 1, zN)
		id_arr = id_arr.astype(int)

		for nn in range(N_bin):

			if nn == N_bin - 1:
				dot = id_arr[nn * n_step:]
			else:
				dot = id_arr[nn * n_step: (nn + 1) * n_step]

			set_z = lis_z[dot]
			set_ra = lis_ra[dot]
			set_dec = lis_dec[dot]

			clus_sub_samp(kk, set_z, set_ra, set_dec, nn)

		for nn in range(N_bin):
			id_arry = np.linspace(0, N_bin -1, N_bin)
			id_arry = id_arry.astype(int)
			jack_id = list(id_arry)
			jack_id.remove(jack_id[nn])
			jack_id = np.array(jack_id)

			clus_jack_samp(kk, jack_id, nn)

if __name__ == "__main__":
	main()
