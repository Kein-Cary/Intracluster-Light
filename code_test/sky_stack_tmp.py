import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import random
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C

from astropy import cosmology as apcy
from resample_modelu import sum_samp, down_samp
from matplotlib.patches import Circle, Ellipse
from light_measure import light_measure, light_measure_Z0

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

pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.250 
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Rpp = (rad2asec / Da_ref) / pixel
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # zero point in unit (erg/s)/cm^-2

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'
band = ['r', 'g', 'i', 'u', 'z']

def sky_oppose(band_id, sub_z, sub_ra, sub_dec):

	## stack sky image
	stack_N = len(sub_z)
	kk = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	sum_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	sum_array_0 = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	sum_array_1 = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	for jj in range(stack_N):
		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]

		## scaled image
		data = fits.open( load + 'sky/sky_resamp/resample_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g) )
		img = data[0].data
		cx, cy = data[0].header['CENTER_X'], data[0].header['CENTER_Y']
		## rule out the edge pixels (due to the errror in flux resampling)
		img[0,:] = np.nan
		img[-1,:] = np.nan
		img[:,0] = np.nan
		img[:,-1] = np.nan
		'''
		## sky-select sample
		data = fits.open( load + 'sky_select_img/sky_set/Cut_edge_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g) )
		img = data[0].data
		cx, cy = data[0].header['CENTER_X'], data[0].header['CENTER_Y']
		'''
		cnx, cny = img.shape[1] - cx, img.shape[0] - cy
		la0 = np.int(y0 - cny)
		la1 = np.int(y0 - cny + img.shape[0])
		lb0 = np.int(x0 - cnx)
		lb1 = np.int(x0 - cnx + img.shape[1])

		idx = np.isnan(img)
		idv = np.where(idx == False)

		sum_array[la0: la1, lb0: lb1][idv] = sum_array[la0:la1, lb0:lb1][idv] + img[idv]
		count_array[la0: la1, lb0: lb1][idv] = img[idv]
		id_nan = np.isnan(count_array)
		id_fals = np.where(id_nan == False)
		p_count[id_fals] = p_count[id_fals] + 1
		count_array[la0: la1, lb0: lb1][idv] = np.nan

		## minus case
		img_add0 = img - np.nanmean(img)
		img_add1 = img - np.nanmedian(img)

		sum_array_0[la0: la1, lb0: lb1][idv] = sum_array_0[la0:la1, lb0:lb1][idv] + img_add0[idv]
		sum_array_1[la0: la1, lb0: lb1][idv] = sum_array_1[la0:la1, lb0:lb1][idv] + img_add1[idv]

	with h5py.File(tmp + 'sky_oppose_sum_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(sum_array)
	with h5py.File(tmp + 'sky_oppose_count_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(p_count)
	## minus case
	with h5py.File(tmp + 'sky_oppose_minus_mean_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(sum_array_0)
	with h5py.File(tmp + 'sky_oppose_minus_media_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(sum_array_1)

	return

def sky_shuffle(band_id, sub_z, sub_ra, sub_dec, shlf_z, shlf_ra, shlf_dec):
	## stack sky image
	stack_N = len(sub_z)
	kk = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	## stack include the cluster region
	sum_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	sum_array_0 = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	sum_array_1 = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	for jj in range(stack_N):

		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]

		pos_ra = shlf_ra[jj]
		pos_dec = shlf_dec[jj]
		pos_z = shlf_z[jj]
		'''
		## scaled image
		data = fits.open( load + 'sky/sky_resamp/resample_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g) )
		img = data[0].data
		ori_cx, ori_cy = data[0].header['CENTER_X'], data[0].header['CENTER_Y']
		## rule out the edge pixels (due to the errror in flux resampling)
		img[0,:] = np.nan
		img[-1,:] = np.nan
		img[:,0] = np.nan
		img[:,-1] = np.nan
		'''
		## sky-select sample
		data = fits.open( load + 'sky_select_img/sky_set/Cut_edge_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g) )
		img = data[0].data
		ori_cx, ori_cy = data[0].header['CENTER_X'], data[0].header['CENTER_Y']

		#pos_data = fits.open( load + 'sky/sky_resamp/resample_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], pos_ra, pos_dec, pos_z) )
		pos_data = fits.open( load + 'sky_select_img/sky_set/Cut_edge_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], pos_ra, pos_dec, pos_z) )
		cx, cy = pos_data[0].header['CENTER_X'], pos_data[0].header['CENTER_Y']
		pos_img = pos_data[0].data
		eta_x = cx / pos_img.shape[1]
		eta_y = cy / pos_img.shape[0]

		cnx = np.int(eta_x * img.shape[1])
		cny = np.int(eta_y * img.shape[0])

		la0 = np.int(y0 - cny)
		la1 = np.int(y0 - cny + img.shape[0])
		lb0 = np.int(x0 - cnx)
		lb1 = np.int(x0 - cnx + img.shape[1])

		idx = np.isnan(img)
		idv = idx == False

		sum_array[la0: la1, lb0: lb1][idv] = sum_array[la0:la1, lb0:lb1][idv] + img[idv]
		count_array[la0: la1, lb0: lb1][idv] = img[idv]
		id_nan = np.isnan(count_array)
		id_fals = np.where(id_nan == False)
		p_count[id_fals] = p_count[id_fals] + 1
		count_array[la0: la1, lb0: lb1][idv] = np.nan

		## minus case
		img_add0 = img - np.nanmean(img)
		img_add1 = img - np.nanmedian(img)

		sum_array_0[la0: la1, lb0: lb1][idv] = sum_array_0[la0:la1, lb0:lb1][idv] + img_add0[idv]
		sum_array_1[la0: la1, lb0: lb1][idv] = sum_array_1[la0:la1, lb0:lb1][idv] + img_add1[idv]

	## save the random center data
	with h5py.File(tmp + 'sky_shuffle_sum_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(sum_array)
	with h5py.File(tmp + 'sky_shuffle_count_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(p_count)
	## minus case
	with h5py.File(tmp + 'sky_shuffle_minus_mean_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(sum_array_0)
	with h5py.File(tmp + 'sky_shuffle_minus_media_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(sum_array_1)

	return

def sky_rndm_cen(band_id, sub_z, sub_ra, sub_dec):

	stack_N = len(sub_z)
	kk = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	## random center as comparison
	rndm_sum = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	rndm_count = np.zeros((len(Ny), len(Nx)), dtype = np.float) * np.nan
	rndm_pcont = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	pos_x = np.zeros(stack_N, dtype = np.float)
	pos_y = np.zeros(stack_N, dtype = np.float)

	sum_array_0 = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	sum_array_1 = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	sqare_f0 = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	sqare_f1 = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	n_add = 0.
	for jj in range(stack_N):
		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]

		## scaled
		data = fits.open( load + 'sky/sky_resamp/resample_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g) )
		img = data[0].data
		cx, cy = data[0].header['CENTER_X'], data[0].header['CENTER_Y']
		## rule out the edge pixels
		img[0,:] = np.nan
		img[-1,:] = np.nan
		img[:,0] = np.nan
		img[:,-1] = np.nan
		'''
		## sky-select sample
		data = fits.open( load + 'sky_select_img/sky_set/Cut_edge_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g) )
		img = data[0].data
		cx, cy = data[0].header['CENTER_X'], data[0].header['CENTER_Y']
		'''
		## the random center test
		rnx, rny = np.random.choice(img.shape[1], 1, replace = False), np.random.choice(img.shape[0], 1, replace = False) ## random center
		pos_x[jj], pos_y[jj] = rnx, rny

		la0 = np.int(y0 - rny)
		la1 = np.int(y0 - rny + img.shape[0])
		lb0 = np.int(x0 - rnx)
		lb1 = np.int(x0 - rnx + img.shape[1])

		idx = np.isnan(img)
		idv = np.where(idx == False)

		rndm_sum[la0:la1, lb0:lb1][idv] = rndm_sum[la0:la1, lb0:lb1][idv] + img[idv]
		rndm_count[la0:la1, lb0:lb1][idv] = img[idv]
		id_nan = np.isnan(rndm_count)
		id_fals = np.where(id_nan == False)
		rndm_pcont[id_fals] = rndm_pcont[id_fals] + 1
		rndm_count[la0:la1, lb0:lb1][idv] = np.nan

		## minus case
		img_add0 = img - np.nanmean(img)
		img_add1 = img - np.nanmedian(img)

		sum_array_0[la0: la1, lb0: lb1][idv] = sum_array_0[la0:la1, lb0:lb1][idv] + img_add0[idv]
		sum_array_1[la0: la1, lb0: lb1][idv] = sum_array_1[la0:la1, lb0:lb1][idv] + img_add1[idv]
		sqare_f0[la0: la1, lb0: lb1][idv] = sqare_f0[la0: la1, lb0: lb1][idv] + img_add0[idv]**2
		sqare_f1[la0: la1, lb0: lb1][idv] = sqare_f1[la0: la1, lb0: lb1][idv] + img_add1[idv]**2

		n_add += 1.
	rndm_pcont[-1, -1] = n_add
	## save the random center data
	with h5py.File(tmp + 'rndm_sum_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(rndm_sum)
	with h5py.File(tmp + 'rndm_sum_pcount_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(rndm_pcont)
	## save the random poisition
	rdn_pos = np.array([pos_x, pos_y])
	with h5py.File(tmp + 'rdnm_pos_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(rdn_pos)
	with h5py.File(tmp + 'rdnm_pos_%d_in_%s_band.h5' % (rank, band[kk]) ) as f:
		for ll in range(len(rdn_pos)):
			f['a'][ll,:] = rdn_pos[ll,:]

	## minus case
	with h5py.File(tmp + 'sky_random_minus_mean_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(sum_array_0)
	with h5py.File(tmp + 'sky_random_minus_media_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(sum_array_1)
	## Variance
	with h5py.File(tmp + 'sky_random_mean_Var_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(sqare_f0)
	with h5py.File(tmp + 'sky_random_media_Var_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(sqare_f1)

	return

def main():
	R_cut, bins = 1280, 80  # in unit of pixel (with 2Mpc inside)
	R_smal, R_max = 10, 1.7e3 # kpc

	## stack all the sky togather (without resampling)
	x0, y0, bins = 2427, 1765, 65
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	d_record = 5 ## record the random test
	N_dd = np.array([2013, 2008, 2002, 2008, 2009]) ## sky-selected sample

	for tt in range(3):
		#N_tot = np.array([50, 100, 200, 500, 1000, 3000]) ## random sub-sample
		'''
		with h5py.File(load + 'mpi_h5/%s_band_sky_catalog.h5' % band[tt], 'r') as f:
			sub_array = np.array(f['a'])
		ra, dec, z, rich, r_mag = sub_array[0,:], sub_array[1,:], sub_array[2,:], sub_array[3,:], sub_array[4,:]

		## sky-select sample
		with h5py.File(load + 'sky_select_img/%s_band_%d_imgs_sky_select.h5' % (band[tt], N_dd[tt]), 'r') as f:
			sub_array = np.array(f['a'])
		ra, dec, z, r_mag = sub_array[0,:], sub_array[1,:], sub_array[2,:], sub_array[3,:]
		'''
		## test for center closed BCG select
		with h5py.File(load + 'sky_select_img/%s_band_sky_0.8Mpc_select.h5' % ( band[tt] ), 'r') as f:
			sub_array = np.array(f['a'])
		ra, dec, z, r_mag = sub_array[0,:], sub_array[1,:], sub_array[2,:], sub_array[3,:]

		zN = len(z)
		N_tot = np.array([zN])
		for aa in range( len(N_tot) ):
			'''
			#np.random.seed(5)
			tt0 = np.random.choice(zN, size = N_tot[aa], replace = False)
			set_z = z[tt0]
			set_ra = ra[tt0]
			set_dec = dec[tt0]
			'''
			set_z = z[:zN]
			set_ra = ra[:zN]
			set_dec = dec[:zN]

			np.random.seed(d_record)
			tt1 = np.random.choice(zN, size = N_tot[aa], replace = False)
			shif_z = z[tt1]
			shif_ra = ra[tt1]
			shif_dec = dec[tt1]

			m, n = divmod(N_tot[aa], cpus)
			N_sub0, N_sub1 = m * rank, (rank + 1) * m
			if rank == cpus - 1:
				N_sub1 += n
			'''
			sky_shuffle(tt, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1], 
				shif_z[N_sub0 :N_sub1], shif_ra[N_sub0 :N_sub1], shif_dec[N_sub0 :N_sub1])
			commd.Barrier()

			sky_oppose(tt, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])
			commd.Barrier()
			'''
			sky_rndm_cen(tt, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])
			commd.Barrier()

			if rank == 0:
				## record the order of set_z....and shif_z..., this is the shuffle record
				## and the former part is also order for oppose case stacking

				tmp_array = np.array([set_z, set_ra, set_dec, shif_z, shif_ra, shif_dec])
				#with h5py.File(load + 'sky/cluster/%d_oppos_shlf_%s_band_%d_order.h5' % (d_record, band[tt], N_tot[aa]), 'w') as f:
				#with h5py.File(load + 'sky_select_img/result/%d_oppos_shlf_%s_band_%d_order.h5' % (d_record, band[tt], N_tot[aa]), 'w') as f:
				with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/%d_oppos_shlf_%s_band_%d_order.h5' % (d_record, band[tt], N_tot[aa]), 'w') as f:
				#with h5py.File(load + 'sky_select_img/test_set/0.65Mpc/%d_oppos_shlf_%s_band_%d_order.h5' % (d_record, band[tt], N_tot[aa]), 'w') as f:
					f['a'] = np.array(tmp_array)

				#with h5py.File(load + 'sky/cluster/%d_oppos_shlf_%s_band_%d_order.h5' % (d_record, band[tt], N_tot[aa]) ) as f:
				#with h5py.File(load + 'sky_select_img/result/%d_oppos_shlf_%s_band_%d_order.h5' % (d_record, band[tt], N_tot[aa]) ) as f:
				with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/%d_oppos_shlf_%s_band_%d_order.h5' % (d_record, band[tt], N_tot[aa]) ) as f:
				#with h5py.File(load + 'sky_select_img/test_set/0.65Mpc/%d_oppos_shlf_%s_band_%d_order.h5' % (d_record, band[tt], N_tot[aa]) ) as f:
					for ll in range(len(tmp_array)):
						f['a'][ll,:] = tmp_array[ll,:]

				tt_N = 0
				oppo_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				oppo_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				oppo_mean = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				oppo_media = np.zeros((len(Ny), len(Nx)), dtype = np.float)

				rand_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				rand_cnt = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				rand_px, rand_py = np.array([0]), np.array([0])
				rand_mean = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				rand_media = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				rand_mean_sqare = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				rand_media_sqare = np.zeros((len(Ny), len(Nx)), dtype = np.float)

				shlf_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				shlf_cnt = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				shlf_mean = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				shlf_media = np.zeros((len(Ny), len(Nx)), dtype = np.float)

				for pp in range(cpus):
					'''
					## shuffle case
					with h5py.File(tmp + 'sky_shuffle_sum_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						df_img = np.array(f['a'])
					with h5py.File(tmp + 'sky_shuffle_count_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						df_cnt = np.array(f['a'])

					with h5py.File(tmp + 'sky_shuffle_minus_mean_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						sub_shlf_mean = np.array(f['a'])
					with h5py.File(tmp + 'sky_shuffle_minus_media_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						sub_shlf_media = np.array(f['a'])

					sub_sum = np.nanmax(df_cnt)
					tt_N += sub_sum

					id_zero = df_cnt == 0
					ivx = id_zero == False
					shlf_img[ivx] = shlf_img[ivx] + df_img[ivx]
					shlf_cnt[ivx] = shlf_cnt[ivx] + df_cnt[ivx]

					shlf_mean[ivx] = shlf_mean[ivx] + sub_shlf_mean[ivx]
					shlf_media[ivx] = shlf_media[ivx] + sub_shlf_media[ivx]

					## oppose case
					with h5py.File(tmp + 'sky_oppose_sum_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						sum_img = np.array(f['a'])
					with h5py.File(tmp + 'sky_oppose_count_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						p_count = np.array(f['a'])

					with h5py.File(tmp + 'sky_oppose_minus_mean_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						sub_op_mean = np.array(f['a'])
					with h5py.File(tmp + 'sky_oppose_minus_media_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						sub_op_media = np.array(f['a'])

					id_zero = p_count == 0
					ivx = id_zero == False
					oppo_img[ivx] = oppo_img[ivx] + sum_img[ivx]
					oppo_count[ivx] = oppo_count[ivx] + p_count[ivx]

					oppo_mean[ivx] = oppo_mean[ivx] + sub_op_mean[ivx]
					oppo_media[ivx] = oppo_media[ivx] + sub_op_media[ivx]
					'''
					## random center case
					with h5py.File(tmp + 'rndm_sum_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						rndm_sum = np.array(f['a'])
					with h5py.File(tmp + 'rndm_sum_pcount_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						rndm_pcont = np.array(f['a'])

					with h5py.File(tmp + 'sky_random_minus_mean_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						sub_rndm_mean = np.array(f['a'])
					with h5py.File(tmp + 'sky_random_minus_media_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						sub_rndm_media = np.array(f['a'])

					with h5py.File(tmp + 'sky_random_mean_Var_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						sqare_f0 = np.array(f['a'])
					with h5py.File(tmp + 'sky_random_media_Var_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						sqare_f1 = np.array(f['a'])

					sub_sum = rndm_pcont[-1, -1]
					tt_N += sub_sum
					id_zero = rndm_pcont == 0
					ivx = id_zero == False
					rand_img[ivx] = rand_img[ivx] + rndm_sum[ivx]
					rand_cnt[ivx] = rand_cnt[ivx] + rndm_pcont[ivx]

					rand_mean[ivx] = rand_mean[ivx] + sub_rndm_mean[ivx]
					rand_media[ivx] = rand_media[ivx] + sub_rndm_media[ivx]

					rand_mean_sqare[ivx] = rand_mean_sqare[ivx] + sqare_f0[ivx]
					rand_media_sqare[ivx] = rand_media_sqare[ivx] + sqare_f1[ivx]

					with h5py.File(tmp + 'rdnm_pos_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						rndm_pos = np.array(f['a'])
					pos_x, pos_y = rndm_pos[0,:], rndm_pos[1,:]
					rand_px = np.r_[rand_px, pos_x]
					rand_py = np.r_[rand_py, pos_y]

				## oppose case
				tt_N = np.int(tt_N)
				'''
				id_zero = oppo_count == 0
				oppo_img[id_zero] = np.nan
				oppo_count[id_zero] = np.nan
				oppo_stack = oppo_img / oppo_count
				id_inf = np.isinf(oppo_stack)
				oppo_stack[id_inf] = np.nan

				oppo_m_mean = oppo_mean / oppo_count
				id_inf = np.isinf(oppo_m_mean)
				oppo_m_mean[id_inf] = np.nan

				oppo_m_media = oppo_media / oppo_count
				id_inf = np.isinf(oppo_m_media)
				oppo_m_media[id_inf] = np.nan	

				with h5py.File(load + 'sky/cluster/%d_sky_oppose_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				#with h5py.File(load + 'sky_select_img/result/sky_oppose_%d_imgs_%s_band.h5' % (tt_N, band[tt]), 'w') as f:
					f['a'] = np.array(oppo_stack)
				with h5py.File(load + 'sky/cluster/%d_sky_oppose_mean_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				#with h5py.File(load + 'sky_select_img/result/sky_oppose_mean_%d_imgs_%s_band.h5' % (tt_N, band[tt]), 'w') as f:
					f['a'] = np.array(oppo_m_mean)
				with h5py.File(load + 'sky/cluster/%d_sky_oppose_media_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				#with h5py.File(load + 'sky_select_img/result/sky_oppose_media_%d_imgs_%s_band.h5' % (tt_N, band[tt]), 'w') as f:
					f['a'] = np.array(oppo_m_media)

				## shuffle center
				id_zero = shlf_cnt == 0
				shlf_img[id_zero] = np.nan
				shlf_cnt[id_zero] = np.nan
				shlf_stack = shlf_img / shlf_cnt
				id_inf = np.isinf(shlf_stack)
				shlf_stack[id_inf] = np.nan

				shlf_m_mean = shlf_mean / shlf_cnt
				id_inf = np.isinf(shlf_m_mean)
				shlf_m_mean[id_inf] = np.nan

				shlf_m_media = shlf_media / shlf_cnt
				id_inf = np.isinf(shlf_m_media)
				shlf_m_media[id_inf] = np.nan

				#with h5py.File(load + 'sky/cluster/%d_sky_shuffle_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				#with h5py.File(load + 'sky_select_img/result/%d_sky_shuffle_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				with h5py.File(load + 'sky_select_img/test_set/%d_sky_shuffle_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
					f['a'] = np.array(shlf_stack)

				#with h5py.File(load + 'sky/cluster/%d_sky_shuffle_mean_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				#with h5py.File(load + 'sky_select_img/result/%d_sky_shuffle_mean_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				with h5py.File(load + 'sky_select_img/test_set/%d_sky_shuffle_mean_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
					f['a'] = np.array(shlf_m_mean)

				#with h5py.File(load + 'sky/cluster/%d_sky_shuffle_media_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				#with h5py.File(load + 'sky_select_img/result/%d_sky_shuffle_media_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				with h5py.File(load + 'sky_select_img/test_set/%d_sky_shuffle_media_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
					f['a'] = np.array(shlf_m_media)

				'''
				## random center
				id_zero = rand_cnt == 0
				rand_img[id_zero] = np.nan
				rand_cnt[id_zero] = np.nan
				random_stack = rand_img / rand_cnt
				id_inf = np.isinf(random_stack)
				random_stack[id_inf] = np.nan

				rand_m_mean = rand_mean / rand_cnt
				id_inf = np.isinf(rand_m_mean)
				rand_m_mean[id_inf] = np.nan

				rand_m_media = rand_media / rand_cnt
				id_inf = np.isinf(rand_m_media)
				rand_m_media[id_inf] = np.nan

				rand_mean_sqare[id_zero] = np.nan
				mean_E_f2 = rand_mean_sqare / rand_cnt
				id_inf = np.isinf(mean_E_f2)
				mean_E_f2[id_inf] = np.nan
				rand_mean_Var = mean_E_f2 - rand_m_mean**2

				rand_media_sqare[id_zero] = np.nan
				media_E_f2 = rand_media_sqare / rand_cnt
				id_inf = np.isinf(media_E_f2)
				media_E_f2[id_inf] = np.nan
				rand_media_Var = media_E_f2 - rand_m_media**2

				#with h5py.File(load + 'sky/cluster/%d_sky_random_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				#with h5py.File(load + 'sky_select_img/result/%d_sky_random_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/%d_sky_random_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				#with h5py.File(load + 'sky_select_img/test_set/0.65Mpc/%d_sky_random_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
					f['a'] = np.array(random_stack)

				#with h5py.File(load + 'sky/cluster/%d_sky_random_mean_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				#with h5py.File(load + 'sky_select_img/result/%d_sky_random_mean_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/%d_sky_random_mean_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				#with h5py.File(load + 'sky_select_img/test_set/0.65Mpc/%d_sky_random_mean_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
					f['a'] = np.array(rand_m_mean)

				#with h5py.File(load + 'sky/cluster/%d_sky_random_media_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				#with h5py.File(load + 'sky_select_img/result/%d_sky_random_media_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/%d_sky_random_media_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				#with h5py.File(load + 'sky_select_img/test_set/0.65Mpc/%d_sky_random_media_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
					f['a'] = np.array(rand_m_media)

				with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/%d_sky_random_media_Var_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
					f['a'] = np.array(rand_media_Var)
				with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/%d_sky_random_mean_Var_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
					f['a'] = np.array(rand_mean_Var)

				## position
				rand_px, rand_py = rand_px[1:], rand_py[1:]
				rand_pos = np.array([rand_px, rand_py])
				#with h5py.File(load + 'sky/cluster/%d_sky_random-pos_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				#with h5py.File(load + 'sky_select_img/result/%d_sky_random-pos_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/%d_sky_random-pos_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
				#with h5py.File(load + 'sky_select_img/test_set/0.65Mpc/%d_sky_random-pos_%d_imgs_%s_band.h5' % (d_record, tt_N, band[tt]), 'w') as f:
					f['a'] = np.array(rand_pos)

			commd.Barrier()
	raise
	## set color range
	color_lim = np.array([  [0.55, 0.22, 1.025, 0.170, 3.2], 
							[0.65, 0.26, 1.225, 0.205, 3.9] ])
	color_sym = np.array([  [-0.05, -0.02, -0.1, -0.025, -0.2], 
							[ 0.05,  0.02,  0.1,  0.025,  0.2]])

	#N_sum = np.array([3308, 3309, 3295, 3308, 3305]) ## after selecting
	#N_sum = np.array([2013, 2008, 2002, 2008, 2009]) ## sky-select(BCG location) sample
	#N_sum = np.array([1291, 1286, 1283, 1294, 1287]) ## 0.8Mpc
	N_sum = np.array([876,  873,  872,  878,  877 ]) ## 0.65Mpc
	'''
	#### subtracted case
	if rank == 0:
		for tt in range(3):

			with h5py.File(load + 'sky/cluster/sky_minus_media_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'r') as f:
			#with h5py.File(load + 'sky/cluster/sky_minus_mean_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'r') as f:
				BCG_minus_img = np.array(f['a'])

			shlf_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
			shlf_minus_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
			shlf_cnt = np.zeros((len(Ny), len(Nx)), dtype = np.float)
			for id_rec in range(1, 6):

				with h5py.File(load + 'sky/cluster/%d_sky_shuffle_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:
				#with h5py.File(load + 'sky/cluster/%d_sky_random_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:
					sub_img = np.array(f['a'])
				id_nan = np.isnan(sub_img)
				id_fals = id_nan == False
				shlf_img[id_fals] = shlf_img[id_fals] + sub_img[id_fals]
				shlf_cnt[id_fals] = shlf_cnt[id_fals] + 1.

				with h5py.File(load + 'sky/cluster/%d_sky_shuffle_media_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:
				#with h5py.File(load + 'sky/cluster/%d_sky_shuffle_mean_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:
				#with h5py.File(load + 'sky/cluster/%d_sky_random_media_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:
				#with h5py.File(load + 'sky/cluster/%d_sky_random_mean_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:
					sub_minus_img = np.array(f['a'])
				shlf_minus_img[id_fals] = shlf_minus_img[id_fals] + sub_minus_img[id_fals]

			shulf_img = shlf_img / shlf_cnt
			shulf_minus_img = shlf_minus_img / shlf_cnt

			with h5py.File(load + 'sky/cluster/mean_sky_shuffle_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:
			#with h5py.File(load + 'sky/cluster/mean_sky_random_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:
				f['a'] = np.array(shulf_img)

			with h5py.File(load + 'sky/cluster/mean_sky_shuffle_media_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:
			#with h5py.File(load + 'sky/cluster/mean_sky_shuffle_mean_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:
			#with h5py.File(load + 'sky/cluster/mean_sky_random_media_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:
			#with h5py.File(load + 'sky/cluster/mean_sky_random_mean_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:
				f['a'] = np.array(shulf_minus_img)

			plt.figure(figsize = (18, 6))
			ax0 = plt.subplot(131)
			ax1 = plt.subplot(132)
			ax2 = plt.subplot(133)

			clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5,)
			#ax0.set_title('%s band %d imgs centered on BCG [subtracted average]' % (band[tt], N_sum[tt]))
			ax0.set_title('%s band %d imgs centered on BCG [subtracted median]' % (band[tt], N_sum[tt]),)
			tf = ax0.imshow(BCG_minus_img, origin = 'lower', cmap = 'seismic', vmin = -2e-4, vmax = 2e-4,)
			ax0.add_patch(clust)
			ax0.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
			ax0.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
			plt.colorbar(tf, ax = ax0, fraction = 0.046, pad = 0.01, label = '$ flux \, [nmaggies]$')

			clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5,)
			#ax1.set_title('%s band %d imgs shuffle case [subtracted average]' % (band[tt], N_sum[tt]),)
			ax1.set_title('%s band %d imgs shuffle case [subtracted median]' % (band[tt], N_sum[tt]),)
			#ax1.set_title('%s band %d imgs random case [subtracted average]' % (band[tt], N_sum[tt]),)
			#ax1.set_title('%s band %d imgs random case [subtracted median]' % (band[tt], N_sum[tt]),)

			tf = ax1.imshow(shulf_minus_img, origin = 'lower', cmap = 'seismic', vmin = -2e-4, vmax = 2e-4,)
			ax1.add_patch(clust)
			ax1.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
			ax1.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
			plt.colorbar(tf, ax = ax1, fraction = 0.046, pad = 0.01, label = '$ flux \, [nmaggies]$')

			clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5,)
			#ax2.set_title('centered on BCG minus shuffle case [subtracted average]',)
			ax2.set_title('centered on BCG minus shuffle case [subtracted median]',)
			#ax2.set_title('centered on BCG minus random case [subtracted average]',)
			#ax2.set_title('centered on BCG minus random case [subtracted median]',)

			tf = ax2.imshow(BCG_minus_img - shulf_minus_img, origin = 'lower', cmap = 'seismic', vmin = -2e-4, vmax = 2e-4,)
			ax2.add_patch(clust)
			ax2.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
			ax2.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
			plt.colorbar(tf, ax = ax2, fraction = 0.046, pad = 0.01, label = '$ flux \, [nmaggies]$')

			plt.tight_layout()
			plt.subplots_adjust(bottom = 0.1, right = 0.9, top = 0.9)

			plt.savefig(load + 'sky/cluster/BCG-shuffle_%d_imgs_%s_band_minus_media.png' % ( N_sum[tt], band[tt]), dpi = 300)
			#plt.savefig(load + 'sky/cluster/BCG-shuffle_%d_imgs_%s_band_minus_mean.png' % ( N_sum[tt], band[tt]), dpi = 300)
			#plt.savefig(load + 'sky/cluster/BCG-random_%d_imgs_%s_band_minus_media.png' % ( N_sum[tt], band[tt]), dpi = 300)
			#plt.savefig(load + 'sky/cluster/BCG-random_%d_imgs_%s_band_minus_mean.png' % ( N_sum[tt], band[tt]), dpi = 300)
			plt.close()

	commd.Barrier()
	'''
	#### more center sample
	if rank == 0:
		for tt in range(3):

			#with h5py.File(load + 'sky_select_img/result/sky_minus_media_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'r') as f:
			#with h5py.File(load + 'sky_select_img/result/sky_minus_mean_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'r') as f:

			#with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/sky_minus_media_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'r') as f:
			#with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/sky_minus_mean_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'r') as f:

			#with h5py.File(load + 'sky_select_img/test_set/0.65Mpc/sky_minus_media_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'r') as f:
			with h5py.File(load + 'sky_select_img/test_set/0.65Mpc/sky_minus_mean_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'r') as f:
				BCG_minus_img = np.array(f['a'])

			shlf_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
			shlf_minus_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
			shlf_cnt = np.zeros((len(Ny), len(Nx)), dtype = np.float)
			for id_rec in range(1, 6):

				#with h5py.File(load + 'sky_select_img/result/%d_sky_shuffle_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:
				#with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/%d_sky_shuffle_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:
				with h5py.File(load + 'sky_select_img/test_set/0.65Mpc/%d_sky_shuffle_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:

				#with h5py.File(load + 'sky_select_img/result/%d_sky_random_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:
				#with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/%d_sky_random_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:
				#with h5py.File(load + 'sky_select_img/test_set/0.65Mpc/%d_sky_random_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:				
					sub_img = np.array(f['a'])
				id_nan = np.isnan(sub_img)
				id_fals = id_nan == False
				shlf_img[id_fals] = shlf_img[id_fals] + sub_img[id_fals]
				shlf_cnt[id_fals] = shlf_cnt[id_fals] + 1.
				## -- >
				#with h5py.File(load + 'sky_select_img/result/%d_sky_shuffle_media_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:
				#with h5py.File(load + 'sky_select_img/result/%d_sky_shuffle_mean_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:

				#with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/%d_sky_shuffle_media_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:
				#with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/%d_sky_shuffle_mean_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:

				#with h5py.File(load + 'sky_select_img/test_set/0.65Mpc/%d_sky_shuffle_media_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:
				with h5py.File(load + 'sky_select_img/test_set/0.65Mpc/%d_sky_shuffle_mean_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:
				## -- >
				#with h5py.File(load + 'sky_select_img/result/%d_sky_random_media_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:
				#with h5py.File(load + 'sky_select_img/result/%d_sky_random_mean_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:

				#with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/%d_sky_random_media_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:
				#with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/%d_sky_random_mean_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:

				#with h5py.File(load + 'sky_select_img/test_set/0.65Mpc/%d_sky_random_media_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:
				#with h5py.File(load + 'sky_select_img/test_set/0.65Mpc/%d_sky_random_mean_%d_imgs_%s_band.h5' % (id_rec, N_sum[tt], band[tt]), 'r') as f:
					sub_minus_img = np.array(f['a'])
				shlf_minus_img[id_fals] = shlf_minus_img[id_fals] + sub_minus_img[id_fals]

			shulf_img = shlf_img / shlf_cnt
			shulf_minus_img = shlf_minus_img / shlf_cnt

			#with h5py.File(load + 'sky_select_img/result/mean_sky_shuffle_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:
			#with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/mean_sky_shuffle_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:
			with h5py.File(load + 'sky_select_img/test_set/0.65Mpc/mean_sky_shuffle_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:

			#with h5py.File(load + 'sky_select_img/result/mean_sky_random_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:
			#with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/mean_sky_random_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:
			#with h5py.File(load + 'sky_select_img/test_set/0.65Mpc/mean_sky_random_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:
				f['a'] = np.array(shulf_img)
			## -- >
			#with h5py.File(load + 'sky_select_img/result/mean_sky_shuffle_media_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:
			#with h5py.File(load + 'sky_select_img/result/mean_sky_shuffle_mean_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:

			#with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/mean_sky_shuffle_media_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:
			#with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/mean_sky_shuffle_mean_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:

			#with h5py.File(load + 'sky_select_img/test_set/0.65Mpc/mean_sky_shuffle_media_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:
			with h5py.File(load + 'sky_select_img/test_set/0.65Mpc/mean_sky_shuffle_mean_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:
			## -- >
			#with h5py.File(load + 'sky_select_img/result/mean_sky_random_media_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:
			#with h5py.File(load + 'sky_select_img/result/mean_sky_random_mean_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:

			#with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/mean_sky_random_media_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:
			#with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/mean_sky_random_mean_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:

			#with h5py.File(load + 'sky_select_img/test_set/0.65Mpc/mean_sky_random_media_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:
			#with h5py.File(load + 'sky_select_img/test_set/0.65Mpc/mean_sky_random_mean_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'w') as f:
				f['a'] = np.array(shulf_minus_img)

			plt.figure(figsize = (18, 6))
			ax0 = plt.subplot(131)
			ax1 = plt.subplot(132)
			ax2 = plt.subplot(133)

			clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5,)
			ax0.set_title('%s band %d imgs centered on BCG [subtracted average]' % (band[tt], N_sum[tt]),)
			#ax0.set_title('%s band %d imgs centered on BCG [subtracted median]' % (band[tt], N_sum[tt]),)
			tf = ax0.imshow(BCG_minus_img, origin = 'lower', cmap = 'seismic', vmin = -2e-4, vmax = 2e-4,)
			ax0.add_patch(clust)
			ax0.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
			ax0.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
			plt.colorbar(tf, ax = ax0, fraction = 0.040, pad = 0.01, label = '$ flux \, [nmaggies]$')

			clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5,)
			ax1.set_title('%s band %d imgs shuffle case [subtracted average]' % (band[tt], N_sum[tt]),)
			#ax1.set_title('%s band %d imgs shuffle case [subtracted median]' % (band[tt], N_sum[tt]),)
			#ax1.set_title('%s band %d imgs random case [subtracted average]' % (band[tt], N_sum[tt]),)
			#ax1.set_title('%s band %d imgs random case [subtracted median]' % (band[tt], N_sum[tt]),)

			tf = ax1.imshow(shulf_minus_img, origin = 'lower', cmap = 'seismic', vmin = -2e-4, vmax = 2e-4,)
			ax1.add_patch(clust)
			ax1.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
			ax1.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
			plt.colorbar(tf, ax = ax1, fraction = 0.040, pad = 0.01, label = '$ flux \, [nmaggies]$')

			clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5,)
			ax2.set_title('centered on BCG minus shuffle case [subtracted average]',)
			#ax2.set_title('centered on BCG minus shuffle case [subtracted median]',)
			#ax2.set_title('centered on BCG minus random case [subtracted average]',)
			#ax2.set_title('centered on BCG minus random case [subtracted median]',)

			tf = ax2.imshow(BCG_minus_img - shulf_minus_img, origin = 'lower', cmap = 'seismic', vmin = -2e-4, vmax = 2e-4,)
			ax2.add_patch(clust)
			ax2.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
			ax2.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
			plt.colorbar(tf, ax = ax2, fraction = 0.040, pad = 0.01, label = '$ flux \, [nmaggies]$')

			plt.tight_layout()
			plt.subplots_adjust(bottom = 0.1, right = 0.9, top = 0.9)
			## -- >
			#plt.savefig(load + 'sky_select_img/result/BCG-shuffle_%d_imgs_%s_band_minus_media.png' % (N_sum[tt], band[tt]), dpi = 300)
			#plt.savefig(load + 'sky_select_img/result/BCG-shuffle_%d_imgs_%s_band_minus_mean.png' % (N_sum[tt], band[tt]), dpi = 300)

			#plt.savefig(load + 'sky_select_img/test_set/0.8Mpc/BCG-shuffle_%d_imgs_%s_band_minus_media.png' % (N_sum[tt], band[tt]), dpi = 300)
			#plt.savefig(load + 'sky_select_img/test_set/0.8Mpc/BCG-shuffle_%d_imgs_%s_band_minus_mean.png' % (N_sum[tt], band[tt]), dpi = 300)

			#plt.savefig(load + 'sky_select_img/test_set/0.65Mpc/BCG-shuffle_%d_imgs_%s_band_minus_media.png' % (N_sum[tt], band[tt]), dpi = 300)
			plt.savefig(load + 'sky_select_img/test_set/0.65Mpc/BCG-shuffle_%d_imgs_%s_band_minus_mean.png' % (N_sum[tt], band[tt]), dpi = 300)
			## -- >
			#plt.savefig(load + 'sky_select_img/result/BCG-random_%d_imgs_%s_band_minus_media.png' % (N_sum[tt], band[tt]), dpi = 300)
			#plt.savefig(load + 'sky_select_img/result/BCG-random_%d_imgs_%s_band_minus_mean.png' % (N_sum[tt], band[tt]), dpi = 300)

			#plt.savefig(load + 'sky_select_img/test_set/0.8Mpc/BCG-random_%d_imgs_%s_band_minus_media.png' % (N_sum[tt], band[tt]), dpi = 300)
			#plt.savefig(load + 'sky_select_img/test_set/0.8Mpc/BCG-random_%d_imgs_%s_band_minus_mean.png' % (N_sum[tt], band[tt]), dpi = 300)

			#plt.savefig(load + 'sky_select_img/test_set/0.65Mpc/BCG-random_%d_imgs_%s_band_minus_media.png' % (N_sum[tt], band[tt]), dpi = 300)
			#plt.savefig(load + 'sky_select_img/test_set/0.65Mpc/BCG-random_%d_imgs_%s_band_minus_mean.png' % (N_sum[tt], band[tt]), dpi = 300)
			plt.close()

	commd.Barrier()

if __name__ == "__main__":
	main()
