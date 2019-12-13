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
from scipy.ndimage import map_coordinates as mapcd
from resample_modelu import sum_samp, down_samp
from astropy.coordinates import SkyCoord
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
band = ['r', 'i', 'g', 'u', 'z']
sky_SB = [21.04, 20.36, 22.01, 22.30, 19.18] # ref_value from SDSS
mag_add = np.array([0, 0, 0, -0.04, 0.02])

def sky_oppose(band_id, sub_z, sub_ra, sub_dec):

	## stack sky image
	stack_N = len(sub_z)
	kk = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	sum_grid = np.array(np.meshgrid(Nx, Ny))

	sum_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

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

	with h5py.File(tmp + 'sky_oppose_sum_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(sum_array)

	with h5py.File(tmp + 'sky_oppose_count_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(p_count)

def sky_shuffle(band_id, sub_z, sub_ra, sub_dec, shlf_z, shlf_ra, shlf_dec):
	## stack sky image
	stack_N = len(sub_z)
	kk = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	sum_grid = np.array(np.meshgrid(Nx, Ny))

	sum_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	for jj in range(stack_N):
		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]

		pos_ra = shlf_ra[jj]
		pos_dec = shlf_dec[jj]
		pos_z = shlf_z[jj]
		## scaled image
		data = fits.open( load + 'sky/sky_resamp/resample_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g) )
		img = data[0].data
		## rule out the edge pixels (due to the errror in flux resampling)
		img[0,:] = np.nan
		img[-1,:] = np.nan
		img[:,0] = np.nan
		img[:,-1] = np.nan

		pos_data = fits.open( load + 'sky/sky_resamp/resample_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], pos_ra, pos_dec, pos_z) )
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
		idv = np.where(idx == False)

		sum_array[la0: la1, lb0: lb1][idv] = sum_array[la0:la1, lb0:lb1][idv] + img[idv]
		count_array[la0: la1, lb0: lb1][idv] = img[idv]
		id_nan = np.isnan(count_array)
		id_fals = np.where(id_nan == False)
		p_count[id_fals] = p_count[id_fals] + 1
		count_array[la0: la1, lb0: lb1][idv] = np.nan

	## save the random center data
	with h5py.File(tmp + 'sky_shuffle_sum_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(sum_array)

	with h5py.File(tmp + 'sky_shuffle_count_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(p_count)

def sky_fig_cen(band_id, sub_z, sub_ra, sub_dec):
	stack_N = len(sub_z)
	kk = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	sum_grid = np.array(np.meshgrid(Nx, Ny))

	sum_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

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

		rnx, rny = img.shape[1] / 2, img.shape[0] / 2
		la0 = np.int(y0 - rny)
		la1 = np.int(y0 - rny + img.shape[0])
		lb0 = np.int(x0 - rnx)
		lb1 = np.int(x0 - rnx + img.shape[1])

		idx = np.isnan(img)
		idv = np.where(idx == False)

		sum_array[la0: la1, lb0: lb1][idv] = sum_array[la0:la1, lb0:lb1][idv] + img[idv]
		count_array[la0: la1, lb0: lb1][idv] = img[idv]
		id_nan = np.isnan(count_array)
		id_fals = np.where(id_nan == False)
		p_count[id_fals] = p_count[id_fals] + 1
		count_array[la0: la1, lb0: lb1][idv] = np.nan

	with h5py.File(tmp + 'sky_fig-cen_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(sum_array)
	with h5py.File(tmp + 'sky_fig-cen_count_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(p_count)

def sky_rndm_cen(band_id, sub_z, sub_ra, sub_dec):
	stack_N = len(sub_z)
	kk = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	sum_grid = np.array(np.meshgrid(Nx, Ny))

	## random center as comparison
	rndm_sum = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	rndm_count = np.zeros((len(Ny), len(Nx)), dtype = np.float) * np.nan
	rndm_pcont = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	pos_x = np.zeros(stack_N, dtype = np.float)
	pos_y = np.zeros(stack_N, dtype = np.float)

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

def main():
	R_cut, bins = 1260, 80  # in unit of pixel (with 2Mpc inside)
	R_smal, R_max = 10, 1.7e3 # kpc

	## stack all the sky togather (without resampling)
	x0, y0, bins = 2427, 1765, 65
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	## set color range
	color_lim = np.array([  [0.55, 1.025, 0.22, 0.170, 3.2], 
							[0.65, 1.225, 0.26, 0.205, 3.9] ])
	color_sym = np.array([  [-0.05, -0.1, -0.02, -0.025, -0.2], 
							[ 0.05,  0.1,  0.02,  0.025,  0.2]])

	#N_tot = np.array([50, 100, 200, 500, 1000, 3000])
	for tt in range( len(band) ):
		with h5py.File(load + 'mpi_h5/%s_band_sample_catalog.h5' % band[tt], 'r') as f:
			sub_array = np.array(f['a'])
		ra, dec, z, rich, r_mag = sub_array[0,:], sub_array[1,:], sub_array[2,:], sub_array[3,:], sub_array[4,:]
		zN = len(z)
		N_tot = np.array([zN])

		for aa in range( len(N_tot) ):

			## random sample
			#np.random.seed(1)
			tt0 = np.random.choice(zN, size = N_tot[aa], replace = False)
			set_z = z[tt0]
			set_ra = ra[tt0]
			set_dec = dec[tt0]

			#np.random.seed(2)
			tt1 = np.random.choice(zN, size = N_tot[aa], replace = False)
			shif_z = z[tt1]
			shif_ra = ra[tt1]
			shif_dec = dec[tt1]

			m, n = divmod(N_tot[aa], cpus)
			N_sub0, N_sub1 = m * rank, (rank + 1) * m
			if rank == cpus - 1:
				N_sub1 += n
			sky_oppose(tt, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])
			#sky_fig_cen(tt, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])

			sky_shuffle(tt, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1], 
				shif_z[N_sub0 :N_sub1], shif_ra[N_sub0 :N_sub1], shif_dec[N_sub0 :N_sub1])
			sky_rndm_cen(tt, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])
			commd.Barrier()

			if rank == 0:
				## record the order of set_z....and shif_z..., this is the shuffle record
				## and the former part is also order for oppose case stacking
				tmp_array = np.array([set_z, set_ra, set_dec, shif_z, shif_ra, shif_dec])
				with h5py.File(load + 'test_h5/oppos_shlf_%s_band_%d_imgs.h5' % (band[tt], N_tot[aa]), 'w') as f:
					f['a'] = np.array(tmp_array)
				with h5py.File(load + 'test_h5/oppos_shlf_%s_band_%d_imgs.h5' % (band[tt], N_tot[aa]) ) as f:
					for ll in range(len(tmp_array)):
						f['a'][ll,:] = tmp_array[ll,:]

				tot_N = 0

				mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				'''
				fig_cimg = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				fig_cont = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				'''
				shlf_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				shlf_cnt = np.zeros((len(Ny), len(Nx)), dtype = np.float)

				rand_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				rand_cnt = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				rand_px, rand_py = np.array([0]), np.array([0])

				for pp in range(cpus):
					## oppose case
					with h5py.File(tmp + 'sky_oppose_sum_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						sum_img = np.array(f['a'])

					with h5py.File(tmp + 'sky_oppose_count_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						p_count = np.array(f['a'])

					sub_Num = np.nanmax(p_count)
					tot_N += sub_Num
					id_zero = p_count == 0
					ivx = id_zero == False
					mean_img[ivx] = mean_img[ivx] + sum_img[ivx]
					p_add_count[ivx] = p_add_count[ivx] + p_count[ivx]

					## shuffle case
					with h5py.File(tmp + 'sky_shuffle_sum_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						df_img = np.array(f['a'])

					with h5py.File(tmp + 'sky_shuffle_count_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						df_cnt = np.array(f['a'])

					id_zero = df_cnt == 0
					ivx = id_zero == False
					shlf_img[ivx] = shlf_img[ivx] + df_img[ivx]
					shlf_cnt[ivx] = shlf_cnt[ivx] + df_cnt[ivx]
					'''
					## fig-cen case
					with h5py.File(tmp + 'sky_fig-cen_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						fig_cen_img = np.array(f['a'])
					with h5py.File(tmp + 'sky_fig-cen_count_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						fig_cen_cnt = np.array(f['a'])

					id_zero = fig_cen_cnt == 0
					ivx = id_zero == False
					fig_cimg[ivx] = fig_cimg[ivx] + fig_cen_img[ivx]
					fig_cont[ivx] = fig_cont[ivx] + fig_cen_cnt[ivx]
					'''
					## random center case
					with h5py.File(tmp + 'rndm_sum_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						rndm_sum = np.array(f['a'])
					with h5py.File(tmp + 'rndm_sum_pcount_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						rndm_pcont = np.array(f['a'])
					id_zero = rndm_pcont == 0
					ivx = id_zero == False
					rand_img[ivx] = rand_img[ivx] + rndm_sum[ivx]
					rand_cnt[ivx] = rand_cnt[ivx] + rndm_pcont[ivx]

					with h5py.File(tmp + 'rdnm_pos_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						rndm_pos = np.array(f['a'])
					pos_x, pos_y = rndm_pos[0,:], rndm_pos[1,:]
					rand_px = np.r_[rand_px, pos_x]
					rand_py = np.r_[rand_py, pos_y]

				## centered on BCG
				tot_N = np.int(tot_N)
				id_zero = p_add_count == 0
				mean_img[id_zero] = np.nan
				p_add_count[id_zero] = np.nan
				stack_img = mean_img / p_add_count
				id_inf = np.isinf(stack_img)
				stack_img[id_inf] = np.nan
				with h5py.File(load + 'sky/sky_oppose_%d_imgs_%s_band.h5' % (N_tot[aa], band[tt]), 'w') as f:
					f['a'] = np.array(stack_img)

				## shuffle center
				id_zero = shlf_cnt == 0
				shlf_img[id_zero] = np.nan
				shlf_cnt[id_zero] = np.nan
				mean_shlf = shlf_img / shlf_cnt
				id_inf = np.isinf(mean_shlf)
				mean_shlf[id_inf] = np.nan
				with h5py.File(load + 'sky/sky_shuffle_%d_imgs_%s_band.h5' % (N_tot[aa], band[tt]), 'w') as f:
					f['a'] = np.array(mean_shlf)
				'''
				## fig-center case
				id_zero = fig_cont == 0
				fig_cimg[id_zero] = np.nan
				fig_cont[id_zero] = np.nan
				mean_fig_cimg = fig_cimg / fig_cont
				id_inf = np.isinf(mean_fig_cimg)
				mean_fig_cimg[id_inf] = np.nan
				with h5py.File(load + 'sky/sky_fig-center_%d_imgs_%s_band.h5' % (N_tot[aa], band[tt]), 'w') as f:
					f['a'] = np.array(mean_fig_cimg)
				'''
				## random center
				id_zero = rand_cnt == 0
				rand_img[id_zero] = np.nan
				rand_cnt[id_zero] = np.nan
				random_cen = rand_img / rand_cnt
				id_inf = np.isinf(random_cen)
				random_cen[id_inf] = np.nan
				with h5py.File(load + 'sky/sky_random_%d_imgs_%s_band.h5' % (N_tot[aa], band[tt]), 'w') as f:
					f['a'] = np.array(random_cen)
				## position
				rand_px, rand_py = rand_px[1:], rand_py[1:]
				rand_pos = np.array([rand_px, rand_py])
				with h5py.File(load + 'test_h5/sky_random-pos_%d_imgs_%s_band.h5' % (N_tot[aa], band[tt]), 'w') as f:
					f['a'] = np.array(rand_pos)

			commd.Barrier()

	N_sum = np.array([3378, 3363, 3377, 3378, 3372])
	if rank == 0:
		for tt in range(len(band)):
			## results
			with h5py.File(load + 'sky/sky_oppose_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'r') as f:
				oppos_img = np.array(f['a'])
			with h5py.File(load + 'sky/sky_shuffle_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'r') as f:
				shulf_img = np.array(f['a'])
			with h5py.File(load + 'sky/sky_random_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'r') as f:
				random_cen = np.array(f['a'])
			## BCG case
			with h5py.File(load + 'sky/sky_stack_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'r') as f:
				BCG_img = np.array(f['a'])

			plt.figure(figsize = (18, 6))
			ax0 = plt.subplot(231)
			ax1 = plt.subplot(232)
			ax2 = plt.subplot(233)

			bx0 = plt.subplot(234)
			bx1 = plt.subplot(235)
			bx2 = plt.subplot(236)

			clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5, label = 'cluster region[1Mpc]')
			ax0.set_title( 'stack [%d] sky img %s band [centered on BCG]' % (N_sum[tt], band[tt] ) )
			tf = ax0.imshow(BCG_img, origin = 'lower', vmin = color_lim[0][tt], vmax = color_lim[1][tt], )
			ax0.add_patch(clust)
			ax0.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
			ax0.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
			plt.colorbar(tf, ax = ax0, fraction = 0.040, pad = 0.01, label = '$ flux \, [nmaggies]$')

			ax1.set_title( 'stack [%d] sky img %s band [oppose case]' % (N_sum[tt], band[tt] ) )
			tf = ax1.imshow(oppos_img, origin = 'lower', vmin = color_lim[0][tt], vmax = color_lim[1][tt], )
			clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5, label = 'cluster region[1Mpc]')
			ax1.add_patch(clust)
			ax1.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
			ax1.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
			plt.colorbar(tf, ax = ax1, fraction = 0.040, pad = 0.01, label = '$ flux \, [nmaggies]$')

			ax2.set_title( 'stack [%d] sky img %s band [shuffle case]' % (N_sum[tt], band[tt] ) )
			tf = ax2.imshow(shulf_img, origin = 'lower', vmin = color_lim[0][tt], vmax = color_lim[1][tt], )
			clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5, label = 'cluster region[1Mpc]')
			ax2.add_patch(clust)
			ax2.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
			ax2.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
			plt.colorbar(tf, ax = ax2, fraction = 0.040, pad = 0.01, label = '$ flux \, [nmaggies]$')

			bx0.set_title( 'stack [%d] sky img %s band [random center]' % (N_sum[tt], band[tt] ) )
			tf = bx0.imshow(random_cen, origin = 'lower', vmin = color_lim[0][tt], vmax = color_lim[1][tt], )
			clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5, label = 'cluster size region[1Mpc]')
			bx0.add_patch(clust)
			bx0.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
			bx0.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
			plt.colorbar(tf, ax = bx0, fraction = 0.040, pad = 0.01, label = '$ flux \, [nmaggies]$')	

			bx1.set_title( 'centered on BCG minus oppose case' )
			tf = bx1.imshow(BCG_img - oppos_img, origin = 'lower', cmap = 'seismic', vmin = color_sym[0][tt], vmax = color_sym[1][tt],)
			clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5, label = 'cluster region[1Mpc]')
			bx1.add_patch(clust)
			bx1.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
			bx1.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
			plt.colorbar(tf, ax = bx1, fraction = 0.040, pad = 0.01, label = '$ flux \, [nmaggies]$')

			bx2.set_title( 'centered on BCG minus shuffle case' )
			tf = bx2.imshow(BCG_img - shulf_img, origin = 'lower', cmap = 'seismic', vmin = color_sym[0][tt], vmax = color_sym[1][tt],)
			clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5, label = 'cluster region[1Mpc]')
			bx2.add_patch(clust)
			bx2.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
			bx2.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
			plt.colorbar(tf, ax = bx2, fraction = 0.040, pad = 0.01, label = '$ flux \, [nmaggies]$')

			plt.tight_layout()
			plt.subplots_adjust(bottom = 0.1, right = 0.8, top = 0.9)
			plt.savefig(load + 'sky/sky_%d_imgs_%s_band_flat_test.png' % (N_sum[tt], band[tt]), dpi = 300)
			plt.close()

			plt.figure()
			ax0 = plt.subplot(111)
			ax0.set_title('%s band random center minus shuffle BCG [%d imgs]' % (band[tt], N_sum[tt]) )
			tf = ax0.imshow(random_cen - shulf_img, cmap = 'seismic', origin = 'lower', vmin = -0.1, vmax = 0.1,)
			clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5, label = 'cluster region[1Mpc]')
			ax0.add_patch(clust)
			ax0.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
			ax0.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
			plt.colorbar(tf, ax = ax0, fraction = 0.040, pad = 0.01, label = '$ flux \, [nmaggies]$')
			#plt.subplots_adjust(bottom = 0.1, right = 0.8, top = 0.9)
			plt.savefig(load + 'sky/sky_%d_imgs_%s_band_random_devi.png' % (N_sum[tt], band[tt]), dpi = 300)
			plt.close()

	commd.Barrier()

if __name__ == "__main__":
	main()
