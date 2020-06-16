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
from light_measure import light_measure, flux_recal
from scipy.stats import binned_statistic as binned

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

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

home = '/mnt/ddnfs/data_users/cxkttwl/ICL/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'
band = ['r', 'g', 'i', 'u', 'z']

#def rand_pont(band_id, sub_z, sub_ra, sub_dec, img_x, img_y):
def rand_pont(band_id, sub_z, sub_ra, sub_dec, ):

	stack_N = len(sub_z)
	kk = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	id_nm = 0.
	for jj in range(stack_N):

		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]
		#xn, yn = img_x[jj], img_y[jj]

		data_A = fits.open(load + 'random_cat/mask_no_dust/random_mask_%s_ra%.3f_dec%.3f_z%.3f.fits' % (band[kk], ra_g, dec_g, z_g)) ## just masking

		img_A = data_A[0].data
		head = data_A[0].header
		wcs_lis = awc.WCS(head)
		xn, yn = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

		# centered on cat.(ra, dec)
		la0 = np.int(y0 - yn)
		la1 = np.int(y0 - yn + img_A.shape[0])
		lb0 = np.int(x0 - xn)
		lb1 = np.int(x0 - xn + img_A.shape[1])
		'''
		#rnx, rny = np.random.choice(img_A.shape[1], 1, replace = False), np.random.choice(img_A.shape[0], 1, replace = False) ## random center
		rnx, rny = np.int(img_A.shape[1] / 2), np.int(img_A.shape[0] / 2) ## image center
		la0 = np.int(y0 - rny)
		la1 = np.int(y0 - rny + img_A.shape[0])
		lb0 = np.int(x0 - rnx)
		lb1 = np.int(x0 - rnx + img_A.shape[1])
		'''
		idx = np.isnan(img_A)
		idv = np.where(idx == False)

		sum_array_A[la0: la1, lb0: lb1][idv] = sum_array_A[la0: la1, lb0: lb1][idv] + img_A[idv]
		count_array_A[la0: la1, lb0: lb1][idv] = img_A[idv]
		id_nan = np.isnan(count_array_A)
		id_fals = np.where(id_nan == False)
		p_count_A[id_fals] = p_count_A[id_fals] + 1.
		count_array_A[la0: la1, lb0: lb1][idv] = np.nan
		id_nm += 1.

	p_count_A[0, 0] = id_nm
	with h5py.File(tmp + 'stack_mask_sum_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(sum_array_A)
	with h5py.File(tmp + 'stack_mask_pcount_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(p_count_A)

	return

def sky_stack(band_id, sub_z, sub_ra, sub_dec):
    stack_N = len(sub_z)
    kk = np.int(band_id)

    x0, y0 = 2427, 1765
    Nx = np.linspace(0, 4854, 4855)
    Ny = np.linspace(0, 3530, 3531)

    sum_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
    count_array = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
    p_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

    id_nm = 0
    for jj in range(stack_N):
        ra_g = sub_ra[jj]
        dec_g = sub_dec[jj]
        z_g = sub_z[jj]

        data = fits.open(load + 'random_cat/sky_img/rand_sky-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[kk]) )
        img = data[0].data
        head = data[0].header
        wcs_lis = awc.WCS(head)
        cx, cy = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)
        '''
        ## catalog (ra, dec)
        la0 = np.int(y0 - cy)
        la1 = np.int(y0 - cy + img.shape[0])
        lb0 = np.int(x0 - cx)
        lb1 = np.int(x0 - cx + img.shape[1])
        '''
        ## image frame center / random center
        #rnx, rny = np.random.choice(img.shape[1], 1, replace = False), np.random.choice(img.shape[0], 1, replace = False)
        rnx, rny = np.int(img.shape[1] / 2), np.int(img.shape[0] / 2)
        la0 = np.int(y0 - rny)
        la1 = np.int(y0 - rny + img.shape[0])
        lb0 = np.int(x0 - rnx)
        lb1 = np.int(x0 - rnx + img.shape[1])

        idx = np.isnan(img)
        idv = np.where(idx == False)

        img = img - np.nanmedian(img)

        sum_array[la0: la1, lb0: lb1][idv] = sum_array[la0:la1, lb0:lb1][idv] + img[idv]
        count_array[la0: la1, lb0: lb1][idv] = img[idv]
        id_nan = np.isnan(count_array)
        id_fals = np.where(id_nan == False)
        p_count[id_fals] = p_count[id_fals] + 1
        count_array[la0: la1, lb0: lb1][idv] = np.nan
        id_nm += 1.

    p_count[0, 0] = id_nm
    with h5py.File(tmp + 'sky_sum_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
        f['a'] = np.array(sum_array)
    with h5py.File(tmp + 'sky_sum_pcount_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
        f['a'] = np.array(p_count)
    return

def main():

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	## use r band only
	for kk in range( 1 ):

		with h5py.File(load + 'random_cat/cat_select/rand_%s_band_catalog.h5' % (band[kk]), 'r') as f:
			tmp_array = np.array(f['a'])
		ra, dec, z, rich = np.array(tmp_array[0]), np.array(tmp_array[1]), np.array(tmp_array[2]), np.array(tmp_array[3])

		### apply the image position of BCGs
		#with h5py.File(home + 'tmp_stack/r_band_cluster_img-position.h5', 'r') as f:
		with h5py.File(home + 'tmp_stack/jack_random/copy_r_band_cluster_img-coord.h5', 'r') as f:
			clus_x = np.array(f['x'])
			clus_y = np.array(f['y'])
		set_x, set_y = clus_x, clus_y

		DN = len(z) ##1000, 4500
		set_ra, set_dec, set_z = ra[:DN], dec[:DN], z[:DN]

		#### test part
		#offD = 200
		#set_x, set_y = np.random.randint(1024 - offD, 1024 + offD, DN), np.random.randint(745 - offD, 745 + offD, DN)
		#np.random.seed(2)
		#tt0 = np.random.choice(4500, DN, replace = False)
		#set_ra, set_dec, set_z = ra[tt0], dec[tt0], z[tt0]

		zN = len(set_z)
		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		rand_pont(kk, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1], )
		#rand_pont(kk, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1], set_x[N_sub0 :N_sub1], set_y[N_sub0 :N_sub1] )
		commd.Barrier()

		if rank == 0:

			tot_N = 0.
			mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
			p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

			for pp in range(cpus):

				with h5py.File(tmp + 'stack_mask_pcount_%d_in_%s_band.h5' % (pp, band[kk]), 'r')as f:
					p_count = np.array(f['a'])

				with h5py.File(tmp + 'stack_mask_sum_%d_in_%s_band.h5' % (pp, band[kk]), 'r') as f:
					sum_img = np.array(f['a'])

				id_zero = p_count == 0
				ivx = id_zero == False
				mean_img[ivx] = mean_img[ivx] + sum_img[ivx]
				p_add_count[ivx] = p_add_count[ivx] + p_count[ivx]
				tot_N += p_count[0, 0]

				## save sub-sample sky
				sub_mean = sum_img / p_count
				id_zero = sub_mean == 0.
				id_inf = np.isinf(sub_mean)
				sub_mean[id_zero] = np.nan
				sub_mean[id_inf] = np.nan

				#with h5py.File(home + 'tmp_stack/%s_band_center-stack_random-img_%d-sub-sample.h5' % (band[kk], pp), 'w') as f:
				#with h5py.File(home + 'tmp_stack/%s_band_cat-stack_random-img_%d-sub-sample.h5' % (band[kk], pp), 'w') as f:

				## sub-sample for jackknife
				with h5py.File(home + 'tmp_stack/jack_random/rand_cat/%s_band_cat-stack_random-img_%d-sub-smp.h5' % (band[kk], pp), 'w') as f:
				#with h5py.File(home + 'tmp_stack/jack_random/apply_bcgs/%s_band_bcgs-stack_random-img_%d-sub-smp.h5' % (band[kk], pp), 'w') as f:
					f['a'] = np.array(sub_mean)

			tot_N = np.int(tot_N)
			id_zero = p_add_count == 0
			mean_img[id_zero] = np.nan
			p_add_count[id_zero] = np.nan
			stack_img = mean_img / p_add_count
			where_are_inf = np.isinf(stack_img)
			stack_img[where_are_inf] = np.nan

			#with h5py.File(home + 'tmp_stack/%s_band_center-stack_random-img.h5' % band[kk], 'w') as f:
			#with h5py.File(home + 'tmp_stack/%s_band_cat-stack_random-img.h5' % band[kk], 'w') as f:
			#with h5py.File(home + 'tmp_stack/%s_band_cat-stack_random-img_clus-position.h5' % band[kk], 'w') as f:

			#with h5py.File(home + 'tmp_stack/%s_band_cat-stack_random-img_pix-r_select_%d.h5' % (band[kk], offD), 'w') as f:

			### random choice case
			#with h5py.File(home + 'tmp_stack/%s_band_cat-stack_random-img_rndm-select_2.h5' % band[kk], 'w') as f:
			#with h5py.File(home + 'tmp_stack/%s_band_cat-stack_random-img_top-1000.h5' % band[kk], 'w') as f:
			#	f['a'] = np.array(stack_img)

		commd.Barrier()
	raise
	for kk in range( 1 ):

		with h5py.File(load + 'random_cat/cat_select/rand_%s_band_catalog.h5' % (band[kk]), 'r') as f:
			tmp_array = np.array(f['a'])
		ra, dec, z, rich = np.array(tmp_array[0]), np.array(tmp_array[1]), np.array(tmp_array[2]), np.array(tmp_array[3])

		DN = 1000
		set_ra, set_dec, set_z = ra[:DN], dec[:DN], z[:DN]

		m, n = divmod(DN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		sky_stack(kk, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])
		commd.Barrier()

		## combine all of the sub-stack imgs
		if rank == 0:

			bcg_stack = np.zeros((len(Ny), len(Nx)), dtype = np.float)
			bcg_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

			for pp in range(cpus):

				with h5py.File(tmp + 'sky_sum_pcount_%d_in_%s_band.h5' % (pp, band[kk]), 'r')as f:
					p_count = np.array(f['a'])
				with h5py.File(tmp + 'sky_sum_%d_in_%s_band.h5' % (pp, band[kk]), 'r') as f:
					sum_img = np.array(f['a'])

				id_zero = p_count == 0
				ivx = id_zero == False
				bcg_stack[ivx] = bcg_stack[ivx] + sum_img[ivx]
				bcg_count[ivx] = bcg_count[ivx] + p_count[ivx]

				## save sub-sample sky
				sub_mean = sum_img / p_count
				id_zero = sub_mean == 0.
				id_inf = np.isinf(sub_mean)
				sub_mean[id_zero] = np.nan
				sub_mean[id_inf] = np.nan

				#with h5py.File(home + 'tmp_stack/%s_band_center-stack_random-sky_%d-sub-sample.h5' % (band[kk], pp), 'w') as f:
				with h5py.File(home + 'tmp_stack/%s_band_center-stack_rand_minu-media_sky_%d-sub-sample.h5' % (band[kk], pp), 'w') as f:
					f['a'] = np.array(sub_mean)

			## centered on BCG
			id_zero = bcg_count == 0
			bcg_stack[id_zero] = np.nan
			bcg_count[id_zero] = np.nan
			stack_img = bcg_stack / bcg_count
			id_inf = np.isinf(stack_img)
			stack_img[id_inf] = np.nan

			#with h5py.File(home + 'tmp_stack/%s_band_center-stack_random-sky-img.h5' % band[kk], 'w') as f:
			with h5py.File(home + 'tmp_stack/%s_band_center-stack_rand_minu-media_sky.h5' % band[kk], 'w') as f:
				f['a'] = np.array(stack_img)

		commd.Barrier()

if __name__ == "__main__":
	main()
