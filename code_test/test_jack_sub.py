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
from scipy import interpolate as interp
from astropy import cosmology as apcy
from light_measure import light_measure, light_measure_rn
from Mass_rich_radius import rich2R_critical_2019, rich2R_2019

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
Angu_ref = (R0/Da_ref)*rad2asec
Rpp = Angu_ref/pixel
M_dot = 4.83 # the absolute magnitude of SUN

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'
band = ['r', 'g', 'i', 'u', 'z']
mag_add = np.array([0, 0, 0, -0.04, 0.02])

def betwn_SB(data, R_low, R_up, cx, cy, pix_size, z0, band_id):

	betwn_r, betwn_Intns, betwn_err = light_measure_rn(data, R_low, R_up, cx, cy, pix_size, z0)
	betwn_lit = 22.5 - 2.5 * np.log10(betwn_Intns) + 2.5 * np.log10(pixel**2) + mag_add[band_id]
	flux0 = betwn_Intns + betwn_err
	flux1 = betwn_Intns - betwn_err
	dSB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2) + mag_add[band_id]
	dSB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2) + mag_add[band_id]
	btn_err0 = betwn_lit - dSB0
	btn_err1 = dSB1 - betwn_lit
	id_nan = np.isnan(dSB1)
	if id_nan == True:
		btn_err1 = 100.

	return betwn_r, betwn_lit, btn_err0, btn_err1, betwn_Intns, betwn_err

def SB_pro(img, R_bins, R_min, R_max, Cx, Cy, pix_size, zg, band_id):
	kk = band_id
	Intns, Intns_r, Intns_err = light_measure(img, R_bins, R_min, R_max, Cx, Cy, pix_size, zg)
	SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2) + mag_add[kk]
	flux0 = Intns + Intns_err
	flux1 = Intns - Intns_err
	dSB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2) + mag_add[kk]
	dSB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2) + mag_add[kk]
	err0 = SB - dSB0
	err1 = dSB1 - SB
	id_nan = np.isnan(SB)
	SB_out, R_out, out_err0, out_err1 = SB[id_nan == False], Intns_r[id_nan == False], err0[id_nan == False], err1[id_nan == False]
	dSB0, dSB1 = dSB0[id_nan == False], dSB1[id_nan == False]
	idx_nan = np.isnan(dSB1)
	out_err1[idx_nan] = 100.

	return R_out, SB_out, out_err0, out_err1, Intns, Intns_r, Intns_err

def rich_divid(band_id, sub_z, sub_ra, sub_dec):

	stack_N = len(sub_z)
	ii = np.int(band_id)

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
		Da_g = Test_model.angular_diameter_distance(z_g).value

		## A mask imgs without edge pixels
		data_A = fits.getdata(load + 
			'edge_cut/sample_img/Edg_cut-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)

		img_A = data_A[0]
		xn = data_A[1]['CENTER_X']
		yn = data_A[1]['CENTER_Y']

		la0 = np.int(y0 - yn)
		la1 = np.int(y0 - yn + img_A.shape[0])
		lb0 = np.int(x0 - xn)
		lb1 = np.int(x0 - xn + img_A.shape[1])

		idx = np.isnan(img_A)
		idv = np.where(idx == False)
		BL = 0.
		sub_BL_img = img_A - BL

		sum_array_A[la0: la1, lb0: lb1][idv] = sum_array_A[la0: la1, lb0: lb1][idv] + sub_BL_img[idv]
		count_array_A[la0: la1, lb0: lb1][idv] = sub_BL_img[idv]
		id_nan = np.isnan(count_array_A)
		id_fals = np.where(id_nan == False)
		p_count_A[id_fals] = p_count_A[id_fals] + 1.
		count_array_A[la0: la1, lb0: lb1][idv] = np.nan
		id_nm += 1.

	p_count_A[0, 0] = id_nm
	with h5py.File(tmp + 'stack_Amask_sum_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(sum_array_A)
	with h5py.File(tmp + 'stack_Amask_pcount_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(p_count_A)

	return

def sky_stack_BCG(band_id, sub_z, sub_ra, sub_dec, cor_id):
	stack_N = len(sub_z)
	kk = np.int(band_id)
	open_id = cor_id

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	sum_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	f2_sum = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	id_nm = 0
	for jj in range(stack_N):
		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]

		## sky-select sample
		data = fits.open( load + 'edge_cut/sample_sky/Edg_cut-sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g) )
		img = data[0].data
		cx, cy = data[0].header['CENTER_X'], data[0].header['CENTER_Y']

		la0 = np.int(y0 - cy)
		la1 = np.int(y0 - cy + img.shape[0])
		lb0 = np.int(x0 - cx)
		lb1 = np.int(x0 - cx + img.shape[1])

		idx = np.isnan(img)
		idv = np.where(idx == False)

		if open_id == 0 :
			img_add = img - np.nanmean(img)
		if open_id == 1 :
			img_add = img - np.nanmedian(img)

		sum_array[la0: la1, lb0: lb1][idv] = sum_array[la0:la1, lb0:lb1][idv] + img_add[idv]
		f2_sum[la0: la1, lb0: lb1][idv] = f2_sum[la0: la1, lb0: lb1][idv] + img_add[idv]**2
		count_array[la0: la1, lb0: lb1][idv] = img_add[idv]
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

	with h5py.File(tmp + 'sky_f2_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(f2_sum)

	return

def sky_stack_rndm(band_id, sub_z, sub_ra, sub_dec, cor_id):

	stack_N = len(sub_z)
	kk = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	## random center as comparison
	rndm_sum = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	rndm_count = np.zeros((len(Ny), len(Nx)), dtype = np.float) * np.nan
	rndm_pcont = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	f2_sum = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	pos_x = np.zeros(stack_N, dtype = np.float)
	pos_y = np.zeros(stack_N, dtype = np.float)
	n_add = 0.
	for jj in range(stack_N):
		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]

		## sky-select sample
		data = fits.open( load + 'edge_cut/sample_sky/Edg_cut-sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g) )
		img = data[0].data
		cx, cy = data[0].header['CENTER_X'], data[0].header['CENTER_Y']

		## the random center test
		rnx, rny = np.random.choice(img.shape[1], 1, replace = False), np.random.choice(img.shape[0], 1, replace = False) ## random center
		pos_x[jj], pos_y[jj] = rnx, rny

		la0 = np.int(y0 - rny)
		la1 = np.int(y0 - rny + img.shape[0])
		lb0 = np.int(x0 - rnx)
		lb1 = np.int(x0 - rnx + img.shape[1])

		idx = np.isnan(img)
		idv = np.where(idx == False)

		if cor_id == 0:
			img_add = img - np.nanmean(img)
		if cor_id == 1:
			img_add = img - np.nanmedian(img)

		rndm_sum[la0:la1, lb0:lb1][idv] = rndm_sum[la0:la1, lb0:lb1][idv] + img_add[idv]
		f2_sum[la0:la1, lb0:lb1][idv] = f2_sum[la0:la1, lb0:lb1][idv] + img_add[idv]**2
		rndm_count[la0:la1, lb0:lb1][idv] = img_add[idv]
		id_nan = np.isnan(rndm_count)
		id_fals = np.where(id_nan == False)
		rndm_pcont[id_fals] = rndm_pcont[id_fals] + 1
		rndm_count[la0:la1, lb0:lb1][idv] = np.nan

		n_add += 1.

	rndm_pcont[0, 0] = n_add
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

	with h5py.File(tmp + 'rndm_Var_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(f2_sum)

	return

def main():
	d_load = load + 'rich_sample/jack_sub-sample/'
	## skg_img - np.nanmean(sky_img) [cor_id == 0] or sky_img - np.nanmedian(sky_img) [cor_id == 1]
	cor_id = 1 # 0, 1 (see description at the beginning)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	rich_a0, rich_a1, rich_a2 = 20, 30, 50 # for lamda_k = 0, 1, 2
	N_bin = 30

	bins, R_smal, R_max = 95, 1, 3.0e3 ## for sky ICL
	bin_1Mpc = 75
	dnoise = 30
	SB_lel = np.arange(27, 31, 1) ## cotour c-label	
	"""
	## stack cluster img
	for kk in range(3):

		with h5py.File(load + 'mpi_h5/%s_band_sky_catalog.h5' % band[kk], 'r') as f:
			set_array = np.array(f['a'])
		set_ra, set_dec, set_z, set_rich = set_array[0,:], set_array[1,:], set_array[2,:], set_array[3,:]
		## calculate the physical size of the sub-sample clusters
		M_vir_c, R_vir_c = rich2R_critical_2019(set_z, set_rich)
		M_vir_m, R_vir_m = rich2R_2019(set_z, set_rich)
		for lamda_k in range(3):
			if lamda_k == 0:
				idx = (set_rich >= rich_a0) & (set_rich <= rich_a1)
			elif lamda_k == 1:
				idx = (set_rich >= rich_a1) & (set_rich <= rich_a2)
			else:
				idx = (set_rich >= rich_a2)

			lis_z = set_z[idx]
			lis_ra = set_ra[idx]
			lis_dec = set_dec[idx]
			lis_r200_c, lis_m200_c = R_vir_c[idx], M_vir_c[idx]
			lis_r200_m, lis_m200_m = R_vir_m[idx], M_vir_m[idx]

			zN = len(lis_z)
			n_step = zN // N_bin
			id_arr = np.linspace(0, zN - 1, zN)
			id_arr = id_arr.astype(int)

			for nn in range(rank, rank + 1):
				if nn == N_bin - 1:
					dot = id_arr[nn * n_step:]
				else:
					dot = id_arr[nn * n_step: (nn + 1) * n_step]

				z_use = lis_z[dot]
				ra_use = lis_ra[dot]
				dec_use = lis_dec[dot]
				rich_divid(kk, z_use, ra_use, dec_use)

				with h5py.File(tmp + 'stack_Amask_pcount_%d_in_%s_band.h5' % (nn, band[kk]), 'r')as f:
					p_count = np.array(f['a'])
				with h5py.File(tmp + 'stack_Amask_sum_%d_in_%s_band.h5' % (nn, band[kk]), 'r') as f:
					sum_img = np.array(f['a'])

				id_zero = p_count == 0
				sum_img[id_zero] = np.nan
				p_count[id_zero] = np.nan
				stack_img = sum_img / p_count
				where_are_inf = np.isinf(stack_img)
				stack_img[where_are_inf] = np.nan

				with h5py.File(d_load + '%d_rich_%d_sub-stack_%s_band_img.h5' % (lamda_k, nn, band[kk]), 'w') as f:
					f['a'] = np.array(stack_img)

				## save the virial radius
				sub_r200_c = lis_r200_c[dot]
				sub_m200_c = lis_m200_c[dot]
				dmp_array_c = np.array([sub_r200_c, sub_m200_c])
				with h5py.File(d_load + '%s_band_%d_rich_%d_sub-sample_R200c.h5' % (band[kk], lamda_k, nn), 'w') as f:
					f['a'] = np.array(dmp_array_c)

				sub_r200_m = lis_r200_m[dot]
				sub_m200_m = lis_m200_m[dot]
				dmp_array_m = np.array([sub_r200_m, sub_m200_m])
				with h5py.File(d_load + '%s_band_%d_rich_%d_sub-sample_R200m.h5' % (band[kk], lamda_k, nn), 'w') as f:
					f['a'] = np.array(dmp_array_m)

			commd.Barrier()

	## stack sky img
	for kk in range( 3 ):

		with h5py.File(load + 'mpi_h5/%s_band_sky_catalog.h5' % band[kk], 'r') as f:
			set_array = np.array(f['a'])
		set_ra, set_dec, set_z, set_rich = set_array[0,:], set_array[1,:], set_array[2,:], set_array[3,:]

		for lamda_k in range(3):

			if lamda_k == 0:
				idx = (set_rich >= rich_a0) & (set_rich <= rich_a1)
			elif lamda_k == 1:
				idx = (set_rich >= rich_a1) & (set_rich <= rich_a2)
			else:
				idx = (set_rich >= rich_a2)

			lis_z = set_z[idx]
			lis_ra = set_ra[idx]
			lis_dec = set_dec[idx]
			lis_rich = set_rich[idx]

			zN = len(lis_z)
			n_step = zN // N_bin
			id_arr = np.linspace(0, zN - 1, zN)
			id_arr = id_arr.astype(int)

			for nn in range(rank, rank + 1):
				if nn == N_bin - 1:
					dot = id_arr[nn * n_step:]
				else:
					dot = id_arr[nn * n_step: (nn + 1) * n_step]

				z_use = lis_z[dot]
				ra_use = lis_ra[dot]
				dec_use = lis_dec[dot]
				sky_stack_BCG(kk, z_use, ra_use, dec_use, cor_id)

				with h5py.File(tmp + 'sky_sum_pcount_%d_in_%s_band.h5' % (nn, band[kk]), 'r')as f:
					p_count = np.array(f['a'])
				with h5py.File(tmp + 'sky_sum_%d_in_%s_band.h5' % (nn, band[kk]), 'r') as f:
					sum_img = np.array(f['a'])

				id_zero = p_count == 0
				sum_img[id_zero] = np.nan
				p_count[id_zero] = np.nan
				stack_img = sum_img / p_count
				id_inf = np.isinf(stack_img)
				stack_img[id_inf] = np.nan

				if cor_id == 0:
					with h5py.File(d_load + '%d_rich_sky-mean_%d_sub-stack_%s_band_img.h5' % (lamda_k, nn, band[kk]), 'w') as f:
						f['a'] = np.array(stack_img)
				if cor_id == 1:
					with h5py.File(d_load + '%d_rich_sky-median_%d_sub-stack_%s_band_img.h5' % (lamda_k, nn, band[kk]), 'w') as f:
						f['a'] = np.array(stack_img)

			commd.Barrier()

	for d_record in range(1, 6):

		for kk in range( 3 ):

			with h5py.File(load + 'mpi_h5/%s_band_sky_catalog.h5' % band[kk], 'r') as f:
				set_array = np.array(f['a'])
			set_ra, set_dec, set_z, set_rich = set_array[0,:], set_array[1,:], set_array[2,:], set_array[3,:]

			for lamda_k in range(3):

				if lamda_k == 0:
					idx = (set_rich >= rich_a0) & (set_rich <= rich_a1)
				elif lamda_k == 1:
					idx = (set_rich >= rich_a1) & (set_rich <= rich_a2)
				else:
					idx = (set_rich >= rich_a2)

				lis_z = set_z[idx]
				lis_ra = set_ra[idx]
				lis_dec = set_dec[idx]
				lis_rich = set_rich[idx]

				zN = len(lis_z)
				n_step = zN // N_bin
				id_arr = np.linspace(0, zN - 1, zN)
				id_arr = id_arr.astype(int)

				for nn in range(rank, rank + 1):
					if nn == N_bin - 1:
						dot = id_arr[nn * n_step:]
					else:
						dot = id_arr[nn * n_step: (nn + 1) * n_step]

					z_use = lis_z[dot]
					ra_use = lis_ra[dot]
					dec_use = lis_dec[dot]
					sky_stack_rndm(kk, z_use, ra_use, dec_use, cor_id)

					with h5py.File(tmp + 'rndm_sum_%d_in_%s_band.h5' % (nn, band[kk]), 'r') as f:
						rndm_sum = np.array(f['a'])
					with h5py.File(tmp + 'rndm_sum_pcount_%d_in_%s_band.h5' % (nn, band[kk]), 'r') as f:
						rndm_pcont = np.array(f['a'])

					id_zero = rndm_pcont == 0
					rndm_sum[id_zero] = np.nan
					rndm_pcont[id_zero] = np.nan
					random_stack = rndm_sum / rndm_pcont
					id_inf = np.isinf(random_stack)
					random_stack[id_inf] = np.nan

					if cor_id == 0:
						with h5py.File(d_load + 
							'%d_rich_rndm_sky-mean_%d_sub-stack_%s_band_img_%d_rnd.h5' % (lamda_k, nn, band[kk], d_record), 'w') as f:
							f['a'] = np.array(random_stack)
					if cor_id == 1:
						with h5py.File(d_load + 
							'%d_rich_rndm_sky-median_%d_sub-stack_%s_band_img_%d_rnd.h5' % (lamda_k, nn, band[kk], d_record), 'w') as f:
							f['a'] = np.array(random_stack)

				commd.Barrier()

	for kk in range(3):
		for lamda_k in range(3):

			for nn in range(rank, rank + 1):

				m_rndm_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				rndm_cnt = np.zeros((len(Ny), len(Nx)), dtype = np.float)

				for id_rec in range(1, 6):
					if cor_id == 0:
						with h5py.File(d_load + 
							'%d_rich_rndm_sky-mean_%d_sub-stack_%s_band_img_%d_rnd.h5' % (lamda_k, nn, band[kk], id_rec), 'r') as f:
							sub_img = np.array(f['a'])
					if cor_id == 1:
						with h5py.File(d_load + 
							'%d_rich_rndm_sky-median_%d_sub-stack_%s_band_img_%d_rnd.h5' % (lamda_k, nn, band[kk], id_rec), 'r') as f:
							sub_img = np.array(f['a'])

					id_nan = np.isnan(sub_img)
					id_fals = id_nan == False
					m_rndm_img[id_fals] = m_rndm_img[id_fals] + sub_img[id_fals]
					rndm_cnt[id_fals] = rndm_cnt[id_fals] + 1.

				m_rndm_img = m_rndm_img / rndm_cnt
				id_zero = m_rndm_img == 0
				id_inf = np.isinf(m_rndm_img)
				m_rndm_img[id_zero] = np.nan

				if cor_id == 0:
					with h5py.File(d_load + '%d_rich_M_rndm_sky-mean_%d_sub-stack_%s_band.h5' % (lamda_k, nn, band[kk]), 'w') as f:
						f['a'] = np.array(m_rndm_img)
				if cor_id == 1:
					with h5py.File(d_load + '%d_rich_M_rndm_sky-median_%d_sub-stack_%s_band.h5' % (lamda_k, nn, band[kk]), 'w') as f:
						f['a'] = np.array(m_rndm_img)
	commd.Barrier()
	"""
	## SB pros.
	for kk in range(rank, rank + 1):
		for lamda_k in range(3):
			for nn in range(N_bin):
				## physical size
				with h5py.File(d_load + '%s_band_%d_rich_%d_sub-sample_R200c.h5' % (band[kk], lamda_k, nn), 'r') as f:
					dmp_array_c = np.array(f['a'])
				R200 = dmp_array_c[0]
				'''
				with h5py.File(d_load + '%s_band_%d_rich_%d_sub-sample_R200m.h5' % (band[kk], lamda_k, nn), 'r') as f:
					dmp_array_m = np.array(f['a'])
				R200 = dmp_array_m[0]
				'''
				m_R200 = np.nanmedian(R200)
				bins_0 = np.int( np.ceil(bin_1Mpc * m_R200 / 1e3) )
				R_min_0, R_max_0 = 1, m_R200 # kpc
				R_min_1, R_max_1 = m_R200 + 100., R_max # kpc

				if R_min_1 < R_max:
					x_quen = np.logspace(0, np.log10(R_max_0), bins_0)
					d_step = np.log10(x_quen[-1]) - np.log10(x_quen[-2])
					bins_1 = len( np.arange(np.log10(R_min_1), np.log10(R_max), d_step) )
				else:
					bins_1 = 0
				r_a0, r_a1 = R_max_0, R_min_1

				## clust imgs
				with h5py.File(d_load + '%d_rich_%d_sub-stack_%s_band_img.h5' % (lamda_k, nn, band[kk]), 'r') as f:
					clust_img = np.array(f['a'])
				## sky imgs
				with h5py.File(d_load + '%d_rich_sky-median_%d_sub-stack_%s_band_img.h5' % (lamda_k, nn, band[kk]), 'r') as f:
					BCG_sky = np.array(f['a'])
				with h5py.File(d_load + '%d_rich_M_rndm_sky-median_%d_sub-stack_%s_band.h5' % (lamda_k, nn, band[kk]), 'r') as f:
					rand_sky = np.array(f['a'])
				differ_img = BCG_sky - rand_sky
				add_img = clust_img + differ_img

				## SB measurement
				Rt_0, SBt_0, t_err0_0, t_err1_0, Intns_0_0, Intns_r_0_0, Intns_err_0_0 = SB_pro(clust_img, bins_0, R_min_0, R_max_0, x0, y0, pixel, z_ref, kk)
				Rt_1, SBt_1, t_err0_1, t_err1_1, Intns_0_1, Intns_r_0_1, Intns_err_0_1 = SB_pro(clust_img, bins_1, R_min_1, R_max_1, x0, y0, pixel, z_ref, kk)
				betwn_r, betwn_lit, btn_err0, btn_err1, betwn_Intns, betwn_err = betwn_SB(clust_img, r_a0, r_a1, x0, y0, pixel, z_ref, kk)
				
				Rt = np.r_[Rt_0, betwn_r, Rt_1]
				SBt = np.r_[SBt_0, betwn_lit, SBt_1]
				t_err0 = np.r_[t_err0_0, btn_err0, t_err0_1]
				t_err1 = np.r_[t_err1_0, btn_err1, t_err1_1]
				Intns_0 = np.r_[Intns_0_0, betwn_Intns, Intns_0_1]
				Intns_r_0 = np.r_[Intns_r_0_0, betwn_r, Intns_r_0_1]
				Intns_err_0 = np.r_[Intns_err_0_0, betwn_err, Intns_err_0_1]
				Intns_0, Intns_err_0 = Intns_0 / pixel**2, Intns_err_0 / pixel**2

				dmp_array = np.array([Intns_r_0, Intns_0, Intns_err_0])
				with h5py.File(d_load + '%s_band_%d_rich_%d_sub-sample_clust_stack_SB.h5' % (band[kk], lamda_k, nn), 'w') as f:
					f['a'] = np.array(dmp_array)
				with h5py.File(d_load + '%s_band_%d_rich_%d_sub-sample_clust_stack_SB.h5' % (band[kk], lamda_k, nn), ) as f:
					for ll in range(len(dmp_array)):
						f['a'][ll,:] = dmp_array[ll,:]

				R_sky, sky_ICL, sky_err0, sky_err1, Intns, Intns_r, Intns_err = SB_pro(differ_img, bins, R_smal, R_max, x0, y0, pixel, z_ref, kk)
				Intns, Intns_err = Intns / pixel**2, Intns_err / pixel**2

				dmp_array = np.array([Intns_r, Intns, Intns_err])
				with h5py.File(d_load + '%s_band_%d_rich_%d_sub-sample_sky_stack_SB.h5' % (band[kk], lamda_k, nn), 'w') as f:
					f['a'] = np.array(dmp_array)
				with h5py.File(d_load + '%s_band_%d_rich_%d_sub-sample_sky_stack_SB.h5' % (band[kk], lamda_k, nn), ) as f:
					for ll in range(len(dmp_array)):
						f['a'][ll,:] = dmp_array[ll,:]			

				R_add_0, SB_add_0, add_err0_0, add_err1_0, Intns_1_0, Intns_r_1_0, Intns_err_1_0 = SB_pro(
					add_img, bins_0, R_min_0, R_max_0, x0, y0, pixel, z_ref, kk)
				R_add_1, SB_add_1, add_err0_1, add_err1_1, Intns_1_1, Intns_r_1_1, Intns_err_1_1 = SB_pro(
					add_img, bins_1, R_min_1, R_max_1, x0, y0, pixel, z_ref, kk)
				betwn_r, betwn_lit, btn_err0, btn_err1, betwn_Intns, betwn_err = betwn_SB(add_img, r_a0, r_a1, x0, y0, pixel, z_ref, kk)

				R_add = np.r_[R_add_0, betwn_r, R_add_1]
				SB_add = np.r_[SB_add_0, betwn_lit, SB_add_1]
				add_err0 = np.r_[add_err0_0, btn_err0, add_err0_1]
				add_err1 = np.r_[add_err1_0, btn_err1, add_err1_1]
				Intns_1 = np.r_[Intns_1_0, betwn_Intns, Intns_1_1]
				Intns_r_1 = np.r_[Intns_r_1_0, betwn_r, Intns_r_1_1]
				Intns_err_1 = np.r_[Intns_err_1_0, betwn_err, Intns_err_1_1]
				Intns_1, Intns_err_1 = Intns_1 / pixel**2, Intns_err_1 / pixel**2

				dmp_array = np.array([Intns_r_1, Intns_1, Intns_err_1])
				with h5py.File(d_load + '%s_band_%d_rich_%d_sub-sample_add-img_SB.h5' % (band[kk], lamda_k, nn), 'w') as f:
					f['a'] = np.array(dmp_array)
				with h5py.File(d_load + '%s_band_%d_rich_%d_sub-sample_add-img_SB.h5' % (band[kk], lamda_k, nn), ) as f:
					for ll in range(len(dmp_array)):
						f['a'][ll,:] = dmp_array[ll,:]

				## estimate the residual background
				Resi_bl = betwn_Intns / pixel**2
				Resi_std = betwn_err / pixel**2
				Resi_sky = betwn_lit
				bl_dSB0, bl_dSB1 = betwn_lit - btn_err0, betwn_lit + btn_err1

				minu_bl_img = differ_img + clust_img - Resi_bl * pixel**2
				# case 1 : measuring based on img
				cli_R_0, cli_SB_0, cli_err0_0, cli_err1_0, Intns_2_0, Intns_r_2_0, Intns_err_2_0 = SB_pro(
					minu_bl_img, bins_0, R_min_0, R_max_0, x0, y0, pixel, z_ref, kk)
				cli_R_1, cli_SB_1, cli_err0_1, cli_err1_1, Intns_2_1, Intns_r_2_1, Intns_err_2_1 = SB_pro(
					minu_bl_img, bins_1, R_min_1, R_max_1, x0, y0, pixel, z_ref, kk)
				betwn_r, betwn_lit, btn_err0, btn_err1, betwn_Intns, betwn_err = betwn_SB(minu_bl_img, r_a0, r_a1, x0, y0, pixel, z_ref, kk)

				cli_R = np.r_[cli_R_0, betwn_r, cli_R_1]
				cli_SB = np.r_[cli_SB_0, betwn_lit, cli_SB_1]
				cli_err0 = np.r_[cli_err0_0, btn_err0, cli_err0_1]
				cli_err1 = np.r_[cli_err1_0, btn_err1, cli_err1_1]
				Intns_2 = np.r_[Intns_2_0, betwn_Intns, Intns_2_1]
				Intns_r_2 = np.r_[Intns_r_2_0, betwn_r, Intns_r_2_1]
				Intns_err_2 = np.r_[Intns_err_2_0, betwn_err, Intns_err_2_1]
				Intns_2, Intns_err_2 = Intns_2 / pixel**2, Intns_err_2 / pixel**2

				dmp_array = np.array([Intns_r_2, Intns_2, Intns_err_2])
				with h5py.File(d_load + '%s_band_%d_rich_%d_sub-sample_ICL_SB.h5' % (band[kk], lamda_k, nn), 'w') as f:
					f['a'] = np.array(dmp_array)
				with h5py.File(d_load + '%s_band_%d_rich_%d_sub-sample_ICL_SB.h5' % (band[kk], lamda_k, nn), ) as f:
					for ll in range(len(dmp_array)):
						f['a'][ll,:] = dmp_array[ll,:]

				## img record for sub-sample
				plt.figure(figsize = (16, 12))
				bx0 = plt.subplot(221)
				bx1 = plt.subplot(222)
				bx2 = plt.subplot(223)
				bx3 = plt.subplot(224)
				if lamda_k == 0:
					bx0.set_title('$ %s \; band \; %d \; sub-sample \; stacking [20 \\leqslant \\lambda \\leqslant 30] $' % (band[kk], nn) )
				elif lamda_k == 1:
					bx0.set_title('$ %s \; band \; %d \; sub-sample \; stacking [30 \\leqslant \\lambda \\leqslant 50] $' % (band[kk], nn) )
				else:
					bx0.set_title('$ %s \; band \; %d \; sub-sample \; stacking [ \\lambda \\geq 50 ] $' % (band[kk], nn) )
				clust00 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5,)
				clust01 = Circle(xy = (x0, y0), radius = 0.5 * Rpp, fill = False, ec = 'r', ls = '--', alpha = 0.5,)
				tf = bx0.imshow(clust_img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
				plt.colorbar(tf, ax = bx0, fraction = 0.042, pad = 0.01, label = 'flux[nmaggy]')
				## add contour
				kernl_img = ndimage.gaussian_filter(clust_img, sigma = dnoise,  mode = 'nearest')
				SB_img = 22.5 - 2.5 * np.log10(kernl_img) + 2.5 * np.log10(pixel**2)
				tg = bx0.contour(SB_img, origin = 'lower', cmap = 'rainbow', levels = SB_lel, )
				plt.clabel(tg, inline = False, fontsize = 6.5, colors = 'k', fmt = '%.0f')

				bx0.add_patch(clust00)
				bx0.add_patch(clust01)
				bx0.axis('equal')
				bx0.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
				bx0.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
				bx0.set_xticks([])
				bx0.set_yticks([])

				bx1.set_title('%s band difference img' % band[kk] )
				clust10 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-',)
				clust11 = Circle(xy = (x0, y0), radius = 0.5 * Rpp, fill = False, ec = 'r', ls = '--',)
				tf = bx1.imshow(differ_img, origin = 'lower', cmap = 'seismic', vmin = -2e-4, vmax = 2e-4)
				plt.colorbar(tf, ax = bx1, fraction = 0.042, pad = 0.01, label = 'flux[nmaggy]')
				bx1.add_patch(clust10)
				bx1.add_patch(clust11)
				bx1.axis('equal')
				bx1.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
				bx1.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
				bx1.set_xticks([])
				bx1.set_yticks([])

				bx2.set_title('%s band difference + stack img' % band[kk] )
				clust20 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5,)
				clust21 = Circle(xy = (x0, y0), radius = 0.5 * Rpp, fill = False, ec = 'r', ls = '--', alpha = 0.5,)
				tf = bx2.imshow(add_img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
				plt.colorbar(tf, ax = bx2, fraction = 0.042, pad = 0.01, label = 'flux[nmaggy]')
				## add contour
				kernl_img = ndimage.gaussian_filter(add_img, sigma = dnoise,  mode = 'nearest')
				SB_img = 22.5 - 2.5 * np.log10(kernl_img) + 2.5 * np.log10(pixel**2)
				tg = bx2.contour(SB_img, origin = 'lower', cmap = 'rainbow', levels = SB_lel, )
				plt.clabel(tg, inline = False, fontsize = 6.5, colors = 'k', fmt = '%.0f')

				bx2.add_patch(clust20)
				bx2.add_patch(clust21)
				bx2.axis('equal')
				bx2.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
				bx2.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
				bx2.set_xticks([])
				bx2.set_yticks([])

				bx3.set_title('%s band difference + stack - RBL' % band[kk] )
				clust30 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5,)
				clust31 = Circle(xy = (x0, y0), radius = 0.5 * Rpp, fill = False, ec = 'r', ls = '--', alpha = 0.5,)

				tf = bx3.imshow(minu_bl_img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
				plt.colorbar(tf, ax = bx3, fraction = 0.042, pad = 0.01, label = 'flux[nmaggy]')
				## add contour
				kernl_img = ndimage.gaussian_filter(minu_bl_img, sigma = dnoise,  mode = 'nearest')
				SB_img = 22.5 - 2.5 * np.log10(kernl_img) + 2.5 * np.log10(pixel**2)
				tg = bx3.contour(SB_img, origin = 'lower', cmap = 'rainbow', levels = SB_lel, )
				plt.clabel(tg, inline = False, fontsize = 6.5, colors = 'k', fmt = '%.0f')

				bx3.add_patch(clust30)
				bx3.add_patch(clust31)
				bx3.axis('equal')
				bx3.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
				bx3.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
				bx3.set_xticks([])
				bx3.set_yticks([])

				plt.tight_layout()
				if lamda_k == 0:
					plt.savefig(d_load + 'low_rich_%s_band_%d_sub-sample.png' % (band[kk], nn), dpi = 300)
				elif lamda_k == 1:
					plt.savefig(d_load + 'median_rich_%s_band_%d_sub-sample.png' % (band[kk], nn), dpi = 300)
				else:
					plt.savefig(d_load + 'high_rich_%s_band_%d_sub-sample.png' % (band[kk], nn), dpi = 300) 
				plt.close()

				plt.figure()
				cx0 = plt.subplot(111)
				if lamda_k == 0:
					cx0.set_title('$ %s \, band %d \, sub-sample \, SB \, profile [20 \\leqslant \\lambda \\leqslant 30] $' % (band[kk], nn) )
				elif lamda_k == 1:
					cx0.set_title('$ %s \, band %d \, sub-sample \, SB \, profile [30 \\leqslant \\lambda \\leqslant 50] $' % (band[kk], nn) )
				else:
					cx0.set_title('$ %s \, band %d \, sub-sample \, SB \, profile [ \\lambda \\geq 50 ] $' % (band[kk], nn) )

				cx0.errorbar(Rt, SBt, yerr = [t_err0, t_err1], xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1, 
					ecolor = 'r', elinewidth = 1, label = 'img ICL + background', alpha = 0.5)
				cx0.errorbar(R_add, SB_add, yerr = [add_err0, add_err1], xerr = None, color = 'g', marker = 'None', ls = '-', linewidth = 1, 
					ecolor = 'g', elinewidth = 1, label = 'img + ICL + background', alpha = 0.5)
				cx0.errorbar(cli_R, cli_SB, yerr = [cli_err0, cli_err1], xerr = None, color = 'b', marker = 'None', ls = '-', linewidth = 1, 
					ecolor = 'b', elinewidth = 1, label = 'img ICL + sky ICL', alpha = 0.5)
				cx0.plot(R_sky, sky_ICL, color = 'm', ls = '-', alpha = 0.5, label = '$ sky \, ICL $',)

				cx0.set_xlabel('$R[kpc]$')
				cx0.set_ylabel('$SB[mag / arcsec^2]$')
				cx0.set_xscale('log')
				cx0.set_ylim(20, 35)
				cx0.set_xlim(1, 2e3)
				cx0.legend(loc = 1)
				cx0.invert_yaxis()
				cx0.grid(which = 'both', axis = 'both')
				cx0.tick_params(axis = 'both', which = 'both', direction = 'in')

				if lamda_k == 0:
					plt.savefig(d_load + '%s_band_low_rich_%d_sub-sample_SB.png' % (band[kk], nn), dpi = 300)
				elif lamda_k == 1:
					plt.savefig(d_load + '%s_band_median_rich_%d_sub-sample_SB.png' % (band[kk], nn), dpi = 300)
				else:
					plt.savefig(d_load + '%s_band_high_rich_%d_sub-sample_SB.png' % (band[kk], nn), dpi = 300) 
				plt.close()

	commd.Barrier()

if __name__ == "__main__":
	main()
