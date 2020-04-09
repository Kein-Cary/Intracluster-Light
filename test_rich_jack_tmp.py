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
import statistics as sts
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.fits as fits

from scipy import ndimage
from Mass_rich_radius import rich2R
from astropy import cosmology as apcy
from light_measure import light_measure, light_measure_rn

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

## profile catalogue [in unit of 'arcsec']
cat_Rii = np.array([0.23,  0.68,  1.03,   1.76,   3.00, 
					4.63,  7.43,  11.42,  18.20,  28.20, 
					44.21, 69.00, 107.81, 168.20, 263.00])
## the band info. of SDSS BCG pro. : 0, 1, 2, 3, 4 --> u, g, r, i, z

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

def jack_SB(SB_array, R_array, band_id, N_bins):
	## stacking profile based on flux
	stack_sb = np.array(SB_array) 
	stack_r = np.array(R_array)
	Stack_R = np.nanmean(stack_r, axis = 0)
	Stack_SB = np.nanmean(stack_sb, axis = 0)
	std_Stack_SB = np.nanstd(stack_sb, axis = 0)
	jk_Stack_err = np.sqrt(N_bins - 1) * std_Stack_SB

	## change flux to magnitude
	jk_Stack_SB = 22.5 - 2.5 * np.log10(Stack_SB) + mag_add[band_id]
	dSB0 = 22.5 - 2.5 * np.log10(Stack_SB + jk_Stack_err) + mag_add[band_id]
	dSB1 = 22.5 - 2.5 * np.log10(Stack_SB - jk_Stack_err) + mag_add[band_id]
	err0 = jk_Stack_SB - dSB0
	err1 = dSB1 - jk_Stack_SB
	id_nan = np.isnan(jk_Stack_SB)
	jk_Stack_SB, jk_Stack_R, jk_Stack_err0, jk_Stack_err1 = jk_Stack_SB[id_nan == False], Stack_R[id_nan == False], err0[id_nan == False], err1[id_nan == False]
	dSB0, dSB1 = dSB0[id_nan == False], dSB1[id_nan == False]
	idx_nan = np.isnan(dSB1)
	jk_Stack_err1[idx_nan] = 100.

	return jk_Stack_SB, jk_Stack_R, jk_Stack_err0, jk_Stack_err1, Stack_R, Stack_SB, jk_Stack_err

def SB_pro(img, R_bins, R_min, R_max, Cx, Cy, pix_size, zg, band_id):
	kk = band_id
	Intns, Intns_r, Intns_err, Npix = light_measure(img, R_bins, R_min, R_max, Cx, Cy, pix_size, zg)
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

def SB_result():
	back_id = 1 # 0, 1
	# calculate the cat_Rii at z = 0.25 in physical unit (kpc)
	ref_Rii = Da_ref * cat_Rii * 10**3 / rad2asec # in unit kpc

	if back_id == 0:
		## background based on 1-1.1Mpc brightness
		R_cut, bins_0, bins_1 = 1280, 70, 7 # around 2Mpc set
		R_min_0, R_max_0 = 1, 1e3 # kpc
		R_min_1, R_max_1 = 1.1e3, 2e3 # kpc
		r_a0, r_a1 = 1e3, 1.1e3
		d_load = load + 'rich_sample/test_img/'
	if back_id == 1:
		## background based on 1-1.1Mpc brightness
		R_cut, bins_0, bins_1 = 1318, 80, 5 # around 2.1Mpc set
		R_min_0, R_max_0 = 1, 1.4e3 # kpc
		R_min_1, R_max_1 = 1.5e3, 2.1e3 # kpc
		r_a0, r_a1 = 1.4e3, 1.5e3
		d_load = load + 'rich_sample/test_14_15/'

	bins, R_smal, R_max = 80, 1, 2e3 ## for sky ICL

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	## R200 calculate parameter
	dnoise = 25
	SB_lel = np.arange(26, 34, 1) ## cotour c-label
	M0, lamd0, z0 = 14.37, 30, 0.5
	F_lamda, G_z = 1.12, 0.18
	V_num = 200
	N_bin = 20

	for kk in range(rank, rank + 1):

		## calculate the mean result of those sub-stack
		m_BCG_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		BCG_cont_Mx = np.zeros((len(Ny), len(Nx)), dtype = np.float)

		m_differ_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		differ_cont = np.zeros((len(Ny), len(Nx)), dtype = np.float)

		## record the process data and calculate the Jackknife varinace
		SB_flux, R_arr = [], []
		stack_sb, stack_r = [], []
		add_sb, add_r = [], []
		sky_lit, sky_r = [], []

		rbl = np.zeros(N_bin, dtype = np.float)
		bcg_pros = np.zeros((N_bin, len(ref_Rii)), dtype = np.float) + np.nan
		clust_2d = np.zeros((N_bin, len(Ny), len(Nx)), dtype = np.float) + np.nan
		clust_cnt = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		differ_2d = np.zeros((N_bin, len(Ny), len(Nx)), dtype = np.float) + np.nan
		differ_cnt = np.zeros((len(Ny), len(Nx)), dtype = np.float)

		fig = plt.figure(figsize = (20, 20))
		fig.suptitle('%s band sub-sample SB profile' % band[kk])
		gs = gridspec.GridSpec(N_bin // 5, 5)

		for nn in range(N_bin):
			with h5py.File(load + 'rich_sample/test_img/%s_band_%d_sub-samp_SB_pro.h5' % (band[kk], nn), 'r') as f:
				tmp_array = np.array(f['a'])
			bcg_Rii, bcg_SB, err0, err1, SB_mean = tmp_array[0,:],tmp_array[1,:],tmp_array[2,:],tmp_array[3,:],tmp_array[4,:]
			id_nan = np.isnan(bcg_SB)
			SB_obs = bcg_SB[id_nan == False]
			R_obs, obs_err0, obs_err1 = bcg_Rii[id_nan == False], err0[id_nan == False], err1[id_nan == False]
			id_nan = np.isnan(obs_err1)
			obs_err1[id_nan] = 100.

			Nr = len(bcg_Rii)
			for mm in range( Nr ):
				dr = np.abs( ref_Rii - bcg_Rii[mm] )
				idy = dr == np.min(dr)
				bcg_pros[nn,:][idy] = SB_mean[mm]

			## cluster image
			with h5py.File(load + 'rich_sample/test_img/%d_sub-stack_%s_band_img.h5' % (nn, band[kk]), 'r') as f:
				clust_img = np.array(f['a'])

			idnx = np.isnan(clust_img)
			idv = np.where(idnx == False)
			clust_2d[nn][idv] = clust_img[idv]
			clust_cnt[idv] = clust_cnt[idv] + 1
			m_BCG_img[idv] = m_BCG_img[idv] + clust_img[idv]
			BCG_cont_Mx[idv] += 1.

			ss_img = clust_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
			Rt_0, SBt_0, t_err0_0, t_err1_0, Intns_0_0, Intns_r_0_0, Intns_err_0_0 = SB_pro(
				ss_img, bins_0, R_min_0, R_max_0, R_cut, R_cut, pixel, z_ref, kk)
			Rt_1, SBt_1, t_err0_1, t_err1_1, Intns_0_1, Intns_r_0_1, Intns_err_0_1 = SB_pro(
				ss_img, bins_1, R_min_1, R_max_1, R_cut, R_cut, pixel, z_ref, kk)
			betwn_r, betwn_lit, btn_err0, btn_err1, betwn_Intns, betwn_err = betwn_SB(ss_img, r_a0, r_a1, R_cut, R_cut, pixel, z_ref, kk)

			Rt = np.r_[Rt_0, betwn_r, Rt_1]
			SBt = np.r_[SBt_0, betwn_lit, SBt_1]
			t_err0 = np.r_[t_err0_0, btn_err0, t_err0_1]
			t_err1 = np.r_[t_err1_0, btn_err1, t_err1_1]
			Intns_0 = np.r_[Intns_0_0, betwn_Intns, Intns_0_1]
			Intns_r_0 = np.r_[Intns_r_0_0, betwn_r, Intns_r_0_1]
			Intns_err_0 = np.r_[Intns_err_0_0, betwn_err, Intns_err_0_1]

			Intns_0, Intns_err_0 = Intns_0 / pixel**2, Intns_err_0 / pixel**2
			stack_sb.append(Intns_0)
			stack_r.append(Intns_r_0)
			#......
			dmp_array = np.array([Intns_r_0, Intns_0, Intns_err_0])
			with h5py.File(d_load + '%s_band_%d_sub-stack_clust_SB.h5' % (band[kk], nn), 'w') as f:
				f['a'] = np.array(dmp_array)
			with h5py.File(d_load + '%s_band_%d_sub-stack_clust_SB.h5' % (band[kk], nn) ) as f:
				for ll in range(len(dmp_array)):
					f['a'][ll,:] = dmp_array[ll,:]
			#......
			id_nan = np.isnan(Intns_0)
			Intns_0, Intns_r_0, Intns_err_0 = Intns_0[id_nan == False], Intns_r_0[id_nan == False], Intns_err_0[id_nan == False]

			## sky image
			with h5py.File(load + 'rich_sample/test_img/sky-median_%d_sub-stack_%s_band_img.h5' % (nn, band[kk]), 'r') as f:
				BCG_sky = np.array(f['a'])
			with h5py.File(load + 'rich_sample/test_img/M_rndm_sky-median_%d_sub-stack_%s_band.h5' % (nn, band[kk]), 'r') as f:
				rand_sky = np.array(f['a'])

			differ_img = BCG_sky - rand_sky
			with h5py.File(d_load + '%s_band_%d_sub-differ_img.h5' % (band[kk], nn), 'w') as f:
				f['a'] = np.array(differ_img)

			idnx = np.isnan(differ_img)
			idv = np.where(idnx == False)
			differ_2d[nn][idv] = differ_img[idv]
			differ_cnt[idv] = differ_cnt[idv] + 1
			m_differ_img[idv] = m_differ_img[idv] + differ_img[idv]
			differ_cont[idv] += 1.

			resi_add = differ_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
			R_sky, sky_ICL, sky_err0, sky_err1, Intns, Intns_r, Intns_err = SB_pro(resi_add, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref, kk)
			Intns, Intns_err = Intns / pixel**2, Intns_err / pixel**2
			sky_lit.append(Intns)
			sky_r.append(Intns_r)
			#......
			dmp_array = np.array([Intns_r, Intns, Intns_err])
			with h5py.File(d_load + '%s_band_%d_sub-stack_sky_ICL.h5' % (band[kk], nn), 'w') as f:
				f['a'] = np.array(dmp_array)
			with h5py.File(d_load + '%s_band_%d_sub-stack_sky_ICL.h5' % (band[kk], nn) ) as f:
				for ll in range(len(dmp_array)):
					f['a'][ll,:] = dmp_array[ll,:]
			#......
			id_nan = np.isnan(Intns)
			Intns, Intns_r, Intns_err = Intns[id_nan == False], Intns_r[id_nan == False], Intns_err[id_nan == False]

			## add the sky difference image
			add_img = ss_img + resi_add
			R_add_0, SB_add_0, add_err0_0, add_err1_0, Intns_1_0, Intns_r_1_0, Intns_err_1_0 = SB_pro(
				add_img, bins_0, R_min_0, R_max_0, R_cut, R_cut, pixel, z_ref, kk)
			R_add_1, SB_add_1, add_err0_1, add_err1_1, Intns_1_1, Intns_r_1_1, Intns_err_1_1 = SB_pro(
				add_img, bins_1, R_min_1, R_max_1, R_cut, R_cut, pixel, z_ref, kk)
			betwn_r, betwn_lit, btn_err0, btn_err1, betwn_Intns, betwn_err = betwn_SB(add_img, r_a0, r_a1, R_cut, R_cut, pixel, z_ref, kk)

			R_add = np.r_[R_add_0, betwn_r, R_add_1]
			SB_add = np.r_[SB_add_0, betwn_lit, SB_add_1]
			add_err0 = np.r_[add_err0_0, btn_err0, add_err0_1]
			add_err1 = np.r_[add_err1_0, btn_err1, add_err1_1]
			Intns_1 = np.r_[Intns_1_0, betwn_Intns, Intns_1_1]
			Intns_r_1 = np.r_[Intns_r_1_0, betwn_r, Intns_r_1_1]
			Intns_err_1 = np.r_[Intns_err_1_0, betwn_err, Intns_err_1_1]

			Intns_1, Intns_err_1 = Intns_1 / pixel**2, Intns_err_1 / pixel**2
			add_sb.append(Intns_1)
			add_r.append(Intns_r_1)
			#......
			dmp_array = np.array([Intns_r_1, Intns_1, Intns_err_1])
			with h5py.File(d_load + '%s_band_%d_sub-stack_add_SB.h5' % (band[kk], nn), 'w') as f:
				f['a'] = np.array(dmp_array)
			with h5py.File(d_load + '%s_band_%d_sub-stack_add_SB.h5' % (band[kk], nn) ) as f:
				for ll in range(len(dmp_array)):
					f['a'][ll,:] = dmp_array[ll,:]
			#......

			## RBL estimation
			Resi_bl = betwn_Intns * 1.
			Resi_std = betwn_err * 1.
			Resi_sky = betwn_lit * 1.
			bl_dSB0, bl_dSB1 = betwn_lit - btn_err0, betwn_lit + btn_err1
			rbl[nn] = Resi_bl * 1.

			## minus the residual background
			cli_R = Intns_r_1 * 1.
			cli_SB = 22.5 - 2.5 * np.log10(Intns_1 - Resi_bl / pixel**2) + mag_add[kk]
			cli_dSB0 = 22.5 - 2.5 * np.log10(Intns_1 + Intns_err_1 - Resi_bl / pixel**2) + mag_add[kk]
			cli_dSB1 = 22.5 - 2.5 * np.log10(Intns_1 - Intns_err_1 - Resi_bl / pixel**2) + mag_add[kk]

			err0 = cli_SB - cli_dSB0
			err1 = cli_dSB1 - cli_SB
			id_nan = np.isnan(cli_SB)
			cli_SB, cli_R, cli_err0, cli_err1 = cli_SB[id_nan == False], cli_R[id_nan == False], err0[id_nan == False], err1[id_nan == False]
			cli_dSB0, cli_dSB1 = cli_dSB0[id_nan == False], cli_dSB1[id_nan == False]
			idx_nan = np.isnan(cli_dSB1)
			cli_err1[idx_nan] = 100.

			Intns_2 = Intns_1 - Resi_bl / pixel**2
			Intns_r_2 = Intns_r_1 * 1.
			Intns_err_2 = Intns_err_1 * 1.

			SB_flux.append(Intns_2)
			R_arr.append(Intns_r_2)
			#......
			dmp_array = np.array([Intns_r_2, Intns_2, Intns_err_2])
			with h5py.File(d_load + '%s_band_%d_sub-stack_cli_SB.h5' % (band[kk], nn), 'w') as f:
				f['a'] = np.array(dmp_array)
			with h5py.File(d_load + '%s_band_%d_sub-stack_cli_SB.h5' % (band[kk], nn) ) as f:
				for ll in range(len(dmp_array)):
					f['a'][ll,:] = dmp_array[ll,:]
			#......
			id_nan = np.isnan(Intns_1)
			Intns_1, Intns_r_1, Intns_err_1 = Intns_1[id_nan == False], Intns_r_1[id_nan == False], Intns_err_1[id_nan == False]

			id_nan = np.isnan(Intns_2)
			Intns_2, Intns_r_2, Intns_err_2 = Intns_2[id_nan == False], Intns_r_2[id_nan == False], Intns_err_2[id_nan == False]

			### fig sub-sample results
			fig = plt.figure()
			cx0 = plt.subplot(111)
			cx0.set_title('%s band %d sub-sample SB' % (band[kk], nn) )

			cx0.errorbar(Intns_r_0, Intns_0, yerr = Intns_err_0, xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1, 
				ecolor = 'r', elinewidth = 1, label = 'img ICL + background', alpha = 0.5)
			cx0.errorbar(Intns_r_1, Intns_1, yerr = Intns_err_1, xerr = None, color = 'g', marker = 'None', ls = '-', linewidth = 1, 
				ecolor = 'g', elinewidth = 1, label = 'img ICL + background + sky ICL', alpha = 0.5)
			cx0.errorbar(Intns_r_2, Intns_2, yerr = Intns_err_2, xerr = None, color = 'b', marker = 'None', ls = '-', linewidth = 1,
				ecolor = 'b', elinewidth = 1, label = 'img ICL + sky ICL', alpha = 0.5)
			cx0.errorbar(Intns_r, Intns, yerr = Intns_err, xerr = None, color = 'm', marker = 'None', ls = '-', linewidth = 1,
				ecolor = 'm', elinewidth = 1, label = 'sky ICL', alpha = 0.5)

			cx0.set_xlabel('$R [kpc] $')
			cx0.set_ylabel('$SB[ nmaggy / arcsec^2]$')
			cx0.set_xscale('log')
			cx0.set_yscale('log')
			cx0.set_xlim(1, 2e3)
			cx0.set_ylim(1e-5, 1e1)
			cx0.legend(loc = 1, frameon = False)
			cx0.grid(which = 'major', axis = 'both')
			cx0.tick_params(axis = 'both', which = 'both', direction = 'in')

			subax = fig.add_axes([0.2, 0.2, 0.32, 0.32])
			subax.errorbar(Intns_r_0, Intns_0, yerr = Intns_err_0, xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1, 
				ecolor = 'r', elinewidth = 1, alpha = 0.5)
			subax.errorbar(Intns_r_1, Intns_1, yerr = Intns_err_1, xerr = None, color = 'g', marker = 'None', ls = '-', linewidth = 1, 
				ecolor = 'g', elinewidth = 1, alpha = 0.5)
			subax.errorbar(Intns_r_2, Intns_2, yerr = Intns_err_2, xerr = None, color = 'b', marker = 'None', ls = '-', linewidth = 1,
				ecolor = 'b', elinewidth = 1, alpha = 0.5)
			subax.errorbar(Intns_r, Intns, yerr = Intns_err, xerr = None, color = 'm', marker = 'None', ls = '-', linewidth = 1,
				ecolor = 'm', elinewidth = 1, alpha = 0.5)

			subax.set_xlim(4e2, 1.4e3)
			subax.set_xscale('log')
			subax.set_ylim(3e-3, 7e-3)
			subax.set_yscale('log')
			subax.grid(which = 'both', axis = 'both')
			subax.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 5.)

			plt.savefig(d_load + '%s_band_%d_sub-sample_SB.png' % (band[kk], nn), dpi = 300)
			plt.close()

			## process img
			plt.figure(figsize = (16, 12))
			bx0 = plt.subplot(221)
			bx1 = plt.subplot(222)
			bx2 = plt.subplot(223)
			bx3 = plt.subplot(224)

			bx0.set_title('$ %s \; band \; %d \; sub-stack \; img [50 \\leqslant \\lambda] $' % (band[kk], nn) )
			clust00 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5,)
			clust01 = Circle(xy = (x0, y0), radius = 0.5 * Rpp, fill = False, ec = 'r', ls = '--', alpha = 0.5,)
			tf = bx0.imshow(clust_img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
			plt.colorbar(tf, ax = bx0, fraction = 0.042, pad = 0.01, label = 'flux[nmaggy]')
			## add contour
			con_img = clust_img * 1.
			kernl_img = ndimage.gaussian_filter(clust_img, sigma = dnoise,  mode = 'nearest')
			SB_img = 22.5 - 2.5 * np.log10(kernl_img) + 2.5 * np.log10(pixel**2)
			tg = bx0.contour(SB_img, origin = 'lower', cmap = 'rainbow', levels = SB_lel, )
			plt.clabel(tg, inline = False, fontsize = 6.5, colors = 'k', fmt = '%.0f')

			bx0.add_patch(clust00)
			bx0.add_patch(clust01)
			bx0.axis('equal')
			bx0.set_xlim(x0 - 0.7 * R_cut, x0 + 0.7 * R_cut)
			bx0.set_ylim(y0 - 0.7 * R_cut, y0 + 0.7 * R_cut)
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
			bx1.set_xlim(x0 - 0.7 * R_cut, x0 + 0.7 * R_cut)
			bx1.set_ylim(y0 - 0.7 * R_cut, y0 + 0.7 * R_cut)
			bx1.set_xticks([])
			bx1.set_yticks([])

			bx2.set_title('%s band difference + stack img' % band[kk] )
			clust20 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5,)
			clust21 = Circle(xy = (x0, y0), radius = 0.5 * Rpp, fill = False, ec = 'r', ls = '--', alpha = 0.5,)
			tf = bx2.imshow(differ_img + clust_img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
			plt.colorbar(tf, ax = bx2, fraction = 0.042, pad = 0.01, label = 'flux[nmaggy]')
			## add contour
			con_img = differ_img + clust_img
			kernl_img = ndimage.gaussian_filter(con_img, sigma = dnoise,  mode = 'nearest')
			SB_img = 22.5 - 2.5 * np.log10(kernl_img) + 2.5 * np.log10(pixel**2)
			tg = bx2.contour(SB_img, origin = 'lower', cmap = 'rainbow', levels = SB_lel, )
			plt.clabel(tg, inline = False, fontsize = 6.5, colors = 'k', fmt = '%.0f')

			bx2.add_patch(clust20)
			bx2.add_patch(clust21)
			bx2.axis('equal')
			bx2.set_xlim(x0 - 0.7 * R_cut, x0 + 0.7 * R_cut)
			bx2.set_ylim(y0 - 0.7 * R_cut, y0 + 0.7 * R_cut)
			bx2.set_xticks([])
			bx2.set_yticks([])

			bx3.set_title('%s band difference + stack - RBL' % band[kk] )
			clust30 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5,)
			clust31 = Circle(xy = (x0, y0), radius = 0.5 * Rpp, fill = False, ec = 'r', ls = '--', alpha = 0.5,)

			tf = bx3.imshow(differ_img + clust_img - Resi_bl, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, 
				norm = mpl.colors.LogNorm())
			plt.colorbar(tf, ax = bx3, fraction = 0.042, pad = 0.01, label = 'flux[nmaggy]')
			## add contour
			con_img = differ_img + clust_img - Resi_bl
			kernl_img = ndimage.gaussian_filter(con_img, sigma = dnoise,  mode = 'nearest')
			SB_img = 22.5 - 2.5 * np.log10(kernl_img) + 2.5 * np.log10(pixel**2)
			tg = bx3.contour(SB_img, origin = 'lower', cmap = 'rainbow', levels = SB_lel, )
			plt.clabel(tg, inline = False, fontsize = 6.5, colors = 'k', fmt = '%.0f')

			bx3.add_patch(clust30)
			bx3.add_patch(clust31)
			bx3.axis('equal')
			bx3.set_xlim(x0 - 0.7 * R_cut, x0 + 0.7 * R_cut)
			bx3.set_ylim(y0 - 0.7 * R_cut, y0 + 0.7 * R_cut)
			bx3.set_xticks([])
			bx3.set_yticks([])

			plt.tight_layout()
			plt.savefig(d_load + '%s_band_%d_sub-stack_process.png' % (band[kk], nn), dpi = 300)
			plt.close()
			### 
			ax = plt.subplot(gs[nn // 5, nn % 5])
			ax.errorbar(Rt, SBt, yerr = [t_err0, t_err1], xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1,
				ecolor = 'r', elinewidth = 1, label = 'img ICL + background', alpha = 0.5)
			ax.errorbar(R_add, SB_add, yerr = [add_err0, add_err1], xerr = None, color = 'g', marker = 'None', ls = '-', linewidth = 1, 
				ecolor = 'g', elinewidth = 1, label = 'img ICL + background + sky ICL', alpha = 0.5)
			ax.errorbar(cli_R, cli_SB, yerr = [cli_err0, cli_err1], xerr = None, color = 'b', marker = 'None', ls = '-', linewidth = 1, 
				ecolor = 'b', elinewidth = 1, label = 'img ICL + sky ICL', alpha = 0.5)
			ax.errorbar(R_sky, sky_ICL, yerr = [sky_err0, sky_err1], xerr = None, color = 'm', marker = 'None', ls = '--', linewidth = 1, 
				ecolor = 'm', elinewidth = 1, label = 'sky ICL', alpha = 0.5)

			ax.axhline(y = Resi_sky, ls = '-', color = 'k', label = 'background', alpha = 0.5)
			ax.errorbar(R_obs, SB_obs, yerr = [obs_err0, obs_err1], xerr = None, color = 'k', marker = 's', ls = '--', linewidth = 1, 
				markersize = 5, ecolor = 'k', elinewidth = 1, label = 'SDSS BCG photo.', alpha = 0.5)

			ax.set_xlabel('$R[kpc]$')
			ax.set_ylabel('$SB[mag / arcsec^2]$')
			ax.set_xscale('log')
			ax.set_ylim(20, 35)
			ax.set_xlim(1, 1.5e3)
			ax.legend(loc = 1, fontsize = 8., frameon = False)
			ax.invert_yaxis()
			ax.grid(which = 'major', axis = 'both')
			ax.tick_params(axis = 'both', which = 'both', direction = 'in')

		plt.tight_layout()
		plt.savefig(d_load + '%s_band_sub-stack_SB_pros.png' % band[kk], dpi = 300)
		plt.close()
		"""
		for nn in range(N_bin):
			with h5py.File(load + 'rich_sample/test_img/%s_band_%d_sub-samp_SB_pro.h5' % (band[kk], nn), 'r') as f:
				tmp_array = np.array(f['a'])
			bcg_Rii, bcg_SB, err0, err1, SB_mean = tmp_array[0,:],tmp_array[1,:],tmp_array[2,:],tmp_array[3,:],tmp_array[4,:]
			id_nan = np.isnan(bcg_SB)
			SB_obs = bcg_SB[id_nan == False]
			R_obs, obs_err0, obs_err1 = bcg_Rii[id_nan == False], err0[id_nan == False], err1[id_nan == False]
			id_nan = np.isnan(obs_err1)
			obs_err1[id_nan] = 100.

			Nr = len(bcg_Rii)
			for mm in range( Nr ):
				dr = np.abs( ref_Rii - bcg_Rii[mm] )
				idy = dr == np.min(dr)
				bcg_pros[nn,:][idy] = SB_mean[mm]

			with h5py.File(d_load + '%s_band_%d_sub-stack_clust_SB.h5' % (band[kk], nn), 'r') as f:
				dmp_array = np.array(f['a'])
			stack_sb.append(dmp_array[1])
			stack_r.append(dmp_array[0])

			with h5py.File(d_load + '%s_band_%d_sub-stack_sky_ICL.h5' % (band[kk], nn), 'r') as f:
				dmp_array = np.array(f['a'])
			sky_lit.append(dmp_array[1])
			sky_r.append(dmp_array[0])

			with h5py.File(d_load + '%s_band_%d_sub-stack_add_SB.h5' % (band[kk], nn), 'r') as f:
				dmp_array = np.array(f['a'])
			add_sb.append(dmp_array[1])
			add_r.append(dmp_array[0])

			with h5py.File(d_load + '%s_band_%d_sub-stack_cli_SB.h5' % (band[kk], nn), 'r') as f:
				dmp_array = np.array(f['a'])
			SB_flux.append(dmp_array[1])
			R_arr.append(dmp_array[0])

			## sky image
			with h5py.File(load + 'rich_sample/test_img/sky-median_%d_sub-stack_%s_band_img.h5' % (nn, band[kk]), 'r') as f:
				BCG_sky = np.array(f['a'])
			with h5py.File(load + 'rich_sample/test_img/M_rndm_sky-median_%d_sub-stack_%s_band.h5' % (nn, band[kk]), 'r') as f:
				rand_sky = np.array(f['a'])

			differ_img = BCG_sky - rand_sky
			with h5py.File(d_load + '%s_band_%d_sub-differ_img.h5' % (band[kk], nn), 'w') as f:
				f['a'] = np.array(differ_img)

			idnx = np.isnan(differ_img)
			idv = np.where(idnx == False)
			differ_2d[nn][idv] = differ_img[idv]
			differ_cnt[idv] = differ_cnt[idv] + 1
			m_differ_img[idv] = m_differ_img[idv] + differ_img[idv]
			differ_cont[idv] += 1.
			## cluster img
			with h5py.File(load + 'rich_sample/test_img/%d_sub-stack_%s_band_img.h5' % (nn, band[kk]), 'r') as f:
				clust_img = np.array(f['a'])

			idnx = np.isnan(clust_img)
			idv = np.where(idnx == False)
			clust_2d[nn][idv] = clust_img[idv]
			clust_cnt[idv] = clust_cnt[idv] + 1
			m_BCG_img[idv] = m_BCG_img[idv] + clust_img[idv]
			BCG_cont_Mx[idv] += 1.

			add_img = clust_img + differ_img
			cen_pos = R_cut * 1 # 1280 pixel, for z = 0.25, larger than 2Mpc
			BL_img = add_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
			grd_x = np.linspace(0, BL_img.shape[1] - 1, BL_img.shape[1])
			grd_y = np.linspace(0, BL_img.shape[0] - 1, BL_img.shape[0])
			grd = np.array( np.meshgrid(grd_x, grd_y) )
			ddr = np.sqrt( (grd[0,:] - cen_pos)**2 + (grd[1,:] - cen_pos)**2 )
			idu = (ddr > r_a0 * Rpp) & (ddr < r_a1 * Rpp)
			Resi_bl = np.nanmean( BL_img[idu] )
			rbl[nn] = Resi_bl * 1.
		"""
		####################

		## stack sub-stacking img result
		M_clust_img = m_BCG_img / BCG_cont_Mx
		id_inf = np.isinf(M_clust_img)
		id_zeros = M_clust_img == 0.
		M_clust_img[id_inf] = np.nan
		M_clust_img[id_zeros] = np.nan
		with h5py.File(d_load + '%s_band_clust_tot-stack_img.h5' % band[kk], 'w') as f:
			f['a'] = np.array(M_clust_img)

		with h5py.File(d_load + '%s_band_clust_tot-stack_img.h5' % band[kk], 'r') as f:
			M_clust_img = np.array(f['a'])

		M_difference = m_differ_img / differ_cont
		id_inf = np.isinf(M_difference)
		id_zeros = M_difference == 0.
		M_difference[id_inf] = np.nan
		M_difference[id_zeros] = np.nan
		with h5py.File(d_load + '%s_band_tot-difference_img.h5' % band[kk], 'w') as f:
			f['a'] = np.array(M_difference)

		with h5py.File(d_load + '%s_band_tot-difference_img.h5' % band[kk], 'r') as f:
			M_difference = np.array(f['a'])

		ss_img = M_clust_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
		## stacking image
		Rt_0, SBt_0, t_err0_0, t_err1_0, Intns_0_0, Intns_r_0_0, Intns_err_0_0 = SB_pro(
			ss_img, bins_0, R_min_0, R_max_0, R_cut, R_cut, pixel, z_ref, kk)
		Rt_1, SBt_1, t_err0_1, t_err1_1, Intns_0_1, Intns_r_0_1, Intns_err_0_1 = SB_pro(
			ss_img, bins_1, R_min_1, R_max_1, R_cut, R_cut, pixel, z_ref, kk)
		betwn_r, betwn_lit, btn_err0, btn_err1, betwn_Intns, betwn_err = betwn_SB(ss_img, r_a0, r_a1, R_cut, R_cut, pixel, z_ref, kk)

		Rt = np.r_[Rt_0, betwn_r, Rt_1]
		SBt = np.r_[SBt_0, betwn_lit, SBt_1]
		t_err0 = np.r_[t_err0_0, btn_err0, t_err0_1]
		t_err1 = np.r_[t_err1_0, btn_err1, t_err1_1]
		Intns_0 = np.r_[Intns_0_0, betwn_Intns, Intns_0_1]
		Intns_r_0 = np.r_[Intns_r_0_0, betwn_r, Intns_r_0_1]
		Intns_err_0 = np.r_[Intns_err_0_0, betwn_err, Intns_err_0_1]

		Intns_0, Intns_err_0 = Intns_0 / pixel**2, Intns_err_0 / pixel**2
		id_nan = np.isnan(Intns_0)
		Intns_0, Intns_r_0, Intns_err_0 = Intns_0[id_nan == False], Intns_r_0[id_nan == False], Intns_err_0[id_nan == False]
		## difference image
		resi_add = M_difference[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
		R_sky, sky_ICL, sky_err0, sky_err1, Intns, Intns_r, Intns_err = SB_pro(resi_add, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref, kk)
		Intns, Intns_err = Intns / pixel**2, Intns_err / pixel**2
		id_nan = np.isnan(Intns)
		Intns, Intns_r, Intns_err = Intns[id_nan == False], Intns_r[id_nan == False], Intns_err[id_nan == False]

		## add image
		add_img = ss_img + resi_add

		R_add_0, SB_add_0, add_err0_0, add_err1_0, Intns_1_0, Intns_r_1_0, Intns_err_1_0 = SB_pro(
			add_img, bins_0, R_min_0, R_max_0, R_cut, R_cut, pixel, z_ref, kk)
		R_add_1, SB_add_1, add_err0_1, add_err1_1, Intns_1_1, Intns_r_1_1, Intns_err_1_1 = SB_pro(
			add_img, bins_1, R_min_1, R_max_1, R_cut, R_cut, pixel, z_ref, kk)
		betwn_r, betwn_lit, btn_err0, btn_err1, betwn_Intns, betwn_err = betwn_SB(add_img, r_a0, r_a1, R_cut, R_cut, pixel, z_ref, kk)

		R_add = np.r_[R_add_0, betwn_r, R_add_1]
		SB_add = np.r_[SB_add_0, betwn_lit, SB_add_1]
		add_err0 = np.r_[add_err0_0, btn_err0, add_err0_1]
		add_err1 = np.r_[add_err1_0, btn_err1, add_err1_1]
		Intns_1 = np.r_[Intns_1_0, betwn_Intns, Intns_1_1]
		Intns_r_1 = np.r_[Intns_r_1_0, betwn_r, Intns_r_1_1]
		Intns_err_1 = np.r_[Intns_err_1_0, betwn_err, Intns_err_1_1]

		Intns_1, Intns_err_1 = Intns_1 / pixel**2, Intns_err_1 / pixel**2
		id_nan = np.isnan(Intns_1)
		Intns_1, Intns_r_1, Intns_err_1 = Intns_1[id_nan == False], Intns_r_1[id_nan == False], Intns_err_1[id_nan == False]

		## RBL estimation
		Resi_bl = betwn_Intns * 1.
		Resi_std = betwn_err * 1.
		Resi_sky = betwn_lit * 1.
		bl_dSB0, bl_dSB1 = betwn_lit - btn_err0, betwn_lit + btn_err1

		## minus the residual background
		cli_R = Intns_r_1 * 1.
		cli_SB = 22.5 - 2.5 * np.log10(Intns_1 - Resi_bl / pixel**2) + mag_add[kk]
		cli_dSB0 = 22.5 - 2.5 * np.log10(Intns_1 + Intns_err_1 - Resi_bl / pixel**2) + mag_add[kk]
		cli_dSB1 = 22.5 - 2.5 * np.log10(Intns_1 - Intns_err_1 - Resi_bl / pixel**2) + mag_add[kk]

		err0 = cli_SB - cli_dSB0
		err1 = cli_dSB1 - cli_SB
		id_nan = np.isnan(cli_SB)
		cli_SB, cli_R, cli_err0, cli_err1 = cli_SB[id_nan == False], cli_R[id_nan == False], err0[id_nan == False], err1[id_nan == False]
		cli_dSB0, cli_dSB1 = cli_dSB0[id_nan == False], cli_dSB1[id_nan == False]
		idx_nan = np.isnan(cli_dSB1)
		cli_err1[idx_nan] = 100.

		Intns_2 = Intns_1 - Resi_bl / pixel**2
		Intns_r_2 = Intns_r_1 * 1.
		Intns_err_2 = Intns_err_1 * 1.

		########## jackknife resampling result
		## mean BCG pros. of SDSS cat.
		m_bcg_pro = np.nanmean(bcg_pros, axis = 0)
		std_bcg_pro = np.nanstd(bcg_pros, axis = 0)
		id_nan = np.isnan(m_bcg_pro)
		bcg_pro_r = ref_Rii[id_nan == False]
		m_bcg_pro = m_bcg_pro[id_nan == False]
		std_bcg_pro = std_bcg_pro[id_nan == False]

		m_BCG_SB = 22.5 - 2.5 * np.log10(m_bcg_pro) + mag_add[kk]
		m_dSB0 = 22.5 - 2.5 * np.log10(m_bcg_pro + std_bcg_pro) + mag_add[kk]
		m_dSB1 = 22.5 - 2.5 * np.log10(m_bcg_pro - std_bcg_pro) + mag_add[kk]
		err0 = m_BCG_SB - m_dSB0
		err1 = m_dSB1 - m_BCG_SB
		id_nan = np.isnan(m_BCG_SB)
		m_BCG_SB, m_BCG_r = m_BCG_SB[id_nan == False], bcg_pro_r[id_nan == False]
		m_BCG_err0, m_BCG_err1 = err0[id_nan == False], err1[id_nan == False]

		### jackknife calculation (mean process SB)
		with h5py.File(d_load + '%s_band_RBL_SB.h5' % (band[kk]), 'w') as f:
			f['a'] = np.array(rbl)

		with h5py.File(d_load + '%s_band_RBL_SB.h5' % (band[kk]), 'r') as f:
			rbl = np.array(f['a'])

		m_rbl = np.nanmean(rbl)
		std_rbl = np.nanstd(rbl)
		jk_std_rbl = std_rbl * np.sqrt(N_bin - 1)
		m_rbl_SB = 22.5 - 2.5 * np.log10(m_rbl) + 2.5 * np.log10(pixel**2) + mag_add[kk]
		m_rbl_dSB0 = 22.5 - 2.5 * np.log10(m_rbl + jk_std_rbl) + 2.5 * np.log10(pixel**2) + mag_add[kk]
		m_rbl_dSB1 = 22.5 - 2.5 * np.log10(m_rbl - jk_std_rbl) + 2.5 * np.log10(pixel**2) + mag_add[kk]
		id_nan = np.isnan(m_rbl_dSB1)
		if id_nan == True:
			m_rbl_dSB1 = 100.

		jk_Stack_SB, jk_Stack_R, jk_Stack_err0, jk_Stack_err1, Stack_R, Stack_SB, jk_Stack_err = jack_SB(stack_sb, stack_r, kk, N_bin)

		jk_add_SB, jk_add_R, jk_add_err0, jk_add_err1, Add_R, Add_SB, jk_Add_err = jack_SB(add_sb, add_r, kk, N_bin)

		jk_sky_SB, jk_sky_R, jk_sky_err0, jk_sky_err1, sky_R, m_sky_SB, jk_sky_err = jack_SB(sky_lit, sky_r, kk, N_bin)

		JK_SB, JK_R, JK_err0, JK_err1, jk_cli_R, jk_cli_SB, jk_cli_err = jack_SB(SB_flux, R_arr, kk, N_bin)

		## pixel jackknife err
		with h5py.File(d_load + '%s_band_all_clust_2D.h5' % (band[kk]), 'w') as f:
			f['a'] = np.array(clust_2d)
		with h5py.File(d_load + '%s_band_all_differ_2D.h5' % (band[kk]), 'w') as f:
			f['a'] = np.array(differ_2d)

		with h5py.File(d_load + '%s_band_all_clust_2D.h5' % (band[kk]), 'r') as f:
			clust_2d = np.array(f['a'])
		with h5py.File(d_load + '%s_band_all_differ_2D.h5' % (band[kk]), 'r') as f:
			differ_2d = np.array(f['a'])

		clus_rms = np.nanstd(clust_2d, axis = 0)
		clust_jk_err = clus_rms * np.sqrt(N_bin - 1)
		differ_rms = np.nanstd(differ_2d, axis = 0)
		differ_jk_err = differ_rms * np.sqrt(N_bin - 1)

		#### figs
		plt.figure()
		ax = plt.subplot(111)
		clust10 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5,)
		clust11 = Circle(xy = (x0, y0), radius = 0.5 * Rpp, fill = False, ec = 'r', ls = '--', alpha = 0.5,)
		ax.set_title('%s band pixel rms [cluster img]' % band[kk])
		tf = ax.imshow(clust_jk_err, cmap = 'rainbow', origin = 'lower', vmin = np.nanmin(clust_jk_err), vmax = np.nanmax(clust_jk_err),)
		plt.colorbar(tf, ax = ax, fraction = 0.042, pad = 0.01, label = 'flux[nmaggy]')
		ax.add_patch(clust10)
		ax.add_patch(clust11)
		ax.axis('equal')
		ax.set_xlim(x0 - 0.7 * R_cut, x0 + 0.7 * R_cut)
		ax.set_ylim(y0 - 0.7 * R_cut, y0 + 0.7 * R_cut)
		ax.set_xticks([])
		ax.set_yticks([])
		plt.savefig(d_load + '%s_band_clust-rms.png' % (band[kk]), dpi = 300)
		plt.close()

		plt.figure()
		ax = plt.subplot(111)
		clust10 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5,)
		clust11 = Circle(xy = (x0, y0), radius = 0.5 * Rpp, fill = False, ec = 'r', ls = '--', alpha = 0.5,)
		ax.set_title('%s band pixel rms [difference img]' % band[kk])
		tf = ax.imshow(differ_jk_err, cmap = 'rainbow', origin = 'lower', vmin = np.nanmin(differ_jk_err), vmax = np.nanmax(differ_jk_err),)
		plt.colorbar(tf, ax = ax, fraction = 0.042, pad = 0.01, label = 'flux[nmaggy]')
		ax.add_patch(clust10)
		ax.add_patch(clust11)
		ax.axis('equal')
		ax.set_xlim(x0 - 0.7 * R_cut, x0 + 0.7 * R_cut)
		ax.set_ylim(y0 - 0.7 * R_cut, y0 + 0.7 * R_cut)
		ax.set_xticks([])
		ax.set_yticks([])
		plt.savefig(d_load + '%s_band_differ-rms.png' % (band[kk]), dpi = 300)
		plt.close()
		##################################################
		## SB profile
		fig = plt.figure()
		ax = plt.subplot(111)
		ax.set_title('%s band jackknife stacking SB' % band[kk])

		ax.errorbar(Stack_R, Stack_SB, yerr = jk_Stack_err, xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1, 
			ecolor = 'r', elinewidth = 1, alpha = 0.5, label = 'image ICL + background')
		ax.errorbar(Add_R, Add_SB, yerr = jk_Add_err, xerr = None, color = 'g', marker = 'None', ls = '-', linewidth = 1, 
			ecolor = 'g', elinewidth = 1, alpha = 0.5, label = 'image ICL + background + sky ICL')
		ax.errorbar(jk_cli_R, jk_cli_SB, yerr = jk_cli_err, xerr = None, color = 'm', marker = 'None', ls = '-', linewidth = 1, 
			ecolor = 'm', elinewidth = 1, alpha = 0.5, label = 'image ICL + sky ICL')
		#ax.errorbar(sky_R - 10, m_sky_SB, yerr = jk_sky_err, xerr = None, color = 'c', marker = 'None', ls = '-', linewidth = 1, 
		#	ecolor = 'c', elinewidth = 1, label = 'sky ICL', alpha = 0.5)
		ax.plot(sky_R, m_sky_SB, color = 'c', ls = '-', linewidth = 1, label = 'sky ICL', alpha = 0.5)

		ax.set_xlabel('$ R[kpc] $')
		ax.set_ylabel('$ SB[nmaggy / arcsec^2] $')
		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.set_xlim(1, 2e3)
		ax.set_ylim(1e-5, 1e1)
		ax.legend(loc = 1, fontsize = 8., frameon = False)
		ax.grid(which = 'major', axis = 'both')
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')

		subax = fig.add_axes([0.2, 0.2, 0.32, 0.32])
		subax.errorbar(Stack_R, Stack_SB, yerr = jk_Stack_err, xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1, 
			ecolor = 'r', elinewidth = 1, alpha = 0.5)
		subax.errorbar(Add_R, Add_SB, yerr = jk_Add_err, xerr = None, color = 'g', marker = 'None', ls = '-', linewidth = 1, 
			ecolor = 'g', elinewidth = 1, alpha = 0.5)
		subax.errorbar(jk_cli_R, jk_cli_SB, yerr = jk_cli_err, xerr = None, color = 'm', marker = 'None', ls = '-', linewidth = 1, 
			ecolor = 'm', elinewidth = 1, alpha = 0.5,)
		subax.plot(sky_R, m_sky_SB, color = 'c', ls = '-', linewidth = 1, alpha = 0.5)

		subax.set_xlim(4e2, 1.4e3)
		subax.set_xscale('log')
		subax.set_ylim(3e-3, 7e-3)
		subax.set_yscale('log')
		subax.grid(which = 'both', axis = 'both')
		subax.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 5.)

		plt.savefig(d_load + '%s_band_tot-stack_flux_dens.png' % band[kk], dpi = 300)
		plt.close()

		plt.figure()
		ax = plt.subplot(111)
		ax.set_title('%s band jackknife stacking SB' % band[kk])

		ax.errorbar(jk_Stack_R, jk_Stack_SB, yerr = [jk_Stack_err0, jk_Stack_err1], xerr = None, color = 'r', marker = 'None', 
			ls = '-', linewidth = 1, ecolor = 'r', elinewidth = 1, alpha = 0.5, label = 'img ICL + background')
		ax.errorbar(jk_add_R, jk_add_SB, yerr = [jk_add_err0, jk_add_err1], xerr = None, color = 'g', marker = 'None', ls = '-', 
			linewidth = 1, ecolor = 'g', elinewidth = 1, alpha = 0.5, label = 'img ICL + background + sky ICL')
		ax.errorbar(JK_R, JK_SB, yerr = [JK_err0, JK_err1], xerr = None, color = 'm', marker = 'None', ls = '-', linewidth = 1, 
			ecolor = 'm', elinewidth = 1, alpha = 0.5, label = 'img ICL + sky ICL')
		#ax.errorbar(jk_sky_R - 10, jk_sky_SB, yerr = [jk_sky_err0, jk_sky_err1], xerr = None, color = 'c', marker = 'None', ls = '-', 
		#	linewidth = 1, ecolor = 'c', elinewidth = 1, alpha = 0.5, label = 'sky ICL')
		ax.plot(jk_sky_R, jk_sky_SB, color = 'c', ls = '-', linewidth = 1, alpha = 0.5, label = 'sky ICL')

		#ax.errorbar(m_BCG_r, m_BCG_SB, yerr = [m_BCG_err0, m_BCG_err1], xerr = None, color = 'k', marker = 'D', ls = '-', linewidth = 1, 
		#	markersize = 3, ecolor = 'k', elinewidth = 1, label = 'SDSS photo. cat.', alpha = 0.5)
		ax.plot(m_BCG_r, m_BCG_SB, color = 'k', ls = '-', linewidth = 1, label = 'SDSS photo. cat.', alpha = 0.5)

		ax.set_xlabel('$R[kpc]$')
		ax.set_ylabel('$SB[mag / arcsec^2]$')
		ax.set_xscale('log')
		ax.set_ylim(19, 34)
		ax.set_xlim(1, 2e3)
		ax.legend(loc = 1, fontsize = 8., frameon = False)
		ax.invert_yaxis()
		ax.grid(which = 'major', axis = 'both')
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')

		plt.savefig(d_load + '%s_band_tot-stack_SB_pros.png' % band[kk], dpi = 300)
		plt.close()
		########################################################
		## process img
		plt.figure(figsize = (16, 12))
		bx0 = plt.subplot(221)
		bx1 = plt.subplot(222)
		bx2 = plt.subplot(223)
		bx3 = plt.subplot(224)

		bx0.set_title('$ %s \; band \; tot-stack \; img [50 \\leqslant \\lambda] $' % band[kk] )
		clust00 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5,)
		clust01 = Circle(xy = (x0, y0), radius = 0.5 * Rpp, fill = False, ec = 'r', ls = '--', alpha = 0.5,)
		tf = bx0.imshow(M_clust_img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
		plt.colorbar(tf, ax = bx0, fraction = 0.042, pad = 0.01, label = 'flux[nmaggy]')
		## add contour
		con_img = M_clust_img * 1.
		kernl_img = ndimage.gaussian_filter(M_clust_img, sigma = dnoise,  mode = 'nearest')
		SB_img = 22.5 - 2.5 * np.log10(kernl_img) + 2.5 * np.log10(pixel**2)
		tg = bx0.contour(SB_img, origin = 'lower', cmap = 'rainbow', levels = SB_lel, )
		plt.clabel(tg, inline = False, fontsize = 6.5, colors = 'k', fmt = '%.0f')

		bx0.add_patch(clust00)
		bx0.add_patch(clust01)
		bx0.axis('equal')
		bx0.set_xlim(x0 - 0.7 * R_cut, x0 + 0.7 * R_cut)
		bx0.set_ylim(y0 - 0.7 * R_cut, y0 + 0.7 * R_cut)
		bx0.set_xticks([])
		bx0.set_yticks([])

		bx1.set_title('%s band difference img' % band[kk] )
		clust10 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-',)
		clust11 = Circle(xy = (x0, y0), radius = 0.5 * Rpp, fill = False, ec = 'r', ls = '--',)
		tf = bx1.imshow(M_difference, origin = 'lower', cmap = 'seismic', vmin = -2e-4, vmax = 2e-4)
		plt.colorbar(tf, ax = bx1, fraction = 0.042, pad = 0.01, label = 'flux[nmaggy]')
		bx1.add_patch(clust10)
		bx1.add_patch(clust11)
		bx1.axis('equal')
		bx1.set_xlim(x0 - 0.7 * R_cut, x0 + 0.7 * R_cut)
		bx1.set_ylim(y0 - 0.7 * R_cut, y0 + 0.7 * R_cut)
		bx1.set_xticks([])
		bx1.set_yticks([])

		bx2.set_title('%s band difference + stack img' % band[kk] )
		clust20 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5,)
		clust21 = Circle(xy = (x0, y0), radius = 0.5 * Rpp, fill = False, ec = 'r', ls = '--', alpha = 0.5,)
		tf = bx2.imshow(M_difference + M_clust_img, 
			cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
		plt.colorbar(tf, ax = bx2, fraction = 0.042, pad = 0.01, label = 'flux[nmaggy]')
		## add contour
		con_img = M_difference + M_clust_img
		kernl_img = ndimage.gaussian_filter(con_img, sigma = dnoise,  mode = 'nearest')
		SB_img = 22.5 - 2.5 * np.log10(kernl_img) + 2.5 * np.log10(pixel**2)
		tg = bx2.contour(SB_img, origin = 'lower', cmap = 'rainbow', levels = SB_lel, )
		plt.clabel(tg, inline = False, fontsize = 6.5, colors = 'k', fmt = '%.0f')

		bx2.add_patch(clust20)
		bx2.add_patch(clust21)
		bx2.axis('equal')
		bx2.set_xlim(x0 - 0.7 * R_cut, x0 + 0.7 * R_cut)
		bx2.set_ylim(y0 - 0.7 * R_cut, y0 + 0.7 * R_cut)
		bx2.set_xticks([])
		bx2.set_yticks([])

		bx3.set_title('%s band difference + stack - RBL' % band[kk] )
		clust30 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-', alpha = 0.5,)
		clust31 = Circle(xy = (x0, y0), radius = 0.5 * Rpp, fill = False, ec = 'r', ls = '--', alpha = 0.5,)

		tf = bx3.imshow(M_difference + M_clust_img - Resi_bl, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, 
			norm = mpl.colors.LogNorm())
		plt.colorbar(tf, ax = bx3, fraction = 0.042, pad = 0.01, label = 'flux[nmaggy]')
		## add contour
		con_img = M_difference + M_clust_img - Resi_bl
		kernl_img = ndimage.gaussian_filter(con_img, sigma = dnoise,  mode = 'nearest')
		SB_img = 22.5 - 2.5 * np.log10(kernl_img) + 2.5 * np.log10(pixel**2)
		tg = bx3.contour(SB_img, origin = 'lower', cmap = 'rainbow', levels = SB_lel, )
		plt.clabel(tg, inline = False, fontsize = 6.5, colors = 'k', fmt = '%.0f')

		bx3.add_patch(clust30)
		bx3.add_patch(clust31)
		bx3.axis('equal')
		bx3.set_xlim(x0 - 0.7 * R_cut, x0 + 0.7 * R_cut)
		bx3.set_ylim(y0 - 0.7 * R_cut, y0 + 0.7 * R_cut)
		bx3.set_xticks([])
		bx3.set_yticks([])

		plt.tight_layout()
		plt.savefig(d_load + '%s_band_tot-stack_process.png' % band[kk], dpi = 300)
		plt.close()

	return

def main():
	cor_id = 1 # 0, 1 (see description at the beginning)
	rich_a0, rich_a1, rich_a2 = 20, 30, 50
	## sersic pro of Zibetti 05
	mu_e = np.array([23.87, 25.22, 23.4])
	r_e = np.array([19.29, 19.40, 20])

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	R_cut, bins = 1280, 80
	R_smal, R_max = 1, 1.7e3 # kpc
	r_a0, r_a1 = 1.0, 1.1

	## R200 calculate parameter
	dnoise = 25
	SB_lel = np.arange(26, 31, 1) ## cotour c-label
	M0, lamd0, z0 = 14.37, 30, 0.5
	F_lamda, G_z = 1.12, 0.18
	V_num = 200

	N_bin = 20
	"""
	## stack cluster img
	for kk in range(3):

		with h5py.File(load + 'mpi_h5/%s_band_sky_catalog.h5' % band[kk], 'r') as f:
			set_array = np.array(f['a'])
		set_ra, set_dec, set_z, set_rich = set_array[0,:], set_array[1,:], set_array[2,:], set_array[3,:]

		idx = set_rich >= rich_a2
		lis_z = set_z[idx]
		lis_ra = set_ra[idx]
		lis_dec = set_dec[idx]

		zN = len(lis_z)
		n_step = zN // N_bin
		id_arr = np.linspace(0, zN - 1, zN)
		id_arr = id_arr.astype(int)

		for nn in range(N_bin):
			if nn == N_bin - 1:
				dot = id_arr[nn * n_step:]
			else:
				dot = id_arr[nn * n_step: (nn + 1) * n_step]
			id_use = list( set( id_arr ).difference( set( dot ) ) )
			id_use = np.array(id_use)

			z_use = lis_z[id_use]
			ra_use = lis_ra[id_use]
			dec_use = lis_dec[id_use]
			m, n = divmod( len(z_use), cpus )
			N_sub0, N_sub1 = m * rank, (rank + 1) * m
			if rank == cpus - 1:
				N_sub1 += n

			rich_divid(kk, z_use[N_sub0 :N_sub1], ra_use[N_sub0 :N_sub1], dec_use[N_sub0 :N_sub1])
			commd.Barrier()

			if rank == 0:

				tot_N = 0
				mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

				for pp in range(cpus):

					with h5py.File(tmp + 'stack_Amask_pcount_%d_in_%s_band.h5' % (pp, band[kk]), 'r')as f:
						p_count = np.array(f['a'])
					with h5py.File(tmp + 'stack_Amask_sum_%d_in_%s_band.h5' % (pp, band[kk]), 'r') as f:
						sum_img = np.array(f['a'])

					tot_N += p_count[0, 0]
					id_zero = p_count == 0
					ivx = id_zero == False
					mean_img[ivx] = mean_img[ivx] + sum_img[ivx]
					p_add_count[ivx] = p_add_count[ivx] + p_count[ivx]

				id_zero = p_add_count == 0
				mean_img[id_zero] = np.nan
				p_add_count[id_zero] = np.nan
				tot_N = np.int(tot_N)
				stack_img = mean_img / p_add_count
				where_are_inf = np.isinf(stack_img)
				stack_img[where_are_inf] = np.nan

				with h5py.File(load + 
					'rich_sample/test_img/%d_sub-stack_%s_band_img.h5' % (nn, band[kk]), 'w') as f:
					f['a'] = np.array(stack_img)

			commd.Barrier()
	## stack sky img
	for kk in range( 3 ):

		with h5py.File(load + 'mpi_h5/%s_band_sky_catalog.h5' % band[kk], 'r') as f:
			set_array = np.array(f['a'])
		set_ra, set_dec, set_z, set_rich = set_array[0,:], set_array[1,:], set_array[2,:], set_array[3,:]

		idx = (set_rich >= rich_a2)
		lis_z = set_z[idx]
		lis_ra = set_ra[idx]
		lis_dec = set_dec[idx]
		lis_rich = set_rich[idx]

		zN = len(lis_z)
		n_step = zN // N_bin
		id_arr = np.linspace(0, zN - 1, zN)
		id_arr = id_arr.astype(int)

		for nn in range( N_bin ):
			if nn == N_bin - 1:
				dot = id_arr[nn * n_step:]
			else:
				dot = id_arr[nn * n_step: (nn + 1) * n_step]
			id_use = list( set( id_arr ).difference( set( dot ) ) )
			id_use = np.array(id_use)

			z_use = lis_z[id_use]
			ra_use = lis_ra[id_use]
			dec_use = lis_dec[id_use]
			m, n = divmod( len(z_use), cpus )
			N_sub0, N_sub1 = m * rank, (rank + 1) * m
			if rank == cpus - 1:
				N_sub1 += n

			sky_stack_BCG(kk, z_use[N_sub0 :N_sub1], ra_use[N_sub0 :N_sub1], dec_use[N_sub0 :N_sub1], cor_id)
			commd.Barrier()

			## combine all of the sub-stack imgs
			if rank == 0:
				tt_N = 0
				bcg_stack = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				bcg_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

				for pp in range(cpus):
					with h5py.File(tmp + 'sky_sum_pcount_%d_in_%s_band.h5' % (pp, band[kk]), 'r')as f:
						p_count = np.array(f['a'])
					with h5py.File(tmp + 'sky_sum_%d_in_%s_band.h5' % (pp, band[kk]), 'r') as f:
						sum_img = np.array(f['a'])

					tt_N += p_count[0, 0]
					id_zero = p_count == 0
					ivx = id_zero == False
					bcg_stack[ivx] = bcg_stack[ivx] + sum_img[ivx]
					bcg_count[ivx] = bcg_count[ivx] + p_count[ivx]

				## sample sky SB
				tt_N = np.int(tt_N)
				## centered on BCG
				id_zero = bcg_count == 0
				bcg_stack[id_zero] = np.nan
				bcg_count[id_zero] = np.nan
				stack_img = bcg_stack / bcg_count
				id_inf = np.isinf(stack_img)
				stack_img[id_inf] = np.nan

				if cor_id == 0:
					with h5py.File(load + 'rich_sample/test_img/sky-mean_%d_sub-stack_%s_band_img.h5' % (nn, band[kk]), 'w') as f:
						f['a'] = np.array(stack_img)
				if cor_id == 1:
					with h5py.File(load + 'rich_sample/test_img/sky-median_%d_sub-stack_%s_band_img.h5' % (nn, band[kk]), 'w') as f:
						f['a'] = np.array(stack_img)
			commd.Barrier()

	for d_record in range(1, 6):

		for kk in range( 3 ):

			with h5py.File(load + 'mpi_h5/%s_band_sky_catalog.h5' % band[kk], 'r') as f:
				set_array = np.array(f['a'])
			set_ra, set_dec, set_z, set_rich = set_array[0,:], set_array[1,:], set_array[2,:], set_array[3,:]

			idx = (set_rich >= rich_a2)
			lis_z = set_z[idx]
			lis_ra = set_ra[idx]
			lis_dec = set_dec[idx]
			lis_rich = set_rich[idx]

			zN = len(lis_z)
			n_step = zN // N_bin
			id_arr = np.linspace(0, zN - 1, zN)
			id_arr = id_arr.astype(int)

			for nn in range( N_bin ):
				if nn == N_bin - 1:
					dot = id_arr[nn * n_step:]
				else:
					dot = id_arr[nn * n_step: (nn + 1) * n_step]
				id_use = list( set( id_arr ).difference( set( dot ) ) )
				id_use = np.array(id_use)

				z_use = lis_z[id_use]
				ra_use = lis_ra[id_use]
				dec_use = lis_dec[id_use]
				m, n = divmod( len(z_use), cpus )
				N_sub0, N_sub1 = m * rank, (rank + 1) * m
				if rank == cpus - 1:
					N_sub1 += n
				sky_stack_rndm(kk, z_use[N_sub0 :N_sub1], ra_use[N_sub0 :N_sub1], dec_use[N_sub0 :N_sub1], cor_id)
				commd.Barrier()

				if rank == 0:
					tt_N = 0
					rand_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
					rand_cnt = np.zeros((len(Ny), len(Nx)), dtype = np.float)
					for pp in range(cpus):
						with h5py.File(tmp + 'rndm_sum_%d_in_%s_band.h5' % (pp, band[kk]), 'r') as f:
							rndm_sum = np.array(f['a'])
						with h5py.File(tmp + 'rndm_sum_pcount_%d_in_%s_band.h5' % (pp, band[kk]), 'r') as f:
							rndm_pcont = np.array(f['a'])
						tt_N += rndm_pcont[0, 0]
						id_zero = rndm_pcont == 0
						ivx = id_zero == False
						rand_img[ivx] = rand_img[ivx] + rndm_sum[ivx]
						rand_cnt[ivx] = rand_cnt[ivx] + rndm_pcont[ivx]

					tt_N = np.int(tt_N)
					id_zero = rand_cnt == 0
					rand_img[id_zero] = np.nan
					rand_cnt[id_zero] = np.nan
					random_stack = rand_img / rand_cnt
					id_inf = np.isinf(random_stack)
					random_stack[id_inf] = np.nan

					if cor_id == 0:
						with h5py.File(load + 
							'rich_sample/test_img/rndm_sky-mean_%d_sub-stack_%s_band_img_%d_rnd.h5' % (nn, band[kk], d_record), 'w') as f:
							f['a'] = np.array(random_stack)
					if cor_id == 1:
						with h5py.File(load + 
							'rich_sample/test_img/rndm_sky-median_%d_sub-stack_%s_band_img_%d_rnd.h5' % (nn, band[kk], d_record), 'w') as f:
							f['a'] = np.array(random_stack)
				commd.Barrier()

	if rank == 0:
		for kk in range(3):

			for nn in range( N_bin ):

				m_rndm_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				rndm_cnt = np.zeros((len(Ny), len(Nx)), dtype = np.float)

				for id_rec in range(1, 6):
					if cor_id == 0:
						with h5py.File(load + 
						'rich_sample/test_img/rndm_sky-mean_%d_sub-stack_%s_band_img_%d_rnd.h5' % (nn, band[kk], id_rec), 'r') as f:
							sub_img = np.array(f['a'])
					if cor_id == 1:
						with h5py.File(load + 
						'rich_sample/test_img/rndm_sky-median_%d_sub-stack_%s_band_img_%d_rnd.h5' % (nn, band[kk], id_rec), 'r') as f:
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
					with h5py.File(load + 'rich_sample/test_img/M_rndm_sky-mean_%d_sub-stack_%s_band.h5' % (nn, band[kk]), 'w') as f:
						f['a'] = np.array(m_rndm_img)
				if cor_id == 1:
					with h5py.File(load + 'rich_sample/test_img/M_rndm_sky-median_%d_sub-stack_%s_band.h5' % (nn, band[kk]), 'w') as f:
						f['a'] = np.array(m_rndm_img)
	commd.Barrier()
	"""
	commd.Barrier()
	SB_result()
	commd.Barrier()

if __name__ == "__main__":
	main()
