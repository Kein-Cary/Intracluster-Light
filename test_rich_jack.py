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
import statistics as sts

from scipy import ndimage
from astropy import cosmology as apcy
from light_measure import light_measure
from Mass_rich_radius import rich2R

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

home = '/mnt/ddnfs/data_users/cxkttwl/ICL/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'
band = ['r', 'g', 'i', 'u', 'z']
mag_add = np.array([0, 0, 0, -0.04, 0.02])

def sers_pro(r, mu_e, r_e, n):
	belta_n = 2 * n - 0.324
	fn = 1.086 * belta_n * ( (r/r_e)**(1/n) - 1)
	mu_r = mu_e + fn
	return mu_r

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

	return R_out, SB_out, out_err0, out_err1

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

	for kk in range(3):

		SB_tt = pds.read_csv( load + 'Zibetti_SB/%s_band_BCG_ICL.csv' % band[kk])
		R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
		R_obs = R_obs**4
		## sersic part
		Mu_e, R_e, n_e = mu_e[kk], r_e[kk], 4.
		SB_Z05 = sers_pro(R_obs, Mu_e, R_e, n_e)

		## calculate the mean result of those sub-stack
		m_BCG_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		BCG_cont_Mx = np.zeros((len(Ny), len(Nx)), dtype = np.float)

		m_differ_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		differ_cont = np.zeros((len(Ny), len(Nx)), dtype = np.float)

		SB_flux = []
		R_arr = []

		fig = plt.figure(figsize = (20, 20))
		fig.suptitle('%s band sub-sample SB profile' % band[kk])
		gs = gridspec.GridSpec(N_bin // 5, 5)

		for nn in range(N_bin):
			## cluster image
			with h5py.File(load + 'rich_sample/test_img/%d_sub-stack_%s_band_img.h5' % (nn, band[kk]), 'r') as f:
				clust_img = np.array(f['a'])

			idnx = np.isnan(clust_img)
			idv = np.where(idnx == False)
			m_BCG_img[idv] = m_BCG_img[idv] + clust_img[idv]
			BCG_cont_Mx[idv] += 1.

			ss_img = clust_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
			Rt, SBt, t_err0, t_err1 = SB_pro(ss_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref, kk)

			## sky image
			with h5py.File(load + 'rich_sample/test_img/sky-median_%d_sub-stack_%s_band_img.h5' % (nn, band[kk]), 'r') as f:
				BCG_sky = np.array(f['a'])
			with h5py.File(load + 'rich_sample/test_img/M_rndm_sky-median_%d_sub-stack_%s_band.h5' % (nn, band[kk]), 'r') as f:
				rand_sky = np.array(f['a'])

			differ_img = BCG_sky - rand_sky
			with h5py.File(load + 'rich_sample/test_img/%s_band_%d_sub-differ_img.h5' % (band[kk], nn), 'w') as f:
				f['a'] = np.array(differ_img)

			idnx = np.isnan(differ_img)
			idv = np.where(idnx == False)
			m_differ_img[idv] = m_differ_img[idv] + differ_img[idv]
			differ_cont[idv] += 1.
			resi_add = differ_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
			R_sky, sky_ICL, sky_err0, sky_err1 = SB_pro(resi_add, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref, kk)

			## add the sky difference image
			add_img = ss_img + resi_add
			R_add, SB_add, add_err0, add_err1 = SB_pro(add_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref, kk)

			## RBL estimation
			cen_pos = R_cut * 1 # 1280 pixel, for z = 0.25, larger than 2Mpc
			BL_img = add_img * 1
			grd_x = np.linspace(0, BL_img.shape[1] - 1, BL_img.shape[1])
			grd_y = np.linspace(0, BL_img.shape[0] - 1, BL_img.shape[0])
			grd = np.array( np.meshgrid(grd_x, grd_y) )
			ddr = np.sqrt( (grd[0,:] - cen_pos)**2 + (grd[1,:] - cen_pos)**2 )
			idu = (ddr > r_a0 * Rpp) & (ddr < r_a1 * Rpp)
			Resi_bl = np.nanmean( BL_img[idu] )
			Resi_sky = 22.5 - 2.5 * np.log10(Resi_bl) + 2.5 * np.log10(pixel**2) + mag_add[kk]

			## minus the residual background
			minus_img = add_img - Resi_bl
			Intns, Intns_r, Intns_err, Npix = light_measure(minus_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
			SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2) + mag_add[kk]

			SB_flux.append(Intns)
			R_arr.append(Intns_r)

			flux0 = Intns + Intns_err
			flux1 = Intns - Intns_err
			dSB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2) + mag_add[kk]
			dSB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2) + mag_add[kk]
			err0 = SB - dSB0
			err1 = dSB1 - SB
			id_nan = np.isnan(SB)
			cli_SB, cli_R, cli_err0, cli_err1 = SB[id_nan == False], Intns_r[id_nan == False], err0[id_nan == False], err1[id_nan == False]
			dSB0, dSB1 = dSB0[id_nan == False], dSB1[id_nan == False]
			idx_nan = np.isnan(dSB1)
			cli_err1[idx_nan] = 100.

			ax = plt.subplot(gs[nn // 5, nn % 5])
			ax.errorbar(Rt, SBt, yerr = [t_err0, t_err1], xerr = None, color = 'r', marker = '.', ls = '-', linewidth = 1, 
			markersize = 5, ecolor = 'r', elinewidth = 1, label = 'Stacking', alpha = 0.5)
			ax.errorbar(R_add, SB_add, yerr = [add_err0, add_err1], xerr = None, color = 'g', marker = '.', ls = '-', linewidth = 1, 
			markersize = 5, ecolor = 'g', elinewidth = 1, label = 'Stacking + Difference', alpha = 0.5)
			ax.errorbar(cli_R, cli_SB, yerr = [cli_err0, cli_err1], xerr = None, color = 'b', marker = '.', ls = '-', linewidth = 1, 
			markersize = 5, ecolor = 'b', elinewidth = 1, label = 'Stacking + Difference - RBL', alpha = 0.5)
			ax.errorbar(R_sky, sky_ICL, yerr = [sky_err0, sky_err1], xerr = None, color = 'm', marker = '^', ls = '--', linewidth = 1, 
			markersize = 5, ecolor = 'm', elinewidth = 1, label = '$ sky \, ICL $', alpha = 0.5)

			ax.plot(R_obs, SB_obs, 'k-.', label = 'Z05', alpha = 0.5)
			#ax.plot(R_obs, SB_Z05, 'k:', label = 'Sersic', alpha = 0.5)
			ax.axhline( y = Resi_sky, ls = '-', color = 'k', label = 'RBL', alpha = 0.5)
			ax.set_xlabel('$R[kpc]$')
			ax.set_ylabel('$SB[mag / arcsec^2]$')
			ax.set_xscale('log')
			ax.set_ylim(20, 35)
			ax.set_xlim(1, 1.5e3)
			ax.legend(loc = 1)
			ax.invert_yaxis()
			ax.grid(which = 'both', axis = 'both')
			ax.tick_params(axis = 'both', which = 'both', direction = 'in')
		plt.tight_layout()
		plt.savefig(load + 'rich_sample/test_img/%s_band_sub-stack_SB_pros.png' % band[kk], dpi = 300)
		plt.close()

		M_clust_img = m_BCG_img / BCG_cont_Mx
		id_inf = np.isinf(M_clust_img)
		id_zeros = M_clust_img == 0.
		M_clust_img[id_inf] = np.nan
		M_clust_img[id_zeros] = np.nan
		with h5py.File(load + 'rich_sample/test_img/%s_band_clust_tot-stack_img.h5' % band[kk], 'w') as f:
			f['a'] = np.array(M_clust_img)

		M_difference = m_differ_img / differ_cont
		id_inf = np.isinf(M_difference)
		id_zeros = M_difference == 0.
		M_difference[id_inf] = np.nan
		M_difference[id_zeros] = np.nan
		with h5py.File(load + 'rich_sample/test_img/%s_band_tot-difference_img.h5' % band[kk], 'w') as f:
			f['a'] = np.array(M_clust_img)

		ss_img = M_clust_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
		Rt, SBt, t_err0, t_err1 = SB_pro(ss_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref, kk)

		resi_add = M_difference[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
		R_sky, sky_ICL, sky_err0, sky_err1 = SB_pro(resi_add, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref, kk)

		add_img = ss_img + resi_add
		R_add, SB_add, add_err0, add_err1 = SB_pro(add_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref, kk)

		## RBL estimation
		cen_pos = R_cut * 1 # 1280 pixel, for z = 0.25, larger than 2Mpc
		BL_img = add_img * 1
		grd_x = np.linspace(0, BL_img.shape[1] - 1, BL_img.shape[1])
		grd_y = np.linspace(0, BL_img.shape[0] - 1, BL_img.shape[0])
		grd = np.array( np.meshgrid(grd_x, grd_y) )
		ddr = np.sqrt( (grd[0,:] - cen_pos)**2 + (grd[1,:] - cen_pos)**2 )
		idu = (ddr > r_a0 * Rpp) & (ddr < r_a1 * Rpp)
		Resi_bl = np.nanmean( BL_img[idu] )
		Resi_sky = 22.5 - 2.5 * np.log10(Resi_bl) + 2.5 * np.log10(pixel**2) + mag_add[kk]
		## minus the residual background
		minus_img = add_img - Resi_bl
		cli_R, cli_SB, cli_err0, cli_err1 = SB_pro(minus_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref, kk)

		## calculate the mean result of sub-SB_profile
		SB_flux = np.array(SB_flux)
		R_arr = np.array(R_arr)

		m_R = np.nanmean(R_arr, axis = 0)
		m_SB = np.nanmean(SB_flux, axis = 0)
		std_SB = np.nanstd(SB_flux, axis = 0)
		jk_std = np.sqrt(N_bin - 1) * std_SB
		jk_SB = 22.5 - 2.5 * np.log10(m_SB) + 2.5 * np.log10(pixel**2) + mag_add[kk]
		dSB0 = 22.5 - 2.5 * np.log10(m_SB + jk_std) + 2.5 * np.log10(pixel**2) + mag_add[kk]
		dSB1 = 22.5 - 2.5 * np.log10(m_SB - jk_std) + 2.5 * np.log10(pixel**2) + mag_add[kk]
		err0 = jk_SB - dSB0
		err1 = dSB1 - jk_SB
		id_nan = np.isnan(jk_SB)
		JK_SB, JK_R, JK_err0, JK_err1 = jk_SB[id_nan == False], m_R[id_nan == False], err0[id_nan == False], err1[id_nan == False]
		dSB0, dSB1 = dSB0[id_nan == False], dSB1[id_nan == False]
		idx_nan = np.isnan(dSB1)
		JK_err1[idx_nan] = 100.

		plt.figure()
		ax = plt.subplot(111)
		ax.set_title('%s band jackknife stacking SB' % band[kk])
		ax.errorbar(Rt, SBt, yerr = [t_err0, t_err1], xerr = None, color = 'r', marker = '.', ls = '-', linewidth = 1, 
		markersize = 5, ecolor = 'r', elinewidth = 1, label = 'Stacking', alpha = 0.5)
		ax.errorbar(R_add, SB_add, yerr = [add_err0, add_err1], xerr = None, color = 'g', marker = '.', ls = '-', linewidth = 1, 
		markersize = 5, ecolor = 'g', elinewidth = 1, label = 'Stacking + Difference', alpha = 0.5)
		ax.errorbar(cli_R, cli_SB, yerr = [cli_err0, cli_err1], xerr = None, color = 'b', marker = '.', ls = '-', linewidth = 1, 
		markersize = 5, ecolor = 'b', elinewidth = 1, label = 'Stacking + Difference - RBL', alpha = 0.5)
		ax.errorbar(R_sky, sky_ICL, yerr = [sky_err0, sky_err1], xerr = None, color = 'c', marker = '^', ls = '--', linewidth = 1, 
		markersize = 5, ecolor = 'c', elinewidth = 1, label = '$ sky \, ICL $', alpha = 0.5)

		ax.errorbar(JK_R, JK_SB, yerr = [JK_err0, JK_err1], xerr = None, color = 'm', marker = 'o', ls = '-', linewidth = 1, 
		markersize = 5, ecolor = 'm', elinewidth = 1, label = '$ Jackknife $', alpha = 0.5)

		ax.plot(R_obs, SB_obs, 'k-.', label = 'Z05', alpha = 0.5)
		ax.axhline( y = Resi_sky, ls = '-', color = 'k', label = 'RBL', alpha = 0.5)
		ax.set_xlabel('$R[kpc]$')
		ax.set_ylabel('$SB[mag / arcsec^2]$')
		ax.set_xscale('log')
		ax.set_ylim(20, 35)
		ax.set_xlim(1, 1.5e3)
		ax.legend(loc = 1)
		ax.invert_yaxis()
		ax.grid(which = 'both', axis = 'both')
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')

		plt.savefig(load + 'rich_sample/test_img/%s_band_tot-stack_SB_pros.png' % band[kk], dpi = 300)
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
	if rank == 0:
		SB_result()
	commd.Barrier()

if __name__ == "__main__":
	main()
