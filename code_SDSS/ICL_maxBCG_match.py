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
import astropy.io.ascii as asc
import astropy.io.fits as fits

from astropy import cosmology as apcy
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d as interp
from light_measure import light_measure, flux_recal, sigmamc

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

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

def catlg_patch():
	## maxBCG cat.
	dat_max = pds.read_csv(load + 'Zibetti_SB/MaxBCG_CAT_cor.dat', sep = '\s+')
	ra_max = dat_max['Ra']
	dec_max = dat_max['Dec']
	z_max = dat_max['Spec_Z_bcg']

	Ra = ['%.4f' % ll for ll in ra_max]
	Dec = ['%.4f' % ll for ll in dec_max]
	Zbcg = ['%.4f' % ll for ll in z_max]

	for kk in range(len(band)):
		## selected redmapper cat.
		dat_red = pds.read_csv(load + 'selection/%s_band_sky_catalog.csv' % band[kk])
		ra_red = np.array(dat_red['ra'])
		dec_red = np.array(dat_red['dec'])
		z_red = np.array(dat_red['z'])

		ra_set = np.zeros(len(z_red), dtype = np.float)
		dec_set = np.zeros(len(z_red), dtype = np.float)
		z_set = np.zeros(len(z_red), dtype = np.float)

		for jj in range(len(z_red)):
			ra_g = '%.4f' % ra_red[jj]
			dec_g = '%.4f' % dec_red[jj]
			z_g = '%.4f' % z_red[jj]

			identy = (ra_g in Ra) & (dec_g in Dec) & (z_g in Zbcg)

			if identy == True:
				ra_set[jj] = ra_red[jj]
				dec_set[jj] = dec_red[jj]
				z_set[jj] = z_red[jj]
			else:
				continue

		id_zero = z_set == 0.
		id_fals = id_zero == False
		ra_set = ra_set[id_fals]
		dec_set = dec_set[id_fals]
		z_set = z_set[id_fals]

		keys = ['ra', 'dec', 'z']
		values = [ra_set, dec_set, z_set]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv(load + 'Zibetti_SB/%s_band_maxBCG_match.csv' % band[kk])
		## save h5 for mpirun
		rec_array = np.array([ra_set, dec_set, z_set])
		with h5py.File(load + 'mpi_h5/%s_band_maxBCG_match.h5' % band[kk], 'w') as f:
			f['a'] = np.array(rec_array)
		with h5py.File(load + 'mpi_h5/%s_band_maxBCG_match.h5' % band[kk],) as f:
			for ii in range(len(rec_array)):
				f['a'][ii, :] = rec_array[ii, :]
	return

def max_stack(band_id, sub_z, sub_ra, sub_dec):
	stack_N = len(sub_z)
	ii = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	n_pix = 0.
	for jj in range(stack_N):

		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]
		Da_g = Test_model.angular_diameter_distance(z_g).value

		data_A = fits.getdata(load + 'resample/1_5sigma_larger_R/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
		img_A = data_A[0]
		xn = data_A[1]['CENTER_X']
		yn = data_A[1]['CENTER_Y']

		la0 = np.int(y0 - yn)
		la1 = np.int(y0 - yn + img_A.shape[0])
		lb0 = np.int(x0 - xn)
		lb1 = np.int(x0 - xn + img_A.shape[1])

		idx = np.isnan(img_A)
		idv = np.where(idx == False)
		'''
		## select the 1 Mpc ~ 1.1 Mpc region as background
		grd_x = np.linspace(0, img_A.shape[1] - 1, img_A.shape[1])
		grd_y = np.linspace(0, img_A.shape[0] - 1, img_A.shape[0])
		grd = np.array( np.meshgrid(grd_x, grd_y) )
		ddr = np.sqrt( (grd[0,:] - xn)**2 + (grd[1,:] - yn)**2 )
		idu = (ddr > Rpp) & (ddr < 1.1 * Rpp) # using SB in 1 Mpc ~ 1.1 Mpc region as residual sky
		'''
		#BL = np.nanmean(img_A[idu])
		BL = 0.
		sub_BL_img = img_A - BL

		sum_array_A[la0: la1, lb0: lb1][idv] = sum_array_A[la0: la1, lb0: lb1][idv] + sub_BL_img[idv]
		count_array_A[la0: la1, lb0: lb1][idv] = sub_BL_img[idv]
		id_nan = np.isnan(count_array_A)
		id_fals = np.where(id_nan == False)
		p_count_A[id_fals] = p_count_A[id_fals] + 1.
		count_array_A[la0: la1, lb0: lb1][idv] = np.nan
		n_pix += 1.

	p_count_A[0, 0] = n_pix
	with h5py.File(tmp + 'stack_max_sum_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(sum_array_A)
	with h5py.File(tmp + 'stack_max_pcount_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(p_count_A)

	return

def main():
	#catlg_patch()
	R_cut, bins = 1280, 80  # in unit of pixel (with 2Mpc inside)
	R_smal, R_max = 10, 1.7e3 # kpc

	x0, y0, bins = 2427, 1765, 65
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	## sersic pro of Zibetti 05
	mu_e = np.array([23.87, 25.22, 23.4])
	r_e = np.array([19.29, 19.40, 20])

	'''
	for tt in range(3):
		with h5py.File( load + 'mpi_h5/%s_band_maxBCG_match.h5' % band[tt], 'r') as f:
			rec_array = np.array(f['a'])
		ra, dec, z = rec_array[0,:], rec_array[1,:], rec_array[2,:]

		zN = len(z)
		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		max_stack(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
		commd.Barrier()

		if rank == 0:

			tot_N = 0
			mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
			p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)
			for pp in range(cpus):

				with h5py.File(tmp + 'stack_max_pcount_%d_in_%s_band.h5' % (pp, band[tt]), 'r')as f:
					p_count = np.array(f['a'])
				with h5py.File(tmp + 'stack_max_sum_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
					sum_img = np.array(f['a'])

				tot_N += p_count[0, 0]
				id_zero = p_count == 0
				ivx = id_zero == False
				mean_img[ivx] = mean_img[ivx] + sum_img[ivx]
				p_add_count[ivx] = p_add_count[ivx] + p_count[ivx]

			## save the stack image
			id_zero = p_add_count == 0
			mean_img[id_zero] = np.nan
			p_add_count[id_zero] = np.nan
			tot_N = np.int(tot_N)
			stack_img = mean_img / p_add_count
			where_are_inf = np.isinf(stack_img)
			stack_img[where_are_inf] = np.nan

			with h5py.File(home + 'fig_ZIT/stack_max_%d_in_%s_band.h5' % (tot_N, band[tt]), 'w') as f:
				f['a'] = np.array(stack_img)

		commd.Barrier()
	'''
	N_sum = np.array([1223, 1224, 1219, 1221, 1219]) ## maxBCG match
	N_sky = np.array([3308, 3309, 3295, 3308, 3305])
	r_a0, r_a1 = 1.0, 1.1
	if rank == 0:
		for kk in range(3):

			SB_tt = pds.read_csv( load + 'Zibetti_SB/%s_band_BCG_ICL.csv' % band[kk])
			R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
			R_obs = R_obs**4
			## sersic part
			Mu_e, R_e, n_e = mu_e[kk], r_e[kk], 4.
			SB_Z05 = sers_pro(R_obs, Mu_e, R_e, n_e)

			with h5py.File(home + 'fig_ZIT/stack_max_%d_in_%s_band.h5' % (N_sum[kk], band[kk]), 'r') as f:
				stack_img = np.array(f['a'])
			ss_img = stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]

			Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
			SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2) + mag_add[kk]
			id_nan = np.isnan(SB)
			SBt, Rt = SB[id_nan == False], Intns_r[id_nan == False]

			with h5py.File(load + 'sky/cluster/sky_minus_media_%d_imgs_%s_band.h5' % (N_sky[kk], band[kk]), 'r') as f:
				BCG_add = np.array(f['a'])
			with h5py.File(load + 'sky/cluster/mean_sky_shuffle_media_%d_imgs_%s_band.h5' % (N_sky[kk], band[kk]), 'r') as f:
			#with h5py.File(load + 'sky/cluster/mean_sky_random_media_%d_imgs_%s_band.h5' % (N_sky[kk], band[kk]), 'r') as f:
				shlf_add = np.array(f['a'])
			resi_add = BCG_add - shlf_add

			add_img = ss_img + resi_add[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
			Intns, Intns_r, Intns_err, Npix = light_measure(add_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
			SB_add = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2) + mag_add[kk]
			R_add = Intns_r * 1

			cen_pos = R_cut * 1 # 1280 pixel, for z = 0.25, larger than 2Mpc
			BL_img = add_img * 1
			grd_x = np.linspace(0, BL_img.shape[1] - 1, BL_img.shape[1])
			grd_y = np.linspace(0, BL_img.shape[0] - 1, BL_img.shape[0])
			grd = np.array( np.meshgrid(grd_x, grd_y) )
			ddr = np.sqrt( (grd[0,:] - cen_pos)**2 + (grd[1,:] - cen_pos)**2 )
			idu = (ddr > r_a0 * Rpp) & (ddr < r_a1 * Rpp)
			Resi_bl = np.nanmean( BL_img[idu] )

			sub_SB = 22.5 - 2.5 * np.log10(Intns - Resi_bl) + 2.5 * np.log10(pixel**2) + mag_add[kk]
			flux0 = Intns + Intns_err - Resi_bl
			flux1 = Intns - Intns_err - Resi_bl
			dSB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2) + mag_add[kk]
			dSB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2) + mag_add[kk]
			err0 = sub_SB - dSB0
			err1 = dSB1 - sub_SB

			id_nan = np.isnan(sub_SB)
			cli_SB, cli_R, cli_err0, cli_err1 = sub_SB[id_nan == False], R_add[id_nan == False], err0[id_nan == False], err1[id_nan == False]
			dSB0, dSB1 = dSB0[id_nan == False], dSB1[id_nan == False]
			idx_nan = np.isnan(dSB1)
			cli_err1[idx_nan] = 100.

			plt.figure()
			ax = plt.subplot(111)
			ax.set_title('maxBCG match stacking [%s band %d imgs]' % (band[kk], N_sum[kk]) )
			ax.plot(R_obs, SB_obs, 'k-.', label = 'Z05', alpha = 0.5)
			ax.plot(R_obs, SB_Z05, 'k:', label = 'Sersic Z05', alpha = 0.5)

			ax.plot(Rt, SBt, linestyle = '-', color = 'r', label = '$ stack \, imgs $', alpha = 0.5)
			ax.plot(R_add, SB_add, linestyle = '-.', color = 'g', label = 'Add over-subtracted light', alpha = 0.5)
			ax.errorbar(cli_R, cli_SB, yerr = [cli_err0, cli_err1], xerr = None, color = 'r', marker = '.', ls = '', linewidth = 1, markersize = 5, 
				ecolor = 'r', elinewidth = 1, label = 'RBL subtracted', alpha = 0.5)

			ax.set_xlabel('$R[kpc]$')
			ax.set_ylabel('$SB[mag / arcsec^2]$')
			ax.set_xscale('log')
			ax.set_ylim(20, 33)
			ax.set_xlim(1, 1.5e3)
			ax.legend(loc = 3, fontsize = 7.5)
			ax.invert_yaxis()
			ax.grid(which = 'both', axis = 'both')
			ax.tick_params(axis = 'both', which = 'both', direction = 'in')

			plt.savefig(home + 'fig_ZIT/maxBCG_match_SB_%d_imgs_%s_band.png' %(N_sum[kk], band[kk]), dpi = 300)
			plt.close()
	commd.Barrier()
	raise

if __name__ == "__main__":
	main()
