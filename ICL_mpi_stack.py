import matplotlib as mpl
mpl.use('Agg')
import handy.scatter as hsc
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
f0 = 3631*10**(-23) # (erg/s)/cm^-2
R0 = 1 # Mpc
Angu_ref = (R0/Da_ref)*rad2asec
Rpp = Angu_ref/pixel
M_dot = 4.83 # the absolute magnitude of SUN

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'
band = ['r', 'g', 'i', 'u', 'z']
bnd_indx = [2, 1, 3, 0, 4]
mag_add = np.array([0, 0, 0, -0.04, 0.02])

def sers_pro(r, mu_e, r_e, n):
	belta_n = 2 * n - 0.324
	fn = 1.086 * belta_n * ( (r/r_e)**(1/n) - 1)
	mu_r = mu_e + fn
	return mu_r

def SB_fit(r, m0, mc, c, m2l):
	bl = m0
	f_bl = 10**(0.4 * ( 22.5 - bl + 2.5 * np.log10(pixel**2) ) )

	surf_mass = sigmamc(r, mc, c)
	surf_lit = surf_mass / m2l
	SB_mod = M_dot - 2.5 * np.log10(surf_lit * 1e-6) + 10 * np.log10(1 + z_ref) + 21.572
	f_mod = 10**(0.4 * ( 22.5 - SB_mod + 2.5 * np.log10(pixel**2) ) )

	f_ref = f_mod + f_bl

	return f_ref

def chi2(X, *args):
	m0 = X[0]
	mc = X[1]
	c = X[2]
	m2l = X[3]
	r, data, yerr = args
	m0 = m0
	mc = mc
	m2l = m2l
	c = c
	mock_L = SB_fit(r, m0, mc, c, m2l)
	chi = np.sum( ( (mock_L - data) / yerr)**2 )
	return chi

def crit_r(Mc, c = 4.5):
	c = c
	M = 10**Mc
	rho_c = (kpc2m / Msun2kg)*(3*H0**2) / (8*np.pi*G)
	r200_c = (3*M / (4*np.pi*rho_c*200))**(1/3)
	rs = r200_c / c
	return rs, r200_c

def stack_process(band_number, subz, subra, subdec):

	stack_N = len(subz)
	ii = np.int(band_number)
	sub_z = subz
	sub_ra = subra
	sub_dec = subdec

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	sum_array_B = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_B = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_B = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	for jj in range(stack_N):

		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]
		Da_g = Test_model.angular_diameter_distance(z_g).value

		## stack A mask
		# 1.5sigma
		data_A = fits.getdata(load + 
			'resample/1_5sigma/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
		'''
		# Zibetti
		data_A = fits.getdata(load + 
			'resample/Zibetti/A_mask/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
		'''
		img_A = data_A[0]
		xn = data_A[1]['CENTER_X']
		yn = data_A[1]['CENTER_Y']

		la0 = np.int(y0 - yn)
		la1 = np.int(y0 - yn + img_A.shape[0])
		lb0 = np.int(x0 - xn)
		lb1 = np.int(x0 - xn + img_A.shape[1])

		idx = np.isnan(img_A)
		idv = np.where(idx == False)

		sum_array_A[la0:la1, lb0:lb1][idv] = sum_array_A[la0:la1, lb0:lb1][idv] + img_A[idv]
		count_array_A[la0: la1, lb0: lb1][idv] = img_A[idv]
		id_nan = np.isnan(count_array_A)
		id_fals = np.where(id_nan == False)
		p_count_A[id_fals] += 1.
		p_count_A[0, 0] += 1.
		count_array_A[la0: la1, lb0: lb1][idv] = np.nan

		## stack B mask
		# 1.5sigma
		data_B = fits.getdata(load + 
			'resample/resam_B/frameB-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
		'''
		# Zibetti
		data_B = fits.getdata(load + 
			'resample/Zibetti/B_mask/frameB-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
		'''
		img_B = data_B[0]
		xn = data_B[1]['CENTER_X']
		yn = data_B[1]['CENTER_Y']

		la0 = np.int(y0 - yn)
		la1 = np.int(y0 - yn + img_B.shape[0])
		lb0 = np.int(x0 - xn)
		lb1 = np.int(x0 - xn + img_B.shape[1])

		idx = np.isnan(img_B)
		idv = np.where(idx == False)

		sum_array_B[la0:la1, lb0:lb1][idv] = sum_array_B[la0:la1, lb0:lb1][idv] + img_B[idv]
		count_array_B[la0: la1, lb0: lb1][idv] = img_B[idv]
		id_nan = np.isnan(count_array_B)
		id_fals = np.where(id_nan == False)
		p_count_B[id_fals] += 1.
		p_count_B[0, 0] += 1.
		count_array_B[la0: la1, lb0: lb1][idv] = np.nan

	with h5py.File(tmp + 'stack_Amask_sum_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(sum_array_A)

	with h5py.File(tmp + 'stack_Amask_pcount_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(p_count_A)

	with h5py.File(tmp + 'stack_Bmask_sum_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(sum_array_B)

	with h5py.File(tmp + 'stack_Bmask_pcount_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(p_count_B)

	return

def main():
	## sersic pro of Zibetti 05
	mu_e = np.array([23.87, 25.22, 23.4])
	r_e = np.array([19.29, 19.40, 20])

	x0, y0, bins = 2427, 1765, 75
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	R_cut = 900
	R_smal, R_max = 1, 10**3.02 # kpc

	for tt in range(3):
		with h5py.File(load + 'mpi_h5/%s_band_sample_catalog.h5' % band[tt], 'r') as f:
			sub_array = np.array(f['a'])
		ra, dec, z, rich, r_mag = sub_array[0,:], sub_array[1,:], sub_array[2,:], sub_array[3,:], sub_array[4,:]
		zN = len(z)
		N_tt = np.array([zN])
		for aa in range(len(N_tt)):
			np.random.seed(1)
			tt0 = np.random.choice(zN, size = N_tt[aa], replace = False)
			set_z, set_ra, set_dec = z[tt0], ra[tt0], dec[tt0]

			m, n = divmod(N_tt[aa], cpus)
			N_sub0, N_sub1 = m * rank, (rank + 1) * m
			if rank == cpus - 1:
				N_sub1 += n
			stack_process(tt, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])
			commd.Barrier()

			### stack all of the sub-stack A mask image
			if rank == 0:
				# read the SB of BCG + ICL + residual sky
				SB_tt = pds.read_csv( load + 'Zibetti_SB/%s_band_BCG_ICL.csv' % band[tt])
				R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
				R_obs = R_obs**4

				tot_N = 0
				mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

				for pp in range(cpus):

					with h5py.File(tmp + 'stack_Amask_pcount_%d_in_%s_band.h5' % (pp, band[tt]), 'r')as f:
						p_count = np.array(f['a'])
					with h5py.File(tmp + 'stack_Amask_sum_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						sum_img = np.array(f['a'])

					sub_Num = np.nanmax(p_count)
					tot_N += sub_Num
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

				with h5py.File(
					'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_Amask_%d_in_%s_band.h5' % (tot_N, band[tt]), 'w') as f:
					f['a'] = np.array(stack_img)

				with h5py.File(
					'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_Amask_%d_in_%s_band.h5' % (tot_N, band[tt]), 'r') as f:
					stack_img = np.array(f['a'])

				ss_img = stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
				Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
				flux0 = Intns + Intns_err
				flux1 = Intns - Intns_err
				SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2) + mag_add[tt]
				SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2) + mag_add[tt]
				SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2) + mag_add[tt]
				err0 = SB - SB0
				err1 = SB1 - SB
				id_nan = np.isnan(SB)
				SBt, SB0, SB1 = SB[id_nan == False], SB0[id_nan == False], SB1[id_nan == False] 
				Rt, t_err0, t_err1 = Intns_r[id_nan == False], err0[id_nan == False], err1[id_nan == False]
				idx_nan = np.isnan(SB1)
				t_err1[idx_nan] = 100.

				## sersic part
				Mu_e, R_e, n_e = mu_e[tt], r_e[tt], 4.
				SB_Z05 = sers_pro(Intns_r, Mu_e, R_e, n_e)

				plt.figure()
				plt.title('[A]stack_%d_%s_band' % (tot_N, band[tt]) )
				ax = plt.imshow(stack_img, cmap = 'Greys', vmin = 1e-7, vmax = 5e1, origin = 'lower', norm = mpl.colors.LogNorm())
				plt.colorbar(ax, fraction = 0.035, pad =  0.01, label = '$flux[nmaggy]$')
				hsc.circles(x0, y0, s = Rpp, fc = '', ec = 'r', linestyle = '-', alpha = 0.5)
				hsc.circles(x0, y0, s = 0.2 * Rpp, fc = '', ec = 'g', linestyle = '--', alpha = 0.5)	
				plt.xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
				plt.ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
				plt.savefig(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/A_mask/A_stack_%d_%s_band.png' % (tot_N, band[tt]), dpi = 300)
				plt.close()

				fig = plt.figure()
				ax = plt.subplot(111)
				ax.set_title('[A mask]SB profile %d %s band' % (tot_N, band[tt]) )
				ax.errorbar(Rt, SBt, yerr = [t_err0, t_err1], xerr = None, color = 'r', marker = '.', ls = '', linewidth = 1, markersize = 5, 
					ecolor = 'r', elinewidth = 1, label = 'stack image', alpha = 0.5)
				ax.plot(R_obs, SB_obs, 'b:', label = 'Z05', alpha = 0.5)
				ax.plot(Intns_r, SB_Z05, 'k--', label = 'Sersic Z05', alpha = 0.5)

				ax.set_xlabel('$R[kpc]$')
				ax.set_ylabel('$SB[mag / arcsec^2]$')
				ax.set_xscale('log')
				ax.set_ylim(20, 32)
				ax.set_xlim(1, 1e3)
				ax.legend(loc = 1)
				ax.invert_yaxis()
				ax.grid(which = 'both', axis = 'both')
				ax.tick_params(axis = 'both', which = 'both', direction = 'in')
				plt.savefig(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/A_mask/A_stack_%d_%s_band_SB.png' % (tot_N, band[tt]), dpi = 300)
				plt.close()

			commd.Barrier()

			### stack all of the sub-stack B mask image
			if rank == 1:
				# read the SB of BCG + ICL + residual sky
				SB_tt = pds.read_csv( load + 'Zibetti_SB/%s_band_tot.csv' % band[tt])
				R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
				R_obs = R_obs**4

				tot_N = 0
				mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				for pp in range(cpus):

					with h5py.File(load + 'test_h5/stack_Bmask_pcount_%d_in_%s_band.h5' % (pp, band[tt]), 'r')as f:
						p_count = np.array(f['a'])
					with h5py.File(load + 'test_h5/stack_Bmask_sum_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						sum_img = np.array(f['a'])

					sub_Num = np.nanmax(p_count)
					tot_N += sub_Num
					id_zero = p_count == 0
					ivx = id_zero == False
					mean_img[ivx] = mean_img[ivx] + sum_img[ivx]
					p_add_count[ivx] = p_add_count[ivx] + p_count[ivx]

					# check sub-results
					p_count[id_zero] = np.nan
					sum_img[id_zero] = np.nan
					sub_mean = sum_img / p_count
					where_are_inf = np.isinf(sub_mean)
					sub_mean[where_are_inf] = np.nan

					plt.figure()
					plt.title('[B]stack_%d_%s_band_%d_cpus' % (np.int(sub_Num), band[tt], pp) )
					ax = plt.imshow(sub_mean, cmap = 'Greys', vmin = 1e-7, vmax = 5e1, origin = 'lower', norm = mpl.colors.LogNorm())
					plt.colorbar(ax, fraction = 0.035, pad =  0.01, label = '$flux[nmaggy]$')
					hsc.circles(x0, y0, s = Rpp, fc = '', ec = 'r', linestyle = '-', alpha = 0.5)
					hsc.circles(x0, y0, s = 0.2 * Rpp, fc = '', ec = 'g', linestyle = '--', alpha = 0.5)
					plt.xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
					plt.ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
					plt.savefig(
					'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/B_mask/B_stack_%d_%s_band_%d_cpus.png' % 
					(np.int(sub_Num), band[tt], pp), dpi = 300)
					plt.close()

				## save the stack image
				id_zero = p_add_count == 0
				mean_img[id_zero] = np.nan
				p_add_count[id_zero] = np.nan
				tot_N = np.int(tot_N)
				stack_img = mean_img / p_add_count
				where_are_inf = np.isinf(stack_img)
				stack_img[where_are_inf] = np.nan

				with h5py.File(
					'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_Bmask_%d_in_%s_band.h5' % (tot_N, band[tt]), 'w') as f:
					f['a'] = np.array(stack_img)

				with h5py.File(
					'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_Bmask_%d_in_%s_band.h5' % (tot_N, band[tt]), 'r') as f:
					stack_img = np.array(f['a'])

				ss_img = stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
				Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
				flux0 = Intns + Intns_err
				flux1 = Intns - Intns_err
				SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2) + mag_add[tt]
				SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2) + mag_add[tt]
				SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2) + mag_add[tt]
				err0 = SB - SB0
				err1 = SB1 - SB
				id_nan = np.isnan(SB)
				SBt, SB0, SB1 = SB[id_nan == False], SB0[id_nan == False], SB1[id_nan == False] 
				Rt, t_err0, t_err1 = Intns_r[id_nan == False], err0[id_nan == False], err1[id_nan == False]
				idx_nan = np.isnan(SB1)
				t_err1[idx_nan] = 100.

				## sersic part
				Mu_e, R_e, n_e = mu_e[tt], r_e[tt], 4.
				SB_Z05 = sers_pro(Intns_r, Mu_e, R_e, n_e)

				plt.figure()
				plt.title('[B]stack_%d_%s_band' % (tot_N, band[tt]) )
				ax = plt.imshow(stack_img, cmap = 'Greys', vmin = 1e-7, vmax = 5e1, origin = 'lower', norm = mpl.colors.LogNorm())
				plt.colorbar(ax, fraction = 0.035, pad =  0.01, label = '$flux[nmaggy]$')
				hsc.circles(x0, y0, s = Rpp, fc = '', ec = 'r', linestyle = '-', alpha = 0.5)
				hsc.circles(x0, y0, s = 0.2 * Rpp, fc = '', ec = 'g', linestyle = '--', alpha = 0.5)	
				plt.xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
				plt.ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
				plt.savefig(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/B_mask/B_stack_%d_%s_band.png' % (tot_N, band[tt]), dpi = 300)
				plt.close()

				fig = plt.figure()
				ax = plt.subplot(111)
				ax.set_title('[B mask]SB profile %d %s band' % (tot_N, band[tt]) )
				ax.errorbar(Rt, SBt, yerr = [t_err0, t_err1], xerr = None, color = 'r', marker = '.', ls = '', linewidth = 1, markersize = 5, 
					ecolor = 'r', elinewidth = 1, label = 'stack image', alpha = 0.5)
				ax.plot(R_obs, SB_obs, 'b:', label = 'Z05', alpha = 0.5)
				ax.plot(Intns_r, SB_Z05, 'k--', label = 'Sersic Z05', alpha = 0.5)

				ax.set_xlabel('$R[kpc]$')
				ax.set_ylabel('$SB[mag / arcsec^2]$')
				ax.set_xscale('log')
				ax.set_ylim(20, 32)
				ax.set_xlim(1, 1e3)
				ax.legend(loc = 1)
				ax.invert_yaxis()
				ax.grid(which = 'both', axis = 'both')
				ax.tick_params(axis = 'both', which = 'both', direction = 'in')
				plt.savefig(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/B_mask/B_stack_%d_%s_band_SB.png' % (tot_N, band[tt]), dpi = 300)
				plt.close()

			commd.Barrier()

if __name__ == "__main__":
	main()
