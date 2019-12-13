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
h = H0/100
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

tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'
d_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])
mag_add = np.array([0, 0, 0, -0.04, 0.02])

Rv = 3.1
sfd = SFDQuery()

def sers_pro(r, mu_e, r_e, n):
	belta_n = 2 * n - 0.324
	fn = 1.086 * belta_n * ( (r/r_e)**(1/n) - 1)
	mu_r = mu_e + fn
	return mu_r

def A_stack(band_number, subz, subra, subdec):

	stack_N = len(subz)
	ii = np.int(band_number)
	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	for jj in range(stack_N):

		ra_g = subra[jj]
		dec_g = subdec[jj]
		z_g = subz[jj]
		Da_g = Test_model.angular_diameter_distance(z_g).value

		data_A = fits.getdata(load + 
			'resample/Zibetti/A_mask/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
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
		p_count_A[0, 0] = p_count_A[0, 0] + 1.
		count_array_A[la0: la1, lb0: lb1][idv] = np.nan

	with h5py.File(tmp + 'stack_Amask_sum_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(sum_array_A)
	with h5py.File(tmp + 'stack_Amask_pcount_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(p_count_A)

	return

def main():
	## sersic pro of Zibetti 05
	mu_e = np.array([23.87, 25.22, 23.4])
	r_e = np.array([19.29, 19.40, 20])
	sky_num = np.array([3378, 3377, 3363])

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	#R_cut, bins = 900, 75
	#R_smal, R_max = 1, 10**3.02 # kpc
	R_cut, bins = 1280, 80
	R_smal, R_max = 1, 1.7e3 # kpc
	N_tt = 500

	for tt in range(3):
		with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/%s_band_%d_sample.h5' % (band[tt], N_tt), 'r') as f:
			set_info = np.array(f['a'])
		set_z, set_ra, set_dec = set_info[0,:], set_info[1,:], set_info[2,:]
		zN = len(set_z)

		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		A_stack(tt, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])
		commd.Barrier()

	if rank == 1:
		for tt in range(3):
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

			id_zero = p_add_count == 0
			mean_img[id_zero] = np.nan
			p_add_count[id_zero] = np.nan
			tot_N = np.int(tot_N)
			stack_img = mean_img / p_add_count
			where_are_inf = np.isinf(stack_img)
			stack_img[where_are_inf] = np.nan
			with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_ZIT/stack_Amask_%d_in_%s_band.h5' % (tot_N, band[tt]), 'w') as f:
				f['a'] = np.array(stack_img)
	commd.Barrier()
	'''
	if rank == 1:
		for tt in range(3):
			with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_ZIT/stack_Amask_%d_in_%s_band.h5' % (N_tt, band[tt]), 'r') as f:
				stack_img = np.array(f['a'])

			plt.figure()
			clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5, label = 'cluster region[1Mpc]')
			ax = plt.subplot(111)
			ax.set_title('Z05 2D stacking %d image in %s band' % (N_tt, band[tt]) )
			tf = ax.imshow(stack_img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
			plt.colorbar(tf, ax = ax, fraction = 0.035, pad = 0.01, label = 'flux[nmaggy]')
			ax.add_patch(clust)
			ax.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
			ax.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
			ax.legend(loc = 1)
			plt.subplots_adjust(left = 0.125, right = 0.9, bottom = 0.1, top = 0.9)
			plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_ZIT/Z05_stack_%d_img_%s_band.png' % (N_tt, band[tt]), dpi = 300)
			plt.close()
	commd.Barrier()
	'''
	if rank == 1:
		for kk in range(3):

			SB_tt = pds.read_csv( load + 'Zibetti_SB/%s_band_BCG_ICL.csv' % band[kk])
			R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
			R_obs = R_obs**4
			## sersic part
			Mu_e, R_e, n_e = mu_e[kk], r_e[kk], 4.
			SB_Z05 = sers_pro(R_obs, Mu_e, R_e, n_e)

			## Z05 with mask adjust
			with h5py.File(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_ZIT/stack_Amask_%d_in_%s_band.h5' % (N_tt, band[kk]), 'r') as f:
				stack_img = np.array(f['a'])

			ss_img = stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
			Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
			SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
			SB_r0 = SB + mag_add[kk]
			R_r0 = Intns_r * 1

			## 1.5sigma without adjust
			with h5py.File(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_Amask_%d_in_%s_band.h5' % (N_tt, band[kk]), 'r') as f:
				stack_img0 = np.array(f['a'])
			ss_img0 = stack_img0[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
			Intns, Intns_r, Intns_err, Npix = light_measure(ss_img0, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
			SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
			SB_r1 = SB + mag_add[kk]
			R_r1 = Intns_r * 1

			## 1.5sigma with adjust
			with h5py.File(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_Amask_%d_in_%s_band_2.80rstar_2.80rgalx.h5' % (N_tt, band[kk]), 'r') as f:
				stack_img1 = np.array(f['a'])
			ss_img1 = stack_img1[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
			Intns, Intns_r, Intns_err, Npix = light_measure(ss_img1, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
			SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
			SB_r2 = SB + mag_add[kk]
			R_r2 = Intns_r * 1

			plt.figure()
			ax = plt.subplot(111)
			ax.plot(R_obs, SB_obs, 'k-.', label = 'Z05', alpha = 0.5)
			ax.plot(R_obs, SB_Z05, 'k:', label = 'Sersic Z05', alpha = 0.5)
			ax.plot(R_r0, SB_r0, 'r--', label = 'pipe Z05 with adjust', alpha = 0.5)
			ax.plot(R_r1, SB_r1, 'g--', label = '1.5sigma without adjust', alpha = 0.5)
			ax.plot(R_r2, SB_r2, 'b--', label = '1.5sigma with adjust', alpha = 0.5)

			ax.set_xscale('log')
			ax.set_xlabel('R[kpc]')
			ax.set_ylabel('$ SB[mag/arcsec^2] $')
			ax.set_ylim(20, 34)
			ax.set_xlim(1, 1.5e3)
			ax.legend(loc = 3)
			ax.invert_yaxis()
			ax.grid(which = 'both', axis = 'both')
			ax.tick_params(axis = 'both', which = 'both', direction = 'in')
			plt.savefig(
			'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_ZIT/Z05_15sigm_SB_%d_img_%s_band.png' % (N_tt, band[kk]), dpi = 300)
			plt.close()
	commd.Barrier()
	'''
	cen_pos = 1280
	R_bl = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])

	if rank == 1:
		for kk in range(3):

			SB_tt = pds.read_csv( load + 'Zibetti_SB/%s_band_BCG_ICL.csv' % band[kk])
			R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
			R_obs = R_obs**4
			## sersic part
			Mu_e, R_e, n_e = mu_e[kk], r_e[kk], 4.
			SB_Z05 = sers_pro(R_obs, Mu_e, R_e, n_e)

			with h5py.File(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_ZIT/stack_Amask_%d_in_%s_band.h5' % (N_tt, band[kk]), 'r') as f:
				stack_img = np.array(f['a'])

			ss_img = stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
			Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
			SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
			SB_r0 = SB + mag_add[kk]
			R_r0 = Intns_r * 1

			BL_img = stack_img[y0 - cen_pos: y0 + cen_pos, x0 - cen_pos: x0 + cen_pos]
			grd_x = np.linspace(0, BL_img.shape[1] - 1, BL_img.shape[1])
			grd_y = np.linspace(0, BL_img.shape[0] - 1, BL_img.shape[0])
			grd = np.array( np.meshgrid(grd_x, grd_y) )

			plt.figure()
			ax = plt.subplot(111)
			ax.set_title('Z05 %s band SB as function of RBL region [%d img] ' % (band[kk], N_tt) )
			ax.plot(R_r0, SB_r0, 'k-', label = '$ stack \; img $', alpha = 0.5)
			ax.plot(R_obs, SB_obs, 'k-.', label = 'Z05', alpha = 0.5)
			ax.plot(R_obs, SB_Z05, 'k:', label = 'Sersic Z05', alpha = 0.5)

			for qq in range(len(R_bl) - 1):
				ddr = np.sqrt( (grd[0,:] - cen_pos)**2 + (grd[1,:] - cen_pos)**2 )
				idu = (ddr > R_bl[qq] * Rpp) & (ddr < R_bl[qq + 1] * Rpp)
				Resi_bl = np.nanmean( BL_img[idu] )
				dd_SB = 22.5 - 2.5 * np.log10(Intns - Resi_bl) + 2.5 * np.log10(pixel**2) + mag_add[kk]
				ax.plot(R_r0, dd_SB, linestyle = '--', color = mpl.cm.rainbow(qq / (len(R_bl) - 1) ), 
					label = '$ subtracted \, RBL \, in \, %.2f \sim %.2fMpc $' % (R_bl[qq], R_bl[qq + 1]), alpha = 0.5)

			ax.set_xlabel('$R[kpc]$')
			ax.set_ylabel('$SB[mag / arcsec^2]$')
			ax.set_xscale('log')
			ax.set_ylim(20, 34)
			ax.set_xlim(1, 1.5e3)
			ax.legend(loc = 3, fontsize = 7.5)
			ax.invert_yaxis()
			ax.grid(which = 'both', axis = 'both')
			ax.tick_params(axis = 'both', which = 'both', direction = 'in')
			plt.savefig(
			'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_ZIT/Z05_SB_%d_img_%s_band_BL_sub.png' % (N_tt, band[kk]), dpi = 300)
			plt.close()
	commd.Barrier()
	'''
if __name__ == "__main__":
	main()
