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

band = ['r', 'g', 'i', 'u', 'z']
bnd_indx = [2, 1, 3, 0, 4]
mag_add = np.array([0, 0, 0, -0.04, 0.02])
cat_Rii = np.array([0.23,  0.68,  1.03,   1.76,   3.00, 
					4.63,  7.43,  11.42,  18.20,  28.20, 
					44.21, 69.00, 107.81, 168.20, 263.00])
def sers_pro(r, mu_e, r_e, n):
	belta_n = 2 * n - 0.324
	fn = 1.086 * belta_n * ( (r/r_e)**(1/n) - 1)
	mu_r = mu_e + fn
	return mu_r

def stack_process(band_number, subz, subra, subdec, N_tt):

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

	for jj in range(stack_N):

		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]
		Da_g = Test_model.angular_diameter_distance(z_g).value

		## stack A mask
		# 1.5sigma
		data_A = fits.getdata(load + 
			'resample/1_5sigma/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
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

	with h5py.File(load + 'test_h5/stack_Amask_sum_%d_in_%s_band_%d_imgs.h5' % (rank, band[ii], N_tt), 'w') as f:
		f['a'] = np.array(sum_array_A)
	with h5py.File(load + 'test_h5/stack_Amask_pcount_%d_in_%s_band_%d_imgs.h5' % (rank, band[ii], N_tt), 'w') as f:
		f['a'] = np.array(p_count_A)

	return

def main():
	## sersic pro of Zibetti 05
	mu_e = np.array([23.87, 25.22, 23.4])
	r_e = np.array([19.29, 19.40, 20])

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	R_cut, bins = 900, 75
	R_smal, R_max = 1, 10**3.02 # kpc
	#R_cut, bins = 1280, 80
	#R_smal, R_max = 1, 1.7e3 # kpc

	#N_tt = np.array([50, 100, 150, 200, 250, 500, 1000, 1500, 2000, 2500, 3000])
	N_tt = np.array([200])

	for tt in range(3):
		with h5py.File(load + 'mpi_h5/%s_band_sample_catalog.h5' % band[tt], 'r') as f:
			sub_array = np.array(f['a'])
		ra, dec, z, rich, r_mag = sub_array[0,:], sub_array[1,:], sub_array[2,:], sub_array[3,:], sub_array[4,:]
		zN = len(z)

		for aa in range(len(N_tt)):
			np.random.seed(1)
			tt0 = np.random.choice(zN, size = N_tt[aa], replace = False)
			set_z, set_ra, set_dec = z[tt0], ra[tt0], dec[tt0]
			set_Dl = Test_model.luminosity_distance(set_z).value
			set_Mag = r_mag[tt0] + 5 - 5 * np.log10(set_Dl * 1e6)

			## save the Mag info. in r band
			keys = ['z', 'ra', 'dec', 'r_Mag']
			values = [set_z, set_ra, set_dec, set_Mag]
			fill = dict(zip(keys, values))
			data = pds.DataFrame(fill)
			data.to_csv(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/%s_band_%d_sample_info.csv' % (band[tt], N_tt[aa]) )

			m, n = divmod(N_tt[aa], cpus)
			N_sub0, N_sub1 = m * rank, (rank + 1) * m
			if rank == cpus - 1:
				N_sub1 += n
			stack_process(tt, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1], N_tt[aa])
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

					with h5py.File(load + 'test_h5/stack_Amask_pcount_%d_in_%s_band_%d_imgs.h5' % (pp, band[tt], N_tt[aa]), 'r')as f:
						p_count = np.array(f['a'])
					with h5py.File(load + 'test_h5/stack_Amask_sum_%d_in_%s_band_%d_imgs.h5' % (pp, band[tt], N_tt[aa]), 'r') as f:
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

				fig = plt.figure()
				ax = plt.subplot(111)
				ax.set_title('SB profile %d %s band' % (tot_N, band[tt]) )
				ax.errorbar(Rt, SBt, yerr = [t_err0, t_err1], xerr = None, color = 'r', marker = '.', ls = '', linewidth = 1, markersize = 5, 
					ecolor = 'r', elinewidth = 1, label = 'stack image', alpha = 0.5)
				ax.plot(R_obs, SB_obs, 'b:', label = 'Z05', alpha = 0.5)
				ax.plot(Intns_r, SB_Z05, 'k--', label = 'Sersic Z05', alpha = 0.5)

				ax.set_xlabel('$R[kpc]$')
				ax.set_ylabel('$SB[mag / arcsec^2]$')
				ax.set_xscale('log')
				ax.set_ylim(20, 33)
				ax.set_xlim(1, 1e3)
				ax.legend(loc = 1)
				ax.text(2, 27, s = '$ \overline{M}_{r} = %.3f$' % np.nanmean(set_Mag), color = 'r')
				ax.invert_yaxis()
				ax.grid(which = 'both', axis = 'both')
				ax.tick_params(axis = 'both', which = 'both', direction = 'in')
				plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/A_mask/tot_pros_%d_%s_band_SB.png' % (tot_N, band[tt]), dpi = 300)
				plt.close()

			commd.Barrier()
			
	'''
	## 2D image
	if rank == 1:
		for kk in range(3):
			for jj in range( len(N_tt) ):
				with h5py.File(
					'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_Amask_%d_in_%s_band.h5' % (N_tt[jj], band[kk]), 'r') as f:
					stack_img = np.array(f['a'])

				## measure the profile in flux

				plt.figure()
				clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5, label = 'cluster region[1Mpc]')
				ax = plt.subplot(111)
				ax.set_title('2D stacking %d image in %s band' % (N_tt[jj], band[kk]) )
				tf = ax.imshow(stack_img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
				plt.colorbar(tf, ax = ax, fraction = 0.035, pad = 0.01, label = 'flux[nmaggy]')
				ax.add_patch(clust)
				ax.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
				ax.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
				ax.legend(loc = 1)
				plt.savefig(
					'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/A_mask/stack_%d_img_%s_band.png' % (N_tt[jj], band[kk]), dpi = 300)
				plt.close()

	commd.Barrier()
	'''
	r_a0, r_a1 = 1.0, 1.1
	'''
	if rank == 2:
		for kk in range(3):

			plt.figure()
			gs = gridspec.GridSpec(2,1, height_ratios = [4,1])
			ax = plt.subplot(gs[0])
			bx = plt.subplot(gs[1])

			ax.set_title(
				'$ %s \, band \, SB \, profile \, [SB \, in \, %.2f \sim %.2f Mpc \, as \, RBL] $' % (band[kk], r_a0, r_a1) )

			for jj in range( len(N_tt) ):
				with h5py.File(
					'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_Amask_%d_in_%s_band.h5' % (N_tt[jj], band[kk]), 'r') as f:
					stack_img = np.array(f['a'])

				cen_pos = 1280
				BL_img = stack_img[y0 - cen_pos: y0 + cen_pos, x0 - cen_pos: x0 + cen_pos] # 1280 pixel, for z = 0.25, larger than 2Mpc
				grd_x = np.linspace(0, BL_img.shape[1] - 1, BL_img.shape[1])
				grd_y = np.linspace(0, BL_img.shape[0] - 1, BL_img.shape[0])
				grd = np.array( np.meshgrid(grd_x, grd_y) )
				ddr = np.sqrt( (grd[0,:] - cen_pos)**2 + (grd[1,:] - cen_pos)**2 )
				idu = (ddr > r_a0 * Rpp) & (ddr < r_a1 * Rpp)
				Resi_bl = np.nanmean( BL_img[idu] )

				ss_img = stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
				Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
				devi_pro = (Intns - Resi_bl) / Intns

				## read the sample info.
				set_info = pds.read_csv(
					'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/%s_band_%d_sample_info.csv' % (band[kk], N_tt[jj]) )
				set_Mag = set_info['r_Mag']

				ax.plot(Intns_r, Intns / pixel**2, linestyle = '-', color = mpl.cm.rainbow(jj / len(N_tt)), 
					label = '$ stack \, %d \, imgs[\overline{M}_{r} = %.3f] $' % (N_tt[jj], np.nanmean(set_Mag)), alpha = 0.5 )
				ax.axhline(y = Resi_bl / pixel**2, linestyle = '--', color = mpl.cm.rainbow(jj / len(N_tt)), alpha = 0.5 )

				bx.plot(Intns_r, devi_pro, linestyle = '-', color = mpl.cm.rainbow(jj / len(N_tt)), alpha = 0.5 )

			ax.set_xlabel('$R[kpc]$')
			ax.set_ylabel('$SB[nmaggy / pixel^{2}]$')
			ax.set_xscale('log')
			ax.set_yscale('log')
			ax.set_xlim(1, 1.5e3)
			ax.legend(loc = 1, fontsize = 7.5)
			ax.grid(which = 'both', axis = 'both')
			ax.tick_params(axis = 'both', which = 'both', direction = 'in')

			bx.set_xlabel('$R[kpc]$')
			bx.set_ylabel('SB(r) - RBL / SB(r)')
			bx.set_xscale('log')
			bx.set_xlim(ax.get_xlim())
			bx.grid(which = 'both', axis = 'both')
			bx.tick_params(axis = 'both', which = 'both', direction = 'in')
			ax.set_xticks([])

			plt.subplots_adjust(hspace = 0.)
			plt.savefig(
			'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/A_mask/SB_pro_%s_band.png' % band[kk], dpi = 300)
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

			plt.figure()
			ax = plt.subplot(111)
			ax.set_title('$ %s \, band \, SB \, [subtract \, SB \, in \, %.2f \sim %.2f Mpc] $' % (band[kk], r_a0, r_a1) )

			for jj in range( len(N_tt) ):
				with h5py.File(
					'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_Amask_%d_in_%s_band.h5' % (N_tt[jj], band[kk]), 'r') as f:
					stack_img = np.array(f['a'])

				cen_pos = 1280
				BL_img = stack_img[y0 - cen_pos: y0 + cen_pos, x0 - cen_pos: x0 + cen_pos] # 1280 pixel, for z = 0.25, larger than 2Mpc
				grd_x = np.linspace(0, BL_img.shape[1] - 1, BL_img.shape[1])
				grd_y = np.linspace(0, BL_img.shape[0] - 1, BL_img.shape[0])
				grd = np.array( np.meshgrid(grd_x, grd_y) )
				ddr = np.sqrt( (grd[0,:] - cen_pos)**2 + (grd[1,:] - cen_pos)**2 )
				idu = (ddr > r_a0 * Rpp) & (ddr < r_a1 * Rpp)
				Resi_bl = np.nanmean( BL_img[idu] )

				ss_img = stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]

				Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
				flux0 = Intns + Intns_err
				flux1 = Intns - Intns_err
				SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2) + mag_add[kk]
				SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2) + mag_add[kk]
				SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2) + mag_add[kk]
				err0 = SB - SB0
				err1 = SB1 - SB
				id_nan = np.isnan(SB)
				SBt, SB0, SB1 = SB[id_nan == False], SB0[id_nan == False], SB1[id_nan == False] 
				Rt, t_err0, t_err1 = Intns_r[id_nan == False], err0[id_nan == False], err1[id_nan == False]
				idx_nan = np.isnan(SB1)
				t_err1[idx_nan] = 100.
				## read the sample info.
				sub_SB = 22.5 - 2.5 * np.log10(Intns - Resi_bl) + 2.5 * np.log10(pixel**2) + mag_add[kk]

				set_info = pds.read_csv('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/%s_band_%d_sample_info.csv' % (band[kk], N_tt[jj]) )
				set_Mag = set_info['r_Mag']

				ax.plot(Rt, SBt, linestyle = '-', color = mpl.cm.rainbow(jj / len(N_tt)), 
					label = '$ stack \, %d \, imgs[\overline{M}_{r} = %.3f] $' % (N_tt[jj], np.nanmean(set_Mag)), alpha = 0.5)
				ax.plot(Intns_r, sub_SB, linestyle = '--', color = mpl.cm.rainbow(jj / len(N_tt)), 
					label = 'RBL subtracted', alpha = 0.5)

			ax.plot(R_obs, SB_obs, 'k-.', label = 'Z05', alpha = 0.5)
			ax.plot(R_obs, SB_Z05, 'k:', label = 'Sersic Z05', alpha = 0.5)
			ax.set_xlabel('$R[kpc]$')
			ax.set_ylabel('$SB[mag / arcsec^2]$')
			ax.set_xscale('log')
			ax.set_ylim(20, 33)
			ax.set_xlim(1, 1e3)
			ax.legend(loc = 3, fontsize = 7.5)
			ax.invert_yaxis()
			ax.grid(which = 'both', axis = 'both')
			ax.tick_params(axis = 'both', which = 'both', direction = 'in')
			plt.savefig(
			'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/A_mask/sub_%.2f-%.2f_bl_SB_pro_%s_band.png' %(r_a0, r_a1, band[kk]), dpi = 300)
			plt.close()
	commd.Barrier()

if __name__ == "__main__":
	main()
