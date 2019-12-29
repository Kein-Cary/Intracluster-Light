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

home = '/mnt/ddnfs/data_users/cxkttwl/ICL/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'
band = ['r', 'g', 'i', 'u', 'z']
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
		'''
		## stack A mask
		# 1.5sigma
		data_A = fits.getdata(load + 
			'resample/1_5sigma/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
		'''
		## sky-select sample
		data_A = fits.getdata(load + 
			'sky_select_img/imgs/cut_edge-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)

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

	with h5py.File(tmp + 'stack_Amask_sum_%d_in_%s_band_%d_imgs.h5' % (rank, band[ii], N_tt), 'w') as f:
		f['a'] = np.array(sum_array_A)
	with h5py.File(tmp + 'stack_Amask_pcount_%d_in_%s_band_%d_imgs.h5' % (rank, band[ii], N_tt), 'w') as f:
		f['a'] = np.array(p_count_A)

	return

def main():
	## sersic pro of Zibetti 05
	mu_e = np.array([23.87, 25.22, 23.4])
	r_e = np.array([19.29, 19.40, 20])

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	#R_cut, bins = 900, 75
	#R_smal, R_max = 1, 10**3.02 # kpc
	R_cut, bins = 1280, 80
	R_smal, R_max = 1, 1.7e3 # kpc

	#N_tt = np.array([50, 100, 150, 200, 250, 500, 1000, 1500, 2000, 2500, 3000])

	N_dd = np.array([2013, 2008, 2002, 2008, 2009]) ## sky-select sample

	for tt in range(3):
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
		with h5py.File(load + 'sky_select_img/test_set/%s_band_sky_0.8Mpc_select.h5' % ( band[tt] ), 'r') as f:
			sub_array = np.array(f['a'])
		ra, dec, z, r_mag = sub_array[0,:], sub_array[1,:], sub_array[2,:], sub_array[3,:]

		zN = len(z)
		N_tt = np.array([zN])
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

			#data.to_csv(load + 'sky/cluster/%s_band_%d_sample_info.csv' % (band[tt], N_tt[aa]) )
			#data.to_csv(load + 'sky_select_img/result/%s_band_%d_sample_info.csv' % (band[tt], N_tt[aa]) )
			data.to_csv(load + 'sky_select_img/test_set/%s_band_%d_sample_info.csv' % (band[tt], N_tt[aa]) )

			m, n = divmod(N_tt[aa], cpus)
			N_sub0, N_sub1 = m * rank, (rank + 1) * m
			if rank == cpus - 1:
				N_sub1 += n
			stack_process(tt, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1], N_tt[aa])
			commd.Barrier()

			### stack all of the sub-stack A mask image
			if rank == 0:

				tot_N = 0
				mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				for pp in range(cpus):

					with h5py.File(tmp + 'stack_Amask_pcount_%d_in_%s_band_%d_imgs.h5' % (pp, band[tt], N_tt[aa]), 'r')as f:
						p_count = np.array(f['a'])
					with h5py.File(tmp + 'stack_Amask_sum_%d_in_%s_band_%d_imgs.h5' % (pp, band[tt], N_tt[aa]), 'r') as f:
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

				#with h5py.File(load + 'sky/cluster/select_Amask_%d_in_%s_band.h5' % (tot_N, band[tt]), 'w') as f:
				#with h5py.File(load + 'sky_select_img/result/stack_cut_A_%d_in_%s_band.h5' % (tot_N, band[tt]), 'w') as f:
				with h5py.File(load + 'sky_select_img/test_set/stack_cut_A_%d_in_%s_band.h5' % (tot_N, band[tt]), 'w') as f:
					f['a'] = np.array(stack_img)

			commd.Barrier()

	#N_sum = np.array([3308, 3309, 3295, 3308, 3305])
	#N_sum = np.array([2013, 2008, 2002, 2008, 2009]) ## sky-select sample
	N_sum = np.array([1291, 1286, 1283, 1294, 1287])

	## 2D image and Sb profile
	r_a0, r_a1 = 1.0, 1.1
	if rank == 1:
		for kk in range(3):

			SB_tt = pds.read_csv( load + 'Zibetti_SB/%s_band_BCG_ICL.csv' % band[kk])
			R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
			R_obs = R_obs**4
			## sersic part
			Mu_e, R_e, n_e = mu_e[kk], r_e[kk], 4.
			SB_Z05 = sers_pro(R_obs, Mu_e, R_e, n_e)

			#with h5py.File(load + 'sky/cluster/select_Amask_%d_in_%s_band.h5' % (N_sum[kk], band[kk]), 'r') as f:
			#with h5py.File(load + 'sky_select_img/result/stack_cut_A_%d_in_%s_band.h5' % (N_sum[kk], band[kk]), 'r') as f:
			with h5py.File(load + 'sky_select_img/test_set/stack_cut_A_%d_in_%s_band.h5' % (N_sum[kk], band[kk]), 'r') as f:
				stack_img = np.array(f['a'])
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

			#set_info = pds.read_csv(load + 'sky/cluster/%s_band_%d_sample_info.csv' % (band[kk], N_sum[kk]) )
			#set_info = pds.read_csv(load + 'sky_select_img/result/%s_band_%d_sample_info.csv' % (band[kk], N_sum[kk]) )
			set_info = pds.read_csv(load + 'sky_select_img/test_set/%s_band_%d_sample_info.csv' % (band[kk], N_sum[kk]) )
			set_Mag = set_info['r_Mag']

			#### add-back the over sky component (BCG case - shuffle case)
			#with h5py.File(load + 'sky/center_set/sky_minus_media_%d_imgs_%s_band.h5' % (N_sum[kk], band[kk]), 'r') as f:
			#with h5py.File(load + 'sky_select_img/result/sky_minus_media_%d_imgs_%s_band.h5' % (N_sum[kk], band[kk]), 'r') as f:
			with h5py.File(load + 'sky_select_img/test_set/sky_minus_media_%d_imgs_%s_band.h5' % (N_sum[kk], band[kk]), 'r') as f:
				BCG_add = np.array(f['a'])
			#############
			#with h5py.File(load + 'sky/center_set/mean_sky_shuffle_media_%d_imgs_%s_band.h5' % (N_sum[kk], band[kk]), 'r') as f:
			#with h5py.File(load + 'sky_select_img/result/mean_sky_shuffle_media_%d_imgs_%s_band.h5' % (N_sum[kk], band[kk]), 'r') as f:
			with h5py.File(load + 'sky_select_img/test_set/mean_sky_shuffle_media_%d_imgs_%s_band.h5' % (N_sum[kk], band[kk]), 'r') as f:
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

			## read the sample info.
			sub_SB = 22.5 - 2.5 * np.log10(Intns - Resi_bl) + 2.5 * np.log10(pixel**2) + mag_add[kk]

			plt.figure()
			ax = plt.subplot(111)
			ax.set_title('$ %s \, band \, SB \, %d imgs\,[subtract \, SB \, in \, %.2f \sim %.2f Mpc] $' % (band[kk], N_sum[kk], r_a0, r_a1) )

			ax.plot(Rt, SBt, linestyle = '-', color = 'r', label = '$ stack \, imgs[\overline{M}_{r} = %.3f] $' % np.nanmean(set_Mag), alpha = 0.5)
			ax.plot(Intns_r, sub_SB, linestyle = '--', color = 'b', label = 'RBL subtracted', alpha = 0.5)
			ax.plot(R_add, SB_add, linestyle = '-.', color = 'g', label = 'Add over-subtracted light', alpha = 0.5)

			ax.plot(R_obs, SB_obs, 'k-.', label = 'Z05', alpha = 0.5)
			ax.plot(R_obs, SB_Z05, 'k:', label = 'Sersic Z05', alpha = 0.5)
			ax.set_xlabel('$R[kpc]$')
			ax.set_ylabel('$SB[mag / arcsec^2]$')
			ax.set_xscale('log')
			ax.set_ylim(20, 33)
			ax.set_xlim(1, 1.5e3)
			ax.legend(loc = 3, fontsize = 7.5)
			ax.invert_yaxis()
			ax.grid(which = 'both', axis = 'both')
			ax.tick_params(axis = 'both', which = 'both', direction = 'in')

			#plt.savefig(load + 'sky/center_set/SB_%d_add_sub_median_pros_%s_band.png' %(N_sum[kk], band[kk]), dpi = 300)
			#plt.savefig(load + 'sky_select_img/result/cutA_%d_imgs_SB_pro_%s_band_add_sub_media.png' %(N_sum[kk], band[kk]), dpi = 300)
			plt.savefig(load + 'sky_select_img/test_set/cutA_%d_imgs_SB_pro_%s_band_add_sub_media.png' %(N_sum[kk], band[kk]), dpi = 300)
			plt.close()

			plt.figure(figsize = (18, 6))
			ax0 = plt.subplot(131)
			ax1 = plt.subplot(132)
			ax2 = plt.subplot(133)

			ax0.set_title('stack %d imgs in %s band' % (N_sum[kk], band[kk]),)
			tf = ax0.imshow(stack_img, origin = 'lower', cmap = 'Greys', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
			clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5, label = 'cluster region[1Mpc]')
			ax0.add_patch(clust)
			ax0.set_xlim(x0 - R_cut, x0 + R_cut)
			ax0.set_ylim(y0 - R_cut, y0 + R_cut)
			plt.colorbar(tf, ax = ax0, fraction = 0.045, pad = 0.01, label = 'flux[nmaggies]')

			ax1.set_title('stacking img + Difference img',)
			tf = ax1.imshow(stack_img + resi_add, origin = 'lower', cmap = 'Greys', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
			clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5, label = 'cluster region[1Mpc]')
			ax1.add_patch(clust)
			ax1.set_xlim(x0 - R_cut, x0 + R_cut)
			ax1.set_ylim(y0 - R_cut, y0 + R_cut)
			plt.colorbar(tf, ax = ax1, fraction = 0.045, pad = 0.01, label = 'flux[nmaggies]')

			ax2.set_title('stacking img + Difference img - RBL',)
			tf = ax2.imshow(stack_img + resi_add - Resi_bl, origin = 'lower', cmap = 'Greys', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
			clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5, label = 'cluster region[1Mpc]')
			ax2.add_patch(clust)
			ax2.set_xlim(x0 - R_cut, x0 + R_cut)
			ax2.set_ylim(y0 - R_cut, y0 + R_cut)
			plt.colorbar(tf, ax = ax2, fraction = 0.045, pad = 0.01, label = 'flux[nmaggies]')

			plt.tight_layout()
			plt.subplots_adjust(bottom = 0.1, right = 0.9, top = 0.9)

			#plt.savefig(load + 'sky/center_set/Amask_stack_%d_imgs_%s_band.png' % (N_sum[kk], band[kk]), dpi = 300)
			#plt.savefig(load + 'sky_select_img/result/cutA_stack_%d_imgs_%s_band.png' % (N_sum[kk], band[kk]), dpi = 300)
			plt.savefig(load + 'sky_select_img/test_set/cutA_stack_%d_imgs_%s_band.png' % (N_sum[kk], band[kk]), dpi = 300)
			plt.close()

	commd.Barrier()
	raise
	### different selection sample compare
	cen_pos = 1280
	if rank == 0:
		for kk in range(3):

			SB_tt = pds.read_csv( load + 'Zibetti_SB/%s_band_BCG_ICL.csv' % band[kk])
			R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
			R_obs = R_obs**4
			## sersic part
			Mu_e, R_e, n_e = mu_e[kk], r_e[kk], 4.
			SB_Z05 = sers_pro(R_obs, Mu_e, R_e, n_e)

			for jj in range( len(N_tt) ):
				## sky selected sample
				with h5py.File(load + 'sky/cluster/select_Amask_%d_in_%s_band.h5' % (N_tt[jj], band[kk]), 'r') as f:
				#with h5py.File(load + 'sky_select_img/result/stack_cut_A_%d_in_%s_band.h5' % (N_tt[jj], band[kk]), 'r') as f:
					stack_img = np.array(f['a'])

				ss_img = stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
				Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
				SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
				SB_sky_select = SB + mag_add[kk]
				R_sky = Intns_r * 1
				# sub-BL
				grd_x = np.linspace(0, ss_img.shape[1] - 1, ss_img.shape[1])
				grd_y = np.linspace(0, ss_img.shape[0] - 1, ss_img.shape[0])
				grd = np.array( np.meshgrid(grd_x, grd_y) )
				ddr = np.sqrt( (grd[0,:] - cen_pos)**2 + (grd[1,:] - cen_pos)**2 )
				idu = (ddr > r_a0 * Rpp) & (ddr < r_a1 * Rpp)
				Resi_bl_0 = np.nanmean( ss_img[idu] )
				corr_SB_sky = 22.5 - 2.5 * np.log10(Intns - Resi_bl_0) + 2.5 * np.log10(pixel**2) + mag_add[kk]

				set_info = pds.read_csv(load + 'sky/cluster/%s_band_%d_sample_info.csv' % (band[kk], N_tt[jj]) )
				#set_info_0 = pds.read_csv(load + 'sky_select_img/result/%s_band_%d_sample_info.csv' % (band[kk], N_tt[jj]) )
				set_Mag_sky = set_info_0['r_Mag']

				## Z05 pipeline
				with h5py.File(home + 'fig_ZIT/stack_Amask_%d_in_%s_band.h5' % (N_tt[jj], band[kk]), 'r') as f:
					stack_img = np.array(f['a'])

				ss_img = stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
				Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
				SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
				SB_Z05_pipe = SB + mag_add[kk]
				R_Z05_pipe = Intns_r * 1

				grd_x = np.linspace(0, ss_img.shape[1] - 1, ss_img.shape[1])
				grd_y = np.linspace(0, ss_img.shape[0] - 1, ss_img.shape[0])
				grd = np.array( np.meshgrid(grd_x, grd_y) )
				ddr = np.sqrt( (grd[0,:] - cen_pos)**2 + (grd[1,:] - cen_pos)**2 )
				idu = (ddr > r_a0 * Rpp) & (ddr < r_a1 * Rpp)
				Resi_bl_1 = np.nanmean( ss_img[idu] )
				corr_SB_Z05 = 22.5 - 2.5 * np.log10(Intns - Resi_bl_1) + 2.5 * np.log10(pixel**2) + mag_add[kk]

				## adjust case
				with h5py.File(home + 
					'fig_15sigma/size_adjust/stack_Amask_%d_in_%s_band_2.80rstar_2.80rgalx.h5' % (N_tt[jj], band[kk]), 'r') as f:
					stack_img = np.array(f['a'])

				ss_img = stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
				Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
				SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
				SB_adjust = SB + mag_add[kk]
				R_adjust = Intns_r * 1

				grd_x = np.linspace(0, ss_img.shape[1] - 1, ss_img.shape[1])
				grd_y = np.linspace(0, ss_img.shape[0] - 1, ss_img.shape[0])
				grd = np.array( np.meshgrid(grd_x, grd_y) )
				ddr = np.sqrt( (grd[0,:] - cen_pos)**2 + (grd[1,:] - cen_pos)**2 )
				idu = (ddr > r_a0 * Rpp) & (ddr < r_a1 * Rpp)
				Resi_bl_2 = np.nanmean( ss_img[idu] )
				corr_SB_adjust = 22.5 - 2.5 * np.log10(Intns - Resi_bl_2) + 2.5 * np.log10(pixel**2) + mag_add[kk]

				set_info_1 = pds.read_csv(home + 'fig_15sigma/stack_img/%s_band_%d_sample_info.csv' % (band[kk], N_tt[jj]) )
				set_Mag_com = set_info_1['r_Mag']

				## no-adjust case
				with h5py.File(home + 'fig_15sigma/stack_img/stack_Amask_%d_in_%s_band.h5' % (N_tt[jj], band[kk]), 'r') as f:
					stack_img = np.array(f['a'])

				ss_img = stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
				Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
				SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
				SB_no_adjust = SB + mag_add[kk]
				R_no_adjust = Intns_r * 1

				grd_x = np.linspace(0, ss_img.shape[1] - 1, ss_img.shape[1])
				grd_y = np.linspace(0, ss_img.shape[0] - 1, ss_img.shape[0])
				grd = np.array( np.meshgrid(grd_x, grd_y) )
				ddr = np.sqrt( (grd[0,:] - cen_pos)**2 + (grd[1,:] - cen_pos)**2 )
				idu = (ddr > r_a0 * Rpp) & (ddr < r_a1 * Rpp)
				Resi_bl_3 = np.nanmean( ss_img[idu] )
				corr_SB_no_adjust = 22.5 - 2.5 * np.log10(Intns - Resi_bl_3) + 2.5 * np.log10(pixel**2) + mag_add[kk]

				plt.figure()
				ax = plt.subplot(111)
				ax.set_title('SB comparison in %s band [%d imgs, RBL: %.2f--%.2f Mpc ]' % (band[kk], N_tt[jj], r_a0, r_a1) )
				ax.plot(R_obs, SB_obs, 'k-.', label = 'Z05', alpha = 0.5)
				ax.plot(R_obs, SB_Z05, 'k:', label = 'Sersic Z05', alpha = 0.5)

				ax.plot(R_sky, SB_sky_select, 'r-', label = '$ sky \; selected \, [M_{r} = %.3f] $' % np.nanmean(set_Mag_sky), alpha = 0.5)
				ax.plot(R_sky, corr_SB_sky, 'r--', alpha = 0.5)

				ax.plot(R_Z05_pipe, SB_Z05_pipe, 'm-', label = '$ Z05 \; pipe \, [M_{r} = %.3f] $' % np.nanmean(set_Mag_com), alpha = 0.5)
				ax.plot(R_Z05_pipe, corr_SB_Z05, 'm--', alpha = 0.5)

				ax.plot(R_no_adjust, SB_no_adjust, 'b-', label = '$ No \; mask \; adjust \, [M_{r} = %.3f]$' % np.nanmean(set_Mag_com), alpha = 0.5)
				ax.plot(R_no_adjust, corr_SB_no_adjust, 'b--', alpha = 0.5)

				ax.plot(R_adjust, SB_adjust, 'g-', label = '$ mask \; adjust \, [M_{r} = %.3f] $' % np.nanmean(set_Mag_com), alpha = 0.5)
				ax.plot(R_adjust, corr_SB_adjust, 'g--', alpha = 0.5)

				ax.set_xscale('log')
				ax.set_xlabel('R[kpc]')
				ax.set_ylabel('$ SB[mag/arcsec^2] $')
				ax.set_ylim(20, 34)
				ax.set_xlim(1, 1.5e3)
				ax.legend(loc = 3)
				ax.invert_yaxis()
				ax.grid(which = 'both', axis = 'both')
				ax.tick_params(axis = 'both', which = 'both', direction = 'in')

				plt.savefig(load + 'sky_select_img/result/SB_compare_%s_band_%d_imgs.png' % (band[kk], N_tt[jj]), dpi = 300)
				plt.close()
	commd.Barrier()

if __name__ == "__main__":
	main()
