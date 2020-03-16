import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import h5py
import pandas as pds
import numpy as np
import astropy.io.fits as fits
import astropy.units as U
from astropy import cosmology as apcy
from matplotlib.patches import Circle, Ellipse
from light_measure import light_measure
from Mass_rich_radius import rich2R

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
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

band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])
mag_add = np.array([0, 0, 0, -0.04, 0.02])

dfile = '/mnt/ddnfs/data_users/cxkttwl/ICL/photo_data/' ## save the catalogue data
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'  ## save the process data
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'

def sers_pro(r, mu_e, r_e, n):
	belta_n = 2 * n - 0.324
	fn = 1.086 * belta_n * ( (r/r_e)**(1/n) - 1)
	mu_r = mu_e + fn
	return mu_r

def img_stack(band_id, sub_z, sub_ra, sub_dec):

	stack_N = len(sub_z)
	ii = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	## initial image which is about twice of the biggest size of images, use to stack images
	sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	f2_sum = np.zeros((len(Ny), len(Nx)), dtype = np.float)  ## also save flux^2 img for Variance calculation
	id_nm = 0.

	for jj in range(stack_N):

		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]
		Da_g = Test_model.angular_diameter_distance(z_g).value
		try:
			'''
			## resample imgs
			data_A = fits.getdata(load + 
				'photo_z/resample/pho_z-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
			'''
			## resample + edge rule out imgs
			data_A = fits.getdata(load + 
				'photo_z/resamp_cut/cut_edge-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)

			img_A = data_A[0]
			xn = data_A[1]['CENTER_X']
			yn = data_A[1]['CENTER_Y']

			la0 = np.int(y0 - yn)
			la1 = np.int(y0 - yn + img_A.shape[0])
			lb0 = np.int(x0 - xn)
			lb1 = np.int(x0 - xn + img_A.shape[1])

			idx = np.isnan(img_A)
			idv = np.where(idx == False)

			sum_array_A[la0: la1, lb0: lb1][idv] = sum_array_A[la0: la1, lb0: lb1][idv] + img_A[idv]
			f2_sum[la0: la1, lb0: lb1][idv] = f2_sum[la0: la1, lb0: lb1][idv] + img_A[idv]**2
			count_array_A[la0: la1, lb0: lb1][idv] = img_A[idv]
			id_nan = np.isnan(count_array_A)
			id_fals = np.where(id_nan == False)
			p_count_A[id_fals] = p_count_A[id_fals] + 1.
			count_array_A[la0: la1, lb0: lb1][idv] = np.nan
			id_nm += 1.
		except FileNotFoundError:
			continue

	p_count_A[0, 0] = id_nm
	with h5py.File(tmp + 'stack_Amask_sum_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(sum_array_A)
	with h5py.File(tmp + 'stack_Amask_pcount_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(p_count_A)
	with h5py.File(tmp + 'stack_Amask_Var_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(f2_sum)

	return

def main():
	rich_a0, rich_a1, rich_a2 = 20, 30, 50
	## sersic pro of Zibetti 05
	mu_e = np.array([23.87, 25.22, 23.4])
	r_e = np.array([19.29, 19.40, 20])

	R_cut, bins = 1280, 80
	R_smal, R_max = 1, 1.7e3 # kpc

	### sub-stack mask A
	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	"""
	for kk in range( 3 ):
		'''
		#with h5py.File(load + 'mpi_h5/phot_z_%s_band_stack_cat.h5' % band[kk], 'r') as f:
		with h5py.File(load + 'photo_z/%s_band_img-center_cat.h5' % ( band[kk] ), 'r') as f:
			dat = np.array(f['a'])
		ra, dec, z = dat[0,:], dat[1,:], dat[2,:]
		zN = len(z)
		da0, da1 = 0, zN - 1
		Ntt = np.int(da1 - da0)
		set_z, set_ra, set_dec = z[da0: da1], ra[da0: da1], dec[da0: da1]
		m, n = divmod(Ntt, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		img_stack(kk, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])
		'''
		#with h5py.File(load + 'mpi_h5/phot_z_%s_band_stack_cat.h5' % band[kk], 'r') as f:
		with h5py.File(load + 'photo_z/%s_band_img-center_cat.h5' % ( band[kk] ), 'r') as f:
			dat = np.array(f['a'])
		ra, dec, z = dat[0,:], dat[1,:], dat[2,:]
		zN = len(z)
		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		img_stack(kk, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])

		commd.Barrier()

	### combine all of the sub-stack A mask image
	if rank == 0:
		for kk in range( 3 ):

			tot_N = 0
			mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
			p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)
			sqare_f = np.zeros((len(Ny), len(Nx)), dtype = np.float)

			for pp in range(cpus):

				with h5py.File(tmp + 'stack_Amask_pcount_%d_in_%s_band.h5' % (pp, band[kk]), 'r')as f:
					p_count = np.array(f['a'])
				with h5py.File(tmp + 'stack_Amask_sum_%d_in_%s_band.h5' % (pp, band[kk]), 'r') as f:
					sum_img = np.array(f['a'])
				with h5py.File(tmp + 'stack_Amask_Var_%d_in_%s_band.h5' % (pp, band[kk]), 'r') as f:
					f2_sum = np.array(f['a'])

				tot_N += p_count[0, 0]
				id_zero = p_count == 0
				ivx = id_zero == False
				mean_img[ivx] = mean_img[ivx] + sum_img[ivx]
				p_add_count[ivx] = p_add_count[ivx] + p_count[ivx]
				sqare_f[ivx] = sqare_f[ivx] + f2_sum[ivx]

			## save the stack image
			id_zero = p_add_count == 0
			mean_img[id_zero] = np.nan
			p_add_count[id_zero] = np.nan
			tot_N = np.int(tot_N)
			stack_img = mean_img / p_add_count
			where_are_inf = np.isinf(stack_img)
			stack_img[where_are_inf] = np.nan
			with h5py.File(load + 'photo_z/stack/stack_maskA_%d_in_%s_band.h5' % (tot_N, band[kk]), 'w') as f:
				f['a'] = np.array(stack_img)

			## save the variance image
			sqare_f[id_zero] = np.nan
			E_f2 = sqare_f / p_add_count
			id_inf = np.isinf(E_f2)
			E_f2[id_inf] = np.nan
			Var_f = E_f2 - stack_img**2  ## Variance img
			with h5py.File(load + 'photo_z/stack/stack_maskA_pix_Var_%d_in_%s_band.h5' % (tot_N, band[kk]), 'w') as f:
				f['a'] = np.array(Var_f)

	commd.Barrier()
	"""
	#N_sum = np.array([1497, 1503, 1492])
	N_sum = np.array([1176, 1176, 1169])

	r_a0, r_a1 = 1.0, 1.1
	## R200 calculate parameter
	M0, lamd0, z0 = 14.37, 30, 0.5
	F_lamda, G_z = 1.12, 0.18
	V_num = 200

	if rank == 0:
		for kk in range(3):

			SB_tt = pds.read_csv( load + 'Zibetti_SB/%s_band_BCG_ICL.csv' % band[kk])
			R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
			R_obs = R_obs**4
			## sersic part
			Mu_e, R_e, n_e = mu_e[kk], r_e[kk], 4.
			SB_Z05 = sers_pro(R_obs, Mu_e, R_e, n_e)

			with h5py.File(load + 'photo_z/stack/stack_maskA_%d_in_%s_band.h5' % (N_sum[kk], band[kk]), 'r') as f:
				stack_img = np.array(f['a'])

			## stack-img
			plt.figure()
			ax = plt.subplot(111)
			ax.set_title('stack %d imgs in %s band' % (N_sum[kk], band[kk]) )

			tf = ax.imshow(stack_img, cmap = 'Greys', vmin = 1e-5, vmax = 1e2, origin = 'lower', norm = mpl.colors.LogNorm())
			clut = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5)
			clut1 = Circle(xy = (x0, y0), radius = 0.2 * Rpp, fill = False, ec = 'g', alpha = 0.5)
			plt.colorbar(tf, ax = ax, fraction = 0.040, pad = 0.01, label = 'flux[nmaggy]')
			ax.add_patch(clut)
			ax.add_patch(clut1)
			ax.set_xlim(x0 - R_cut, x0 + R_cut)
			ax.set_ylim(y0 - R_cut, y0 + R_cut)
			plt.savefig(load + 'photo_z/stack/stack_%d_imgs_%s_band.png' % (N_sum[kk], band[kk]), dpi = 300)
			plt.close()

			ss_img = stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
			Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
			SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2) + mag_add[kk]
			id_nan = np.isnan(SB)
			SBt, Rt = SB[id_nan == False], Intns_r[id_nan == False]
			'''
			## read difference img
			# mean difference
			with h5py.File(load + 'photo_z/stack/stack_sky_mean_%d_imgs_%s_band.h5' % (N_sum[kk], band[kk]), 'r') as f:
				BCG_sky = np.array(f['a'])
			with h5py.File(load + 'photo_z/stack/M_sky_rndm_mean_%d_imgs_%s_band.h5' % (N_sum[kk], band[kk]), 'r') as f:
				rand_sky = np.arrayf['a'])
			'''
			# median difference
			with h5py.File(load + 'photo_z/stack/stack_sky_median_%d_imgs_%s_band.h5' % (N_sum[kk], band[kk]), 'r') as f:
				BCG_sky = np.array(f['a'])
			with h5py.File(load + 'photo_z/stack/M_sky_rndm_median_%d_imgs_%s_band.h5' % (N_sum[kk], band[kk]), 'r') as f:
				rand_sky = np.array(f['a'])

			differ_img = BCG_sky - rand_sky
			resi_add = differ_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]

			## differ-img
			plt.figure()
			ax = plt.subplot(111)
			ax.set_title('[%d imgs] difference img %s band' % (N_sum[kk], band[kk]) )

			tf = ax.imshow(differ_img, origin = 'lower', cmap = 'seismic', vmin = -2e-4, vmax = 2e-4,)
			clut = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5)
			clut1 = Circle(xy = (x0, y0), radius = 0.2 * Rpp, fill = False, ec = 'g', alpha = 0.5)
			plt.colorbar(tf, ax = ax, fraction = 0.040, pad = 0.01, label = 'flux[nmaggy]')
			ax.add_patch(clut)
			ax.add_patch(clut1)
			ax.set_xlim(x0 - R_cut, x0 + R_cut)
			ax.set_ylim(y0 - R_cut, y0 + R_cut)
			plt.savefig(load + 'photo_z/stack/%d_imgs_difference_%s_band.png' % (N_sum[kk], band[kk]), dpi = 300)
			plt.close()

			add_img = ss_img + resi_add
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

			# minus the RBL
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
			ax.set_title('SB profile [%d imgs %s band]' % (N_sum[kk], band[kk]) )

			ax.plot(Rt, SBt, 'r-', alpha = 0.5, label = 'stack imgs')
			ax.plot(R_add, SB_add, 'g--', alpha = 0.5, label = 'stack + difference img')
			ax.plot(cli_R, cli_SB, 'b-.', alpha = 0.5, label = 'stack + difference img - RBL')

			ax.plot(R_obs, SB_obs, 'k-.', label = 'Z05', alpha = 0.5)
			ax.plot(R_obs, SB_Z05, 'k:', label = 'Sersic', alpha = 0.5)
			ax.set_xlabel('$R[kpc]$')
			ax.set_ylabel('$SB[mag / arcsec^2]$')
			ax.set_xscale('log')
			ax.set_ylim(20, 32)
			ax.set_xlim(1, 1.5e3)
			ax.legend(loc = 1)
			ax.invert_yaxis()
			ax.grid(which = 'both', axis = 'both')
			ax.tick_params(axis = 'both', which = 'both', direction = 'in')

			plt.savefig(load + 'photo_z/stack/phot-z_SB_%s_band_%d_imgs.png' % (band[kk], N_sum[kk]), dpi = 300)
			plt.close()
	commd.Barrier()
	raise

if __name__ == "__main__":
	main()
