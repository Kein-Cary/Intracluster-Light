import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import random
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C

from astropy import cosmology as apcy
from scipy.ndimage import map_coordinates as mapcd
from resample_modelu import sum_samp, down_samp
from astropy.coordinates import SkyCoord
from matplotlib.patches import Circle, Ellipse
from light_measure import light_measure, light_measure_Z0

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

pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.250 
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Rpp = (rad2asec / Da_ref) / pixel
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # zero point in unit (erg/s)/cm^-2

d_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/'
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
band = ['r', 'i', 'g', 'u', 'z']
sky_SB = [21.04, 20.36, 22.01, 22.30, 19.18] # ref_value from SDSS
mag_add = np.array([0, 0, 0, -0.04, 0.02])

## rule out sample
with h5py.File(load + 'mpi_h5/sky_rule_out_cat.h5', 'r') as f:
	rule_cat = np.array(f['a'])
except_ra = ['%.3f' % ll for ll in rule_cat[0,:] ]
except_dec = ['%.3f' % ll for ll in rule_cat[1,:] ]
except_z = ['%.3f' % ll for ll in rule_cat[2,:] ]

def sky_add(band_id, z_set, ra_set, dec_set):
	kk = np.int(band_id)
	Nz = len(z_set)
	for jj in range(Nz):
		ra_g = ra_set[jj]
		dec_g = dec_set[jj]
		z_g = z_set[jj]

		data = fits.open(d_file + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%(band[kk], ra_g, dec_g, z_g) )
		img = data[0].data
		head_inf = data[0].header
		wcs = awc.WCS(head_inf)
		cenx, ceny = wcs.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)
		R_ph = rad2asec / (Test_model.angular_diameter_distance(z_g).value)
		R_p = R_ph / pixel

		sky0 = data[2].data['ALLSKY'][0]
		sky_x = data[2].data['XINTERP'][0]
		sky_y = data[2].data['YINTERP'][0]
		inds = np.array(np.meshgrid(sky_x, sky_y))
		t_sky = mapcd(sky0, [inds[1,:], inds[0,:]], order = 1, mode = 'nearest')
		sky_bl = t_sky * (data[0].header['NMGY'])
		cimg = img + sky_bl ## PS: here the original image, do not apply Galactic extinction calibration
		SB_sky = 22.5 - 2.5 * np.log10( np.mean(sky_bl) ) + 2.5 * np.log10(pixel**2)

		## save the img with appling sky
		hdu = fits.PrimaryHDU()
		hdu.data = cimg
		hdu.header = data[0].header
		hdu.writeto(load + 'sky_add_img/cimg-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[kk]), overwrite = True)
		## save the sky img
		hdu = fits.PrimaryHDU()
		hdu.data = sky_bl
		hdu.header = data[0].header
		hdu.writeto(load + 'sky/sky_arr/sky-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[kk]), overwrite = True)

		fig = plt.figure(figsize = (16, 8))
		fig.suptitle('image comparison ra%.3f dec%.3f z%.3f %s band' % (ra_g, dec_g, z_g, band[kk]) )
		cluster0 = Circle(xy = (cenx, ceny), radius = R_p, fill = False, ec = 'b', alpha = 0.5, label = 'cluster region[1Mpc]')
		cluster1 = Circle(xy = (cenx, ceny), radius = R_p, fill = False, ec = 'b', alpha = 0.5, label = 'cluster region[1Mpc]')
		ax0 = plt.subplot(121)
		ax1 = plt.subplot(122)

		tf = ax0.imshow(img, cmap = 'Greys', vmin = 1e-5, vmax = 1e2, origin = 'lower', norm = mpl.colors.LogNorm())
		plt.colorbar(tf, ax = ax0, orientation = 'horizontal', fraction = 0.05, pad = 0.01, label = '$flux[nmaggy]$')
		ax0.add_patch(cluster0)
		ax0.set_title('calibrated image')
		ax0.set_xlim(0, img.shape[1])
		ax0.set_ylim(0, img.shape[0])
		ax0.set_xticks([])

		tf = ax1.imshow(cimg, cmap = 'Greys', vmin = 1e-2, vmax = 1e2, origin = 'lower', norm = mpl.colors.LogNorm())
		plt.colorbar(tf, ax = ax1, orientation = 'horizontal', fraction = 0.05, pad = 0.01, label = '$flux[nmaggy]$')
		ax1.add_patch(cluster1)
		ax1.set_title('sky-added image')
		ax1.set_xlim(0, cimg.shape[1])
		ax1.set_ylim(0, cimg.shape[0])
		ax1.set_xticks([])

		plt.tight_layout()
		plt.savefig(
			'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_class/sky_add/cimage_ra%.3f_dec%.3f_z%.3f_%s_band.png' % (ra_g, dec_g, z_g, band[kk]), dpi = 300)
		plt.close()

	return

def resamp_sky(band_id, sub_z, sub_ra, sub_dec):
	ii = np.int(band_id)
	zn = len(sub_z)
	for k in range(zn):
		ra_g = sub_ra[k]
		dec_g = sub_dec[k]
		z_g = sub_z[k]
		Da_g = Test_model.angular_diameter_distance(z_g).value

		data = fits.open(load + 'sky/sky_arr/sky-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[ii]) )
		img = data[0].data
		head = data[0].header
		cx0 = data[0].header['CRPIX1']
		cy0 = data[0].header['CRPIX2']
		RA0 = data[0].header['CRVAL1']
		DEC0 = data[0].header['CRVAL2']
		wcs = awc.WCS(head)
		cx, cy = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

		Angur = rad2asec / Da_g
		Rp = Angur / pixel
		L_ref = Da_ref * pixel / rad2asec
		L_z0 = Da_g * pixel / rad2asec
		b = L_ref / L_z0
		Rref = (rad2asec / Da_ref) / pixel
		ix0 = np.int(cx0 / b)
		iy0 = np.int(cy0 / b)

		if b > 1:
			resam, xn, yn = sum_samp(b, b, img, cx, cy)
		else:
			resam, xn, yn = down_samp(b, b, img, cx, cy)

		xn = np.int(xn)
		yn = np.int(yn)
		x0 = resam.shape[1]
		y0 = resam.shape[0]

		keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2','CENTER_X','CENTER_Y',
				'CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z', 'P_SCALE']
		value = ['T', 32, 2, x0, y0, ix0, iy0, xn, yn, RA0, DEC0, ra_g, dec_g, z_g, pixel]
		ff = dict(zip(keys,value))
		fil = fits.Header(ff)
		fits.writeto(load + 'sky/sky_resamp/resample_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), 
			resam, header = fil, overwrite = True)

def sky_stack(band_id, sub_z, sub_ra, sub_dec, cen_dx):

	## stack sky image
	stack_N = len(sub_z)
	kk = np.int(band_id)
	open_id = cen_dx

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	sum_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	sum_cen = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_cen = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_cen = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	## f^2 image and save the random position for each image
	f2_sum = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	id_n = 0

	for jj in range(stack_N):
		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]
		'''
		identy = ('%.3f'%ra_g in except_ra) & ('%.3f'%dec_g in except_dec) & ('%.3f'%z_g in except_z)
		if identy == True:
			continue
		else:
		'''
		## scaled
		data = fits.open( load + 'sky/sky_resamp/resample_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g) )
		img = data[0].data
		cx, cy = data[0].header['CENTER_X'], data[0].header['CENTER_Y']
		## rule out the edge pixels
		img[0,:] = np.nan
		img[-1,:] = np.nan
		img[:,0] = np.nan
		img[:,-1] = np.nan
		'''
		## unscale image
		data = fits.open( load + 'sky/sky_arr/sky-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[kk]) ) 
		wcs = awc.WCS(data[0].header)
		img = data[0].data
		cx, cy = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)
		'''
		la0 = np.int(y0 - cy)
		la1 = np.int(y0 - cy + img.shape[0])
		lb0 = np.int(x0 - cx)
		lb1 = np.int(x0 - cx + img.shape[1])

		idx = np.isnan(img)
		idv = np.where(idx == False)

		img_add = img - 0.
		img_add = img - np.nanmean(img)
		#img_add = img - np.nanmedian(img)

		sum_array[la0: la1, lb0: lb1][idv] = sum_array[la0:la1, lb0:lb1][idv] + img_add[idv]
		f2_sum[la0: la1, lb0: lb1][idv] = f2_sum[la0: la1, lb0: lb1][idv] + img_add[idv]**2
		count_array[la0: la1, lb0: lb1][idv] = img_add[idv]
		id_nan = np.isnan(count_array)
		id_fals = np.where(id_nan == False)
		p_count[id_fals] = p_count[id_fals] + 1
		count_array[la0: la1, lb0: lb1][idv] = np.nan
		id_n += 1.

		if open_id == 1:
			rnx, rny = img.shape[1] / 2, img.shape[0] / 2
			la0 = np.int(y0 - rny)
			la1 = np.int(y0 - rny + img.shape[0])
			lb0 = np.int(x0 - rnx)
			lb1 = np.int(x0 - rnx + img.shape[1])

			sum_cen[la0: la1, lb0: lb1][idv] = sum_cen[la0:la1, lb0:lb1][idv] + img_add[idv]
			count_cen[la0: la1, lb0: lb1][idv] = img_add[idv]
			id_nan = np.isnan(count_cen)
			id_fals = np.where(id_nan == False)
			p_count_cen[id_fals] = p_count_cen[id_fals] + 1
			count_cen[la0: la1, lb0: lb1][idv] = np.nan

	p_count[-1, -1] = id_n
	with h5py.File(tmp + 'sky_sum_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(sum_array)
	with h5py.File(tmp + 'sky_sum_pcount_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(p_count)
	## save the flux square img and the random position
	with h5py.File(tmp + 'sky_f2_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(f2_sum)

	if open_id == 1:
		with h5py.File(tmp + 'sky_fig-cen_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
			f['a'] = np.array(sum_cen)
		with h5py.File(tmp + 'sky_fig-cen_count_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
			f['a'] = np.array(p_count_cen)

def main():
	R_cut, bins = 1260, 80  # in unit of pixel (with 2Mpc inside)
	R_smal, R_max = 10, 1.7e3 # kpc

	de_cen = np.array([0.1, 0.1, 0.07, 0.09, 0.16]) # scaled
	#de_cen = np.array([0.095, 0.095, 0.065, 0.085, 0.155]) #unscaled

	'''
	## re-applied the sky to image (also save the sky img)
	for tt in range(len(band)):
		m, n = divmod(Ntot, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		sky_add(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])

	## resample the sky image
	for tt in range(len(band)):
		m, n = divmod(Ntot, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		resamp_sky(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
	'''
	## stack all the sky togather (without resampling)
	x0, y0, bins = 2427, 1765, 65
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	#N_tot = np.array([50, 100, 200, 500, 1000, 3000])
	open_dx = 0 # 0 / 1 , if stack with the frame center

	N_sum = []
	for tt in range(3):
		with h5py.File(load + 'mpi_h5/%s_band_sample_catalog.h5' % band[tt], 'r') as f:
			sub_array = np.array(f['a'])
		ra, dec, z, rich, r_mag = sub_array[0,:], sub_array[1,:], sub_array[2,:], sub_array[3,:], sub_array[4,:]
		zN = len(z)

		N_tot = np.array([zN])
		for aa in range( len(N_tot) ):
			'''
			## random sample
			np.random.seed(1)
			tt0 = np.random.choice(zN, size = N_tot[aa], replace = False)
			set_z = z[tt0]
			set_ra = ra[tt0]
			set_dec = dec[tt0]
			'''
			## for imgs selection
			set_z = z[:zN]
			set_ra = ra[:zN]
			set_dec = dec[:zN]

			m, n = divmod(N_tot[aa], cpus)
			N_sub0, N_sub1 = m * rank, (rank + 1) * m
			if rank == cpus - 1:
				N_sub1 += n
			sky_stack(tt, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1], open_dx)
			commd.Barrier()

			if rank == 0:

				tt_N = 0
				mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				sqr_f = np.zeros((len(Ny), len(Nx)), dtype = np.float)

				fig_cen = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				fig_cen_cnt = np.zeros((len(Ny), len(Nx)), dtype = np.float)

				for pp in range(cpus):

					with h5py.File(tmp + 'sky_sum_pcount_%d_in_%s_band.h5' % (pp, band[tt]), 'r')as f:
						p_count = np.array(f['a'])
					with h5py.File(tmp + 'sky_sum_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						sum_img = np.array(f['a'])

					with h5py.File(tmp + 'sky_f2_%d_in_%s_band.h5' % (pp, band[tt]), 'r') as f:
						f2_sum = np.array(f['a'])

					tt_N += p_count[-1, -1]
					id_zero = p_count == 0
					ivx = id_zero == False
					mean_img[ivx] = mean_img[ivx] + sum_img[ivx]
					sqr_f[ivx] = sqr_f[ivx] + f2_sum[ivx]
					p_add_count[ivx] = p_add_count[ivx] + p_count[ivx]

					if open_dx == 1:
						with h5py.File(tmp + 'sky_fig-cen_%d_in_%s_band.h5' % (pp, band[tt]), 'w') as f:
							sub_cen = np.array(f['a'])
						with h5py.File(tmp + 'sky_fig-cen_count_%d_in_%s_band.h5' % (pp, band[tt]), 'w') as f:
							sub_cen_cnt = np.array(f['a'])
						id_zero = sub_cen_cnt == 0
						ivx = id_zero == False
						fig_cen[ivx] = fig_cen[ivx] + sub_cen[ivx]				
						fig_cen_cnt[ivx] = fig_cen_cnt[ivx] + sub_cen_cnt[ivx]


				## sample sky SB
				tt_N = np.int(tt_N)
				N_sum.append(tt_N)

				## centered on BCG
				id_zero = p_add_count == 0
				mean_img[id_zero] = np.nan
				p_add_count[id_zero] = np.nan
				stack_img = mean_img / p_add_count
				id_inf = np.isinf(stack_img)
				stack_img[id_inf] = np.nan

				#with h5py.File(load + 'sky/sky_stack_%d_imgs_%s_band.h5' % (tt_N, band[tt]), 'w') as f:
				#with h5py.File(load + 'sky/sky_minus_media_%d_imgs_%s_band.h5' % (tt_N, band[tt]), 'w') as f:
				with h5py.File(load + 'sky/sky_minus_mean_%d_imgs_%s_band.h5' % (tt_N, band[tt]), 'w') as f:
					f['a'] = np.array(stack_img)

				## save the square flux and sky SB
				sqr_f[id_zero] = np.nan
				E_f2 = sqr_f / p_add_count
				id_inf = np.isinf(E_f2)
				E_f2[id_inf] = np.nan
				Var_f = E_f2 - stack_img**2

				#with h5py.File(load + 'sky/sky_Var_%d_in_%s_band.h5' % (tt_N, band[tt]), 'w') as f:
				#with h5py.File(load + 'sky/sky_Var_minus_media_%d_in_%s_band.h5' % (tt_N, band[tt]), 'w') as f:
				with h5py.File(load + 'sky/sky_Var_minus_mean_%d_in_%s_band.h5' % (tt_N, band[tt]), 'w') as f:
					f['a'] = np.array(Var_f)

				if open_dx == 1:
					id_zero = fig_cen_cnt == 0
					fig_cen[id_zero] = np.nan
					fig_cen_cnt[id_zero] = np.nan
					fig_cen_stack = fig_cen / fig_cen_cnt
					id_inf = np.isinf(fig_cen_stack)
					fig_cen_stack[id_inf] = np.nan

					#with h5py.File(load + 'sky/sky_fig-center_%d_imgs_%s_band.h5' % (tt_N, band[tt]), 'w') as f:
					#with h5py.File(load + 'sky/sky_fig-center_media_%d_imgs_%s_band.h5' % (tt_N, band[tt]), 'w') as f:
					with h5py.File(load + 'sky/sky_fig-center_mean_%d_imgs_%s_band.h5' % (tt_N, band[tt]), 'w') as f:
						f['a'] = np.array(fig_cen_stack)			

			commd.Barrier()

	## origin case
	color_lim = np.array([  [0.55, 1.025, 0.22, 0.170, 3.2], 
							[0.65, 1.225, 0.26, 0.205, 3.9] ])

	N_sum = np.array(N_sum)
	#N_sum = np.array([3378, 3363, 3377, 3378, 3372]) ## before select
	#N_sum = np.array([3355, 3340, 3354, 3355, 3349]) ## after select
	if rank == 0:
		for qq in range(3):
			## results
			#with h5py.File(load + 'sky/sky_stack_%d_imgs_%s_band.h5' % (N_sum[qq], band[qq]), 'r') as f:
			#with h5py.File(load + 'sky/sky_minus_media_%d_imgs_%s_band.h5' % (N_sum[qq], band[qq]), 'r') as f:
			with h5py.File(load + 'sky/sky_minus_mean_%d_imgs_%s_band.h5' % (N_sum[qq], band[qq]), 'r') as f:
				stack_img = np.array(f['a'])

			## Var_img
			#with h5py.File(load + 'sky/sky_Var_%d_in_%s_band.h5' % (N_sum[qq], band[qq]), 'r') as f:
			#with h5py.File(load + 'sky/sky_Std_minus_media_%d_in_%s_band.h5' % (N_sum[qq], band[qq]), 'r') as f:
			with h5py.File(load + 'sky/sky_Std_minus_mean_%d_in_%s_band.h5' % (N_sum[qq], band[qq]), 'r') as f:
				Var_img = np.array(f['a'])

			plt.figure(figsize = (12, 6))
			ax = plt.subplot(121)
			bx = plt.subplot(122)
			ax.set_title('stack %d imgs %s band [centered on BCG]' % (N_sum[qq], band[qq]) )
			bx.set_title('Variance plot %d imgs  %s band [centered on BCG]' % (N_sum[qq], band[qq]) )

			clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5, label = 'cluster region[1Mpc]')
			#tf = ax.imshow(stack_img, origin = 'lower', vmin = color_lim[0][qq], vmax = color_lim[1][qq], )
			tf = ax.imshow(stack_img, origin = 'lower', vmin = -5e-4, vmax = 5e-4, )
			ax.add_patch(clust)
			ax.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
			ax.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
			plt.colorbar(tf, ax = ax, fraction = 0.040, pad = 0.01, label = '$ flux \, [nmaggies]$')

			clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5, label = 'cluster region[1Mpc]')
			tf = bx.imshow(Var_img, origin = 'lower', )
			bx.add_patch(clust)
			bx.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
			bx.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
			plt.colorbar(tf, ax = bx, fraction = 0.040, pad = 0.01, label = '$ flux^2 $')				

			plt.tight_layout()
			#plt.savefig(load + 'sky/sky_BCG_Var_%d_imgs_%s_band.png' % (N_sum[qq], band[qq]), dpi = 300)
			#plt.savefig(load + 'sky/sky_BCG_Var_minus_media_%d_imgs_%s_band.png' % (N_sum[qq], band[qq]), dpi = 300)
			plt.savefig(load + 'sky/sky_BCG_Var_minus_mean_%d_imgs_%s_band.png' % (N_sum[qq], band[qq]), dpi = 300)
			plt.close()

	commd.Barrier()
	'''
	## sky histgram
	if rank == 0:
		plt.figure()
		ax = plt.subplot(111)
		ax.set_title('sky SB hist')

		for qq in range(len(band)):
			with h5py.File(load + 'sky/sky_SB_%d_in_%s_band.h5' % (N_sum[qq], band[qq]), 'r') as f:
				sky_measure = np.array(f['a'])
			sky_measure = sky_measure + mag_add[qq] # correction for AB trans
			ax.hist(sky_measure, histtype = 'step', color = mpl.cm.rainbow(qq / len(band)), density = True, 
				label = '%s band' % band[qq], alpha = 0.5)
			ax.axvline(x = sky_SB[qq], linestyle = '-.', color = mpl.cm.rainbow(qq / len(band)), alpha = 0.5, label = 'SDSS')
			ax.axvline(x = np.nanmean(sky_measure), linestyle = '--', color = mpl.cm.rainbow(qq / len(band)), alpha = 0.5)
		ax.set_xlabel('$ sky \;SB [mag / arcsec^2]$')
		ax.set_ylabel('PDF')
		ax.legend(loc = 1)
		plt.savefig(load + 'sky/sky_SB_PDF.png', dpi = 300)
		plt.close()

	commd.Barrier()

	## sky form Z05 method
	if rank == 0:
		plt.figure()
		ax = plt.subplot(111)
		ax.set_title('sky SB comparison')

		for qq in range(len(band)):
			with h5py.File(load + 'sky/sky_SB_%d_in_%s_band.h5' % (N_sum[qq], band[qq]), 'r') as f:
				sky_mine = np.array(f['a'])		
			sky_mine = sky_mine + mag_add[qq]
			# Z05 sky SB
			with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_ZIT/Z05_sky_%s_band.h5' % band[qq], 'r') as f:
				Z05_arr = np.array(f['a'])
			com_z, com_ra, com_dec = Z05_arr[0,:][1:], Z05_arr[1,:][1:], Z05_arr[2,:][1:]
			sky_com = Z05_arr[3,:][1:] + mag_add[qq]
			## select the stack sample data
			with h5py.File(load + 'mpi_h5/%s_band_sample_catalog.h5' % band[qq], 'r') as f:
				sub_array = np.array(f['a'])
			ra_in, dec_in, z_in = sub_array[0,:], sub_array[1,:], sub_array[2,:]

			includ_ra = ['%.3f' % ll for ll in ra_in ]
			includ_dec = ['%.3f' % ll for ll in dec_in ]
			includ_z = ['%.3f' % ll for ll in z_in ]

			str_z = ['%.3f' % ll for ll in com_z]
			str_ra = ['%.3f' % ll for ll in com_ra]
			str_dec = ['%.3f' % ll for ll in com_dec]

			Z05_set = np.zeros(len(com_z), dtype = np.float)
			for rr in range(len(com_z)):
				identy = (str_ra[rr] in includ_ra) & (str_dec[rr] in includ_dec) & (str_z[rr] in includ_z)
				if identy == True:
					Z05_set[rr] += 1.
				else:
					continue
			idx = Z05_set != 0.
			sky_Z05 = sky_com[idx]
			id_nan = np.isnan(sky_Z05)
			sky_Z05[id_nan] = 0.
			sky_Z05 = sky_Z05[sky_Z05 != 0]

			ax.hist(sky_mine, histtype = 'bar', color = 'k', density = True, alpha = 0.45)
			ax.axvline(x = np.nanmean(sky_mine), linestyle = '-', color = 'k', alpha = 0.5)

			ax.axvline(x = sky_SB[qq], linestyle = ':', color = mpl.cm.rainbow(qq / len(band)), alpha = 0.5, label = 'SDSS')
			ax.hist(sky_Z05, histtype = 'bar', color = mpl.cm.rainbow(qq / len(band)), density = True, label = 'Z05 %s band' % band[qq], alpha = 0.5)
			ax.axvline(x = np.nanmean(sky_Z05), linestyle = '-.', color = mpl.cm.rainbow(qq / len(band)), alpha = 0.5)

		ax.set_xlabel('$ sky \;SB [mag / arcsec^2]$')
		ax.set_ylabel('PDF')
		ax.legend(loc = 2)
		plt.savefig(load + 'sky/sky_SB_compare.png', dpi = 300)
		plt.close()
	commd.Barrier()
	'''
if __name__ == "__main__" :
	main()
