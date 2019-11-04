import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import handy.scatter as hsc
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
from light_measure_tmp import light_measure
from scipy.ndimage import map_coordinates as mapcd
from resample_modelu import sum_samp, down_samp
from astropy.coordinates import SkyCoord
from matplotlib.patches import Circle, Ellipse

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
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2

with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sample_catalog.h5', 'r') as f:
	catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]
d_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
band = ['r', 'i', 'g', 'u', 'z']
sky_SB = [21.04, 20.36, 22.01, 22.30, 19.18] # ref_value from SDSS

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

		## print sky SB for test
		SB_sky = 22.5 - 2.5 * np.log10( np.mean(sky_bl) ) + 2.5 * np.log10(pixel**2)
		print('sky = %.2f mag/arcsec^2' % SB_sky)

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

def sky_stack(band_id, sub_z, sub_ra, sub_dec):

	## stack sky image
	stack_N = len(sub_z)
	kk = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	sum_grid = np.array(np.meshgrid(Nx, Ny))
	sum_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	for jj in range(stack_N):
		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]

		## rule out bad image (once a cluster image is bad in a band, all the five band image will be ruled out!)
		except_cat = pds.read_csv(load + 'Except_%s_sample.csv' % band[kk])
		except_ra = ['%.3f' % ll for ll in except_cat['ra'] ]
		except_dec = ['%.3f' % ll for ll in except_cat['dec'] ]
		except_z = ['%.3f' % ll for ll in except_cat['z'] ]
		identi = ('%.3f'%ra_g in except_ra) & ('%.3f'%dec_g in except_dec) & ('%.3f'%z_g in except_z)

		if  identi == True: 
			continue
		else:
			data = fits.open( load + 'sky/sky_resamp/resample_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g) ) ## scaled
			img = data[0].data
			cx, cy = data[0].header['CENTER_X'], data[0].header['CENTER_Y']
			'''
			data = fits.open( load + 'sky/sky_arr/sky-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[kk]) ) ## without scale
			img = data[0].data
			head_inf = data[0].header
			wcs = awc.WCS(head_inf)
			cenx, ceny = wcs.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)
			cx, cy = np.int(cenx), np.int(ceny)
			'''

			la0 = np.int(y0 - cy)
			la1 = np.int(y0 - cy + img.shape[0])
			lb0 = np.int(x0 - cx)
			lb1 = np.int(x0 - cx + img.shape[1])

			sum_array[la0: la1, lb0: lb1] = sum_array[la0:la1, lb0:lb1] + img
			count_array[la0: la1, lb0: lb1] = img
			id_nan = np.isnan(count_array)
			id_fals = np.where(id_nan == False)
			p_count[id_fals] = p_count[id_fals] + 1
			count_array[la0: la1, lb0: lb1] = np.nan

	with h5py.File(load + 'test_h5/sky_sum_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(sum_array)

	with h5py.File(load + 'test_h5/sky_sum_pcount_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(p_count)

def main():
	t0 = time.time()
	Ntot = len(z)
	#commd.Barrier()
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
	N_tot = np.array([50, 100, 200, 500, 1000, 3000])
	x0, y0, bins = 2427, 1765, 65
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	popu = np.linspace(0, len(z) - 1, len(z))
	popu = popu.astype( int )
	popu = set(popu)

	for aa in range( len(N_tot) ):
		tt0 = random.sample(popu, N_tot[aa])
		set_z = z[tt0]
		set_ra = ra[tt0]
		set_dec = dec[tt0]

		for tt in range(3):
			m, n = divmod(N_tot[aa], cpus)
			N_sub0, N_sub1 = m * rank, (rank + 1) * m
			if rank == cpus - 1:
				N_sub1 += n
			sky_stack(tt, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])
		commd.Barrier()

		mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

		if rank == 0:
			for qq in range(3):
				tot_N = 0
				for pp in range(cpus):

					with h5py.File(load + 'test_h5/sky_sum_pcount_%d_in_%s_band.h5' % (pp, band[qq]), 'r')as f:
						p_count = np.array(f['a'])
					with h5py.File(load + 'test_h5/sky_sum_%d_in_%s_band.h5' % (pp, band[qq]), 'r') as f:
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

				R_cut = 800
				ss_img = stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
				Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, 10, Rpp, R_cut, R_cut, pixel, z_ref)
				flux0 = Intns + Intns_err
				flux1 = Intns - Intns_err
				SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
				SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2)
				SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2)
				err0 = SB - SB0
				err1 = SB1 - SB
				id_nan = np.isnan(SB)
				SB, SB0, SB1 = SB[id_nan == False], SB0[id_nan == False], SB1[id_nan == False] 
				pR, err0, err1 = Intns_r[id_nan == False], err0[id_nan == False], err1[id_nan == False]
				id_nan = np.isnan(SB1)
				err1[id_nan] = 100. # set a large value for show the break out errorbar

				plt.figure()
				ax = plt.subplot(111)
				ax.set_title( 'stack [%d] sky img %s band' % (tot_N, band[qq]) )
				tf = ax.imshow(stack_img, origin = 'lower')
				plt.colorbar(tf, ax = ax, fraction = 0.035, pad = 0.01, label = '$ flux \, [maggies]$')
				ax.set_xlim(x0 - 900, x0 + 900)
				ax.set_ylim(y0 - 900, y0 + 900)
				plt.savefig(load + 'sky/sky_%d_stack_%s_band.png' % (tot_N, band[qq]), dpi = 300)
				plt.close()

				plt.figure()
				gs = gridspec.GridSpec(2,1, height_ratios = [1, 2])
				bx = plt.subplot(gs[0])
				ax = plt.subplot(gs[1])

				bx.set_title('stack [%d] sky SB %s band' % (tot_N, band[qq]) )
				bx.errorbar(pR, SB, yerr = [err0, err1], xerr = None, ls = '', fmt = 'go', label = 'stack sky image', alpha = 0.5)
				bx.axhline(y = np.nanmean(SB), linestyle = '--', color = 'r', label = 'mean value', alpha = 0.5)
				bx.set_xscale('log')
				bx.set_xlabel('R[kpc]')
				bx.set_xlim(9, 1010)
				bx.set_ylabel('$ sky \; SB \; [mag/arcsec^2] $')
				bx.set_ylim(np.nanmean(SB) - 0.5, np.nanmean(SB) + 0.5)
				bx.legend(loc = 1)
				bx.invert_yaxis()

				ax.plot(pR, SB - SB[0], 'g--', label = 'deviation to central SB', alpha = 0.5)
				ax.axhline(y = 0, linestyle = '-.', color = 'r', label = '$ \Delta = 0 $', alpha = 0.5)
				ax.set_xscale('log')
				ax.set_xlabel('R [kpc]')
				ax.set_ylabel('SB(R) - SB(0)')
				ax.legend(loc = 1)

				plt.subplots_adjust(hspace = 0)
				plt.savefig(load + 'sky/sky_SB_%d_stack_%s_band.png' % (tot_N, band[qq]), dpi = 300)
				plt.close()

		commd.Barrier()
	t1 = time.time() - t0
	print('t = ', t1)
	raise

if __name__ == "__main__" :
	main()
