import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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

d_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/photo_data/'
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
band = ['r', 'g', 'i', 'u', 'z']
sky_SB = [21.04, 22.01, 20.36, 22.30, 19.18] # ref_value from SDSS
mag_add = np.array([0, 0, 0, -0.04, 0.02])

def photo_sky(band_id, z_set, ra_set, dec_set):

	kk = np.int(band_id)
	Nz = len(z_set)
	for jj in range(Nz):
		ra_g = ra_set[jj]
		dec_g = dec_set[jj]
		z_g = z_set[jj]

		try:
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

			## save the sky img
			hdu = fits.PrimaryHDU()
			hdu.data = sky_bl
			hdu.header = data[0].header
			hdu.writeto(load + 
				'photo_z/sky/sky_img/sky-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[kk]), overwrite = True)
		except FileNotFoundError:
			continue

def photo_sky_resamp(band_id, sub_z, sub_ra, sub_dec):

	ii = np.int(band_id)
	zn = len(sub_z)
	for k in range(zn):
		ra_g = sub_ra[k]
		dec_g = sub_dec[k]
		z_g = sub_z[k]
		Da_g = Test_model.angular_diameter_distance(z_g).value

		try:
			data = fits.open(load + 'photo_z/sky/sky_img/sky-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[ii]) )
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
			fits.writeto(load + 
				'photo_z/sky/sky_resam_img/resampl_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), 
				resam, header = fil, overwrite = True)
		except FileNotFoundError:
			continue

def photo_sky_cut(band_id, sub_z, sub_ra, sub_dec):

	ii = np.int(band_id)
	zn = len(sub_z)
	for k in range(zn):
		ra_g = sub_ra[k]
		dec_g = sub_dec[k]
		z_g = sub_z[k]
		try:
			data = fits.getdata(load + 'photo_z/sky/sky_resam_img/resampl_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % 
				(band[ii], ra_g, dec_g, z_g), header = True)
			img = data[0]
			BCGx, BCGy = data[1]['CENTER_X'], data[1]['CENTER_Y']
			RA0, DEC0 = data[1]['CRVAL1'], data[1]['CRVAL2']

			xc, yc = np.int(img.shape[1] / 2), np.int(img.shape[0] / 2)
			re_img = img[yc - np.int(Rpp): yc + np.int(Rpp), xc - np.int(1.3 * Rpp): xc + np.int(1.3 * Rpp)]

			New_bcgx = BCGx - (xc - np.int(1.3 * Rpp))
			New_bcgy = BCGy - (yc - np.int(Rpp))

			Lx = re_img.shape[1]
			Ly = re_img.shape[0]
			Crx = np.int(1.3 * Rpp)
			Cry = np.int(Rpp)

			keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2','CENTER_X','CENTER_Y',
					'CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z', 'P_SCALE']
			value = ['T', 32, 2, Lx, Ly, Crx, Cry, New_bcgx, New_bcgy, RA0, DEC0, ra_g, dec_g, z_g, pixel]
			ff = dict(zip(keys,value))
			fil = fits.Header(ff)
			fits.writeto(load + 'photo_z/sky/sky_cut_img/Cut_edge_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % 
			(band[ii], ra_g, dec_g, z_g), re_img, header = fil, overwrite=True)
		except FileNotFoundError:
			continue

def phot_z_center_cat(band_id, sub_z, sub_ra, sub_dec, sub_rmag, sub_rich):

	ii = np.int(band_id)
	zn = len(sub_z)

	ra_fit = np.zeros(zn, dtype = np.float)
	dec_fit = np.zeros(zn, dtype = np.float)
	z_fit = np.zeros(zn, dtype = np.float)
	rmag_fit = np.zeros(zn, dtype = np.float)
	rich_fit = np.zeros(zn, dtype = np.float)

	#cen_dst = 0.65 ## centric distance: 1 Mpc, 0.8Mpc, 0.65Mpc

	for k in range(zn):
		ra_g = sub_ra[k]
		dec_g = sub_dec[k]
		z_g = sub_z[k]

		try:
			data_A = fits.getdata(load + 
				'photo_z/resample/pho_z-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
			img_A = data_A[0]
			xn = data_A[1]['CENTER_X']
			yn = data_A[1]['CENTER_Y']

			cx = np.int(img_A.shape[1] / 2)
			cy = np.int(img_A.shape[0] / 2)

			x_side = np.array( [cx - np.int(1.3 * Rpp), cx + np.int(1.3 * Rpp)] )
			y_side = np.array( [cy - np.int(Rpp), cy + np.int(Rpp)] )

			idx = (x_side[0] < xn) & (xn < x_side[1])
			idy = (y_side[0] < yn) & (yn < y_side[1])
			idz = idx & idy

			if idz == True:
				ra_fit[k], dec_fit[k], z_fit[k] = ra_g, dec_g, z_g
				rmag_fit[k] = sub_rmag[k]
				rich_fit[k] = sub_rich[k]
			else:
				continue

		except FileNotFoundError:
			continue

	id_zeros = ra_fit == 0
	id_false = id_zeros == False

	dec_fit = dec_fit[id_false]
	z_fit = z_fit[id_false]
	rmag_fit = rmag_fit[id_false]
	ra_fit = ra_fit[id_false]
	rich_fit = rich_fit[id_false]

	sel_arr = np.array([ra_fit, dec_fit, z_fit, rmag_fit, rich_fit])

	with h5py.File(tmp + 'sky_select_%d_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(sel_arr)
	with h5py.File(tmp + 'sky_select_%d_%s_band.h5' % (rank, band[ii]) ) as f:
		for ll in range(len(sel_arr)):
			f['a'][ll,:] = sel_arr[ll,:]

	return

def main():
	'''
	with h5py.File(load + 'mpi_h5/photo_z_difference_sample.h5', 'r') as f:
		dat = np.array(f['a'])
	ra, dec, z, rich, r_mag = dat[0,:], dat[1,:], dat[2,:], dat[3,:], dat[4,:]
	zN = len(z)

	## read the sky image (also save the sky img)
	for tt in range(3):
		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		photo_sky(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
	commd.Barrier()

	## resample the sky image
	for tt in range(3):
		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		photo_sky_resamp(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
	commd.Barrier()

	## rule out the edges pixels
	for tt in range(3):
		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		photo_sky_cut(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
	commd.Barrier()
	'''
	## build img-center catalogue
	for tt in range(3):
		with h5py.File(load + 'mpi_h5/phot_z_%s_band_stack_cat.h5' % band[tt], 'r') as f:
			dat = np.array(f['a'])
		ra, dec, z, rich, r_mag = dat[0,:], dat[1,:], dat[2,:], dat[3,:], dat[4,:]
		zN = len(z)

		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		phot_z_center_cat(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1], r_mag[N_sub0 :N_sub1], rich[N_sub0 :N_sub1])
		commd.Barrier()

	if rank == 0:
		for tt in range(3):

			set_ra = np.array([0.])
			set_dec = np.array([0.])
			set_z = np.array([0.])
			set_mag = np.array([0.])
			set_rich = np.array([0.])
			for pp in range(cpus):
				with h5py.File(tmp + 'sky_select_%d_%s_band.h5' % (pp, band[tt]), 'r') as f:
					sel_arr = np.array(f['a'])
				set_ra = np.r_[ set_ra, sel_arr[0,:] ]
				set_dec = np.r_[ set_dec, sel_arr[1,:] ]
				set_z = np.r_[ set_z, sel_arr[2,:] ]
				set_mag = np.r_[ set_mag, sel_arr[3,:] ]
				set_rich = np.r_[ set_rich, sel_arr[4,:] ]

			set_ra = set_ra[1:]
			set_dec = set_dec[1:]
			set_z = set_z[1:]
			set_mag = set_mag[1:]
			set_rich = set_rich[1:]

			n_sum = len(set_z)
			set_array = np.array([set_ra, set_dec, set_z, set_rich, set_mag])
			with h5py.File(load + 'photo_z/%s_band_img-center_cat.h5' % ( band[tt] ), 'w') as f:
				f['a'] = np.array(set_array)
			with h5py.File(load + 'photo_z/%s_band_img-center_cat.h5' % ( band[tt] ) ) as f:
				for ll in range( len(set_array) ):
					f['a'][ll,:] = set_array[ll,:]

			print( len(set_z) )
	commd.Barrier()

if __name__ == "__main__":
	main()
