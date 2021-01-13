import h5py
import pandas as pds
import numpy as np

import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
import astropy.io.fits as fits

# constant
vc = C.c.to(U.km/U.s).value
G = C.G.value # gravitation constant
Ms = C.M_sun.value # solar mass
kpc2m = U.kpc.to(U.m)
Msun2kg = U.M_sun.to(U.kg)

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
Lsun2erg_s = U.L_sun.to(U.erg/U.s)
rad2arcsec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
DH = vc/H0

def pix_flux_set_func(x0, y0, m_cen_x, m_cen_y, set_ra, set_dec, set_z, set_x, set_y, img_file, band_str,):
	"""
	x0, y0 : the point which need to check flux (in the aveged image [applied astacking process])
	m_cen_x, m_cen_y : the center pixel of stacking image
	set_ra, set_dec, set_z, set_x, set_y : img information of stacking sample imgs,
			including ra, dec, z, and BCG position on img (set_x, set_y)
	img_file : imgs for given catalog ('XXX/XX.fits')
	band_str : the filter information
	"""
	targ_f = []
	Ns = len( set_z )

	for kk in range( Ns ):

		ra_g, dec_g, z_g = set_ra[kk], set_dec[kk], set_z[kk]
		img_x, img_y = set_x[kk], set_y[kk]

		data = fits.open(img_file % (band_str, ra_g, dec_g, z_g) )
		img = data[0].data

		dev_05_x = img_x - np.int( img_x )
		dev_05_y = img_y - np.int( img_y )

		if dev_05_x > 0.5:
			x_cen = np.int( img_x ) + 1
		else:
			x_cen = np.int( img_x )

		if dev_05_y > 0.5:
			y_cen = np.int( img_y ) + 1
		else:
			y_cen = np.int( img_y )

		x_ori_0, y_ori_0 = x0 + x_cen - m_cen_x, y0 + y_cen - m_cen_y

		identy_0 = ( (x_ori_0 >= 0) & (x_ori_0 < 2048) ) & ( (y_ori_0 >= 0) & (y_ori_0 < 1489) )

		if identy_0 == True:
			targ_f.append( img[y_ori_0, x_ori_0] )

	targ_f = np.array( targ_f )

	return targ_f

def radi_bin_flux_set_func(targ_R, R_edgs, stack_img, m_cen_x, m_cen_y, R_bins, set_ra, set_dec, set_z, set_x, set_y,
	img_file, pix_size, band_str, out_file,):
	"""
	targ_R : the radius bin in which need to check flux (in unit arcsec or kpc)
	R_edgs : the radius bin edges, the unit is the same as targ_R, arcsec or kpc
	stack_img : the stacking img file (in .h5 format, img = np.array(f['a']),)
	R_bins : the radius bins will be applied on the stacking img or imgs, in unit of pixel
	set_ra, set_dec, set_z, set_x, set_y : the catalog imformation, (set_x, set_y) is BCG position
		on img
	img_file : imgs for given catalog ('XXX/XX.fits')
	pix_size : the pixel scale of imgs
	band_str : filter, str type
	out_file : out-put the data (.h5 file)
	"""
	with h5py.File(stack_img, 'r') as f:
		tt_img = np.array( f['a'] )

	R_cen = 0.5 * (R_edgs[1:] + R_edgs[:-1])

	Nx = np.linspace(0, tt_img.shape[1] - 1, tt_img.shape[1] )
	Ny = np.linspace(0, tt_img.shape[0] - 1, tt_img.shape[0] )
	grd = np.array( np.meshgrid(Nx, Ny) )
	cen_dR = np.sqrt( (grd[0] - m_cen_x)**2 + (grd[1] - m_cen_y)**2 )

	ddr = np.abs( R_cen - targ_R )
	idx = np.where( ddr == ddr.min() )[0]
	edg_lo = R_bins[idx]
	edg_hi = R_bins[idx + 1]

	id_flux = (cen_dR >= edg_lo) & (cen_dR < edg_hi)
	id_nn = np.isnan( tt_img )
	id_effect = ( id_nn == False ) & id_flux
	f_stack = tt_img[ id_effect ]

	sub_f_arr = []
	Ns = len( set_z )

	for kk in range( Ns ):

		ra_g, dec_g, z_g = set_ra[kk], set_dec[kk], set_z[kk]
		img_x, img_y = set_x[kk], set_y[kk]

		data = fits.open( img_file % (band_str, ra_g, dec_g, z_g),)
		img = data[0].data

		lx = np.linspace(0, img.shape[1] - 1, img.shape[1])
		ly = np.linspace(0, img.shape[0] - 1, img.shape[0])
		grd_lxy = np.array( np.meshgrid(lx, ly) )

		dev_05_x = img_x - np.int( img_x )
		dev_05_y = img_y - np.int( img_y )

		if dev_05_x > 0.5:
			x_cen = np.int( img_x ) + 1
		else:
			x_cen = np.int( img_x )

		if dev_05_y > 0.5:
			y_cen = np.int( img_y ) + 1
		else:
			y_cen = np.int( img_y )

		## identy by radius edges
		dr = np.sqrt( (grd_lxy[0] - x_cen)**2 + (grd_lxy[1] - y_cen)**2)
		id_last = (dr >= edg_lo) & (dr < edg_hi)
		idnn = np.isnan( img )
		idin = id_last & (idnn == False)

		if np.sum( idin ) == 0:
			continue
		else:
			sub_f_arr.append( img[ idin ] )

	#### in case of on pixel contribution (for pixel resampling case)
	if sub_f_arr == []:
		tmp_f_arr = [np.nan, np.nan, np.nan]
		sub_f_arr = tmp_f_arr

	dtf = np.hstack( sub_f_arr )
	dtf = dtf.astype( np.float32 )

	with h5py.File( out_file % targ_R, 'w') as f:
		f['pix_flux'] = np.array( dtf )

	return

def Rbin_flux_track(targ_R, R_lim, flux_lim, img_file, set_ra, set_dec, set_z, set_x, set_y, out_file,):
	"""
	targ_R : the radius bin in which the flux will be collected
	R_lim : the limited radius edges, R_lim[0] is the inner one, and R_lim[1] is the outer one
			in unit of pixel 
	flux_lim : the flux range in which those pixels will be collected,
				flux_lim[0] is the smaller one and flux_lim[1] is the larger one
	set_ra, set_dec, set_z, set_x, set_y : the catalog information, including the BCG position (set_x, set_y)
				in image region
	img_file : .fits files
	out_file : out put data of the collected flux, .csv file
	"""
	Ns = len(set_z)

	for mm in range( Ns ):

		ra_g, dec_g, z_g = set_ra[mm], set_dec[mm], set_z[mm]
		cen_x, cen_y = set_x[mm], set_y[mm]

		data = fits.open( img_file % (ra_g, dec_g, z_g),)
		img = data[0].data

		lx = np.linspace(0, img.shape[1] - 1, img.shape[1])
		ly = np.linspace(0, img.shape[0] - 1, img.shape[0])
		grd_lxy = np.array( np.meshgrid(lx, ly) )

		pix_dR = np.sqrt( (grd_lxy[0] - cen_x)**2 + (grd_lxy[1] - cen_y)**2)

		id_rx = (pix_dR >= R_lim[0]) & (pix_dR < R_lim[1])
		idnn = np.isnan( img )

		id_vlim = ( img >= flux_lim[0] ) & ( img <= flux_lim[1] )

		id_set = id_rx & (idnn == False) & id_vlim

		my, mx = np.where( id_set == True )

		flux_in = img[ id_set ]

		flux_in = flux_in.astype( np.float32 )
		my = my.astype( np.int32 )
		mx = mx.astype( np.int32 )

		# save the flux info.
		with h5py.File( out_file % (targ_R, ra_g, dec_g, z_g), 'w') as f:
			f['pix_flux'] = np.array( flux_in )
			f['pos_x'] = np.array( mx )
			f['pos_y'] = np.array( my )

	return

