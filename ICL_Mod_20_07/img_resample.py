import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.ascii as asc
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C

from light_measure import flux_recal
from resample_modelu import sum_samp, down_samp
from astropy import cosmology as apcy

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
# constant
rad2asec = U.rad.to(U.arcsec)

def resamp_func(d_file, z_set, ra_set, dec_set, img_x, img_y, band, out_file, z_ref, 
	stack_info = None, pixel = 0.396, id_dimm = False, ):
	"""
	d_file : path where save the masked data (include file-name structure:'/xxx/xxx/xxx.xxx')
	z_set, ra_set, dec_set : ra, dec, z of will be resampled imgs
	band : the band of imgs, 'str' type
	out_file : path where to save the resampling img
	pixel : pixel scale, in unit 'arcsec' (default is 0.396)
	z_ref : reference redshift, the redshift to which all clusters will be scaled
	id_dimm : if do cosmic dimming correction or not
	img_x, img_y : BCG location on image frame before pixel resampling
	"""
	zn = len(z_set)
	bcg_x, bcg_y = [], []

	for k in range(zn):

		ra_g = ra_set[k]
		dec_g = dec_set[k]
		z_g = z_set[k]

		file = d_file % (band, ra_g, dec_g, z_g)
		data = fits.getdata(file, header = True)

		img = data[0]
		cx0 = data[1]['CRPIX1']
		cy0 = data[1]['CRPIX2']
		RA0 = data[1]['CRVAL1']
		DEC0 = data[1]['CRVAL2']

		#. convert (ra, dec) to location in image frame
		#wcs = awc.WCS(data[1])
		#cx, cy = wcs.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

		#. read BCG position from catalog
		cx, cy = img_x[k], img_y[k]

		Da_g = Test_model.angular_diameter_distance( z_g ).value
		Da_ref = Test_model.angular_diameter_distance( z_ref ).value

		Dl_g = Test_model.luminosity_distance( z_g ).value 
		Dl_ref = Test_model.luminosity_distance( z_ref ).value

		#. observation angle and flux factor at z_ref
		pixel_ref = pixel * ( Da_g / Da_ref )
		eta_flux = Dl_g**2 / Dl_ref**2       #... flux change due to distance

		eta_pix = pixel / pixel_ref

		if id_dimm == True:

			dimm_flux = flux_recal( img, z_g, z_ref )
			pre_img = dimm_flux * eta_flux

		else:
			pre_img = img * 1.

		ix0 = np.int( cx0 / eta_pix )
		iy0 = np.int( cy0 / eta_pix )

		if eta_pix > 1:
			resam, xn, yn = sum_samp( eta_pix, eta_pix, pre_img, cx, cy )
		else:
			resam, xn, yn = down_samp( eta_pix, eta_pix, pre_img, cx, cy )

		# cheng the data type
		out_data = resam.astype('float32')

		bcg_x.append( xn )
		bcg_y.append( yn )

		x0 = resam.shape[1]
		y0 = resam.shape[0]

		keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2','CENTER_X','CENTER_Y',
				'CRVAL1','CRVAL2','BCG_RA','BCG_DEC','ORIGN_Z', 'P_SCALE']
		value = ['T', 32, 2, x0, y0, ix0, iy0, xn, yn, RA0, DEC0, ra_g, dec_g, z_g, pixel]
		ff = dict(zip(keys,value))
		fil = fits.Header(ff)
		fits.writeto(out_file % (band, ra_g, dec_g, z_g), out_data, header = fil, overwrite = True)

	bcg_x = np.array(bcg_x)
	bcg_y = np.array(bcg_y)

	if stack_info != None:
		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
		values = [ra_set, dec_set, z_set, bcg_x, bcg_y]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv(stack_info)

	return

