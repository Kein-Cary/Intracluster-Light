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

z_ref = 0.250 
Da_ref = Test_model.angular_diameter_distance(z_ref).value
# constant
rad2asec = U.rad.to(U.arcsec)

def resamp_func(d_file, z_set, ra_set, dec_set, band, out_file, pixel = 0.396):
	"""
	d_file : path where save the masked data (include file-name structure:'/xxx/xxx/xxx.xxx')
	z_set, ra_set, dec_set : ra, dec, z of will be resampled imgs
	band : the band of imgs, 'str' type
	out_file : path where to save the resampling img
	pixel : pixel scale, in unit 'arcsec' (default is 0.396)
	"""
	zn = len(z_set)

	for k in range(zn):
		ra_g = ra_set[k]
		dec_g = dec_set[k]
		z_g = z_set[k]
		Da_g = Test_model.angular_diameter_distance(z_g).value

		file = d_file % (band, ra_g, dec_g, z_g)
		data = fits.getdata(file, header = True)
		img = data[0]
		cx0 = data[1]['CRPIX1']
		cy0 = data[1]['CRPIX2']
		RA0 = data[1]['CRVAL1']
		DEC0 = data[1]['CRVAL2']

		wcs = awc.WCS(data[1])
		cx, cy = wcs.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

		L_ref = Da_ref * pixel / rad2asec
		L_z0 = Da_g * pixel / rad2asec
		b = L_ref / L_z0

		f_goal = flux_recal(img, z_g, z_ref)
		ix0 = np.int(cx0 / b)
		iy0 = np.int(cy0 / b)

		if b > 1:
			resam, xn, yn = sum_samp(b, b, f_goal, cx, cy)
		else:
			resam, xn, yn = down_samp(b, b, f_goal, cx, cy)
		# cheng the data type
		out_data = resam.astype('float32')

		xn = np.int(xn)
		yn = np.int(yn)
		x0 = resam.shape[1]
		y0 = resam.shape[0]

		keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2','CENTER_X','CENTER_Y',
				'CRVAL1','CRVAL2','BCG_RA','BCG_DEC','ORIGN_Z', 'P_SCALE']
		value = ['T', 32, 2, x0, y0, ix0, iy0, xn, yn, RA0, DEC0, ra_g, dec_g, z_g, pixel]
		ff = dict(zip(keys,value))
		fil = fits.Header(ff)
		fits.writeto(out_file + 'resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band, ra_g, dec_g, z_g), out_data, header = fil, overwrite = True)

	return

def main():

	dat = pds.read_csv('/home/xkchen/mywork/ICL/r_band_sky_catalog.csv')
	ra, dec, z = dat.ra, dat.dec, dat.z
	Nz = 10
	set_ra, set_dec, set_z = ra[:10], dec[:10], z[:10]

	out_file = '/home/xkchen/mywork/ICL/data/tmp_img/source_find/'
	d_file = '/home/xkchen/mywork/ICL/data/tmp_img/source_find/mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
	band = 'r'
	resamp_func(d_file, set_z, set_ra, set_dec, band, out_file)

if __name__ == "__main__":
	main()
