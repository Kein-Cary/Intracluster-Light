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
		Da_g = Test_model.angular_diameter_distance(z_g).value
		Da_ref = Test_model.angular_diameter_distance(z_ref).value

		file = d_file % (band, ra_g, dec_g, z_g)
		data = fits.getdata(file, header = True)
		img = data[0]
		cx0 = data[1]['CRPIX1']
		cy0 = data[1]['CRPIX2']
		RA0 = data[1]['CRVAL1']
		DEC0 = data[1]['CRVAL2']

		#wcs = awc.WCS(data[1])
		#cx, cy = wcs.all_world2pix(ra_g * U.deg, dec_g * U.deg, 0)
		cx, cy = img_x[k], img_y[k]

		L_ref = Da_ref * pixel / rad2asec
		L_z0 = Da_g * pixel / rad2asec
		b = L_ref / L_z0

		if id_dimm == True:
			f_goal = flux_recal(img, z_g, z_ref)
		else:
			f_goal = img * 1.

		ix0 = np.int(cx0 / b)
		iy0 = np.int(cy0 / b)

		if b > 1:
			resam, xn, yn = sum_samp(b, b, f_goal, cx, cy)
		else:
			resam, xn, yn = down_samp(b, b, f_goal, cx, cy)

		# cheng the data type
		out_data = resam.astype('float32')

		bcg_x.append(xn)
		bcg_y.append(yn)

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

def main():

	from mpi4py import MPI
	commd = MPI.COMM_WORLD
	rank = commd.Get_rank()
	cpus = commd.Get_size()

	home = '/home/xkchen/data/SDSS/'
	load = '/home/xkchen/data/SDSS/'

	dat = pds.read_csv('/home/xkchen/fig_tmp/test_1000_no_select.csv')
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	zN = len( z )
	m, n = divmod(zN, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	size_arr = np.array([10, 20])

	band = 'r'
	z_ref = 0.25

	for mm in range( 2 ):

		d_file = '/home/xkchen/fig_tmp/norm_mask/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_' + '%d-FWHM-ov2.fits' % (size_arr[mm])
		out_file = '/home/xkchen/fig_tmp/pix_resample/resamp-%s-ra%.3f-dec%.3f-redshift%.3f_' + '%d-FWHM-ov2.fits' % (size_arr[mm])

		resamp_func(d_file, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1], clus_x[N_sub0 :N_sub1], clus_y[N_sub0 :N_sub1], 
			band, out_file, z_ref, id_dimm = True,)

	raise

if __name__ == "__main__":
	main()

