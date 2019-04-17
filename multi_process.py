# this file use to fast resample data
import numpy as np
import astropy.io.fits as fits
import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
import astropy.wcs as awc

from itertools import product
import h5py
from multiprocessing import Pool
from itertools import starmap
from fast_resamp import gen, flux_recal

c0 = U.kpc.to(U.cm)
c1 = U.Mpc.to(U.pc)
c2 = U.Mpc.to(U.cm)
c3 = U.L_sun.to(U.erg/U.s)
c4 = U.rad.to(U.arcsec)
c5 = U.pc.to(U.cm)
Lsun = C.L_sun.value*10**7
# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
# constant
pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.250 
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2

def fill_name():
	with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sample_catalog.h5') as f:
		catalogue = np.array(f['a'])
	z = catalogue[0]
	ra = catalogue[1]
	dec = catalogue[2]
	lamb = catalogue[3]
	"""
	# name is the file name, 
	name_u : u band file name
	the formation: 'frame-u-raXXX.XXX-decXXX.XXX-redshiftX.XXX.fits.bz2' 
	"""
	name_u = []
	name_g = []
	name_r = []
	name_i = []
	name_z = []
	for k in range(len(z)):
		name_u.append('frame-u-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%(ra[k], dec[k], z[k]))
		name_g.append('frame-g-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%(ra[k], dec[k], z[k]))
		name_r.append('frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%(ra[k], dec[k], z[k]))
		name_i.append('frame-i-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%(ra[k], dec[k], z[k]))
		name_z.append('frame-z-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%(ra[k], dec[k], z[k]))
	f_name = [name_u, name_g, name_r, name_i, name_z]
	return f_name, ra, dec, z

def mutli_resamp(data_name, rac, dec, z):
	fill_nas = data_name
	ra = rac
	dec = dec
	z0 = z
	D_ref = Test_model.angular_diameter_distance(z_ref).value
	L_ref = D_ref*pixel/c4
	D_z = Test_model.angular_diameter_distance(z0).value
	L_z = D_z*pixel/c4
	b = L_ref/L_z
	b = np.float('%.4f'%b)
	f = fits.getdata('/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/'+fill_nas, header=True)
	wcs = awc.WCS(f[1])
	cx, cy = wcs.all_world2pix(ra*U.deg, dec*U.deg, 1)
	scale_f = flux_recal(f[0], z0, z_ref)
	xn, yn, resam = gen(scale_f, 1, b, cx, cy)

	x1 = resam.shape[1]
	y1 = resam.shape[0]
	intx = np.ceil(f[1]['CRPIX1'] // b)
	inty = np.ceil(f[1]['CRPIX2'] // b)
	keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2',
	'CENTER_X','CENTER_Y','CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z','Z_REF',]
	value = ['T', 32, 2, x1, y1, intx, inty, xn, yn, f[1]['CRVAL1'], f[1]['CRVAL2'],
	ra, dec, z0, z_ref ]
	head = dict(zip(keys, value))
	file_s = fits.Header(head)
	fits.writeto(
			'/mnt/ddnfs/data_users/cxkttwl/ICL/data/cut_sample/cut_record/resamp-'+fill_nas, resam, header = file_s, overwrite = True) 

	return 
def test_print(data_name):
	print(data_name +'00')

def test():
	#p = Pool(5)
	file_str, ra, dec, z = fill_name()
	'''
	result = map(mutli_resamp, file_str[0], ra, dec, z)
	result = list(result) # use map as a test run
	'''
	p = Pool(5)
	result = p.starmap(mutli_resamp, [(file_str[0], ra, dec, z),(file_str[1], ra, dec, z),
			(file_str[2], ra, dec, z),(file_str[3], ra, dec, z),(file_str[4], ra, dec, z)])
	p.close()
	p.join()

	return 
def main():
	test()

if __name__ == '__main__' :

	main()