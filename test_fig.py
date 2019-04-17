# see the angular diameter distance and angular size
import h5py
import numpy as np
import pandas as pd
import astropy.io.ascii as asc
import astropy.io.fits as fits
import astropy.wcs as awc

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

import handy.scatter as hsc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from resamp import gen # test model
from scipy import interpolate as interp
from light_measure import light_measure

import subprocess as subpro
# constant
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
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2
# total catalogue with redshift
with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
	catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]

def mask_B():

	load = '/home/xkchen/mywork/ICL/data/test_data/'
	file = 'frame-r-ra36.455-dec-5.896-redshift0.233.fits'
	out_load = '/home/xkchen/mywork/ICL/data/SEX/result/test_1.cat'
	cmd = 'sex '+ load +file + ' -CATALOG_NAME %s'%out_load
	print(cmd)
	a = subpro.Popen(cmd, shell = True)
	a.wait()
	raise
	source = asc.read(out_load)
	Numb = source['NUMBER'][-1]
	A = source['A_IMAGE']
	B = source['B_IMAGE']
	theta = source['THETA_IMAGE']
	cx = source['X_IMAGE']
	cy = source['Y_IMAGE']
	#Kron = source['KRON_RADIUS']

	Kron = 6
	a = Kron*A
	b = Kron*B
	data_f = fits.open(load+file)
	img = data_f[0].data
	head_inf = data_f[0].header
	ra =  36.455
	dec = -5.896
	wcs1 = awc.WCS(head_inf)
	cx_BCG, cy_BCG = wcs1.all_world2pix(ra*U.deg, dec*U.deg, 1)
	
	plt.imshow(data_f[0].data, cmap = 'Greys', origin = 'lower', norm = mpl.colors.LogNorm())
	hsc.ellipses(cx, cy, w = a, h = b, rot = theta, fc = '', ec = 'r', lw = 0.5)
	plt.savefig('/home/xkchen/mywork/ICL/code/source_select.png', dpi = 300)
	plt.show()

	## mask B process
	mask_star = np.ones((img.shape[0], img.shape[1]), dtype = np.float)
	ox = np.linspace(0,2047,2048)
	oy = np.linspace(0,1488,1489)
	basic_coord = np.array(np.meshgrid(ox,oy))
	major = a/2
	minor = b/2 
	senior = np.sqrt(major**2 - minor**2)

	for k in range(Numb):
		xc = cx[k]
		yc = cy[k]

		lr = major[k]
		sr = minor[k]
		cr = senior[k]

		chi = theta[k]*np.pi/180
		df1 = lr**2 - cr**2*np.cos(chi)**2
		df2 = lr**2 - cr**2*np.sin(chi)**2

		fr = (basic_coord[0,:] - xc)**2*df1 +(basic_coord[1,:] - yc)**2*df2 \
			- cr**2*np.sin(2*chi)*(basic_coord[0,:] - xc)*(basic_coord[1,:] - yc)
		idr = fr/(lr**2*sr**2)
		jx = idr <= 1
		jx = (-1)*jx+1
		mask_star = mask_star*jx
	mirro_no_star = mask_star*img
	
	plt.imshow(mirro_no_star, cmap = 'Greys', origin = 'lower', norm = mpl.colors.LogNorm())
	hsc.ellipses(cx, cy, w = a, h = b, rot = theta, fc = '', ec = 'r', lw = 0.5)
	plt.savefig('/home/xkchen/mywork/ICL/code/star_mask.png', dpi = 300)
	plt.show()
	
	hdu = fits.PrimaryHDU()
	hdu.data = mirro_no_star
	hdu.header = head_inf
	hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/star_mask_1.fits',overwrite = True)
	raise
	return

def mask_A():
	#thresh = 24.5-10*np.log10((1+z_ref)/(1+0.233))

	load = '/home/xkchen/mywork/ICL/data/test_data/'
	file = 'frame-r-ra36.455-dec-5.896-redshift0.233.fits'
	out_load = '/home/xkchen/mywork/ICL/data/SEX/result/test_1.cat'
	cmd = 'sex '+ load +file + ' -CATALOG_NAME %s'%out_load
	print(cmd)
	a = subpro.Popen(cmd, shell = True)
	a.wait()
	
	source = asc.read(out_load)
	Numb = source['NUMBER'][-1]
	A = source['A_IMAGE']
	B = source['B_IMAGE']
	theta = source['THETA_IMAGE']
	cx = source['X_IMAGE']
	cy = source['Y_IMAGE']
	#Kron = source['KRON_RADIUS']
	
	Kron = 6
	a = Kron*A
	b = Kron*B
	data_f = fits.open(load+file)
	img = data_f[0].data
	head_inf = data_f[0].header
	ra =  36.455
	dec = -5.896
	wcs1 = awc.WCS(head_inf)
	cx_BCG, cy_BCG = wcs1.all_world2pix(ra*U.deg, dec*U.deg, 1)
	'''
	plt.imshow(data_f[0].data, cmap = 'Greys', origin = 'lower', norm = mpl.colors.LogNorm())
	hsc.ellipses(cx, cy, w = a, h = b, rot = theta, fc = '', ec = 'r', lw = 0.5)
	plt.plot(cx_BCG, cy_BCG, 'bo', alpha = 0.5)
	plt.savefig('/home/xkchen/mywork/ICL/code/source_select.png', dpi = 300)
	plt.show()
	'''
	mask_A = np.ones((img.shape[0], img.shape[1]), dtype = np.float)
	ox = np.linspace(0,2047,2048)
	oy = np.linspace(0,1488,1489)
	basic_coord = np.array(np.meshgrid(ox,oy))
	major = a/2
	minor = b/2 # set the star mask based on the major and minor radius
	senior = np.sqrt(major**2 - minor**2)
	for k in range(Numb):
		xc = cx[k]
		yc = cy[k]
		dcr = np.sqrt((xc - cx_BCG)**2 +(yc - cy_BCG)**2)
		if dcr <= 5 :
			mask_A = mask_A
		else:
			lr = major[k]
			sr = minor[k]
			cr = senior[k]
			chi = theta[k]*np.pi/180
			df1 = lr**2 - cr**2*np.cos(chi)**2
			df2 = lr**2 - cr**2*np.sin(chi)**2
			fr = (basic_coord[0,:] - xc)**2*df1 +(basic_coord[1,:] - yc)**2*df2 \
				- cr**2*np.sin(2*chi)*(basic_coord[0,:] - xc)*(basic_coord[1,:] - yc)
			idr = fr/(lr**2*sr**2)
			jx = idr<=1
			jx = (-1)*jx+1
			mask_A = mask_A*jx

	mirro_A = mask_A*img
	plt.imshow(mirro_A, cmap = 'Greys', origin = 'lower', norm = mpl.colors.LogNorm())
	hsc.ellipses(cx, cy, w = a, h = b, rot = theta, fc = '', ec = 'r', lw = 0.5)
	plt.plot(cx_BCG, cy_BCG, 'bo', alpha = 0.5)
	plt.savefig('/home/xkchen/mywork/ICL/code/all_mask.png', dpi = 300)
	plt.show()
	
	hdu = fits.PrimaryHDU()
	hdu.data = mirro_A
	hdu.header = head_inf
	hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/star_mask_A0.fits',overwrite = True)
	raise
	return

def test():
	mask_B()
	#mask_A()

def main():
	test()

if __name__ == "__main__":
	main()
