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
	band = ['u', 'g', 'r', 'i', 'z']
	load = '/home/xkchen/mywork/ICL/data/test_data/'
	mask = '/home/xkchen/mywork/ICL/query/SDSS_SQL_data.txt'

	for q in range(len(band)):

		file = 'frame-%s-ra36.455-dec-5.896-redshift0.233.fits' % band[q]
		#file = 'frame-r-ra36.455-dec-5.896-redshift0.233.fits'
		ra_g =  36.455
		dec_g = -5.896
		data_f = fits.open(load+file)
		img = data_f[0].data
		head_inf = data_f[0].header
		wcs = awc.WCS(head_inf)
		x_side = data_f[0].data.shape[1]
		y_side = data_f[0].data.shape[0]
		
		cat = pd.read_csv('/home/xkchen/mywork/ICL/query/SDSS_SQL_data.txt', skiprows = 1)
		ra_s = np.array(cat['ra'])
		dec_s = np.array(cat['dec'])
		mag = np.array(cat['r'])
		R0 = np.array(cat['psffwhm_r'])
		'''
		# with radius parameter
		iu = R0 >= 0
		R = 4.25*R0[iu]/pixel
		
		Ra = ra_s[iu]
		Dec = dec_s[iu]
		Mag = mag[iu]

		x, y = wcs.all_world2pix(Ra*U.deg, Dec*U.deg, 1)
		ia = (x >= 0) & (x <= x_side)
		ib = (y >= 0) & (y <= y_side)
		ie = (Mag <= 20)
		ic = ia & ib & ie
		comx = x[ic]
		comy = y[ic]
		cr = R[ic]
		'''
		# without radius parameter
		x, y = wcs.all_world2pix(ra_s*U.deg, dec_s*U.deg, 1)
		ia = (x >= 0) & (x <= x_side)
		ib = (y >= 0) & (y <= y_side)
		ie = (mag <= 20)
		ic = ia & ib & ie
		comx = x[ic]
		comy = y[ic]
		comr = 2*1.5/pixel
		cr = R0[ic]		
		
		R_ph = rad2asec/(Test_model.angular_diameter_distance(z = 0.233).value)
		R_p = R_ph/pixel
		cenx, ceny = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

		Numb = len(cr)
		mask_B = np.ones((img.shape[0], img.shape[1]), dtype = np.float)
		ox = np.linspace(0,2047,2048)
		oy = np.linspace(0,1488,1489)
		basic_coord = np.array(np.meshgrid(ox,oy))
		for k in range(Numb):
			xc = comx[k]
			yc = comy[k]
			#idr = np.sqrt((xc - basic_coord[0,:])**2 + (yc - basic_coord[1,:])**2)/cr[k]
			idr = np.sqrt((xc - basic_coord[0,:])**2 + (yc - basic_coord[1,:])**2)/comr
			jx = idr <= 1
			jx = (-1)*jx+1
			mask_B = mask_B*jx

		mirro_B = mask_B*img

		plt.imshow(mirro_B, cmap = 'Greys', origin = 'lower', vmin = 1e-3, norm = mpl.colors.LogNorm())
		hsc.circles(comx, comy, s = comr, fc = '', ec = 'r', lw = 1)
		hsc.circles(cenx, ceny, s = R_p, fc = '', ec = 'b', )
		hsc.circles(cenx, ceny, s = 1.1*R_p, fc = '', ec = 'b', ls = '--')
		plt.ylim(0, 1489)
		plt.xlim(0, 2048)
		plt.savefig('sdss_mask_test_band%s.png'%band[q], dpi = 600)
		plt.close()
		
		hdu = fits.PrimaryHDU()
		hdu.data = mirro_B
		hdu.header = head_inf
		hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/B_mask_data_%s_ra%.3f_dec%.3f.fits'%(band[q], ra_g, dec_g),overwrite = True)
		# aslo save the mask_matrix
		hdu = fits.PrimaryHDU()
		hdu.data = mask_B
		hdu.header = head_inf
		hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/B_mask_metrx_ra%.3f_dec%.3f.fits'%(ra_g, dec_g),overwrite = True)

	return

def mask_A():
	band = ['r', 'i', 'z']
	r_star = 2*1.5/pixel #mask star radius
	ZP = 22.5
	#band_limit = 24.5 - 10*np.log10((1 + z_ref) / (1 + z0))
	load = '/home/xkchen/mywork/ICL/data/test_data/'

	param_A = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A.sex'
	param_B = '/home/xkchen/mywork/ICL/data/SEX/default_mask_B.sex'
	out_cat = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A.param'
	out_load_A = '/home/xkchen/mywork/ICL/data/SEX/result/mask_A_test.cat'
	out_load_B = '/home/xkchen/mywork/ICL/data/SEX/result/mask_B_test.cat'
	file = 'frame-r-ra36.455-dec-5.896-redshift0.233.fits'

	ra_g =  36.455
	dec_g = -5.896
	data_f = fits.open(load+file)
	img = data_f[0].data
	head_inf = data_f[0].header
	wcs1 = awc.WCS(head_inf)
	cx_BCG, cy_BCG = wcs1.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)
	R_ph = rad2asec/(Test_model.angular_diameter_distance(z = 0.233).value)
	R_p = R_ph/pixel

	combine = np.zeros((1489, 2048), dtype = np.float)
	for q in range(len(band)):
		file_q = 'frame-%s-ra36.455-dec-5.896-redshift0.233.fits' % band[q]
		data_q = fits.open(load + file_q)
		img_q = data_q[0].data
		combine = combine + img_q
	# combine data
	hdu = fits.PrimaryHDU()
	hdu.data = combine
	hdu.header = head_inf
	hdu.writeto(load + 'combine_data_ra%.3f_dec%.3f.fits'%(ra_g, dec_g), overwrite = True)

	file_source = load + 'combine_data_ra%.3f_dec%.3f.fits'%(ra_g, dec_g)
	cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s'%(param_A, out_load_A, out_cat)
	print(cmd)
	a = subpro.Popen(cmd, shell = True)
	a.wait()
	
	source = asc.read(out_load_A)
	Numb = np.array(source['NUMBER'][-1])
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])
	theta = np.array(source['THETA_IMAGE'])
	cx = np.array(source['X_IMAGE']) - 1
	cy = np.array(source['Y_IMAGE']) - 1
	p_type = np.array(source['CLASS_STAR'])
	#Kron = source['KRON_RADIUS']
	Kron = 6
	a = Kron*A
	b = Kron*B

	cat = pd.read_csv('/home/xkchen/mywork/ICL/query/SDSS_SQL_data.txt', skiprows = 1)
	ra_s = np.array(cat['ra'])
	dec_s = np.array(cat['dec'])
	mag = np.array(cat['r'])
	x_side = img.shape[1]
	y_side = img.shape[0]
	x, y = wcs1.all_world2pix(ra_s*U.deg, dec_s*U.deg, 1)
	ia = (x >= 0) & (x <= x_side)
	ib = (y >= 0) & (y <= y_side)
	ie = (mag <= 20)
	ic = ia & ib & ie
	comx = x[ic]
	comy = y[ic]
	comr = np.ones(len(comx), dtype = np.float)*r_star
	com_chi = np.zeros(len(comx), dtype = np.float)

	cx = np.r_[cx, comx]
	cy = np.r_[cy, comy]
	a = np.r_[a, 2*comr]
	b = np.r_[b, 2*comr]
	theta = np.r_[theta, com_chi]
	Numb = Numb + len(comx)
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
		if dcr <= R_p/20 :
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
	mirro_A = mask_A *img

	plt.imshow(img, cmap = 'Greys', origin = 'lower', vmin = 1e-3, norm = mpl.colors.LogNorm())
	hsc.ellipses(cx, cy, w = a, h = b, rot = theta, fc = '', ec = 'r')
	plt.title('mine')
	plt.ylim(0, 1489)
	plt.xlim(0, 2048)
	plt.savefig('source_catalog.png', dpi = 600)
	plt.show()
	
	'''
	plt.imshow(mirro_A, cmap = 'Greys', origin = 'lower', vmin = 1e-3, norm = mpl.colors.LogNorm())
	hsc.ellipses(cx, cy, w = a, h = b, rot = theta, fc = '', ec = 'r', lw = 0.5)
	hsc.circles(cx_BCG, cy_BCG, s = R_p, fc = '', ec = 'b', )
	hsc.circles(cx_BCG, cy_BCG, s = 1.1*R_p, fc = '', ec = 'b', ls = '--')
	plt.plot(cx_BCG, cy_BCG, 'bo', alpha = 0.5)
	plt.xlim(0, 2048)
	plt.ylim(0, 1489)
	plt.savefig('/home/xkchen/mywork/ICL/code/pro_A_mask.png', dpi = 300)
	plt.show()
	'''
	'''
	# for compare
	Kron1 = np.arange(4,7,0.5)
	bin_number = 80
	light_test = []
	Ar_test = []
	for tt in range(len(Kron1)):
		a1 = Kron1[tt]*A
		b1 = Kron1[tt]*B
		mask_A1 = np.ones((img.shape[0], img.shape[1]), dtype = np.float)
		major1 = a1/2
		minor1 = b1/2 # set the star mask based on the major and minor radius
		senior1 = np.sqrt(major1**2 - minor1**2)
		for k in range(Numb):
			xc = cx[k]
			yc = cy[k]
			dcr = np.sqrt((xc - cx_BCG)**2 +(yc - cy_BCG)**2)
			if dcr <= 10 :
				mask_A1 = mask_A1
			else:
				lr = major1[k]
				sr = minor1[k]
				cr = senior1[k]
				chi = theta[k]*np.pi/180
				df1 = lr**2 - cr**2*np.cos(chi)**2
				df2 = lr**2 - cr**2*np.sin(chi)**2
				fr = (basic_coord[0,:] - xc)**2*df1 +(basic_coord[1,:] - yc)**2*df2 \
					- cr**2*np.sin(2*chi)*(basic_coord[0,:] - xc)*(basic_coord[1,:] - yc)
				idr = fr/(lr**2*sr**2)
				jx = idr<=1
				jx = (-1)*jx+1
				mask_A1 = mask_A1*jx
		mirro_A1 = mask_A1 *img
		light1, R1, Ar1, erro1 = light_measure(mirro_A1, bin_number, 1, R_p, cx_BCG, cy_BCG, pixel, 0.233)
		light_test.append(light1)
		Ar_test.append(Ar1)

	light, R, Ar, erro = light_measure(mirro_A, bin_number, 1, R_p, cx_BCG, cy_BCG, pixel, 0.233)
	
	for tt in range(len(Kron1)):
		rkk = Kron1[tt]/2
		plt.plot(Ar_test[tt], light_test[tt], ls = '--', color = mpl.cm.rainbow(tt/len(Kron1)), label = '$R_{mask}$ = %.2f'%rkk)	
	plt.legend(loc = 3)
	plt.xscale('log')
	plt.xlabel('$R[arcsec]$')
	plt.ylabel('$SB[mag/arcsec^2]$')
	plt.gca().invert_yaxis()
	plt.savefig('light_test.png', dpi = 600)
	plt.show()
	'''
	# joint mask
	'''
	mark_B = fits.getdata(load + 'B_mask_metrx_ra36.455_dec-5.896.fits', header = True)
	sum_mark = mask_A *mark_B[0]
	mirro = sum_mark *img
	'''
	plt.imshow(mirro_A, cmap = 'Greys', origin = 'lower', vmin = 1e-3, norm = mpl.colors.LogNorm())
	hsc.ellipses(cx, cy, w = a, h = b, rot = theta, fc = '', ec = 'r', ls = '--', lw = 0.5)
	hsc.circles(comx, comy, s = comr, fc = '', ec = 'b', ls = '-', lw = 0.5)
	hsc.circles(cx_BCG, cy_BCG, s = R_p, fc = '', ec = 'b', )
	hsc.circles(cx_BCG, cy_BCG, s = 1.1*R_p, fc = '', ec = 'b', ls = '--')
	plt.plot(cx_BCG, cy_BCG, 'bo', alpha = 0.5)
	plt.xlim(0, 2048)
	plt.ylim(0, 1489)
	plt.savefig('/home/xkchen/mywork/ICL/code/add_mask_AB.png', dpi = 300)
	plt.show()
	raise
	hdu = fits.PrimaryHDU()
	hdu.data = mirro_A
	hdu.header = head_inf
	hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/A_mask_data_%s_ra%.3f_dec%.3f.fits'%(band[q], ra_g, dec_g),overwrite = True)

	hdu = fits.PrimaryHDU()
	hdu.data = sum_mark
	hdu.header = head_inf
	hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/A_mask_metrx_ra%.3f_dec%.3f.fits'%(ra_g, dec_g),overwrite = True)
	raise
	return

def test():
	#mask_B()
	mask_A()

def main():
	test()

if __name__ == "__main__":
	main()
