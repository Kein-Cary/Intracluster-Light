# see the angular diameter distance and angular size
import matplotlib as mpl
import handy.scatter as hsc
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pd
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

from resamp import gen
import subprocess as subpro
from dustmaps.sfd import SFDQuery
from extinction_redden import A_wave
from astropy.coordinates import SkyCoord
from light_measure import light_measure, flux_recal

import time
import sfdmap
m = sfdmap.SFDMap('/home/xkchen/mywork/ICL/data/redmapper/sfddata_maskin', scaling = 0.86)
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
# dust correct
Rv = 3.1
sfd = SFDQuery()
band = ['u', 'g', 'r', 'i', 'z']
l_wave = np.array([3551, 4686, 6166, 7480, 8932])
mag_add = np.array([-0.04, 0, 0, 0, 0.02])
zop = np.array([22.46, 22.5, 22.5, 22.5, 22.52])
sb_lim = np.array([24.35, 25, 24.5, 24, 22.9])
def mask_wit():
	x0 = np.linspace(0, 2047, 2048)
	y0 = np.linspace(0, 1488, 1489)
	img_grid = np.array(np.meshgrid(x0, y0))

	load = '/home/xkchen/mywork/ICL/data/total_data/sample_02_03/'
	mask = '/home/xkchen/mywork/ICL/data/star_catalog/'
	for q in range(len(band)):
		'''
		file = 'frame-%s-ra260.613-dec32.133-redshift0.223.fits.bz2' % band[q]
		ra_g = 260.613
		dec_g = 32.133
		z_g = 0.223
		
		file = 'frame-%s-ra36.455-dec-5.896-redshift0.233.fits.bz2' % band[q]
		ra_g = 36.455
		dec_g = -5.896
		z_g = 0.233

		file = 'frame-%s-ra240.829-dec3.279-redshift0.222.fits.bz2' % band[q]
		ra_g = 240.829
		dec_g = 3.279
		z_g = 0.222
		'''

		dust_map_11 = np.zeros((1489, 2048), dtype = np.float)
		dust_map_98 = np.zeros((1489, 2048), dtype = np.float)
		mapN = np.int(20)
		for k in range(mapN):
			file = 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[q], ra[k], dec[k], z[k])
			data_f = fits.open(load+file)
			img = data_f[0].data
			head_inf = data_f[0].header
			wcs = awc.WCS(head_inf)
			x_side = data_f[0].data.shape[1]
			y_side = data_f[0].data.shape[0]

			t0 = time.time()
			ra_img, dec_img = wcs.all_pix2world(img_grid[0,:], img_grid[1,:], 1)
			pos = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
			BEV = sfd(pos)
			bev = m.ebv(pos)

			dust_map_11 = dust_map_11 + bev
			dust_map_98 = dust_map_98 + BEV * 0.86

		map_11 = dust_map_11 / mapN
		map_98 = dust_map_98 / mapN

		plt.figure()
		gf1 = plt.imshow(map_11, cmap = 'rainbow', origin = 'lower')
		plt.title('$map_{2011} \; stack \; %.0f \; in \; %s \; band$' % (mapN, band[q]))
		plt.colorbar(gf1, fraction = 0.035, pad = 0.01, label = '$f[nmagy]$')
		plt.savefig('map_11_stack_%s_band.png' % band[q], dpi = 600)
		plt.close()

		plt.figure()
		gf2 = plt.imshow(map_98, cmap = 'rainbow', origin = 'lower')
		plt.title('$map_{1998} \; stack \; %.0f \; in \; %s \; band$' % (mapN, band[q]))
		plt.colorbar(gf1, fraction = 0.035, pad = 0.01, label = '$f[nmagy]$')
		plt.savefig('map_98_stack_%s_band.png' % band[q], dpi = 600)
		plt.close()
		'''
		data_f = fits.open(load+file)
		img = data_f[0].data
		head_inf = data_f[0].header
		wcs = awc.WCS(head_inf)
		x_side = data_f[0].data.shape[1]
		y_side = data_f[0].data.shape[0]

		t0 = time.time()
		ra_img, dec_img = wcs.all_pix2world(img_grid[0,:], img_grid[1,:], 1)
		pos = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
		BEV = sfd(pos, order = 1)
		bev = m.ebv(pos)

		Av = Rv * BEV * 0.86
		Al = A_wave(l_wave[q], Rv) * Av
		img = img*10**(Al / 2.5)

		cat = pd.read_csv('/home/xkchen/mywork/ICL/data/star_catalog/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g), skiprows = 1)
		ra_s = np.array(cat['ra'])
		dec_s = np.array(cat['dec'])
		mag = np.array(cat['r'])
		R0 = np.array(cat['psffwhm_r'])
		# without radius parameter
		x, y = wcs.all_world2pix(ra_s*U.deg, dec_s*U.deg, 1)
		ia = (x >= 0) & (x <= x_side)
		ib = (y >= 0) & (y <= y_side)
		ie = (mag <= 20)
		ic = ia & ib & ie
		comx = x[ic]
		comy = y[ic]
		comr = 2*1.5/pixel

		R_ph = rad2asec/(Test_model.angular_diameter_distance(z_g).value)
		R_p = R_ph/pixel
		cenx, ceny = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)
		
		plt.figure(figsize = (7, 4.5))
		ax = plt.subplot(111)
		gf = ax.imshow(BEV * 0.86, cmap = 'rainbow', origin = 'lower')
		hsc.circles(cenx, ceny, s = R_p, fc = '', ec = 'b', )
		hsc.circles(cenx, ceny, s = 1.1*R_p, fc = '', ec = 'b', ls = '--')
		ax.set_title(r'$dust \; map_{SFD_{1998}} \; %s \; ra_{%.3f} \; dec_{%.3f} \; z_{%.3f}$' % (band[q], ra_g, dec_g, z_g))
		plt.colorbar(gf, fraction = 0.035, pad = 0.01, label = '$E[B-V]$')
		ax.set_ylim(0, 1489)
		ax.set_xlim(0, 2048)
		plt.subplots_adjust(left = 0.01, right = 0.85)
		plt.savefig('/home/xkchen/mywork/ICL/code/dust_map98_%s_ra%.3f_dec%.3f_z%.3f.png' % (band[q], ra_g, dec_g, z_g), dpi = 600)
		plt.close()
				
		plt.figure(figsize = (7, 4.5))
		ax = plt.subplot(111)
		gf = ax.imshow(bev, cmap = 'rainbow', origin = 'lower')
		hsc.circles(cenx, ceny, s = R_p, fc = '', ec = 'b', )
		hsc.circles(cenx, ceny, s = 1.1*R_p, fc = '', ec = 'b', ls = '--')
		ax.set_title(r'$dust \; map_{SFD_{2011}} \; %s \; ra_{%.3f} \; dec_{%.3f} \; z_{%.3f}$' % (band[q], ra_g, dec_g, z_g))
		plt.colorbar(gf, fraction = 0.035, pad = 0.01, label = '$E[B-V]$')
		ax.set_ylim(0, 1489)
		ax.set_xlim(0, 2048)
		plt.subplots_adjust(left = 0.01, right = 0.85)
		plt.savefig('/home/xkchen/mywork/ICL/code/dust_map11_%s_ra%.3f_dec%.3f_z%.3f.png' % (band[q], ra_g, dec_g, z_g), dpi = 600)
		plt.close()
		
		plt.figure(figsize = (7, 4.5))
		ax = plt.subplot(111)
		gf = ax.imshow(bev - BEV * 0.86, cmap = 'rainbow', origin = 'lower')
		hsc.circles(cenx, ceny, s = R_p, fc = '', ec = 'b', )
		hsc.circles(cenx, ceny, s = 1.1*R_p, fc = '', ec = 'b', ls = '--')
		ax.set_title(r'$dust \; map_{SFD_{2011} - SFD_{1998}} \; %s \; ra_{%.3f} \; dec_{%.3f} \; z_{%.3f}$' % (band[q], ra_g, dec_g, z_g))
		plt.colorbar(gf, fraction = 0.035, pad = 0.01, label = '$E[B-V]$')
		ax.set_ylim(0, 1489)
		ax.set_xlim(0, 2048)
		plt.subplots_adjust(left = 0.01, right = 0.85)
		plt.savefig('/home/xkchen/mywork/ICL/code/dust_map_deviation_%s_band.png' % band[q], dpi = 600)
		plt.close()
		raise

		plt.figure(figsize = (7, 4.5))
		plt.imshow(Al, cmap = 'rainbow', origin = 'lower')
		hsc.circles(cenx, ceny, s = R_p, fc = '', ec = 'b', )
		hsc.circles(cenx, ceny, s = 1.1*R_p, fc = '', ec = 'b', ls = '--')
		plt.colorbar(fraction = 0.035, pad = 0.01, label = '$Al$')
		plt.title(r'$A_{\lambda}/A_{\nu} \; in \; %s \; ra_{%.3f} \; dec_{%.3f} \; z_{%.3f}$' % (band[q], ra_g, dec_g, z_g))
		plt.ylim(0, 1489)
		plt.xlim(0, 2048)
		plt.subplots_adjust(left = 0.01, right = 0.85)
		plt.savefig('/home/xkchen/mywork/ICL/code/extinction_%s_ra%.3f_dec%.3f_z%.3f.png' % (band[q], ra_g, dec_g, z_g), dpi = 600)
		plt.close()
		
		Numb = len(comx)
		mask_B = np.ones((img.shape[0], img.shape[1]), dtype = np.float)
		ox = np.linspace(0,2047,2048)
		oy = np.linspace(0,1488,1489)
		basic_coord = np.array(np.meshgrid(ox,oy))
		for k in range(Numb):
			xc = comx[k]
			yc = comy[k]
			set_r = np.int(np.ceil(1.2 * comr))

			la0 = np.int(xc - set_r)
			la1 = np.int(xc + set_r +1)
			lb0 = np.int(yc - set_r)
			lb1 = np.int(yc + set_r +1)

			idr = np.sqrt((xc - basic_coord[0,:][lb0: lb1, la0: la1])**2 + (yc - basic_coord[1,:][lb0: lb1, la0: la1])**2)/comr
			jx = idr <= 1
			jx = (-1)*jx+1
			mask_B[lb0: lb1, la0: la1] = mask_B[lb0: lb1, la0: la1]*jx

		mirro_B = mask_B *img
		t1 = time.time() - t0
		
		plt.figure()
		plt.imshow(mirro_B, cmap = 'Greys', origin = 'lower', vmin = 1e-3, norm = mpl.colors.LogNorm())
		hsc.circles(comx, comy, s = comr, fc = '', ec = 'r', lw = 1)
		hsc.circles(cenx, ceny, s = R_p, fc = '', ec = 'b', )
		hsc.circles(cenx, ceny, s = 1.1*R_p, fc = '', ec = 'b', ls = '--')
		plt.ylim(0, 1489)
		plt.xlim(0, 2048)
		plt.savefig('/home/xkchen/mywork/ICL/code/sdss_mask_test_band%s.png'%band[q], dpi = 600)
		plt.close()
		
		hdu = fits.PrimaryHDU()
		hdu.data = mirro_B
		hdu.header = head_inf
		hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/mask/B_mask_data_%s_ra%.3f_dec%.3f.fits'%(band[q], ra_g, dec_g),overwrite = True)
		# aslo save the mask_matrix
		hdu = fits.PrimaryHDU()
		hdu.data = mask_B
		hdu.header = head_inf
		hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/mask/B_mask_metrx_ra%.3f_dec%.3f.fits'%(ra_g, dec_g),overwrite = True)
		'''
	raise
	return

def mask_A():
	kb = 2

	t0 = time.time()

	x0 = np.linspace(0, 2047, 2048)
	y0 = np.linspace(0, 1488, 1489)
	img_grid = np.array(np.meshgrid(x0, y0))

	bin_number = 80
	r_star = 2*1.5/pixel #mask star radius
	load = '/home/xkchen/mywork/ICL/data/total_data/sample_02_03/'

	param_A = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A.sex'
	param_A_tal = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A_Tal.sex' # Tal 2011
	param_sky = '/home/xkchen/mywork/ICL/data/SEX/default_sky_mask.sex'

	out_cat = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A.param'

	out_load_A = '/home/xkchen/mywork/ICL/data/SEX/result/mask_A_test.cat'
	out_load_B = '/home/xkchen/mywork/ICL/data/SEX/result/mask_B_test.cat'
	out_load_sky = '/home/xkchen/mywork/ICL/data/SEX/result/mask_sky_test.cat'
	'''
	file = 'frame-%s-ra260.613-dec32.133-redshift0.223.fits.bz2' % band[kb]	
	ra_g = 260.613
	dec_g = 32.133
	z_g = 0.223
	
	file = 'frame-%s-ra36.455-dec-5.896-redshift0.233.fits.bz2' % band[kb]	
	ra_g = 36.455
	dec_g = -5.896
	z_g = 0.233
	'''
	file = 'frame-%s-ra240.829-dec3.279-redshift0.222.fits.bz2' % band[kb]
	ra_g = 240.829
	dec_g = 3.279
	z_g = 0.222

	data_f = fits.open(load+file)
	img = data_f[0].data
	head_inf = data_f[0].header
	wcs = awc.WCS(head_inf)
	x_side = data_f[0].data.shape[1]
	y_side = data_f[0].data.shape[0]

	ra_img, dec_img = wcs.all_pix2world(img_grid[0,:], img_grid[1,:], 1)
	pos = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
	BEV = sfd(pos)
	Av = Rv * BEV
	Al = A_wave(l_wave[kb], Rv) * Av
	img = img*10**(Al / 2.5)

	cx_BCG, cy_BCG = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)
	R_ph = rad2asec/(Test_model.angular_diameter_distance(z_g).value)
	R_p = R_ph/pixel
	if (kb == 0) | (kb == 4):
		# Tal et.al
		combine = np.zeros((1489, 2048), dtype = np.float)
		for q in range(len(band)):
			file_q = 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[q], ra_g, dec_g, z_g)
			data_q = fits.open(load + file_q)
			img_q = data_q[0].data
			combine = combine + img_q
		# combine data
		hdu = fits.PrimaryHDU()
		hdu.data = combine
		hdu.header = head_inf
		hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/mask/' + 'combine_data_ra%.3f_dec%.3f.fits'%(ra_g, dec_g), overwrite = True)
		file_source = '/home/xkchen/mywork/ICL/data/test_data/mask/' + 'combine_data_ra%.3f_dec%.3f.fits'%(ra_g, dec_g)
		cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s'%(param_A_tal, out_load_A, out_cat)
	else:
		file_source = '/home/xkchen/mywork/ICL/data/test_data/' + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits'%(band[kb], ra_g, dec_g, z_g)
		# Zibetti et.al
		dete_thresh = sb_lim[kb] + 10*np.log10((1 + z_g)/(1 + z_ref))
		dete_thresh = '%.3f' % dete_thresh + ',%.2f' % zop[kb]
		dete_min = '10'
		ana_thresh = dete_thresh *1
		cmd = (
			'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s -DETECT_MINAREA %s -DETECT_THRESH %s -ANALYSIS_THRESH %s'
			%(param_A, out_load_A, out_cat, dete_min, dete_thresh, ana_thresh))
	
	print(cmd)
	a = subpro.Popen(cmd, shell = True)
	a.wait()

	source = asc.read(out_load_A)
	Numb = np.array(source['NUMBER'][-1])
	Nz = Numb *1
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

	cat = pd.read_csv('/home/xkchen/mywork/ICL/data/star_catalog/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g), skiprows = 1)
	ra_s = np.array(cat['ra'])
	dec_s = np.array(cat['dec'])
	mag = np.array(cat['r'])
	x_side = img.shape[1]
	y_side = img.shape[0]
	x, y = wcs.all_world2pix(ra_s*U.deg, dec_s*U.deg, 1)
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

	tdr = np.sqrt((cx - cx_BCG)**2 + (cy - cy_BCG)**2)
	dr00 = np.where(tdr == np.min(tdr))[0]

	for k in range(Numb):
		xc = cx[k]
		yc = cy[k]
		set_r = np.int(np.ceil(1.2 * major[k]))

		la0 = np.int(xc - set_r)
		la1 = np.int(xc + set_r +1)
		lb0 = np.int(yc - set_r)
		lb1 = np.int(yc + set_r +1)

		#dcr = np.sqrt((xc - cx_BCG)**2 +(yc - cy_BCG)**2)
		if k == dr00[0] :
			continue
		else:
			lr = major[k]
			sr = minor[k]
			cr = senior[k]
			chi = theta[k]*np.pi/180
			df1 = lr**2 - cr**2*np.cos(chi)**2
			df2 = lr**2 - cr**2*np.sin(chi)**2
			fr = ((basic_coord[0,:][lb0: lb1, la0: la1] - xc)**2*df1 + (basic_coord[1,:][lb0: lb1, la0: la1] - yc)**2*df2
				- cr**2*np.sin(2*chi)*(basic_coord[0,:][lb0: lb1, la0: la1] - xc)*(basic_coord[1,:][lb0: lb1, la0: la1] - yc))
			idr = fr/(lr**2*sr**2)
			jx = idr<=1
			jx = (-1)*jx+1
			mask_A[lb0: lb1, la0: la1] = mask_A[lb0: lb1, la0: la1]*jx
	mirro_A = mask_A *img

	t1 = time.time() - t0
	print('t = ', t1)

	plt.imshow(mirro_A, cmap = 'Greys', origin = 'lower', vmin = 1e-3, norm = mpl.colors.LogNorm())
	hsc.circles(cx_BCG, cy_BCG, s = 20, fc = '', ec = 'r', lw = 1)
	plt.plot(cx[dr00[0]], cy[dr00[0]], c = 'r', marker = 'P')
	hsc.ellipses(cx, cy, w = a, h = b, rot = theta, fc = '', ec = 'r', ls = '--', lw = 0.5)
	hsc.circles(comx, comy, s = comr, fc = '', ec = 'b', ls = '-', lw = 0.5)
	hsc.circles(cx_BCG, cy_BCG, s = R_p, fc = '', ec = 'b', )
	hsc.circles(cx_BCG, cy_BCG, s = 1.1*R_p, fc = '', ec = 'b', ls = '--')
	plt.xlim(0, 2048)
	plt.ylim(0, 1489)
	plt.savefig('find_BCG.png', dpi = 600)
	plt.show()
	raise
	light, R, Ar, erro = light_measure(mirro_A, bin_number, 1, R_p, cx_BCG, cy_BCG, pixel, z_g)
	light = light + mag_add[kb]

	ax = plt.subplot(111)
	ax.plot(Ar, light, 'b-', label = r'$SB_{ccd} \; Zibetti$', alpha = 0.5)
	ax.set_title(r'$SB \; in \; %s \; band \; ra%.3f \; dec%.3f \; z%.3f$' % (band[kb], ra_g, dec_g, z_g))
	ax.legend(loc = 3)
	ax.set_xscale('log')
	ax.text(1e2, 24, s = r'$N_{Z} \, = \, %.0f$' % Nz)
	ax.set_xlabel('$R[arcsec]$')
	ax.set_ylabel('$SB[mag/arcsec^2]$')
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax1 = ax.twiny()
	ax1.plot(R, light, 'b-', alpha = 0.5)
	ax1.set_xscale('log')
	ax1.set_xlabel('$R[kpc]$')
	ax1.tick_params(axis = 'x', which = 'both', direction = 'in')
	ax.invert_yaxis()
	plt.savefig('/home/xkchen/mywork/ICL/code/light_test_%s.png' % band[kb], dpi = 600)
	plt.show()

	plt.imshow(mirro_A, cmap = 'Greys', origin = 'lower', vmin = 1e-3, norm = mpl.colors.LogNorm())
	plt.title(r'$source \; compare \; %s \; ra%.3f \; dec%.3f \; z%.3f$' % (band[kb], ra_g, dec_g, z_g))
	hsc.ellipses(cx, cy, w = a, h = b, rot = theta, fc = '', ec = 'r', ls = '--', lw = 0.5)
	hsc.circles(comx, comy, s = comr, fc = '', ec = 'b', ls = '-', lw = 0.5)
	hsc.circles(cx_BCG, cy_BCG, s = R_p, fc = '', ec = 'b', )
	hsc.circles(cx_BCG, cy_BCG, s = 1.1*R_p, fc = '', ec = 'b', ls = '--')
	hsc.circles(cx_BCG, cy_BCG, s = 20, fc = 'r', ec = 'r', alpha = 0.5)
	plt.xlim(0, 2048)
	plt.ylim(0, 1489)
	plt.savefig('/home/xkchen/mywork/ICL/code/add_mask_AB_%s.png' % band[kb], dpi = 300)
	plt.show()

	hdu = fits.PrimaryHDU()
	hdu.data = mirro_A
	hdu.header = head_inf
	hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/mask/A_mask_data_%s_ra%.3f_dec%.3f.fits'%(band[kb], ra_g, dec_g),overwrite = True)

	hdu = fits.PrimaryHDU()
	hdu.data = mask_A
	hdu.header = head_inf
	hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/mask/A_mask_metrx_ra%.3f_dec%.3f.fits'%(ra_g, dec_g),overwrite = True)

	return

def main():
	mask_wit()
	mask_A()

if __name__ == "__main__":
	main()
