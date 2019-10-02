import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates as mapcd

from matplotlib.patches import Circle, Ellipse
from astropy import cosmology as apcy
from dustmaps.sfd import SFDQuery
from extinction_redden import A_wave
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp2d  as inter2

import h5py
import numpy as np
import pandas as pd
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.ascii as asc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C
import pandas as pds

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Lsun = C.L_sun.value*10**7

Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396 # arcsec
z_ref = 0.250 
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*10**(-23) # ZP : (erg/s)/cm^-2
# sample
with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
    catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]

def mask_B():
	cra = 203.834
	cdec = 41.001
	z_g = 0.228
	load = '/home/xkchen/mywork/ICL/data/total_data/'
	file = 'frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (cra, cdec, z_g)
	data = fits.open(load + file)
	img = data[0].data
	wcs = awc.WCS(data[0].header)
	x_side = data[0].data.shape[1]
	y_side = data[0].data.shape[0]

	R_ph = rad2asec/(Test_model.angular_diameter_distance(z_g).value)
	R_p = R_ph / pixel

	cenx, ceny = wcs.all_world2pix(cra*U.deg, cdec*U.deg, 1)
	'''	
	cat = pd.read_csv('/home/xkchen/mywork/ICL/data/star_catalog/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' %(z_g, cra, cdec) )
	set_ra = np.array(cat['ra'])
	set_dec = np.array(cat['dec'])
	mag = np.array(cat['r'])

	A = np.array(cat['isoA_r'])
	B = np.array(cat['isoB_r'])
	chi = np.array(cat['isoPhi_r'])
	OBJ = np.array(cat['type'])
	xt = cat['Unnamed: 24']
	'''
	cat = pd.read_csv('/home/xkchen/mywork/ICL/data/star_dr12/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, cra, cdec), skiprows = 1)
	"""
	'psffwhm_r', 'petroR90_r', // 'deVRad_r', 'deVAB_r', 'deVPhi_r', // 'expRad_r', 'expAB_r', 'expPhi_r',
	"""
	set_ra = np.array(cat['ra'])
	set_dec = np.array(cat['dec'])
	mag = np.array(cat['r'])
	xt = cat['Column1']
	OBJ = np.array(cat['type'])

	#A = np.array(cat['petroR90_r']) / pixel
	#B = np.array(cat['petroR90_r']) / pixel
	A = np.array(cat['psffwhm_r']) * 2.6 / pixel
	B = np.array(cat['psffwhm_r']) * 2.6 / pixel
	chi = np.zeros(len(A), dtype = np.float)

	#A = np.array(cat['expRad_r']) / pixel
	#B = np.array(cat['expRad_r']) / pixel
	#chi = np.array(cat['expPhi_r'])

	#A = np.array(cat['deVRad_r']) / pixel
	#B = np.array(cat['deVRad_r']) / pixel
	#chi = np.array(cat['deVPhi_r'])

	# bright stars
	x, y = wcs.all_world2pix(set_ra * U.deg, set_dec * U.deg, 1)
	ia = (x >= 0) & (x <= x_side)
	ib = (y >= 0) & (y <= y_side)
	ie = (mag <= 20)
	ig = OBJ == 6
	ic = (ia & ib & ie & ig)
	sub_x0 = x[ic]
	sub_y0 = y[ic]
	sub_A0 = A[ic]
	sub_B0 = B[ic]
	sub_chi0 = chi[ic]

	# saturated objs
	ih = (mag <= 14)
	ikk = (ia & ib & ig & ih)
	sub_x1 = x[ikk]
	sub_y1 = y[ikk]
	sub_A1 = 3 * A[ikk]
	sub_B1 = 3 * B[ikk]
	sub_chi1 = chi[ikk]

	# saturated source(may not stars)
	xa = ['SATURATED' in kk for kk in xt]
	xv = np.array(xa)
	idx = np.where(xv == True)[0]
	sub_x2 = x[idx]
	sub_y2 = y[idx]
	sub_A2 = 3 * A[idx]
	sub_B2 = 3 * B[idx]
	sub_chi2 = chi[idx]

	comx = np.r_[sub_x0[sub_A0 > 0], sub_x2[sub_A2 > 0]]
	comy = np.r_[sub_y0[sub_A0 > 0], sub_y2[sub_A2 > 0]]
	Lr = np.r_[sub_A0[sub_A0 > 0], sub_A2[sub_A2 > 0]]
	Sr = np.r_[sub_B0[sub_A0 > 0], sub_B2[sub_A2 > 0]]
	phi = np.r_[sub_chi0[sub_A0 > 0], sub_chi2[sub_A2 > 0]]
	'''
	plt.figure()
	ax = plt.subplot(111)
	#ax.set_title('source form dr12 ')
	ax.set_title('source form dr7 ')
	ax.imshow(img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, norm = mpl.colors.LogNorm())
	ax.scatter(sub_x2, sub_y2, s = 5, marker = 'o', facecolors = '', edgecolors = 'r', label = '$Saturated$')
	ax.scatter(sub_x0, sub_y0, s = 5, marker = 's', facecolors = '', edgecolors = 'b', label = '$bright \; stars$')
	cluster = Circle(xy = (cenx, ceny), radius = R_p, fill = False, ec = 'b', alpha = 0.5, label = 'cluster region[1Mpc]')
	ax.add_patch(cluster)
	plt.legend(loc = 4)
	plt.xlim(0, img.shape[1])
	plt.ylim(0, img.shape[0])
	#plt.savefig('source_dr12.png', dpi = 300)
	plt.savefig('source_dr7.png', dpi = 300)
	plt.show()
	'''
	Numb = len(comx)
	mask_B = np.ones((img.shape[0], img.shape[1]), dtype = np.float)
	ox = np.linspace(0,2047,2048)
	oy = np.linspace(0,1488,1489)
	basic_coord = np.array(np.meshgrid(ox,oy))

	major = Lr / 2
	minor = Sr / 2
	senior = np.sqrt(major**2 - minor**2)
	## mask B
	for k in range(Numb):
		xc = comx[k]
		yc = comy[k]
		lr = major[k]
		sr = minor[k]
		cr = senior[k]
		theta = phi[k] * np.pi / 180

		set_r = np.int(np.ceil(1.2 * lr))
		la0 = np.max( [np.int(xc - set_r), 0])
		la1 = np.min( [np.int(xc + set_r +1), img.shape[1] - 1] )
		lb0 = np.max( [np.int(yc - set_r), 0] ) 
		lb1 = np.min( [np.int(yc + set_r +1), img.shape[0] - 1] )

		df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.cos(theta) + (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.sin(theta)
		df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.cos(theta) - (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.sin(theta)
		fr = df1**2 / lr**2 + df2**2 / sr**2
		jx = fr <= 1

		iu = np.where(jx == True)
		iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float)
		iv[iu] = np.nan
		mask_B[lb0: lb1, la0: la1] = mask_B[lb0: lb1, la0: la1] * iv
	mirro_B = mask_B * img

	fig = plt.figure()

	#fig.suptitle('mask B with isoRad')
	#fig.suptitle('mask B with petroR90')
	fig.suptitle('mask B with FWHM')
	#fig.suptitle('mask B with expRad')
	#fig.suptitle('mask B with deVRad')

	ax = plt.subplot(111)
	ax.imshow(mirro_B, cmap = 'Greys', origin = 'lower', vmin = 1e-5, norm = mpl.colors.LogNorm())
	cluster = Circle(xy = (cenx, ceny), radius = R_p, fill = False, ec = 'r', alpha = 0.5)
	ax.add_patch(cluster)
	ax.set_xlim(0, img.shape[1])
	ax.set_ylim(0, img.shape[0])

	#plt.savefig('mask_B_with_isoRad.png', dpi = 300)
	#plt.savefig('mask_B_with_petroR90.png', dpi = 300)
	plt.savefig('mask_B_with_PSF.png', dpi = 300)
	#plt.savefig('mask_B_with_expRad.png', dpi = 300)
	#plt.savefig('mask_B_with_deVRad.png', dpi = 300)

	plt.show()

	raise
	return

def mask_test():
	for qq in range(10):
		cra = 38.382
		cdec = 1.911
		z_g = 0.243
		load = '/home/xkchen/mywork/ICL/data/total_data/'
		file = 'frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (cra, cdec, z_g)
		data = fits.open(load + file)
		img = data[0].data
		wcs = awc.WCS(data[0].header)
		x_side = data[0].data.shape[1]
		y_side = data[0].data.shape[0]

		R_ph = rad2asec/(Test_model.angular_diameter_distance(z_g).value)
		R_p = R_ph / pixel
		cenx, ceny = wcs.all_world2pix(cra*U.deg, cdec*U.deg, 1)

		cat = pd.read_csv('/home/xkchen/mywork/ICL/data/star_dr12/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, cra, cdec), skiprows = 1)
		"""
		'psffwhm_r', 'petroR90_r', // 'deVRad_r', 'deVAB_r', 'deVPhi_r', // 'expRad_r', 'expAB_r', 'expPhi_r',
		"""
		set_ra = np.array(cat['ra'])
		set_dec = np.array(cat['dec'])
		mag = np.array(cat['r'])
		xt = cat['Column1']
		OBJ = np.array(cat['type'])
		'''
		DECAY = 10000
		tau = np.sqrt(np.log(DECAY) / np.log(2))
		print('tau = ', tau)
		'''
		tau = 6
		A = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i']]) * tau / pixel
		B = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i']]) * tau / pixel
		chi = np.zeros(A.shape[1], dtype = np.float)

		lln = np.array([len(A[:,ll][A[:,ll] > 0 ]) for ll in range(A.shape[1]) ])
		lr_iso = np.array([np.max(A[:,ll]) for ll in range(A.shape[1]) ])
		sr_iso = np.array([np.max(B[:,ll]) for ll in range(B.shape[1]) ])

		# bright stars
		x, y = wcs.all_world2pix(set_ra * U.deg, set_dec * U.deg, 1)
		ia = (x >= 0) & (x <= x_side)
		ib = (y >= 0) & (y <= y_side)
		ie = (mag <= 20)
		ig = OBJ == 6
		iq = lln >= 2
		ic = (ia & ib & ie & ig & iq)
		sub_x0 = x[ic]
		sub_y0 = y[ic]
		sub_A0 = lr_iso[ic]
		sub_B0 = sr_iso[ic]
		sub_chi0 = chi[ic]

		# saturated objs
		ih = (mag <= 14)
		ikk = (ia & ib & ig & ih)
		sub_x1 = x[ikk]
		sub_y1 = y[ikk]
		sub_A1 = 3 * A[ikk]
		sub_B1 = 3 * B[ikk]
		sub_chi1 = chi[ikk]

		# saturated source(may not stars)
		xa = ['SATURATED' in kk for kk in xt]
		xv = np.array(xa)
		idx = np.where(xv == True)[0]
		sub_x2 = x[idx]
		sub_y2 = y[idx]
		sub_A2 = 3 * A[idx]
		sub_B2 = 3 * B[idx]
		sub_chi2 = chi[idx]

		comx = np.r_[sub_x0[sub_A0 > 0], sub_x2[sub_A2 > 0]]
		comy = np.r_[sub_y0[sub_A0 > 0], sub_y2[sub_A2 > 0]]
		Lr = np.r_[sub_A0[sub_A0 > 0], sub_A2[sub_A2 > 0]]
		Sr = np.r_[sub_B0[sub_A0 > 0], sub_B2[sub_A2 > 0]]
		phi = np.r_[sub_chi0[sub_A0 > 0], sub_chi2[sub_A2 > 0]]
		Numb = len(comx)
		mask_B = np.ones((img.shape[0], img.shape[1]), dtype = np.float)
		ox = np.linspace(0,2047,2048)
		oy = np.linspace(0,1488,1489)
		basic_coord = np.array(np.meshgrid(ox,oy))

		major = Lr / 2
		minor = Sr / 2
		senior = np.sqrt(major**2 - minor**2)
		## mask B
		for k in range(Numb):
			xc = comx[k]
			yc = comy[k]
			lr = major[k]
			sr = minor[k]
			cr = senior[k]
			theta = phi[k] * np.pi / 180

			set_r = np.int(np.ceil(1.2 * lr))
			la0 = np.max( [np.int(xc - set_r), 0])
			la1 = np.min( [np.int(xc + set_r +1), img.shape[1] - 1] )
			lb0 = np.max( [np.int(yc - set_r), 0] ) 
			lb1 = np.min( [np.int(yc + set_r +1), img.shape[0] - 1] )

			df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.cos(theta) + (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.sin(theta)
			df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.cos(theta) - (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.sin(theta)
			fr = df1**2 / lr**2 + df2**2 / sr**2
			jx = fr <= 1

			iu = np.where(jx == True)
			iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float)
			iv[iu] = np.nan
			mask_B[lb0: lb1, la0: la1] = mask_B[lb0: lb1, la0: la1] * iv
		mirro_B = mask_B * img

		fig = plt.figure(figsize = (16, 8))
		fig.suptitle('mask B [%.2f FWHM]' % tau)
		ax0 = plt.subplot(121)
		ax1 = plt.subplot(122)
		ax0.imshow(img, cmap = 'Greys', vmin = 1e-5, vmax = 1e2, origin = 'lower', norm = mpl.colors.LogNorm())
		ax0.set_xlim(0, img.shape[1])
		ax0.set_ylim(0, img.shape[0])
		ax1.imshow(mirro_B, cmap = 'Greys', vmin = 1e-5, vmax = 1e2, origin = 'lower', norm = mpl.colors.LogNorm())
		ax1.set_xlim(0, img.shape[1])
		ax1.set_ylim(0, img.shape[0])
		plt.savefig('fwhm_mask_%d.png' % qq, dpi = 300)
		plt.close()
	raise

	return

def add_sky():
	load = '/home/xkchen/mywork/ICL/query/'
	file = 'frame-r-ra203.834-dec41.001-z0.228.fits'
	data = fits.open(load + file)
	img = data[0].data
	## add sky 
	sky0 = data[2].data['ALLSKY'][0]
	eta = (img.shape[0] * img.shape[1]) / (sky0.shape[0] * sky0.shape[1])
	sky_x = data[2].data['XINTERP'][0]
	sky_y = data[2].data['YINTERP'][0]
	x0 = np.linspace(0, sky0.shape[1] - 1, sky0.shape[1])
	y0 = np.linspace(0, sky0.shape[0] - 1, sky0.shape[0])
	f_sky = inter2(x0, y0, sky0, kind = 'linear')
	New_sky = f_sky(sky_x, sky_y)
	sky_bl = New_sky * data[0].header['NMGY'] / eta
	cimg = img + sky_bl

	hdu = fits.PrimaryHDU()
	hdu.data = cimg
	hdu.header = data[0].header
	hdu.writeto(load + 'cimg-r-ra203.834-dec41.001-z0.228.fits', overwrite = True)

	inds = np.array(np.meshgrid(sky_x, sky_y))
	t_sky = mapcd(sky0, [inds[1,:], inds[0,:]], order = 1, mode = 'nearest')

	plt.figure(figsize = (16, 8))
	ax0 = plt.subplot(221)
	ax1 = plt.subplot(222)
	ax2 = plt.subplot(223)
	ax3 = plt.subplot(224)

	ax0.set_title('interp2d')
	g0 = ax0.imshow(New_sky, origin = 'lower')
	plt.colorbar(g0, ax = ax0, fraction = 0.035, pad = 0.01, label = 'counts')

	ax1.set_title('map_coordinates')
	g1 = ax1.imshow(t_sky, origin = 'lower')
	plt.colorbar(g1, ax = ax1, fraction = 0.035, pad = 0.01, label = 'counts')

	ax2.set_title('interp2d - map_coordinates / interp2d')
	g2 = ax2.imshow((New_sky - t_sky) / New_sky, origin = 'lower')
	plt.colorbar(g2, ax = ax2, fraction = 0.035, pad = 0.01)

	ax3.set_title('sky background')
	g3 = ax3.imshow(sky_bl, origin = 'lower')
	plt.colorbar(g3, ax = ax3, fraction = 0.035, pad = 0.01, label = '$flux[nmaggy]$')

	plt.tight_layout()
	plt.savefig('sky_information.png', dpi = 300)
	plt.show()
	raise
	return

def source_detect():

	load = '/home/xkchen/mywork/ICL/query/'
	file = 'frame-r-ra203.834-dec41.001-z0.228.fits'
	tmp_load = '/home/xkchen/mywork/ICL/data/test_data/'

	data0 = fits.open(load + file)
	img0 = data0[0].data
	wcs = awc.WCS(data0[0].header)
	x_side = data0[0].data.shape[1]
	y_side = data0[0].data.shape[0]

	R_ph = rad2asec/(Test_model.angular_diameter_distance(0.228).value)
	R_p = R_ph / pixel
	cra = 203.834
	cdec = 41.001
	cenx, ceny = wcs.all_world2pix(cra*U.deg, cdec*U.deg, 0)

	sky = data0[2].data['ALLSKY'][0]
	eta = (1489 * 2048) / (sky.shape[0] * sky.shape[1])
	sky_bl = np.mean(sky * data0[0].header['NMGY']) / eta

	data1 = fits.open(load + 'cimg-r-ra203.834-dec41.001-z0.228.fits')
	cimg = data1[0].data
	#1 source from calibrated img
	param_A = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A.sex'
	out_cat = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A.param'
	out_load_A = '/home/xkchen/mywork/ICL/data/SEX/result/mask_A_test.cat'

	file_source = load + file
	cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s -DETECT_MINAREA %s'%(param_A, out_load_A, out_cat, '5')
	print(cmd)
	a = subpro.Popen(cmd, shell = True)
	a.wait()

	Kron = 6
	source0 = asc.read(out_load_A)
	Numb0 = np.array(source0['NUMBER'][-1])
	A0 = np.array(source0['A_IMAGE'])
	B0 = np.array(source0['B_IMAGE'])
	chi0 = np.array(source0['THETA_IMAGE'])
	cx0 = np.array(source0['X_IMAGE']) - 1
	cy0 = np.array(source0['Y_IMAGE']) - 1
	Lr0 = Kron * A0
	Sr0 = Kron * B0
	#1 source from sky added img
	file_source = load + 'cimg-r-ra203.834-dec41.001-z0.228.fits'
	cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s -DETECT_MINAREA %s'%(param_A, out_load_A, out_cat, '5')
	print(cmd)
	a = subpro.Popen(cmd, shell = True)
	a.wait()
	source1 = asc.read(out_load_A)
	Numb1 = np.array(source1['NUMBER'][-1])
	A1 = np.array(source1['A_IMAGE'])
	B1 = np.array(source1['B_IMAGE'])
	chi1 = np.array(source1['THETA_IMAGE'])
	cx1 = np.array(source1['X_IMAGE']) - 1
	cy1 = np.array(source1['Y_IMAGE']) - 1
	Lr1 = Kron * A1
	Sr1 = Kron * B1

	fig = plt.figure(figsize = (16, 8))
	fig.suptitle('comparison of source detection')
	cluster0 = Circle(xy = (cenx, ceny), radius = R_p, fill = False, ec = 'b', alpha = 0.5, label = 'cluster region[1Mpc]')
	cluster1 = Circle(xy = (cenx, ceny), radius = R_p, fill = False, ec = 'b', alpha = 0.5, label = 'cluster region[1Mpc]')
	ax0 = plt.subplot(121)
	ax1 = plt.subplot(122)

	ax0.imshow(img0, cmap = 'Greys', vmin = 1e-5, origin = 'lower', norm = mpl.colors.LogNorm())
	for kk in range(Numb0):
		clco = Ellipse(xy = (cx0[kk], cy0[kk]), width = Lr0[kk], height = Sr0[kk], angle = chi0[kk], fill = False, ec = 'r', alpha = 0.5)
		ax0.add_patch(clco)
	ax0.add_patch(cluster0)
	ax0.set_title('source from calibrated image')
	ax0.set_xlim(0, img0.shape[1])
	ax0.set_ylim(0, img0.shape[0])

	ax1.imshow(cimg, cmap = 'Greys', vmin = 1e-5, origin = 'lower', norm = mpl.colors.LogNorm())
	for kk in range(Numb1):
		clco = Ellipse(xy = (cx1[kk], cy1[kk]), width = Lr1[kk], height = Sr1[kk], angle = chi1[kk], fill = False, ec = 'r', alpha = 0.5)
		ax1.add_patch(clco)
	ax1.add_patch(cluster1)
	ax1.set_title('source from sky-added image')
	ax1.set_xlim(0, cimg.shape[1])
	ax1.set_ylim(0, cimg.shape[0])	

	plt.tight_layout()
	plt.savefig('source_detect_compare.png', dpi = 300)
	plt.close()

	fig = plt.figure(figsize = (16, 8))
	fig.suptitle('image comparison')
	cluster0 = Circle(xy = (cenx, ceny), radius = R_p, fill = False, ec = 'b', alpha = 0.5, label = 'cluster region[1Mpc]')
	cluster1 = Circle(xy = (cenx, ceny), radius = R_p, fill = False, ec = 'b', alpha = 0.5, label = 'cluster region[1Mpc]')
	ax0 = plt.subplot(121)
	ax1 = plt.subplot(122)

	tf = ax0.imshow(img0, cmap = 'Greys', vmin = 1e-5, origin = 'lower', norm = mpl.colors.LogNorm())
	plt.colorbar(tf, ax = ax0, orientation = 'horizontal', fraction = 0.05, pad = 0.01, label = '$flux[nmaggy]$')
	ax0.add_patch(cluster0)
	ax0.set_title('calibrated image')
	ax0.set_xlim(0, cimg.shape[1])
	ax0.set_ylim(0, cimg.shape[0])	
	ax0.set_xticks([])

	tf = ax1.imshow(cimg, cmap = 'Greys', vmin = 1e-5, origin = 'lower', norm = mpl.colors.LogNorm())
	plt.colorbar(tf, ax = ax1, orientation = 'horizontal', fraction = 0.05, pad = 0.01, label = '$flux[nmaggy]$')
	ax1.add_patch(cluster1)
	ax1.set_title('sky-added image')
	ax1.set_xlim(0, cimg.shape[1])
	ax1.set_ylim(0, cimg.shape[0])	
	ax1.set_xticks([])

	plt.tight_layout()
	plt.savefig('image_compare.png', dpi = 300)
	plt.close()

	a0 = np.max([0, cenx - 0.8 * R_p])
	a1 = np.min([cenx + 0.8 * R_p, 2048])
	b0 = np.max([0, ceny - 0.8 * R_p])
	b1 = np.min([ceny + 0.8 * R_p, 1489])
	plt.figure(figsize = (16, 8))
	ax0 = plt.subplot(121)
	ax1 = plt.subplot(122)
	tf = ax0.imshow(img0, cmap = 'Greys', vmin = 1e-5, origin = 'lower', norm = mpl.colors.LogNorm())
	plt.colorbar(tf, ax = ax0, fraction = 0.045, pad = 0.01, label = '$flux[nmaggy]$')
	ax0.set_title('calibrated image')
	ax0.set_xlim(a0, a1)
	ax0.set_ylim(b0, b1)

	tf = ax1.imshow(cimg, cmap = 'Greys', vmin = 1e-5, origin = 'lower', norm = mpl.colors.LogNorm())
	plt.colorbar(tf, ax = ax1, fraction = 0.045, pad = 0.01, label = '$flux[nmaggy]$')
	ax1.set_title('sky-added image')
	ax1.set_xlim(a0, a1)
	ax1.set_ylim(b0, b1)

	plt.tight_layout()
	plt.savefig('cluster_compare.png', dpi = 300)
	plt.close()

	raise
	return

def data_select():
	lod = '/home/xkchen/mywork/ICL/data/redmapper/'
	csv_1 = 'Except_r_sample.csv'
	csv_2 = 'Except_g_sample.csv'
	csv_3 = 'Except_i_sample.csv'
	csv_4 = 'Except_z_sample.csv'
	csv_5 = 'Except_u_sample.csv'

	except_ra_r = ['128.611', '133.229', '177.809', '183.580', '221.403', '242.344', '130.537',
					'32.906', '38.382', '168.057', '202.923', '212.342', '237.591', '180.927']
	except_dec_r = ['36.515', '18.196', '33.577', '32.835', '8.243', '36.393', '38.226', '0.387',
					'1.911', '15.760', '13.325', '39.952', '2.756', '1.032']
	except_z_r = ['0.289', '0.257', '0.212', '0.226', '0.268', '0.281', '0.244', '0.295', '0.243',
					'0.275', '0.242', '0.267', '0.240', '0.255']
	x_ra = np.array( [ np.float(ll) for ll in except_ra_r] )
	x_dec = np.array( [ np.float(ll) for ll in except_dec_r] )
	x_z = np.array( [np.float(ll) for ll in except_z_r] )
	keys = ['ra', 'dec', 'z']
	values = [x_ra, x_dec, x_z]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(lod + csv_1)

	except_ra_g = ['168.215', '8.320', '128.611', '133.229', '177.809', '242.344', '351.680',
					'35.888', '180.927', '186.666', '130.537', '34.675', '121.566', '124.701',
					'172.033', '188.950', '215.244', '227.367']
	except_dec_g = ['56.464', '-0.633', '36.515', '18.196', '33.577', '36.393', '1.134', '-7.228',
					'1.032', '3.383', '38.226', '0.114', '19.247', '4.898', '41.322', '15.556', 
					'0.229', '0.899']
	except_z_g = ['0.227', '0.261', '0.289', '0.257', '0.212', '0.281', '0.277', '0.279', '0.255',
					'0.226', '0.244', '0.272', '0.285', '0.252', '0.288', '0.285', '0.277', '0.263']
	x_ra = np.array( [ np.float(ll) for ll in except_ra_g] )
	x_dec = np.array( [ np.float(ll) for ll in except_dec_g] )
	x_z = np.array( [np.float(ll) for ll in except_z_g] )
	keys = ['ra', 'dec', 'z']
	values = [x_ra, x_dec, x_z]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(lod + csv_2)

	except_ra_i = ['5.200', '8.320', '10.708', '21.224', '24.320', '37.108', '37.662', '117.410',
					'117.585', '126.251', '129.462', '141.919', '143.950', '148.541', '162.793',
					'174.080', '179.024', '180.927', '183.580', '186.372', '130.537', '190.946',
					'196.951', '214.913', '218.217', '242.344', '330.691']
	except_dec_i = ['1.585', '-0.633', '0.215', '2.508', '7.882', '1.503', '-4.991', '27.287',
					'26.293', '4.430', '6.777', '4.352', '12.151', '36.054', '9.066', '50.425',
					'42.024', '1.032', '32.835', '0.726', '38.226', '51.831', '43.153', '43.195',
					'30.489', '36.393', '-8.544']
	except_z_i = ['0.208', '0.261', '0.269', '0.217', '0.258', '0.264', '0.292', '0.243', '0.205',
					'0.224', '0.236', '0.275', '0.255', '0.291', '0.221', '0.286', '0.245', '0.255',
					'0.226', '0.238', '0.244', '0.268', '0.210', '0.219', '0.266', '0.281', '0.210']
	x_ra = np.array( [ np.float(ll) for ll in except_ra_i] )
	x_dec = np.array( [ np.float(ll) for ll in except_dec_i] )
	x_z = np.array( [np.float(ll) for ll in except_z_i] )
	keys = ['ra', 'dec', 'z']
	values = [x_ra, x_dec, x_z]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(lod + csv_3)

	except_ra_z = ['24.320', '126.251', '130.537', '140.078', '148.541', '153.281', '186.372', '190.640',
					'192.743', '196.951', '222.069', '228.094', '242.344', '334.357', '346.984']
	except_dec_z = ['7.882', '4.430', '38.226', '37.105', '36.054', '17.932', '0.726', '35.537', '9.056',
					'43.153', '13.622', '59.217', '36.393', '27.836', '15.373']
	except_z_z = ['0.258', '0.224', '0.244', '0.235', '0.291', '0.261', '0.238', '0.269', '0.298', '0.210',
					'0.226', '0.295', '0.281', '0.298', '0.249']
	x_ra = np.array( [ np.float(ll) for ll in except_ra_z] )
	x_dec = np.array( [ np.float(ll) for ll in except_dec_z] )
	x_z = np.array( [np.float(ll) for ll in except_z_z] )
	keys = ['ra', 'dec', 'z']
	values = [x_ra, x_dec, x_z]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(lod + csv_4)

	except_ra_u = ['35.888', '130.537', '140.296', '147.783', '155.783', '189.004', '204.525', '208.380',
					'219.787', '242.344', '244.422', '327.180', '334.357', '346.734']
	except_dec_u = ['-7.228', '38.226', '34.860', '43.745', '7.108', '53.394', '5.434', '53.575', '29.039',
					'36.393', '42.539', '7.024', '27.836', '15.375']
	except_z_u = ['0.279', '0.244', '0.238', '0.287', '0.290', '0.275', '0.237', '0.255', '0.251', '0.281',
					'0.294', '0.282', '0.298', '0.220']
	x_ra = np.array( [ np.float(ll) for ll in except_ra_u] )
	x_dec = np.array( [ np.float(ll) for ll in except_dec_u] )
	x_z = np.array( [np.float(ll) for ll in except_z_u] )
	keys = ['ra', 'dec', 'z']
	values = [x_ra, x_dec, x_z]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(lod + csv_5)

	# dr7 select [mainly include "over-masking" part]
	except_ra = ['1.980', '6.566', '29.162', '34.432', '119.661', '129.197', '142.096', '162.870', 
				'167.538', '180.610', '247.334', '337.717', '180.211', '242.027']
	except_dec = ['-8.845', '1.221', '0.842', '-9.105', '13.152', '15.150', '18.003', '17.351', 
				'8.920', '10.562', '47.148', '-0.538', '-1.465', '25.346']
	except_z = ['0.260', '0.292', '0.217', '0.239', '0.230', '0.279', '0.291', '0.202', '0.225', 
				'0.229', '0.211', '0.282', '0.267', '0.270']
	x_ra = np.array( [ np.float(ll) for ll in except_ra] )
	x_dec = np.array( [ np.float(ll) for ll in except_dec] )
	x_z = np.array( [np.float(ll) for ll in except_z] )
	keys = ['ra', 'dec', 'z']
	values = [x_ra, x_dec, x_z]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(lod + 'Bad_match_dr7_cat.csv')

	raise
	return

def main():
	# mask_B()
	# mask_test()
	add_sky()
	# source_detect()
	# data_select()
if __name__ == "__main__":
	main()
