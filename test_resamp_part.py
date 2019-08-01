import matplotlib as mpl
import handy.scatter as hsc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
import pandas as pd
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

import subprocess as subpro
import astropy.units as U
import astropy.constants as C

import scipy.stats as sts
from dustmaps.sfd import SFDQuery
from scipy.optimize import curve_fit
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d as interp

from resamp import gen
from extinction_redden import A_wave
from light_measure import sigmamc
from light_measure import light_measure, flux_recal, weit_l_measure, flux_scale

import time
import sfdmap
# constant
m = sfdmap.SFDMap('/home/xkchen/mywork/ICL/data/redmapper/sfddata_maskin', scaling = 0.86)
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

with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
	catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]

R0 = 1 # Mpc
pixel = 0.396
z_ref = 0.25
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Angu_ref = (R0/Da_ref)*rad2asec
Rpp = Angu_ref/pixel

# dust correct
Rv = 3.1
sfd = SFDQuery()
band = ['u', 'g', 'r', 'i', 'z']
sum_band = ['r', 'i', 'z']
l_wave = np.array([3551, 4686, 6166, 7480, 8932])
mag_add = np.array([-0.04, 0, 0, 0, 0.02])
zopt = np.array([22.46, 22.5, 22.5, 22.5, 22.52])
sb_lim = np.array([24.35, 25, 24.5, 24, 22.9])

stack_N = 10
def source_compa():
	bins = 90
	kd = -20

	zg = z[kd]
	rag = ra[kd]
	decg = dec[kd]
	x0 = np.linspace(0, 2047, 2048)
	y0 = np.linspace(0, 1488, 1489)
	grd = np.array(np.meshgrid(x0, y0))

	param_A = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A.sex'
	out_cat = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A.param'
	out_load_A = '/home/xkchen/mywork/ICL/data/SEX/result/mask_A_test.cat'

	tmp_load = '/home/xkchen/mywork/ICL/data/test_data/'
	load = '/home/xkchen/mywork/ICL/data/total_data/sample_02_03/'

	file = 'frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (rag, decg, zg)
	data = fits.open(load + file)
	img = data[0].data
	Head = data[0].header
	wcs = awc.WCS(Head)
	Da = Test_model.angular_diameter_distance(zg).value
	Ar = rad2asec/Da
	Rp = Ar/pixel
	cx_BCG, cy_BCG = wcs.all_world2pix(rag*U.deg, decg*U.deg, 1)

	ra_img, dec_img = wcs.all_pix2world(grd[0,:], grd[1,:], 1)
	pos = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
	EBV = sfd(pos)
	Av = Rv * EBV * 0.86
	Al = A_wave(l_wave[2], Rv) * Av
	img1 = img*10**(Al / 2.5)
	## threshold 1.3sigma
	hdu = fits.PrimaryHDU()
	hdu.data = img1
	hdu.header = Head
	hdu.writeto(tmp_load + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % ('r', rag, decg, zg), overwrite = True)

	file_source = tmp_load + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % ('r', rag, decg, zg)
	cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s -DETECT_THRESH %s -ANALYSIS_THRESH %s'%(param_A, out_load_A, out_cat, '1.5', '1.5')
	print(cmd)
	a = subpro.Popen(cmd, shell = True)
	a.wait()

	source = asc.read(out_load_A)
	Numb = np.array(source['NUMBER'][-1])
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])
	chi1 = np.array(source['THETA_IMAGE'])
	cx1 = np.array(source['X_IMAGE']) - 1
	cy1 = np.array(source['Y_IMAGE']) - 1

	Kron = 5
	Lr1 = Kron*A
	Sr1 = Kron*B

	file_source = tmp_load + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % ('r', rag, decg, zg)
	cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s -DETECT_THRESH %s -ANALYSIS_THRESH %s'%(param_A, out_load_A, out_cat, '1.3', '1.3')
	print(cmd)
	a = subpro.Popen(cmd, shell = True)
	a.wait()

	source = asc.read(out_load_A)
	Numb = np.array(source['NUMBER'][-1])
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])
	chi2 = np.array(source['THETA_IMAGE'])
	cx2 = np.array(source['X_IMAGE']) - 1
	cy2 = np.array(source['Y_IMAGE']) - 1

	Kron = 7
	Lr2 = Kron*A
	Sr2 = Kron*B

	a0 = np.max([cx_BCG - 1.1*Rp, 0])
	a1 = np.min([cx_BCG + 1.1*Rp, 2047])
	b0 = np.max([cy_BCG - 1.1*Rp, 0])
	b1 = np.min([cy_BCG + 1.1*Rp, 1488])

	plt.figure()
	plt.title(r'$Cluster \; ra%.3f \; dec%.3f \; z%.3f$' % (rag, decg, zg))
	plt.imshow(img1, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
	hsc.circles(cx_BCG, cy_BCG, s = Rp, fc = '', ec ='b', alpha = 0.5)
	hsc.ellipses(cx1, cy1, w = Lr1, h = Sr1, rot = chi1, fc = '', ec = 'g', ls = '-', linewidth = 0.5, label = r'$1.5 \sigma$', alpha = 0.5)
	hsc.ellipses(cx2, cy2, w = Lr2, h = Sr2, rot = chi2, fc = '', ec = 'r', ls = '-', linewidth = 0.5, label = r'$1.3\sigma$', alpha = 0.5)
	plt.xlim(a0, a1)
	plt.ylim(b0, b1)
	plt.savefig('/home/xkchen/mywork/ICL/code/source_C_ra%.3f_dec%.3f.png' % (rag, decg), dpi = 600)
	plt.show()

	raise
	return

def color():
	z_ref = 0.25
	Da_ref = Test_model.angular_diameter_distance(z_ref).value
	Angu_ref = (R0/Da_ref)*rad2asec
	Rpp = Angu_ref/pixel

	bins = 90
	bint = 90

	kd = 0 
	#kd = 19

	x0 = np.linspace(0, 2047, 2048)
	y0 = np.linspace(0, 1488, 1489)
	grd = np.array(np.meshgrid(x0, y0))
	r_star = 2*1.5/pixel

	zg = z[kd]
	rag = ra[kd]
	decg = dec[kd]

	load = '/home/xkchen/mywork/ICL/data/total_data/sample_02_03/'

	file1 = 'frame-g-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (rag, decg, zg)
	data = fits.open(load + file1)
	img01 = data[0].data
	Head = data[0].header
	wcs = awc.WCS(Head)
	Da = Test_model.angular_diameter_distance(zg).value
	Ar = rad2asec/Da
	Rp = Ar/pixel
	cx_BCG, cy_BCG = wcs.all_world2pix(rag*U.deg, decg*U.deg, 1)
	SB01, R01, Anr01, err01 = light_measure(img01, bins, 1, Rp, cx_BCG, cy_BCG, pixel, zg)

	file2 = 'frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (rag, decg, zg)
	data = fits.open(load + file2)
	img02 = data[0].data
	SB02, R02, Anr02, err02 = light_measure(img02, bins, 1, Rp, cx_BCG, cy_BCG, pixel, zg)

	## redden
	ra_img, dec_img = wcs.all_pix2world(grd[0,:], grd[1,:], 1)
	pos = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
	EBV = sfd(pos)
	Av = Rv * EBV * 0.86
	Al1 = A_wave(l_wave[1], Rv) * Av
	img1 = img01*10**(Al1 / 2.5)
	SB1, R1, Anr1, err1 = light_measure(img1, bins, 1, Rp, cx_BCG, cy_BCG, pixel, zg)

	Al2 = A_wave(l_wave[2], Rv) * Av
	img2 = img02*10**(Al2 / 2.5)
	SB2, R2, Anr2, err2 = light_measure(img2, bins, 1, Rp, cx_BCG, cy_BCG, pixel, zg)

	DM1 = SB01 - SB02
	DM2 = SB1 - SB2
	plt.figure(figsize = (16, 9))
	gs = gridspec.GridSpec(2,1, height_ratios = [4,1])
	ax = plt.subplot(gs[0])
	bx = plt.subplot(gs[1])

	ax.plot(R01, DM1, 'g-', label = r'$g-r_{before \, correct}$', alpha = 0.5)
	ax.plot(R1, DM2, 'r-', label = r'$g-r_{after \, correct}$', alpha = 0.5)
	ax.set_xlabel('$R[arcsec]$')
	ax.set_ylabel('$g-r$')
	ax.set_xscale('log')
	ax.legend(loc = 1)

	bx.plot(R01, DM1 - DM2, 'b-', label = r'$\Delta_{c_{before} - c_{after}}$')
	bx.set_xlabel('$R[arcsec]$')
	bx.set_xscale('log')
	bx.set_ylabel(r'$\delta_{g-r}$')
	bx.set_yscale('log')
	bx.legend(loc = 1, fontsize = 15)
	plt.savefig('/home/xkchen/mywork/ICL/code/color_test.png', dpi = 300)
	plt.close()

	raise
	return

def resamp_test():
	for kd in range(10):

		z_ref = 0.25
		Da_ref = Test_model.angular_diameter_distance(z_ref).value
		Angu_ref = (R0/Da_ref)*rad2asec
		Rpp = Angu_ref/pixel

		bins = 50

		x0 = np.linspace(0, 2047, 2048)
		y0 = np.linspace(0, 1488, 1489)
		grd = np.array(np.meshgrid(x0, y0))
		r_star = 2*1.5/pixel

		zg = z[kd]
		rag = ra[kd]
		decg = dec[kd]

		param_A = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A.sex'
		out_cat = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A.param'
		out_load_A = '/home/xkchen/mywork/ICL/data/SEX/result/mask_A_test.cat'

		tmp_load = '/home/xkchen/mywork/ICL/data/test_data/'
		load = '/home/xkchen/mywork/ICL/data/total_data/sample_02_03/'

		file = 'frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (rag, decg, zg)
		data = fits.open(load + file)
		img = data[0].data
		Head = data[0].header
		wcs = awc.WCS(Head)
		Da = Test_model.angular_diameter_distance(zg).value
		Ar = rad2asec/Da
		Rp = Ar/pixel
		cx_BCG, cy_BCG = wcs.all_world2pix(rag*U.deg, decg*U.deg, 1)

		ra_img, dec_img = wcs.all_pix2world(grd[0,:], grd[1,:], 1)
		pos = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
		EBV = sfd(pos)
		Av = Rv * EBV * 0.86
		Al = A_wave(l_wave[2], Rv) * Av
		img1 = img*10**(Al / 2.5)

		## part1: find the source and mask in original image
		hdu = fits.PrimaryHDU()
		hdu.data = img1
		hdu.header = Head
		hdu.writeto(tmp_load + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % ('r', rag, decg, zg), overwrite = True)

		file_source = tmp_load + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % ('r', rag, decg, zg)
		cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s'%(param_A, out_load_A, out_cat)
		print(cmd)
		A = subpro.Popen(cmd, shell = True)
		A.wait()

		source = asc.read(out_load_A)
		Numb = np.array(source['NUMBER'][-1])
		A = np.array(source['A_IMAGE'])
		B = np.array(source['B_IMAGE'])
		chi = np.array(source['THETA_IMAGE'])
		cx = np.array(source['X_IMAGE']) - 1
		cy = np.array(source['Y_IMAGE']) - 1

		Kron = 6
		Lr = Kron*A
		Sr = Kron*B

		cat = pd.read_csv('/home/xkchen/mywork/ICL/data/star_catalog/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (zg, rag, decg), skiprows = 1)
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

		CX = np.r_[cx, comx]
		CY = np.r_[cy, comy]
		a = np.r_[Lr, 2*comr]
		b = np.r_[Sr, 2*comr]
		theta = np.r_[chi, com_chi]
		Numb = Numb + len(comx)

		mask_A = np.ones((img.shape[0], img.shape[1]), dtype = np.float)
		ox = np.linspace(0, img.shape[1]-1, img.shape[1])
		oy = np.linspace(0, img.shape[0]-1, img.shape[0])
		basic_coord = np.array(np.meshgrid(ox,oy))
		major = a/2
		minor = b/2
		senior = np.sqrt(major**2 - minor**2)

		tdr = np.sqrt((CX - cx_BCG)**2 + (CY - cy_BCG)**2)
		dr00 = np.where(tdr == np.min(tdr))[0]
		for k in range(Numb):
			xc = CX[k]
			yc = CY[k]
			lr = major[k]
			sr = minor[k]
			cr = senior[k]

			set_r = np.int(np.ceil(1.2 * lr))
			la0 = np.int(xc - set_r)
			la1 = np.int(xc + set_r +1)
			lb0 = np.int(yc - set_r)
			lb1 = np.int(yc + set_r +1)

			if k == dr00[0] :
				continue
			else:
				phi = theta[k]*np.pi/180
				df1 = lr**2 - cr**2*np.cos(phi)**2
				df2 = lr**2 - cr**2*np.sin(phi)**2
				fr = ((basic_coord[0,:][lb0: lb1, la0: la1] - xc)**2*df1 + (basic_coord[1,:][lb0: lb1, la0: la1] - yc)**2*df2
					- cr**2*np.sin(2*phi)*(basic_coord[0,:][lb0: lb1, la0: la1] - xc)*(basic_coord[1,:][lb0: lb1, la0: la1] - yc))
				idr = fr/(lr**2*sr**2)
				jx = idr < 1
				#jx = (-1)*jx+1
				#mask_A[lb0: lb1, la0: la1] = mask_A[lb0: lb1, la0: la1] * jx

				iu = np.where(jx == True)
				iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float)
				iv[iu] = np.nan
				mask_A[lb0: lb1, la0: la1] = mask_A[lb0: lb1, la0: la1] * iv

		mirro_A = mask_A * img1
		SB2, R2, Anr2, err2 = light_measure(mirro_A, bins, 1, Rp, cx_BCG, cy_BCG, pixel, zg)
		'''
		## test the flux distribution in each bins
		sub_R = np.logspace(0, np.log10(Rp), bins)
		cdr = np.sqrt((grd[0,:] - cx_BCG)**2 + (grd[1,:] - cy_BCG)**2)
		Nr = len(sub_R)
		for xx in range(Nr - 1):
			idr = (cdr >= sub_R[xx]) & (cdr < sub_R[xx + 1])
			sub_f = mirro_A[idr]
			id_nan = np.isnan(sub_f)
			iv = np.where(id_nan == False)[0]
			bl_array = sub_f[iv]

			plt.figure()
			plt.title('$flux \; distribution \; in \; %d \; bins$' % (xx))
			plt.hist(bl_array, histtype = 'step', color = 'b', normed = True)
			plt.xlabel('$flux[nmagy]$')
			plt.savefig('/home/xkchen/mywork/ICL/code/flux_%dbins_DF_%.3fra_%.3fdec_%.3fz.png' % (xx, rag, decg, zg), dpi = 300)
			plt.close()
		'''
		# flux scale and pixel resample
		eta = Da_ref / Da
		mu = 1 / eta
		SB_ref = SB2 - 10*np.log10((1 + zg) / (1 + z_ref))
		Ar_ref = Anr2 * mu
		f_SB = interp(Ar_ref, SB_ref, kind = 'cubic')

		imgt = flux_recal(img1, zg, z_ref)
		mirroA = flux_recal(mirro_A, zg, z_ref)

		xn, yn, resam = gen(mirroA, 1, eta, cx_BCG, cy_BCG) #############???????
		xn = np.int(xn)
		yn = np.int(yn)
		if eta > 1:
			resam = resam[1:, 1:]
		elif eta == 1:
			resam = resam[1:-1, 1:-1]
		else:
			resam = resam
		SBn, Rn, Anrn, errn = light_measure(resam, bins, 1, Rpp, xn, yn, pixel, z_ref)
		arn = Anrn[(Anrn >= np.min(Ar_ref)) & (Anrn <= np.max(Ar_ref))]

		plt.figure(figsize = (16, 8))
		gs = gridspec.GridSpec(1, 2, width_ratios = [1,1])
		ax = plt.subplot(gs[0])
		bx = plt.subplot(gs[1])

		ax.plot(Anrn, SBn, 'b-', label = '$SB_{resample}$', alpha = 0.5)
		ax.plot(Ar_ref, SB_ref, 'g--', label = '$SB_{ref}$', alpha = 0.5)
		bx.plot(arn, SBn[(Anrn >= np.min(Ar_ref)) & (Anrn <= np.max(Ar_ref))] - f_SB(arn), 'g*')
		bx.axhline(y = np.mean(SBn[(Anrn >= np.min(Ar_ref)) & (Anrn <= np.max(Ar_ref))] - f_SB(arn)), ls = '--', color = 'b')

		ax.set_ylabel('$SB[mag/arcsec^2]$')
		ax.set_xlabel('$R[arcsec]$')
		ax.set_xscale('log')
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')
		ax.invert_yaxis()
		ax.legend(loc = 1, fontsize = 15)

		bx.set_xlabel('$R[arcsec]$')
		bx.set_xscale('log')
		bx.set_ylabel('$ \Delta{SB}[mag/arcsec^2] $')
		bx.tick_params(axis = 'both', which = 'both', direction = 'in')
		bx.legend(loc = 3, fontsize = 12)
		plt.subplots_adjust(hspace = 0)
		plt.tight_layout()
		plt.savefig('/home/xkchen/mywork/ICL/code/resamp_err_SB_%d_ra%.3f_dec%.3f_z%.3f.png' % (bins, rag, decg, zg), dpi = 300)
		plt.close()		#############???????????????

		'''
		xn, yn, resam = gen(imgt, 1, eta, cx_BCG, cy_BCG)
		xn = np.int(xn)
		yn = np.int(yn)
		if eta > 1:
		    resam = resam[1:, 1:]
		elif eta == 1:
		    resam = resam[1:-1, 1:-1]
		else:
			resam = resam

		## handle with these pixels around 0
		# from mask
		id_zero = np.where(mask_A == 0)
		tt_mask = mask_A * 1
		tt_mask[id_zero] = np.nan
		ttx, tty, ttmask = gen(tt_mask, 1, eta, cx_BCG, cy_BCG)

		if eta > 1:
		    ttmask = ttmask[1:, 1:]
		elif eta == 1:
		    ttmask = ttmask[1:-1, 1:-1]
		else:
			ttmask = ttmask
		id_nan = np.isnan(ttmask)
		ttmask[id_nan] = 0
		idnzeo = np.where(ttmask != 0)
		ttmask[idnzeo] = 1

		tt_img = ttmask * resam
		SB3, R3, Anr3, err3 = light_measure(tt_img, bins, 1, Rpp, xn, yn, pixel, z_ref)

		# from masked img
		id_zero = np.where(mirroA == 0)
		tt_mirro = mirroA * 1
		tt_mirro[id_zero] = np.nan 
		x2, y2, resam2 = gen(tt_mirro, 1, eta, cx_BCG, cy_BCG)
		x2 = np.int(x2)
		y2 = np.int(y2)
		if eta > 1:
		    resam2 = resam2[1:, 1:]
		elif eta == 1:
		    resam2 = resam2[1:-1, 1:-1]
		else:
			resam2 = resam2
		id_nan = np.isnan(resam2)
		resam2[id_nan] = 0
		SB5, R5, Anr5, err5 = light_measure(resam2, bins, 1, Rpp, x2, y2, pixel, z_ref)

		### part2: mask after resample (mask with the same source)
		CX_ = CX * mu
		CY_ = CY * mu
		a_ = a * mu
		b_ = b * mu

		res_mask_1 = np.ones((resam.shape[0], resam.shape[1]), dtype = np.float)
		ox_ = np.linspace(0, resam.shape[1] - 1, resam.shape[1])
		oy_ = np.linspace(0, resam.shape[0] - 1, resam.shape[0])
		basic_coord = np.array(np.meshgrid(ox_, oy_))
		major = a_ / 2
		minor = b_ / 2
		senior = np.sqrt(major**2 - minor**2)

		tdr = np.sqrt((CX_ - xn)**2 + (CY_ - yn)**2)
		dr00 = np.where(tdr == np.min(tdr))[0]

		for k in range(Numb):
			xc = CX_[k]
			yc = CY_[k]
			lr = major[k]
			sr = minor[k]
			cr = senior[k]

			set_r = np.int(np.ceil(1.2 * lr))
			la0 = np.int(xc - set_r)
			la1 = np.int(xc + set_r +1)
			lb0 = np.int(yc - set_r)
			lb1 = np.int(yc + set_r +1)

			if k == dr00[0] :
				continue
			else:
				phi = theta[k]*np.pi/180
				df1 = lr**2 - cr**2*np.cos(phi)**2
				df2 = lr**2 - cr**2*np.sin(phi)**2
				fr = ((basic_coord[0,:][lb0: lb1, la0: la1] - xc)**2*df1 + (basic_coord[1,:][lb0: lb1, la0: la1] - yc)**2*df2
					- cr**2*np.sin(2*phi)*(basic_coord[0,:][lb0: lb1, la0: la1] - xc)*(basic_coord[1,:][lb0: lb1, la0: la1] - yc))
				idr = fr/(lr**2*sr**2)
				jx = idr<=1
				jx = (-1)*jx+1
				res_mask_1[lb0: lb1, la0: la1] = res_mask_1[lb0: lb1, la0: la1]*jx
		resam1 = res_mask_1 * resam
		SBt, Rt, Anrt, errt = light_measure(resam1, bins, 1, Rpp, xn, yn, pixel, z_ref)

		# cut these pixel which is 0 before resamp
		xm, ym, res_mask_2 = gen(mask_A, 1, eta, cx_BCG, cy_BCG)
		if eta > 1:
		    res_mask_2 = res_mask_2[1:, 1:]
		elif eta == 1:
		    res_mask_2 = res_mask_2[1:-1, 1:-1]
		else:
			res_mask_2 = res_mask_2

		mix, miy, mirro_img = gen(mirroA, 1, eta, cx_BCG, cy_BCG)
		mix = np.int(mix)
		miy = np.int(miy)
		if eta > 1:
		    mirro_img = mirro_img[1:, 1:]
		elif eta == 1:
		    mirro_img = mirro_img[1:-1, 1:-1]
		else:
			mirro_img = mirro_img

		val, cont = sts.find_repeats(res_mask_2)
		ids = np.where(cont == np.max(cont))[0]
		res_mask2 = res_mask_2 / val[ids[0]]
		print('scale factor = ', val[ids[0]])
		SB4, R4, Anr4, err4 = weit_l_measure(mirro_img, res_mask2, bins, 1, Rpp, mix, miy, pixel, z_ref)

		ar4 = Anr4[(Anr4 >= np.min(Ar_ref)) & (Anr4 <= np.max(Ar_ref))]
		ar3 = Anr3[(Anr3 >= np.min(Ar_ref)) & (Anr3 <= np.max(Ar_ref))]
		art = Anrt[(Anrt >= np.min(Ar_ref)) & (Anrt <= np.max(Ar_ref))]

		plt.figure(figsize = (16, 8))
		gs = gridspec.GridSpec(1, 2, width_ratios = [1,1])
		ax = plt.subplot(gs[0])
		bx = plt.subplot(gs[1])

		ax.plot(Anrt, SBt, 'g-.', label = '$SB_{re-load \, source \, at \, z_{0}}$', alpha = 0.5)
		ax.plot(Anr4, SB4, 'b-', label = '$SB_{resample \, mask \, Metrix}$', alpha = 0.5)
		ax.plot(Anr3, SB3, 'r:', label = '$SB_{correct \, resample}$', alpha = 0.5)
		ax.plot(Ar_ref, SB_ref, 'k--', label = '$SB_{ref}$', alpha = 0.5)

		bx.plot(ar4, SB4[(Anr4 >= np.min(Ar_ref)) & (Anr4 <= np.max(Ar_ref))] - f_SB(ar4), 
			'b*', label = '$ [SB_{resample \, mask} - SB_{ref}] $', alpha = 0.5)
		bx.plot(ar3, SB3[(Anr3 >= np.min(Ar_ref)) & (Anr3 <= np.max(Ar_ref))] - f_SB(ar3), 
			'r*', label = '$ [SB_{correct \, resample} - SB_{ref}] $', alpha = 0.5)
		bx.plot(art, SBt[(Anrt >= np.min(Ar_ref)) & (Anrt <= np.max(Ar_ref))] - f_SB(art), 
			'g*', label = '$ [SB_{re-load \, source} - SB_{ref}] $', alpha = 0.5)
		bx.axhline(y = np.mean(SB4[(Anr4 >= np.min(Ar_ref)) & (Anr4 <= np.max(Ar_ref))] - f_SB(ar4)), color = 'b', ls = '--', alpha = 0.5)
		bx.axhline(y = np.mean(SB3[(Anr3 >= np.min(Ar_ref)) & (Anr3 <= np.max(Ar_ref))] - f_SB(ar3)), color = 'r', ls = '--', alpha = 0.5)
		bx.axhline(y = np.mean(SBt[(Anrt >= np.min(Ar_ref)) & (Anrt <= np.max(Ar_ref))] - f_SB(art)), color = 'g', ls = '--', alpha = 0.5)

		ax.set_title('resample SB profile comparation')
		ax.set_ylabel('$SB[mag/arcsec^2]$')
		ax.set_xlabel('$R[arcsec]$')
		ax.set_xscale('log')
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')
		ax.invert_yaxis()
		ax.legend(loc = 1, fontsize = 15)

		bx.set_xlabel('$R[arcsec]$')
		bx.set_xscale('log')
		bx.set_ylabel('$ \Delta{SB}[mag/arcsec^2] $')
		bx.tick_params(axis = 'both', which = 'both', direction = 'in')
		bx.legend(loc = 3, fontsize = 12)
		plt.subplots_adjust(hspace = 0)
		plt.tight_layout()
		plt.savefig('/home/xkchen/mywork/ICL/code/resample_test_SB_%d_ra%.3f_dec%.3f_z%.3f.png' % (bins, rag, decg, zg), dpi = 300)
		plt.close()
		'''
	raise
	return

def stack_test():

	x0 = 2427
	y0 = 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	sum_grid = np.array(np.meshgrid(Nx, Ny))

	sum_array_0 = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_0 = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_0 = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	SB_ref = []
	Ar_ref = []
	bins = 60
	for kd in range(stack_N):

		zg = z[kd]
		rag = ra[kd]
		decg = dec[kd]

		param_A = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A.sex'
		out_cat = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A.param'
		out_load_A = '/home/xkchen/mywork/ICL/data/SEX/result/mask_A_test.cat'

		tmp_load = '/home/xkchen/mywork/ICL/data/test_data/'
		load = '/home/xkchen/mywork/ICL/data/total_data/sample_02_03/'

		file = 'frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (rag, decg, zg)
		data = fits.getdata(load + file, header = True)
		img = data[0]
		Head = data[1]
		wcs = awc.WCS(Head)
		Da = Test_model.angular_diameter_distance(zg).value
		Ar = rad2asec/Da
		Rp = Ar/pixel

		xt = np.linspace(0, img.shape[1] - 1, img.shape[1])
		yt = np.linspace(0, img.shape[0] - 1, img.shape[0])
		grd = np.array(np.meshgrid(xt, yt))
		cx_BCG, cy_BCG = wcs.all_world2pix(rag*U.deg, decg*U.deg, 1)

		ra_img, dec_img = wcs.all_pix2world(grd[0,:], grd[1,:], 1)
		pos = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
		EBV = sfd(pos)
		Av = Rv * EBV * 0.86
		Al = A_wave(l_wave[2], Rv) * Av
		img1 = img*10**(Al / 2.5)

		imgt = flux_recal(img1, zg, z_ref)
		eta = Da_ref / Da
		mu = 1 / eta
		xn, yn, resam = gen(imgt, 1, eta, cx_BCG, cy_BCG)
		xn = np.int(xn)
		yn = np.int(yn)
		if eta > 1:
			resam = resam[1:, 1:]
		elif eta == 1:
			resam = resam[1:-1, 1:-1]
		else:
			resam = resam

		ix0 = np.int(Head['CRPIX1'] * mu)
		iy0 = np.int(Head['CRPIX2'] * mu)
		RA0 = Head['CRVAL1']
		DEC0 = Head['CRVAL2']

		keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2',
		        'CENTER_X','CENTER_Y','CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z']
		value = ['T', 32, 2, resam.shape[1], resam.shape[0], ix0, iy0, xn, yn, RA0, DEC0, rag, decg, zg]
		ff = dict(zip(keys,value))
		fil = fits.Header(ff)
		fits.writeto(tmp_load + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % ('r', rag, decg, zg), resam, header = fil, overwrite=True)

		file_source = tmp_load + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % ('r', rag, decg, zg)
		cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s'%(param_A, out_load_A, out_cat)
		print(cmd)
		A = subpro.Popen(cmd, shell = True)
		A.wait()

		source = asc.read(out_load_A)
		Numb = np.array(source['NUMBER'][-1])
		A = np.array(source['A_IMAGE'])
		B = np.array(source['B_IMAGE'])
		chi = np.array(source['THETA_IMAGE'])
		cx = np.array(source['X_IMAGE']) - 1
		cy = np.array(source['Y_IMAGE']) - 1

		Kron = 6
		Lr = Kron*A
		Sr = Kron*B

		CX = cx * 1
		CY = cy * 1
		a = Lr * 1
		b = Sr * 1
		theta = chi * 1

		mask_A = np.ones((resam.shape[0], resam.shape[1]), dtype = np.float)
		ox = np.linspace(0, resam.shape[1]-1, resam.shape[1])
		oy = np.linspace(0, resam.shape[0]-1, resam.shape[0])
		basic_coord = np.array(np.meshgrid(ox,oy))
		major = a/2
		minor = b/2
		senior = np.sqrt(major**2 - minor**2)

		tdr = np.sqrt((CX - xn)**2 + (CY - yn)**2)
		dr00 = np.where(tdr == np.min(tdr))[0]
		for k in range(Numb):
			xc = CX[k]
			yc = CY[k]
			lr = major[k]
			sr = minor[k]
			cr = senior[k]

			set_r = np.int(np.ceil(1.2 * lr))
			la0 = np.int(xc - set_r)
			la1 = np.int(xc + set_r +1)
			lb0 = np.int(yc - set_r)
			lb1 = np.int(yc + set_r +1)

			if k == dr00[0] :
				continue
			else:
				phi = theta[k]*np.pi/180
				df1 = lr**2 - cr**2*np.cos(phi)**2
				df2 = lr**2 - cr**2*np.sin(phi)**2
				fr = ((basic_coord[0,:][lb0: lb1, la0: la1] - xc)**2*df1 + (basic_coord[1,:][lb0: lb1, la0: la1] - yc)**2*df2
					- cr**2*np.sin(2*phi)*(basic_coord[0,:][lb0: lb1, la0: la1] - xc)*(basic_coord[1,:][lb0: lb1, la0: la1] - yc))
				idr = fr/(lr**2*sr**2)
				jx = idr < 1

				iu = np.where(jx == True)
				iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float)
				iv[iu] = np.nan
				mask_A[lb0: lb1, la0: la1] = mask_A[lb0: lb1, la0: la1] * iv

		mirro_A = mask_A * resam

		plt.figure()
		ax = plt.subplot(111)
		ax.set_title('source mask after resample %d' % kd)
		ax.imshow(resam, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
		hsc.ellipses(CX, CY, w = a, h = b, rot = theta, fc = '', ec = 'r', alpha = 0.5)
		plt.xlim(0, resam.shape[1])
		plt.ylim(0, resam.shape[0])
		plt.savefig('/home/xkchen/mywork/ICL/code/source_%d.png' % kd, dpi = 300)
		plt.close()

		ox = np.linspace(0, resam.shape[1]-1, resam.shape[1])
		oy = np.linspace(0, resam.shape[0]-1, resam.shape[0])
		oo_grd = np.array(np.meshgrid(ox, oy))
		cdr = np.sqrt((oo_grd[0,:] - xn)**2 + (oo_grd[1,:] - yn)**2)
		idd = (cdr > Rpp) & (cdr < 1.1 * Rpp)
		cut_res = mirro_A[idd]
		id_nan = np.isnan(cut_res)
		idx = np.where(id_nan == False)
		bl_array = cut_res[idx]
		back_cc = np.mean(bl_array)

		mirroA = mirro_A - back_cc
		SB2, R2, Anr2, err2 = light_measure(mirroA, bins, 1, Rpp, xn, yn, pixel, z_ref)
		SB_ = SB2[1:]
		Ar_ = Anr2[1:]
		R_ = R2[1:]
		err_ = err2[1:]
		SB_ref.append(SB_)
		Ar_ref.append(Ar_)
		
		la0 = np.int(y0 - yn)
		la1 = np.int(y0 - yn + resam.shape[0])
		lb0 = np.int(x0 - xn)
		lb1 = np.int(x0 - xn + resam.shape[1])

		idx = np.isnan(resam)
		idv = np.where(idx == False)
		sum_array_0[la0:la1, lb0:lb1][idv] = sum_array_0[la0:la1, lb0:lb1][idv] + mirroA[idv]
		count_array_0[la0: la1, lb0: lb1][idv] = resam[idv]
		id_nan = np.isnan(count_array_0)
		id_fals = np.where(id_nan == False)
		p_count_0[id_fals] = p_count_0[id_fals] + 1
		count_array_0[la0: la1, lb0: lb1][idv] = np.nan

	ll0 = [np.min(kk / Angu_ref) for kk in Ar_ref]
	tar0 = np.min(ll0)
	ll1 = [np.max(kk / Angu_ref) for kk in Ar_ref]
	tar1 = np.max(ll1)

	tar_down = tar0 * Angu_ref
	tar_up = tar1 * Angu_ref
	inter_frac = np.logspace(np.log10(tar0), np.log10(tar1), bins)
	inter_ar = inter_frac * Angu_ref

	m_flux = np.ones((stack_N, bins), dtype = np.float) * np.nan
	for pp in range(len(SB_ref)):
		id_count = np.zeros(bins, dtype = np.float)
		tsb = SB_ref[pp]
		tar = Ar_ref[pp] / Angu_ref
		t_flux = 10**((22.5 + 2.5*np.log10(pixel**2) - tsb) / 2.5)
		for kk in range(len(tar)):
			sub_ar = np.abs(inter_frac - tar[kk])
			id_min = np.where(sub_ar == np.min(sub_ar))[0]
			id_count[id_min[0]] = id_count[id_min[0]] + 1
		id_nuzero = id_count != 0
		id_g = np.where(id_nuzero == True)[0]
		m_flux[pp, id_g] = t_flux

	m_count = np.zeros(bins, dtype = np.float)
	inter_flux = np.zeros(bins, dtype = np.float)
	for pp in range(bins):
		sub_flux = m_flux[:, pp]
		iy = np.isnan(sub_flux)
		iv = np.where(iy == False)[0]
		m_count[pp] = len(iv)
		inter_flux[pp] = inter_flux[pp] + np.sum(sub_flux[iv])
	inter_flux = inter_flux / m_count

	id_nan = np.isnan(inter_flux)
	id_x = id_nan == False
	id_inf = np.isinf(inter_flux)
	id_y = id_inf == False
	id_zero = inter_flux == 0
	id_z = id_zero == False
	id_set = id_x & id_y & id_z

	ref_ar = inter_ar[id_set]
	ref_flux = inter_flux[id_set]
	ref_SB = 22.5 - 2.5 * np.log10(ref_flux) + 2.5*np.log10(pixel**2)
	f_SB = interp(ref_ar, ref_SB, kind = 'cubic')

	mean_array_0 = sum_array_0 / p_count_0
	where_are_inf = np.isinf(mean_array_0)
	mean_array_0[where_are_inf] = np.nan
	id_zeros = np.where(p_count_0 == 0)
	mean_array_0[id_zeros] = np.nan

	SB, R, Ar, error = light_measure(mean_array_0, bins, 1, Rpp, x0, y0, pixel, z_ref)
	SB_0 = SB[1:] + 0
	R_0 = R[1:]
	Ar_0 = Ar[1:]
	err_0 = error[1:]
	Ar0 = (Ar_0 / Angu_ref) * Angu_ref

	plt.figure(figsize = (16, 8))
	gs = gridspec.GridSpec(1, 2, width_ratios = [1,1])
	ax = plt.subplot(gs[0])
	bx = plt.subplot(gs[1])

	ax.plot(ref_ar, ref_SB, 'r-', label = '$stacking \, profile$', alpha = 0.5)
	ax.plot(Ar0, SB_0, 'g--', label = '$stacking \, img$', alpha = 0.5)

	bx.plot(Ar0[(Ar0 > tar_down * 1.01) & (Ar0 < tar_up / 1.01)], SB_0[(Ar0 > tar_down * 1.01) & (Ar0 < tar_up / 1.01)] - 
		f_SB(Ar0[(Ar0 > tar_down * 1.01) & (Ar0 < tar_up / 1.01)]), 'b*', alpha = 0.5)

	ax.set_xscale('log')
	ax.set_xlabel('$R[arcsec]$')
	ax.set_ylabel('$SB[mag/arcsec^2]$')
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax.legend(loc = 1, fontsize = 12)
	ax.invert_yaxis()

	bx.set_xlabel('$R[arcsec]$')
	bx.set_xscale('log')
	bx.set_ylabel('$ \Delta{SB}[mag/arcsec^2] $')
	bx.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.subplots_adjust(hspace = 0)
	plt.tight_layout()
	plt.savefig('/home/xkchen/mywork/ICL/code/stack_test.png', dpi = 300)
	plt.close()

	raise
	return

def main():
	#source_compa()
	#color()
	#resamp_test()
	stack_test()

if __name__ == "__main__":
	main()