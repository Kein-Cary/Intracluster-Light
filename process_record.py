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
from scipy.interpolate import interp1d as interp
from dustmaps.sfd import SFDQuery
from scipy.optimize import curve_fit
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord

from resamp import gen
from extinction_redden import A_wave
from light_measure import light_measure, flux_recal, weit_l_measure
from light_measure import sigmamc

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

pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.25
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2
# total catalogue with redshift
with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
	catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]

R0 = 1 # in unit Mpc
Angu_ref = (R0/Da_ref)*rad2asec
Rpp = Angu_ref/pixel

# dust correct
Rv = 3.1
stack_N = 5
sfd = SFDQuery()

band = ['r', 'g', 'i', 'u', 'z']
sum_band = ['r', 'i', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])
mag_add = np.array([0, 0, 0, -0.04, 0.02])
zpot = np.array([22.5, 22.5, 22.5, 22.46, 22.52])
sb_lim = np.array([24.5, 25, 24, 24.35, 22.9])

# read redMapper catalog as comparation for sextractor
goal_data = fits.getdata('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
sub_data = fits.getdata('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')

repeat = sts.find_repeats(sub_data.ID)
rept_ID = np.int0(repeat)
ID_array = np.int0(sub_data.ID)
sub_redshift = np.array(sub_data.Z_SPEC) 
center_distance = np.array(sub_data.R)/h # in unit Mpc
member_pos = np.array([sub_data.RA,sub_data.DEC])

redshift = np.array(goal_data.Z_SPEC)
richness = np.array(goal_data.LAMBDA)
host_ID = np.array(goal_data.ID)
# select 0.2 <= z <= 0.3
Lambd = richness[(redshift >= 0.2) & (redshift <= 0.3)]
ID_set = host_ID[(redshift >= 0.2) & (redshift <= 0.3)]
use_z = redshift *1

def mask_part():

	x0 = np.linspace(0, 2047, 2048)
	y0 = np.linspace(0, 1488, 1489)
	grd = np.array(np.meshgrid(x0, y0))
	r_star = 2*1.5/pixel

	bins = 90
	#kd = 0
	kd = 19
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
	SB, R, Anr, err = light_measure(img, bins, 1, Rp, cx_BCG, cy_BCG, pixel, zg)

	## redden
	ra_img, dec_img = wcs.all_pix2world(grd[0,:], grd[1,:], 1)
	pos = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
	EBV = sfd(pos)
	Av = Rv * EBV * 0.86
	Al = A_wave(l_wave[2], Rv) * Av
	img1 = img*10**(Al / 2.5)
	SB1, R1, Anr1, err1 = light_measure(img1, bins, 1, Rp, cx_BCG, cy_BCG, pixel, zg)

	### compare part
	hdu = fits.PrimaryHDU()
	hdu.data = img1
	hdu.header = Head
	hdu.writeto(tmp_load + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % ('r', rag, decg, zg), overwrite = True)

	file_source = tmp_load + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % ('r', rag, decg, zg)
	minS = str(5)
	cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s -DETECT_MINAREA %s'%(param_A, out_load_A, out_cat, '5')
	print(cmd)
	a = subpro.Popen(cmd, shell = True)
	a.wait()

	source = asc.read(out_load_A)
	Numb = np.array(source['NUMBER'][-1])
	Nz = Numb *1
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])
	chi = np.array(source['THETA_IMAGE'])
	cx = np.array(source['X_IMAGE']) - 1
	cy = np.array(source['Y_IMAGE']) - 1

	Kron = 6
	Lr = Kron*A
	Sr = Kron*B

	# catalog from redMapper
	IA = np.where(use_z == zg)[0]
	IA = IA[0]
	N_sat = rept_ID[1][IA]
	sum_ID = np.sum(rept_ID[1][:IA])
	Cr_sat = center_distance[sum_ID:sum_ID + N_sat]
	R_sat = Cr_sat[Cr_sat <= 1.]
	poa = member_pos[0][sum_ID:sum_ID+rept_ID[1][IA]]
	pob = member_pos[1][sum_ID:sum_ID+rept_ID[1][IA]]
	posx = poa[poa != rag]
	posy = pob[poa != rag]
	Sx, Sy = wcs.all_world2pix(posx * U.deg, posy * U.deg, 1)

	# star form SDSS
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
			jx = idr<=1
			jx = (-1)*jx+1
			mask_A[lb0: lb1, la0: la1] = mask_A[lb0: lb1, la0: la1]*jx
	mirro_A = mask_A *img1
	SBt, Rt, Anrt, errt = light_measure(mirro_A, bins, 1, Rp, cx_BCG, cy_BCG, pixel, zg)

	## flux scale
	img_scale = flux_recal(mirro_A, zg, z_ref)
	eta = Da_ref / Da
	mu = 1 / eta
	SB_ref = SBt - 10*np.log10((1 + zg) / (1 + z_ref))
	Ar_ref = Anrt * mu
	f_SB = interp(Ar_ref, SB_ref, kind = 'cubic')
	SB2, R2, Anr2, err2 = light_measure(img_scale, bins, 1, Rp, cx_BCG, cy_BCG, pixel * mu, zg)

	## resample
	xn, yn, resam = gen(img_scale, 1, eta, cx_BCG, cy_BCG)
	xn = np.int(xn)
	yn = np.int(yn)
	
	if eta > 1:
	    resam = resam[1:, 1:]
	elif eta == 1:
	    resam = resam[1:-1, 1:-1]
	else:
		resam = resam	
	SB3, R3, Anr3, err3 = light_measure(resam, bins, 1, Rpp, xn, yn, pixel, z_ref)

	## reproduce mask
	xm, ym, res_mask_2 = gen(mask_A, 1, eta, cx_BCG, cy_BCG)
	if eta > 1:
	    res_mask_2 = res_mask_2[1:, 1:]
	elif eta == 1:
	    res_mask_2 = res_mask_2[1:-1, 1:-1]
	else:
		res_mask_2 = res_mask_2

	val, cont = sts.find_repeats(res_mask_2)
	ids = np.where(cont == np.max(cont))[0]
	res_mask2 = res_mask_2 / val[ids[0]]
	SB4, R4, Anr4, err4 = weit_l_measure(resam, res_mask2, bins, 1, Rpp, xn, yn, pixel, z_ref)

	plt.figure()
	ax = plt.subplot(111)
	ax.plot(Ar_ref, SB_ref, 'g--', label = '$SB_{ref}$', alpha = 0.5)
	ax.plot(Anr2, SB2, 'b-', label = '$SB_{rescale}$', alpha = 0.25)
	ax.set_title('scale SB profile comparation')
	ax.set_xlabel('$R[arcsec]$')
	ax.set_xscale('log')
	ax.set_ylabel('$SB[mag/arcsec^2]$')
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax.invert_yaxis()
	ax.legend(loc = 1)
	plt.savefig('/home/xkchen/mywork/ICL/code/rescale_SB.png', dpi = 300)
	plt.close()

	plt.figure()
	ax = plt.subplot(111)
	ax.plot(Anr1, SB1, 'b-', label = '$SB_{dust \, correct}$', alpha = 0.25)
	ax.plot(Anrt, SBt, 'r--', label = '$SB_{source \, masked}$', alpha = 0.5)
	ax.set_title('masked SB profile comparation')
	ax.set_xlabel('$R[arcsec]$')
	ax.set_xscale('log')
	ax.set_ylabel('$SB[mag/arcsec^2]$')
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax.invert_yaxis()
	ax.legend(loc = 1)
	plt.savefig('/home/xkchen/mywork/ICL/code/masked_SB.png', dpi = 300)
	plt.close()

	plt.figure(figsize = (16, 8))
	gs = gridspec.GridSpec(1, 2, width_ratios = [1,1])
	ax = plt.subplot(gs[0])
	bx = plt.subplot(gs[1])

	ax.plot(Anr3, SB3, 'b-', label = '$SB_{resample}$', alpha = 0.5)
	ax.plot(Ar_ref, SB_ref, 'g--', label = '$SB_{ref}$', alpha = 0.25)
	ax.set_title('resample SB comparation')
	ax.set_ylabel('$SB[mag/arcsec^2]$')
	ax.set_xlabel('$R[arcsec]$')
	ax.set_xscale('log')
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')

	ax1 = ax.twiny()
	xtick = ax.get_xticks()
	R_tick = Da * (xtick/rad2asec) * 10**3
	ax1.set_xticks(xtick)
	ax1.set_xticklabels(["%.2f" % ll for ll in R_tick])
	ax1.set_xlim(ax.get_xlim())
	ax1.tick_params(axis = 'x', which = 'both', direction = 'in')
	ax.invert_yaxis()
	ax.legend(loc = 1)

	ar3 = Anr3[(Anr3 >= np.min(Ar_ref)) & (Anr3 <= np.max(Ar_ref))]
	bx.plot(ar3, SB3[(Anr3 >= np.min(Ar_ref)) & (Anr3 <= np.max(Ar_ref))] - f_SB(ar3), 
		'g*', label = '$\Delta_{SB}[{SB_{dust \, correct} - SB_{origin}}]$')
	bx.axhline(y = np.mean(SB3[(Anr3 >= np.min(Ar_ref)) & (Anr3 <= np.max(Ar_ref))] - f_SB(ar3)), ls = '--', color = 'g')
	bx.set_ylabel('$\Delta_{SB}[mag/arcsec^2]$')
	bx.set_xlabel('$R[arcsec]$')
	bx.set_xscale('log')
	bx.tick_params(axis = 'both', which = 'both', direction = 'in')
	bx.legend(loc = 1)

	plt.subplots_adjust(hspace = 0)
	plt.tight_layout()
	plt.savefig('/home/xkchen/mywork/ICL/code/previous_resamp_SB.png', dpi = 300)
	plt.close()

	plt.figure(figsize = (16, 8))
	gs = gridspec.GridSpec(1, 2, width_ratios = [1,1])
	ax = plt.subplot(gs[0])
	bx = plt.subplot(gs[1])

	ax.plot(Anr4, SB4, 'b-', label = '$SB_{masked \, at \, z_{ref}}$', alpha = 0.25)
	ax.plot(Ar_ref, SB_ref, 'g--', label = '$SB_{ref}$', alpha = 0.5)
	ax.plot(Anrt, SBt, 'r-', label = '$SB_{masked \, at \, z_0}$', alpha = 0.5)
	ax.set_title('resample SB profile comparation')
	ax.set_ylabel('$SB[mag/arcsec^2]$')
	ax.set_xlabel('$R[arcsec]$')
	ax.set_xscale('log')
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax.invert_yaxis()
	ax.legend(loc = 1)

	ar4 = Anr4[(Anr4 >= np.min(Ar_ref)) & (Anr4 <= np.max(Ar_ref))]
	bx.plot(ar4, SB4[(Anr4 >= np.min(Ar_ref)) & (Anr4 <= np.max(Ar_ref))] - f_SB(ar4), 
		'b*', label = '$\Delta_{SB}[{SB_{masked \, z_{ref}} - SB_{ref}}]$')
	bx.axhline(y = np.mean(SB4[(Anr4 >= np.min(Ar_ref)) & (Anr4 <= np.max(Ar_ref))] - f_SB(ar4)), ls = '--', color = 'b')
	bx.set_ylabel('$\Delta_{SB}[mag/arcsec^2]$')
	bx.set_xlabel('$R[arcsec]$')
	bx.set_xscale('log')
	bx.tick_params(axis = 'both', which = 'both', direction = 'in')
	bx.legend(loc = 1)

	plt.subplots_adjust(hspace = 0)
	plt.tight_layout()
	plt.savefig('/home/xkchen/mywork/ICL/code/masked_resamp_SB.png', dpi = 300)
	plt.close()

	raise
	return

def mask_B():
	x0 = np.linspace(0, 2047, 2048)
	y0 = np.linspace(0, 1488, 1489)
	img_grid = np.array(np.meshgrid(x0, y0))

	load = '/home/xkchen/mywork/ICL/data/total_data/sample_02_03/'
	mask = '/home/xkchen/mywork/ICL/data/star_catalog/'

	for pp in range(stack_N):

		ra_g = ra[pp]
		dec_g = dec[pp]
		z_g = z[pp]
		for q in range(len(band)):

			file = 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[q], ra_g, dec_g, z_g)

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

			Av = Rv * BEV * 0.86
			Al = A_wave(l_wave[q], Rv) * Av
			img = img*10**(Al / 2.5)

			cat = pd.read_csv('/home/xkchen/mywork/ICL/data/star_catalog/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g), skiprows = 1)
			ra_s = np.array(cat['ra'])
			dec_s = np.array(cat['dec'])
			mag = np.array(cat['r'])
			R0 = np.array(cat['psffwhm_r'])

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
			
			hdu = fits.PrimaryHDU()
			hdu.data = mirro_B
			hdu.header = head_inf
			hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/mask/B_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[q], ra_g, dec_g, z_g),overwrite = True)
			# aslo save the mask_matrix
			hdu = fits.PrimaryHDU()
			hdu.data = mask_B
			hdu.header = head_inf
			hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/mask/B_mask_metrx_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[q], ra_g, dec_g, z_g),overwrite = True)

	return

def mask_A():
	t0 = time.time()

	x0 = np.linspace(0, 2047, 2048)
	y0 = np.linspace(0, 1488, 1489)
	img_grid = np.array(np.meshgrid(x0, y0))

	bin_number = 90
	r_star = 2*1.5/pixel #mask star radius
	load = '/home/xkchen/mywork/ICL/data/total_data/sample_02_03/'

	param_A = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A.sex'
	param_A_Tal = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A_Tal.sex' # Tal 2011
	param_A_Ze = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A_Ze.sex' # Zibetti 2005
	param_sky = '/home/xkchen/mywork/ICL/data/SEX/default_sky_mask.sex'

	out_cat = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A.param'
	out_load_A = '/home/xkchen/mywork/ICL/data/SEX/result/mask_A_test.cat'
	out_load_B = '/home/xkchen/mywork/ICL/data/SEX/result/mask_B_test.cat'
	out_load_sky = '/home/xkchen/mywork/ICL/data/SEX/result/mask_sky_test.cat'

	for i in range(len(band)):
		for q in range(stack_N):
			ra_g = ra[q]
			dec_g = dec[q]
			z_g = z[q]

			file = 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[i], ra_g, dec_g, z_g)
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
			Al = A_wave(l_wave[i], Rv) * Av
			img = img*10**(Al / 2.5)

			cx_BCG, cy_BCG = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)
			R_ph = rad2asec/(Test_model.angular_diameter_distance(z_g).value)
			R_p = R_ph/pixel

			if (i == 0) | (i == 4):
				# Tal et.al 
				combine = np.zeros((1489, 2048), dtype = np.float)
				for q in range(len(sum_band)):
					file_q = 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (sum_band[q], ra_g, dec_g, z_g)
					data_q = fits.open(load + file_q)
					img_q = data_q[0].data
					combine = combine + img_q

				hdu = fits.PrimaryHDU()
				hdu.data = combine
				hdu.header = head_inf
				hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/' + 'combine_data_ra%.3f_dec%.3f.fits'%(ra_g, dec_g), overwrite = True)
				file_source = '/home/xkchen/mywork/ICL/data/test_data/' + 'combine_data_ra%.3f_dec%.3f.fits'%(ra_g, dec_g)
				cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s'%(param_A_Tal,
					'/home/xkchen/mywork/ICL/data/SEX/result/mask_A_%.3fra_%.3fdec_%.3fz.cat' % (ra_g, dec_g, z_g), out_cat)
			else:
				hdu = fits.PrimaryHDU()
				hdu.data = data_f[0].data
				hdu.header = head_inf
				hdu.writeto('source_data.fits', overwrite = True)
				file_source = './source_data.fits'	
				dete_thresh = sb_lim[i] + 10*np.log10((1 + z_g)/(1 + z_ref))
				dete_thresh = '%.3f' % dete_thresh + ',%.2f' % zpot[i]
				dete_min = '10'
				ana_thresh = dete_thresh *1
				cmd = (
					'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s -DETECT_MINAREA %s -DETECT_THRESH %s -ANALYSIS_THRESH %s'
					%(param_A, '/home/xkchen/mywork/ICL/data/SEX/result/mask_A_%.3fra_%.3fdec_%.3fz.cat' % 
						(ra_g, dec_g, z_g), out_cat, dete_min, dete_thresh, ana_thresh))

			print(cmd)
			A = subpro.Popen(cmd, shell = True)
			A.wait()

			source = asc.read('/home/xkchen/mywork/ICL/data/SEX/result/mask_A_%.3fra_%.3fdec_%.3fz.cat' % (ra_g, dec_g, z_g))
			Numb = np.array(source['NUMBER'][-1])
			Nz = Numb *1
			A = np.array(source['A_IMAGE'])
			B = np.array(source['B_IMAGE'])
			theta = np.array(source['THETA_IMAGE'])
			cx = np.array(source['X_IMAGE']) - 1
			cy = np.array(source['Y_IMAGE']) - 1
			p_type = np.array(source['CLASS_STAR'])
			#Kron = source['KRON_RADIUS']
			Kron = 5
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
			minor = b/2
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

			hdu = fits.PrimaryHDU()
			hdu.data = mirro_A
			hdu.header = head_inf
			hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/mask/A_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[i], ra_g, dec_g, z_g),overwrite = True)

			hdu = fits.PrimaryHDU()
			hdu.data = mask_A
			hdu.header = head_inf
			hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/mask/A_mask_metrx_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[i], ra_g, dec_g, z_g),overwrite = True)

	return

def stack_A():
	un_mask = 0.15
	load = '/home/xkchen/mywork/ICL/data/test_data/mask/'
	x0 = 2427
	y0 = 1765
	bins = 90
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	sum_grid = np.array(np.meshgrid(Nx, Ny))

	sky_lev = np.zeros(len(band), dtype = np.float)
	# stack cluster
	for ii in range(len(band)):
		sum_array_0 = np.zeros((len(Ny), len(Nx)), dtype = np.float) 
		count_array_0 = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		p_count_0 = np.zeros((len(Ny), len(Nx)), dtype = np.float)

		sum_array_1 = np.zeros((len(Ny), len(Nx)), dtype = np.float) # reload source
		count_array_1 = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		p_count_1 = np.zeros((len(Ny), len(Nx)), dtype = np.float)

		sum_array_2 = np.zeros((len(Ny), len(Nx)), dtype = np.float) # set 0 around
		count_array_2 = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		p_count_2 = np.zeros((len(Ny), len(Nx)), dtype = np.float)

		sum_array_3 = np.zeros((len(Ny), len(Nx)), dtype = np.float) # weight with mask
		count_array_3 = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		p_count_3 = np.zeros((len(Ny), len(Nx)), dtype = np.float)

		SB_ref = []
		Ar_ref = []
		for jj in range(stack_N):
			ra_g = ra[jj]
			dec_g = dec[jj]
			z_g = z[jj]
			Da_g = Test_model.angular_diameter_distance(z_g).value
			data = fits.getdata(load + 'A_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[ii], ra_g, dec_g, z_g), header = True)
			img = data[0]
			wcs = awc.WCS(data[1])
			cx, cy = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

			Angur = (R0*rad2asec/Da_g)
			Rp = Angur/pixel
			L_ref = Da_ref*pixel/rad2asec
			L_z0 = Da_g*pixel/rad2asec
			eta = L_ref/L_z0
			miu = 1 / eta
			Rref = (R0*rad2asec/Da_ref)/pixel

			SB_ll, R_ll, Ar_ll, error_ll = light_measure(img, bins, 1, Rp, cx, cy, pixel, z_g)
			L_SB = SB_ll[1:]
			L_R = R_ll[1:]
			L_Ar = Ar_ll[1:]
			L_erro = error_ll[1:]

			SB_set = L_SB - 10*np.log10((1 + z_g) / (1 + z_ref)) # reference SB at z_ref
			Ar_set = L_Ar * miu
			SB_ref.append(SB_set)
			Ar_ref.append(Ar_set)

			f_goal = flux_recal(img, z_g, z_ref)
			xn, yn, resam = gen(f_goal, 1, eta, cx, cy)
			xn = np.int(xn)
			yn = np.int(yn)
			if eta > 1:
			    resam = resam[1:, 1:]
			elif eta == 1:
			    resam = resam[1:-1, 1:-1]
			else:
			    resam = resam
			# get the background of each img
			dr = np.sqrt((sum_grid[0,:] - x0)**2 + (sum_grid[1,:] - y0)**2)
			ia = dr >= Rpp
			ib = dr <= 1.1 * Rpp
			ic = ia & ib
			sky_set = resam[ic]
			sky_light = np.sum(sky_set[sky_set != 0])/len(sky_set[sky_set != 0])

			la0 = np.int(y0 - yn)
			la1 = np.int(y0 - yn + resam.shape[0])
			lb0 = np.int(x0 - xn)
			lb1 = np.int(x0 - xn + resam.shape[1])

			sum_array_0[la0:la1, lb0:lb1] = sum_array_0[la0:la1, lb0:lb1] + (resam - sky_light)
			count_array_0[la0: la1, lb0: lb1] = resam
			ia = np.where(count_array_0 != 0)
			p_count_0[ia[0], ia[1]] = p_count_0[ia[0], ia[1]] + 1
			count_array_0[la0: la1, lb0: lb1] = 0

			## re-corect part
			# reload_source
			source = asc.read('/home/xkchen/mywork/ICL/data/SEX/result/mask_A_%.3fra_%.3fdec_%.3fz.cat' % (ra_g, dec_g, z_g))
			Numb = np.array(source['NUMBER'][-1])
			Nz = Numb *1
			A = np.array(source['A_IMAGE'])
			B = np.array(source['B_IMAGE'])
			theta = np.array(source['THETA_IMAGE'])
			cx = np.array(source['X_IMAGE']) - 1
			cy = np.array(source['Y_IMAGE']) - 1
			p_type = np.array(source['CLASS_STAR'])

			Kron = 5
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
			cx = cx * miu
			cy = cy * miu
			a = a * miu
			b = b * miu

			Numb = Numb + len(comx)
			tt_mask = np.ones((resam.shape[0], resam.shape[1]), dtype = np.float)
			ox = np.linspace(0, resam.shape[1] - 1, resam.shape[1])
			oy = np.linspace(0, resam.shape[0] - 1, resam.shape[0])
			basic_coord = np.array(np.meshgrid(ox,oy))
			major = a/2
			minor = b/2
			senior = np.sqrt(major**2 - minor**2)

			tdr = np.sqrt((cx - xn)**2 + (cy - yn)**2)
			dr00 = np.where(tdr == np.min(tdr))[0]
			for k in range(Numb):
				xc = cx[k]
				yc = cy[k]
				set_r = np.int(np.ceil(1.2 * major[k]))

				la0 = np.int(xc - set_r)
				la1 = np.int(xc + set_r +1)
				lb0 = np.int(yc - set_r)
				lb1 = np.int(yc + set_r +1)

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
					jx = (-1) * jx+1
					tt_mask[lb0: lb1, la0: la1] = tt_mask[lb0: lb1, la0: la1] * jx
			mirro_A = tt_mask * resam

			la0 = np.int(y0 - yn)
			la1 = np.int(y0 - yn + mirro_A.shape[0])
			lb0 = np.int(x0 - xn)
			lb1 = np.int(x0 - xn + mirro_A.shape[1])

			sum_array_1[la0:la1, lb0:lb1] = sum_array_1[la0:la1, lb0:lb1] + (mirro_A - sky_light)
			count_array_1[la0: la1, lb0: lb1] = resam
			ia = np.where(count_array_1 != 0)
			p_count_1[ia[0], ia[1]] = p_count_1[ia[0], ia[1]] + 1
			count_array_1[la0: la1, lb0: lb1] = 0

			# set pixels (around 0) as 0
			id_zero = np.where(f_goal == 0)
			tt_img = f_goal * 1
			tt_img[id_zero] = np.nan
			xt, yt, resamt = gen(tt_img, 1, eta, cx_BCG, cy_BCG)
			xt = np.int(xt)
			yt = np.int(yt)
			if eta > 1:
			    resamt = resamt[1:, 1:]
			elif eta == 1:
			    resamt = resamt[1:-1, 1:-1]
			else:
				resamt = resamt
			id_nan = np.isnan(resamt)
			resamt[id_nan] = 0	
			la0 = np.int(y0 - yn)
			la1 = np.int(y0 - yn + resamt.shape[0])
			lb0 = np.int(x0 - xn)
			lb1 = np.int(x0 - xn + resamt.shape[1])

			sum_array_2[la0:la1, lb0:lb1] = sum_array_2[la0:la1, lb0:lb1] + (resamt - sky_light)
			count_array_2[la0: la1, lb0: lb1] = resamt
			ia = np.where(count_array_2 != 0)
			p_count_2[ia[0], ia[1]] = p_count_2[ia[0], ia[1]] + 1
			count_array_2[la0: la1, lb0: lb1] = 0

			# set maks metrix as weight
			mask = fits.getdata(load + 'A_mask_metrx_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[ii], ra_g, dec_g, z_g), header = True)
			mask_img = mask[0]
			xm, ym, res_mask = gen(mask_img, 1, eta, cx_BCG, cy_BCG)
			if eta > 1:
				res_mask = res_mask[1:, 1:]
			elif eta == 1:
				res_mask = res_mask[1:-1, 1:-1]
			else:
				res_mask = res_mask
			val, cont = sts.find_repeats(res_mask)
			ids = np.where(cont == np.max(cont))[0]
			resmask = res_mask / val[ids[0]]
			weit_img = resmask * resam

			la0 = np.int(y0 - yn)
			la1 = np.int(y0 - yn + resam.shape[0])
			lb0 = np.int(x0 - xn)
			lb1 = np.int(x0 - xn + resam.shape[1])

			sum_array_3[la0:la1, lb0:lb1] = sum_array_3[la0:la1, lb0:lb1] + (weit_img - sky_light)
			count_array_3[la0: la1, lb0: lb1] = resam
			ia = np.where(count_array_3 != 0)
			p_count_3[ia[0], ia[1]] = p_count_3[ia[0], ia[1]] + 1
			count_array_3[la0: la1, lb0: lb1] = 0			

		mean_array_0 = sum_array_0/p_count_0
		where_are_nan = np.isnan(mean_array_0)
		mean_array_0[where_are_nan] = 0
		SB, R, Ar, error = light_measure(mean_array_0, bins, 1, Rpp, x0, y0, pixel, z_ref)
		SB_0 = SB[1:] + mag_add[ii]
		R_0 = R[1:]
		Ar_0 = Ar[1:]
		err_0 = error[1:]
		'''
		stack_A = np.array([SB_0, R_0, Ar_0, err_0])
		with h5py.File('/home/xkchen/mywork/ICL/data/test_data/SB_stack_Amask_%s_band.h5' % band[ii], 'w') as f:
			f['a'] = np.array(stack_A)
		with h5py.File('/home/xkchen/mywork/ICL/data/test_data/SB_stack_Amask_%s_band.h5' % band[ii]) as f:
			for tt in range(len(stack_A)):
				f['a'][tt,:] = stack_A[tt,:]
		'''
		mean_array_1 = sum_array_1/p_count_1
		where_are_nan = np.isnan(mean_array_1)
		mean_array_1[where_are_nan] = 0
		SB, R, Ar, error = light_measure(mean_array_1, bins, 1, Rpp, x0, y0, pixel, z_ref)
		SB_1 = SB[1:] + mag_add[ii]
		R_1 = R[1:]
		Ar_1 = Ar[1:]
		err_1 = error[1:]

		mean_array_2 = sum_array_2/p_count_2
		where_are_nan = np.isnan(mean_array_2)
		mean_array_2[where_are_nan] = 0
		SB, R, Ar, error = light_measure(mean_array_2, bins, 1, Rpp, x0, y0, pixel, z_ref)
		SB_2 = SB[1:] + mag_add[ii]
		R_2 = R[1:]
		Ar_2 = Ar[1:]
		err_2 = error[1:]

		mean_array_3 = sum_array_3/p_count_3
		where_are_nan = np.isnan(mean_array_3)
		mean_array_3[where_are_nan] = 0
		SB, R, Ar, error = light_measure(mean_array_3, bins, 1, Rpp, x0, y0, pixel, z_ref)
		SB_3 = SB[1:] + mag_add[ii]
		R_3 = R[1:]
		Ar_3 = Ar[1:]
		err_3 = error[1:]

		ll0 = [np.max(kk) for kk in Ar_ref]
		tar1 = np.min(ll0)
		ll1 = [np.min(kk) for kk in Ar_ref]
		tar0 = np.max(ll1)

		ll = [len(kk) for kk in SB_ref]
		id_ref = np.where(ll == np.max(ll))[0]
		tAr = np.array(Ar_ref[id_ref[0]])
		tSB = np.array(SB_ref[id_rer[0]])
		inter_ar = tAr[(tAr >= tar0) & (tAr <= tar1)]
		
		m_SB = []
		for pp in range(len(SB_ref)):			
			tsb = SB_ref[pp]
			tar = AR_ref[pp]
			SB_ff = interp(tar, tsb)
			mSB = SB_ff(inter_ar)
			m_SB.append(mSB)
		inter_SB = np.mean(m_SB, axis = 0)

		# stack results without handle
		plt.figure(figsize = (16, 8))
		gs = gridspec.GridSpec(1, 2, width_ratios = [1,1])
		ax = plt.subplot(gs[0])
		bx = plt.subplot(gs[1])

		ax.plot(inter_ar, inter_SB, 'r--', label = '$SB_{ref} \, stacked$', alpha = 0.5)
		ax.plot(Ar_0, SB_0, 'b-', label = '$SB_{stack} \, without \, correct$', alpha = 0.5)

		f_SB = interp(inter_ar, inter_SB)
		bx.plot(Ar_0[(Ar_0 >= tar0)&(Ar_0 <= tar1)], SB_0[(Ar_0 >= tar0)&(Ar_0 <= tar1)] - f_SB(Ar_0[(Ar_0 >= tar0)&(Ar_0 <= tar1)]), 'g*', alpha = 0.5)
		bx.axhline(y = np.mean(SB[(Ar_0 >= tar0)&(Ar_0 <= tar1)] - f_SB(Ar_0[(Ar_0 >= tar0)&(Ar_0 <= tar1)])), ls = '--', color = 'b', alpha = 0.5)

		ax.set_xscale('log')
		ax.set_xlabel('$R[arcsec]$')
		ax.set_ylabel('$SB[mag/arcsec^2]$')
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')
		ax.invert_yaxis()
		ax.legend(loc = 1, fontsize = 15)
		ax.set_title('stacked SB with no resample correct')
		bx.set_xlabel('$R[arcsec]$')
		bx.set_xscale('log')
		bx.set_ylabel('$ \Delta{SB}[mag/arcsec^2] $')
		bx.tick_params(axis = 'both', which = 'both', direction = 'in')
		bx.legend(loc = 3, fontsize = 12)

		plt.subplots_adjust(hspace = 0)
		plt.tight_layout()
		plt.savefig('/home/xkchen/mywork/ICL/code/stack_profile_origin.png', dpi = 300)
		plt.close()

		# stack results with handle
		plt.figure(figsize = (16, 8))
		gs = gridspec.GridSpec(1, 2, width_ratios = [1,1])
		ax = plt.subplot(gs[0])
		bx = plt.subplot(gs[1])

		ax.plot(inter_ar, inter_SB, 'r--', label = '$SB_{ref} \, stacked$', alpha = 0.5)
		ax.plot(Ar_1, SB_1, 'g-', label = '$SB_{reload \, source}$', alpha = 0.5)
		ax.plot(Ar_2, SB_2, 'b-', label = '$SB_{set \, 0 \, nearest}$', alpha = 0.5)
		ax.plot(Ar_3, SB_3, 'm-', label = '$SB_{weight \, with \, matrix}$', alpha = 0.5)

		f_SB = interp(inter_ar, inter_SB)
		bx.plot(Ar_1[(Ar_1 >= tar0) & (Ar_1 <= tar1)], 
			SB_1[(Ar_1 >= tar0) & (Ar_1 <= tar1)] - f_SB(Ar_1[(Ar_1 >= tar0) & (Ar_1 <= tar1)]), 'g*', label = '$SB_{reload \, source} - SB_{ref}$', alpha = 0.5)
		bx.plot(Ar_2[(Ar_2 >= tar0) & (Ar_2 <= tar1)], 
			SB_2[(Ar_2 >= tar0) & (Ar_2 <= tar1)] - f_SB(Ar_2[(Ar_2 >= tar0) & (Ar_2 <= tar1)]), 'b*', label = '$SB_{set \, 0 \, around} - SB_{ref}$', alpha = 0.5)
		bx.plot(Ar_3[(Ar_3 >= tar0) & (Ar_3 <= tar1)], 
			SB_3[(Ar_3 >= tar0) & (Ar_3 <= tar1)] - f_SB(Ar_3[(Ar_3 >= tar0) & (Ar_3 <= tar1)]), 'm*', label = '$SB_{weight \, with \, mask} - SB_{ref}$', alpha = 0.5)
		
		bx.axhline(y = np.mean(SB_1[(Ar_1 >= tar0)&(Ar_1 <= tar1)] - f_SB(Ar_1[(Ar_1 >= tar0)&(Ar_1 <= tar1)])), ls = '--', color = 'g', alpha = 0.5)
		bx.axhline(y = np.mean(SB_2[(Ar_2 >= tar0)&(Ar_2 <= tar1)] - f_SB(Ar_2[(Ar_2 >= tar0)&(Ar_2 <= tar1)])), ls = '--', color = 'g', alpha = 0.5)
		bx.axhline(y = np.mean(SB_3[(Ar_3 >= tar0)&(Ar_3 <= tar1)] - f_SB(Ar_3[(Ar_3 >= tar0)&(Ar_3 <= tar1)])), ls = '--', color = 'g', alpha = 0.5)

		ax.set_xscale('log')
		ax.set_xlabel('$R[arcsec]$')
		ax.set_ylabel('$SB[mag/arcsec^2]$')
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')
		ax.invert_yaxis()
		ax.legend(loc = 1, fontsize = 15)
		ax.set_title('stacked SB with resample correct')
		bx.set_xlabel('$R[arcsec]$')
		bx.set_xscale('log')
		bx.set_ylabel('$ \Delta{SB}[mag/arcsec^2] $')
		bx.tick_params(axis = 'both', which = 'both', direction = 'in')
		bx.legend(loc = 3, fontsize = 12)

		plt.subplots_adjust(hspace = 0)
		plt.tight_layout()
		plt.savefig('/home/xkchen/mywork/ICL/code/stack_profile_correct.png', dpi = 300)
		plt.close()

	raise
	return

def stack_B():
	un_mask = 0.15
	load = '/home/xkchen/mywork/ICL/data/test_data/mask/'
	x0 = 2427
	y0 = 1765
	bins = 90
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	sum_grid = np.array(np.meshgrid(Nx, Ny))

	for ii in range(len(band)):
		tot_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		tot_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		p_count_total = np.zeros((len(Ny), len(Nx)), dtype = np.float)

		for jj in range(stack_N):
			ra_g = ra[jj]
			dec_g = dec[jj]
			z_g = z[jj]
			Da_g = Test_model.angular_diameter_distance(z_g).value
			data = fits.getdata(load + 'B_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[ii], ra_g, dec_g, z_g), header = True)
			img = data[0]
			wcs = awc.WCS(data[1])
			cx, cy = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

			Angur = (R0*rad2asec/Da_g)
			Rp = Angur/pixel
			L_ref = Da_ref*pixel/rad2asec
			L_z0 = Da_g*pixel/rad2asec
			b = L_ref/L_z0
			Rref = (R0*rad2asec/Da_ref)/pixel

			f_goal = flux_recal(img, z_g, z_ref)
			xn, yn, resam = gen(f_goal, 1, b, cx, cy)
			xn = np.int(xn)
			yn = np.int(yn)
			if b > 1:
			    resam = resam[1:, 1:]
			elif b == 1:
			    resam = resam[1:-1, 1:-1]
			else:
			    resam = resam
			la0 = np.int(y0 - yn)
			la1 = np.int(y0 - yn + resam.shape[0])
			lb0 = np.int(x0 - xn)
			lb1 = np.int(x0 - xn + resam.shape[1])

			tot_array[la0:la1, lb0:lb1] = tot_array[la0:la1, lb0:lb1] + resam
			tot_count[la0: la1, lb0: lb1] = resam
			ia = np.where(tot_count != 0)
			p_count_total[ia[0], ia[1]] = p_count_total[ia[0], ia[1]] + 1
			tot_count[la0: la1, lb0: lb1] = 0

		mean_total = tot_array/p_count_total
		where_are_nan = np.isnan(mean_total)
		mean_total[where_are_nan] = 0

		SB_tot, R_tot, Ar_tot, error_tot = light_measure(mean_total, bins, 1, Rpp, x0, y0, pixel, z_ref)
		SB_TT = SB_tot[1:] + mag_add[ii]
		R_TT = R_tot[1:]
		Ar_TT = Ar_tot[1:]
		err_TT = error_tot[1:]

		stack_B = np.array([SB_TT, R_TT, Ar_TT, err_TT])
		with h5py.File('/home/xkchen/mywork/ICL/data/test_data/SB_stack_Bmask_%s_band.h5' % band[ii], 'w') as f:
			f['a'] = np.array(stack_B)
		with h5py.File('/home/xkchen/mywork/ICL/data/test_data/SB_stack_Bmask_%s_band.h5' % band[ii]) as f:
			for tt in range(len(stack_B)):
				f['a'][tt,:] = stack_B[tt,:]
		'''
		plt.figure()
		gf = plt.imshow(mean_total, cmap = 'Greys', origin = 'lower', vmin = 1e-3, norm = mpl.colors.LogNorm())
		plt.colorbar(gf, fraction = 0.036, pad = 0.01, label = '$f[nmagy]$')
		hsc.circles(x0, y0, s = Rpp, fc = '', ec = 'b', ls = '-', lw = 0.5)
		hsc.circles(x0, y0, s = 1.1*Rpp,  fc = '', ec = 'b', ls = '--', lw = 0.5)
		plt.xlim(x0 - 1.2*Rpp, x0 + 1.2*Rpp)
		plt.ylim(y0 - 1.2*Rpp, y0 + 1.2*Rpp)
		plt.subplots_adjust(left = 0.01, right = 0.85)
		plt.title('stack %.0f mean image in %s band' % (stack_N, band[ii]))
		plt.savefig('/home/xkchen/mywork/ICL/code/stack_mask_B_%s_band.png' % band[ii], dpi = 600)
		plt.close()
		'''
	return

def SB_fit(r, m0, Mc, c, M2L):
	skyl = m0
	surf_mass = sigmamc(r, Mc, c)
	surf_lit = surf_mass / M2L

	# case 1
	#mock_SB = 21.572 + 4.75 - 2.5*np.log10(10**(-6)*surf_lit) + 10*np.log10(1 + z_ref)

	# case 2
	Lz = surf_lit / ((1 + z_ref)**4 * np.pi * 4 * rad2asec**2)
	Lob = Lz * Lsun / kpc2cm**2
	mock_SB = 22.5 - 2.5 * np.log10(Lob/(10**(-9)*f0))

	mock_L = mock_SB + skyl

	return mock_L

def SB_ICL():
	f_unmask = 0.15
	with h5py.File('/home/xkchen/mywork/ICL/data/test_data/sky_light.h5') as f:
		SB_sky = np.array(f['a'])

	for ii in range(len(band)):
		with h5py.File('/home/xkchen/mywork/ICL/data/test_data/SB_stack_Amask_%s_band.h5' % band[ii]) as f:
			A_stack = np.array(f['a'])
		SB_diff = A_stack[0,:]
		R_diff = A_stack[1,:]
		Ar_diff = A_stack[2,:]
		err_diff = A_stack[3,:]

		with h5py.File('/home/xkchen/mywork/ICL/data/test_data/SB_stack_Bmask_%s_band.h5' % band[ii]) as f:
			B_stack = np.array(f['a'])
		SB_tot = B_stack[0,:]
		R_tot = B_stack[1,:]
		Ar_tot = B_stack[2,:]
		err_tot = B_stack[3,:]

		SB_ICL = SB_diff/(1 - f_unmask) - SB_tot * f_unmask/(1 - f_unmask)
		# fit the light profile
		ix = R_diff >= 100
		iy = R_diff <= 900
		iz = ix & iy
		r_fit = R_diff[iz]
		sb_fit = SB_diff[iz]

		m0 = SB_sky[ii]
		mc = 14
		cc = 4
		m2l = 120
		po = np.array([m0, mc, cc, m2l])
		popt, pcov = curve_fit(SB_fit, r_fit, sb_fit, p0 = po, method = 'trf')

		M0 = popt[0]
		Mc = popt[1]
		Cc = popt[2]
		M2L = popt[3]

		print('*'*10)
		print('m0 = ', M0)
		print('Mc = ', Mc)
		print('C = ', Cc)
		print('M2L = ', M2L)

		fit_line = SB_fit(r_fit, M0, Mc, Cc, M2L)

		fig = plt.figure(figsize = (16, 9))
		bx = plt.subplot(111)
		bx.set_title('$fit \; for \; background \; estimate \; in \; %s \; band$' % band[ii])
		bx.errorbar(R_diff[iz], SB_diff[iz], yerr = err_diff[iz], xerr = None, ls = '', fmt = 'ro', label = '$SB_{obs}$')
		bx.plot(r_fit, fit_line, 'b-', label = '$NFW+C$')
		bx.set_xlabel('$R[kpc]$')
		bx.set_xscale('log')
		bx.set_ylabel('$SB[mag/arcsec^2]$')
		bx.tick_params(axis = 'x', which = 'both', direction = 'in')
		bx.invert_yaxis()
		subax = fig.add_axes([0.15, 0.15, 0.2, 0.2])
		subax.errorbar(R_diff, SB_diff, yerr = err_diff, xerr = None, ls = '', fmt = 'ro', label = '$SB_{obs}$')
		subax.set_xscale('log')
		subax.tick_params(axis = 'x', which = 'both', direction = 'in')
		subax.invert_yaxis()
		plt.savefig('/home/xkchen/mywork/ICL/code/fit_for_BG_%s_band.png' % band[ii], dpi = 600)
		plt.close()

	raise
	return

def main():
	#mask_part()
	#mask_A()
	stack_A()
	#mask_B()
	#stack_B()
	SB_ICL()

if __name__ == "__main__":
	main()
