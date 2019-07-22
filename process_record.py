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
from light_measure import light_measure, flux_recal, flux_scale
from light_measure import sigmamc
from resample_modelu import down_samp

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

stack_N = 10
def stack_no_mask():
	x0 = 2427
	y0 = 1765
	bins = 50
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	sum_grid = np.array(np.meshgrid(Nx, Ny))
	load = '/home/xkchen/mywork/ICL/data/total_data/sample_02_03/'
	for ii in range(len(band)):
		sum_array_0 = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		count_array_0 = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
		p_count_0 = np.zeros((len(Ny), len(Nx)), dtype = np.float)

		SB_ref = []
		Ar_ref = []
		for jj in range(stack_N):
			ra_g = ra[jj]
			dec_g = dec[jj]
			z_g = z[jj]
			Da_g = Test_model.angular_diameter_distance(z_g).value

			data = fits.getdata(load + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[ii], ra_g, dec_g, z_g), header = True)
			img = data[0]
			wcs = awc.WCS(data[1])
			cx_BCG, cy_BCG = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

			Angur = (R0*rad2asec/Da_g)
			Rp = Angur/pixel
			L_ref = Da_ref*pixel/rad2asec
			L_z0 = Da_g*pixel/rad2asec
			eta = L_ref/L_z0
			miu = 1 / eta
			Rref = (R0*rad2asec/Da_ref)/pixel

			SB_ll, R_ll, Ar_ll, error_ll = light_measure(img, bins, 1, Rp, cx_BCG, cy_BCG, pixel, z_g)
			L_SB = SB_ll[1:]
			L_R = R_ll[1:]
			L_Ar = Ar_ll[1:]
			L_erro = error_ll[1:]

			SB_set = L_SB - 10*np.log10((1 + z_g) / (1 + z_ref))
			Ar_set = L_Ar * miu
			SB_ref.append(SB_set)
			Ar_ref.append(Ar_set)

			cc_img = flux_recal(img, z_g, z_ref)
			xn, yn, resam = gen(cc_img, 1, eta, cx_BCG, cy_BCG)
			xn = np.int(xn)
			yn = np.int(yn)
			if eta > 1:
			    resam = resam[1:, 1:]
			elif eta == 1:
			    resam = resam[1:-1, 1:-1]
			else:
			    resam = resam

			la0 = np.int(y0 - yn)
			la1 = np.int(y0 - yn + resam.shape[0])
			lb0 = np.int(x0 - xn)
			lb1 = np.int(x0 - xn + resam.shape[1])

			sum_array_0[la0:la1, lb0:lb1] = sum_array_0[la0:la1, lb0:lb1] + resam
			count_array_0[la0: la1, lb0: lb1] = resam
			id_nan = np.isnan(count_array_0)
			id_fals = np.where(id_nan == False)
			p_count_0[id_fals] = p_count_0[id_fals] + 1
			count_array_0[la0: la1, lb0: lb1] = np.nan

		mean_array_0 = sum_array_0 / p_count_0
		where_are_inf = np.isinf(mean_array_0)
		mean_array_0[where_are_inf] = np.nan
		id_zeros = np.where(p_count_0 == 0)
		mean_array_0[id_zeros] = np.nan

		SB, R, Ar, error = light_measure(mean_array_0, bins, 1, Rpp, x0, y0, pixel, z_ref)
		SB_0 = SB[1:] + mag_add[ii]
		R_0 = R[1:]
		Ar_0 = Ar[1:]
		err_0 = error[1:]

		ll0 = [np.max(kk / np.max(kk)) for kk in Ar_ref]
		tar1 = np.min(ll0)
		ll1 = [np.min(kk / np.max(kk)) for kk in Ar_ref]
		tar0 = np.max(ll1)

		tar_down = tar0 * Angu_ref
		tar_up = tar1 * Angu_ref
		inter_frac = np.logspace(np.log10(tar0 * 1.01), np.log10(tar1 / 1.01), bins)
		inter_ar = inter_frac * Angu_ref

		m_flux = []
		for pp in range(len(SB_ref)):
			tsb = SB_ref[pp]
			tar = Ar_ref[pp] / np.max(Ar_ref[pp])
			t_flux = 10**((22.5 - tsb) / 2.5)
			flux_ff = interp(tar, t_flux, kind = 'cubic')
			mflux = flux_ff(inter_frac)
			m_flux.append(mflux)

		inter_flux = np.mean(m_flux, axis = 0)
		inter_SB = 22.5 - 2.5 * np.log10(inter_flux)
		f_SB = interp(inter_ar, inter_SB, kind = 'cubic')
		Ar_0 = (Ar_0 / np.max(Ar_0)) * Angu_ref

		plt.figure(figsize = (16, 8))
		gs = gridspec.GridSpec(1, 2, width_ratios = [1,1])
		ax = plt.subplot(gs[0])
		bx = plt.subplot(gs[1])

		ax.plot(inter_ar, inter_SB, 'r--', label = '$SB_{ref}$', alpha = 0.5)
		ax.plot(inter_ar, inter_SB, 'ro')
		ax.plot(Ar_0, SB_0, 'b-', label = '$SB_{stack} \, no \, correct$', alpha = 0.5)
		ax.plot(Ar_0, SB_0, 'b*')

		bx.plot(Ar_0[(Ar_0 > tar_down) & (Ar_0 < tar_up)], SB_0[(Ar_0 > tar_down) & (Ar_0 < tar_up)] - 
			f_SB(Ar_0[(Ar_0 > tar_down) & (Ar_0 < tar_up)]), 'g*', alpha = 0.5)
		bx.axhline(y = np.mean(SB_0[(Ar_0 > tar_down) & (Ar_0 < tar_up)] - 
			f_SB(Ar_0[(Ar_0 > tar_down) & (Ar_0 < tar_up)])), ls = '--', color = 'b', alpha = 0.5)

		ax.set_xscale('log')
		ax.set_xlabel('$R[arcsec]$')
		ax.set_ylabel('$SB[mag/arcsec^2]$')
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')
		ax.invert_yaxis()
		ax.legend(loc = 1, fontsize = 12)
		ax.set_title('stacked SB with no resample correct')

		bx.set_xlabel('$R[arcsec]$')
		bx.set_xscale('log')
		bx.set_ylabel('$ \Delta{SB}[mag/arcsec^2] $')
		bx.tick_params(axis = 'both', which = 'both', direction = 'in')

		plt.subplots_adjust(hspace = 0)
		plt.tight_layout()
		plt.savefig('/home/xkchen/mywork/ICL/code/stack_test_%s_band.png' % band[ii], dpi = 300)
		plt.close()		
	raise
	return

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
			#jx = (-1)*jx+1
			#mask_A[lb0: lb1, la0: la1] = mask_A[lb0: lb1, la0: la1] * jx

			iu = np.where(jx == True)
			iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float)
			iv[iu] = np.nan
			mask_A[lb0: lb1, la0: la1] = mask_A[lb0: lb1, la0: la1] * iv

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
				#jx = (-1)*jx+1
				#mask_B[lb0: lb1, la0: la1] = mask_B[lb0: lb1, la0: la1]*jx

				iu = np.where(jx == True)
				iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float)
				iv[iu] = np.nan
				mask_B[lb0: lb1, la0: la1] = mask_B[lb0: lb1, la0: la1] * iv

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

	r_star = 2*1.5/pixel #mask star radius
	load = '/home/xkchen/mywork/ICL/data/total_data/sample_02_03/'

	param_A = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A.sex'
	out_cat = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A.param'
	out_load_A = '/home/xkchen/mywork/ICL/data/SEX/result/mask_A_test.cat'
	out_load_B = '/home/xkchen/mywork/ICL/data/SEX/result/mask_B_test.cat'
	out_load_sky = '/home/xkchen/mywork/ICL/data/SEX/result/mask_sky_test.cat'

	red_rich = Lambd[(Lambd >= 25) & (Lambd <= 27.5)]
	red_z = z[(Lambd >= 25) & (Lambd <= 27.5)]
	red_ra = ra[(Lambd >= 25) & (Lambd <= 27.5)]
	red_dec = dec[(Lambd >= 25) & (Lambd <= 27.5)]

	for i in range(len(band)):
		for q in range(stack_N):
			'''
			ra_g = ra[q]
			dec_g = dec[q]
			z_g = z[q]
			'''
			ra_g = red_ra[q]
			dec_g = red_dec[q]
			z_g = red_z[q]
			
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

			hdu = fits.PrimaryHDU()
			hdu.data = data_f[0].data
			hdu.header = head_inf
			hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/' + 'source_data_ra%.3f_dec%.3f.fits'%(ra_g, dec_g), overwrite = True)
			file_source = '/home/xkchen/mywork/ICL/data/test_data/' + 'source_data_ra%.3f_dec%.3f.fits'%(ra_g, dec_g)
			cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s'%(param_A,
				'/home/xkchen/mywork/ICL/data/SEX/result/mask_A_%.3fra_%.3fdec_%.3fz_%s_band.cat' % (ra_g, dec_g, z_g, band[i]), out_cat)
			print(cmd)
			A = subpro.Popen(cmd, shell = True)
			A.wait()

			source = asc.read('/home/xkchen/mywork/ICL/data/SEX/result/mask_A_%.3fra_%.3fdec_%.3fz_%s_band.cat' % (ra_g, dec_g, z_g, band[i]))
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
					#jx = (-1)*jx+1
					#mask_A[lb0: lb1, la0: la1] = mask_A[lb0: lb1, la0: la1]*jx

					iu = np.where(jx == True)
					iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float)
					iv[iu] = np.nan
					mask_A[lb0: lb1, la0: la1] = mask_A[lb0: lb1, la0: la1] * iv

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
		tot_count = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
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
			id_nan = np.isnan(tot_count)
			id_fals = np.where(id_nan == False)
			p_count_total[id_fals] = p_count_total[id_fals] + 1
			tot_count[la0: la1, lb0: lb1] = np.nan

		mean_total = tot_array / p_count_total
		where_are_inf = np.isinf(mean_total)
		mean_total[where_are_inf] = np.nan
		id_zeros = np.where(p_count_total == 0)
		mean_total[id_zeros] = np.nan

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

def stack_A():
	un_mask = 0.15
	r_star = 2*1.5/pixel
	load = '/home/xkchen/mywork/ICL/data/test_data/mask/'
	x0 = 2427
	y0 = 1765
	bins = 50
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	sum_grid = np.array(np.meshgrid(Nx, Ny))

	red_rich = Lambd[(Lambd >= 25) & (Lambd <= 27.5)]
	red_z = z[(Lambd >= 25) & (Lambd <= 27.5)]
	red_ra = ra[(Lambd >= 25) & (Lambd <= 27.5)]
	red_dec = dec[(Lambd >= 25) & (Lambd <= 27.5)]

	for ii in range(1):
		sum_array_0 = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		count_array_0 = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
		p_count_0 = np.zeros((len(Ny), len(Nx)), dtype = np.float)

		sum_array_s = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		count_array_s = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
		p_count_s = np.zeros((len(Ny), len(Nx)), dtype = np.float)

		sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
		p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)

		sum_array_D = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		count_array_D = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
		p_count_D = np.zeros((len(Ny), len(Nx)), dtype = np.float)	

		SB_ref = []
		Ar_ref = []

		flux_ori = []
		Ar_ori = []

		flux_cc = []
		SB_cc = []
		Ar_cc = []

		for jj in range(10):
			
			ra_g = ra[jj]
			dec_g = dec[jj]
			z_g = z[jj]
			'''
			ra_g = red_ra[jj]
			dec_g = red_dec[jj]
			z_g = red_z[jj]
			'''
			Da_g = Test_model.angular_diameter_distance(z_g).value
			data = fits.getdata(load + 'A_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[ii], ra_g, dec_g, z_g), header = True)
			img = data[0]
			wcs = awc.WCS(data[1])
			cx_BCG, cy_BCG = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

			Angur = (R0*rad2asec/Da_g)
			Rp = Angur / pixel
			L_ref = Da_ref*pixel/rad2asec
			L_z0 = Da_g*pixel/rad2asec
			Rref = (R0*rad2asec/Da_ref)/pixel

			eta = L_ref/L_z0
			miu = 1 / eta

			ox = np.linspace(0, img.shape[1]-1, img.shape[1])
			oy = np.linspace(0, img.shape[0]-1, img.shape[0])
			oo_grd = np.array(np.meshgrid(ox, oy))
			cdr = np.sqrt((oo_grd[0,:] - cx_BCG)**2 + (oo_grd[1,:] - cy_BCG)**2)
			idd = (cdr > Rp) & (cdr < 1.1 * Rp)
			cut_region = img[idd]
			id_nan = np.isnan(cut_region)
			idx = np.where(id_nan == False)
			sky_origin = np.mean(cut_region[idx])
			sky_SB = 22.5 - 2.5*np.log10(np.abs(sky_origin)) + 2.5*np.log10(pixel**2)
			cc_img = img - sky_origin

			pro_SB, pro_R, pro_Ar, pro_err = light_measure(img, bins, 1, Rp, cx_BCG, cy_BCG, pixel, z_g)
			pro_SB = pro_SB[1:]
			pro_R = pro_R[1:]
			pro_Ar = pro_Ar[1:]
			pro_err = pro_err[1:]

			################ test for background subtract compare
			p2s_flux = 10**((22.5 + 2.5*np.log10(pixel**2) - pro_SB) / 2.5) - sky_origin
			
			#fluxcc = flux_recal(p2s_flux, z_g, z_ref)
			fluxcc = flux_scale(p2s_flux, z_g, z_ref)
			flux_cc.append(fluxcc)
			'''
			SB_cet = 22.5 - 2.5 * np.log10(p2s_flux) + 2.5 * np.log10(pixel**2)
			SBc = SB_cet - 10*np.log10((1 + z_g) / (1 + z_ref))
			SB_cc.append(SBc)
			'''
			Arcc = pro_Ar * miu
			Ar_cc.append(Arcc)
			###############

			SB_ll, R_ll, Ar_ll, error_ll = light_measure(cc_img, bins, 1, Rp, cx_BCG, cy_BCG, pixel, z_g)
			L_SB = SB_ll[1:]
			L_R = R_ll[1:]
			L_Ar = Ar_ll[1:]
			L_err = error_ll[1:]

			SB_set = L_SB - 10*np.log10((1 + z_g) / (1 + z_ref))
			Ar_set = L_Ar * miu
			SB_ref.append(SB_set)
			Ar_ref.append(Ar_set)

			#f_goal = flux_recal(img, z_g, z_ref)
			f_goal = flux_scale(img, z_g, z_ref)
			xn, yn, resam = gen(f_goal, 1, eta, cx_BCG, cy_BCG)
			xn = np.int(xn)
			yn = np.int(yn)
			if eta > 1:
				resam = resam[1:, 1:]
			elif eta == 1:
				resam = resam[1:-1, 1:-1]
			else:
				resam = resam

			ox = np.linspace(0, resam.shape[1]-1, resam.shape[1])
			oy = np.linspace(0, resam.shape[0]-1, resam.shape[0])
			oo_grd = np.array(np.meshgrid(ox, oy))
			cdr = np.sqrt((oo_grd[0,:] - xn)**2 + (oo_grd[1,:] - yn)**2)
			idd = (cdr > Rpp) & (cdr < 1.1 * Rpp)
			cut_res = resam[idd]
			id_nan = np.isnan(cut_res)
			idx = np.where(id_nan == False)
			back_lel = np.mean(cut_res[idx])
			'''
			print('before resample', sky_origin)
			print('After resample', back_lel)
			print('scale factor', (1+z_ref)**4/(1+z_g)**4)
			print('ratio ', back_lel / sky_origin)
			'''
			la0 = np.int(y0 - yn)
			la1 = np.int(y0 - yn + resam.shape[0])
			lb0 = np.int(x0 - xn)
			lb1 = np.int(x0 - xn + resam.shape[1])

			idx = np.isnan(resam)
			idv = np.where(idx == False)
			sum_array_0[la0:la1, lb0:lb1][idv] = sum_array_0[la0:la1, lb0:lb1][idv] + (resam - back_lel)[idv]
			count_array_0[la0: la1, lb0: lb1][idv] = resam[idv]
			id_nan = np.isnan(count_array_0)
			id_fals = np.where(id_nan == False)
			p_count_0[id_fals] = p_count_0[id_fals] + 1
			count_array_0[la0: la1, lb0: lb1][idv] = np.nan

			############# check resample process
			SB_res = pro_SB - 10*np.log10((1 + z_g) / (1 + z_ref))
			flux_res = 10**((22.5 + 2.5*np.log10(pixel**2) - SB_res) / 2.5) - back_lel
			Ar_res = pro_Ar * miu
			flux_ori.append(flux_res)
			Ar_ori.append(Ar_res)

			############# stack the image without subtract background
			idx = np.isnan(resam)
			idv = np.where(idx == False)
			sum_array_A[la0:la1, lb0:lb1][idv] = sum_array_A[la0:la1, lb0:lb1][idv] + resam[idv]
			count_array_A[la0: la1, lb0: lb1][idv] = resam[idv]
			id_nan = np.isnan(count_array_A)
			id_fals = np.where(id_nan == False)
			p_count_A[id_fals] = p_count_A[id_fals] + 1
			count_array_A[la0: la1, lb0: lb1][idv] = np.nan			

			############# stack the image with background subtraction
			#f_D = flux_recal(cc_img, z_g, z_ref)
			f_D = flux_scale(cc_img, z_g, z_ref)
			xnd, ynd, resam_dd = gen(f_D, 1, eta, cx_BCG, cy_BCG)
			xnd = np.int(xnd)
			ynd = np.int(ynd)
			if eta > 1:
				resam_d = resam_dd[1:, 1:]
			elif eta == 1:
				resam_d = resam_dd[1:-1, 1:-1]
			else:
				resam_d = resam_dd

			la0 = np.int(y0 - ynd)
			la1 = np.int(y0 - ynd + resam_d.shape[0])
			lb0 = np.int(x0 - xnd)
			lb1 = np.int(x0 - xnd + resam_d.shape[1])

			idx = np.isnan(resam_d)
			idv = np.where(idx == False)
			sum_array_D[la0:la1, lb0:lb1][idv] = sum_array_D[la0:la1, lb0:lb1][idv] + resam_d[idv]
			count_array_D[la0: la1, lb0: lb1][idv] = resam_d[idv]
			id_nan = np.isnan(count_array_D)
			id_fals = np.where(id_nan == False)
			p_count_D[id_fals] = p_count_D[id_fals] + 1
			count_array_D[la0: la1, lb0: lb1][idv] = np.nan	

			############# reload souce and mask
			'''
			s_load = '/home/xkchen/mywork/ICL/data/total_data/sample_02_03/'
			file = 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[ii], ra_g, dec_g, z_g)
			data_f = fits.open(s_load + file)
			s_img = data_f[0].data
			x_side = s_img.shape[1]
			y_side = s_img.shape[0]

			xs = np.linspace(0, x_side - 1, x_side)
			ys = np.linspace(0, y_side - 1, y_side)
			img_grid = np.array(np.meshgrid(xs, ys))

			ra_img, dec_img = wcs.all_pix2world(img_grid[0,:], img_grid[1,:], 1)
			pos = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
			BEV = sfd(pos)
			Av = Rv * BEV
			Al = A_wave(l_wave[ii], Rv) * Av
			s_img = s_img * 10**(Al / 2.5)

			simg = flux_recal(s_img, z_g, z_ref)
			xns, yns, s_img_resam = gen(simg, 1, eta, cx_BCG, cy_BCG)
			xns = np.int(xns)
			yns = np.int(yns)
			if eta > 1:
				s_resam = s_img_resam[1:, 1:]
			elif eta == 1:
				s_resam = s_img_resam[1:-1, 1:-1]
			else:
				s_resam = s_img_resam

			source = asc.read('/home/xkchen/mywork/ICL/data/SEX/result/mask_A_%.3fra_%.3fdec_%.3fz_%s_band.cat' % (ra_g, dec_g, z_g, band[ii]))
			Numb = np.array(source['NUMBER'][-1])
			A = np.array(source['A_IMAGE'])
			B = np.array(source['B_IMAGE'])
			theta = np.array(source['THETA_IMAGE'])
			cx = np.array(source['X_IMAGE']) - 1
			cy = np.array(source['Y_IMAGE']) - 1
			Kron = 6
			a = Kron*A
			b = Kron*B

			CX = cx * miu
			CY = cy * miu
			a_ = a * miu
			b_ = b * miu
			res_mask_1 = np.ones((s_resam.shape[0], s_resam.shape[1]), dtype = np.float)
			ox_ = np.linspace(0, s_resam.shape[1] - 1, s_resam.shape[1])
			oy_ = np.linspace(0, s_resam.shape[0] - 1, s_resam.shape[0])
			basic_coord = np.array(np.meshgrid(ox_, oy_))
			major = a_ / 2
			minor = b_ / 2
			senior = np.sqrt(major**2 - minor**2)

			tdr = np.sqrt((CX - xns)**2 + (CY - yns)**2)
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

					iu = np.where(jx == True)
					iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float)
					iv[iu] = np.nan
					res_mask_1[lb0: lb1, la0: la1] = res_mask_1[lb0: lb1, la0: la1] * iv
			ss_resam = res_mask_1 * s_resam
			
			ox = np.linspace(0, ss_resam.shape[1]-1, ss_resam.shape[1])
			oy = np.linspace(0, ss_resam.shape[0]-1, ss_resam.shape[0])
			oo_grd = np.array(np.meshgrid(ox, oy))
			cdr = np.sqrt((oo_grd[0,:] - xns)**2 + (oo_grd[1,:] - yns)**2)
			idd = (cdr > Rpp) & (cdr < 1.1 * Rpp)
			cut_res = ss_resam[idd]
			id_nan = np.isnan(cut_res)
			idx = np.where(id_nan == False)
			ss_back = np.mean(cut_res[idx])

			la0 = np.int(y0 - yns)
			la1 = np.int(y0 - yns + ss_resam.shape[0])
			lb0 = np.int(x0 - xns)
			lb1 = np.int(x0 - xns + ss_resam.shape[1])

			idx = np.isnan(ss_resam)
			idv = np.where(idx == False)
			sum_array_s[la0:la1, lb0:lb1][idv] = sum_array_s[la0:la1, lb0:lb1][idv] + (ss_resam - ss_back)[idv]
			count_array_s[la0: la1, lb0: lb1][idv] = ss_resam[idv]
			id_nan = np.isnan(count_array_s)
			id_fals = np.where(id_nan == False)
			p_count_s[id_fals] = p_count_s[id_fals] + 1
			count_array_s[la0: la1, lb0: lb1][idv] = np.nan
			'''
		mean_array_0 = sum_array_0 / p_count_0
		where_are_inf = np.isinf(mean_array_0)
		mean_array_0[where_are_inf] = np.nan
		id_zeros = np.where(p_count_0 == 0)
		mean_array_0[id_zeros] = np.nan

		SB, R, Ar, error = light_measure(mean_array_0, bins, 1, Rpp, x0, y0, pixel, z_ref)
		SB_0 = SB[1:] + mag_add[ii]
		R_0 = R[1:]
		Ar_0 = Ar[1:]
		err_0 = error[1:]

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

		inter_ar = inter_ar[id_set]
		inter_flux = inter_flux[id_set]
		inter_SB = 22.5 - 2.5 * np.log10(inter_flux) + 2.5*np.log10(pixel**2)
		f_SB = interp(inter_ar, inter_SB, kind = 'cubic')
		Ar0 = (Ar_0 / Angu_ref) * Angu_ref

		############# test for background subtract compare
		flux_obs = np.ones((stack_N, bins), dtype = np.float) * np.nan
		for pp in range(stack_N):
			id_count = np.zeros(bins, dtype = np.float)
			tflux = flux_cc[pp]
			#tflux = 10**((22.5 + 2.5*np.log10(pixel**2) - SB_cc[pp]) / 2.5)
			tar = Ar_cc[pp] / Angu_ref
			for kk in range(len(tar)):
				sub_ar = np.abs(inter_frac - tar[kk])
				id_min = np.where(sub_ar == np.min(sub_ar))[0]
				id_count[id_min[0]] = id_count[id_min[0]] + 1
			id_nuzero = id_count != 0
			id_g = np.where(id_nuzero == True)[0]
			flux_obs[pp, id_g] = tflux

		count_obs = np.zeros(bins, dtype = np.float)
		mflux_obs = np.zeros(bins, dtype = np.float)
		for pp in range(bins):
			sub_flux = flux_obs[:, pp]
			iy = np.isnan(sub_flux)
			iv = np.where(iy == False)[0]
			count_obs[pp] = len(iv)
			mflux_obs[pp] = mflux_obs[pp] + np.sum(sub_flux[iv])
		mflux = mflux_obs / count_obs

		id_nan = np.isnan(mflux)
		id_x = id_nan == False
		id_inf = np.isinf(mflux)
		id_y = id_inf == False
		id_zero = mflux == 0
		id_z = id_zero == False
		id_set = id_x & id_y & id_z

		obs_ar = inter_ar[id_set]
		obs_flux = mflux[id_set]
		obs_SB = 22.5 - 2.5 * np.log10(obs_flux) + 2.5*np.log10(pixel**2)

		############# reload souce and mask
		'''
		mean_array_s = sum_array_s / p_count_s
		where_are_inf = np.isinf(mean_array_s)
		mean_array_s[where_are_inf] = np.nan
		id_zeros = np.where(p_count_s == 0)
		mean_array_s[id_zeros] = np.nan

		s_SB, s_R, s_Ar, s_error = light_measure(mean_array_s, bins, 1, Rpp, x0, y0, pixel, z_ref)
		SB_s = s_SB[1:] + mag_add[ii]
		R_s = s_R[1:]
		Ar_s = s_Ar[1:]
		err_s = s_error[1:]
		'''
		##########subtract background after stacking
		mean_array_A = sum_array_A / p_count_A
		where_are_inf = np.isinf(mean_array_A)
		mean_array_A[where_are_inf] = np.nan
		id_zeros = np.where(p_count_A == 0)
		mean_array_A[id_zeros] = np.nan

		ox = np.linspace(0, len(Nx)-1, len(Nx))
		oy = np.linspace(0, len(Ny)-1, len(Ny))
		oo_grd = np.array(np.meshgrid(ox, oy))
		cdr = np.sqrt((oo_grd[0,:] - x0)**2 + (oo_grd[1,:] - y0)**2)
		idd = (cdr > Rpp) & (cdr < 1.1 * Rpp)
		cut_res = mean_array_A[idd]
		id_nan = np.isnan(cut_res)
		idx = np.where(id_nan == False)
		cc_back = np.mean(cut_res[idx])
		cc_resam = mean_array_A - cc_back

		cc_SB, cc_R, cc_Ar, cc_error = light_measure(cc_resam, bins, 1, Rpp, x0, y0, pixel, z_ref)
		c_SB = cc_SB[1:] + mag_add[ii]
		c_R = cc_R[1:]
		c_Ar = cc_Ar[1:]
		c_err = cc_error[1:]

		############# check resample process
		flux_com = np.ones((stack_N, bins), dtype = np.float) * np.nan
		for pp in range(len(Ar_ori)):
			id_count = np.zeros(bins, dtype = np.float)
			tflux = flux_ori[pp]
			tar = Ar_ori[pp] / Angu_ref
			tt_flux = 10**((22.5 + 2.5*np.log10(pixel**2) - tsb) / 2.5)
			for kk in range(len(tar)):
				sub_ar = np.abs(inter_frac - tar[kk])
				id_min = np.where(sub_ar == np.min(sub_ar))[0]
				id_count[id_min[0]] = id_count[id_min[0]] + 1
			id_nuzero = id_count != 0
			id_g = np.where(id_nuzero == True)[0]
			flux_com[pp, id_g] = tflux

		ori_count = np.zeros(bins, dtype = np.float)
		ori_flux = np.zeros(bins, dtype = np.float)
		for pp in range(bins):
			sub_flux = flux_com[:, pp]
			iy = np.isnan(sub_flux)
			iv = np.where(iy == False)[0]
			ori_count[pp] = len(iv)
			ori_flux[pp] = ori_flux[pp] + np.sum(sub_flux[iv])
		com_flux = ori_flux / ori_count

		id_nan = np.isnan(com_flux)
		id_x = id_nan == False
		id_inf = np.isinf(com_flux)
		id_y = id_inf == False
		id_zero = com_flux == 0
		id_z = id_zero == False
		id_set = id_x & id_y & id_z

		com_ar = inter_ar[id_set]
		comflux = com_flux[id_set]
		com_SB = 22.5 - 2.5 * np.log10(comflux) + 2.5*np.log10(pixel**2)

		############ stack subtracted image
		mean_array_D = sum_array_D / p_count_D
		where_are_inf = np.isinf(mean_array_D)
		mean_array_D[where_are_inf] = np.nan
		id_zeros = np.where(p_count_D == 0)
		mean_array_D[id_zeros] = np.nan

		SB, R, Ar, error = light_measure(mean_array_D, bins, 1, Rpp, x0, y0, pixel, z_ref)
		SB_1 = SB[1:] + mag_add[ii]
		R_1 = R[1:]
		Ar_1 = Ar[1:]
		err_1 = error[1:]
		Ar1 = (Ar_1 / Angu_ref) * Angu_ref
		###################

		plt.figure(figsize = (16, 8))
		gs = gridspec.GridSpec(1, 2, width_ratios = [1,1])
		ax = plt.subplot(gs[0])
		bx = plt.subplot(gs[1])

		ax.plot(inter_ar, inter_SB, 'r-', label = '$SB_{ref}$', alpha = 0.5)
		ax.plot(com_ar, com_SB, 'k--', label = '$SB_{check \; resample}$', alpha = 0.5)
		ax.plot(obs_ar, obs_SB, 'y--', label = '$SB_{BL \; comparation}$', alpha = 0.5)

		ax.plot(Ar0, SB_0, 'b*-', label = '$SB_{stack \, corrected}$', alpha = 0.5)
		ax.plot(Ar1, SB_1, 'g*--', label = '$SB_{stack \, subtracted \, img}$', alpha = 0.5)
		ax.plot(c_Ar, c_SB, 'm*:', label = '$SB_{subtract \, BL \, from \, stacked \, img}$', alpha = 0.5)

		bx.plot(Ar0[(Ar0 > tar_down * 1.01) & (Ar0 < tar_up / 1.01)], SB_0[(Ar0 > tar_down * 1.01) & (Ar0 < tar_up / 1.01)] - 
			f_SB(Ar0[(Ar0 > tar_down * 1.01) & (Ar0 < tar_up / 1.01)]), 'b*', alpha = 0.5)
		bx.axhline(y = np.mean(SB_0[(Ar0 > tar_down * 1.01) & (Ar0 < tar_up / 1.01)] - 
			f_SB(Ar0[(Ar0 > tar_down * 1.01) & (Ar0 < tar_up / 1.01)])), ls = '--', color = 'b', alpha = 0.5)
		
		bx.plot(Ar1[(Ar1 > tar_down * 1.01) & (Ar1 < tar_up / 1.01)], SB_1[(Ar1 > tar_down * 1.01) & (Ar1 < tar_up / 1.01)] - 
			f_SB(Ar1[(Ar1 > tar_down * 1.01) & (Ar1 < tar_up / 1.01)]), 'g*', alpha = 0.5)
		bx.axhline(y = np.mean(SB_1[(Ar1 > tar_down * 1.01) & (Ar1 < tar_up / 1.01)] - 
			f_SB(Ar1[(Ar1 > tar_down * 1.01) & (Ar1 < tar_up / 1.01)])), ls = '--', color = 'g', alpha = 0.5)
		
		bx.plot(c_Ar[(c_Ar > tar_down * 1.01) & (c_Ar < tar_up / 1.01)], c_SB[(c_Ar > tar_down * 1.01) & (c_Ar < tar_up / 1.01)] - 
			f_SB(c_Ar[(c_Ar > tar_down * 1.01) & (c_Ar < tar_up / 1.01)]), 'm*', alpha = 0.5)
		bx.axhline(y = np.mean(c_SB[(c_Ar > tar_down * 1.01) & (c_Ar < tar_up / 1.01)] - 
			f_SB(c_Ar[(c_Ar > tar_down * 1.01) & (c_Ar < tar_up / 1.01)])), ls = '--', color = 'm', alpha = 0.5)

		ax.set_xscale('log')
		ax.set_xlabel('$R[arcsec]$')
		ax.set_ylabel('$SB[mag/arcsec^2]$')
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')
		ax.invert_yaxis()
		ax.legend(loc = 1, fontsize = 12)
		ax.set_title('stacked SB with resample correction')
		ax.set_xlim(3e-1, 2.5e2)

		bx.set_title('$SB_{measured} - SB_{ref} \; comparation$')
		bx.set_xlabel('$R[arcsec]$')
		bx.set_xscale('log')
		bx.set_ylabel('$ \Delta{SB}[mag/arcsec^2] $')
		bx.tick_params(axis = 'both', which = 'both', direction = 'in')
		bx.set_xlim(3e-1, 2.5e2)

		plt.subplots_adjust(hspace = 0)
		plt.tight_layout()
		plt.savefig('/home/xkchen/mywork/ICL/code/stack_profile_%s_band.png' % band[ii], dpi = 300)
		#plt.savefig('/home/xkchen/mywork/ICL/code/stack_profile_origin_%s_band_sample_%d.png' % (band[ii], jj), dpi = 300)
		#plt.savefig('/home/xkchen/mywork/ICL/code/stack_profile_origin_%s_band_subtract_%d.png' % (band[ii], jj), dpi = 300)
		plt.close()

	raise
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
	#stack_no_mask()
	#mask_part()
	#mask_A()
	stack_A()
	#mask_B()
	#stack_B()
	#SB_ICL()

if __name__ == "__main__":
	main()
