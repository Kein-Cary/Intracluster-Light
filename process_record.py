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
from scipy.optimize import curve_fit, minimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord

from resamp import gen
from extinction_redden import A_wave
from light_measure import light_measure, flux_recal, flux_scale
from light_measure import sigmamc

import time
import random
import sfdmap
from matplotlib.patches import Circle
# constant
m = sfdmap.SFDMap('/home/xkchen/mywork/ICL/data/redmapper/sfddata_maskin', scaling = 0.86)
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
kpc2m = U.kpc.to(U.m)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Msun2kg = U.M_sun.to(U.kg)

Lsun = C.L_sun.value*10**7
G = C.G.value
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
f0 = 3631 * Jy # zero point in unit (erg/s)/cm^-2

with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
	catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]

R0 = 1 # in unit Mpc
Angu_ref = (R0 / Da_ref) * rad2asec
Rpp = Angu_ref / pixel

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

stack_N = 20
def resam_test():
	load = '/home/xkchen/mywork/ICL/data/total_data/sample_02_03/'
	for jj in range(10):
		ra_g = ra[jj]
		dec_g = dec[jj]
		z_g = z[jj]
		data = fits.getdata(load + 
			'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % ('r', ra_g, dec_g, z_g), header = True)
		img = data[0]
		wcs = awc.WCS(data[1])
		cx_BCG, cy_BCG = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)
		Da_g = Test_model.angular_diameter_distance(z_g).value
		Angur = (R0 * rad2asec / Da_g)
		Rp = Angur / pixel
		L_ref = Da_ref * pixel / rad2asec
		L_z0 = Da_g * pixel / rad2asec
		eta = L_ref / L_z0
		'''
		# check total flux
		x0 = [random.randint(900, 1100) for _ in range(100)]
		y0 = [random.randint(700, 800) for _ in range(100)]
		cut_size = 200
		df2f0 = np.zeros(len(x0), dtype = np.float)
		for kk in range(len(x0)):
			sub_img = img[y0[kk] - 200: y0[kk] + 200, x0[kk] - 200: x0[kk] + 200]
			tot_f0 = np.sum(sub_img)
			xm, ym, new_sub = gen(sub_img, 1, eta, 200, 200)
			if eta > 1:
				new_smg = new_sub[1:, 1:]
			elif eta == 1:
				new_smg = new_sub[1:-1, 1:-1]
			else:
				new_smg = new_sub
			tot_f1 = np.sum(new_smg)
			df2f0[kk] = (tot_f1 - tot_f0) / tot_f0
		plt.figure()
		ax = plt.subplot(111)
		ax.hist(df2f0, bins = 20, histtype = 'step', color = 'b')
		ax.set_ylabel('Number of sample images')
		ax.set_xlabel('$f_{after \, resampling } - f_{before \, resampling} / f_{before \, resampling}$')
		plt.savefig('total_flux_test.png', dpi = 300)
		plt.close()
		'''
		'''
		# select points
		x0 = np.array([1365, 1004, 1809, 1813, 1803, 1816, 1777, 1777, 1780, 
						1787, 1031, 1031, 1011, 1005, 1003, 1019, 832, 838, 
						825, 821, 360, 375])
		y0 = np.array([198, 1002, 801, 804, 835, 815, 823, 817, 809, 
						801, 996, 1005, 1014, 1007, 1001, 984, 358, 
						362, 374, 369, 467, 463]) # points close to sources
		x1 = np.array([1456, 1505, 1069, 1074, 1692, 1097, 1088, 1680, 1487, 
						1485, 1502, 1514, 83, 90, 101, 102, 1202, 1218, 1221, 
						1187, 1198, 1209])
		y1 = np.array([1149, 230, 643, 636, 283, 632, 616, 286, 1102, 
						1096, 1088, 1100, 901, 894, 885, 915, 1048, 1041, 
						1033, 1053, 1038, 1028]) # points far away sources

		xn, yn, resam = gen(img, 1, eta, cx_BCG, cy_BCG)
		xn = np.int(xn)
		yn = np.int(yn)
		if eta > 1:
			resam = resam[1:, 1:]
		elif eta == 1:
			resam = resam[1:-1, 1:-1]
		else:
			resam = resam
		x0n = [np.int(kk / eta) for kk in x0]
		y0n = [np.int(kk / eta) for kk in y0]
		x1n = [np.int(kk / eta) for kk in x1]
		y1n = [np.int(kk / eta) for kk in y1]
		fract0 = (resam[y0n, x0n] / (eta * pixel)**2 - img[y0, x0] / pixel**2) / (img[y0, x0] / pixel**2)
		fract1 = (resam[y1n, x1n] / (eta * pixel)**2 - img[y1, x1] / pixel**2) / (img[y1, x1] / pixel**2)

		plt.figure()
		ax0 = plt.subplot(121)
		ax1 = plt.subplot(122)
		ax0.imshow(img, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
		ax0.scatter(x0, y0, s = 5, marker = 'o', facecolors = '', edgecolors = 'r', linewidth = 0.5)
		ax0.scatter(x1, y1, s = 5, marker = 's', facecolors = '', edgecolors = 'b', linewidth = 0.5)
		ax1.imshow(resam, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
		ax1.scatter(x0n, y0n, s = 5, marker = 'o', facecolors = '', edgecolors = 'r', linewidth = 0.5)
		ax1.scatter(x1n, y1n, s = 5, marker = 's', facecolors = '', edgecolors = 'b', linewidth = 0.5)

		plt.tight_layout()
		plt.savefig('points_select.png', dpi = 300)
		plt.close()

		plt.figure()
		ax = plt.subplot(111)
		ax.set_title('pixel SB changes')
		ax.hist(fract0, histtype = 'step', color = 'r', label = '$ pixels[close \; to \; source] $', alpha = 0.5)
		ax.hist(fract1, histtype = 'step', color = 'b', label = '$ pixels[from \; background] $', alpha = 0.5)
		ax.set_ylabel('Pixel Number')
		ax.set_xlabel('$ SB_{after \, resampling} - SB_{before \, resampling} / SB_{before \, resampling}$')
		ax.legend(loc = 2)
		plt.savefig('pixel_SB_change.png', dpi = 300)
		plt.close()
		'''
		# mean SB of pixel
		xn, yn, resam = gen(img, 1, eta, cx_BCG, cy_BCG)
		xn = np.int(xn)
		yn = np.int(yn)
		if eta > 1:
			resam = resam[1:, 1:]
		elif eta == 1:
			resam = resam[1:-1, 1:-1]
		else:
			resam = resam
		bins = 45
		Nx = np.linspace(0, img.shape[1] - 1, img.shape[1])
		Ny = np.linspace(0, img.shape[0] - 1, img.shape[0])
		ngrd = np.array(np.meshgrid(Nx, Ny))
		ndr = np.sqrt(( (2 * ngrd[0,:] + 1) / 2 - (2 * cx_BCG + 1) / 2)**2 + 
					( (2 * ngrd[1,:] + 1) / 2 - (2 * cy_BCG + 1) / 2)**2)
		ndr1 = ndr * pixel * Da_g * 10**3 / rad2asec

		Nx = np.linspace(0, resam.shape[1] - 1, resam.shape[1])
		Ny = np.linspace(0, resam.shape[0] - 1, resam.shape[0])
		ngrd = np.array(np.meshgrid(Nx, Ny))
		ndr = np.sqrt(( (2 * ngrd[0,:] + 1) / 2 - (2 * xn + 1) / 2)**2 + 
					( (2 * ngrd[1,:] + 1) / 2 - (2 * yn + 1) / 2)**2)
		ndr2 = ndr * pixel * Da_ref * 10**3 / rad2asec

		nr = np.linspace(0, Rpp, bins) * pixel * Da_ref * 10**3 / rad2asec

		fract = np.zeros(len(nr), dtype = np.float)
		for kk in range(len(nr) - 1):
			idr1 = (ndr1 >= nr[kk]) & (ndr1 < nr[kk + 1])
			subf1 = np.mean(img[idr1] / pixel**2)
			idr2 = (ndr2 >= nr[kk]) & (ndr2 < nr[kk + 1])
			subf2 = np.mean(resam[idr2] / (eta * pixel)**2)
			fract[kk + 1] = (subf2 - subf1) / subf1
		rbin = 0.5 * (nr[1:] + nr[:-1])

		fig = plt.figure()
		plt.title('mean pixel surface brightness variation')
		plt.plot(rbin, fract[1:], 'g*')
		plt.axhline(y = 0, linestyle = '--', color = 'r', label = 'y = 0')
		plt.legend(loc = 1)
		plt.xlabel('R[kpc]')
		plt.ylabel('$ SB_{after \, resampling} - SB_{before \, resampling} / SB_{before \, resampling}$')
		plt.savefig('mean_pixel_SB_%d.png' % jj, dpi = 300)
		plt.show()

	raise
	return

def stack_no_mask():
	x0 = 2427
	y0 = 1765
	bins = 50
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	sum_grid = np.array(np.meshgrid(Nx, Ny))
	load = '/home/xkchen/mywork/ICL/data/total_data/sample_02_03/'
	'''
	red_rich = Lambd[(Lambd >= 25) & (Lambd <= 27.5)]
	red_z = z[(Lambd >= 25) & (Lambd <= 27.5)]
	red_ra = ra[(Lambd >= 25) & (Lambd <= 27.5)]
	red_dec = dec[(Lambd >= 25) & (Lambd <= 27.5)]
	'''
	red_rich = Lambd[Lambd > 100]
	red_z = z[Lambd > 100]
	red_ra = ra[Lambd > 100]
	red_dec = dec[Lambd > 100]

	for ii in range(len(band)):
		sum_array_0 = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		count_array_0 = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
		p_count_0 = np.zeros((len(Ny), len(Nx)), dtype = np.float)

		SB_ref = []
		Ar_ref = []
		for jj in range(10):
			ra_g = red_ra[jj]
			dec_g = red_dec[jj]
			z_g = red_z[jj]
			Da_g = Test_model.angular_diameter_distance(z_g).value

			data = fits.getdata(load + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[ii], ra_g, dec_g, z_g), header = True)
			img = data[0]
			wcs = awc.WCS(data[1])
			cx_BCG, cy_BCG = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

			x_set = [random.randint(50, img.shape[1] - 51) for _ in range(1000)]
			y_set = [random.randint(50, img.shape[0] - 51) for _ in range(1000)]

			Angur = (R0*rad2asec/Da_g)
			Rp = Angur/pixel
			L_ref = Da_ref*pixel/rad2asec
			L_z0 = Da_g*pixel/rad2asec
			eta = L_ref/L_z0
			miu = 1 / eta
			Rref = (R0*rad2asec/Da_ref)/pixel

			a0 = np.max([0, cx_BCG - 0.8 * Rp])
			a1 = np.min([cx_BCG + 0.8 * Rp, 2048])
			b0 = np.max([0, cy_BCG - 0.8 * Rp])
			b1 = np.min([cy_BCG + 0.8 * Rp, 1489])
			'''
			plt.figure()
			plt.title('cluster ra%.3f dec%.3f z%.3f %s band' % (ra_g, dec_g, z_g, band[ii]) )
			ax = plt.imshow(img, cmap = 'Greys', vmin = 1e-10, origin = 'lower', norm = mpl.colors.LogNorm())
			plt.colorbar(ax, fraction = 0.045, pad = 0.01, label = '$flux[nmaggy]$')

			plt.xlim(cx_BCG - 0.8*Rp, cx_BCG + 0.8*Rp)
			plt.ylim(cy_BCG - 0.8*Rp, cy_BCG + 0.8*Rp)
			plt.subplots_adjust(bottom = 0.1, right = 0.8, top = 0.9)
			plt.savefig('cluster_ra%.3f_dec%.3f_z%.3f_%s_band.png' % (ra_g, dec_g, z_g, band[ii]), dpi = 300)
			plt.close()
			'''
			R_set = 10 # kpc
			Rp_set = 10**(-2) * rad2asec / Da_ref
			cc_img = flux_recal(img, z_g, z_ref)
			n_set = np.int(Rp_set / (miu * pixel))

			f_set = np.zeros(len(y_set), dtype = np.float)
			for qq in range(len(y_set)):
				f_set[qq] = np.sum(cc_img[y_set[qq] - n_set : y_set[qq] + n_set, 
					x_set[qq] - n_set : x_set[qq] + n_set]) / (2 * n_set * miu * pixel**2)

			plt.figure()
			plt.title('cluster ra%.3f dec%.3f z%.3f %s band[point select]' % (ra_g, dec_g, z_g, band[ii]) )
			ax = plt.imshow(img, cmap = 'Greys', vmin = 1e-10, origin = 'lower', norm = mpl.colors.LogNorm())
			plt.colorbar(ax, fraction = 0.035, pad = 0.01, label = '$flux[nmaggy]$')
			plt.scatter(x_set, y_set, s = 1, marker = 'o', facecolors = '', edgecolors = 'b', linewidth = 0.5)
			plt.xlim(0, img.shape[1])
			plt.ylim(0, img.shape[0])
			plt.subplots_adjust(bottom = 0.1, right = 0.8, top = 0.9)
			plt.savefig('cluster_ra%.3f_dec%.3f_z%.3f_%s_band_points.png' % (ra_g, dec_g, z_g, band[ii]), dpi = 300)
			plt.close()
			'''
			plt.figure()
			plt.title('rescale ra%.3f dec%.3f z%.3f %s band' % (ra_g, dec_g, z_g, band[ii]) )
			ax = plt.imshow(cc_img, cmap = 'Greys', vmin = 1e-10, origin = 'lower', norm = mpl.colors.LogNorm())
			plt.colorbar(ax, fraction = 0.045, pad = 0.01, label = '$flux[nmaggy]$')
			plt.xlim(cx_BCG - 0.8 * Rp, cx_BCG + 0.8 * Rp)
			plt.ylim(cy_BCG - 0.8 * Rp, cy_BCG + 0.8 * Rp)
			plt.subplots_adjust(bottom = 0.1, right = 0.8, top = 0.9)
			plt.savefig('rescales_ra%.3f_dec%.3f_z%.3f_%s_band.png' % (ra_g, dec_g, z_g, band[ii]), dpi = 300)
			plt.close()
			'''
			alpha = pixel / (miu * pixel)
			xn, yn, resam = gen(cc_img, 1, alpha, cx_BCG, cy_BCG)
			xn = np.int(xn)
			yn = np.int(yn)
			if alpha > 1:
				resam = resam[1:, 1:]
			elif alpha == 1:
				resam = resam[1:-1, 1:-1]
			else:
				resam = resam
			nx_set = [np.int( ll / alpha ) for ll in x_set]
			ny_set = [np.int( ll / alpha ) for ll in y_set]
			n_set = np.int(Rp_set / pixel)
			nf_set = np.zeros(len(nx_set), dtype = np.float)
			for qq in range(len(nx_set)):
				nf_set[qq] = np.sum(resam[ny_set[qq] - n_set : ny_set[qq] + n_set, 
					nx_set[qq] - n_set : nx_set[qq] + n_set]) / (2 * n_set * pixel**2)

			f_fract = (nf_set - f_set) / f_set

			plt.figure()
			plt.title('rescale+resample ra%.3f dec%.3f z%.3f %s band[point select]' % (ra_g, dec_g, z_g, band[ii]) )
			ax = plt.imshow(resam, cmap = 'Greys', vmin = 1e-10, origin = 'lower', norm = mpl.colors.LogNorm())
			plt.colorbar(ax, fraction = 0.035, pad = 0.01, label = '$flux[nmaggy]$')
			plt.scatter(nx_set, ny_set, s = 1, marker = 'o', facecolors = '', edgecolors = 'r', linewidth = 0.5)
			plt.xlim(0, resam.shape[1])
			plt.ylim(0, resam.shape[0])
			plt.subplots_adjust(bottom = 0.1, right = 0.8, top = 0.9)
			plt.savefig('rescale+resample_ra%.3f_dec%.3f_z%.3f_%s_band_points.png' % (ra_g, dec_g, z_g, band[ii]), dpi = 300)
			plt.close()
			'''
			plt.figure()
			plt.title('rescale + resample ra%.3f dec%.3f z%.3f %s band' % (ra_g, dec_g, z_g, band[ii]) )
			ax = plt.imshow(resam, cmap = 'Greys', vmin = 1e-10, origin = 'lower', norm = mpl.colors.LogNorm())
			plt.colorbar(ax, fraction = 0.045, pad = 0.01, label = '$flux[nmaggy]$')
			plt.xlim(cx_BCG / alpha - 0.8 * Rpp, cx_BCG / alpha + 0.8 * Rpp)
			plt.ylim(cy_BCG / alpha - 0.8 * Rpp, cy_BCG / alpha + 0.8 * Rpp)
			plt.subplots_adjust(bottom = 0.1, right = 0.8, top = 0.9)
			plt.savefig('resample_ra%.3f_dec%.3f_z%.3f_%s_band.png' % (ra_g, dec_g, z_g, band[ii]), dpi = 300)
			plt.close()
			'''
			plt.figure()
			ax = plt.subplot(111)
			ax.set_title('SB of pixel variation distribution [ra%.3f dec%.3f z%.3f %s band]' % (ra_g, dec_g, z_g, band[ii]) )
			ax.hist(f_fract, bins = 45, histtype = 'step', color = 'b')
			ax.axvline(x = 0, linestyle = '--', color = 'r', label = '$ x = 0 $', alpha = 0.5)
			ax.axvline(x = np.mean(f_fract), linestyle = '--', color = 'b', label = '$ mean \; value$', alpha = 0.5)
			ax.set_xlabel('SB variation fraction of pixel')
			ax.set_ylabel('pixel numbers')
			ax.set_yscale('log')

			ax.text(0, 1e2, s = '$ \sigma = %.3f $' % np.std(f_fract) + '\n' + 'Mean = %.3f' % np.mean(f_fract) 
				+ '\n' + 'Median = %.3f' % np.median(f_fract) )

			plt.legend(loc = 2)
			plt.savefig('SB variation distribution [ra%.3f dec%.3f z%.3f %s band].png' % (ra_g, dec_g, z_g, band[ii]), dpi = 300)
			plt.close()

		raise

	return

def mask_part():

	x0 = np.linspace(0, 2047, 2048)
	y0 = np.linspace(0, 1488, 1489)
	grd = np.array(np.meshgrid(x0, y0))
	r_star = 2*1.5/pixel

	bins = 65
	kd = -13
	#kd = 19

	#kd = -13 # test for richness
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
	SB, R, Anr, err = light_measure(img, bins, 1, Rp, cx_BCG, cy_BCG, pixel, zg)[:4]
	
	fig = plt.figure(figsize = (16, 8))
	gs = gridspec.GridSpec(1, 2, width_ratios = [1, 1])
	ax = plt.subplot(gs[0])
	bx = plt.subplot(gs[1])
	cluster = Circle(xy = (cx_BCG, cy_BCG), radius = Rp, fill = False, ec = 'r', alpha = 0.5)
	ax.set_title('origin img and SB profile', fontsize = 15)
	gf = ax.imshow(img, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
	fig.colorbar(gf, ax = ax, fraction = 0.035, pad = 0.01, label = '$flux[nmag]$')
	ax.add_patch(cluster)
	ax.scatter(cx_BCG, cy_BCG, marker = 'X', facecolors = '', edgecolors = 'r', alpha = 0.5)
	ax.set_xlim(0, 2048)
	ax.set_ylim(0, 1489)
	bx.plot(R, SB, label = '$origin \; img$')
	bx.set_xlabel('$R[kpc]$')
	bx.set_xscale('log')
	bx.set_ylabel('$SB[mag/arcsec^2]$')
	bx.invert_yaxis()
	bx.axis('scaled')
	bx.set_xlim(3e0, 1e3)
	bx.tick_params(axis = 'both', which = 'both', direction = 'in')
	bx.legend(loc = 3, fontsize = 12)

	bx1 = bx.twiny()
	xtik = bx.get_xticks(minor = True)
	xR = xtik * 10**(-3) * rad2asec / Da
	ix = xR > 2.3e1
	bx1.set_xticks(xtik[ix])
	bx1.set_xticklabels(["%.2f" % uu for uu in xR[ix]])
	bx1.set_xlim(bx.get_xlim())
	bx1.set_xlabel('$R[arcsec]$')
	bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.tight_layout()
	plt.savefig('/home/xkchen/mywork/ICL/code/proces_origin_img.png', dpi = 300)
	plt.close()

	## redden
	ra_img, dec_img = wcs.all_pix2world(grd[0,:], grd[1,:], 1)
	pos = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
	EBV = sfd(pos)
	Av = Rv * EBV * 0.86
	Al = A_wave(l_wave[2], Rv) * Av
	img1 = img*10**(Al / 2.5)
	SB1, R1, Anr1, err1 = light_measure(img1, bins, 1, Rp, cx_BCG, cy_BCG, pixel, zg)[:4]

	fig = plt.figure(figsize = (16, 8))
	gs = gridspec.GridSpec(1, 2, width_ratios = [1, 1])
	ax = plt.subplot(gs[0])
	bx = plt.subplot(gs[1])
	cluster = Circle(xy = (cx_BCG, cy_BCG), radius = Rp, fill = False, ec = 'r', alpha = 0.5)
	ax.set_title('redden calibration img and SB profile', fontsize = 15)
	gf = ax.imshow(img1, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
	fig.colorbar(gf, ax = ax, fraction = 0.035, pad = 0.01, label = '$flux[nmag]$')
	ax.add_patch(cluster)
	ax.scatter(cx_BCG, cy_BCG, marker = 'X', facecolors = '', edgecolors = 'r', alpha = 0.5)
	ax.set_xlim(0, 2048)
	ax.set_ylim(0, 1489)	
	bx.plot(R, SB, 'k--', label = '$origin \; img$', alpha = 0.5)
	bx.plot(R1, SB1, 'r-', label = '$redden \; calibration$', alpha = 0.5)
	bx.set_xlabel('$R[kpc]$')
	bx.set_xscale('log')
	bx.set_ylabel('$SB[mag/arcsec^2]$')
	bx.invert_yaxis()
	bx.axis('scaled')
	bx.set_xlim(3e0, 1e3)
	bx.tick_params(axis = 'both', which = 'both', direction = 'in')
	bx.legend(loc = 3, fontsize = 12)

	bx1 = bx.twiny()
	xtik = bx.get_xticks(minor = True)
	xR = xtik * 10**(-3) * rad2asec / Da
	ix = xR > 2.3e1
	bx1.set_xticks(xtik[ix])
	bx1.set_xticklabels(["%.2f" % uu for uu in xR[ix]])
	bx1.set_xlim(bx.get_xlim())
	bx1.set_xlabel('$R[arcsec]$')
	bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.tight_layout()
	plt.savefig('/home/xkchen/mywork/ICL/code/proces_redden_img.png', dpi = 300)
	plt.close()

	### compare part
	hdu = fits.PrimaryHDU()
	hdu.data = img1
	hdu.header = Head
	hdu.writeto(tmp_load + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % ('r', rag, decg, zg), overwrite = True)

	file_source = tmp_load + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % ('r', rag, decg, zg)
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

	CX = cx * 1
	CY = cy * 1
	a = Lr * 1
	b = Sr * 1
	theta = chi * 1

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

			iu = np.where(jx == True)
			iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float)
			iv[iu] = np.nan
			mask_A[lb0: lb1, la0: la1] = mask_A[lb0: lb1, la0: la1] * iv

	mirro_A = mask_A *img1
	SBt, Rt, Anrt, errt = light_measure(mirro_A, bins, 1, Rp, cx_BCG, cy_BCG, pixel, zg)[:4]

	fig = plt.figure(figsize = (16, 8))
	gs = gridspec.GridSpec(1, 2, width_ratios = [1, 1])
	ax = plt.subplot(gs[0])
	bx = plt.subplot(gs[1])
	ax.set_title('masked img and SB profile', fontsize = 15)
	cluster = Circle(xy = (cx_BCG, cy_BCG), radius = Rp, fill = False, ec = 'r', alpha = 0.5)
	gf = ax.imshow(mirro_A, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
	fig.colorbar(gf, ax = ax, fraction = 0.035, pad = 0.01, label = '$flux[nmag]$')
	ax.add_patch(cluster)
	ax.scatter(cx_BCG, cy_BCG, marker = 'X', facecolors = '', edgecolors = 'r', alpha = 0.5)
	ax.set_xlim(0, 2048)
	ax.set_ylim(0, 1489)
	bx.plot(Rt, SBt, 'b-', label = '$masked \; img$', alpha = 0.5)
	bx.plot(R1, SB1, 'r--', label = '$redden \; calibration$', alpha = 0.5)
	bx.set_xlabel('$R[kpc]$')
	bx.set_xscale('log')
	bx.set_ylabel('$SB[mag/arcsec^2]$')
	bx.invert_yaxis()
	bx.axis('scaled')
	bx.set_xlim(3e0, 1e3)
	bx.tick_params(axis = 'both', which = 'both', direction = 'in')
	bx.legend(loc = 3, fontsize = 12)

	bx1 = bx.twiny()
	xtik = bx.get_xticks(minor = True)
	xR = xtik * 10**(-3) * rad2asec / Da
	ix = xR > 2.3e1
	bx1.set_xticks(xtik[ix])
	bx1.set_xticklabels(["%.2f" % uu for uu in xR[ix]])
	bx1.set_xlim(bx.get_xlim())
	bx1.set_xlabel('$R[arcsec]$')
	bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.tight_layout()
	plt.savefig('/home/xkchen/mywork/ICL/code/proces_mask_img.png', dpi = 300)
	plt.close()

	## flux scale
	img_scale = flux_recal(mirro_A, zg, z_ref)
	eta = Da_ref / Da
	mu = 1 / eta
	SB_ref = SBt + 10*np.log10((1 + z_ref) / (1 + zg))
	Ar_ref = Anrt * mu
	id_nan = np.isnan(SB_ref)
	ivx = id_nan == False
	f_SB = interp(Ar_ref[ivx], SB_ref[ivx], kind = 'cubic')
	SB2, R2, Anr2, err2 = light_measure(img_scale, bins, 1, Rp, cx_BCG, cy_BCG, pixel * mu, z_ref)[:4]

	fig = plt.figure(figsize = (16, 8))
	gs = gridspec.GridSpec(1, 2, width_ratios = [1, 1])
	ax = plt.subplot(gs[0])
	bx = plt.subplot(gs[1])
	ax.set_title('scaled img and SB profile', fontsize = 15)
	cluster = Circle(xy = (cx_BCG, cy_BCG), radius = Rp, fill = False, ec = 'r', alpha = 0.5)
	gf = ax.imshow(img_scale, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
	fig.colorbar(gf, ax = ax, fraction = 0.035, pad = 0.01, label = '$flux[nmag]$')
	ax.add_patch(cluster)
	ax.scatter(cx_BCG, cy_BCG, marker = 'X', facecolors = '', edgecolors = 'r', alpha = 0.5)
	ax.set_xlim(0, 2048)
	ax.set_ylim(0, 1489)
	bx.plot(Rt, SBt, 'k--', label = '$masked \; img$', alpha = 0.5)
	bx.plot(Rt, SB_ref, 'r:', label = '$reference \; SB$', alpha = 0.5)
	bx.plot(R2, SB2, 'b-', label = '$scaled \; img$', alpha = 0.5)
	bx.set_xlabel('$R[kpc]$')
	bx.set_xscale('log')
	bx.set_ylabel('$SB[mag/arcsec^2]$')
	bx.invert_yaxis()
	bx.axis('scaled')
	bx.set_xlim(3e0, 1e3)
	bx.tick_params(axis = 'both', which = 'both', direction = 'in')
	bx.legend(loc = 3, fontsize = 12)

	bx1 = bx.twiny()
	xtik = bx.get_xticks(minor = True)
	xR = xtik * 10**(-3) * rad2asec / Da
	ix = xR > 2.3e1
	bx1.set_xticks(xtik[ix])
	bx1.set_xticklabels(["%.2f" % uu for uu in xR[ix]])
	bx1.set_xlim(bx.get_xlim())
	bx1.set_xlabel('$R[arcsec]$')
	bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.tight_layout()
	plt.savefig('/home/xkchen/mywork/ICL/code/proces_scale_img.png', dpi = 300)
	plt.close()

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
	SB3, R3, Anr3, err3 = light_measure(resam, bins, 1, Rpp, xn, yn, pixel, z_ref)[:4]

	fig = plt.figure(figsize = (16, 8))
	gs = gridspec.GridSpec(1, 2, width_ratios = [1, 1])
	ax = plt.subplot(gs[0])
	bx = plt.subplot(gs[1])
	ax.set_title('resampled img and SB profile', fontsize = 15)
	cluster = Circle(xy = (cx_BCG * mu, cy_BCG * mu), radius = Rpp, fill = False, ec = 'r', alpha = 0.5)
	gf = ax.imshow(resam, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
	fig.colorbar(gf, ax = ax, fraction = 0.035, pad = 0.01, label = '$flux[nmag]$')
	ax.add_patch(cluster)
	ax.scatter(cx_BCG * mu, cy_BCG * mu, marker = 'X', facecolors = '', edgecolors = 'r', alpha = 0.5)
	ax.set_xlim(0, resam.shape[1])
	ax.set_ylim(0, resam.shape[0])
	bx.plot(R3, SB3, 'g--', label = '$resampled \; img$', alpha = 0.5)
	bx.plot(Rt, SB_ref, 'r--', label = '$reference \; SB$', alpha = 0.5)
	bx.plot(R2, SB2, 'b:', label = '$scaled \; img$', alpha = 0.5)
	bx.set_xlabel('$R[kpc]$')
	bx.set_xscale('log')
	bx.set_ylabel('$SB[mag/arcsec^2]$')
	bx.invert_yaxis()
	bx.axis('scaled')
	bx.set_xlim(3e0, 1e3)
	bx.tick_params(axis = 'both', which = 'both', direction = 'in')
	bx.legend(loc = 3, fontsize = 12)

	bx1 = bx.twiny()
	xtik = bx.get_xticks(minor = True)
	xR = xtik * 10**(-3) * rad2asec / Da
	ix = xR > 2.3e1
	bx1.set_xticks(xtik[ix])
	bx1.set_xticklabels(["%.2f" % uu for uu in xR[ix]])
	bx1.set_xlim(bx.get_xlim())
	bx1.set_xlabel('$R[arcsec]$')
	bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.tight_layout()
	plt.savefig('/home/xkchen/mywork/ICL/code/proces_resamp_img.png', dpi = 300)
	plt.close()

	plt.figure(figsize = (16, 10))
	gs = gridspec.GridSpec(2, 1, height_ratios = [4, 1])
	ax = plt.subplot(gs[0])
	bx = plt.subplot(gs[1], sharex = ax)

	ax.plot(Anr3, SB3, 'g-', label = '$resampled \; img$')
	ax.plot(Ar_ref, SB_ref, 'r--', label = '$reference \; SB$')
	ax.set_ylabel('$SB[mag/arcsec^2]$')
	ax.invert_yaxis()
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax.legend(loc = 3)

	ddsb = SB3[(Anr3 >= np.min(Ar_ref[ivx])) & (Anr3 <= np.max(Ar_ref[ivx]))] - f_SB(Anr3[(Anr3 >= np.min(Ar_ref[ivx])) & (Anr3 <= np.max(Ar_ref[ivx]))])
	bx.plot(Anr3[(Anr3 >= np.min(Ar_ref[ivx])) & (Anr3 <= np.max(Ar_ref[ivx]))], ddsb, 'g*', alpha = 0.5)

	bx.axhline(y = np.nanmean(ddsb), c = 'g', linestyle = '--', alpha = 0.5)
	bx.set_ylabel('$SB_{resampled} - SB_{ref} [mag/arcsec^2]$')
	bx.tick_params(axis = 'both', which = 'both', direction = 'in')
	#bx.set_yscale('log')
	bx.set_xscale('log')
	bx.set_xlabel('$R[arcsec]$')

	plt.subplots_adjust(hspace = 0)
	plt.savefig('/home/xkchen/mywork/ICL/code/process_SB_compare.png', dpi = 300)
	plt.show()

	raise
	return

def mask_B():
	bins = 65
	x0 = np.linspace(0, 2047, 2048)
	y0 = np.linspace(0, 1488, 1489)
	img_grid = np.array(np.meshgrid(x0, y0))

	load = '/home/xkchen/mywork/ICL/data/total_data/sample_02_03/'
	mask = '/home/xkchen/mywork/ICL/data/star_catalog/'
	'''
	red_rich = Lambd[(Lambd >= 25) & (Lambd <= 27.5)]
	red_z = z[(Lambd >= 25) & (Lambd <= 27.5)]
	red_ra = ra[(Lambd >= 25) & (Lambd <= 27.5)]
	red_dec = dec[(Lambd >= 25) & (Lambd <= 27.5)]
	'''
	red_rich = Lambd[Lambd > 100]
	red_z = z[Lambd > 100]
	red_ra = ra[Lambd > 100]
	red_dec = dec[Lambd > 100]

	for pp in range(10):

		ra_g = red_ra[pp]
		dec_g = red_dec[pp]
		z_g = red_z[pp]
		for q in range(1):

			file = 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[q], ra_g, dec_g, z_g)

			data_f = fits.open(load+file)
			img = data_f[0].data
			head_inf = data_f[0].header
			wcs = awc.WCS(head_inf)
			x_side = data_f[0].data.shape[1]
			y_side = data_f[0].data.shape[0]
			cenx, ceny = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)
			Da = Test_model.angular_diameter_distance(z_g).value
			R_ph = rad2asec / Da
			R_p = R_ph / pixel
			'''
			SB, R, Anr, err = light_measure(img, bins, 1, R_p, cenx, ceny, pixel, z_g)[:4]
			fig = plt.figure(figsize = (16, 9))
			gs = gridspec.GridSpec(1, 2, width_ratios = [1, 1])
			ax = plt.subplot(gs[0])
			bx = plt.subplot(gs[1])

			cluster = Circle(xy = (cenx, ceny), radius = R_p, fill = False, ec = 'r', alpha = 0.5)
			ax.set_title('origin img and SB profile', fontsize = 15)
			gf = ax.imshow(img, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
			fig.colorbar(gf, ax = ax, fraction = 0.035, pad = 0.01, label = '$flux[nmag]$')
			ax.add_patch(cluster)
			ax.scatter(cenx, ceny, marker = 'X', facecolors = '', edgecolors = 'r', alpha = 0.5)
			ax.set_xlim(0, 2048)
			ax.set_ylim(0, 1489)
			bx.plot(R, SB, label = '$origin \; img$')
			bx.set_xlabel('$R[kpc]$')
			bx.set_xscale('log')
			bx.set_ylabel('$SB[mag/arcsec^2]$')
			bx.invert_yaxis()
			bx.axis('scaled')
			bx.set_xlim(3e0, 1e3)
			bx.tick_params(axis = 'both', which = 'both', direction = 'in')
			bx.legend(loc = 3, fontsize = 12)

			bx1 = bx.twiny()
			xtik = bx.get_xticks(minor = True)
			xR = xtik * 10**(-3) * rad2asec / Da
			ix = xR > 2.3e1
			bx1.set_xticks(xtik[ix])
			bx1.set_xticklabels(["%.2f" % uu for uu in xR[ix]])
			bx1.set_xlim(bx.get_xlim())
			bx1.set_xlabel('$R[arcsec]$')
			bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

			plt.tight_layout()
			plt.savefig('/home/xkchen/mywork/ICL/code/Bplane_origin_img.png', dpi = 300)		
			plt.close()
			'''
			t0 = time.time()
			ra_img, dec_img = wcs.all_pix2world(img_grid[0,:], img_grid[1,:], 1)
			pos = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
			BEV = sfd(pos, order = 1)

			Av = Rv * BEV * 0.86
			Al = A_wave(l_wave[q], Rv) * Av
			img1 = img * 10**(Al / 2.5)
			'''
			SB1, R1, Anr1, err1 = light_measure(img1, bins, 1, R_p, cenx, ceny, pixel, z_g)[:4]			
			fig = plt.figure(figsize = (16, 9))
			gs = gridspec.GridSpec(1, 2, width_ratios = [1, 1])
			ax = plt.subplot(gs[0])
			bx = plt.subplot(gs[1])

			cluster = Circle(xy = (cenx, ceny), radius = R_p, fill = False, ec = 'r', alpha = 0.5)
			ax.set_title('redden calibration and SB profile', fontsize = 15)
			gf = ax.imshow(img, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
			fig.colorbar(gf, ax = ax, fraction = 0.035, pad = 0.01, label = '$flux[nmag]$')
			ax.add_patch(cluster)
			ax.scatter(cenx, ceny, marker = 'X', facecolors = '', edgecolors = 'r', alpha = 0.5)
			ax.set_xlim(0, 2048)
			ax.set_ylim(0, 1489)
			bx.plot(R1, SB1, 'r-', label = '$redden \; calibration$', alpha = 0.5)
			bx.plot(R, SB, 'k--', label = '$origin \; img$', alpha = 0.5)
			bx.set_xlabel('$R[kpc]$')
			bx.set_xscale('log')
			bx.set_ylabel('$SB[mag/arcsec^2]$')
			bx.invert_yaxis()
			bx.axis('scaled')
			bx.set_xlim(3e0, 1e3)
			bx.tick_params(axis = 'both', which = 'both', direction = 'in')
			bx.legend(loc = 3, fontsize = 12)

			bx1 = bx.twiny()
			xtik = bx.get_xticks(minor = True)
			xR = xtik * 10**(-3) * rad2asec / Da
			ix = xR > 2.3e1
			bx1.set_xticks(xtik[ix])
			bx1.set_xticklabels(["%.2f" % uu for uu in xR[ix]])
			bx1.set_xlim(bx.get_xlim())
			bx1.set_xlabel('$R[arcsec]$')
			bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

			plt.tight_layout()
			plt.savefig('/home/xkchen/mywork/ICL/code/Bplane_redden_img.png', dpi = 300)		
			plt.close()
			'''
			cat = pd.read_csv('/home/xkchen/mywork/ICL/data/star_catalog/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g), skiprows = 1)
			ra_s = np.array(cat['ra'])
			dec_s = np.array(cat['dec'])
			mag = np.array(cat['r'])

			x, y = wcs.all_world2pix(ra_s*U.deg, dec_s*U.deg, 1)
			ia = (x >= 0) & (x <= x_side)
			ib = (y >= 0) & (y <= y_side)
			ie = (mag <= 20)
			ic = ia & ib & ie
			comx = x[ic]
			comy = y[ic]
			comr = 2 * 1.5 / pixel
			Numb = len(comx)

			mask_B = np.ones((img1.shape[0], img1.shape[1]), dtype = np.float)
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

				iu = np.where(jx == True)
				iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float)
				iv[iu] = np.nan
				mask_B[lb0: lb1, la0: la1] = mask_B[lb0: lb1, la0: la1] * iv

			mirro_B = mask_B * img1
			t1 = time.time() - t0
			'''
			SB2, R2, Anr2, err2 = light_measure(mirro_B, bins, 1, R_p, cenx, ceny, pixel, z_g)[:4]
			fig = plt.figure(figsize = (16, 9))
			gs = gridspec.GridSpec(1, 2, width_ratios = [1, 1])
			ax = plt.subplot(gs[0])
			bx = plt.subplot(gs[1])

			cluster = Circle(xy = (cenx, ceny), radius = R_p, fill = False, ec = 'r', alpha = 0.5)
			ax.set_title('masked img and SB profile', fontsize = 15)
			gf = ax.imshow(mirro_B, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
			fig.colorbar(gf, ax = ax, fraction = 0.035, pad = 0.01, label = '$flux[nmag]$')
			ax.add_patch(cluster)
			ax.scatter(cenx, ceny, marker = 'X', facecolors = '', edgecolors = 'r', alpha = 0.5)
			ax.set_xlim(0, 2048)
			ax.set_ylim(0, 1489)
			bx.plot(R1, SB1, 'r--', label = '$redden \; calibration$', alpha = 0.5)
			bx.plot(R2, SB2, 'g-', label = '$masked \; img$', alpha = 0.5)
			bx.set_xlabel('$R[kpc]$')
			bx.set_xscale('log')
			bx.set_ylabel('$SB[mag/arcsec^2]$')
			bx.invert_yaxis()
			bx.axis('scaled')
			bx.set_xlim(3e0, 1e3)
			bx.tick_params(axis = 'both', which = 'both', direction = 'in')
			bx.legend(loc = 3, fontsize = 12)

			bx1 = bx.twiny()
			xtik = bx.get_xticks(minor = True)
			xR = xtik * 10**(-3) * rad2asec / Da
			ix = xR > 2.3e1
			bx1.set_xticks(xtik[ix])
			bx1.set_xticklabels(["%.2f" % uu for uu in xR[ix]])
			bx1.set_xlim(bx.get_xlim())
			bx1.set_xlabel('$R[arcsec]$')
			bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

			plt.tight_layout()
			plt.savefig('/home/xkchen/mywork/ICL/code/Bplane_mask_img.png', dpi = 300)		
			plt.close()
			'''
			hdu = fits.PrimaryHDU()
			hdu.data = mirro_B
			hdu.header = head_inf
			hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/mask/B_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[q], ra_g, dec_g, z_g),overwrite = True)

	return

def resamp_B():
	load = '/home/xkchen/mywork/ICL/data/test_data/'
	'''
	red_rich = Lambd[(Lambd >= 25) & (Lambd <= 27.5)]
	red_z = z[(Lambd >= 25) & (Lambd <= 27.5)]
	red_ra = ra[(Lambd >= 25) & (Lambd <= 27.5)]
	red_dec = dec[(Lambd >= 25) & (Lambd <= 27.5)]
	'''
	red_rich = Lambd[Lambd > 100]
	red_z = z[Lambd > 100]
	red_ra = ra[Lambd > 100]
	red_dec = dec[Lambd > 100]
	bins = 65
	for ii in range(1):
		for jj in range(10):

			ra_g = red_ra[jj]
			dec_g = red_dec[jj]
			z_g = red_z[jj]
			Da_g = Test_model.angular_diameter_distance(z_g).value
			data = fits.getdata(load + 'mask/B_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[ii], ra_g, dec_g, z_g), header = True)
			img = data[0]
			wcs = awc.WCS(data[1])
			cx, cy = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

			cx0 = data[1]['CRPIX1']
			cy0 = data[1]['CRPIX2']
			RA0 = data[1]['CRVAL1']
			DEC0 = data[1]['CRVAL2']

			Angur = (R0 * rad2asec / Da_g)
			Rp = Angur / pixel
			L_ref = Da_ref * pixel / rad2asec
			L_z0 = Da_g * pixel / rad2asec
			b = L_ref / L_z0
			Rref = (R0 * rad2asec / Da_ref)/pixel
			mu = 1 / b

			f_goal = flux_recal(img, z_g, z_ref)

			SB1, R1, Anr1, err1 = light_measure(img, bins, 1, Rp, cx, cy, pixel, z_g)[:4]
			SB_ref = SB1 + 10*np.log10((1 + z_ref) / (1 + z_g))
			Ar_ref = Anr1 * mu

			id_nan = np.isnan(SB_ref)
			ivx = id_nan == False
			f_SB = interp(Ar_ref[ivx], SB_ref[ivx], kind = 'cubic')

			SB2, R2, Anr2, err2 = light_measure(f_goal, bins, 1, Rp, cx, cy, pixel * mu, z_ref)[:4]
			'''
			fig = plt.figure(figsize = (16, 9))
			gs = gridspec.GridSpec(1, 2, width_ratios = [1, 1])
			ax = plt.subplot(gs[0])
			bx = plt.subplot(gs[1])

			cluster = Circle(xy = (cx, cy), radius = Rp, fill = False, ec = 'r', alpha = 0.5)
			ax.set_title('scaled img and SB profile', fontsize = 15)
			gf = ax.imshow(f_goal, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
			fig.colorbar(gf, ax = ax, fraction = 0.035, pad = 0.01, label = '$flux[nmag]$')
			ax.add_patch(cluster)
			ax.scatter(cx, cy, marker = 'X', facecolors = '', edgecolors = 'r', alpha = 0.5)
			ax.set_xlim(0, 2048)
			ax.set_ylim(0, 1489)
			bx.plot(R1, SB1, 'g--', label = '$masked \; img$', alpha = 0.5)
			bx.plot(R2, SB2, 'b-', label = '$scaled \; img$', alpha = 0.5)
			bx.plot(R1, SB_ref, 'r:', label = '$reference \; SB$', alpha = 0.5)
			bx.set_xlabel('$R[kpc]$')
			bx.set_xscale('log')
			bx.set_ylabel('$SB[mag/arcsec^2]$')
			bx.invert_yaxis()
			bx.axis('scaled')
			bx.set_xlim(3e0, 1e3)
			bx.tick_params(axis = 'both', which = 'both', direction = 'in')
			bx.legend(loc = 3, fontsize = 12)

			bx1 = bx.twiny()
			xtik = bx.get_xticks(minor = True)
			xR = xtik * 10**(-3) * rad2asec / Da_g
			ix = xR > 2.3e1
			bx1.set_xticks(xtik[ix])
			bx1.set_xticklabels(["%.2f" % uu for uu in xR[ix]])
			bx1.set_xlim(bx.get_xlim())
			bx1.set_xlabel('$R[arcsec]$')
			bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

			plt.tight_layout()
			plt.savefig('/home/xkchen/mywork/ICL/code/Bplane_scale_img.png', dpi = 300)		
			plt.close()
			'''
			xn, yn, resam = gen(f_goal, 1, b, cx, cy)
			xn = np.int(xn)
			yn = np.int(yn)
			ix0 = np.int(cx0 * mu)
			iy0 = np.int(cy0 * mu)
			if b > 1:
				resam = resam[1:, 1:]
			elif b == 1:
				resam = resam[1:-1, 1:-1]
			else:
				resam = resam
			'''
			SB3, R3, Anr3, err3 = light_measure(resam, bins, 1, Rpp, xn, yn, pixel, z_ref)[:4]
			fig = plt.figure(figsize = (16, 9))
			gs = gridspec.GridSpec(1, 2, width_ratios = [1, 1])
			ax = plt.subplot(gs[0])
			bx = plt.subplot(gs[1])

			cluster = Circle(xy = (xn, yn), radius = Rpp, fill = False, ec = 'r', alpha = 0.5)
			ax.set_title('resampled img and SB profile', fontsize = 15)
			gf = ax.imshow(resam, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
			fig.colorbar(gf, ax = ax, fraction = 0.035, pad = 0.01, label = '$flux[nmag]$')
			ax.add_patch(cluster)
			ax.scatter(xn, yn, marker = 'X', facecolors = '', edgecolors = 'r', alpha = 0.5)
			ax.set_xlim(0, resam.shape[1])
			ax.set_ylim(0, resam.shape[0])
			bx.plot(R1, SB1, 'g--', label = '$masked \; img$', alpha = 0.5)
			bx.plot(R3, SB3, 'b-', label = '$resampled \; img$', alpha = 0.5)
			bx.plot(R1, SB_ref, 'r:', label = '$reference \; SB$', alpha = 0.5)
			bx.set_xlabel('$R[kpc]$')
			bx.set_xscale('log')
			bx.set_ylabel('$SB[mag/arcsec^2]$')
			bx.invert_yaxis()
			bx.axis('scaled')
			bx.set_xlim(3e0, 1e3)
			bx.tick_params(axis = 'both', which = 'both', direction = 'in')
			bx.legend(loc = 3, fontsize = 12)

			bx1 = bx.twiny()
			xtik = bx.get_xticks(minor = True)
			xR = xtik * 10**(-3) * rad2asec / Da_g
			ix = xR > 2.3e1
			bx1.set_xticks(xtik[ix])
			bx1.set_xticklabels(["%.2f" % uu for uu in xR[ix]])
			bx1.set_xlim(bx.get_xlim())
			bx1.set_xlabel('$R[arcsec]$')
			bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

			plt.tight_layout()
			plt.savefig('/home/xkchen/mywork/ICL/code/Bplane_resamp_img.png', dpi = 300)		
			plt.close()

			plt.figure(figsize = (16, 10))
			gs = gridspec.GridSpec(2, 1, height_ratios = [4, 1])
			ax = plt.subplot(gs[0])
			bx = plt.subplot(gs[1], sharex = ax)

			ax.plot(Anr3, SB3, 'g-', label = '$resampled \; img$')
			ax.plot(Ar_ref, SB_ref, 'r--', label = '$reference \; SB$')
			ax.set_ylabel('$SB[mag/arcsec^2]$')
			ax.invert_yaxis()
			ax.tick_params(axis = 'both', which = 'both', direction = 'in')
			ax.legend(loc = 3)

			ddsb = SB3[(Anr3 >= np.min(Ar_ref[ivx])) & (Anr3 <= np.max(Ar_ref[ivx]))] - f_SB(Anr3[(Anr3 >= np.min(Ar_ref[ivx])) & (Anr3 <= np.max(Ar_ref[ivx]))])
			bx.plot(Anr3[(Anr3 >= np.min(Ar_ref[ivx])) & (Anr3 <= np.max(Ar_ref[ivx]))], ddsb, 'g*', alpha = 0.5)

			bx.axhline(y = np.nanmean(ddsb), c = 'g', linestyle = '--', alpha = 0.5)
			bx.set_ylabel('$SB_{resampled} - SB_{ref} [mag/arcsec^2]$')
			bx.tick_params(axis = 'both', which = 'both', direction = 'in')
			#bx.set_yscale('log')
			bx.set_xscale('log')
			bx.set_xlabel('$R[arcsec]$')

			plt.subplots_adjust(hspace = 0)
			plt.savefig('/home/xkchen/mywork/ICL/code/Bplane_SB_compare.png', dpi = 300)
			plt.show()
			'''
			x0 = resam.shape[1]
			y0 = resam.shape[0]
			keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2','CENTER_X','CENTER_Y',
					'CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z', 'P_SCALE']
			value = ['T', 32, 2, x0, y0, ix0, iy0, xnd, ynd, RA0, DEC0, ra_g, dec_g, z_g, pixel]
			ff = dict(zip(keys,value))
			fil = fits.Header(ff)
			fits.writeto(load + 
				'resamp/resamp_B-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), resam, header = fil, overwrite=True)
	raise
	return

def stack_B():

	load = '/home/xkchen/mywork/ICL/data/test_data/resamp/'
	x0 = 2427
	y0 = 1765
	bins = 90
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	sum_grid = np.array(np.meshgrid(Nx, Ny))

	'''
	red_rich = Lambd[(Lambd >= 25) & (Lambd <= 27.5)]
	red_z = z[(Lambd >= 25) & (Lambd <= 27.5)]
	red_ra = ra[(Lambd >= 25) & (Lambd <= 27.5)]
	red_dec = dec[(Lambd >= 25) & (Lambd <= 27.5)]
	'''
	red_rich = Lambd[Lambd > 100]
	red_z = z[Lambd > 100]
	red_ra = ra[Lambd > 100]
	red_dec = dec[Lambd > 100]

	for ii in range(1):
		tot_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		tot_count = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
		p_count_total = np.zeros((len(Ny), len(Nx)), dtype = np.float)

		for jj in range(10):
			ra_g = red_ra[jj]
			dec_g = red_dec[jj]
			z_g = red_z[jj]

			Angur = (R0*rad2asec/Da_g)
			Rp = Angur/pixel
			L_ref = Da_ref*pixel/rad2asec
			L_z0 = Da_g*pixel/rad2asec
			b = L_ref/L_z0
			Rref = (R0*rad2asec/Da_ref)/pixel

			Da_g = Test_model.angular_diameter_distance(z_g).value
			data = fits.getdata(load + 'resamp_B-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
			img = data[0]
			xn = data_tt[1]['CENTER_X']
			yn = data_tt[1]['CENTER_Y']

			la0 = np.int(y0 - yn)
			la1 = np.int(y0 - yn + img.shape[0])
			lb0 = np.int(x0 - xn)
			lb1 = np.int(x0 - xn + img.shape[1])

			tot_array[la0:la1, lb0:lb1] = tot_array[la0:la1, lb0:lb1] + img
			tot_count[la0: la1, lb0: lb1] = img
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
	'''
	red_rich = Lambd[(Lambd >= 25) & (Lambd <= 27.5)]
	red_z = z[(Lambd >= 25) & (Lambd <= 27.5)]
	red_ra = ra[(Lambd >= 25) & (Lambd <= 27.5)]
	red_dec = dec[(Lambd >= 25) & (Lambd <= 27.5)]
	'''
	red_rich = Lambd[Lambd > 100]
	red_z = z[Lambd > 100]
	red_ra = ra[Lambd > 100]
	red_dec = dec[Lambd > 100]

	for i in range(1):
		for q in range(10):

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
					jx = idr <= 1

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
	return

def resamp_A():
	load = '/home/xkchen/mywork/ICL/data/test_data/resamp/'
	'''
	red_rich = Lambd[(Lambd >= 25) & (Lambd <= 27.5)]
	red_z = z[(Lambd >= 25) & (Lambd <= 27.5)]
	red_ra = ra[(Lambd >= 25) & (Lambd <= 27.5)]
	red_dec = dec[(Lambd >= 25) & (Lambd <= 27.5)]
	'''
	red_rich = Lambd[Lambd > 100]
	red_z = z[Lambd > 100]
	red_ra = ra[Lambd > 100]
	red_dec = dec[Lambd > 100]

	for ii in range(1):
		for jj in range(10):

			ra_g = red_ra[jj]
			dec_g = red_dec[jj]
			z_g = red_z[jj]
			
			Da_g = Test_model.angular_diameter_distance(z_g).value
			data = fits.getdata('/home/xkchen/mywork/ICL/data/test_data/mask/'
			+'A_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[ii], ra_g, dec_g, z_g), header = True)
			img = data[0]
			wcs = awc.WCS(data[1])
			cx_BCG, cy_BCG = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

			cx0 = data[1]['CRPIX1']
			cy0 = data[1]['CRPIX2']
			RA0 = data[1]['CRVAL1']
			DEC0 = data[1]['CRVAL2']

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
			'''
			cdr = np.sqrt((oo_grd[0,:] - cx_BCG)**2 + (oo_grd[1,:] - cy_BCG)**2)
			idd = (cdr > Rp) & (cdr < 1.1 * Rp)
			'''
			cdr = np.sqrt(((2 * oo_grd[0,:] + 1)/2 - (2 * cx_BCG + 1) / 2)**2 + ((2 * oo_grd[1,:] + 1)/2 - (2 * cy_BCG + 1)/2)**2)
			idd = (cdr > (2 * Rp + 1)/2) & (cdr < 1.1 * (2 * Rp + 1)/2)
			cut_region = img[idd]
			id_nan = np.isnan(cut_region)
			idx = np.where(id_nan == False)
			bl_array = cut_region[idx]
			bl_array = np.sort(bl_array)
			sky_compare = np.mean(bl_array)
			cc_img = img - sky_compare

			f_D = flux_recal(cc_img, z_g, z_ref)
			xnd, ynd, resam_dd = gen(f_D, 1, eta, cx_BCG, cy_BCG)
			xnd = np.int(xnd)
			ynd = np.int(ynd)
			ix0 = np.int(cx0 * miu)
			iy0 = np.int(cy0 * miu)
			if eta > 1:
				resam_d = resam_dd[1:, 1:]
			elif eta == 1:
				resam_d = resam_dd[1:-1, 1:-1]
			else:
				resam_d = resam_dd

			x0 = resam_d.shape[1]
			y0 = resam_d.shape[0]

			keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2','CENTER_X','CENTER_Y',
					'CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z', 'P_SCALE']
			value = ['T', 32, 2, x0, y0, ix0, iy0, xnd, ynd, RA0, DEC0, ra_g, dec_g, z_g, pixel]
			ff = dict(zip(keys,value))
			fil = fits.Header(ff)
			fits.writeto(load + 
				'resamp_A-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), resam_d, header = fil, overwrite=True)

	return

def stack_A():
	un_mask = 0.15
	r_star = 2*1.5/pixel
	load = '/home/xkchen/mywork/ICL/data/test_data/mask/'
	x0 = 2427
	y0 = 1765
	bins = 60
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	sum_grid = np.array(np.meshgrid(Nx, Ny))
	'''
	red_rich = Lambd[(Lambd >= 25) & (Lambd <= 27.5)]
	red_z = z[(Lambd >= 25) & (Lambd <= 27.5)]
	red_ra = ra[(Lambd >= 25) & (Lambd <= 27.5)]
	red_dec = dec[(Lambd >= 25) & (Lambd <= 27.5)]
	'''
	red_rich = Lambd[Lambd > 100]
	red_z = z[Lambd > 100]
	red_ra = ra[Lambd > 100]
	red_dec = dec[Lambd > 100]

	for ii in range(1):

		sum_array_D = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		count_array_D = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
		p_count_D = np.zeros((len(Ny), len(Nx)), dtype = np.float)

		for jj in range(10):

			ra_g = red_ra[jj]
			dec_g = red_dec[jj]
			z_g = red_z[jj]

			Da_g = Test_model.angular_diameter_distance(z_g).value
			data = fits.getdata(load + 'A_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[ii], ra_g, dec_g, z_g), header = True)
			img = data[0]
			wcs = awc.WCS(data[1])
			cx_BCG, cy_BCG = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

			Angur = (R0 * rad2asec / Da_g)
			Rp = Angur / pixel
			L_ref = Da_ref * pixel / rad2asec
			L_z0 = Da_g * pixel / rad2asec
			Rref = (R0 * rad2asec / Da_ref) / pixel

			eta = L_ref/L_z0
			miu = 1 / eta

			plt.figure()
			plt.title('original img ra%.3f dec%.3f z%.3f in %s band' % (ra_g, dec_g, z_g, band[ii]))
			ax = plt.imshow(img, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
			plt.colorbar(ax, fraction = 0.035, pad =  0.01, label = '$flux[nmaggy]$')
			plt.scatter(cx_BCG, cy_BCG, s = 10, marker = 'X', facecolors = '', edgecolors = 'b', linewidth = 0.5)
			hsc.circles(cx_BCG, cy_BCG, s = Rp, fc = '', ec = 'r', linestyle = '-')
			hsc.circles(cx_BCG, cy_BCG, s = 1.1*Rp, fc = '', ec = 'r', linestyle = '--')
			plt.xlim(0, img.shape[1])
			plt.ylim(0, img.shape[0])
			plt.savefig('/home/xkchen/mywork/ICL/code/sample_%d_img.png' % jj, dpi = 300)
			plt.close()

			ox = np.linspace(0, img.shape[1]-1, img.shape[1])
			oy = np.linspace(0, img.shape[0]-1, img.shape[0])
			oo_grd = np.array(np.meshgrid(ox, oy))
			cdr = np.sqrt(((2 * oo_grd[0,:] + 1)/2 - (2 * cx_BCG + 1) / 2)**2 + ((2 * oo_grd[1,:] + 1)/2 - (2 * cy_BCG + 1)/2)**2)
			idd = (cdr > (2 * Rp + 1)/2) & (cdr < 1.1 * (2 * Rp + 1)/2)
			ivx = np.where(idd == True)
			cut_region = img[idd]
			id_nan = np.isnan(cut_region)
			idx = np.where(id_nan == False)
			bl_array = cut_region[idx]
			sky_compare = np.mean(bl_array)
			cc_img = img - 0.

			############# stack the image with background subtraction
			f_D = flux_recal(img, z_g, z_ref)
			xnd, ynd, resam_dd = gen(f_D, 1, eta, cx_BCG, cy_BCG)
			xnd = np.int(xnd)
			ynd = np.int(ynd)
			if eta > 1:
				resam_d = resam_dd[1:, 1:]
			elif eta == 1:
				resam_d = resam_dd[1:-1, 1:-1]
			else:
				resam_d = resam_dd

			ox = np.linspace(0, resam_d.shape[1] - 1, resam_d.shape[1])
			oy = np.linspace(0, resam_d.shape[0] - 1, resam_d.shape[0])
			oo_grd = np.array(np.meshgrid(ox, oy))
			cdr = np.sqrt(((2 * oo_grd[0,:] + 1)/2 - (2 * xnd + 1) / 2)**2 + ((2 * oo_grd[1,:] + 1)/2 - (2 * ynd + 1)/2)**2)
			idd = (cdr > (2 * Rpp + 1)/2) & (cdr < 1.1 * (2 * Rpp + 1)/2)
			ivx = np.where(idd == True)
			cut_region = resam_d[idd]
			id_nan = np.isnan(cut_region)
			idx = np.where(id_nan == False)
			bl_array = cut_region[idx]
			back_lel = np.mean(bl_array)

			resamt = resam_d - 0.

			la0 = np.int(y0 - ynd)
			la1 = np.int(y0 - ynd + resamt.shape[0])
			lb0 = np.int(x0 - xnd)
			lb1 = np.int(x0 - xnd + resamt.shape[1])

			idx = np.isnan(resamt)
			idv = np.where(idx == False)
			sum_array_D[la0:la1, lb0:lb1][idv] = sum_array_D[la0:la1, lb0:lb1][idv] + resamt[idv]
			count_array_D[la0: la1, lb0: lb1][idv] = resamt[idv]
			id_nan = np.isnan(count_array_D)
			id_fals = np.where(id_nan == False)
			p_count_D[id_fals] = p_count_D[id_fals] + 1
			count_array_D[la0: la1, lb0: lb1][idv] = np.nan

			sub_mean = sum_array_D / p_count_D
			id_inf = np.isinf(sub_mean)
			sub_mean[id_inf] = np.nan
			id_zero = np.where(p_count_D == 0)
			sub_mean[id_zero] = np.nan

			'''
			plt.figure()
			ax = plt.imshow(sum_array_D, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
			plt.colorbar(ax, fraction = 0.035, pad = 0.01, label = '$flux[nmaggy]$')
			plt.title('Now add %d sample' % np.int(jj + 1))
			plt.plot(x0, y0, 'rP')
			hsc.circles(x0, y0, s = Rpp, fc = '', ec = 'b')
			plt.savefig('/home/xkchen/mywork/ICL/code/sub_show_%d.png' % jj, dpi = 300)
			plt.close()

			plt.figure()
			ax = plt.imshow(p_count_D, origin = 'lower', vmin = 1e-3)
			plt.colorbar(ax, fraction = 0.035, pad = 0.01, label = '$flux[nmaggy]$')
			plt.plot(x0, y0, 'rP')
			hsc.circles(x0, y0, s = Rpp, fc = '', ec = 'b')
			plt.title('Now add %d sample' % np.int(jj + 1))
			plt.savefig('/home/xkchen/mywork/ICL/code/sub_count_%d.png' % jj, dpi = 300)
			plt.close()
			'''
			plt.figure()
			ax = plt.imshow(sub_mean, cmap = 'Greys', origin = 'lower', vmin = 1e-3, norm = mpl.colors.LogNorm())
			plt.colorbar(ax, fraction = 0.035, pad = 0.01, label = '$flux[nmaggy]$')
			plt.plot(x0, y0, 'rP')
			hsc.circles(x0, y0, s = Rpp, fc = '', ec = 'b')
			plt.title('mean fllux of %d sample' % np.int(jj + 1))
			plt.savefig('/home/xkchen/mywork/ICL/code/sub_mean_%d.png' % jj, dpi = 300)
			plt.close()

		mean_array_D = sum_array_D / p_count_D
		where_are_inf = np.isinf(mean_array_D)
		mean_array_D[where_are_inf] = np.nan
		id_zeros = np.where(p_count_D == 0)
		mean_array_D[id_zeros] = np.nan

		SB, R, Ar, err, mm_flux = light_measure(mean_array_D, bins, 1, Rpp, x0, y0, pixel, z_ref)[:5]
		SB1 = SB + mag_add[ii]
		Ar1 = (Ar / Angu_ref) * Angu_ref
		R1 = R * 1
		err1 = err * 1
		stackA = np.array([SB1, R1, Ar1, err1])

		'''
		with h5py.File(
			'/home/xkchen/mywork/ICL/data/test_data/SB_stack_Amask_%s_band.h5' % band[ii], 'w') as f:
			f['a'] = stackA
		with h5py.File('/home/xkchen/mywork/ICL/data/test_data/SB_stack_Amask_%s_band.h5' % band[ii]) as f:
			for tt in range(len(stackA)):
				f['a'][tt,:] = stackA[tt,:]

		with h5py.File(
			'/home/xkchen/mywork/ICL/data/test_data/SB_stack_Amask_%s_band_low.h5' % band[ii], 'w') as f:
			f['a'] = stackA
		with h5py.File('/home/xkchen/mywork/ICL/data/test_data/SB_stack_Amask_%s_band_low.h5' % band[ii]) as f:
			for tt in range(len(stackA)):
				f['a'][tt,:] = stackA[tt,:]
		'''
		plt.figure(figsize = (16, 8))
		ax = plt.subplot(111)

		ax.plot(Ar1, SB1, 'g-', alpha = 0.5)

		ax.set_xscale('log')
		ax.set_xlabel('$R[arcsec]$')
		ax.set_ylabel('$SB[mag/arcsec^2]$')
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')
		ax.invert_yaxis()
		ax.set_xlim(1, 2.5e2)

		plt.savefig('/home/xkchen/mywork/ICL/code/stack_profile_%s_band.png' % band[ii], dpi = 300)
		plt.show()

		raise
	return

def SB_fit(r, m0, mc, c, m2l):
	bl = m0
	surf_mass = sigmamc(r, mc, c)
	surf_lit = surf_mass / m2l

	Lz = surf_lit / ((1 + z_ref)**4 * np.pi * 4 * rad2asec**2)
	Lob = Lz * Lsun / kpc2cm**2
	fob = Lob / (10**(-9)*f0)
	mock_SB = 22.5 - 2.5 * np.log10(fob)

	mock_L = mock_SB + bl

	return mock_L

def chi2(X, *args):
	m0 = X[0]
	mc = X[1]
	c = X[2]
	m2l = X[3]
	r, data, yerr = args
	m0 = m0
	mc = mc
	m2l = m2l
	c = c
	mock_L = SB_fit(r, m0, mc, c, m2l)
	chi = np.sum(((mock_L - data) / yerr)**2)
	return chi

def crit_r(Mc, c):
	c = c
	M = 10**Mc
	rho_c = (kpc2m/Msun2kg)*(3*H0**2)/(8*np.pi*G)
	r200_c = (3*M/(4*np.pi*rho_c*200))**(1/3) 
	rs = r200_c / c
	return rs, r200_c

def SB_ICL():

	for ii in range(len(band)):

		with h5py.File('/home/xkchen/mywork/ICL/data/test_data/SB_stack_Amask_%s_band.h5' % band[ii]) as f:
			A_stack = np.array(f['a'])
		SB_diff = A_stack[0,:]
		R_diff = A_stack[1,:]
		Ar_diff = A_stack[2,:]
		err_diff = A_stack[3,:]
		'''
		with h5py.File('/home/xkchen/mywork/ICL/data/test_data/SB_stack_Amask_%s_band_low.h5' % band[id_color]) as f:
			A_stack = np.array(f['a'])
		SB_diff = A_stack[0,:]
		R_diff = A_stack[1,:]
		Ar_diff = A_stack[2,:]
		err_diff = A_stack[3,:]
		'''
		ix = R_diff >= 100
		iy = R_diff <= 900
		iz = ix & iy
		r_fit = R_diff[iz]
		sb_fit = SB_diff[iz]
		err_fit = err_diff[iz]

		m0 = np.arange(30.5, 35.5, 0.25)
		mc = np.arange(13.5, 15, 0.25)
		cc = np.arange(1, 5, 0.25)
		m2l = np.arange(200, 274, 2)
		'''
		popt, pcov = curve_fit(SB_fit, r_fit, sb_fit, p0 = po, bounds = ([30, 13.5, 1, 200], [37, 15, 6, 270]), method = 'trf')
		M0 = popt[0]
		Mc = popt[1]
		Cc = popt[2]
		M2L = popt[3]
		'''
		popt = minimize(chi2, x0 = np.array([m0[0], mc[0], cc[0], m2l[10]]), args = (r_fit, sb_fit, err_fit), method = 'Powell', tol = 1e-5)
		M0 = popt.x[0]
		Mc = popt.x[1]
		Cc = popt.x[2]
		M2L = popt.x[3]

		print('*'*10)
		print('m0 = ', M0)
		print('Mc = ', Mc)
		print('C = ', Cc)
		print('M2L = ', M2L)

		fit_line = SB_fit(r_fit, M0, Mc, Cc, M2L)
		rs, r200 = crit_r(Mc, Cc)

		fig = plt.figure(figsize = (16, 9))
		plt.suptitle('stack profile in %s band' % band[ii])
		ax = plt.subplot(111)
		ax.errorbar(R_diff, SB_diff, yerr = err_diff, xerr = None, ls = '', fmt = 'ro')
		ax.set_xlabel('$R[kpc]$')
		ax.set_xscale('log')
		ax.set_ylabel('$SB[mag/arcsec^2]$')
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')
		ax.invert_yaxis()
		ax.set_xlim(np.min(R_diff + 1), np.max(R_diff + 20))
		ax1 = ax.twiny()
		xtik = ax.get_xticks(minor = True)
		xR = xtik * 10**(-3) * rad2asec / Da_ref
		id_tt = xtik >= 9e1
		ax1.set_xticks(xtik[id_tt])
		ax1.set_xticklabels(["%.2f" % uu for uu in xR[id_tt]])
		ax1.set_xlim(ax.get_xlim())
		ax1.set_xlabel('$R[arcsec]$')
		ax1.tick_params(axis = 'both', which = 'both', direction = 'in')

		plt.savefig('/home/xkchen/mywork/ICL/code/stack_profile_%sband.png' % band[ii], dpi = 300)
		plt.show()

		fig = plt.figure(figsize = (16, 9))
		plt.suptitle('$fit \; for \; background \; estimate \; in \; %s \; band$' % band[ii])
		bx = plt.subplot(111)
		cx = fig.add_axes([0.15, 0.25, 0.175, 0.175])

		bx.errorbar(R_diff[iz], SB_diff[iz], yerr = err_diff[iz], xerr = None, ls = '', fmt = 'ro', label = '$observation$')
		bx.plot(r_fit, fit_line, 'b-', label = '$NFW+C$')
		bx.axvline(x = rs, linestyle = '--', linewidth = 1, color = 'b', label = '$r_s$')

		bx.set_xlabel('$R[kpc]$')
		bx.set_xscale('log')
		bx.set_ylabel('$SB[mag/arcsec^2]$')
		bx.tick_params(axis = 'both', which = 'both', direction = 'in')
		bx.invert_yaxis()
		bx.set_xlim(1e2, 9e2)
		bx1 = bx.twiny()
		xtik = bx.get_xticks(minor = True)
		xR = xtik * 10**(-3) * rad2asec / Da_ref
		bx1.set_xticks(xtik)
		bx1.set_xticklabels(["%.2f" % uu for uu in xR])
		bx1.set_xlim(bx.get_xlim())
		bx1.set_xlabel('$R[arcsec]$')
		bx1.tick_params(axis = 'both', which = 'both', direction = 'in')
		
		cx.text(0, 0, s = 'BL = %.2f' % M0 + '\n' + '$Mc = %.2fM_\odot $' % Mc + '\n' + 
			'C = %.2f' % Cc + '\n' + 'M2L = %.2f' % M2L, fontsize = 15)
		cx.axis('off')
		cx.set_xticks([])
		cx.set_yticks([])

		bx.legend(loc = 3, fontsize = 15)
		plt.savefig('/home/xkchen/mywork/ICL/code/fit_for_BG_%s_band.png' % band[ii], dpi = 600)
		plt.show()
		
		raise
	return

def main():
	#resam_test()
	stack_no_mask()
	#mask_part()

	#mask_B()
	#resamp_B()
	#stack_B() ## not include background subtraction

	#mask_A()
	#resamp_A()
	stack_A()

	SB_ICL()

if __name__ == "__main__":
	main()
