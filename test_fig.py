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
R0 = 1 # in unit Mpc
Angu_ref = (R0/Da_ref)*rad2asec
Rpp = Angu_ref/pixel
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

def mask_A(eta):
	eta = eta
	kb = 2
	stack_N = np.int(5)
	x0 = np.linspace(0, 2047, 2048)
	y0 = np.linspace(0, 1488, 1489)
	img_grid = np.array(np.meshgrid(x0, y0))

	for tt in range(stack_N):

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

		ra_g = ra[tt]
		dec_g = dec[tt]
		z_g = z[tt]
		file = 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[kb], ra_g, dec_g, z_g)

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
			hdu = fits.PrimaryHDU()
			hdu.data = img
			hdu.header = head_inf
			hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/' + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits'% (band[kb], ra_g, dec_g, z_g), overwrite = True)

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
		chi = np.array(source['THETA_IMAGE'])
		cx = np.array(source['X_IMAGE']) - 1
		cy = np.array(source['Y_IMAGE']) - 1
		p_type = np.array(source['CLASS_STAR'])
		#Kron = source['KRON_RADIUS']
		Kron = 5
		Lr = Kron*A
		Sr = Kron*B

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

		CX = np.r_[cx, comx]
		CY = np.r_[cy, comy]
		a = np.r_[Lr, 2*comr]
		b = np.r_[Sr, 2*comr]
		theta = np.r_[chi, com_chi]
		Numb = Numb + len(comx)
		mask_A = np.ones((img.shape[0], img.shape[1]), dtype = np.float)
		ox = np.linspace(0,2047,2048)
		oy = np.linspace(0,1488,1489)
		basic_coord = np.array(np.meshgrid(ox,oy))
		major = a/2
		minor = b/2
		senior = np.sqrt(major**2 - minor**2)

		id_cover = []
		for q in range(len(comx)):
			dd = np.sqrt((cx - comx[q])**2 + (cy - comy[q])**2)
			idd = np.where(dd == np.min(dd))[0]
			id_cover.append(idd[0])
		id_cover = np.array(id_cover)
		CCx = cx[id_cover]
		CCy = cy[id_cover]
		Phi = chi[id_cover]
		LLr = Lr[id_cover]
		SSr = Sr[id_cover]

		tdr = np.sqrt((CX - cx_BCG)**2 + (CY - cy_BCG)**2)
		dr00 = np.where(tdr == np.min(tdr))[0]

		for k in range(Numb):
			xc = CX[k]
			yc = CY[k]

			t_cen = np.isin(CCx, xc)
			id_t = np.sum(t_cen)
			if id_t == 1:
				lr = major[k] * eta
				sr = minor[k] * eta
				cr = senior[k] * eta
			else:
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

		hdu = fits.PrimaryHDU()
		hdu.data = mirro_A
		hdu.header = head_inf
		hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/mask/star_test/mask_test_%s_ra%.3f_dec%.3f_z%.3f.fits' % (band[kb], ra_g, dec_g, z_g), overwrite = True)
		if tt == 4:
			plt.figure()
			plt.title('$star \; mask \; test \; with \; %.2f r_{iso} \; $' % eta)
			plt.imshow(mirro_A, cmap = 'Greys', origin = 'lower', vmin = 1e-3, norm = mpl.colors.LogNorm())
			hsc.circles(cx_BCG, cy_BCG, s = R_p, fc = '', ec = 'b', )
			hsc.circles(cx_BCG, cy_BCG, s = 1.1*R_p, fc = '', ec = 'b', ls = '--')

			hsc.circles(cx_BCG, cy_BCG, s = 20, fc = '', ec = 'r', lw = 1, alpha = 0.5)
			hsc.circles(comx, comy, s = comr, fc = '', ec = 'b', ls = '-', lw = 0.5, alpha = 0.5)
			hsc.ellipses(CX, CY, w = a, h = b, rot = theta, fc = '', ec = 'r', ls = '--', lw = 0.5, alpha = 0.5)

			hsc.ellipses(CCx, CCy, w = LLr * eta, h = SSr * eta, rot = Phi, fc ='', ec = 'g', ls = '-', alpha = 0.5)

			plt.xlim(0, 2048)
			plt.ylim(0, 1489)
			plt.savefig('/home/xkchen/mywork/ICL/code/mask_star_%.2fRiso_%s_ra%.3f_dec%.3f_z%.3f.png' % (eta, band[kb], ra_g, dec_g, z_g), dpi = 600)
			plt.close()

	return

def stack_A(eta):
	kb = 2
	stack_N = np.int(5)
	eta = eta

	x0 = 2427
	y0 = 1765
	bins = 90
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	sum_grid = np.array(np.meshgrid(Nx, Ny))

	get_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	p_count_1 = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	load = '/home/xkchen/mywork/ICL/data/test_data/mask/star_test/'
	for jj in range(stack_N):
		ra_g = ra[jj]
		dec_g = dec[jj]
		z_g = z[jj]
		Da_g = Test_model.angular_diameter_distance(z_g).value
		data = fits.getdata(load + 'mask_test_%s_ra%.3f_dec%.3f_z%.3f.fits' % (band[kb], ra_g, dec_g, z_g), header = True)
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

		get_array[la0:la1, lb0:lb1] = get_array[la0:la1, lb0:lb1] + resam
		count_array[la0: la1, lb0: lb1] = resam
		ia = np.where(count_array != 0)
		p_count_1[ia[0], ia[1]] = p_count_1[ia[0], ia[1]] + 1
		count_array[la0: la1, lb0: lb1] = 0

	mean_array = get_array/p_count_1
	where_are_nan = np.isnan(mean_array)
	mean_array[where_are_nan] = 0

	SB, R, Ar, error = light_measure(mean_array, bins, 1, Rpp, x0, y0, pixel, z_ref)

	SB_diff = SB[1:] + mag_add[kb]
	R_diff = R[1:]
	Ar_diff = Ar[1:]
	err_diff = error[1:]
	# background level
	dr = np.sqrt((sum_grid[0,:] - x0)**2 + (sum_grid[1,:] - y0)**2)
	ia = dr >= Rpp
	ib = dr <= 1.1*Rpp
	ic = ia & ib
	sky_set = mean_array[ic]
	sky_light = np.sum(sky_set[sky_set != 0])/len(sky_set[sky_set != 0])
	sky_mag = 22.5 - 2.5*np.log10(sky_light) + 2.5*np.log10(pixel**2) + mag_add[kb]

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('$SB \; with \; \eta \, = \, %.2f$' % eta)
	ax.errorbar(Ar_diff, SB_diff, yerr = err_diff, xerr = None, ls = '', fmt = 'go', label = '$Stack_{%.0f}$' % stack_N, alpha = 0.5)
	ax.legend(loc = 3)
	ax.set_xscale('log')
	ax.set_xlabel('$R[arcsec]$')
	ax.set_ylabel('$SB[mag/arcsec^2]$')
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax1 = ax.twiny()
	ax1.plot(R_diff, SB_diff, 'b-', alpha = 0.5)
	ax1.set_xscale('log')
	ax1.set_xlabel('$R[kpc]$')
	ax1.tick_params(axis = 'x', which = 'both', direction = 'in')
	ax.invert_yaxis()
	plt.savefig('/home/xkchen/mywork/ICL/code/light_test_%.2fRiso_%s.png' % (eta, band[kb]), dpi = 600)
	plt.close()
	
	return SB_diff, R_diff, Ar_diff, err_diff

def main():
	import matplotlib.gridspec as gridspec
	kb = 2
	eta = np.linspace(5, 10, 6)/5
	R_star = eta * 5
	'''
	for pp in range(len(R_star)):
		mask_A(R_star[pp])
	'''
	fig = plt.figure(figsize = (16,9))
	gs = gridspec.GridSpec(2,1, height_ratios = [4,1])
	bx = plt.subplot(gs[0])
	cx = plt.subplot(gs[1])
	for k in range(len(R_star)):
		SB, R, Ar, err = stack_A(eta[k])
		if k == 0:
			SB0 = SB * 1

		bx.plot(Ar, SB, ls = '-', color = mpl.cm.rainbow(k/len(R_star)), label = '$ \eta = %.2f Riso $' % R_star[k], alpha = 0.5)
		bx.set_xscale('log')
		bx.set_xlabel('$R[arcsec]$')
		bx.set_ylabel('$SB[mag/arcsec^2]$')
		bx.tick_params(axis = 'both', which = 'both', direction = 'in')
		handles, labels = plt.gca().get_legend_handles_labels()

		bx1 = bx.twiny()
		bx1.plot(R, SB, ls = '-', color = mpl.cm.rainbow(k/len(R_star)))
		bx1.set_xscale('log')
		bx1.set_xlabel('$R[kpc]$')
		bx1.tick_params(axis = 'x', which = 'both', direction = 'in')

		cx.plot(Ar, SB - SB0, ls = '-', color = mpl.cm.rainbow(k/len(R_star)), label = '$SB_{%.2fRiso} - SB_{%.2fRiso}$' % (R_star[k], R_star[0]))
		cx.set_xlabel('$R[arcsec]$')
		cx.set_xscale('log')
		cx.set_ylabel('$\Delta{SB}[mag/arcsec^{2}]$')
		cx.tick_params(axis = 'x', which = 'both', direction = 'in')

	bx.invert_yaxis()
	bx.legend( loc = 1)
	plt.savefig('/home/xkchen/mywork/ICL/code/light_test_%s_band.png' % band[kb], dpi = 600)
	plt.close()

if __name__ == "__main__":
	main()
