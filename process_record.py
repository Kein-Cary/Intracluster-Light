import matplotlib as mpl
import handy.scatter as hsc
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pd
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

import subprocess as subpro
import astropy.units as U
import astropy.constants as C

from astropy import cosmology as apcy
from dustmaps.sfd import SFDQuery
from scipy.optimize import curve_fit
from astropy.coordinates import SkyCoord

from resamp import gen
from extinction_redden import A_wave
from light_measure import light_measure, flux_recal
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

R0 = 1 # in unit Mpc
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

stack_N = 5
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

			plt.figure()
			plt.imshow(mirro_B, cmap = 'Greys', origin = 'lower', vmin = 1e-3, norm = mpl.colors.LogNorm())
			hsc.circles(comx, comy, s = comr, fc = '', ec = 'r', lw = 1)
			hsc.circles(cenx, ceny, s = R_p, fc = '', ec = 'b', )
			hsc.circles(cenx, ceny, s = 1.1*R_p, fc = '', ec = 'b', ls = '--')
			plt.ylim(0, 1489)
			plt.xlim(0, 2048)
			plt.savefig('/home/xkchen/mywork/ICL/code/sdss_mask_B_band%s.png'%band[q], dpi = 600)
			plt.close()	
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
	#param_A = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A_Tal.sex' # Tal 2011
	#param_A = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A_Ze.sex' # Zibetti 2005
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
				cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s'%(param_A, out_load_A, out_cat)
			else:
				hdu = fits.PrimaryHDU()
				hdu.data = data_f[0].data
				hdu.header = head_inf
				hdu.writeto('source_data.fits', overwrite = True)
				file_source = './source_data.fits'	
				dete_thresh = sb_lim[i] + 10*np.log10((1 + z_g)/(1 + z_ref))
				dete_thresh = '%.3f' % dete_thresh + ',%.2f' % zopt[i]
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

			plt.figure()
			plt.imshow(mirro_A, cmap = 'Greys', origin = 'lower', vmin = 1e-3, norm = mpl.colors.LogNorm())
			hsc.ellipses(cx, cy, w = a, h = b, rot = theta, fc = '', ec = 'r', ls = '--', lw = 0.5)
			hsc.circles(comx, comy, s = comr, fc = '', ec = 'b', lw = 1)
			hsc.circles(cx_BCG, cy_BCG, s = R_p, fc = '', ec = 'b', )
			hsc.circles(cx_BCG, cy_BCG, s = 1.1*R_p, fc = '', ec = 'b', ls = '--')
			plt.ylim(0, 1489)
			plt.xlim(0, 2048)
			plt.savefig('/home/xkchen/mywork/ICL/code/sdss_mask_A_band%s.png' % band[i], dpi = 600)
			plt.close()

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
		get_array = np.zeros((len(Ny), len(Nx)), dtype = np.float) 
		count_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		p_count_1 = np.zeros((len(Ny), len(Nx)), dtype = np.float)

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
		SB_measure = SB[1:] + mag_add[ii]
		R_measure = R[1:]
		Ar_measure = Ar[1:]
		SB_error = error[1:]

		stack_A = np.array([SB_measure, R_measure, Ar_measure, SB_error])
		with h5py.File('/home/xkchen/mywork/ICL/data/test_data/SB_stack_Amask_%s_band.h5' % band[ii], 'w') as f:
			f['a'] = np.array(stack_A)
		with h5py.File('/home/xkchen/mywork/ICL/data/test_data/SB_stack_Amask_%s_band.h5' % band[ii]) as f:
			for tt in range(len(stack_A)):
				f['a'][tt,:] = stack_A[tt,:]

		dr = np.sqrt((sum_grid[0,:] - x0)**2 + (sum_grid[1,:] - y0)**2)
		ia = dr >= Rpp
		ib = dr <= 1.1*Rpp
		ic = ia & ib
		sky_set = mean_array[ic]
		sky_light = np.sum(sky_set[sky_set != 0])/len(sky_set[sky_set != 0])
		sky_mag = 22.5 - 2.5*np.log10(sky_light) + 2.5*np.log10(pixel**2) + mag_add[ii]
		sky_lev[ii] = sky_mag

		plt.figure()
		gf = plt.imshow(mean_array, cmap = 'Greys', origin = 'lower', vmin = 1e-3, norm = mpl.colors.LogNorm())
		plt.colorbar(gf, fraction = 0.036, pad = 0.01, label = '$f[nmagy]$')
		hsc.circles(x0, y0, s = Rpp, fc = '', ec = 'b', ls = '-', lw = 0.5)
		hsc.circles(x0, y0, s = 1.1*Rpp,  fc = '', ec = 'b', ls = '--', lw = 0.5)
		plt.xlim(x0 - 1.2*Rpp, x0 + 1.2*Rpp)
		plt.ylim(y0 - 1.2*Rpp, y0 + 1.2*Rpp)
		plt.subplots_adjust(left = 0.01, right = 0.85)
		plt.title('stack %.0f mean image in %s band' % (stack_N, band[ii]))
		plt.savefig('/home/xkchen/mywork/ICL/code/stack_mask_A_%s_band.png' % band[ii], dpi = 600)
		plt.close()

	with h5py.File('/home/xkchen/mywork/ICL/data/test_data/sky_light.h5', 'w') as f:
		f['a'] = np.array(sky_lev)

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

	return

def SB_fit(r, m0, Mc, c, M2L):
	skyl = m0
	surf_mass = sigmamc(r, Mc, c)
	surf_lit = surf_mass / M2L
	mock_SB = 21.572 + 4.75 - 2.5*np.log10(10**(-6)*surf_lit) + 10*np.log10(1 + z_ref)
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
		'''
		plt.figure()
		ax = plt.subplot(111)
		ax.set_title(r'$stack_{5} \; test \; in \; %s \; band $' % band[ii])
		ax.plot(Ar_diff, SB_ICL, 'b-', label = '$SB_{obs}$')
		ax.set_xlabel('$R[arcsec]$')
		ax.set_xscale('log')
		ax.set_ylabel('$SB[mag/arcsec^2]$')
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')
		ax.axhline(SB_sky[ii], c = 'r', ls = '-.', label = '$local_{BG}$')
		ax1 = ax.twiny()
		ax1.plot(R_diff, SB_ICL, 'b-')
		ax1.set_xscale('log')
		ax1.set_xlabel('$R[kpc]$')
		ax1.tick_params(axis = 'x', which = 'both', direction = 'in')
		ax.legend(loc = 1)
		ax.invert_yaxis()

		plt.savefig('stack_5_in_%s_band.png' % band[ii], dpi = 600)
		plt.close()
		'''
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
		print(popt)

		fit_line = SB_fit(r_fit, M0, Mc, Cc, M2L)

		plt.figure()
		ax = plt.subplot(111)
		ax.set_title(r'$stack_{5} \; in \; %s \; band$' % band[ii])
		ax.errorbar(R_diff[iz], SB_diff[iz], yerr = err_diff[iz], xerr = None, ls = '', fmt = 'ro', label = '$SB_{obs}$')
		ax.plot(r_fit, fit_line, 'b-', label = '$NFB+C$')
		ax.set_xlabel('$R[kpc]$')
		ax.set_ylabel('$SB[mag/arcsec^2]$')
		ax.set_xscale('log')
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')
		ax1 = ax.twiny()
		ax1.plot(Ar_diff[iz], SB_diff[iz], 'b-', alpha = 0.5)
		ax1.set_xscale('log')
		ax1.set_xlabel('$R[arcsec]$')
		ax1.tick_params(axis = 'x', which = 'both', direction = 'in')
		ax.legend(loc = 1)
		ax.invert_yaxis()

		plt.savefig('sky_estimate.png', dpi = 600)
		plt.close()
		raise

	return

def main():
	#mask_B()
	#mask_A()
	#stack_A()
	#stack_B()
	SB_ICL()

if __name__ == "__main__":
	main()
