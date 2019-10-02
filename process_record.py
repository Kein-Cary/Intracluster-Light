import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

import scipy.stats as sts
import astropy.units as U
import subprocess as subpro
import astropy.constants as C

from dustmaps.sfd import SFDQuery
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from matplotlib.patches import Circle, Ellipse
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d as interp
from scipy.interpolate import interp2d as interp2

from resamp import gen
from light_measure import sigmamc
from extinction_redden import A_wave
from resample_modelu import sum_samp, down_samp
from light_measure import light_measure, flux_recal

import time
import random
import sfdmap
Rv = 3.1
sfd = SFDQuery()

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

R0 = 1 # in unit Mpc
Angu_ref = (R0 / Da_ref) * rad2asec
Rpp = Angu_ref / pixel

band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])
mag_add = np.array([0, 0, 0, -0.04, 0.02])
zpot = np.array([22.5, 22.5, 22.5, 22.46, 22.52])
sb_lim = np.array([24.5, 25, 24, 24.35, 22.9])

with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
	catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]

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
Lambd = richness[(redshift >= 0.2) & (redshift <= 0.3)]

csv_UN = pds.read_csv('/home/xkchen/mywork/ICL/data/redmapper/No_star_query_match.csv') # for dr7 catalog
except_ra_Nu = ['%.3f' % ll for ll in csv_UN['ra'] ]
except_dec_Nu = ['%.3f' % ll for ll in csv_UN['dec'] ]
except_z_Nu = ['%.3f' % ll for ll in csv_UN['z'] ]

def sky_03(band_id, ra_g, dec_g, z_g):

	load = '/home/xkchen/mywork/ICL/data/total_data/'
	tmp_load = '/home/xkchen/mywork/ICL/data/test_data/'

	param_sky = '/home/xkchen/mywork/ICL/data/SEX/default_sky_mask.sex'
	out_cat = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A.param'
	out_load_sky = '/home/xkchen/mywork/ICL/data/SEX/result/sky_mask_test.cat'

	data = fits.open(load + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[band_id], ra_g, dec_g, z_g) )
	img = data[0].data
	head_inf = data[0].header
	wcs = awc.WCS(head_inf)
	cx_BCG, cy_BCG = wcs.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)
	R_ph = rad2asec / (Test_model.angular_diameter_distance(z_g).value)
	R_p = R_ph / pixel

	sky0 = data[2].data['ALLSKY'][0]
	sky_x = data[2].data['XINTERP'][0]
	sky_y = data[2].data['YINTERP'][0]
	fact = img.size / sky0.size
	x0 = np.linspace(0, sky0.shape[1] - 1, sky0.shape[1])
	y0 = np.linspace(0, sky0.shape[0] - 1, sky0.shape[0])
	f_sky = interp2(x0, y0, sky0)
	sky_bl = f_sky(sky_x, sky_y) * data[0].header['NMGY'] / fact

	cimg = img + sky_bl
	hdu = fits.PrimaryHDU()
	hdu.data = cimg
	hdu.header = head_inf
	hdu.writeto(tmp_load + 'cimg_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[band_id], ra_g, dec_g, z_g), overwrite = True)

	file_source = tmp_load + 'cimg_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[band_id], ra_g, dec_g, z_g)
	cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s'%(param_sky, out_load_sky, out_cat)
	print(cmd)
	a = subpro.Popen(cmd, shell = True)
	a.wait()

	source = asc.read(out_load_sky)
	Numb = np.array(source['NUMBER'][-1])
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])
	theta = np.array(source['THETA_IMAGE'])
	cx = np.array(source['X_IMAGE']) - 1
	cy = np.array(source['Y_IMAGE']) - 1
	p_type = np.array(source['CLASS_STAR'])
	Kron = 6
	a = Kron * A
	b = Kron * B

	ddr = np.sqrt((cx - cx_BCG)**2 + (cy - cy_BCG)**2)
	ix = ddr >= 0.95 * R_p
	iy = ddr <= 1.15 * R_p
	iz = ix & iy
	s_cx = cx[iz]
	s_cy = cy[iz]
	s_a = a[iz]
	s_b = b[iz]
	s_phi = theta[iz]
	s_Num = len(s_b)

	mask_sky = np.ones((cimg.shape[0], cimg.shape[1]), dtype = np.float)
	ox = np.linspace(0, cimg.shape[1] - 1, cimg.shape[1])
	oy = np.linspace(0, cimg.shape[0] - 1, cimg.shape[0])
	basic_coord = np.array(np.meshgrid(ox,oy))
	major = s_a / 2
	minor = s_b / 2
	senior = np.sqrt(major**2 - minor**2)
	for k in range(s_Num):
		xc = s_cx[k]
		yc = s_cy[k]

		lr = major[k]
		sr = minor[k]
		cr = senior[k]
		chi = s_phi[k]*np.pi/180

		set_r = np.int(np.ceil(1.2 * lr))
		la0 = np.max( [np.int(xc - set_r), 0])
		la1 = np.min( [np.int(xc + set_r +1), img.shape[1] - 1] )
		lb0 = np.max( [np.int(yc - set_r), 0] ) 
		lb1 = np.min( [np.int(yc + set_r +1), img.shape[0] - 1] )

		df1 = lr**2 - cr**2*np.cos(chi)**2
		df2 = lr**2 - cr**2*np.sin(chi)**2
		fr = (basic_coord[0,:][lb0: lb1, la0: la1] - xc)**2*df1 +(basic_coord[1,:][lb0: lb1, la0: la1] - yc)**2*df2\
		- cr**2*np.sin(2*chi)*(basic_coord[0,:][lb0: lb1, la0: la1] - xc)*(basic_coord[1,:][lb0: lb1, la0: la1] - yc)
		idr = fr/(lr**2*sr**2)
		jx = idr <= 1

		iu = np.where(jx == True)
		iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float)
		iv[iu] = np.nan
		mask_sky[lb0: lb1, la0: la1] = mask_sky[lb0: lb1, la0: la1] * iv

	mirro_sky = cimg * mask_sky
	dr = np.sqrt((basic_coord[0,:] - cx_BCG)**2 + (basic_coord[1,:] - cy_BCG)**2)
	idr = (dr >= R_p) & (dr <= 1.1 * R_p)
	pix_cut = mirro_sky[idr]
	BL = np.nanmean(pix_cut)
	'''
	plt.figure()
	ax = plt.subplot(111)
	ax.imshow(mirro_sky, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
	cluster0 = Circle(xy = (cx_BCG, cy_BCG), radius = R_p, fill = False, ec = 'b', alpha = 0.5)
	cluster1 = Circle(xy = (cx_BCG, cy_BCG), radius = 1.1 * R_p, fill = False, ec = 'b', alpha = 0.5)
	ax.add_patch(cluster0)
	ax.add_patch(cluster1)
	ax.set_xlim(0, cimg.shape[1])
	ax.set_ylim(0, cimg.shape[0])
	plt.savefig('/home/xkchen/mywork/ICL/code/sky_0.3mask.png', dpi = 300)
	plt.show()
	'''
	return BL

def mask_A():
	bins = 65
	t0 = time.time()

	load = '/home/xkchen/mywork/ICL/data/total_data/'
	tmp_load = '/home/xkchen/mywork/ICL/data/test_data/'
	param_A = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A.sex'
	out_cat = '/home/xkchen/mywork/ICL/data/SEX/default_mask_A.param'
	out_load_A = '/home/xkchen/mywork/ICL/data/SEX/result/mask_A_test.cat'

	red_rich = Lambd[Lambd > 100]
	red_z = z[Lambd > 100]
	red_ra = ra[Lambd > 100]
	red_dec = dec[Lambd > 100]

	for i in range(1):
		for q in range(20):
			ra_g = red_ra[q]
			dec_g = red_dec[q]
			z_g = red_z[q]

			file = 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[i], ra_g, dec_g, z_g)
			data_f = fits.open(load+file)
			img = data_f[0].data
			head_inf = data_f[0].header
			wcs = awc.WCS(head_inf)
			cx_BCG, cy_BCG = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)
			Da_g = Test_model.angular_diameter_distance(z_g).value
			R_ph = rad2asec / Da_g
			R_p = R_ph / pixel

			x_side = data_f[0].data.shape[1]
			y_side = data_f[0].data.shape[0]
			'''
			sky0 = data_f[2].data['ALLSKY'][0]
			sky_x = data_f[2].data['XINTERP'][0]
			sky_y = data_f[2].data['YINTERP'][0]
			fact = img.size / sky0.size
			x0 = np.linspace(0, sky0.shape[1] - 1, sky0.shape[1])
			y0 = np.linspace(0, sky0.shape[0] - 1, sky0.shape[0])
			f_sky = interp2(x0, y0, sky0)
			sky_bl = f_sky(sky_x, sky_y) * data_f[0].header['NMGY'] / fact
			Back_lel = sky_03(np.int(i), ra_g, dec_g, z_g)
			cimg = img + sky_bl
			sub_img = cimg - Back_lel
			'''
			cimg = img + 0.
			sub_img = cimg - 0.

			x0 = np.linspace(0, img.shape[1] - 1, img.shape[1])
			y0 = np.linspace(0, img.shape[0] - 1, img.shape[0])
			img_grid = np.array(np.meshgrid(x0, y0))
			ra_img, dec_img = wcs.all_pix2world(img_grid[0,:], img_grid[1,:], 1)
			pos = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
			BEV = sfd(pos)
			Av = Rv * BEV
			Al = A_wave(l_wave[i], Rv) * Av
			cc_img = sub_img * 10**(Al / 2.5)

			# SB record
			SB_0, R_0, Ar_0, err_0 = light_measure(img, bins, 1, R_p, cx_BCG, cy_BCG, pixel, z_g)[:4]
			SB3, R3, Ar3, err3 = light_measure(cc_img, bins, 1, R_p, cx_BCG, cy_BCG, pixel, z_g)[:4]

			fig = plt.figure()
			fig.suptitle('SB variation before masking')
			ax = plt.subplot(111)
			ax.plot(R_0, SB_0, 'r-', label = '$original \; image$', alpha = 0.5)
			ax.plot(R3, SB3, 'm-', label = '$extinction \; calibration$', alpha = 0.5)
			ax.set_xscale('log')
			ax.set_xlim(np.nanmin(R_0) + 1, np.nanmax(R_0) + 50)
			ax.set_xlabel('$Radius[kpc]$')
			ax.set_ylabel('$SB[mag/arcsec^2]$')
			ax.legend(loc = 1)
			ax.invert_yaxis()

			bx1 = ax.twiny()
			xtik = ax.get_xticks(minor = True)
			xtik = np.array(xtik)
			xR = xtik * 10**(-3) * rad2asec / Da_g
			xR = xtik * 10**(-3) * rad2asec / Da_g
			id_tt = xtik >= 9e1 
			bx1.set_xticks(xtik[id_tt])
			bx1.set_xticklabels(['$%.2f^{ \prime \prime }$' % uu for uu in xR[id_tt]])
			bx1.set_xlim(ax.get_xlim())
			bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

			plt.savefig('SB_befo_mask_ra%.3f_dec%.3f_z%.3f_%s_band.png' % (ra_g, dec_g, z_g, band[i]), dpi = 300)
			plt.close()

			hdu = fits.PrimaryHDU()
			hdu.data = cc_img
			hdu.header = head_inf
			hdu.writeto(tmp_load + 'source/source_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[i], ra_g, dec_g, z_g), overwrite = True)

			file_source = tmp_load + 'source/source_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[i], ra_g, dec_g, z_g)
			'''
			cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s'%(param_A, out_load_A, out_cat) # 1.5sigma
			'''
			dete_thresh = sb_lim[i] + 10*np.log10((1 + z_g)/(1 + z_ref))
			dete_thresh = '%.3f' % dete_thresh + ',%.2f' % zpot[i]
			dete_min = '10'
			ana_thresh = dete_thresh *1
			cmd = (
				'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s -DETECT_MINAREA %s -DETECT_THRESH %s -ANALYSIS_THRESH %s'
				%(param_A, out_load_A, out_cat, dete_min, dete_thresh, ana_thresh))
			print(cmd)
			tpp = subpro.Popen(cmd, shell = True)
			tpp.wait()

			source = asc.read(out_load_A)
			Numb = np.array(source['NUMBER'][-1])
			A = np.array(source['A_IMAGE'])
			B = np.array(source['B_IMAGE'])
			theta = np.array(source['THETA_IMAGE'])
			cx = np.array(source['X_IMAGE']) - 1
			cy = np.array(source['Y_IMAGE']) - 1
			p_type = np.array(source['CLASS_STAR'])
			Kron = 6
			a = Kron * A
			b = Kron * B
			# photometric catalogue

			cat = pds.read_csv('/home/xkchen/mywork/ICL/data/star_dr12/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g) ) # dr8
			cat = pds.read_csv(mask, skiprows = 1)
			set_ra = np.array(cat['ra'])
			set_dec = np.array(cat['dec'])
			set_mag = np.array(cat['r'])
			OBJ = np.array(cat['type'])
			xt = cat['Column1']
			tau = 6 # the mask size set as 6 * FWHM from dr12

			set_A = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i']]) * tau / pixel
			set_B = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i']]) * tau / pixel
			set_chi = np.zeros(set_A.shape[1], dtype = np.float)

			lln = np.array([len(set_A[:,ll][set_A[:,ll] > 0 ]) for ll in range(set_A.shape[1]) ])
			lr_iso = np.array([np.max(set_A[:,ll]) for ll in range(set_A.shape[1]) ])
			sr_iso = np.array([np.max(set_B[:,ll]) for ll in range(set_B.shape[1]) ])

			# bright stars
			x, y = wcs.all_world2pix(set_ra * U.deg, set_dec * U.deg, 1)
			ia = (x >= 0) & (x <= img.shape[1])
			ib = (y >= 0) & (y <= img.shape[0])
			ie = (set_mag <= 20)
			iq = lln >= 2
			ig = OBJ == 6
			ic = (ia & ib & ie & ig & iq)
			sub_x0 = x[ic]
			sub_y0 = y[ic]
			sub_A0 = lr_iso[ic]
			sub_B0 = sr_iso[ic]
			sub_chi0 = set_chi[ic]
			# saturated source(may not stars)
			xa = ['SATURATED' in pp for pp in xt]
			xv = np.array(xa)
			idx = xv == True
			ipx = (idx & ia & ib)

			sub_x2 = x[ipx]
			sub_y2 = y[ipx]
			sub_A2 = 3 * lr_iso[ipx]
			sub_B2 = 3 * sr_iso[ipx]
			sub_chi2 = set_chi[ipx]

			comx = np.r_[sub_x0[sub_A0 > 0], sub_x2[sub_A2 > 0]]
			comy = np.r_[sub_y0[sub_A0 > 0], sub_y2[sub_A2 > 0]]
			Lr = np.r_[sub_A0[sub_A0 > 0], sub_A2[sub_A2 > 0]]
			Sr = np.r_[sub_B0[sub_A0 > 0], sub_B2[sub_A2 > 0]]
			phi = np.r_[sub_chi0[sub_A0 > 0], sub_chi2[sub_A2 > 0]]

			cx = np.r_[cx, comx]
			cy = np.r_[cy, comy]
			a = np.r_[a, Lr]
			b = np.r_[b, Sr]
			theta = np.r_[theta, phi]

			Numb = Numb + len(comx)
			mask_A = np.ones((cimg.shape[0], cimg.shape[1]), dtype = np.float)
			ox = np.linspace(0, cimg.shape[1] - 1, cimg.shape[1])
			oy = np.linspace(0, cimg.shape[0] - 1, cimg.shape[0])
			basic_coord = np.array(np.meshgrid(ox,oy))
			major = a / 2
			minor = b / 2
			senior = np.sqrt(major**2 - minor**2)

			tdr = np.sqrt((cx - cx_BCG)**2 + (cy - cy_BCG)**2)
			dr00 = np.where(tdr == np.min(tdr))[0]
			for k in range(Numb):
				xc = cx[k]
				yc = cy[k]
				lr = major[k]
				sr = minor[k]
				cr = senior[k]
				chi = theta[k]*np.pi/180

				set_r = np.int(np.ceil(1.2 * lr))
				la0 = np.max( [np.int(xc - set_r), 0])
				la1 = np.min( [np.int(xc + set_r +1), img.shape[1] - 1] )
				lb0 = np.max( [np.int(yc - set_r), 0] ) 
				lb1 = np.min( [np.int(yc + set_r +1), img.shape[0] - 1] )

				if k == dr00[0] :
					continue
				else:
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

			mirro_A = mask_A * cc_img
			SB4, R4, Ar4, err4 = light_measure(mirro_A, bins, 1, R_p, cx_BCG, cy_BCG, pixel, z_g)[:4]

			fig = plt.figure()
			fig.suptitle('A mask ra%.3f dec%.3f z%.3f %s band' % (ra_g, dec_g, z_g, band[i]) )
			cluster1 = Circle(xy = (cx_BCG, cy_BCG), radius = R_p, fill = False, ec = 'b', alpha = 0.5, label = 'cluster region[1Mpc]')
			ax1 = plt.subplot(111)
			tf = ax1.imshow(mirro_A, cmap = 'Greys', vmin = 1e-5, origin = 'lower', norm = mpl.colors.LogNorm())
			plt.colorbar(tf, ax = ax1, fraction = 0.035, pad = 0.01, label = '$flux[nmaggy]$')
			for kk in range(Numb):
				clco = Ellipse(xy = (cx[kk], cy[kk]), width = a[kk], height = b[kk], angle = theta[kk], fill = False, ec = 'r', alpha = 0.5)
				ax1.add_patch(clco)
			ax1.add_patch(cluster1)
			ax1.set_title('A masked image')
			ax1.set_xlim(0, cc_img.shape[1])
			ax1.set_ylim(0, cc_img.shape[0])
			plt.savefig('A_mask_ra%.3f_dec%.3f_z%.3f_%s_band.png' % (ra_g, dec_g, z_g, band[i]), dpi = 300)
			plt.close()

			fig = plt.figure()
			fig.suptitle('SB variation during A mask ra%.3f dec%.3f z%.3f %s band' % (ra_g, dec_g, z_g, band[i]) )
			ax = plt.subplot(111)
			ax.plot(R4, SB4, 'g-', label = '$allpying \; A \; mask$', alpha = 0.5)
			ax.plot(R3, SB3, 'r-', label = '$extinction \; calibration$', alpha = 0.5)
			ax.set_xscale('log')
			ax.set_xlim(np.nanmin(R3) + 1, np.nanmax(R3) + 50)
			ax.set_xlabel('$Radius[kpc]$')
			ax.set_ylabel('$SB[mag/arcsec^2]$')
			ax.legend(loc = 1)
			ax.invert_yaxis()
			bx1 = ax.twiny()
			xtik = ax.get_xticks(minor = True)
			xtik = np.array(xtik)
			xR = xtik * 10**(-3) * rad2asec / Da_g
			xR = xtik * 10**(-3) * rad2asec / Da_g
			id_tt = xtik >= 9e1 
			bx1.set_xticks(xtik[id_tt])
			bx1.set_xticklabels(['$%.2f^{ \prime \prime }$' % uu for uu in xR[id_tt]])
			bx1.set_xlim(ax.get_xlim())
			bx1.tick_params(axis = 'both', which = 'both', direction = 'in')
			plt.savefig('SB_with_mask_ra%.3f_dec%.3f_z%.3f_%s_band.png' % (ra_g, dec_g, z_g, band[i]), dpi = 300)
			plt.close()

			t1 = time.time() - t0
			print('t = ', t1)
			hdu = fits.PrimaryHDU()
			hdu.data = mirro_A
			hdu.header = head_inf
			hdu.writeto('/home/xkchen/mywork/ICL/data/test_data/mask/A_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[i], ra_g, dec_g, z_g),overwrite = True)

	raise
	return

def mask_B():
	bins = 65
	tmp_load = '/home/xkchen/mywork/ICL/data/test_data/'
	load = '/home/xkchen/mywork/ICL/data/total_data/'
	mask = '/home/xkchen/mywork/ICL/data/star_catalog/'

	red_rich = Lambd[Lambd > 100]
	red_z = z[Lambd > 100]
	red_ra = ra[Lambd > 100]
	red_dec = dec[Lambd > 100]

	for pp in range(20):
		ra_g = red_ra[pp]
		dec_g = red_dec[pp]
		z_g = red_z[pp]
		for q in range(1):

			file = 'source/source_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[q], ra_g, dec_g, z_g)

			data_f = fits.open(tmp_load + file)
			img = data_f[0].data
			head_inf = data_f[0].header
			wcs = awc.WCS(head_inf)
			x_side = data_f[0].data.shape[1]
			y_side = data_f[0].data.shape[0]
			cenx, ceny = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)
			Da_g = Test_model.angular_diameter_distance(z_g).value
			R_ph = rad2asec / Da_g
			R_p = R_ph / pixel

			cat = pds.read_csv('/home/xkchen/mywork/ICL/data/star_dr12/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g) ) # dr8
			cat = pds.read_csv(mask, skiprows = 1)
			set_ra = np.array(cat['ra'])
			set_dec = np.array(cat['dec'])
			set_mag = np.array(cat['r'])
			OBJ = np.array(cat['type'])
			xt = cat['Column1']
			tau = 6 # the mask size set as 6 * FWHM from dr12

			set_A = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i']]) * tau / pixel
			set_B = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i']]) * tau / pixel
			set_chi = np.zeros(set_A.shape[1], dtype = np.float)

			lln = np.array([len(set_A[:,ll][set_A[:,ll] > 0 ]) for ll in range(set_A.shape[1]) ])
			lr_iso = np.array([np.max(set_A[:,ll]) for ll in range(set_A.shape[1]) ])
			sr_iso = np.array([np.max(set_B[:,ll]) for ll in range(set_B.shape[1]) ])

			# bright stars
			x, y = wcs.all_world2pix(set_ra * U.deg, set_dec * U.deg, 1)
			ia = (x >= 0) & (x <= img.shape[1])
			ib = (y >= 0) & (y <= img.shape[0])
			ie = (set_mag <= 20)
			iq = lln >= 2
			ig = OBJ == 6
			ic = (ia & ib & ie & ig & iq)
			sub_x0 = x[ic]
			sub_y0 = y[ic]
			sub_A0 = lr_iso[ic]
			sub_B0 = sr_iso[ic]
			sub_chi0 = set_chi[ic]
			# saturated source(may not stars)
			xa = ['SATURATED' in pp for pp in xt]
			xv = np.array(xa)
			idx = xv == True
			ipx = (idx & ia & ib)

			sub_x2 = x[ipx]
			sub_y2 = y[ipx]
			sub_A2 = 3 * lr_iso[ipx]
			sub_B2 = 3 * sr_iso[ipx]
			sub_chi2 = set_chi[ipx]

			comx = np.r_[sub_x0[sub_A0 > 0], sub_x2[sub_A2 > 0]]
			comy = np.r_[sub_y0[sub_A0 > 0], sub_y2[sub_A2 > 0]]
			Lr = np.r_[sub_A0[sub_A0 > 0], sub_A2[sub_A2 > 0]]
			Sr = np.r_[sub_B0[sub_A0 > 0], sub_B2[sub_A2 > 0]]
			phi = np.r_[sub_chi0[sub_A0 > 0], sub_chi2[sub_A2 > 0]]

			Numb = len(comx)
			mask_B = np.ones((img.shape[0], img.shape[1]), dtype = np.float)
			ox = np.linspace(0, img.shape[1] - 1, img.shape[1])
			oy = np.linspace(0, img.shape[0] - 1, img.shape[0])
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

			hdu = fits.PrimaryHDU()
			hdu.data = mirro_B
			hdu.header = head_inf
			hdu.writeto(
				'/home/xkchen/mywork/ICL/data/test_data/mask/B_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[q], ra_g, dec_g, z_g),overwrite = True)

			SB_0, R_0, Ar_0, err_0 = light_measure(img, bins, 1, R_p, cenx, ceny, pixel, z_g)[:4]
			SB_1, R_1, Ar_1, err_1 = light_measure(mirro_B, bins, 1, R_p, cenx, ceny, pixel, z_g)[:4]

			fig = plt.figure()
			fig.suptitle('B mask ra%.3f dec%.3f z%.3f %s band' % (ra_g, dec_g, z_g, band[q]) )
			cluster1 = Circle(xy = (cenx, ceny), radius = R_p, fill = False, ec = 'b', alpha = 0.5, label = 'cluster region[1Mpc]')
			ax1 = plt.subplot(111)
			ax1.imshow(mirro_B, cmap = 'Greys', vmin = 1e-5, origin = 'lower', norm = mpl.colors.LogNorm())
			for kk in range(Numb):
				clco = Ellipse(xy = (comx[kk], comy[kk]), width = Lr[kk], height = Sr[kk], angle = phi[kk], 
					fill = False, ec = 'r', linewidth = 0.5, alpha = 0.5)
				ax1.add_patch(clco)
			ax1.add_patch(cluster1)
			ax1.set_title('B Mask image')
			ax1.set_xlim(0, img.shape[1])
			ax1.set_ylim(0, img.shape[0])

			plt.savefig('B_mask_ra%.3f_dec%.3f_z%.3f_%s_band.png' % (ra_g, dec_g, z_g, band[q]), dpi = 600)
			plt.close()

			fig = plt.figure(figsize = (16, 8))
			fig.suptitle('SB during B mask ra%.3f dec%.3f z%.3f %s band' % (ra_g, dec_g, z_g, band[q]) )
			ax = plt.subplot(111)
			ax.plot(R_0, SB_0, 'r-', label = '$Extinction \; correct \; image$', alpha = 0.5)
			ax.plot(R_1, SB_1, 'g-', label = '$B \; mask \; image$', alpha = 0.5)	
			ax.set_xscale('log')
			ax.set_xlim(np.nanmin(R_0) + 1, np.nanmax(R_0) + 50)
			ax.set_xlabel('$Radius[kpc]$')
			ax.set_ylabel('$SB[mag/arcsec^2]$')
			ax.legend(loc = 1)
			ax.invert_yaxis()

			bx1 = ax.twiny()
			xtik = ax.get_xticks(minor = True)
			xR = xtik * 10**(-3) * rad2asec / Da_g
			xR = xtik * 10**(-3) * rad2asec / Da_g
			id_tt = xtik >= 9e1 
			bx1.set_xticks(xtik[id_tt])
			bx1.set_xticklabels(['$%.2f^{ \prime \prime }$' % uu for uu in xR[id_tt]])
			bx1.set_xlim(ax.get_xlim())
			bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

			plt.savefig('SB_with_Bmask_ra%.3f_dec%.3f_z%.3f_%s_band.png' % (ra_g, dec_g, z_g, band[q]), dpi = 300)
			plt.close()

	raise
	return

def resamp_B():
	bins = 65
	load = '/home/xkchen/mywork/ICL/data/test_data/'

	red_rich = Lambd[Lambd > 100]
	red_z = z[Lambd > 100]
	red_ra = ra[Lambd > 100]
	red_dec = dec[Lambd > 100]
	bins = 65
	for ii in range(1):
		for jj in range(20):
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
			x0 = resam.shape[1]
			y0 = resam.shape[0]

			keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2','CENTER_X','CENTER_Y',
					'CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z', 'P_SCALE']
			value = ['T', 32, 2, x0, y0, ix0, iy0, xn, yn, RA0, DEC0, ra_g, dec_g, z_g, pixel]
			ff = dict(zip(keys,value))
			fil = fits.Header(ff)
			fits.writeto(load + 
				'resamp/resamp_B-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), resam, header = fil, overwrite=True)

			SB1, R1, Anr1, err1 = light_measure(img, bins, 1, Rp, cx, cy, pixel, z_g)[:4]
			SB_ref = SB1 + 10*np.log10((1 + z_ref) / (1 + z_g))
			Ar_ref = Anr1 * mu
			id_nan = np.isnan(SB_ref)
			ivx = id_nan == False
			f_SB = interp(R1[ivx], SB_ref[ivx], kind = 'cubic')
			SB2, R2, Anr2, err2 = light_measure(f_goal, bins, 1, Rp, cx, cy, pixel * mu, z_ref)[:4]
			SB3, R3, Anr3, err3 = light_measure(resam, bins, 1, Rpp, xn, yn, pixel, z_ref)[:4]

			fig = plt.figure(figsize = (16, 8))
			gs = gridspec.GridSpec(2,1, height_ratios = [4,1])
			fig.suptitle('SB variation during resampling ra%.3f dec%.3f z%.3f %s band' % (ra_g, dec_g, z_g, band[ii]) )
			ax = plt.subplot(gs[0])
			cx = plt.subplot(gs[1], sharex = ax)
			ax.plot(R1, SB1, 'r-', label = '$ B \; Mask$', alpha = 0.5)
			ax.plot(R2, SB2, 'g-', label = '$ scaled \; image$', alpha = 0.5)
			ax.plot(R1, SB_ref, 'r--', label = '$reference \; profile$', alpha = 0.5)
			ax.plot(R3, SB3, 'b-', label = '$scaled + resample \; image$', alpha = 0.5)

			ax.set_xscale('log')
			ax.set_xlabel('$Radius[kpc]$')
			ax.set_ylabel('$SB[mag/arcsec^2]$')
			ax.set_xlim(np.nanmin(R3) + 1, np.nanmax(R3) + 50)
			ax.legend(loc = 1)
			ax.invert_yaxis()

			bx1 = ax.twiny()
			xtik = ax.get_xticks(minor = True)
			xR = xtik * 10**(-3) * rad2asec / Da_g
			xR = xtik * 10**(-3) * rad2asec / Da_g
			id_tt = xtik >= 9e1 
			bx1.set_xticks(xtik[id_tt])
			bx1.set_xticklabels(['$%.2f^{ \prime \prime }$' % uu for uu in xR[id_tt]])
			bx1.set_xlim(ax.get_xlim())
			bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

			id_nan = np.isnan(SB3)
			iux = id_nan == False
			ddbr = R3[iux][ (R3[iux] > np.min(R1[ivx])) & (R3[iux] < np.max(R1[ivx])) ]
			ddb = SB3[iux][ (R3[iux] > np.min(R1[ivx])) & (R3[iux] < np.max(R1[ivx])) ] - f_SB(ddbr)
			cx.plot(ddbr, ddb, 'g*')
			cx.axhline(y = 0, linestyle = '--', color = 'b')
			cx.set_xscale('log')
			cx.set_xlabel('$Radius[kpc]$')
			cx.set_ylabel('$SB_{after \; resample} - SB_{ref}$')
			plt.subplots_adjust(hspace = 0)

			plt.savefig('B_mask_resamp_ra%.3f_dec%.3f_z%.3f_%s_band.png' % (ra_g, dec_g, z_g, band[ii]), dpi = 300)
			plt.close()
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

	red_rich = Lambd[Lambd > 100]
	red_z = z[Lambd > 100]
	red_ra = ra[Lambd > 100]
	red_dec = dec[Lambd > 100]

	for ii in range(1):
		tot_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
		tot_count = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
		p_count_total = np.zeros((len(Ny), len(Nx)), dtype = np.float)

		for jj in range(20):
			ra_g = red_ra[jj]
			dec_g = red_dec[jj]
			z_g = red_z[jj]
			Da_g = Test_model.angular_diameter_distance(z_g).value
			if ('%.3f' % ra_g in except_ra_Nu ) & ('%.3f' % dec_g in except_dec_Nu) & ('%.3f' % z_g in except_z_Nu):
				continue
			else:
				data = fits.getdata(load + 'resamp_B-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
				img = data[0]
				xn = data[1]['CENTER_X']
				yn = data[1]['CENTER_Y']

				la0 = np.int(y0 - yn)
				la1 = np.int(y0 - yn + img.shape[0])
				lb0 = np.int(x0 - xn)
				lb1 = np.int(x0 - xn + img.shape[1])

				idx = np.isnan(img)
				idv = idx == False
				tot_array[la0:la1, lb0:lb1][idv] = tot_array[la0:la1, lb0:lb1][idv] + img[idv]
				tot_count[la0: la1, lb0: lb1][idv] = img[idv]
				id_nan = np.isnan(tot_count)
				id_fals = np.where(id_nan == False)
				p_count_total[id_fals] = p_count_total[id_fals] + 1
				tot_count[id_fals] = np.nan

		mean_total = tot_array / p_count_total
		where_are_inf = np.isinf(mean_total)
		mean_total[where_are_inf] = np.nan
		id_zeros = np.where(p_count_total == 0)
		mean_total[id_zeros] = np.nan

		SB_tot, R_tot, Ar_tot, err_tot = light_measure(mean_total, bins, 1, Rpp, x0, y0, pixel, z_ref)[:4]
		SB_tot = SB_tot + mag_add[ii]

		stack_B = np.array([SB_tot, R_tot, Ar_tot, err_tot])
		with h5py.File('/home/xkchen/mywork/ICL/data/test_data/SB_stack_Bmask_%s_band.h5' % band[ii], 'w') as f:
			f['a'] = np.array(stack_B)
		with h5py.File('/home/xkchen/mywork/ICL/data/test_data/SB_stack_Bmask_%s_band.h5' % band[ii]) as f:
			for tt in range(len(stack_B)):
				f['a'][tt,:] = stack_B[tt,:]

		plt.figure(figsize = (16, 8))
		ax0 = plt.subplot(121)
		ax1 = plt.subplot(122)
		Clus0 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-')
		Clus1 = Circle(xy = (x0, y0), radius = 0.2 * Rpp, fill = False, ec = 'r', ls = '--')
		tf = ax0.imshow(mean_total, cmap = 'Greys', vmin = 1e-5, origin = 'lower', norm = mpl.colors.LogNorm())
		plt.colorbar(tf, ax = ax0, fraction = 0.045, pad = 0.01, label = '$flux[nmaggy]$')
		ax0.add_patch(Clus0)
		ax0.add_patch(Clus1)
		ax0.set_title('stack mask B img')
		ax0.set_xlim(x0 - 1.2 * Rpp, x0 + 1.2 * Rpp)
		ax0.set_ylim(y0 - 1.2 * Rpp, y0 + 1.2 * Rpp)

		ax1.errorbar(R_tot, SB_tot, yerr = err_tot, xerr = None, ls = '', fmt = 'ro')
		ax1.set_xscale('log')
		ax1.set_xlabel('$Radius[kpc]$')
		ax1.set_ylabel('$SB[mag/arcsec^2]$')
		ax1.set_xlim(np.nanmin(R_tot) + 1, np.nanmax(R_tot) + 50)
		ax1.invert_yaxis()

		bx1 = ax1.twiny()
		xtik = ax1.get_xticks(minor = True)
		xR = xtik * 10**(-3) * rad2asec / Da_g
		xR = xtik * 10**(-3) * rad2asec / Da_g
		id_tt = xtik >= 9e1 
		bx1.set_xticks(xtik[id_tt])
		bx1.set_xticklabels(['$%.2f^{ \prime \prime }$' % uu for uu in xR[id_tt]])
		bx1.set_xlim(ax1.get_xlim())
		bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

		plt.tight_layout()
		plt.savefig('stack_mask_B.png', dpi = 300)
		plt.close()
	raise
	return

def resamp_A():
	load = '/home/xkchen/mywork/ICL/data/test_data/resamp/'
	red_rich = Lambd[Lambd > 100]
	red_z = z[Lambd > 100]
	red_ra = ra[Lambd > 100]
	red_dec = dec[Lambd > 100]
	bins = 65
	for ii in range(1):
		for jj in range(20):
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

			f_D = flux_recal(img, z_g, z_ref)
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

			SB1, R1, Anr1, err1 = light_measure(img, bins, 1, Rp, cx_BCG, cy_BCG, pixel, z_g)[:4]
			SB_ref = SB1 + 10*np.log10((1 + z_ref) / (1 + z_g))
			Ar_ref = Anr1 * miu
			id_nan = np.isnan(SB_ref)
			ivx = id_nan == False
			f_SB = interp(R1[ivx], SB_ref[ivx], kind = 'cubic')
			SB2, R2, Anr2, err2 = light_measure(f_D, bins, 1, Rp, cx_BCG, cy_BCG, pixel * miu, z_ref)[:4]
			SB3, R3, Anr3, err3 = light_measure(resam_d, bins, 1, Rpp, xnd, ynd, pixel, z_ref)[:4]

			fig = plt.figure(figsize = (16, 8))
			gs = gridspec.GridSpec(2,1, height_ratios = [4,1])
			fig.suptitle('A mask resampling ra%.3f dec%.3f z%.3f %s band' % (ra_g, dec_g, z_g, band[ii]) )
			ax = plt.subplot(gs[0])
			cx = plt.subplot(gs[1], sharex = ax)
			ax.plot(R1, SB1, 'r-', label = '$ A \; Mask$', alpha = 0.5)
			ax.plot(R2, SB2, 'g-', label = '$ scaled \; image$', alpha = 0.5)
			ax.plot(R1, SB_ref, 'r--', label = '$reference \; profile$', alpha = 0.5)
			ax.plot(R3, SB3, 'b-', label = '$scaled + resample \; image$', alpha = 0.5)

			ax.set_xscale('log')
			ax.set_xlabel('$Radius[kpc]$')
			ax.set_ylabel('$SB[mag/arcsec^2]$')
			ax.set_xlim(np.nanmin(R3) + 1, np.nanmax(R3) + 50)
			ax.legend(loc = 1)
			ax.invert_yaxis()

			bx1 = ax.twiny()
			xtik = ax.get_xticks(minor = True)
			xR = xtik * 10**(-3) * rad2asec / Da_g
			xR = xtik * 10**(-3) * rad2asec / Da_g
			id_tt = xtik >= 9e1 
			bx1.set_xticks(xtik[id_tt])
			bx1.set_xticklabels(['$%.2f^{ \prime \prime }$' % uu for uu in xR[id_tt]])
			bx1.set_xlim(ax.get_xlim())
			bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

			id_nan = np.isnan(SB3)
			iux = id_nan == False
			ddbr = R3[iux][ (R3[iux] > np.min(R1[ivx])) & (R3[iux] < np.max(R1[ivx])) ]
			ddb = SB3[iux][ (R3[iux] > np.min(R1[ivx])) & (R3[iux] < np.max(R1[ivx])) ] - f_SB(ddbr)
			cx.plot(ddbr, ddb, 'g*')
			cx.axhline(y = 0, linestyle = '--', color = 'b')
			cx.set_xscale('log')
			cx.set_xlabel('$Radius[kpc]$')
			cx.set_ylabel('$SB_{after \; resample} - SB_{ref}$')
			plt.subplots_adjust(hspace = 0)
			plt.savefig('A_mask_resamp_ra%.3f_dec%.3f_z%.3f_%s_band.png' % (ra_g, dec_g, z_g, band[ii]), dpi = 300)
			plt.close()

	raise
	return

def stack_A():

	load = '/home/xkchen/mywork/ICL/data/test_data/resamp/'
	x0 = 2427
	y0 = 1765
	bins = 65
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	sum_grid = np.array(np.meshgrid(Nx, Ny))

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

			data = fits.getdata(load + 'resamp_A-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
			img = data[0]
			xn = data[1]['CENTER_X']
			yn = data[1]['CENTER_Y']

			la0 = np.int(y0 - yn)
			la1 = np.int(y0 - yn + img.shape[0])
			lb0 = np.int(x0 - xn)
			lb1 = np.int(x0 - xn + img.shape[1])

			idx = np.isnan(img)
			idv = np.where(idx == False)
			sum_array_D[la0:la1, lb0:lb1][idv] = sum_array_D[la0:la1, lb0:lb1][idv] + img[idv]
			count_array_D[la0: la1, lb0: lb1][idv] = img[idv]
			id_nan = np.isnan(count_array_D)
			id_fals = np.where(id_nan == False)
			p_count_D[id_fals] = p_count_D[id_fals] + 1
			count_array_D[la0: la1, lb0: lb1][idv] = np.nan

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
		with h5py.File(
			'/home/xkchen/mywork/ICL/data/test_data/SB_stack_Amask_%s_band.h5' % band[ii], 'w') as f:
			f['a'] = stackA
		with h5py.File('/home/xkchen/mywork/ICL/data/test_data/SB_stack_Amask_%s_band.h5' % band[ii]) as f:
			for tt in range(len(stackA)):
				f['a'][tt,:] = stackA[tt,:]

		plt.figure(figsize = (16, 8))
		ax0 = plt.subplot(121)
		ax1 = plt.subplot(122)
		Clus0 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', ls = '-')
		Clus1 = Circle(xy = (x0, y0), radius = 0.2 * Rpp, fill = False, ec = 'r', ls = '--')
		tf = ax0.imshow(mean_array_D, cmap = 'Greys', vmin = 1e-5, origin = 'lower', norm = mpl.colors.LogNorm())
		plt.colorbar(tf, ax = ax0, fraction = 0.045, pad = 0.01, label = '$flux[nmaggy]$')
		ax0.add_patch(Clus0)
		ax0.add_patch(Clus1)
		ax0.set_title('stack mask A img')
		ax0.set_xlim(x0 - 1.2 * Rpp, x0 + 1.2 * Rpp)
		ax0.set_ylim(y0 - 1.2 * Rpp, y0 + 1.2 * Rpp)

		ax1.errorbar(R1, SB1, yerr = err1, xerr = None, ls = '', fmt = 'ro')
		ax1.set_xscale('log')
		ax1.set_xlabel('$Radius[kpc]$')
		ax1.set_ylabel('$SB[mag/arcsec^2]$')
		ax1.set_xlim(np.nanmin(R1) + 1, np.nanmax(R1) + 50)
		ax1.invert_yaxis()

		bx1 = ax1.twiny()
		xtik = ax1.get_xticks(minor = True)
		xR = xtik * 10**(-3) * rad2asec / Da_g
		id_tt = xtik >= 9e1 
		bx1.set_xticks(xtik[id_tt])
		bx1.set_xticklabels(['$%.2f^{ \prime \prime }$' % uu for uu in xR[id_tt]])
		bx1.set_xlim(ax1.get_xlim())
		bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

		plt.tight_layout()
		plt.savefig('stack_mask_A.png', dpi = 300)
		plt.close()
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
	rho_c = (kpc2m / Msun2kg)*(3*H0**2) / (8*np.pi*G)
	r200_c = (3*M / (4*np.pi*rho_c*200))**(1/3) 
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

		po = np.array( [m0[0], mc[0], cc[0], m2l[0]] )
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
		'''
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
		ax.set_xlim(np.nanmin(R_diff + 1), np.nanmax(R_diff + 20))
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
	#sky_03()

	#mask_A()
	#mask_B()

	#resamp_B()
	#stack_B()

	#resamp_A()
	stack_A()

	#SB_ICL()

if __name__ == "__main__":
	main()
