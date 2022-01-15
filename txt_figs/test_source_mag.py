import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import astropy.io.fits as fits

import wget
import mechanize
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.wcs as awc
import astropy.io.ascii as asc

from io import StringIO
from scipy import optimize
from scipy import interpolate as interp
from scipy import integrate as integ
from astropy import cosmology as apcy

import time
from img_pre_selection import WCS_to_pixel_func
from light_measure import light_measure_Z0_weit, light_measure_weit
# from fig_out_module import cumu_mass_func

### === ### cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

band = ['r', 'g', 'i']
Mag_sun = [ 4.65, 5.11, 4.53 ]
L_wave = np.array([ 6166, 4686, 7480 ])
psf_FWHM = [ 1.56, 1.67, 1.50 ] # arcsec

pixel = 0.396

ref_Rii = np.array([0.23,  0.68,  1.03,   1.76,   3.00, 
					4.63,  7.43,  11.42,  18.20,  28.20, 
					44.21, 69.00, 107.81, 168.20, 263.00]) # in unit 'arcsec'
ref_R_pix = ref_Rii / pixel

### === ### funcs
def deV_func(r, I_0, r_e):
	I_r = I_0 * np.exp( -7.67 * (r / r_e)**(1/4) )
	return I_r

def exp_func(r, I_0, r_e):
	I_r = I_0 * np.exp( -1.68 * r / r_e )
	return I_r

def comp_func(r, I_0, re_0, I_1, re_1, frac_deV):

	Ir_0 = deV_func( r, I_0, re_0 )
	Ir_1 = exp_func( r, I_1, re_1 )
	Ir = frac_deV * Ir_0 + ( 1 - frac_deV ) * Ir_1
	return Ir

def cumu_mass_func(rp, surf_mass, N_grid = 100):

	try:
		NR = len(rp)
	except:
		rp = np.array([ rp ])
		NR = len(rp)

	intep_sigma_F = interp.interp1d( rp, surf_mass, kind = 'linear', fill_value = 'extrapolate',)
	cumu_mass = np.zeros( NR, )

	for ii in range( NR ):

		lg_r_min = np.log10( rp.min() )
		new_rp = np.logspace( lg_r_min, np.log10( rp[ii] ), N_grid)
		new_mass = intep_sigma_F( new_rp )
		cumu_mass[ ii ] = integ.simps( 2 * np.pi * new_rp * new_mass, new_rp)

	return cumu_mass

def star_pos_func( star_cat, Head_info):

	wcs_lis = awc.WCS( Head_info )

	## stars catalog
	p_cat = pds.read_csv( star_cat, skiprows = 1)
	set_ra = np.array( p_cat['ra'])
	set_dec = np.array( p_cat['dec'])
	set_mag = np.array( p_cat['r'])
	OBJ = np.array( p_cat['type'])
	xt = p_cat['Column1']
	flags = [str(qq) for qq in xt]

	x, y = wcs_lis.all_world2pix( set_ra * U.deg, set_dec * U.deg, 0, ra_dec_order = True,)

	set_A = np.array( [ p_cat['psffwhm_r'] , p_cat['psffwhm_g'], p_cat['psffwhm_i']]) / pixel
	set_B = np.array( [ p_cat['psffwhm_r'] , p_cat['psffwhm_g'], p_cat['psffwhm_i']]) / pixel
	set_chi = np.zeros(set_A.shape[1], dtype = np.float32)

	lln = np.array([len(set_A[:,ll][set_A[:,ll] > 0 ]) for ll in range(set_A.shape[1]) ])
	lr_iso = np.array([np.max(set_A[:,ll]) for ll in range(set_A.shape[1]) ])
	sr_iso = np.array([np.max(set_B[:,ll]) for ll in range(set_B.shape[1]) ])

	# normal stars
	iq = lln >= 2 ## at lest observed in 2 band
	ig = OBJ == 6
	ie = (set_mag <= 20)

	ic = (ie & ig & iq)
	sub_x0 = x[ic]
	sub_y0 = y[ic]

	sub_A0 = lr_iso[ic] * 3 # 5 # 7.5
	sub_B0 = sr_iso[ic] * 3 # 5 # 7.5
	sub_chi0 = set_chi[ic]

	sub_ra0, sub_dec0 = set_ra[ic], set_dec[ic]

	## for stars
	ddx = np.around( sub_x0 )
	d_x0 = np.array( list( set( ddx ) ) )

	m_A0, m_B0, m_chi0 = [], [], []
	m_x0, m_y0 = [], []

	for jj in range( len( d_x0 ) ):
		dex_0 = list( ddx ).index( d_x0[jj] )

		m_x0.append( sub_x0[dex_0] )
		m_y0.append( sub_y0[dex_0] )
		m_A0.append( sub_A0[dex_0] )
		m_B0.append( sub_B0[dex_0] )
		m_chi0.append( sub_chi0[dex_0] )

	m_A0, m_B0, m_chi0 = np.array( m_A0 ), np.array( m_B0 ), np.array( m_chi0 )
	m_x0, m_y0 = np.array( m_x0 ), np.array( m_y0 )

	cm_x0, cm_y0 = sub_x0 + 0, sub_y0 + 0
	cm_A0, cm_B0, cm_chi0 = sub_A0 + 0, sub_B0 + 0, sub_chi0 + 0

	return cm_x0, cm_y0, cm_A0, cm_B0, cm_chi0, sub_ra0, sub_dec0

def sub_mask_func( img, around_arr):

	_cx, _cy, _a, _b, _theta = around_arr[:]

	major = _a / 2
	minor = _b / 2
	senior = np.sqrt(major**2 - minor**2)

	Ns = len( major )

	mask_path = np.ones((img.shape[0], img.shape[1]), dtype = np.float32)
	ox = np.linspace(0, img.shape[1] - 1, img.shape[1])
	oy = np.linspace(0, img.shape[0] - 1, img.shape[0])
	basic_coord = np.array(np.meshgrid(ox, oy))

	for k in range( Ns ):

		xc = _cx[k]
		yc = _cy[k]

		lr = major[k]
		sr = minor[k]
		cr = senior[k]
		chi = _theta[k] * np.pi / 180

		set_r = np.int(np.ceil(1.2 * lr))
		la0 = np.max( [np.int(xc - set_r), 0])
		la1 = np.min( [np.int(xc + set_r + 1), img.shape[1] ] )
		lb0 = np.max( [np.int(yc - set_r), 0] ) 
		lb1 = np.min( [np.int(yc + set_r + 1), img.shape[0] ] )

		df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.cos(chi) + (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.sin(chi)
		df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.cos(chi) - (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.sin(chi)
		fr = df1**2 / lr**2 + df2**2 / sr**2
		jx = fr <= 1

		iu = np.where(jx == True)
		iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float32)
		iv[iu] = np.nan
		mask_path[lb0: lb1, la0: la1] = mask_path[lb0: lb1, la0: la1] * iv

	mask_img = mask_path * img

	return mask_img

def obj_sb_func( img, cen_x, cen_y, tag_a, tag_b, tag_phi, tag_ra, tag_dec, put_file, cut_img, pairs_obj, obj_dex):

	# around source masking
	mask_img = sub_mask_func( img, pairs_obj )

	# target obj.
	major = tag_a / 2
	minor = tag_b / 2
	senior = np.sqrt(major**2 - minor**2)

	ox = np.linspace(0, img.shape[1] - 1, img.shape[1])
	oy = np.linspace(0, img.shape[0] - 1, img.shape[0])
	basic_coord = np.array( np.meshgrid(ox, oy) )
	mask_path = np.zeros( (img.shape[0], img.shape[1]), dtype = np.int32 )

	xc = cen_x
	yc = cen_y

	lr = major
	sr = minor
	cr = senior
	chi = tag_phi * np.pi / 180

	set_r = np.int(np.ceil(1.3 * lr))

	la0 = np.max( [np.int(xc - set_r), 0])
	la1 = np.min( [np.int(xc + set_r + 1), img.shape[1] ] )
	lb0 = np.max( [np.int(yc - set_r), 0] )
	lb1 = np.min( [np.int(yc + set_r + 1), img.shape[0] ] )

	df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.cos(chi) + (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.sin(chi)
	df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.cos(chi) - (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.sin(chi)
	fr = df1**2 / lr**2 + df2**2 / sr**2

	jx = (fr <= 1)
	iu = np.where(jx == False)

	tag_patch = mask_img[lb0: lb1, la0: la1] + 0.
	tag_patch[ iu ] = np.nan

	tag_weit = np.ones( (tag_patch.shape[0], tag_patch.shape[1]),)

	id_nan = np.isnan( tag_patch )
	tag_weit[ id_nan ] = np.nan

	r_bins = np.logspace( 0, np.log10(set_r), 15)

	if xc - set_r < 0:
		xn = set_r + ( xc - set_r )
	else:
		xn = set_r

	if yc - set_r < 0:
		yn = set_r + ( yc - set_r )
	else:
		yn = set_r

	Intns, Angl_r, Intns_err, npix, nratio = light_measure_Z0_weit(tag_patch, tag_weit, pixel, xn, yn, r_bins)

	id_val = npix > 1.
	sb_arr, sb_err_arr = Intns[ id_val ], Intns_err[ id_val ]
	r_arr = Angl_r[ id_val ]

	plt.figure()
	plt.imshow( tag_patch, origin = 'lower', cmap = 'rainbow', norm = mpl.colors.LogNorm(),)
	plt.plot(xn, yn, 'k+')
	plt.savefig('/home/xkchen/%d_obj_cut_test.png' % tt )
	plt.close()

	keys = [ 'r', 'sb', 'sb_err' ]
	values = [ r_arr, sb_arr, sb_err_arr ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( put_file,)

	#. img record
	keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2', 'CENTER_X','CENTER_Y', 'CEN_RA', 'CEN_DEC', 'P_SCALE']
	value = ['T', 32, 2, tag_patch.shape[1], tag_patch.shape[0], xn, yn, tag_ra, tag_dec, pixel ]
	ff = dict(zip(keys,value))
	fil = fits.Header(ff)
	fits.writeto(cut_img, tag_patch, header = fil, overwrite = True)

	return

def sb_fit_func(p, x, y, yerr):

	I_0, re_0, I_1, re_1, frac_deV = p[:]
	mode_sb = comp_func(x, I_0, re_0, I_1, re_1, frac_deV)

	delta = mode_sb - y
	chi2 = np.sum( delta**2 / yerr**2 )

	if np.isfinite( chi2 ):
		return chi2
	return np.inf

### === ### data load
dat = pds.read_csv('/home/xkchen/mywork/ICL/data/cat_random/match_2_28/random_i-band_tot_remain_cat_set_200-grid_6.0-sigma.csv')
ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
imgx, imgy = np.array( dat['bcg_x'] ), np.array( dat['bcg_y'] )

img_path = '/home/xkchen/figs/i_mag_test/img_cat/'
cat_path = '/home/xkchen/figs/i_mag_test/source_cat/'

band_str = 'i'

"""
#. all sources
for ii in range( 0,1 ):

	ra_g, dec_g, z_g = ra[ii], dec[ii], z[ii]

	#. img
	img_file = img_path + 'rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band_str, ra_g, dec_g, z_g)
	data = fits.open( img_file )
	img = data[0].data
	Head = data[0].header
	wcs_lis = awc.WCS( Head )

	#. detected sources
	gal_file = img_path + 'random_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
	gal_list = asc.read( gal_file % (band_str, ra_g, dec_g, z_g), )
	Numb = np.array( gal_list['NUMBER'] )[-1]
	A = np.array( gal_list['A_IMAGE'])
	B = np.array( gal_list['B_IMAGE'])
	theta = np.array( gal_list['THETA_IMAGE'])
	cx = np.array( gal_list['X_IMAGE'])
	cy = np.array( gal_list['Y_IMAGE'])
	p_type = np.array( gal_list['CLASS_STAR'])

	Kron = 16
	a = Kron * A
	b = Kron * B

	s_ra, s_dec = wcs_lis.all_pix2world( cx, cy, 0)

	#.source information
	keys = [ 'cx', 'cy', 'A', 'B', 'phi', 'ra', 'dec']
	values = [ cx, cy, A, B, theta, s_ra, s_dec ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv(img_path + 'rand_img-%s_ra%.3f_dec%.3f_z%.3f_objs_list.csv' % (band_str, ra_g, dec_g, z_g),)

	print( ra_g, dec_g, z_g)
"""

"""
#. stars rule out
for ii in ( 0, 1, 2):

	ra_g, dec_g, z_g = ra[ii], dec[ii], z[ii]

	#. img
	img_file = img_path + 'rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band_str, ra_g, dec_g, z_g)
	data = fits.open( img_file )
	img = data[0].data
	Head = data[0].header
	wcs_lis = awc.WCS( Head )

	#. detected sources
	gal_file = img_path + 'random_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
	gal_list = asc.read( gal_file % (band_str, ra_g, dec_g, z_g), )
	Numb = np.array( gal_list['NUMBER'] )[-1]
	A = np.array( gal_list['A_IMAGE'])
	B = np.array( gal_list['B_IMAGE'])
	theta = np.array( gal_list['THETA_IMAGE'])
	cx = np.array( gal_list['X_IMAGE'])
	cy = np.array( gal_list['Y_IMAGE'])
	p_type = np.array( gal_list['CLASS_STAR'])

	Kron = 16
	a = Kron * A
	b = Kron * B
	s_ra, s_dec = wcs_lis.all_pix2world( cx, cy, 0)

	#. stars
	star_file = img_path + 'source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g)
	cm_x0, cm_y0, cm_A0, cm_B0, cm_chi0, cm_ra0, cm_dec0 = star_pos_func( star_file, Head )

	keys = [ 'cx', 'cy', 'A', 'B', 'phi', 'ra', 'dec' ]
	values = [ cm_x0, cm_y0, cm_A0 / 3, cm_B0 / 3, cm_chi0, cm_ra0, cm_dec0 ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv(img_path + 'rand_img-%s_ra%.3f_dec%.3f_z%.3f_stars_list.csv' % (band_str, ra_g, dec_g, z_g),)

	#. source without stars
	tag_dex = []
	for tt in range( Numb ):
		dL = np.sqrt( (cx[tt] - cm_x0)**2 + (cy[tt] - cm_y0)**2 )
		id_s = dL == dL.min()

		_com_R = cm_A0[ id_s ]
		_dl_ = dL[ id_s ] < _com_R

		if np.sum(_dl_) > 0:
			tag_dex.append( tt )

		else:
			continue

	remov_dex = tuple( tag_dex )
	order_arr = np.arange(0, Numb, 1)
	remin_dex = np.delete( order_arr, remov_dex )

	dd_cx, dd_cy = cx[ remin_dex ], cy[ remin_dex ]
	dd_ra, dd_dec = s_ra[ remin_dex ], s_dec[ remin_dex ]
	dd_a, dd_b, dd_chi = a[ remin_dex ], b[ remin_dex ], theta[ remin_dex ]
	dd_A, dd_B = A[ remin_dex ], B[ remin_dex ]

	keys = [ 'cx', 'cy', 'A', 'B', 'phi', 'ra', 'dec' ]
	values = [ dd_cx, dd_cy, dd_A, dd_B, dd_chi, dd_ra, dd_dec ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv(img_path + 'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_list.csv' % (band_str, ra_g, dec_g, z_g),)

	# fig = plt.figure( figsize = (13.12, 4.8) )
	# ax0 = fig.add_axes( [0.05, 0.10, 0.40, 0.80] )
	# ax1 = fig.add_axes( [0.55, 0.10, 0.40, 0.80] )

	# ax0.set_title('ra%.3f dec%.3f z%.3f' % (ra_g, dec_g, z_g) )
	# ax0.imshow( img, origin = 'lower', cmap = 'Greys', vmin = -5e-3, vmax = 3e-2, alpha = 0.75)

	# Ng = len( dd_cx )
	# for ll in range( Ng ):
	# 	ellips = Ellipse( xy = (dd_cx[ll], dd_cy[ll]), width = dd_a[ll], height = dd_b[ll], angle = dd_chi[ll], fill = False, 
	# 		ec = 'm', ls = '-', linewidth = 0.75, )
	# 	ax0.add_patch( ellips )

	# Ns0 = len( cm_A0 )
	# for ll in range( Ns0 ):
	# 	ellips = Ellipse( xy = (cm_x0[ll], cm_y0[ll]), width = 3 * cm_A0[ll], height = 3 * cm_B0[ll], angle = cm_chi0[ll], fill = False, 
	# 		ec = 'c', ls = '-', linewidth = 0.75,)
	# 	ax0.add_patch( ellips )

	# ax1.imshow( img, origin = 'lower', cmap = 'Greys', vmin = -5e-3, vmax = 3e-2, alpha = 0.75)

	# Ns0 = len( cm_A0 )
	# for ll in range( Ns0 ):
	# 	ellips = Ellipse( xy = (cm_x0[ll], cm_y0[ll]), width = 3 * cm_A0[ll], height = 3 * cm_B0[ll], angle = cm_chi0[ll], fill = False, 
	# 		ec = 'c', ls = '-', linewidth = 0.75,)
	# 	ax1.add_patch( ellips )

	# for ll in range( Numb ):
	# 	ellips = Ellipse( xy = (cx[ll], cy[ll]), width = a[ll], height = b[ll], angle = theta[ll], fill = False, 
	# 		ec = 'r', ls = '-', linewidth = 0.75, )
	# 	ax1.add_patch( ellips )

	# plt.savefig('/home/xkchen/source_divid_test_%d.png' % ii, dpi = 300)
	# plt.close()
"""

"""
#. SB profiles of sources without stars
for ii in range( 0,1 ):

	ra_g, dec_g, z_g = ra[ii], dec[ii], z[ii]

	#. img
	img_file = img_path + 'rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band_str, ra_g, dec_g, z_g)
	data = fits.open( img_file )
	img_load = data[0].data
	Head = data[0].header
	wcs_lis = awc.WCS( Head )

	#. stars
	s_dat = pds.read_csv(img_path + 'rand_img-%s_ra%.3f_dec%.3f_z%.3f_stars_list.csv' % (band_str, ra_g, dec_g, z_g),)
	cm_cx, cm_cy = np.array( s_dat['cx']), np.array( s_dat['cy'])
	cm_A, cm_B, cm_theta = np.array( s_dat['A']), np.array( s_dat['B']), np.array( s_dat['phi'])

	#. no-stars sources
	pdat = pds.read_csv(img_path + 'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_list.csv' % (band_str, ra_g, dec_g, z_g),)
	cx, cy, A, B, theta = np.array(pdat['cx']), np.array(pdat['cy']), np.array(pdat['A']), np.array(pdat['B']), np.array(pdat['phi'])
	s_ra, s_dec = np.array(pdat['ra']), np.array(pdat['dec'])

	Kron = 18
	a = Kron * A
	b = Kron * B

	Ns = len( s_ra )

	#. source profile
	# for tt in range( Ns ):
	for tt in range( 489,490 ):

		tag_a, tag_b = a[ tt ], b[ tt ]
		tag_phi = theta[ tt ]
		cen_x, cen_y = cx[ tt ], cy[ tt ]
		tag_ra, tag_dec = s_ra[ tt ], s_dec[ tt ]

		#.around galaxies
		dl_objs = np.sqrt( (cen_x - cx)**2 + (cen_y - cy)**2 )
		arg_dex = np.argsort( dl_objs )

		cp_cx = cx[arg_dex][1:6]
		cp_cy = cy[arg_dex][1:6]
		cp_a, cp_b = A[arg_dex][1:6] * 6, B[arg_dex][1:6] * 6
		cp_phi = theta[arg_dex][1:6]

		#.around stars
		dl_stars = np.sqrt( (cm_cx - cen_x)**2 + (cm_cy - cen_y)**2 )
		arg_dex_s = np.argsort( dl_stars )

		s_cp_cx = cm_cx[ arg_dex_s][:6]
		s_cp_cy = cm_cy[ arg_dex_s][:6]
		s_cp_a, s_cp_b = cm_A[ arg_dex_s][:6] * 3, cm_B[ arg_dex_s][:6] * 3
		s_cp_phi = cm_theta[ arg_dex_s][:6]

		near_cx = np.r_[ cp_cx, s_cp_cx ]
		near_cy = np.r_[ cp_cy, s_cp_cy ]
		near_a = np.r_[ cp_a, s_cp_b ]
		near_b = np.r_[ cp_b, s_cp_b ]
		near_phi = np.r_[ cp_phi, s_cp_phi ]

		near_arr = [ near_cx, near_cy, near_a, near_b, near_phi ]

		put_file = cat_path + 'frame-ra%.3f_dec%.3f_obj_ra%.3f_dec%.3f_sb.csv' % (ra_g, dec_g, tag_ra, tag_dec)
		cut_img = cat_path + 'frame-ra%.3f_dec%.3f_obj_ra%.3f_dec%.3f_img.fits' % (ra_g, dec_g, tag_ra, tag_dec)

		obj_sb_func( img_load, cen_x, cen_y, tag_a, tag_b, tag_phi, tag_ra, tag_dec, put_file, cut_img, near_arr, tt)
"""


## SB profile fitting
for ii in range( 0,1 ):

	ra_g, dec_g, z_g = ra[ii], dec[ii], z[ii]

	#. img
	img_file = img_path + 'rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band_str, ra_g, dec_g, z_g)
	data = fits.open( img_file )
	img_arr = data[0].data
	Head = data[0].header
	wcs_lis = awc.WCS( Head )

	#. no-star cat
	pdat = pds.read_csv(img_path + 'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_list.csv' % (band_str, ra_g, dec_g, z_g),)

	cx, cy, A, B, phi = np.array(pdat['cx']), np.array(pdat['cy']), np.array(pdat['A']), np.array(pdat['B']), np.array(pdat['phi'])
	s_ra, s_dec = np.array(pdat['ra']), np.array(pdat['dec'])

	Ns = len(s_ra)
	OBJ_mag = np.zeros( Ns,)
	obj_re1, obj_re2 = np.zeros( Ns,), np.zeros( Ns,)
	obj_I1, obj_I2 = np.zeros( Ns,), np.zeros( Ns,)

	#..... SDSS compare
	sdss_compare_0 = pds.read_csv( img_path + 
		'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_re-scaled-cmag_SDSS_compare.csv' % (band_str, ra_g, dec_g, z_g),)
	delt_mag = np.array( sdss_compare_0['delt_Mag'] )
	match_dex = np.array( sdss_compare_0['order'] )
	mag_sdss = np.array( sdss_compare_0['mag_sdss'] )
	id_devi = delt_mag <= -1
	test_dex_0 = match_dex[ id_devi ] # including test_dex_1

	sdss_compare_1 = pds.read_csv( img_path + 
		'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_3-R-Kron_re-scaled-cmag_SDSS_compare.csv' % (band_str, ra_g, dec_g, z_g),)
	delt_mag = np.array( sdss_compare_1['delt_Mag'] )
	match_dex = np.array( sdss_compare_1['order'] )
	mag_sdss = np.array( sdss_compare_1['mag_sdss'] )
	id_devi = delt_mag <= -1
	test_dex_1 = match_dex[ id_devi ]

	diff_list = list( set( test_dex_0 ).difference( set( test_dex_1 ) ) )
	diff_dex = np.array( diff_list )

	#.too brighter
	To_bri_dex = np.loadtxt( img_path + 
		'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_too-brighter_adjust.txt' % (band_str, ra_g, dec_g, z_g),)
	To_bri_dex = To_bri_dex.astype( np.int32 )


	_tmp_dex = np.array([4, 6, 10, 12, 29, 46, 51, 82, 86, 111, 124, 131, 142, 147, 156, 162, 187, 
						189, 220, 231, 232, 243, 248, 256, 271, 279, 283, 302, 312, 333, 353, 355, 
						366, 370, 386, 388, 391, 407, 408, 409, 417, 424, 435, 439, 443, 445, 447, 
						448, 454, 462, 473, 476, 481, 483] )
	# for tt in range( Ns ):
	for jj in range( _tmp_dex.shape[0] ): # SDSS compare test
		tt = _tmp_dex[ jj ]

		#. obj SB profile
		tag_ra, tag_dec = s_ra[ tt ], s_dec[ tt ]
		put_file = cat_path + 'frame-ra%.3f_dec%.3f_obj_ra%.3f_dec%.3f_sb.csv' % (ra_g, dec_g, tag_ra, tag_dec)
		sb_dat = pds.read_csv( put_file )
		tt_r, tt_sb, tt_err = np.array(sb_dat['r']), np.array(sb_dat['sb']), np.array(sb_dat['sb_err'])

		#. obj img
		cut_file = cat_path + 'frame-ra%.3f_dec%.3f_obj_ra%.3f_dec%.3f_img.fits' % (ra_g, dec_g, tag_ra, tag_dec)
		cut_data = fits.open( cut_file )
		cut_img = cut_data[0].data
		px, py = cut_data[0].header['CENTER_X'], cut_data[0].header['CENTER_Y']

		#.. fitting
		if tt in test_dex_1 or tt in To_bri_dex:
			tot_F = np.nansum( cut_img )
			I0_fit, re0_fit, I1_fit, re1_fit, f_dev_fit = [ np.nan ] * 5

		else:
			if tt in diff_dex:
				id_point = tt_sb > 0
				id_err = tt_err != 0

				id_r_up = tt_r <= (3 * A[tt] * pixel)
				id_lim = id_point & id_err & id_r_up

				fit_r, fit_sb, fit_err = tt_r[ id_lim ], tt_sb[ id_lim ], tt_err[ id_lim ]

			else:
				id_point = tt_sb > 0
				id_err = tt_err != 0
				id_lim = id_point & id_err

				fit_r, fit_sb, fit_err = tt_r[ id_lim ], tt_sb[ id_lim ], tt_err[ id_lim ]

			po = [ 0.5, 0.5, 0.1, 1.5, 0.65]
			bounds = [ [0, np.inf], [0, 50], [0, np.inf], [0, 200], [0, 1] ]
			E_return = optimize.minimize( sb_fit_func, x0 = np.array( po ), args = ( fit_r, fit_sb, fit_err), 
				method = 'L-BFGS-B', bounds = bounds,)
			popt = E_return.x

			I0_fit, re0_fit, I1_fit, re1_fit, f_dev_fit = popt
			fit_sb = comp_func( tt_r, I0_fit, re0_fit, I1_fit, re1_fit, f_dev_fit )
			fit_l0 = deV_func( tt_r, I0_fit, re0_fit ) * f_dev_fit
			fit_l1 = exp_func( tt_r, I1_fit, re1_fit ) * ( 1 - f_dev_fit )

			#.. magnitude (R_e scaled region fitting)
			new_r0 = np.logspace(np.log10( re0_fit / 100), np.log10( re0_fit * 7), 50)
			new_r1 = np.logspace(np.log10( re1_fit / 100), np.log10( re1_fit * 3), 50)
			extra_sb_F0 = deV_func( new_r0, I0_fit, re0_fit ) * f_dev_fit
			extra_sb_F1 = exp_func( new_r1, I1_fit, re1_fit ) * ( 1 - f_dev_fit )
			cumu_F_0 = cumu_mass_func( new_r0, extra_sb_F0 )
			cumu_F_1 = cumu_mass_func( new_r1, extra_sb_F1 )

			sub_F_arr = np.array( [ cumu_F_1[-1], cumu_F_0[-1] ] )
			tot_F = np.nansum( sub_F_arr )

		#.. magnitude (mask aperture limited region fitting)
		# po = [ 0.5, 0.5, 0.1, 1.5, 0.65]
		# bounds = [ [0, np.inf], [0, 50], [0, np.inf], [0, 200], [0, 1] ]
		# E_return = optimize.minimize( sb_fit_func, x0 = np.array( po ), args = ( fit_r, fit_sb, fit_err), 
		# 	method = 'L-BFGS-B', bounds = bounds,)
		# popt = E_return.x

		# I0_fit, re0_fit, I1_fit, re1_fit, f_dev_fit = popt
		# fit_sb = comp_func( tt_r, I0_fit, re0_fit, I1_fit, re1_fit, f_dev_fit )
		# fit_l0 = deV_func( tt_r, I0_fit, re0_fit ) * f_dev_fit
		# fit_l1 = exp_func( tt_r, I1_fit, re1_fit ) * ( 1 - f_dev_fit )

		# new_r = np.logspace( np.log10(np.min(fit_r) / 10), np.log10( 8 * A[tt] * pixel ), 50)
		# extra_sb_F0 = deV_func( new_r, I0_fit, re0_fit ) * f_dev_fit
		# extra_sb_F1 = exp_func( new_r, I1_fit, re1_fit ) * ( 1 - f_dev_fit )
		# cumu_F_0 = cumu_mass_func( new_r, extra_sb_F0 )
		# cumu_F_1 = cumu_mass_func( new_r, extra_sb_F1 )
		# sub_F_arr = np.array( [ cumu_F_1[-1], cumu_F_0[-1] ] )

		cMag = 22.5 - 2.5 * np.log10( tot_F )
		OBJ_mag[ tt ] = cMag

		obj_re1[tt], obj_re2[tt] = re0_fit, re1_fit
		obj_I1[tt], obj_I2[tt] = I0_fit, I1_fit

		print( cMag )

		fig = plt.figure( figsize = (13.12, 4.8) )
		ax0 = fig.add_axes( [0.10, 0.10, 0.40, 0.80] )
		ax1 = fig.add_axes( [0.55, 0.10, 0.40, 0.80] )

		ax0.plot( tt_r, tt_sb, 'r-*', label = '$ obj_{obs} $',)
		ax0.fill_between( tt_r, y1 = tt_sb - tt_err, y2 = tt_sb + tt_err, color = 'r', alpha = 0.15,)

		ax0.plot( tt_r, fit_sb, 'b--', label = 'fit')
		ax0.plot( tt_r, fit_l0, 'b:', )
		ax0.plot( tt_r, fit_l1, 'b-.', )

		ax0.legend( loc = 3,)
		ax0.set_xscale('log')
		ax0.set_yscale('log')
		ax0.set_xlabel('R [arcsec]')
		ax0.set_ylabel('$ SB [nanomaggies / arcsec^2] $')

		# ax1.set_title('$ mag - mag_{SDSS} = %.2f $' % delt_mag[id_devi][jj] )
		# ax1.set_title('$mag - mag_{SDSS} = %.2f$' % (cMag - mag_sdss[id_devi][jj] ),)

		ax1.imshow( cut_img, origin = 'lower', cmap = 'Greys', norm = mpl.colors.LogNorm(),)
		ax1.plot( px, py, 'c+', markersize = 5,)

		##... too brighter
		# ax1.imshow( img_arr, origin = 'lower', cmap = 'Greys', norm = mpl.colors.LogNorm(),)
		# ax1.plot( cx[tt], cy[tt], 'r+', markersize = 5,)
		# ax1.set_xlim( cx[tt] - 50, cx[tt] + 50 )
		# ax1.set_ylim( cy[tt] - 50, cy[tt] + 50 )
		plt.savefig('/home/xkchen/fit_fig/obj_sb_compare_%d.png' % tt, dpi = 300)
		plt.close()

	raise
	order = np.arange(0, Ns, 1)
	keys = [ 'cx', 'cy', 'A', 'B', 'phi', 'ra', 'dec', 'cMag', 'order', 'obj_I1', 'obj_re1', 'obj_I2', 'obj_re2']
	values = [ cx, cy, A, B, phi, s_ra, s_dec, OBJ_mag, order, obj_I1, obj_re1, obj_I2, obj_re2 ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )

	# out_data.to_csv( img_path + 'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_re-scaled-cmag.csv' % (band_str, ra_g, dec_g, z_g),)
	# out_data.to_csv( img_path + 'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_3-R-Kron_re-scaled-cmag.csv' % (band_str, ra_g, dec_g, z_g),)
	out_data.to_csv( img_path + 'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_3-R-Kron_pix-sum_re-scaled-cmag.csv' % (band_str, ra_g, dec_g, z_g),)

	# out_data.to_csv( img_path + 'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_cmag.csv' % (band_str, ra_g, dec_g, z_g),)

raise

for ii in range( 0,1 ):

	ra_g, dec_g, z_g = ra[ii], dec[ii], z[ii]

	img_file = img_path + 'rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band_str, ra_g, dec_g, z_g)
	data = fits.open( img_file )
	img = data[0].data

	# c_dat = pds.read_csv( img_path + 'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_re-scaled-cmag.csv' % (band_str, ra_g, dec_g, z_g),)
	# c_dat = pds.read_csv( img_path + 'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_3-R-Kron_re-scaled-cmag.csv' % (band_str, ra_g, dec_g, z_g),)
	c_dat = pds.read_csv( img_path + 'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_3-R-Kron_pix-sum_re-scaled-cmag.csv' % (band_str, ra_g, dec_g, z_g),)

	# c_dat = pds.read_csv( img_path + 'rand_img-%s_ra%.3f_dec%.3f_z%.3f_no-stars-objs_cmag.csv' % (band_str, ra_g, dec_g, z_g),)

	obj_mag = np.array( c_dat['cMag'] )
	cx, cy = np.array( c_dat['cx'] ), np.array( c_dat['cy'] )
	A, B, phi = np.array( c_dat['A'] ), np.array( c_dat['B'] ), np.array( c_dat['phi'] )
	s_ra, s_dec = np.array( c_dat['ra'] ), np.array( c_dat['dec'] )

	order_lis = np.array( c_dat['order'] )

	id_inf = np.isinf( obj_mag )

	plt.figure()
	plt.hist( obj_mag[ id_inf == False], bins = 35, density = True, histtype = 'step', color = 'r')
	plt.axvline( x = np.nanmedian(obj_mag), ls = '--', color = 'r', label = 'median(%.3f)' % np.nanmedian(obj_mag),)
	plt.legend( loc = 1)
	plt.xlim( 10, 30)
	plt.savefig('/home/xkchen/img-%s-band_ra%.3f_dec%.3f_z%.3f_obj-mag_hist.png' % (band_str, ra_g, dec_g, z_g), dpi = 300)
	plt.close()

	idnn = np.isnan( obj_mag )
	id_faint = obj_mag > 21

	plt.figure()
	ax = plt.subplot(111)
	ax.imshow( img, origin = 'lower', cmap = 'Greys', norm = mpl.colors.LogNorm(),)
	ax.scatter( cx[idnn], cy[idnn], marker = 'o', s = 10, facecolors = 'none', edgecolors = 'r')
	ax.scatter( cx[id_faint], cy[id_faint], marker = 's', s = 20, facecolors = 'none', edgecolors = 'b')
	plt.savefig('/home/xkchen/obj_mag_check.png', dpi = 300)
	plt.close()

	idm1 = obj_mag <= 16.864 # SDSS mag low limit in this frame

	plt.figure()
	ax = plt.subplot(111)
	ax.imshow( img, origin = 'lower', cmap = 'Greys', 
		norm = mpl.colors.SymLogNorm( linthresh = 0.01, linscale = 0.001, vmin = -3e-2, vmax = 3e-2, base = 10),)
	ax.scatter( cx[idm1], cy[idm1], marker = 's', s = 20, facecolors = 'none', edgecolors = 'b')
	plt.savefig('/home/xkchen/obj_mag_compare.png', dpi = 300)
	plt.close()

