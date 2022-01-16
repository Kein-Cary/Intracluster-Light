import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc
import scipy.interpolate as interp
from astropy import cosmology as apcy

from img_resample import resamp_func
from fig_out_module import zref_BCG_pos_func
from img_mask import source_detect_func, mask_func

from img_pre_selection import cat_match_func
from img_pre_selection import extra_match_func, gri_common_cat_func
from img_pre_selection import WCS_to_pixel_func

from img_mask_tmp import adjust_mask_func
# from img_mask_adjust import adjust_mask_func

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

###.cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']

### === ###
def cc_star_pos_func( star_cat, Head_info):

	wcs_lis = awc.WCS( Head_info )

	## stars catalog
	p_cat = pds.read_csv( star_cat, skiprows = 1)
	set_ra = np.array( p_cat['ra'])
	set_dec = np.array( p_cat['dec'])
	set_mag = np.array( p_cat['r'])
	OBJ = np.array( p_cat['type'])
	xt = p_cat['Column1']
	flags = [str(qq) for qq in xt]

	# x, y = wcs_lis.all_world2pix( set_ra * U.deg, set_dec * U.deg, 1, ra_dec_order = True,) ## previous
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

	sub_A0 = lr_iso[ic] * 15
	sub_B0 = sr_iso[ic] * 15
	sub_chi0 = set_chi[ic]

	sub_ra0, sub_dec0 = set_ra[ic], set_dec[ic]

	# saturated source(may not stars)
	xa = ['SATURATED' in qq for qq in flags]
	xv = np.array(xa)
	idx = xv == True
	ipx = (idx)

	sub_x2 = x[ipx]
	sub_y2 = y[ipx]

	sub_A2 = lr_iso[ipx] * 5
	sub_B2 = sr_iso[ipx] * 5
	sub_chi2 = set_chi[ipx]

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

	# cm_x0, cm_y0 = m_x0 + 0, m_y0 + 0
	# cm_A0, cm_B0, cm_chi0 = m_A0 + 0, m_B0 + 0, m_chi0 + 0

	cm_x0, cm_y0 = sub_x0 + 0, sub_y0 + 0
	cm_A0, cm_B0, cm_chi0 = sub_A0 + 0, sub_B0 + 0, sub_chi0 + 0

	# for jj in range( len( d_x0 ) - 1 ):

	# 	fill_s = np.ones( len(m_x0), )

	# 	for pp in range( jj + 1, len( d_x0 ) ):

	# 		l_dx = np.abs( cm_x0[jj] - m_x0[pp] )
	# 		l_dy = np.abs( cm_y0[jj] - m_y0[pp] )

	# 		if ( l_dx < 1) & ( l_dy < 1):

	# 			fill_s[pp] = 0

	# 		else:
	# 			fill_s[pp] = 1

	# 	id_nul = np.where( fill_s < 1 )[0]

	# 	cm_x0[id_nul] = np.nan
	# 	cm_y0[id_nul] = np.nan
	# 	cm_A0[id_nul] = np.nan
	# 	cm_B0[id_nul] = np.nan
	# 	cm_chi0[id_nul] = np.nan

	return cm_x0, cm_y0, cm_A0, cm_B0, cm_chi0, sub_ra0, sub_dec0

def star_pos_func( star_cat, Head_info):

	## stars catalog
	p_cat = pds.read_csv( star_cat, skiprows = 1)
	set_ra = np.array( p_cat['ra'])
	set_dec = np.array( p_cat['dec'])
	set_mag = np.array( p_cat['r'])
	OBJ = np.array( p_cat['type'])
	xt = p_cat['Column1']
	flags = [str(qq) for qq in xt]

	x, y = WCS_to_pixel_func( set_ra, set_dec, Head_info ) ## SDSS EDR paper relation

	set_A = np.array( [ p_cat['psffwhm_r'] , p_cat['psffwhm_g'], p_cat['psffwhm_i']]) / pixel
	set_B = np.array( [ p_cat['psffwhm_r'] , p_cat['psffwhm_g'], p_cat['psffwhm_i']]) / pixel
	set_chi = np.zeros(set_A.shape[1], dtype = np.float32)

	lln = np.array([len(set_A[:,ll][ set_A[:,ll] > 0 ]) for ll in range(set_A.shape[1]) ])
	lr_iso = np.array([np.max(set_A[:,ll]) for ll in range(set_A.shape[1]) ])
	sr_iso = np.array([np.max(set_B[:,ll]) for ll in range(set_B.shape[1]) ])

	# normal stars
	iq = lln >= 2 ## at lest observed in 2 band
	ig = OBJ == 6
	ie = (set_mag <= 20)

	ic = (ie & ig & iq)
	sub_x0 = x[ic]
	sub_y0 = y[ic]

	sub_A0 = lr_iso[ic] * 15
	sub_B0 = sr_iso[ic] * 15
	sub_chi0 = set_chi[ic]

	sub_ra0, sub_dec0 = set_ra[ic], set_dec[ic]

	# saturated source(may not stars)
	xa = ['SATURATED' in qq for qq in flags]
	xv = np.array(xa)
	idx = xv == True
	ipx = (idx)

	sub_x2 = x[ipx]
	sub_y2 = y[ipx]

	sub_A2 = lr_iso[ipx] * 5
	sub_B2 = sr_iso[ipx] * 5
	sub_chi2 = set_chi[ipx]

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

	# cm_x0, cm_y0 = m_x0 + 0, m_y0 + 0
	# cm_A0, cm_B0, cm_chi0 = m_A0 + 0, m_B0 + 0, m_chi0 + 0

	cm_x0, cm_y0 = sub_x0 + 0, sub_y0 + 0
	cm_A0, cm_B0, cm_chi0 = sub_A0 + 0, sub_B0 + 0, sub_chi0 + 0

	# for jj in range( len( d_x0 ) - 1 ):

	# 	fill_s = np.ones( len(m_x0), )

	# 	for pp in range( jj + 1, len( d_x0 ) ):

	# 		l_dx = np.abs( cm_x0[jj] - m_x0[pp] )
	# 		l_dy = np.abs( cm_y0[jj] - m_y0[pp] )

	# 		if ( l_dx < 1) & ( l_dy < 1):

	# 			fill_s[pp] = 0

	# 		else:
	# 			fill_s[pp] = 1

	# 	id_nul = np.where( fill_s < 1 )[0]

	# 	cm_x0[id_nul] = np.nan
	# 	cm_y0[id_nul] = np.nan
	# 	cm_A0[id_nul] = np.nan
	# 	cm_B0[id_nul] = np.nan
	# 	cm_chi0[id_nul] = np.nan

	return cm_x0, cm_y0, cm_A0, cm_B0, cm_chi0, sub_ra0, sub_dec0

def tractor_peak_pos( img_file, gal_cat ):

	data = fits.open( img_file )
	img = data[0].data

	source = asc.read(gal_cat)
	Numb = np.array(source['NUMBER'][-1])
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])
	theta = np.array(source['THETA_IMAGE'])
	cx = np.array(source['X_IMAGE'])
	cy = np.array(source['Y_IMAGE'])
	p_type = np.array(source['CLASS_STAR'])

	Kron = 7
	a = Kron * A
	b = Kron * B

	major = a / 2
	minor = b / 2
	senior = np.sqrt(major**2 - minor**2)

	x_peak, y_peak = [], []

	for k in range( Numb ):

		xc = cx[k]
		yc = cy[k]

		lr = major[k]
		sr = minor[k]
		cr = senior[k]
		chi = theta[k] * np.pi / 180

		set_r = np.int(np.ceil(1.2 * lr))
		la0 = np.max( [np.int(xc - set_r), 0])
		la1 = np.min( [np.int(xc + set_r + 1), img.shape[1] ] )
		lb0 = np.max( [np.int(yc - set_r), 0] ) 
		lb1 = np.min( [np.int(yc + set_r + 1), img.shape[0] ] )

		cut_img = img[ lb0 : lb1, la0 : la1 ]
		x_p, y_p = np.where( cut_img == np.nanmax( cut_img ) )

		x_peak.append( x_p[0] + la0 )
		y_peak.append( y_p[0] + lb0 )

	x_peak, y_peak = np.array( x_peak ), np.array( y_peak )

	return cx, cy, x_peak, y_peak

### === ### 
load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'
out_path = home + 'photo_files/pos_offset_correct_imgs/'

# band_str = band[ rank ]
"""
#...fixed BCG Mstar
cat_lis = [ 'low-rich', 'hi-rich' ]

for mm in range( 2 ):

	dat = pds.read_csv( load + 'bcg_M_simi_cat/%s_%s-band_photo-z-match_BCG-pos_cat_z-ref.csv' % (cat_lis[mm], band_str),)
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	Ns = len( z )

	tmp_ra, tmp_dec, tmp_z = [], [], []

	for jj in range( Ns ):

		ra_g, dec_g, z_g = ra[jj], dec[jj], z[jj]
		img_file = out_path + 'mask_img/photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

		try:
			img_data = fits.open( img_file % (band_str, ra_g, dec_g, z_g),)
			img_arr = img_data[0].data

		except FileNotFoundError:
			tmp_ra.append( ra_g )
			tmp_dec.append( dec_g )
			tmp_z.append( z_g )

	tmp_ra, tmp_dec, tmp_z = np.array( tmp_ra ), np.array( tmp_dec ), np.array( tmp_z )

	keys = [ 'ra', 'dec', 'z' ]
	values = [ tmp_ra, tmp_dec, tmp_z ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( load + 'pkoffset_cat/%s_%s-band_no-corrected_yet.csv' % (cat_lis[mm], band_str,),)

#...
cat_lis = [ 'low-age', 'hi-age' ]

for mm in range( 2 ):

	dat = pds.read_csv( load + 'bcg_M_simi_cat/%s_%s-band_photo-z-match_BCG-pos_cat_z-ref.csv' % (cat_lis[mm], band_str),)
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	Ns = len( z )

	tmp_ra, tmp_dec, tmp_z = [], [], []

	for jj in range( Ns ):

		ra_g, dec_g, z_g = ra[jj], dec[jj], z[jj]
		img_file = out_path + 'mask_img/photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

		try:
			img_data = fits.open( img_file % (band_str, ra_g, dec_g, z_g),)
			img_arr = img_data[0].data

		except FileNotFoundError:
			tmp_ra.append( ra_g )
			tmp_dec.append( dec_g )
			tmp_z.append( z_g )

	tmp_ra, tmp_dec, tmp_z = np.array( tmp_ra ), np.array( tmp_dec ), np.array( tmp_z )

	keys = [ 'ra', 'dec', 'z' ]
	values = [ tmp_ra, tmp_dec, tmp_z ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( load + 'pkoffset_cat/%s_%s-band_no-corrected_yet.csv' % (cat_lis[mm], band_str,),)

#...fixed richness
cat_lis = [ 'younger', 'older' ]

for mm in range( 2 ):

	dat = pds.read_csv( load + 'z_formed_cat/%s_%s-band_photo-z-match_BCG-pos_cat_z-ref.csv' % (cat_lis[mm], band_str),)
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	Ns = len( z )

	tmp_ra, tmp_dec, tmp_z = [], [], []

	for jj in range( Ns ):

		ra_g, dec_g, z_g = ra[jj], dec[jj], z[jj]
		img_file = out_path + 'mask_img/photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
		
		try:
			img_data = fits.open( img_file % (band_str, ra_g, dec_g, z_g),)
			img_arr = img_data[0].data

		except FileNotFoundError:
			tmp_ra.append( ra_g )
			tmp_dec.append( dec_g )
			tmp_z.append( z_g )

	tmp_ra, tmp_dec, tmp_z = np.array( tmp_ra ), np.array( tmp_dec ), np.array( tmp_z )

	keys = [ 'ra', 'dec', 'z' ]
	values = [ tmp_ra, tmp_dec, tmp_z ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( load + 'pkoffset_cat/%s_%s-band_no-corrected_yet.csv' % (cat_lis[mm], band_str,),)

##...
cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']

for mm in range( 2 ):

	dat = pds.read_csv( load + 'photo_z_cat/%s_%s-band_photo-z-match_BCG-pos_cat_z-ref.csv' % (cat_lis[mm], band_str),)
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	Ns = len( z )

	tmp_ra, tmp_dec, tmp_z = [], [], []

	for jj in range( Ns ):

		ra_g, dec_g, z_g = ra[jj], dec[jj], z[jj]
		img_file = out_path + 'mask_img/photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

		try:
			img_data = fits.open( img_file % (band_str, ra_g, dec_g, z_g),)
			img_arr = img_data[0].data

		except FileNotFoundError:
			tmp_ra.append( ra_g )
			tmp_dec.append( dec_g )
			tmp_z.append( z_g )

	tmp_ra, tmp_dec, tmp_z = np.array( tmp_ra ), np.array( tmp_dec ), np.array( tmp_z )

	keys = [ 'ra', 'dec', 'z' ]
	values = [ tmp_ra, tmp_dec, tmp_z ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( load + 'pkoffset_cat/%s_%s-band_no-corrected_yet.csv' % (cat_lis[mm], band_str,),)
"""

### === ### position offset record

for jj in range( 4 ):

	if jj == 0:
		cat_lis = [ 'low-rich', 'hi-rich' ]

	if jj == 1:
		cat_lis = [ 'low-age', 'hi-age' ]

	if jj == 2:
		cat_lis = [ 'younger', 'older' ]

	if jj == 3:
		cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']

	for mm in range( 2 ):

		dat = pds.read_csv( load + 'pkoffset_cat/%s_%s-band_no-corrected_yet.csv' % (cat_lis[mm], band_str),)
		ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

		Ns = len( ra )

		off_pk_x = np.zeros( Ns )
		off_pk_y = np.zeros( Ns )

		for jj in range( Ns ):

			ra_g, dec_g, z_g = ra[jj], dec[jj], z[jj]

			Da_g = Test_model.angular_diameter_distance(z_g).value
			L_pix = Da_g * 10**3 * pixel / rad2asec

			##...SExTractor sources
			galx_file = home + 'photo_files/detect_source_cat/photo-z_img_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat' % (band_str, ra_g, dec_g, z_g)
			img_file = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band_str, ra_g, dec_g, z_g)

			cen_x, cen_y, peak_x, peak_y = tractor_peak_pos( img_file, galx_file )

			img_data = fits.open( img_file )
			Head_info = img_data[0].header

			bcg_x, bcg_y = WCS_to_pixel_func( ra_g, dec_g, Head_info)

			## star catalog
			star_file = home + 'photo_files/star_cats/source_SQL_Z%.3f_ra%.3f_dec%.3f.csv' % (z_g, ra_g, dec_g)
			cm_x0, cm_y0, cm_A0, cm_B0, cm_chi0 = star_pos_func( star_file, Head_info )[:5]

			Ns0 = len( cm_x0 )

			targ_order = []

			for ii in range( Ns0 ):

				ds = np.sqrt( ( cen_x - cm_x0[ii] )**2 + ( cen_y - cm_y0[ii] )**2 )
				id_x = np.where( ds == np.nanmin( ds ) )[0][0]
				targ_order.append( id_x )

			targ_order = np.array( targ_order )

			## stars in frame region
			id_limx = ( cm_x0 >= 0 ) & ( cm_x0 <= 2048 )
			id_limy = ( cm_y0 >= 0 ) & ( cm_y0 <= 1489 )
			id_lim = id_limx & id_limy

			lim_s_x0, lim_s_y0 = cm_x0[ id_lim ], cm_y0[ id_lim ]
			lim_order = targ_order[ id_lim ]

			## offset
			off_cen = np.sqrt( ( cen_x[ lim_order ] - lim_s_x0 )**2 + ( cen_y[ lim_order ] - lim_s_y0 )**2 )
			off_peak = np.sqrt( ( peak_x[ lim_order ] - lim_s_x0 )**2 + ( peak_y[ lim_order ] - lim_s_y0 )**2 )

			devi_cenx = cen_x[ lim_order ] - lim_s_x0
			devi_ceny = cen_y[ lim_order ] - lim_s_y0

			devi_pkx = peak_x[ lim_order ] - lim_s_x0
			devi_pky = peak_y[ lim_order ] - lim_s_y0

			## corrected BCG position
			medi_x2pk_off = np.median( devi_pkx )
			medi_y2pk_off = np.median( devi_pky )

			p_pk_x, p_pk_y = bcg_x + medi_x2pk_off, bcg_y + medi_y2pk_off

			off_pk_x[jj] = p_pk_x
			off_pk_y[jj] = p_pk_y

			##.. save the off set array
			keys = [ 'offset2cen', 'offset2peak', 'devi_cenx', 'devi_ceny', 'devi_pk_x', 'devi_pk_y' ]
			values = [ off_cen, off_peak, devi_cenx, devi_ceny, devi_pkx, devi_pky ]
			fill = dict( zip( keys, values ) )
			data = pds.DataFrame( fill )
			data.to_csv( out_path + 'offset/%s-band_ra%.3f_dec%.3f_z%.3f_star-pos-offset.csv' % (band_str, ra_g, dec_g, z_g),)

		keys = [ 'ra', 'dec', 'z', 'bcg_x', 'bcg_y' ]
		values = [ ra, dec, z, off_pk_x, off_pk_y ]
		fill = dict( zip(keys, values) )
		data = pds.DataFrame( fill )
		data.to_csv( load + 'pkoffset_cat/%s_%s-band_no-corrected_yet_pk-offset_cat.csv' % (cat_lis[mm], band_str),)

		print( '%s band done !' % band_str )


"""
### === 
band_str = band[ rank ]
sf_len = 5
f2str = '%.' + '%df' % sf_len

for jj in range( 4 ):

	if jj == 0:
		cat_lis = [ 'low-rich', 'hi-rich' ]
		dat_file = load + 'bcg_M_simi_cat/%s_%s-band_photo-z-match_BCG-pos_cat_z-ref.csv'

	if jj == 1:
		cat_lis = [ 'low-age', 'hi-age' ]
		dat_file = load + 'bcg_M_simi_cat/%s_%s-band_photo-z-match_BCG-pos_cat_z-ref.csv'

	if jj == 2:
		cat_lis = [ 'younger', 'older' ]
		dat_file = load + 'z_formed_cat/%s_%s-band_photo-z-match_BCG-pos_cat_z-ref.csv'

	if jj == 3:
		cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']
		dat_file = load + 'photo_z_cat/%s_%s-band_photo-z-match_BCG-pos_cat_z-ref.csv'

	for mm in range( 2 ):

		sub_ra, sub_dec, sub_z = np.array([]), np.array([]), np.array([])
		sub_imgx, sub_imgy = np.array([]), np.array([])

		dat = pds.read_csv( dat_file  % (cat_lis[mm], band_str),)
		ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
		bcg_x, bcg_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

		out_ra = [ f2str % ll for ll in ra ]
		out_dec = [ f2str % ll for ll in dec ]
		out_z = [ f2str % ll for ll in z ]

		## gri-common imgs
		ref_dat = pds.read_csv( load + 'pkoffset_cat/' + 
								'%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_cat_z-ref.csv' % (cat_lis[mm], band_str),)
		ref_ra_0, ref_dec_0, ref_z_0 = np.array(ref_dat.ra), np.array(ref_dat.dec), np.array(ref_dat.z)
		ref_imgx_0, ref_imgy_0 = np.array(ref_dat.bcg_x), np.array(ref_dat.bcg_y)

		match_ra, match_dec, match_z, match_x, match_y, order_0 = extra_match_func(
			out_ra, out_dec, out_z, ref_ra_0, ref_dec_0, ref_z_0, ref_imgx_0, ref_imgy_0)

		sub_ra = np.r_[ sub_ra, match_ra ]
		sub_dec = np.r_[ sub_dec, match_dec ]
		sub_z = np.r_[ sub_z, match_z ]
		sub_imgx = np.r_[ sub_imgx, match_x ]
		sub_imgy = np.r_[ sub_imgy, match_y ]

		## no-common imgs
		z0_file = load + 'pkoffset_cat/%s_%s-band_no-corrected_yet_pk-offset_cat.csv' % (cat_lis[mm], band_str)
		zref_file = load + 'pkoffset_cat/%s_%s-band_no-corrected_yet_pk-offset_cat_z-ref.csv' % (cat_lis[mm], band_str)
		zref_BCG_pos_func( z0_file, z_ref, zref_file, pixel)

		ref_dat = pds.read_csv( load + 'pkoffset_cat/' + 
								'%s_%s-band_no-corrected_yet_pk-offset_cat_z-ref.csv' % (cat_lis[mm], band_str),)
		ref_ra_1, ref_dec_1, ref_z_1 = np.array(ref_dat.ra), np.array(ref_dat.dec), np.array(ref_dat.z)
		ref_imgx_1, ref_imgy_1 = np.array(ref_dat.bcg_x), np.array(ref_dat.bcg_y)

		match_ra, match_dec, match_z, match_x, match_y, order_1 = extra_match_func(
			out_ra, out_dec, out_z, ref_ra_1, ref_dec_1, ref_z_1, ref_imgx_1, ref_imgy_1)

		sub_ra = np.r_[ sub_ra, match_ra ]
		sub_dec = np.r_[ sub_dec, match_dec ]
		sub_z = np.r_[ sub_z, match_z ]
		sub_imgx = np.r_[ sub_imgx, match_x ]
		sub_imgy = np.r_[ sub_imgy, match_y ]


		keys = [ 'ra', 'dec', 'z', 'bcg_x', 'bcg_y' ]
		values = [ sub_ra, sub_dec, sub_z, sub_imgx, sub_imgy ]
		fill = dict( zip(keys, values) )
		data = pds.DataFrame( fill )
		data.to_csv( load + 'pkoffset_cat/%s_%s-band_photo-z-match_pk-offset_BCG-pos_cat_z-ref.csv' % (cat_lis[mm], band_str),)

		print( '%s band done !' % band_str )
"""

### === ### masking and resampling
for jj in range( 4 ):

	if jj == 0:
		cat_lis = [ 'low-rich', 'hi-rich' ]

	if jj == 1:
		cat_lis = [ 'low-age', 'hi-age' ]

	if jj == 2:
		cat_lis = [ 'younger', 'older' ]

	if jj == 3:
		cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']

	for kk in range( 3 ):

		band_str = band[ kk ]

		for mm in range( 2 ):

			dat = pds.read_csv(load + 'pkoffset_cat/%s_%s-band_no-corrected_yet_pk-offset_cat.csv' % (cat_lis[mm], band_str),)
			ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
			clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)    ###. position have applied offset correction

			zN = len( z )
			print( zN )

			m, n = divmod(zN, cpus)
			N_sub0, N_sub1 = m * rank, (rank + 1) * m
			if rank == cpus - 1:
				N_sub1 += n

			set_z, set_ra, set_dec = z[N_sub0 : N_sub1], ra[N_sub0 : N_sub1], dec[N_sub0 : N_sub1]
			set_imgx, set_imgy = clus_x[N_sub0 : N_sub1], clus_y[N_sub0 : N_sub1]

			# #.. masking
			# d_file = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
			# cat_file = home + 'photo_files/star_cats/source_SQL_Z%.3f_ra%.3f_dec%.3f.csv'
			# offset_file = home + 'photo_files/pos_offset_correct_imgs/offset/%s-band_ra%.3f_dec%.3f_z%.3f_star-pos-offset.csv'

			# gal_file = home + 'photo_files/detect_source_cat/photo-z_img_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
			# bcg_photo_file = home + 'photo_files/BCG_photometry/BCG_photo_Z%.3f_ra%.3f_dec%.3f.txt'

			# out_file = home + 'photo_files/pos_offset_correct_imgs/mask_img/photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

			# bcg_mask = 0

			# if band_str == 'r':
			# 	extra_cat = [ home + 'photo_files/detect_source_cat/photo-z_img_g-band_mask_ra%.3f_dec%.3f_z%.3f.cat', 
			# 				  home + 'photo_files/detect_source_cat/photo-z_img_i-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

			# 	extra_img = [ home + 'photo_data/frame-g-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
			# 				  home + 'photo_data/frame-i-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

			# if band_str == 'g':
			# 	extra_cat = [ home + 'photo_files/detect_source_cat/photo-z_img_r-band_mask_ra%.3f_dec%.3f_z%.3f.cat', 
			# 				  home + 'photo_files/detect_source_cat/photo-z_img_i-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

			# 	extra_img = [ home + 'photo_data/frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
			# 				  home + 'photo_data/frame-i-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

			# if band_str == 'i':
			# 	extra_cat = [ home + 'photo_files/detect_source_cat/photo-z_img_r-band_mask_ra%.3f_dec%.3f_z%.3f.cat',
			# 				  home + 'photo_files/detect_source_cat/photo-z_img_g-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

			# 	extra_img = [ home + 'photo_data/frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
			# 				  home + 'photo_data/frame-g-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

			# adjust_mask_func( d_file, cat_file, set_z, set_ra, set_dec, band_str, gal_file, out_file, bcg_mask,
			# 	offset_file = offset_file, bcg_photo_file = bcg_photo_file, extra_cat = extra_cat, extra_img = extra_img,)

			# print( '%d, %s band, masking done !' % (mm, band_str),)

			#.. pixel resample
			mask_file = home + 'photo_files/pos_offset_correct_imgs/mask_img/photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
			resamp_file = home + 'photo_files/pos_offset_correct_imgs/resamp_img/photo-z_resamp_%s_ra%.3f_dec%.3f_z%.3f.fits'

			resamp_func( mask_file, set_z, set_ra, set_dec, set_imgx, set_imgy, band_str, resamp_file, z_ref,
				stack_info = None, pixel = 0.396, id_dimm = True,)

			print( '%d, %s band, resample done !' % (mm, band_str),)

