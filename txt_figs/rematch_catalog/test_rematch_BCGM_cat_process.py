import time
import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits
import subprocess as subpro

import astropy.units as U
import astropy.constants as C
import astropy.wcs as awc
import astropy.io.ascii as asc

from io import StringIO
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord

#.
from fig_out_module import star_pos_func, tractor_peak_pos
from fig_out_module import zref_BCG_pos_func
from img_pre_selection import WCS_to_pixel_func

from img_mask_adjust import adjust_mask_func
from img_resample import resamp_func

#.
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


### === ### offset correction
load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'
out_path = '/home/xkchen/data/SDSS/photo_files/pos_offset_correct_imgs/'

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

"""
for mm in range( 2 ):

	band_str = band[ rank ]

	dat = pds.read_csv( load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_BCG_cat.csv' % (cat_lis[mm], band_str),)
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

	Ns = len( z )

	off_pk_x = np.zeros( Ns )
	off_pk_y = np.zeros( Ns )

	### offset to peak position record
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
		cm_x0, cm_y0, cm_A0, cm_B0, cm_chi0 = star_pos_func( star_file, Head_info, pixel )[:5]

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

	### BCG position with offset adjust
	#. save the BCG position with offset correction
	keys = [ 'ra', 'dec', 'z', 'bcg_x', 'bcg_y' ]
	values = [ ra, dec, z, off_pk_x, off_pk_y ]
	fill = dict( zip(keys, values) )
	data = pds.DataFrame( fill )
	data.to_csv( load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_pk-offset_cat.csv' % (cat_lis[mm], band_str),)

	#. BCG position at z_ref
	cat_file = load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_pk-offset_cat.csv' % (cat_lis[mm], band_str)
	out_file = load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_pk-offset_cat_z-ref.csv' % (cat_lis[mm], band_str)
	zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)

	print( '%s band done !' % band_str )

"""

### === ### masking and resampling
for kk in range( 3 ):

	band_str = band[ kk ]

	for mm in range( 2 ):

		dat = pds.read_csv( load + 'Extend_Mbcg_cat/%s_%s-band_pre-diffi_BCG_cat.csv' % (cat_lis[mm], band_str),)
		ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

		sub_coord = SkyCoord( ra * U.deg, dec * U.deg )

		#. match the position of BCGs
		ref_dat = pds.read_csv( load + 
								'Extend_Mbcg_cat/%s_%s-band_photo-z-match_pk-offset_cat.csv' % (cat_lis[mm], band_str),)
		ref_ra, ref_dec = np.array( ref_dat['ra'] ), np.array( ref_dat['dec'] )
		ref_bcgx, ref_bcgy = np.array( ref_dat['bcg_x'] ), np.array( ref_dat['bcg_y'] )

		ref_coord = SkyCoord( ref_ra * U.deg, ref_dec * U.deg )

		idx, sep, d3d = sub_coord.match_to_catalog_sky( ref_coord )
		id_lim = sep.value < 2.7e-4

		clus_x, clus_y = ref_bcgx[ idx[ id_lim ] ], ref_bcgy[ idx[ id_lim ] ]   ###. position have applied offset correction

		zN = len( ra )
		print( zN )
		print( clus_x.shape )

		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		set_z, set_ra, set_dec = z[N_sub0 : N_sub1], ra[N_sub0 : N_sub1], dec[N_sub0 : N_sub1]
		set_imgx, set_imgy = clus_x[N_sub0 : N_sub1], clus_y[N_sub0 : N_sub1]


		##.. masking (exclude BCGs)
		d_file = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
		cat_file = home + 'photo_files/star_cats/source_SQL_Z%.3f_ra%.3f_dec%.3f.csv'
		offset_file = home + 'photo_files/pos_offset_correct_imgs/offset/%s-band_ra%.3f_dec%.3f_z%.3f_star-pos-offset.csv'

		gal_file = home + 'photo_files/detect_source_cat/photo-z_img_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
		bcg_photo_file = home + 'photo_files/BCG_photometry/BCG_photo_Z%.3f_ra%.3f_dec%.3f.txt'

		out_file = home + 'photo_files/pos_offset_correct_imgs/mask_img/photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

		bcg_mask = 0

		if band_str == 'r':
			extra_cat = [ home + 'photo_files/detect_source_cat/photo-z_img_g-band_mask_ra%.3f_dec%.3f_z%.3f.cat', 
						  home + 'photo_files/detect_source_cat/photo-z_img_i-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

			extra_img = [ home + 'photo_data/frame-g-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
						  home + 'photo_data/frame-i-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

		if band_str == 'g':
			extra_cat = [ home + 'photo_files/detect_source_cat/photo-z_img_r-band_mask_ra%.3f_dec%.3f_z%.3f.cat', 
						  home + 'photo_files/detect_source_cat/photo-z_img_i-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

			extra_img = [ home + 'photo_data/frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
						  home + 'photo_data/frame-i-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

		if band_str == 'i':
			extra_cat = [ home + 'photo_files/detect_source_cat/photo-z_img_r-band_mask_ra%.3f_dec%.3f_z%.3f.cat',
						  home + 'photo_files/detect_source_cat/photo-z_img_g-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

			extra_img = [ home + 'photo_data/frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
						  home + 'photo_data/frame-g-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

		adjust_mask_func( d_file, cat_file, set_z, set_ra, set_dec, band_str, gal_file, out_file, bcg_mask,
			offset_file = offset_file, bcg_photo_file = bcg_photo_file, extra_cat = extra_cat, extra_img = extra_img,)

		print( '%d, %s band, masking done !' % (mm, band_str),)


		##.. pixel resample
		mask_file = home + 'photo_files/pos_offset_correct_imgs/mask_img/photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
		resamp_file = home + 'photo_files/pos_offset_correct_imgs/resamp_img/photo-z_resamp_%s_ra%.3f_dec%.3f_z%.3f.fits'

		resamp_func( mask_file, set_z, set_ra, set_dec, set_imgx, set_imgy, band_str, resamp_file, z_ref,
			stack_info = None, pixel = 0.396, id_dimm = True,)

		print( '%d, %s band, resample done !' % (mm, band_str),)


