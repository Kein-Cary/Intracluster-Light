import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C

from astropy.coordinates import SkyCoord
from astropy import cosmology as apcy

#. dust map with the recalibration by Schlafly & Finkbeiner (2011)
import sfdmap
E_map = sfdmap.SFDMap('/home/xkchen/module/dust_map/sfddata_maskin')
from extinction_redden import A_wave

### === ### cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone( H0 = 67.74, Om0 = 0.311 )
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1. - Omega_m
Omega_k = 1. - (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

band = ['r', 'g', 'i']
L_wave = np.array( [ 6166, 4686, 7480 ] )
Mag_sun = [ 4.65, 5.11, 4.53 ]

###=====### parameter for Galactic
Rv = 3.1
pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.25

home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

"""
### === ### cluster region dust map query
#. separate samples
for kk in range( 3 ):

	###... bcg age binned sample
	hi_dat = pds.read_csv( load + 'bcg_M_simi_cat/hi-age_%s-band_photo-z-match_BCG-pos_cat.csv' % band[kk] )
	hi_age_ra, hi_age_dec, hi_age_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )

	lo_dat = pds.read_csv( load + 'bcg_M_simi_cat/low-age_%s-band_photo-z-match_BCG-pos_cat.csv' % band[kk] )
	lo_age_ra, lo_age_dec, lo_age_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )


	###... bcg mass binned sample
	hi_dat = pds.read_csv( load + 'photo_z_cat/high_BCG_star-Mass_%s-band_photo-z-match_BCG-pos_cat.csv' % band[kk] )
	hi_m_ra, hi_m_dec, hi_m_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )

	lo_dat = pds.read_csv( load + 'photo_z_cat/low_BCG_star-Mass_%s-band_photo-z-match_BCG-pos_cat.csv' % band[kk] )
	lo_m_ra, lo_m_dec, lo_m_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )


	###... richness binned sample
	hi_dat = pds.read_csv(  load + 'bcg_M_simi_cat/hi-rich_%s-band_photo-z-match_BCG-pos_cat.csv' % band[kk] )
	hi_rich_ra, hi_rich_dec, hi_rich_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )

	lo_dat = pds.read_csv(  load + 'bcg_M_simi_cat/low-rich_%s-band_photo-z-match_BCG-pos_cat.csv' % band[kk] )
	lo_rich_ra, lo_rich_dec, lo_rich_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )


	age_ra = np.r_[ lo_age_ra, hi_age_ra ]
	age_dec = np.r_[ lo_age_dec, hi_age_dec ]
	age_z = np.r_[ lo_age_z, hi_age_z ]

	pos_age = SkyCoord( age_ra, age_dec, unit = 'deg')
	p_EBV = E_map.ebv( pos_age )
	A_v = Rv * p_EBV
	A_l = A_wave( L_wave[ kk ], Rv) * A_v

	N_Low = len( lo_age_ra )
	orin_dex = np.ones( len(age_ra), dtype = int )
	
	_cc_order = np.arange( 0, len(age_ra),)
	id_vx = _cc_order >= N_Low
	orin_dex[ id_vx ] = 2

	keys = ['ra', 'dec', 'z', 'orin_dex', 'E_bv', 'A_l']
	values = [ age_ra, age_dec, age_z, orin_dex, p_EBV, A_l ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/BCG-age_bin_%s-band_dust_value.csv' % band[kk] )


	m_ra = np.r_[ lo_m_ra, hi_m_ra ]
	m_dec = np.r_[ lo_m_dec, hi_m_dec ]
	m_z = np.r_[ lo_m_z, hi_m_z ]

	pos_m = SkyCoord( m_ra, m_dec, unit = 'deg')
	p_EBV = E_map.ebv( pos_m )
	A_v = Rv * p_EBV
	A_l = A_wave( L_wave[ kk ], Rv) * A_v

	N_Low = len( lo_m_ra )
	orin_dex = np.ones( len(m_ra), dtype = int )
	
	_cc_order = np.arange( 0, len(m_ra),)
	id_vx = _cc_order >= N_Low
	orin_dex[ id_vx ] = 2

	keys = ['ra', 'dec', 'z', 'orin_dex', 'E_bv', 'A_l']
	values = [ m_ra, m_dec, m_z, orin_dex, p_EBV, A_l ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/BCG-Mstar_bin_%s-band_dust_value.csv' % band[kk] )


	rich_ra = np.r_[ lo_rich_ra, hi_rich_ra ]
	rich_dec = np.r_[ lo_rich_dec, hi_rich_dec ]
	rich_z = np.r_[ lo_rich_z, hi_rich_z ]

	pos_rich = SkyCoord( rich_ra, rich_dec, unit = 'deg')
	p_EBV = E_map.ebv( pos_rich )
	A_v = Rv * p_EBV
	A_l = A_wave( L_wave[ kk ], Rv) * A_v

	N_Low = len( lo_rich_ra )
	orin_dex = np.ones( len(rich_ra), dtype = int )
	
	_cc_order = np.arange( 0, len(rich_ra),)
	id_vx = _cc_order >= N_Low
	orin_dex[ id_vx ] = 2

	keys = ['ra', 'dec', 'z', 'orin_dex', 'E_bv', 'A_l']
	values = [ rich_ra, rich_dec, rich_z, orin_dex, p_EBV, A_l ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/clust-rich_bin_%s-band_dust_value.csv' % band[kk] )

#. rgi common samples
for kk in range( 3 ):

	###... bcg age binned sample
	hi_dat = pds.read_csv( load + 'pkoffset_cat/' + 
				'hi-age_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_cat_z-ref.csv' % band[kk] )
	hi_age_ra, hi_age_dec, hi_age_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )

	lo_dat = pds.read_csv( load + 'pkoffset_cat/' + 
				'low-age_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_cat_z-ref.csv' % band[kk] )
	lo_age_ra, lo_age_dec, lo_age_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )


	###... bcg mass binned sample
	hi_dat = pds.read_csv( load + 'pkoffset_cat/' + 
				'high_BCG_star-Mass_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_cat_z-ref.csv' % band[kk] )
	hi_m_ra, hi_m_dec, hi_m_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )

	lo_dat = pds.read_csv( load + 'pkoffset_cat/' + 
				'low_BCG_star-Mass_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_cat_z-ref.csv' % band[kk] )
	lo_m_ra, lo_m_dec, lo_m_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )


	###... richness binned sample
	hi_dat = pds.read_csv( load + 'pkoffset_cat/' + 
				'hi-rich_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_cat_z-ref.csv' % band[kk] )
	hi_rich_ra, hi_rich_dec, hi_rich_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )

	lo_dat = pds.read_csv( load + 'pkoffset_cat/' + 
				'low-rich_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_cat_z-ref.csv' % band[kk] )
	lo_rich_ra, lo_rich_dec, lo_rich_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )


	age_ra = np.r_[ lo_age_ra, hi_age_ra ]
	age_dec = np.r_[ lo_age_dec, hi_age_dec ]
	age_z = np.r_[ lo_age_z, hi_age_z ]

	pos_age = SkyCoord( age_ra, age_dec, unit = 'deg')
	p_EBV = E_map.ebv( pos_age )
	A_v = Rv * p_EBV
	A_l = A_wave( L_wave[ kk ], Rv) * A_v

	N_Low = len( lo_age_ra )
	orin_dex = np.ones( len(age_ra), dtype = int )
	
	_cc_order = np.arange( 0, len(age_ra),)
	id_vx = _cc_order >= N_Low
	orin_dex[ id_vx ] = 2

	keys = ['ra', 'dec', 'z', 'orin_dex', 'E_bv', 'A_l']
	values = [ age_ra, age_dec, age_z, orin_dex, p_EBV, A_l ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/BCG-age_bin_gri-common-cat_%s-band_dust_value.csv' % band[kk] )


	m_ra = np.r_[ lo_m_ra, hi_m_ra ]
	m_dec = np.r_[ lo_m_dec, hi_m_dec ]
	m_z = np.r_[ lo_m_z, hi_m_z ]

	pos_m = SkyCoord( m_ra, m_dec, unit = 'deg')
	p_EBV = E_map.ebv( pos_m )
	A_v = Rv * p_EBV
	A_l = A_wave( L_wave[ kk ], Rv) * A_v

	N_Low = len( lo_m_ra )
	orin_dex = np.ones( len(m_ra), dtype = int )
	
	_cc_order = np.arange( 0, len(m_ra),)
	id_vx = _cc_order >= N_Low
	orin_dex[ id_vx ] = 2

	keys = ['ra', 'dec', 'z', 'orin_dex', 'E_bv', 'A_l']
	values = [ m_ra, m_dec, m_z, orin_dex, p_EBV, A_l ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/BCG-Mstar_bin_gri-common-cat_%s-band_dust_value.csv' % band[kk] )


	rich_ra = np.r_[ lo_rich_ra, hi_rich_ra ]
	rich_dec = np.r_[ lo_rich_dec, hi_rich_dec ]
	rich_z = np.r_[ lo_rich_z, hi_rich_z ]

	pos_rich = SkyCoord( rich_ra, rich_dec, unit = 'deg')
	p_EBV = E_map.ebv( pos_rich )
	A_v = Rv * p_EBV
	A_l = A_wave( L_wave[ kk ], Rv) * A_v

	N_Low = len( lo_rich_ra )
	orin_dex = np.ones( len(rich_ra), dtype = int )
	
	_cc_order = np.arange( 0, len(rich_ra),)
	id_vx = _cc_order >= N_Low
	orin_dex[ id_vx ] = 2

	keys = ['ra', 'dec', 'z', 'orin_dex', 'E_bv', 'A_l']
	values = [ rich_ra, rich_dec, rich_z, orin_dex, p_EBV, A_l ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/clust-rich_gri-common-cat_%s-band_dust_value.csv' % band[kk] )

"""

### === ### for all low-z cluster sample (using for stacking)
dat = pds.read_csv('/home/xkchen/data/SDSS/photo_files/lowz_cluster_cat/' + 
					'clslowz_z0.17-0.30_img-cat_match.csv')
ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

pos_deg = SkyCoord( ra, dec, unit = 'deg')
p_EBV = E_map.ebv( pos_deg )
A_v = Rv * p_EBV

Al_r = A_wave( L_wave[ 0 ], Rv) * A_v
Al_g = A_wave( L_wave[ 1 ], Rv) * A_v
Al_i = A_wave( L_wave[ 2 ], Rv) * A_v

keys = [ 'ra', 'dec', 'z', 'E_bv', 'Al_r', 'Al_g', 'Al_i' ]
values = [ ra, dec, z, p_EBV, Al_r, Al_g, Al_i ]
fill = dict( zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/clslowz_z0.17-0.30_img-cat_dust_value.csv')

