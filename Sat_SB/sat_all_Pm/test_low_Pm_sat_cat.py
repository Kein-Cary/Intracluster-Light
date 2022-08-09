"""
record the satellite galaxy with P_mem < 0.8,
and extract their galaxy and background image
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pds
import h5py

import mechanize
from io import StringIO
import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable

#.
from Mass_rich_radius import rich2R_Simet
from img_sat_fig_out_mode import zref_sat_pos_func


###... cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)


### === data load
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_all_Pm/cat/'

dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/' + 
                    'Extend-BCGM_rgi-common_frame-limit_member-cat.csv')

keys = dat.columns[1:]
N_ks = len( keys )

sat_ra, sat_dec = np.array( dat['ra'] ), np.array( dat['dec'] )

pre_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )


cat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/' + 
                    'Extend-BCGM_rgi-common_frame-limit_Pm-cut_member-cat.csv')

c_ra, c_dec = np.array( cat['ra'] ), np.array( cat['dec'] )

c_coord = SkyCoord( ra = c_ra * U.deg, dec = c_dec * U.deg )


idx, sep, d3d = c_coord.match_to_catalog_sky( pre_coord )

#. P_mem >= 0.8
id_lim = sep.value < 2.7e-4
mp_ra, mp_dec = sat_ra[ idx[ id_lim ] ], sat_dec[ idx[ id_lim ] ]

#. P_mem < 0.8
N_all = len( sat_ra )

ord_dex = np.arange( N_all )

res_dex = np.delete( ord_dex, tuple( idx ) )

res_ra, res_dec = sat_ra[ res_dex ], sat_dec[ res_dex ]

tmp_arr = []

for dd in range( N_ks ): 

    sub_arr = np.array( dat[ keys[dd] ], )

    tmp_arr.append( sub_arr[ res_dex ] )

fill = dict( zip( keys, tmp_arr ) )
data = pds.DataFrame( fill )
data.to_csv( cat_path + 'Extend-BCGM_rgi-common_frame-lim_lower-Pm_member-cat.csv')


### === satellite position record and match
band = ['r', 'g', 'i']


s_dat = pds.read_csv( cat_path + 'Extend-BCGM_rgi-common_frame-lim_lower-Pm_member-cat.csv' )
bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
p_ra, p_dec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] )

pre_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

for kk in range( 3 ):

	dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/' + 
			'Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos.csv' % band[ kk ])
	kk_ra, kk_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
	kk_imgx, kk_imgy = np.array( dat['cut_cx'] ), np.array( dat['cut_cy'] )

	kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

	idx, sep, d3d = pre_coord.match_to_catalog_sky( kk_coord )
	id_lim = sep.value < 2.7e-4

	mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]
	mp_imgx, mp_imgy = kk_imgx[ idx[ id_lim ] ], kk_imgy[ idx[ id_lim ] ]

	keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'cut_cx', 'cut_cy']
	values = [ bcg_ra, bcg_dec, bcg_z, p_ra, p_dec, mp_imgx, mp_imgy ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( cat_path + 
            'Extend-BCGM_rgi-common_frame-lim_lower-Pm_member_%s-band_pos.csv' % band[kk] )

	#. z-ref position
	dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/' + 
			'Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos_z-ref.csv' % band[kk] )
	kk_ra, kk_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
	kk_imgx, kk_imgy = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

	kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

	idx, sep, d3d = pre_coord.match_to_catalog_sky( kk_coord )
	id_lim = sep.value < 2.7e-4

	mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]	
	mp_imgx, mp_imgy = kk_imgx[ idx[ id_lim ] ], kk_imgy[ idx[ id_lim ] ]

	keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_x', 'sat_y']
	values = [ bcg_ra, bcg_dec, bcg_z, p_ra, p_dec, mp_imgx, mp_imgy ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( cat_path + 
			'Extend-BCGM_rgi-common_frame-lim_lower-Pm_member_%s-band_pos_z-ref.csv' % band[kk] )

