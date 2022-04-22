import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Ellipse

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc
import scipy.stats as sts

from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord


##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1. - Omega_m
Omega_k = 1. - (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']


### use the P_mem cut sample only
dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/' + 
					'Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv')

bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
sat_ra, sat_dec = np.array( dat['ra'] ), np.array( dat['dec'] )
clus_IDs = np.array( dat['clus_ID'] )

set_IDs = np.array( list( set(clus_IDs) ) )
set_IDs = set_IDs.astype( int )

"""
for kk in range( 3 ):

	pat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/BG_tract_cat/' + 
		'Extend-BCGM_rgi-common_frame-limit_exlu-BCG_Sat_%s-band_origin-img_position.csv' % band[kk],)

	kk_ra, kk_dec = np.array( pat['sat_ra'] ), np.array( pat['sat_dec'] )
	kk_bcg_x, kk_bcg_y = np.array( pat['bcg_x'] ), np.array( pat['bcg_y'] )
	kk_sat_x, kk_sat_y = np.array( pat['sat_x'] ), np.array( pat['sat_y'] )
	kk_PA = np.array( pat['sat_PA2bcg'] )

	kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

	#. match P_mem cut sample
	idx, sep, d3d = sub_coord.match_to_catalog_sky( kk_coord )
	id_lim = sep.value < 2.7e-4

	#. satellite information (for check)
	mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]
	mp_bcg_x, mp_bcg_y = kk_bcg_x[ idx[ id_lim ] ], kk_bcg_y[ idx[ id_lim ] ]
	mp_sat_x, mp_sat_y = kk_sat_x[ idx[ id_lim ] ], kk_sat_y[ idx[ id_lim ] ]
	mp_sat_PA = kk_PA[ idx[ id_lim ] ]

	#. save
	keys = pat.columns[1:]
	mp_arr = [ bcg_ra, bcg_dec, bcg_z, mp_ra, mp_dec, mp_bcg_x, mp_bcg_y, 
								mp_sat_x, mp_sat_y, mp_sat_PA, clus_IDs ]
	fill = dict( zip( keys, mp_arr ) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/Extend-BCGM_rgi-common_frame-limit_Pm-cut_exlu-BCG_Sat_%s-band_origin-img_position.csv' % band[kk],)

"""

raise


### satellites P_mem < 0.8
pat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/BG_tract_cat/' + 
	'Extend-BCGM_rgi-common_frame-limit_exlu-BCG_Sat_r-band_origin-img_position.csv',)

kk_ra, kk_dec = np.array( pat['sat_ra'] ), np.array( pat['sat_dec'] )
kk_bcg_ra, kk_bcg_dec, kk_bcg_z = np.array( pat['bcg_ra'] ), np.array( pat['bcg_dec'] ), np.array( pat['bcg_z'] )
kk_clusID = np.array( pat['clus_ID'] )

N_cs = len( set_IDs )


dif_bcg_ra, dif_bcg_dec, dif_bcg_z = np.array([]), np.array([]), np.array([])
dif_sat_ra, dif_sat_dec = np.array([]), np.array([])
dif_clus_ID = np.array([])

for tt in range( N_cs ):

	id_x = clus_IDs == set_IDs[ tt ]

	tt_ra, tt_dec = sat_ra[ id_x ], sat_dec[ id_x ]
	tt_cen_ra, tt_cen_dec, tt_cen_z = bcg_ra[ id_x ][0], bcg_dec[ id_x ][0], bcg_z[ id_x ][0]

	sub_coord = SkyCoord( ra = tt_ra * U.deg, dec = tt_dec * U.deg)


	id_y = kk_clusID == set_IDs[ tt ]

	cp_ra, cp_dec = kk_ra[ id_y ], kk_dec[ id_y ]

	kk_coord = SkyCoord( ra = cp_ra * U.deg, dec = cp_dec * U.deg )


	#. match P_mem cut sample
	idx, sep, d3d = kk_coord.match_to_catalog_sky( sub_coord )

	id_lim = sep.value < 2.7e-4
	id_out = id_lim == False

	#. satellite information (for check)
	mp_ra, mp_dec = cp_ra[ id_out ], cp_dec[ id_out ]

	tt_n = len( mp_ra )

	mp_bcg_ra, mp_bcg_dec = np.ones( tt_n,) * tt_cen_ra, np.ones( tt_n,) * tt_cen_dec
	mp_bcg_z = np.ones( tt_n,) * tt_cen_z
	mp_clusID = np.ones( tt_n,) * set_IDs[ tt ]

	dif_bcg_ra = np.r_[ dif_bcg_ra, mp_bcg_ra ]
	dif_bcg_dec = np.r_[ dif_bcg_dec, mp_bcg_dec ]
	dif_bcg_z = np.r_[ dif_bcg_z, mp_bcg_z ]

	dif_sat_ra = np.r_[ dif_sat_ra, mp_ra ]
	dif_sat_dec = np.r_[ dif_sat_dec, mp_dec ]

	dif_clus_ID = np.r_[ dif_clus_ID, mp_clusID ]


print( dif_sat_ra.shape )

#. save
keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'clus_ID']
mp_arr = [ dif_bcg_ra, dif_bcg_dec, dif_bcg_z, dif_sat_ra, dif_sat_dec, dif_clus_ID ]
fill = dict( zip( keys, mp_arr ) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/Extend-BCGM_rgi-common_frame-limit_below_Pm-cut_Sat.csv')

