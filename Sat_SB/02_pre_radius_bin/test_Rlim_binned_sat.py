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


### === ### satellites view and division
def scale_R_divid():

	s_dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv')
	bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
	p_ra, p_dec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] )

	p_Rsat = np.array( s_dat['R_cen'] )
	p_R2Rv = np.array( s_dat['Rcen/Rv'] )
	clus_IDs = np.array( s_dat['clus_ID'] )


	## divide by scaled R
	R_cut = 0.191 # np.median( p_Rcen ) = 0.191 * R200m
	id_vx = p_R2Rv <= R_cut

	#. inner part
	out_c_ra, out_c_dec, out_c_z = bcg_ra[ id_vx ], bcg_dec[ id_vx ], bcg_z[ id_vx ]
	out_s_ra, out_s_dec = p_ra[ id_vx ], p_dec[ id_vx ]
	out_Rsat = p_Rsat[ id_vx ]
	out_R2Rv = p_R2Rv[ id_vx ]
	out_clus_ID = clus_IDs[ id_vx ]

	keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'R_sat', 'R2Rv', 'clus_ID'] 
	values = [ out_c_ra, out_c_dec, out_c_z, out_s_ra, out_s_dec, out_Rsat, out_R2Rv, out_clus_ID]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/' + 
		'Extend-BCGM_rgi-common_frame-lim_Pm-cut_inner-mem_cat.csv')

	#. outer part
	out_c_ra, out_c_dec, out_c_z = bcg_ra[ id_vx == False ], bcg_dec[ id_vx == False ], bcg_z[ id_vx == False ]
	out_s_ra, out_s_dec = p_ra[ id_vx == False ], p_dec[ id_vx == False ]
	out_Rsat = p_Rsat[ id_vx == False ]
	out_R2Rv = p_R2Rv[ id_vx == False ]
	out_clus_ID = clus_IDs[ id_vx == False ]

	keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'R_sat', 'R2Rv', 'clus_ID']
	values = [ out_c_ra, out_c_dec, out_c_z, out_s_ra, out_s_dec, out_Rsat, out_R2Rv, out_clus_ID ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/' + 
				'Extend-BCGM_rgi-common_frame-lim_Pm-cut_outer-mem_cat.csv')

	return

def phyR_divid():

	s_dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv')
	bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
	p_ra, p_dec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] )

	p_Rsat = np.array( s_dat['R_cen'] )
	p_R2Rv = np.array( s_dat['Rcen/Rv'] )
	clus_IDs = np.array( s_dat['clus_ID'] )


	## divide by R
	R_cut = 0.213
	id_vx = p_Rsat <= R_cut

	#. inner part
	out_c_ra, out_c_dec, out_c_z = bcg_ra[ id_vx ], bcg_dec[ id_vx ], bcg_z[ id_vx ]
	out_s_ra, out_s_dec = p_ra[ id_vx ], p_dec[ id_vx ]
	out_Rsat = p_Rsat[ id_vx ]
	out_R2Rv = p_R2Rv[ id_vx ]
	out_clus_ID = clus_IDs[ id_vx ]

	keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'R_sat', 'R2Rv', 'clus_ID' ] 
	values = [ out_c_ra, out_c_dec, out_c_z, out_s_ra, out_s_dec, out_Rsat, out_R2Rv, out_clus_ID ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_R-phy_inner-mem_cat.csv')


	#. outer part
	out_c_ra, out_c_dec, out_c_z = bcg_ra[ id_vx == False ], bcg_dec[ id_vx == False ], bcg_z[ id_vx == False ]
	out_s_ra, out_s_dec = p_ra[ id_vx == False ], p_dec[ id_vx == False ]
	out_Rsat = p_Rsat[ id_vx == False ]
	out_R2Rv = p_R2Rv[ id_vx == False ]
	out_clus_ID = clus_IDs[ id_vx == False ]

	keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'R_sat', 'R2Rv', 'clus_ID' ]
	values = [ out_c_ra, out_c_dec, out_c_z, out_s_ra, out_s_dec, out_Rsat, out_R2Rv, out_clus_ID ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_R-phy_outer-mem_cat.csv')

	return


##.. stacking information match
def sat_cut_xy_match():

	band = ['r', 'g', 'i']

	##... divided by scaled radius
	# s_dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_inner-mem_cat.csv')
	s_dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_outer-mem_cat.csv')
	bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
	p_ra, p_dec = np.array( s_dat['sat_ra'] ), np.array( s_dat['sat_dec'] )


	##... divided by physic-R
	# s_dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_R-phy_inner-mem_cat.csv')
	# s_dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_R-phy_outer-mem_cat.csv')
	# bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
	# p_ra, p_dec = np.array( s_dat['sat_ra'] ), np.array( s_dat['sat_dec'] )


	pre_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

	for kk in range( 3 ):

		dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos.csv' % band[ kk ])
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

		# data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_inner-mem_%s-band_pos.csv' % band[kk] )
		data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_outer-mem_%s-band_pos.csv' % band[kk] )
		
		# data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_R-phy_inner-mem_%s-band_pos.csv' % band[kk] )
		# data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_R-phy_outer-mem_%s-band_pos.csv' % band[kk] )

		#. z-ref position
		dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos_z-ref.csv' % band[ kk ])
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

		# data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_inner-mem_%s-band_pos_z-ref.csv' % band[kk] )
		data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_outer-mem_%s-band_pos_z-ref.csv' % band[kk] )

		# data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_R-phy_inner-mem_%s-band_pos_z-ref.csv' % band[kk] )
		# data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_R-phy_outer-mem_%s-band_pos_z-ref.csv' % band[kk] )

	return

# sat_cut_xy_match()


### === record the member or richness of cluster sample (for Ng_weit stacking in background estimation)
dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_cat.csv')
ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

rich = np.array( dat['rich'] )
clus_IDs = np.array( dat['clust_ID'] )
R_vir = np.array( dat['R_vir'] )


tmp_Ng, tmp_Ng_80 = [], []

#. Ng record for subsamples division
tt_Ng_80_in, tt_Ng_80_out = [], []

N_ss = len( ra )

for kk in range( N_ss ):

	pre_dat = Table.read('/home/xkchen/figs/extend_Zphoto_cat/zphot_01_033_cat/' + 
						'redMaPPer_z-phot_0.1-0.33_cluster-sate_record.h5', path = 'clust_%d/mem_table' % clus_IDs[ kk ],)

	sub_Pm = np.array( pre_dat['Pm'] )
	sub_ra = np.array( pre_dat['ra'] )
	sub_Rsat = np.array( pre_dat['Rcen'] )

	_kk_N0 = len( sub_ra )

	id_Px = sub_Pm >= 0.8
	_kk_N1 = np.sum( id_Px )

	#. radius division
	kk_R2Rv = sub_Rsat[ id_Px ] / R_vir[ kk ]

	id_Rx = kk_R2Rv <= 0.191

	tt_Ng_80_in.append( np.sum( id_Rx ) )
	tt_Ng_80_out.append( np.sum( id_Rx == False ) )

	tmp_Ng.append( _kk_N0 )
	tmp_Ng_80.append( _kk_N1 )

##...
# keys = ['ra', 'dec', 'z', 'Ng', 'Ng_80', 'rich']
# values = [ ra, dec, z, np.array( tmp_Ng ), np.array(tmp_Ng_80), rich ]
# fill = dict( zip( keys, values ) )
# data = pds.DataFrame( fill )
# data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_cat_Ng.csv')

##... 
keys = ['ra', 'dec', 'z', 'Ng_80']
values = [ ra, dec, z, np.array( tt_Ng_80_in) ]
fill = dict( zip( keys, values ) )
data = pds.DataFrame( fill )
data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_cat_inner-mem_Ng.csv')

##...
keys = ['ra', 'dec', 'z', 'Ng_80']
values = [ ra, dec, z, np.array( tt_Ng_80_out ) ]
fill = dict( zip( keys, values ) )
data = pds.DataFrame( fill )
data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_cat_outer-mem_Ng.csv')

