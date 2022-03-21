import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pds

import mechanize
from io import StringIO

import scipy.stats as sts
import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
import scipy.interpolate as interp

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable


###... cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)

band = ['r', 'g', 'i']
pixel = 0.396
z_ref = 0.25



### === Extended BCG_Mstar catalog
"""
#. SDSS redMaPPer
dat = pds.read_csv('/home/xkchen/Downloads/new_zlw_cat/BCGM_gri-common_extend_cat.csv')
ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )


#. W15 cluster
z_dat = np.loadtxt( '/home/xkchen/Downloads/new_zlw_cat/W15_cat/WH15_chenxk_extend.dat' )

c_ID = z_dat[:,0]
c_ID = c_ID.astype( int )

c_ra, c_dec, c_z = z_dat[:,1], z_dat[:,2], z_dat[:,3]
r_mag = z_dat[:,4]

R500 = z_dat[:,5]
rich = z_dat[:,6]

N_mem = z_dat[:,7]
z_fag = z_dat[:,8]


#. whether repeat-map or not
da, dn = sts.find_repeats( c_ID )

if len( da ) > 0:

	N_mp = len( da )

	clos_ra, clos_dec = np.zeros( N_mp,), np.zeros( N_mp,)
	clos_z = np.zeros( N_mp,)

	for tt in range( N_mp ):
		
		_tt_id = int( da[ tt ] )
		_tt_dx = c_ID == _tt_id

		_tt_ra, _tt_dec = c_ra[ _tt_dx ], c_dec[ _tt_dx ]
		_tt_z = c_z[ _tt_dx ]

		d_sep = np.sqrt( ( _tt_ra - ra[_tt_id] )**2 + ( _tt_dec - dec[_tt_id] )**2 )

		clos_ra[ tt ] = _tt_ra[ d_sep == d_sep.min() ]
		clos_dec[ tt ] = _tt_dec[ d_sep == d_sep.min() ]
		clos_z[ tt ] = _tt_z[ d_sep == d_sep.min() ]

	#.
	set_IDs = np.array( list( set( c_ID ) ) )
	cp_ra, cp_dec, cp_z = ra[ set_IDs], dec[ set_IDs], z[ set_IDs]

	cp_Ns = len( cp_ra )
	mp_ra, mp_dec, mp_z = [], [], []

	for tt in range( cp_Ns ):

		tt_ID = set_IDs[ tt ]

		if tt_ID in da:

			idx = np.where( da == tt_ID )[0][0]

			mp_ra.append( clos_ra[ idx ] )
			mp_dec.append( clos_dec[ idx ] )
			mp_z.append( clos_z[ idx ] )

		else:

			idx = np.where( c_ID == tt_ID )[0][0]
			mp_ra.append( c_ra[ idx ] )
			mp_dec.append( c_dec[ idx ] )
			mp_z.append( c_z[ idx ] )

	mp_ra, mp_dec, mp_z = np.array( mp_ra), np.array( mp_dec), np.array( mp_z)

	keys = ['ra', 'dec', 'z', 'ra_W15', 'dec_W15', 'z_W15', 'map_order']
	values = [ cp_ra, cp_dec, cp_z, mp_ra, mp_dec, mp_z, set_IDs ]
	fill = dict( zip( keys, values ) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/Downloads/new_zlw_cat/W15_cat/Extend_SDSS_map_clust-cat.csv')

else:

	keys = ['ra', 'dec', 'z', 'ra_W15', 'dec_W15', 'z_W15', 'map_order' ]
	values = [ ra[c_ID], dec[c_ID], z[c_ID], c_ra, c_dec, c_z, c_ID ]
	fill = dict( zip( keys, values ) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/Downloads/new_zlw_cat/W15_cat/Extend_SDSS_map_clust-cat.csv')


##.. member table
c_dat = pds.read_csv('/home/xkchen/Downloads/new_zlw_cat/W15_cat/Extend_SDSS_map_clust-cat.csv')
c_ra, c_dec, c_ID = np.array( c_dat['ra_W15']), np.array( c_dat['dec_W15']), np.array( c_dat['map_order'])


#. W15 members
m_dat = np.loadtxt('/home/xkchen/Downloads/new_zlw_cat/W15_cat/clustmem_chenxk_extend.dat')

m_IDs = m_dat[:,0]
m_IDs = m_IDs.astype( int )

m_ra, m_dec, m_z = m_dat[:,1], m_dat[:,2], m_dat[:,3]

m_Rs = m_dat[:,7]
m_zfag = m_dat[:,8]


N_ss = len( c_ra )

tmp_ra, tmp_dec, tmp_z = np.array( [] ), np.array( [] ), np.array( [] )
tmp_mp_ra, tmp_mp_dec, tmp_mp_z = np.array( [] ), np.array( [] ), np.array( [] )
tmp_sat_ra, tmp_sat_dec = np.array( [] ), np.array( [] )

tmp_Rs = np.array( [] )
tmp_ordex = np.array( [] )

tt_N = []

for pp in range( N_ss ):

	id_vx = m_IDs == c_ID[ pp ]

	_pp_n = np.sum( id_vx )

	tt_N.append( _pp_n )

	tmp_sat_ra = np.r_[ tmp_sat_ra, m_ra[ id_vx ] ]
	tmp_sat_dec = np.r_[ tmp_sat_dec, m_dec[ id_vx ] ]

	tmp_Rs = np.r_[ tmp_Rs, m_Rs[ id_vx ] ]
	tmp_ordex = np.r_[ tmp_ordex, m_IDs[ id_vx ] ]

	tmp_ra = np.r_[ tmp_ra, np.ones( _pp_n,) * c_ra[ pp ] ]
	tmp_dec = np.r_[ tmp_dec, np.ones( _pp_n,) * c_dec[ pp ] ]
	tmp_z = np.r_[ tmp_z, np.ones( _pp_n,) * c_z[ pp ] ]

	tmp_mp_ra = np.r_[ tmp_mp_ra, np.ones( _pp_n,) * ra[ c_ID[ pp ] ] ]
	tmp_mp_dec = np.r_[ tmp_mp_dec, np.ones( _pp_n,) * dec[ c_ID[ pp ] ] ]
	tmp_mp_z =	np.r_[ tmp_mp_z, np.ones( _pp_n,) * z[ c_ID[ pp ] ] ]

#.
keys = ['ra', 'dec', 'R_sat', 'bcg_ra_W15', 'bcg_dec_W15', 'bcg_z_W15', 'map_order', 
											'bcg_ra_sdss', 'bcg_dec_sdss', 'bcg_z_sdss']
values = [ tmp_sat_ra, tmp_sat_dec, tmp_Rs, tmp_ra, tmp_dec, tmp_z, tmp_ordex, 
											tmp_mp_ra, tmp_mp_dec, tmp_mp_z ]
fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/Downloads/new_zlw_cat/W15_cat/W15_extend_clust-cat_match-sat.csv')

"""


"""
### === previous match
##... sdss catalog
# lo_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/age_bin/' + 
# 					'low-age_r-band_photo-z-match_rgi-common_cat_params.csv')

# hi_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/age_bin/' + 
# 					'hi-age_r-band_photo-z-match_rgi-common_cat_params.csv')


lo_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_gri_common/' + 
						'low_BCG_star-Mass_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv')

hi_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_gri_common/' + 
						'high_BCG_star-Mass_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv')

lo_ra, lo_dec, lo_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )

hi_ra, hi_dec, hi_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )



##... Wen+2015 catalog
# pat = np.loadtxt('/home/xkchen/mywork/Sat_SB/data/remap_W15_cat/age_bin_map/WH15_chenxk.dat')
pat = np.loadtxt('/home/xkchen/mywork/Sat_SB/data/remap_W15_cat/ZLW_cat_15/WH15_chenxk_BCG.dat')

map_ID = pat[:,0]
c_ra, c_dec, c_z = pat[:,1], pat[:,2], pat[:,3]

mag_r = pat[:,4]
R_500 = pat[:,5]

c_rich = pat[:,6]
N_g = pat[:,7]

set_ID = pat[:,9]


id_vx = set_ID == 1

sub_ID0 = map_ID[ id_vx ]
sub_ID0 = sub_ID0.astype( int )
sub_ra_0, sub_dec_0, sub_z_0 = lo_ra[ sub_ID0 ], lo_dec[ sub_ID0 ], lo_z[ sub_ID0 ]


id_vx = set_ID == 2
sub_ID1 = map_ID[ id_vx ]
sub_ID1 = sub_ID1.astype( int )
sub_ra_1, sub_dec_1, sub_z_1 = hi_ra[ sub_ID1 ], hi_dec[ sub_ID1 ], hi_z[ sub_ID1 ]


sdss_ra = np.r_[ sub_ra_0, sub_ra_1 ]
sdss_dec = np.r_[ sub_dec_0, sub_dec_1 ] 
sdss_z = np.r_[ sub_z_0, sub_z_1 ]

#.
keys = ['ra', 'dec', 'z', 'ra_W15', 'dec_W15', 'z_W15', 'map_order', 'sample_ID']
values = [ sdss_ra, sdss_dec, sdss_z, c_ra, c_dec, c_z, map_ID, set_ID ]
fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )

# out_data.to_csv( '/home/xkchen/mywork/Sat_SB/data/remap_W15_cat/age_bin_map/redMaPPer_map_clust-cat.csv')
out_data.to_csv( '/home/xkchen/mywork/Sat_SB/data/remap_W15_cat/ZLW_cat_15/redMaPPer_BCG-Mstar_map_clust-cat.csv')


##.. Wen+2015 member and member-cluster mapping
# sat_dat = np.loadtxt('/home/xkchen/mywork/Sat_SB/data/remap_W15_cat/age_bin_map/clustmem_chenxk.dat')
sat_dat = np.loadtxt('/home/xkchen/mywork/Sat_SB/data/remap_W15_cat/ZLW_cat_15/clustmem_chenxk_BCG.dat')

clust_id = sat_dat[:,0]

sat_ra, sat_dec = sat_dat[:,1], sat_dat[:,2]
sat_z = sat_dat[:,3]

sat_rmag = sat_dat[:,4]
sat_rMag = sat_dat[:,5]

cen_dL = sat_dat[:,6]   ## need to record
sat_z_flag = sat_dat[:,7]


N_low = np.sum( set_ID == 1 )
N_high = np.sum( set_ID == 2 )
N_tot = N_low + N_high
N_rep_dex = np.min( [ N_low, N_high ] )


tt_N = []

tmp_ra, tmp_dec, tmp_z = np.array( [] ), np.array( [] ), np.array( [] )
tmp_sat_ra, tmp_sat_dec = np.array( [] ), np.array( [] )

tmp_R_sat = np.array( [] )
tmp_map_IDs = np.array( [] )

tcp_ra, tcp_dec, tcp_z = np.array( [] ), np.array( [] ), np.array( [] )

#. the first subsample
for tt in range( N_low ):

	ra_g, dec_g, z_g = c_ra[ tt ], c_dec[ tt ], c_z[ tt ]

	if tt < N_rep_dex:
		cc_ra_g, cc_dec_g, cc_z_g = c_ra[ tt + N_low ], c_dec[ tt + N_low ], c_z[ tt + N_low ]

	else:
		cc_ra_g, cc_dec_g, cc_z_g = -1000, -1000, -1000


	tag_dex = clust_id == sub_ID0[ tt ]

	daa = np.where( tag_dex )[0]
	dev = np.diff( daa )

	id_rep = dev > 1

	if np.sum( id_rep ) > 0:

		mid_dex = np.where( id_rep )[0]

		#. identi data_arr one by one
		_n_mid = len( mid_dex )
		_bond_x0 = daa[ mid_dex ]

		bond_dex = np.zeros( int(2 * (_n_mid + 1) ), dtype = int )

		bond_dex[::2] = np.r_[ daa[0], daa[ mid_dex + 1] ]
		bond_dex[1::2] = np.r_[ daa[mid_dex], daa[-1] ]

		_n_bond = len( bond_dex )

		#. record array
		sub_ra, sub_dec, sub_z = np.array([]), np.array([]), np.array([])
		sub_Rsat = np.array([])

		for pp in range(0, _n_bond -1, 2 ):

			da0, da1 = bond_dex[ pp ], bond_dex[ pp + 1 ] + 1

			mid_ra, mid_dec = sat_ra[ da0 : da1], sat_dec[ da0 : da1]
			mid_z = sat_z[ da0 : da1]
			mid_cen_dL = cen_dL[ da0 : da1]

			deta_ra_0 = np.mean( np.abs( mid_ra - ra_g ) )
			deta_ra_1 = np.mean( np.abs( mid_ra - cc_ra_g) )

			if deta_ra_0 < deta_ra_1:

				sub_ra = np.r_[ sub_ra, mid_ra ]
				sub_dec = np.r_[ sub_dec, mid_dec ]
				sub_z = np.r_[ sub_z, mid_z ]
				sub_Rsat = np.r_[ sub_Rsat, mid_cen_dL ]

			else:

				continue

		_sub_n = len( sub_Rsat )

	else:
		da0 = daa[ 0 ]
		da1 = daa[-1 ] + 1

		sub_ra, sub_dec, sub_z = sat_ra[ da0: da1 ], sat_dec[ da0: da1 ], sat_z[ da0: da1 ]
		sub_Rsat = cen_dL[ da0: da1 ]

		_sub_n = len( sub_ra )

	tt_N.append( _sub_n )

	tmp_ra = np.r_[ tmp_ra, np.ones( _sub_n, ) * ra_g ]
	tmp_dec = np.r_[ tmp_dec, np.ones( _sub_n, ) * dec_g ]
	tmp_z = np.r_[ tmp_z, np.ones( _sub_n, ) * z_g ]

	tmp_sat_ra = np.r_[ tmp_sat_ra, sub_ra ]
	tmp_sat_dec = np.r_[ tmp_sat_dec, sub_dec ]
	tmp_R_sat = np.r_[ tmp_R_sat, sub_Rsat ]

	tmp_map_IDs = np.r_[ tmp_map_IDs, np.ones( _sub_n, ) * sub_ID0[ tt ] ]

	tcp_ra = np.r_[ tcp_ra, np.ones( _sub_n, ) * sub_ra_0[ tt ] ]
	tcp_dec = np.r_[ tcp_dec, np.ones( _sub_n, ) * sub_dec_0[ tt ] ]
	tcp_z = np.r_[ tcp_z, np.ones( _sub_n, ) * sub_z_0[ tt ] ]


tp_N = []

#. the second subsample
for tt in range( N_low, N_tot ):

	ra_g, dec_g, z_g = c_ra[ tt ], c_dec[ tt ], c_z[ tt ]

	sub_tt = np.int( tt - N_low )

	if sub_tt < N_rep_dex:
		cc_ra_g, cc_dec_g, cc_z_g = c_ra[ sub_tt ], c_dec[ sub_tt ], c_z[ sub_tt ]

	else:
		cc_ra_g, cc_dec_g, cc_z_g = -1000, -1000, -1000

	tag_dex = clust_id == sub_ID1[ sub_tt ]

	daa = np.where( tag_dex )[0]
	dev = np.diff( daa )

	id_rep = dev > 1

	if np.sum( id_rep ) > 0:

		mid_dex = np.where( id_rep )[0]

		#. identi data_arr one by one
		_n_mid = len( mid_dex )
		_bond_x0 = daa[ mid_dex ]

		bond_dex = np.zeros( int(2 * (_n_mid + 1) ), dtype = int )

		bond_dex[::2] = np.r_[ daa[0], daa[ mid_dex + 1] ]
		bond_dex[1::2] = np.r_[ daa[mid_dex], daa[-1] ]

		_n_bond = len( bond_dex )

		#. record array
		sub_ra, sub_dec, sub_z = np.array([]), np.array([]), np.array([])
		sub_Rsat = np.array([])		

		for pp in range(0, _n_bond -1, 2 ):

			da0, da1 = bond_dex[ pp ], bond_dex[ pp + 1 ] + 1

			mid_ra, mid_dec = sat_ra[ da0 : da1], sat_dec[ da0 : da1]
			mid_z = sat_z[ da0 : da1]
			mid_cen_dL = cen_dL[ da0 : da1]

			deta_ra_0 = np.mean( np.abs( mid_ra - ra_g ) )
			deta_ra_1 = np.mean( np.abs( mid_ra - cc_ra_g) )

			if deta_ra_0 < deta_ra_1:

				sub_ra = np.r_[ sub_ra, mid_ra ]
				sub_dec = np.r_[ sub_dec, mid_dec ]
				sub_z = np.r_[ sub_z, mid_z ]
				sub_Rsat = np.r_[ sub_Rsat, mid_cen_dL ]

			else:
				continue

		_sub_n = len( sub_Rsat )

	else:
		da0 = daa[ 0 ]
		da1 = daa[ -1 ] + 1

		sub_Rsat = cen_dL[ da0: da1 ]
		sub_ra, sub_dec, sub_z = sat_ra[ da0: da1 ], sat_dec[ da0: da1 ], sat_z[ da0: da1 ]

		_sub_n = len( sub_Rsat )

	tp_N.append( _sub_n )

	tmp_ra = np.r_[ tmp_ra, np.ones( _sub_n, ) * ra_g ]
	tmp_dec = np.r_[ tmp_dec, np.ones( _sub_n, ) * dec_g ]
	tmp_z = np.r_[ tmp_z, np.ones( _sub_n, ) * z_g ]

	tmp_sat_ra = np.r_[ tmp_sat_ra, sub_ra ]
	tmp_sat_dec = np.r_[ tmp_sat_dec, sub_dec ]
	tmp_R_sat = np.r_[ tmp_R_sat, sub_Rsat ]

	tmp_map_IDs = np.r_[ tmp_map_IDs, np.ones( _sub_n, ) * sub_ID1[ sub_tt ] ]

	tcp_ra = np.r_[ tcp_ra, np.ones( _sub_n, ) * sub_ra_1[ sub_tt ] ]
	tcp_dec = np.r_[ tcp_dec, np.ones( _sub_n, ) * sub_dec_1[ sub_tt ] ]
	tcp_z = np.r_[ tcp_z, np.ones( _sub_n, ) * sub_z_1[ sub_tt ] ]

set_divid = np.r_[ np.ones( np.sum(tt_N),), np.ones( np.sum(tp_N),) * 2 ]

##. save
keys = ['ra', 'dec', 'R_sat', 'bcg_ra_W15', 'bcg_dec_W15', 'bcg_z_W15', 
			'map_order', 'sample_sep', 'bcg_ra_sdss', 'bcg_dec_sdss', 'bcg_z_sdss']
values = [ tmp_sat_ra, tmp_sat_dec, tmp_R_sat, tmp_ra, tmp_dec, tmp_z, 
			tmp_map_IDs, set_divid, tcp_ra, tcp_dec, tcp_z ]

fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )

# out_data.to_csv( '/home/xkchen/mywork/Sat_SB/data/remap_W15_cat/age_bin_map/W15_age-bin_clust-cat_match-sat.csv')
out_data.to_csv( '/home/xkchen/mywork/Sat_SB/data/remap_W15_cat/ZLW_cat_15/W15_BCG-Mstar_clust-cat_match-sat.csv')

"""


### === rich-bin match
##... sdss catalog

hi_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/' + 
						'rich_bin/hi-rich_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv')

lo_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/' + 
						'rich_bin/low-rich_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv')

hi_ra, hi_dec, hi_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )

lo_ra, lo_dec, lo_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )


##... Wen+2015 catalog
pat = np.loadtxt('/home/xkchen/mywork/Sat_SB/data/remap_W15_cat/ZLW_cat_15/cluster_chenxk_rich.dat')

map_ID = pat[:,0]
c_ra, c_dec, c_z = pat[:,1], pat[:,2], pat[:,3]

mag_r = pat[:,4]
z_spec = pat[:,5]

set_ID = pat[:,6]


id_vx = set_ID == 1

sub_ID0 = map_ID[ id_vx ]
sub_ID0 = sub_ID0.astype( int )
sub_ra_0, sub_dec_0, sub_z_0 = lo_ra[ sub_ID0 ], lo_dec[ sub_ID0 ], lo_z[ sub_ID0 ]


id_vx = set_ID == 2
sub_ID1 = map_ID[ id_vx ]
sub_ID1 = sub_ID1.astype( int )
sub_ra_1, sub_dec_1, sub_z_1 = hi_ra[ sub_ID1 ], hi_dec[ sub_ID1 ], hi_z[ sub_ID1 ]


sdss_ra = np.r_[ sub_ra_0, sub_ra_1 ]
sdss_dec = np.r_[ sub_dec_0, sub_dec_1 ] 
sdss_z = np.r_[ sub_z_0, sub_z_1 ]

#.
keys = ['ra', 'dec', 'z', 'ra_W15', 'dec_W15', 'z_W15', 'map_order', 'sample_ID']
values = [ sdss_ra, sdss_dec, sdss_z, c_ra, c_dec, c_z, map_ID, set_ID ]
fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/mywork/Sat_SB/data/remap_W15_cat/ZLW_cat_15/redMaPPer_rich-bin_map_clust-cat.csv')


##... W15 members
sat_dat = np.loadtxt('/home/xkchen/mywork/Sat_SB/data/remap_W15_cat/ZLW_cat_15/clustmem_chenxk_rich.dat')

clust_id = sat_dat[:,0]

sat_ra, sat_dec = sat_dat[:,1], sat_dat[:,2]
sat_z = sat_dat[:,3]

sat_rmag = sat_dat[:,4]
sat_rMag = sat_dat[:,5]

cen_dL = sat_dat[:,6]   ## need to record
sat_z_flag = sat_dat[:,7]


N_low = np.sum( set_ID == 1 )
N_high = np.sum( set_ID == 2 )
N_tot = N_low + N_high
N_rep_dex = np.min( [ N_low, N_high ] )


tt_N = []

tmp_ra, tmp_dec, tmp_z = np.array( [] ), np.array( [] ), np.array( [] )
tmp_sat_ra, tmp_sat_dec = np.array( [] ), np.array( [] )

tmp_R_sat = np.array( [] )
tmp_map_IDs = np.array( [] )

tcp_ra, tcp_dec, tcp_z = np.array( [] ), np.array( [] ), np.array( [] )

#. the first subsample
for tt in range( N_low ):

	ra_g, dec_g, z_g = c_ra[ tt ], c_dec[ tt ], c_z[ tt ]

	if tt < N_rep_dex:
		cc_ra_g, cc_dec_g, cc_z_g = c_ra[ tt + N_low ], c_dec[ tt + N_low ], c_z[ tt + N_low ]

	else:
		cc_ra_g, cc_dec_g, cc_z_g = -1000, -1000, -1000


	tag_dex = clust_id == sub_ID0[ tt ]

	daa = np.where( tag_dex )[0]
	dev = np.diff( daa )

	id_rep = dev > 1

	if np.sum( id_rep ) > 0:

		mid_dex = np.where( id_rep )[0]

		#. identi data_arr one by one
		_n_mid = len( mid_dex )
		_bond_x0 = daa[ mid_dex ]

		bond_dex = np.zeros( int(2 * (_n_mid + 1) ), dtype = int )

		bond_dex[::2] = np.r_[ daa[0], daa[ mid_dex + 1] ]
		bond_dex[1::2] = np.r_[ daa[mid_dex], daa[-1] ]

		_n_bond = len( bond_dex )

		#. record array
		sub_ra, sub_dec, sub_z = np.array([]), np.array([]), np.array([])
		sub_Rsat = np.array([])

		for pp in range(0, _n_bond -1, 2 ):

			da0, da1 = bond_dex[ pp ], bond_dex[ pp + 1 ] + 1

			mid_ra, mid_dec = sat_ra[ da0 : da1], sat_dec[ da0 : da1]
			mid_z = sat_z[ da0 : da1]
			mid_cen_dL = cen_dL[ da0 : da1]

			deta_ra_0 = np.mean( np.abs( mid_ra - ra_g ) )
			deta_ra_1 = np.mean( np.abs( mid_ra - cc_ra_g) )

			if deta_ra_0 < deta_ra_1:

				sub_ra = np.r_[ sub_ra, mid_ra ]
				sub_dec = np.r_[ sub_dec, mid_dec ]
				sub_z = np.r_[ sub_z, mid_z ]
				sub_Rsat = np.r_[ sub_Rsat, mid_cen_dL ]

			else:

				continue

		_sub_n = len( sub_Rsat )

	else:
		da0 = daa[ 0 ]
		da1 = daa[-1 ] + 1

		sub_ra, sub_dec, sub_z = sat_ra[ da0: da1 ], sat_dec[ da0: da1 ], sat_z[ da0: da1 ]
		sub_Rsat = cen_dL[ da0: da1 ]

		_sub_n = len( sub_ra )

	tt_N.append( _sub_n )

	tmp_ra = np.r_[ tmp_ra, np.ones( _sub_n, ) * ra_g ]
	tmp_dec = np.r_[ tmp_dec, np.ones( _sub_n, ) * dec_g ]
	tmp_z = np.r_[ tmp_z, np.ones( _sub_n, ) * z_g ]

	tmp_sat_ra = np.r_[ tmp_sat_ra, sub_ra ]
	tmp_sat_dec = np.r_[ tmp_sat_dec, sub_dec ]
	tmp_R_sat = np.r_[ tmp_R_sat, sub_Rsat ]

	tmp_map_IDs = np.r_[ tmp_map_IDs, np.ones( _sub_n, ) * sub_ID0[ tt ] ]

	tcp_ra = np.r_[ tcp_ra, np.ones( _sub_n, ) * sub_ra_0[ tt ] ]
	tcp_dec = np.r_[ tcp_dec, np.ones( _sub_n, ) * sub_dec_0[ tt ] ]
	tcp_z = np.r_[ tcp_z, np.ones( _sub_n, ) * sub_z_0[ tt ] ]


tp_N = []

#. the second subsample
for tt in range( N_low, N_tot ):

	ra_g, dec_g, z_g = c_ra[ tt ], c_dec[ tt ], c_z[ tt ]

	sub_tt = np.int( tt - N_low )

	if sub_tt < N_rep_dex:
		cc_ra_g, cc_dec_g, cc_z_g = c_ra[ sub_tt ], c_dec[ sub_tt ], c_z[ sub_tt ]

	else:
		cc_ra_g, cc_dec_g, cc_z_g = -1000, -1000, -1000

	tag_dex = clust_id == sub_ID1[ sub_tt ]

	daa = np.where( tag_dex )[0]
	dev = np.diff( daa )

	id_rep = dev > 1

	if np.sum( id_rep ) > 0:

		mid_dex = np.where( id_rep )[0]

		#. identi data_arr one by one
		_n_mid = len( mid_dex )
		_bond_x0 = daa[ mid_dex ]

		bond_dex = np.zeros( int(2 * (_n_mid + 1) ), dtype = int )

		bond_dex[::2] = np.r_[ daa[0], daa[ mid_dex + 1] ]
		bond_dex[1::2] = np.r_[ daa[mid_dex], daa[-1] ]

		_n_bond = len( bond_dex )

		#. record array
		sub_ra, sub_dec, sub_z = np.array([]), np.array([]), np.array([])
		sub_Rsat = np.array([])		

		for pp in range(0, _n_bond -1, 2 ):

			da0, da1 = bond_dex[ pp ], bond_dex[ pp + 1 ] + 1

			mid_ra, mid_dec = sat_ra[ da0 : da1], sat_dec[ da0 : da1]
			mid_z = sat_z[ da0 : da1]
			mid_cen_dL = cen_dL[ da0 : da1]

			deta_ra_0 = np.mean( np.abs( mid_ra - ra_g ) )
			deta_ra_1 = np.mean( np.abs( mid_ra - cc_ra_g) )

			if deta_ra_0 < deta_ra_1:

				sub_ra = np.r_[ sub_ra, mid_ra ]
				sub_dec = np.r_[ sub_dec, mid_dec ]
				sub_z = np.r_[ sub_z, mid_z ]
				sub_Rsat = np.r_[ sub_Rsat, mid_cen_dL ]

			else:
				continue

		_sub_n = len( sub_Rsat )

	else:
		da0 = daa[ 0 ]
		da1 = daa[ -1 ] + 1

		sub_Rsat = cen_dL[ da0: da1 ]
		sub_ra, sub_dec, sub_z = sat_ra[ da0: da1 ], sat_dec[ da0: da1 ], sat_z[ da0: da1 ]

		_sub_n = len( sub_Rsat )

	tp_N.append( _sub_n )

	tmp_ra = np.r_[ tmp_ra, np.ones( _sub_n, ) * ra_g ]
	tmp_dec = np.r_[ tmp_dec, np.ones( _sub_n, ) * dec_g ]
	tmp_z = np.r_[ tmp_z, np.ones( _sub_n, ) * z_g ]

	tmp_sat_ra = np.r_[ tmp_sat_ra, sub_ra ]
	tmp_sat_dec = np.r_[ tmp_sat_dec, sub_dec ]
	tmp_R_sat = np.r_[ tmp_R_sat, sub_Rsat ]

	tmp_map_IDs = np.r_[ tmp_map_IDs, np.ones( _sub_n, ) * sub_ID1[ sub_tt ] ]

	tcp_ra = np.r_[ tcp_ra, np.ones( _sub_n, ) * sub_ra_1[ sub_tt ] ]
	tcp_dec = np.r_[ tcp_dec, np.ones( _sub_n, ) * sub_dec_1[ sub_tt ] ]
	tcp_z = np.r_[ tcp_z, np.ones( _sub_n, ) * sub_z_1[ sub_tt ] ]

set_divid = np.r_[ np.ones( np.sum(tt_N),), np.ones( np.sum(tp_N),) * 2 ]

##. save
keys = ['ra', 'dec', 'R_sat', 'bcg_ra_W15', 'bcg_dec_W15', 'bcg_z_W15', 
			'map_order', 'sample_sep', 'bcg_ra_sdss', 'bcg_dec_sdss', 'bcg_z_sdss']
values = [ tmp_sat_ra, tmp_sat_dec, tmp_R_sat, tmp_ra, tmp_dec, tmp_z, 
			tmp_map_IDs, set_divid, tcp_ra, tcp_dec, tcp_z ]

fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/mywork/Sat_SB/data/remap_W15_cat/ZLW_cat_15/W15_rich-bin_clust-cat_match-sat.csv')

