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

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from fig_out_module import absMag_to_Lumi_func

### === ### cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone( H0 = 67.74, Om0 = 0.311 )
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

pixel = 0.396
band = ['r', 'g', 'i']
L_wave = np.array([ 6166, 4686, 7480 ])
Mag_sun = [ 4.65, 5.11, 4.53 ]

###===### SDSS redMaPPer catalog compare
"""
###... bcg age binned sample
hi_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/' + 
						'age_bin/hi-age_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv')
hi_age_ra, hi_age_dec, hi_age_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )

lo_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/' + 
						'age_bin/low-age_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv')
lo_age_ra, lo_age_dec, lo_age_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )


###... bcg mass binned sample
hi_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_gri_common/' + 
						'high_BCG_star-Mass_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv')
hi_m_ra, hi_m_dec, hi_m_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )

lo_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_gri_common/' + 
						'low_BCG_star-Mass_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv')
lo_m_ra, lo_m_dec, lo_m_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )


###... richness binned sample
hi_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/' + 
						'rich_bin/hi-rich_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv')
hi_ra, hi_dec, hi_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )

lo_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/' + 
						'rich_bin/low-rich_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv')
lo_ra, lo_dec, lo_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )


###... different sample compared to t_age sample
age_ra = np.r_[ lo_age_ra, hi_age_ra ]
age_dec = np.r_[ lo_age_dec, hi_age_dec ]

m_ra = np.r_[ lo_m_ra, hi_m_ra ]
m_dec = np.r_[ lo_m_dec, hi_m_dec ]

rich_ra = np.r_[ lo_ra, hi_ra ]
rich_dec = np.r_[ lo_dec, hi_dec ]

Ns_t = len( age_ra )
Ns_m = len( m_ra )
Ns_r = len( rich_ra )

age_coord = SkyCoord( ra = age_ra*U.deg, dec = age_dec*U.deg,)
m_coord = SkyCoord( ra = m_ra*U.deg, dec = m_dec*U.deg,)
rich_coord = SkyCoord( ra = rich_ra*U.deg, dec = rich_dec*U.deg,)

m_idx, m_sep, m_d3d = m_coord.match_to_catalog_sky( age_coord )
r_idx, r_sep, r_d3d = rich_coord.match_to_catalog_sky( age_coord )

id_lim_0 = m_sep.value < 2.7e-4
id_lim_1 = r_sep.value < 2.7e-4

m_dex = np.where( id_lim_0 == False )[0][0]
diff_ra = m_ra[ m_dex ]
diff_dec = m_dec[ m_dex ]

# #... check the diffi is in ZLWen catalog or not
# hi_tag = pds.read_csv('/home/xkchen/figs/ZLW_cat_15/redMapper_matched_high_BCG_star-Mass.csv')
# hi_tag_ra, hi_tag_dec = np.array( hi_tag['ra'] ), np.array( hi_tag['dec'] )
# hi_tag_order = np.array( hi_tag['order'] )

# low_tag = pds.read_csv('/home/xkchen/figs/ZLW_cat_15/redMapper_matched_low_BCG_star-Mass.csv')
# low_tag_ra, low_tag_dec = np.array( low_tag['ra'] ), np.array( low_tag['dec'] )
# low_tag_order = np.array( low_tag['order'] )

# tag_coord_0 = SkyCoord( ra = hi_tag_ra*U.deg, dec = hi_tag_dec*U.deg,)
# tag_coord_1 = SkyCoord( ra = low_tag_ra*U.deg, dec = low_tag_dec*U.deg,)

# # idx, d2d, d3d = m_coord[ m_dex ].match_to_catalog_sky( tag_coord_0 )
# idx, d2d, d3d = m_coord[ m_dex ].match_to_catalog_sky( tag_coord_1 )
# _id_tag = d2d.value < 2.7e-4

#. member properties matching for the lack samples in redMaPPer
with h5py.File('/home/xkchen/figs/ZLW_cat_15/low_BCG_star-Mass_clus-sat_record.h5', 'r') as f:
	sub_arr = f["/clust_%d/arr/" % m_dex ][()]
	sub_IDs = f["/clust_%d/IDs/" % m_dex ][()]

sat_ra, sat_dec, sat_z = sub_arr[0], sub_arr[1], sub_arr[2]
sat_dL, sat_Pmem = sub_arr[3], sub_arr[4]
order_x = np.arange(0, len(sub_IDs) )

# put_arr = np.array([ order_x, sat_ra, sat_dec ]).T
# np.savetxt('/home/xkchen/figs/ZLW_cat_15/lo_M_check.dat', put_arr, fmt = ('%.0f', '%.5f', '%.5f'), )

sql_dat = pds.read_csv('/home/xkchen/figs/ZLW_cat_15/lo_M_check_sat_mag.csv', skiprows = 1)
sql_IDs, sql_ra, sql_dec = np.array( sql_dat['objID'] ), np.array( sql_dat['ra'] ), np.array( sql_dat['dec'] )
r_mod_mag, g_mod_mag, i_mod_mag = np.array( sql_dat['modelMag_r'] ), np.array( sql_dat['modelMag_g'] ), np.array( sql_dat['modelMag_i'] )
r_dered_mag, g_dered_mag, i_dered_mag = np.array( sql_dat['dered_r'] ), np.array( sql_dat['dered_g'] ), np.array( sql_dat['dered_i'] )

ra_g, dec_g, z_g = lo_m_ra[m_dex], lo_m_dec[m_dex], lo_m_z[m_dex]

sub_r_mag_err = np.ones( len(sub_IDs), ) * 0.5
sub_g_mag_err = np.ones( len(sub_IDs), ) * 0.5
sub_i_mag_err = np.ones( len(sub_IDs), ) * 0.5

keys = [ 'centric_R(Mpc/h)', 'r_mags', 'g_mags', 'i_mags', 'P_member', 'r_mag_err', 'g_mag_err', 'i_mag_err', 
			'dered_r_mags', 'dered_g_mags', 'dered_i_mags', 'ra', 'dec', 'z']
values = [ sat_dL, r_mod_mag, g_mod_mag, i_mod_mag, sat_Pmem, sub_r_mag_err, sub_g_mag_err, sub_i_mag_err, 
			r_dered_mag, g_dered_mag, i_dered_mag, sat_ra, sat_dec, sat_z ]
fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )
out_data.to_csv('/home/xkchen/photo-z_%s-band_ra%.3f_dec%.3f_z%.3f_members_mag.csv' % ('r', ra_g, dec_g, z_g), )

raise
"""


###===### ZLWen matched catalog compare
'''
hi_dat = pds.read_csv( '/home/xkchen/figs/ZLW_cat_15/redMapper_matched_high-age.csv' )
hi_age_ra, hi_age_dec, hi_age_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )

lo_dat = pds.read_csv( '/home/xkchen/figs/ZLW_cat_15/redMapper_matched_low-age.csv' )
lo_age_ra, lo_age_dec, lo_age_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )

###... bcg mass binned sample
hi_dat = pds.read_csv( '/home/xkchen/figs/ZLW_cat_15/redMapper_matched_high_BCG_star-Mass.csv' )
hi_m_ra, hi_m_dec, hi_m_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )

lo_dat = pds.read_csv( '/home/xkchen/figs/ZLW_cat_15/redMapper_matched_low_BCG_star-Mass.csv' )
lo_m_ra, lo_m_dec, lo_m_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )

###... richness binned sample
hi_dat = pds.read_csv( '/home/xkchen/figs/ZLW_cat_15/redMapper_matched_high-rich.csv' )
hi_ra, hi_dec, hi_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )

lo_dat = pds.read_csv( '/home/xkchen/figs/ZLW_cat_15/redMapper_matched_low-rich.csv' )
lo_ra, lo_dec, lo_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )

###... different sample compared to t_age sample
age_ra = np.r_[ lo_age_ra, hi_age_ra ]
age_dec = np.r_[ lo_age_dec, hi_age_dec ]

m_ra = np.r_[ lo_m_ra, hi_m_ra ]
m_dec = np.r_[ lo_m_dec, hi_m_dec ]
m_z = np.r_[ lo_m_z, hi_m_z ]

rich_ra = np.r_[ lo_ra, hi_ra ]
rich_dec = np.r_[ lo_dec, hi_dec ]

Ns_t = len( age_ra )
Ns_m = len( m_ra )
Ns_r = len( rich_ra )

age_coord = SkyCoord( ra = age_ra*U.deg, dec = age_dec*U.deg,)
m_coord = SkyCoord( ra = m_ra*U.deg, dec = m_dec*U.deg,)
rich_coord = SkyCoord( ra = rich_ra*U.deg, dec = rich_dec*U.deg,)

m_idx, m_sep, m_d3d = m_coord.match_to_catalog_sky( age_coord )
r_idx, r_sep, r_d3d = rich_coord.match_to_catalog_sky( age_coord )

id_lim_0 = m_sep.value < 2.7e-4
id_lim_1 = r_sep.value < 2.7e-4

diff_ra_0, diff_dec_0 = m_ra[ id_lim_0 == False ], m_dec[ id_lim_0 == False ]
diff_ra_1, diff_dec_1 = rich_ra[ id_lim_1 == False ], rich_dec[ id_lim_1 == False ] 

#. save the different catalog
diff_ra, diff_dec, diff_z = m_ra[ id_lim_0 == False ], m_dec[ id_lim_0 == False ], m_z[ id_lim_0 == False ]

order_dex = np.where( id_lim_0 == False )[0]
dex_div = np.ones( len(order_dex),)

N_low = len( lo_m_ra )
id_div = order_dex >= N_low
dex_div[ id_div ] = 2

orin_dex = order_dex.copy()
orin_dex[ id_div ] = order_dex[ id_div ] - N_low

put_arr = np.array( [ diff_ra, diff_dec, diff_z, orin_dex, dex_div] ).T
np.savetxt( '/home/xkchen/figs/ZLW_cat_15/ZLW_age_bin_diffi_cat/age-bin_different_cat.dat', put_arr, 
			fmt = ('%.5f', '%.5f', '%.5f', '%.0f', '%.0f'), )

'''


###===### match age-bin different catalog to member galaxies
# different catalog information in redMaPPer
diffi_dat = np.loadtxt('/home/xkchen/figs/ZLW_cat_15/ZLW_age_bin_diffi_cat/age-bin_different_cat.dat')
diffi_ra, diffi_dec, diffi_z = diffi_dat[:,0], diffi_dat[:,1], diffi_dat[:,2]
diffi_samp_id, diffi_order = diffi_dat[:,4], diffi_dat[:,3]

#... overall ZLWen cluster cat
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
cen_dat = np.loadtxt('/home/xkchen/figs/ZLW_cat_15/ZLW_sat/WH15_chenxk_BCG.dat')

cen_ra, cen_dec = cen_dat[:,1], cen_dat[:,2]
orin_dex = cen_dat[:,0]
clus_z = cen_dat[:,3]
r_bcg_m = cen_dat[:,4]
R_500c = cen_dat[:,5] # Mpc
clus_rich = cen_dat[:,6]
N_sat = cen_dat[:,7]
clus_z_flag = cen_dat[:,8]
samp_id = cen_dat[:,9]

id_div = samp_id == 1

lo_ra = cen_ra[ id_div ]
lo_dec = cen_dec[ id_div ]
lo_z = clus_z[ id_div ]

# keys = [ 'ra', 'dec', 'z' ]
# values = [ lo_ra, lo_dec, lo_z ]
# fill = dict(zip( keys, values) )
# out_data = pds.DataFrame( fill )
# out_data.to_csv( '/home/xkchen/figs/sat_cat_ZLW/ZLWen_%s_match_cat.csv' % cat_lis[ 0 ] )


hi_ra = cen_ra[ id_div == False ]
hi_dec = cen_dec[ id_div == False ]
hi_z = clus_z[ id_div == False ]

# keys = [ 'ra', 'dec', 'z' ]
# values = [ hi_ra, hi_dec, hi_z ]
# fill = dict(zip( keys, values) )
# out_data = pds.DataFrame( fill )
# out_data.to_csv( '/home/xkchen/figs/sat_cat_ZLW/ZLWen_%s_match_cat.csv' % cat_lis[ 1 ] )


zlw_coord = SkyCoord( cen_ra * U.deg, cen_dec * U.deg,)
red_coord = SkyCoord( diffi_ra * U.deg, diffi_dec * U.deg,)

idx, d2d, d3d = red_coord.match_to_catalog_sky( zlw_coord )
id_lim = d2d.value < 2.7e-4

lim_ra, lim_dec, lim_z = cen_ra[ idx[ id_lim ] ], cen_dec[ idx[ id_lim ] ], clus_z[ idx[ id_lim ] ]
lim_orin_dex, lim_samp_id = orin_dex[ idx[ id_lim ] ], samp_id[ idx[ id_lim ] ]

# put_arr = np.array( [ lim_ra, lim_dec, lim_z, lim_orin_dex, lim_samp_id] ).T
# np.savetxt('/home/xkchen/figs/ZLW_cat_15/ZLW_age_bin_diffi_cat/age-bin_different_ZLW-cat.dat', put_arr, 
# 			fmt = ('%.5f', '%.5f', '%.5f', '%.0f', '%.0f'),)


#... ZLWen member cat
mem_dat = np.loadtxt('/home/xkchen/figs/ZLW_cat_15/ZLW_sat/clustmem_chenxk_BCG.dat')
clus_id = mem_dat[:,0]
mem_ra, mem_dec, mem_z = mem_dat[:,1], mem_dat[:,2], mem_dat[:,3]
mem_rmag, mem_rMag = mem_dat[:,4], mem_dat[:,5]
cen_dL, mem_z_flag = mem_dat[:,6], mem_dat[:,7]

N_Low = len( lo_ra )
N_hig = len( hi_ra )
N_tot = N_Low + N_hig
N_rep_dex = np.min( [ N_Low, N_hig ] )

"""
#... pre-match
low_F_tree = h5py.File( '/home/xkchen/figs/ZLW_cat_15/ZLW_cat_%s_clus-sat_record.h5' % cat_lis[0], 'w')

for tt in range( N_Low ):

	ra_g, dec_g, z_g = cen_ra[ tt ], cen_dec[ tt ], clus_z[ tt ]

	if tt < N_rep_dex:
		cc_ra_g, cc_dec_g, cc_z_g = cen_ra[ tt + N_Low ], cen_dec[ tt + N_Low ], clus_z[ tt + N_Low ]

	else:
		cc_ra_g, cc_dec_g, cc_z_g = -1000, -1000, -1000

	tag_dex = clus_id == tt
	daa = np.where( tag_dex )[0]
	dev = np.diff( daa )

	id_rep = dev > 1

	if np.sum( id_rep ) > 0:

		mid_dex = np.where( id_rep )[0]

		_n_mid = len( mid_dex )
		_bond_x0 = daa[ mid_dex ]

		bond_dex = np.zeros( int(2 * (_n_mid + 1) ), dtype = int )

		bond_dex[::2] = np.r_[ daa[0], daa[ mid_dex + 1] ]
		bond_dex[1::2] = np.r_[ daa[mid_dex], daa[-1] ]

		_n_bond = len( bond_dex )

		#. record array
		sub_ra, sub_dec, sub_z = np.array([]), np.array([]), np.array([])
		sub_cen_R = np.array([])

		for pp in range(0, _n_bond -1, 2 ):

			pa0, pa1 = bond_dex[ pp ], bond_dex[ pp + 1 ] + 1

			mid_ra, mid_dec = mem_ra[pa0: pa1], mem_dec[ pa0 : pa1]
			mid_z = mem_z[pa0: pa1]
			mid_cen_dL = cen_dL[pa0: pa1]

			deta_ra_0 = np.mean( np.abs( mid_ra - ra_g ) )
			deta_ra_1 = np.mean( np.abs( mid_ra - cc_ra_g) )

			if deta_ra_0 < deta_ra_1:
				sub_ra = np.r_[ sub_ra, mid_ra ]
				sub_dec = np.r_[ sub_dec, mid_dec ]
				sub_z = np.r_[ sub_z, mid_z ]
				sub_cen_R = np.r_[ sub_cen_R, mid_cen_dL ]

			else:
				continue

		_sub_n = len( sub_cen_R )

	else:
		da0 = daa[ 0 ]
		da1 = daa[ -1 ] + 1
		_sub_n = len( cen_dL[ da0: da1 ] )

		sub_cen_R = cen_dL[ da0: da1 ]
		sub_ra, sub_dec, sub_z = mem_ra[ da0: da1 ], mem_dec[ da0: da1 ], mem_z[ da0: da1 ]

	out_arr = np.array( [ sub_ra, sub_dec, sub_z, sub_cen_R ] )
	gk = low_F_tree.create_group( "clust_%d/" % tt )
	dk0 = gk.create_dataset( "arr", data = out_arr )
low_F_tree.close()

print(' low sample matched')


hig_F_tree = h5py.File( '/home/xkchen/figs/ZLW_cat_15/ZLW_cat_%s_clus-sat_record.h5' % cat_lis[1], 'w')
tpx = []
for tt in range( N_Low, N_tot ):

	ra_g, dec_g, z_g = cen_ra[ tt ], cen_dec[ tt ], clus_z[ tt ]

	sub_tt = np.int( tt - N_Low )

	if sub_tt < N_rep_dex:
		cc_ra_g, cc_dec_g, cc_z_g = cen_ra[ sub_tt ], cen_dec[ sub_tt ], clus_z[ sub_tt ]

	else:
		cc_ra_g, cc_dec_g, cc_z_g = -1000, -1000, -1000

	tag_dex = clus_id == sub_tt

	daa = np.where( tag_dex )[0]
	dev = np.diff( daa )

	id_rep = dev > 1

	if np.sum( id_rep ) > 0:

		mid_dex = np.where( id_rep )[0]

		_n_mid = len( mid_dex )
		_bond_x0 = daa[ mid_dex ]

		bond_dex = np.zeros( int(2 * (_n_mid + 1) ), dtype = int )

		bond_dex[::2] = np.r_[ daa[0], daa[ mid_dex + 1] ]
		bond_dex[1::2] = np.r_[ daa[mid_dex], daa[-1] ]

		_n_bond = len( bond_dex )

		#. record array
		sub_ra, sub_dec, sub_z = np.array([]), np.array([]), np.array([])
		sub_cen_R = np.array([])

		for pp in range(0, _n_bond -1, 2 ):

			pa0, pa1 = bond_dex[ pp ], bond_dex[ pp + 1 ] + 1

			mid_ra, mid_dec = mem_ra[pa0: pa1], mem_dec[ pa0 : pa1]
			mid_z = mem_z[pa0: pa1]
			mid_cen_dL = cen_dL[pa0: pa1]

			deta_ra_0 = np.mean( np.abs( mid_ra - ra_g ) )
			deta_ra_1 = np.mean( np.abs( mid_ra - cc_ra_g) )

			if deta_ra_0 < deta_ra_1:
				sub_ra = np.r_[ sub_ra, mid_ra ]
				sub_dec = np.r_[ sub_dec, mid_dec ]
				sub_z = np.r_[ sub_z, mid_z ]
				sub_cen_R = np.r_[ sub_cen_R, mid_cen_dL ]

			else:
				continue

		_sub_n = len( sub_cen_R )

	else:
		da0 = daa[ 0 ]
		da1 = daa[ -1 ] + 1
		_sub_n = len( cen_dL[ da0: da1 ] )

		sub_cen_R = cen_dL[ da0: da1 ]
		sub_ra, sub_dec, sub_z = mem_ra[ da0: da1 ], mem_dec[ da0: da1 ], mem_z[ da0: da1 ]

	tpx.append( _sub_n )
	out_arr = np.array( [ sub_ra, sub_dec, sub_z, sub_cen_R ] )
	gk = hig_F_tree.create_group( "clust_%d/" % sub_tt )
	dk0 = gk.create_dataset( "arr", data = out_arr )
hig_F_tree.close()

print(' high sample matched')
"""

#... record the different catalog
sql_dat = np.loadtxt('/home/xkchen/figs/ZLW_cat_15/ZLW_age_bin_diffi_cat/age-bin_different_ZLW-cat.dat')
sql_ra, sql_dec, sql_z = sql_dat[:,0], sql_dat[:,1], sql_dat[:,2]
sql_samp_id, sql_order = sql_dat[:,4], sql_dat[:,3]

tmp_ra, tmp_dec = np.array( [] ), np.array( [] )
tmp_dex, tmp_div = np.array( [] ), np.array( [] )
N_diff = len( sql_ra )
_N_sat = []

for jj in range( N_diff ):

	if sql_samp_id[jj] == 1:
		with h5py.File( '/home/xkchen/figs/ZLW_cat_15/ZLW_cat_%s_clus-sat_record.h5' % cat_lis[0], 'r') as f:
			sub_arr = f["/clust_%d/arr/" % sql_order[jj] ][()]
		tmp_ra = np.r_[ tmp_ra, sub_arr[0] ]
		tmp_dec = np.r_[ tmp_dec, sub_arr[1] ]
		_n_tt_ = len( sub_arr[0] )
		tmp_dex = np.r_[ tmp_dex, np.ones( _n_tt_ ) * sql_order[jj] ]
		tmp_div = np.r_[ tmp_div, np.ones( _n_tt_ ) * 1 ]

	if sql_samp_id[jj] == 2:
		with h5py.File( '/home/xkchen/figs/ZLW_cat_15/ZLW_cat_%s_clus-sat_record.h5' % cat_lis[1], 'r') as f:
			sub_arr = f["/clust_%d/arr/" % sql_order[jj] ][()]
		tmp_ra = np.r_[ tmp_ra, sub_arr[0] ]
		tmp_dec = np.r_[ tmp_dec, sub_arr[1] ]
		_n_tt_ = len( sub_arr[0] )
		tmp_dex = np.r_[ tmp_dex, np.ones( _n_tt_ ) * sql_order[jj] ]
		tmp_div = np.r_[ tmp_div, np.ones( _n_tt_ ) * 2 ]

	if _n_tt_ > 1000:
		print( _n_tt_ )
		print( sql_samp_id[jj] )
		print( sql_order[jj] )

	_N_sat.append( _n_tt_ )

#. .dat files for SDSS table query
# put_arr = np.array( [ tmp_dex, tmp_ra, tmp_dec] ).T
# np.savetxt('/home/xkchen/figs/ZLW_cat_15/ZLW_age_bin_diffi_cat/ZLWen_age-diffi-cat_sat.dat', 
# 			put_arr, fmt = ('%.0f', '%.5f', '%.5f'),)

# put_arr = np.array( [ tmp_dex, tmp_div, tmp_ra, tmp_dec] ).T
# np.savetxt('/home/xkchen/figs/ZLW_cat_15/ZLW_age_bin_diffi_cat/ZLWen_age-diffi-cat_clus-sat_pair.dat', 
# 			put_arr, fmt = ('%.0f', '%.0f', '%.5f', '%.5f'),)


#... match the different catalog with SDSS matched member properties
#. combine ZLW-sat-cat with sdss sql table
# sat_dat = pds.read_csv('/home/xkchen/figs/ZLW_cat_15/ZLW_age_bin_diffi_cat/ZLWen_age-diff-cat_sat_mag.csv', skiprows = 1,)
# kk_str = sat_dat.columns
# sat_ra, sat_dec, sat_z = np.array( sat_dat['ra'] ), np.array( sat_dat['dec'] ), np.array( sat_dat['z'] )

# mem_coord = SkyCoord( mem_ra * U.deg, mem_dec * U.deg )
# sat_coord = SkyCoord( sat_ra * U.deg, sat_dec * U.deg )

# sql_idx, sql_sep, sql_d3d = sat_coord.match_to_catalog_sky( mem_coord )

# id_lim = sql_sep.value < 2.7e-4

# keys = ['clust_id', 'cat_ra', 'cat_dec', 'cat_z', 'cat_r_mag', 'cat_r_Mag', 'centric_R', 'z_flag' ]
# sat_out_arr = [ clus_id[ sql_idx[id_lim] ], mem_ra[ sql_idx[id_lim] ], mem_dec[ sql_idx[id_lim] ], mem_z[ sql_idx[id_lim] ], 
# 				mem_rmag[ sql_idx[id_lim] ], mem_rMag[ sql_idx[id_lim] ], cen_dL[ sql_idx[id_lim] ], mem_z_flag[ sql_idx[id_lim] ] ]

# for jj in range( len(kk_str) ):

# 	keys.append( kk_str[ jj ] )
# 	sat_out_arr.append( sat_dat[ '%s' % kk_str[jj] ] )

# fill = dict( zip( keys, sat_out_arr ) )
# out_data = pds.DataFrame( fill )
# out_data.to_csv( '/home/xkchen/figs/ZLW_cat_15/ZLW_age_bin_diffi_cat/' + 'ZLWen_age-diff-cat_sat_sql_match.csv' )


sat_dat = pds.read_csv( '/home/xkchen/figs/ZLW_cat_15/ZLW_age_bin_diffi_cat/' + 'ZLWen_age-diff-cat_sat_sql_match.csv' )
sat_ra, sat_dec, sat_z = np.array( sat_dat['cat_ra'] ), np.array( sat_dat['cat_dec'] ), np.array( sat_dat['cat_z'] )
sat_zf, centric_L = np.array( sat_dat['z_flag'] ), np.array( sat_dat['centric_R'] )
sat_r_mag, sat_g_mag, sat_i_mag = np.array( sat_dat['modelMag_r'] ), np.array( sat_dat['modelMag_g'] ), np.array( sat_dat['modelMag_i'] )
sat_dered_rmag, sat_dered_gmag, sat_dered_imag = [ np.array( sat_dat['dered_r'] ), np.array( sat_dat['dered_g'] ), 
													np.array( sat_dat['dered_i'] ) ]
sat_rMag, sat_gMag, sat_iMag = np.array( sat_dat['absMagR'] ), np.array( sat_dat['absMagG'] ), np.array( sat_dat['absMagI'] )
sat_Lumi_r = absMag_to_Lumi_func( sat_rMag, 'r' )
sat_Lumi_g = absMag_to_Lumi_func( sat_gMag, 'g' )
sat_Lumi_i = absMag_to_Lumi_func( sat_iMag, 'i' )

cumu_N = np.cumsum( _N_sat )
cumu_N = np.r_[ 0, cumu_N ]

for jj in range( N_diff ):

	ra_g, dec_g, z_g = sql_ra[jj], sql_dec[jj], sql_z[jj]

	if sql_samp_id[jj] == 1:
		with h5py.File( '/home/xkchen/figs/ZLW_cat_15/ZLW_cat_%s_clus-sat_record.h5' % cat_lis[0], 'r') as f:
			sub_arr = f["/clust_%d/arr/" % sql_order[jj] ][()]
		
		sub_ra, sub_dec, sub_z = sub_arr[0], sub_arr[1], sub_arr[2]
		sub_cen_R = sub_arr[3]

	if sql_samp_id[jj] == 2:
		with h5py.File( '/home/xkchen/figs/ZLW_cat_15/ZLW_cat_%s_clus-sat_record.h5' % cat_lis[1], 'r') as f:
			sub_arr = f["/clust_%d/arr/" % sql_order[jj] ][()]

		sub_ra, sub_dec, sub_z = sub_arr[0], sub_arr[1], sub_arr[2]
		sub_cen_R = sub_arr[3]

	sub_r_mag = sat_r_mag[ cumu_N[jj] : cumu_N[jj+1] ]
	sub_g_mag = sat_g_mag[ cumu_N[jj] : cumu_N[jj+1] ]
	sub_i_mag = sat_i_mag[ cumu_N[jj] : cumu_N[jj+1] ]

	sud_dered_rmag = sat_dered_rmag[ cumu_N[jj] : cumu_N[jj+1] ]
	sud_dered_gmag = sat_dered_gmag[ cumu_N[jj] : cumu_N[jj+1] ]
	sud_dered_imag = sat_dered_imag[ cumu_N[jj] : cumu_N[jj+1] ]

	sub_Lumi_r = sat_Lumi_r[ cumu_N[jj] : cumu_N[jj+1] ]
	sub_Lumi_g = sat_Lumi_g[ cumu_N[jj] : cumu_N[jj+1] ]
	sub_Lumi_i = sat_Lumi_i[ cumu_N[jj] : cumu_N[jj+1] ]

	_sub_n = len( sub_ra )

	sub_Pmem = np.ones( _sub_n, )
	sub_clus_z_arr = np.ones( _sub_n ) * z_g

	sub_r_mag_err = np.ones( _sub_n, ) * 0.5
	sub_g_mag_err = np.ones( _sub_n, ) * 0.5
	sub_i_mag_err = np.ones( _sub_n, ) * 0.5

	keys = [ 'centric_R(Mpc/h)', 'r_mags', 'g_mags', 'i_mags', 'P_member', 'r_mag_err', 'g_mag_err', 'i_mag_err', 
				'dered_r_mags', 'dered_g_mags', 'dered_i_mags', 'ra', 'dec', 'z', 'clus_z', 'L_r', 'L_g', 'L_i' ]
	values = [ sub_cen_R, sub_r_mag, sub_g_mag, sub_i_mag, sub_Pmem, sub_r_mag_err, sub_g_mag_err, sub_i_mag_err, 
				sud_dered_rmag, sud_dered_gmag, sud_dered_imag, sub_ra, sub_dec, sub_z, sub_clus_z_arr, 
				sub_Lumi_r, sub_Lumi_g, sub_Lumi_i ]

	fill = dict( zip( keys, values ) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/figs/sat_cat_ZLW/mem_match/' + 
		'ZLW_%s-band_ra%.3f_dec%.3f_z%.3f_members_mag.csv' % ('r', ra_g, dec_g, z_g), )

print('done!')

raise

###===### rich-bin sample division and member galaxy match
cat_lis = [ 'low-rich', 'hi-rich' ]

cen_dat = np.loadtxt('/home/xkchen/figs/ZLW_cat_15/ZLW_sat/cluster_chenxk_rich.dat')
cen_ra, cen_dec = cen_dat[:,1], cen_dat[:,2]
orin_dex = cen_dat[:,0]
clus_z = cen_dat[:,3]
r_bcg_m = cen_dat[:,4] 
z_spec = cen_dat[:,5]
samp_id = cen_dat[:,6]

_id_nul = clus_z < 0
clus_z[ _id_nul ] = z_spec[ _id_nul ]

# id_div = samp_id == 1

# lo_rich_ra = cen_ra[ id_div ]
# lo_rich_dec = cen_dec[ id_div ]
# lo_rich_z = clus_z[ id_div ]

# hi_rich_ra = cen_ra[ id_div == False ]
# hi_rich_dec = cen_dec[ id_div == False ]
# hi_rich_z = clus_z[ id_div == False ]

# keys = [ 'ra', 'dec', 'z' ]
# values = [ lo_rich_ra, lo_rich_dec, lo_rich_z ]
# fill = dict(zip( keys, values) )
# out_data = pds.DataFrame( fill )
# out_data.to_csv( '/home/xkchen/figs/sat_cat_ZLW/ZLWen_%s_match_cat.csv' % cat_lis[ 0 ] )

# keys = [ 'ra', 'dec', 'z' ]
# values = [ hi_rich_ra, hi_rich_dec, hi_rich_z ]
# fill = dict(zip( keys, values) )
# out_data = pds.DataFrame( fill )
# out_data.to_csv( '/home/xkchen/figs/sat_cat_ZLW/ZLWen_%s_match_cat.csv' % cat_lis[ 1 ] )


## number check
id_div = samp_id == 1

N_Low = np.sum( id_div )
N_hig = np.sum( id_div == False )
N_tot = N_Low + N_hig
N_rep_dex = np.min( [ N_Low, N_hig ] )

mem_dat = np.loadtxt('/home/xkchen/figs/ZLW_cat_15/ZLW_sat/clustmem_chenxk_rich.dat')
clus_id = mem_dat[:,0]
mem_ra, mem_dec, mem_z = mem_dat[:,1], mem_dat[:,2], mem_dat[:,3]
cen_dL = mem_dat[:,6]

#... pre-match
low_F_tree = h5py.File( '/home/xkchen/figs/ZLW_cat_15/ZLW_cat_%s_clus-sat_record.h5' % cat_lis[0], 'w')
dpx_0 = []
for tt in range( N_Low ):

	ra_g, dec_g, z_g = cen_ra[ tt ], cen_dec[ tt ], clus_z[ tt ]

	if tt < N_rep_dex:
		cc_ra_g, cc_dec_g, cc_z_g = cen_ra[ tt + N_Low ], cen_dec[ tt + N_Low ], clus_z[ tt + N_Low ]

	else:
		cc_ra_g, cc_dec_g, cc_z_g = -1000, -1000, -1000

	tag_dex = clus_id == tt
	daa = np.where( tag_dex )[0]
	dev = np.diff( daa )

	id_rep = dev > 1

	if np.sum( id_rep ) > 0:

		mid_dex = np.where( id_rep )[0]

		_n_mid = len( mid_dex )
		_bond_x0 = daa[ mid_dex ]

		bond_dex = np.zeros( int(2 * (_n_mid + 1) ), dtype = int )

		bond_dex[::2] = np.r_[ daa[0], daa[ mid_dex + 1] ]
		bond_dex[1::2] = np.r_[ daa[mid_dex], daa[-1] ]

		_n_bond = len( bond_dex )

		#. record array
		sub_ra, sub_dec, sub_z = np.array([]), np.array([]), np.array([])
		sub_cen_R = np.array([])

		for pp in range(0, _n_bond -1, 2 ):

			pa0, pa1 = bond_dex[ pp ], bond_dex[ pp + 1 ] + 1

			mid_ra, mid_dec = mem_ra[pa0: pa1], mem_dec[ pa0 : pa1]
			mid_z = mem_z[pa0: pa1]
			mid_cen_dL = cen_dL[pa0: pa1]

			deta_ra_0 = np.mean( np.abs( mid_ra - ra_g ) )
			deta_ra_1 = np.mean( np.abs( mid_ra - cc_ra_g) )

			if deta_ra_0 < deta_ra_1:
				sub_ra = np.r_[ sub_ra, mid_ra ]
				sub_dec = np.r_[ sub_dec, mid_dec ]
				sub_z = np.r_[ sub_z, mid_z ]
				sub_cen_R = np.r_[ sub_cen_R, mid_cen_dL ]

			else:
				continue

		_sub_n = len( sub_cen_R )

	else:
		da0 = daa[ 0 ]
		da1 = daa[ -1 ] + 1
		_sub_n = len( cen_dL[ da0: da1 ] )

		sub_cen_R = cen_dL[ da0: da1 ]
		sub_ra, sub_dec, sub_z = mem_ra[ da0: da1 ], mem_dec[ da0: da1 ], mem_z[ da0: da1 ]

	dpx_0.append( _sub_n )

	out_arr = np.array( [ sub_ra, sub_dec, sub_z, sub_cen_R ] )
	gk = low_F_tree.create_group( "clust_%d/" % tt )
	dk0 = gk.create_dataset( "arr", data = out_arr )
low_F_tree.close()

print(' low sample matched')


hig_F_tree = h5py.File( '/home/xkchen/figs/ZLW_cat_15/ZLW_cat_%s_clus-sat_record.h5' % cat_lis[1], 'w')
dpx_1 = []
for tt in range( N_Low, N_tot ):

	ra_g, dec_g, z_g = cen_ra[ tt ], cen_dec[ tt ], clus_z[ tt ]

	sub_tt = np.int( tt - N_Low )

	if sub_tt < N_rep_dex:
		cc_ra_g, cc_dec_g, cc_z_g = cen_ra[ sub_tt ], cen_dec[ sub_tt ], clus_z[ sub_tt ]

	else:
		cc_ra_g, cc_dec_g, cc_z_g = -1000, -1000, -1000

	tag_dex = clus_id == sub_tt

	daa = np.where( tag_dex )[0]
	dev = np.diff( daa )

	id_rep = dev > 1

	if np.sum( id_rep ) > 0:

		mid_dex = np.where( id_rep )[0]

		_n_mid = len( mid_dex )
		_bond_x0 = daa[ mid_dex ]

		bond_dex = np.zeros( int(2 * (_n_mid + 1) ), dtype = int )

		bond_dex[::2] = np.r_[ daa[0], daa[ mid_dex + 1] ]
		bond_dex[1::2] = np.r_[ daa[mid_dex], daa[-1] ]

		_n_bond = len( bond_dex )

		#. record array
		sub_ra, sub_dec, sub_z = np.array([]), np.array([]), np.array([])
		sub_cen_R = np.array([])

		for pp in range(0, _n_bond -1, 2 ):

			pa0, pa1 = bond_dex[ pp ], bond_dex[ pp + 1 ] + 1

			mid_ra, mid_dec = mem_ra[pa0: pa1], mem_dec[ pa0 : pa1]
			mid_z = mem_z[pa0: pa1]
			mid_cen_dL = cen_dL[pa0: pa1]

			deta_ra_0 = np.mean( np.abs( mid_ra - ra_g ) )
			deta_ra_1 = np.mean( np.abs( mid_ra - cc_ra_g) )

			if deta_ra_0 < deta_ra_1:
				sub_ra = np.r_[ sub_ra, mid_ra ]
				sub_dec = np.r_[ sub_dec, mid_dec ]
				sub_z = np.r_[ sub_z, mid_z ]
				sub_cen_R = np.r_[ sub_cen_R, mid_cen_dL ]

			else:
				continue

		_sub_n = len( sub_cen_R )

	else:
		da0 = daa[ 0 ]
		da1 = daa[ -1 ] + 1
		_sub_n = len( cen_dL[ da0: da1 ] )

		sub_cen_R = cen_dL[ da0: da1 ]
		sub_ra, sub_dec, sub_z = mem_ra[ da0: da1 ], mem_dec[ da0: da1 ], mem_z[ da0: da1 ]

	dpx_1.append( _sub_n )

	out_arr = np.array( [ sub_ra, sub_dec, sub_z, sub_cen_R ] )
	gk = hig_F_tree.create_group( "clust_%d/" % sub_tt )
	dk0 = gk.create_dataset( "arr", data = out_arr )
hig_F_tree.close()

print(' high sample matched')

