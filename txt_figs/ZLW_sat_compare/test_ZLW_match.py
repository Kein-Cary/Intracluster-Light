"""
This file match overlap clusters in ZLWen catalog with their identified member galaxies
"""
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
import scipy.signal as signal

from astropy import cosmology as apcy
from scipy import interpolate as interp
from scipy import integrate as integ
from astropy.coordinates import SkyCoord
from fig_out_module import absMag_to_Lumi_func

### === ### cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
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

"""
###.....### Z.L. Wen et al. 2012 cluster catalog
def ZLW12_cat_read_func( cat_file, skip_row):

	fi = open( cat_file )
	lines = fi.readlines()
	fi.close()

	Nl = len( lines )

	#. bcg information
	tt_ra = np.zeros( Nl - skip_row,)
	tt_dec = np.zeros( Nl - skip_row,)
	tt_zs = np.zeros( Nl - skip_row,)
	tt_mr_bcg = np.zeros( Nl - skip_row,) # r-band magnitude

	#. cluster information
	tt_clus_zp = np.zeros( Nl - skip_row,)

	tt_R_200 = np.zeros( Nl - skip_row,)
	tt_rich_200 = np.zeros( Nl - skip_row,)
	tt_Ng_200 = np.zeros( Nl - skip_row,)

	tt_cat_name = []

	for ii in range( skip_row, Nl ):

		sub_ii = ii - skip_row
		sub_str = lines[ ii ].split()

		tt_ra[ sub_ii ] = np.float( sub_str[2] )
		tt_dec[ sub_ii ] = np.float( sub_str[3] ) 

		tt_zs[ sub_ii ] = np.float( sub_str[5] )
		tt_mr_bcg[ sub_ii ] = np.float( sub_str[6] )
		tt_clus_zp[ sub_ii ] = np.float( sub_str[4] )

		tt_R_200[ sub_ii ] = np.float( sub_str[7] )
		tt_rich_200[ sub_ii ] = np.float( sub_str[8] )
		tt_Ng_200[ sub_ii ] = np.float( sub_str[9] )

		tt_cat_name.append( sub_str[0] + sub_str[1] )

	tt_cat_name = np.array( tt_cat_name )

	#. this parameter need to query in ZLWen et al. 2015
	tt_R_500 = np.zeros( Nl - skip_row,)
	tt_rich_500 = np.zeros( Nl - skip_row,)
	tt_Ng_500 = np.zeros( Nl - skip_row,)

	return_arr = [ tt_ra, tt_dec, tt_zs, tt_mr_bcg, tt_clus_zp, 
				tt_R_200, tt_rich_200, tt_Ng_200, tt_R_500, tt_rich_500, tt_Ng_500, tt_cat_name ]

	return return_arr

def ZLW_cat_read_func( cat_file, skip_row):

	fi = open( cat_file )
	lines = fi.readlines()
	fi.close()

	Nl = len( lines )

	#. bcg information
	tt_ra = np.zeros( Nl - skip_row,)
	tt_dec = np.zeros( Nl - skip_row,)
	tt_zs = np.zeros( Nl - skip_row,)
	tt_mr_bcg = np.zeros( Nl - skip_row,) # r-band magnitude

	#. cluster information
	tt_clus_zp = np.zeros( Nl - skip_row,)

	tt_R_200 = np.zeros( Nl - skip_row,)
	tt_rich_200 = np.zeros( Nl - skip_row,)
	tt_Ng_200 = np.zeros( Nl - skip_row,)

	tt_R_500 = np.zeros( Nl - skip_row,)
	tt_rich_500 = np.zeros( Nl - skip_row,)
	tt_Ng_500 = np.zeros( Nl - skip_row,)

	tt_cat_name = []

	for ii in range( skip_row, Nl ):

		sub_ii = ii - skip_row
		sub_str = lines[ ii ].split()

		tt_ra[ sub_ii ] = np.float( sub_str[2] )
		tt_dec[ sub_ii ] = np.float( sub_str[3] ) 

		tt_zs[ sub_ii ] = np.float( sub_str[5] )
		tt_mr_bcg[ sub_ii ] = np.float( sub_str[6] )
		tt_clus_zp[ sub_ii ] = np.float( sub_str[4] )

		tt_R_200[ sub_ii ] = np.float( sub_str[7] )
		tt_rich_200[ sub_ii ] = np.float( sub_str[8] )
		tt_Ng_200[ sub_ii ] = np.float( sub_str[9] )

		tt_R_500[ sub_ii ] = np.float( sub_str[10] )
		tt_rich_500[ sub_ii ] = np.float( sub_str[11] )
		tt_Ng_500[ sub_ii ] = np.float( sub_str[13] )		

		tt_cat_name.append( sub_str[0] + sub_str[1] )

	tt_cat_name = np.array( tt_cat_name )

	return_arr = [ tt_ra, tt_dec, tt_zs, tt_mr_bcg, tt_clus_zp, 
				tt_R_200, tt_rich_200, tt_Ng_200, tt_R_500, tt_rich_500, tt_Ng_500, tt_cat_name ]

	return return_arr


wen_file = '/home/xkchen/figs/ZLW_cat_12/apjs425031t1_mrt.txt' # Wen et al. 2012
bcg_ra, bcg_dec, bcg_zs, mr_bcg, clus_zp, R_200, rich_200, Ng_200, R_500, rich_500, Ng_500, cat_name = ZLW12_cat_read_func( wen_file, 30)

# wen_file = '/home/xkchen/figs/ZLW_cat_15/apj515105t3_mrt.txt' # Wen et al. 2015
# wen_file = '/home/xkchen/figs/ZLW_cat_15/apj515105t4_mrt.txt'
# bcg_ra, bcg_dec, bcg_zs, mr_bcg, clus_zp, R_200, rich_200, Ng_200, R_500, rich_500, Ng_500, cat_name = ZLW_cat_read_func( wen_file, 26)

sql_coord = SkyCoord( bcg_ra, bcg_dec, unit = 'deg')

###... match with our catalog.
hi_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/' + 
						'age_bin/hi-age_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv')

# hi_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/' + 
# 						'rich_bin/hi-rich_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv')

# hi_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_gri_common/' + 
# 						'high_BCG_star-Mass_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv')

hi_ra, hi_dec, hi_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )


lo_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/' + 
						'age_bin/low-age_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv')

# lo_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/' + 
# 						'rich_bin/low-rich_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv')

# lo_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_gri_common/' + 
# 						'low_BCG_star-Mass_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv')

lo_ra, lo_dec, lo_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )


lo_mp_coord = SkyCoord( lo_ra, lo_dec, unit = 'deg')
hi_mp_coord = SkyCoord( hi_ra, hi_dec, unit = 'deg')

idx_l, d2d_l, d3d_l = lo_mp_coord.match_to_catalog_sky( sql_coord )
dex_lim_l = d2d_l.value <= 2.6e-4 # within 1 arcsec

idx_h, d2d_h, d3d_h = hi_mp_coord.match_to_catalog_sky( sql_coord )
dex_lim_h = d2d_h.value <= 2.6e-4 # within 1 arcsec

##... position check
plt.figure()
# plt.plot( lo_ra, lo_dec, 'ro', alpha = 0.5)
plt.plot( lo_ra[dex_lim_l], lo_dec[dex_lim_l], 'ro', alpha = 0.5)
plt.plot( bcg_ra[ idx_l[ dex_lim_l ] ], bcg_dec[ idx_l[ dex_lim_l ] ], 'g*', alpha = 0.5,)

# plt.plot( hi_ra, hi_dec, 'ro', alpha = 0.5)
# plt.plot( bcg_ra[ idx_h[ dex_lim_h ] ], bcg_dec[ idx_h[ dex_lim_h ] ], 'g*', alpha = 0.5,)
plt.show()

#. matched ZLWen catalog information
keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'bcg_Mr', 'clus_zp', 'R200c', 'rich', 'N_mem', 'clust_name']

lo_mat_arr = [ bcg_ra[ idx_l[ dex_lim_l ] ], bcg_dec[ idx_l[ dex_lim_l ] ], bcg_zs[ idx_l[ dex_lim_l ] ], mr_bcg[ idx_l[ dex_lim_l ] ], 
				clus_zp[ idx_l[ dex_lim_l ] ], R_200[ idx_l[ dex_lim_l ] ], rich_200[ idx_l[ dex_lim_l ] ], Ng_200[ idx_l[ dex_lim_l ] ],
				cat_name[ idx_l[ dex_lim_l ] ] ]

fill = dict( zip( keys, lo_mat_arr ) )

out_data = pds.DataFrame( fill )
out_data.to_csv('/home/xkchen/figs/ZLW_clust_match_low-age.csv')
# out_data.to_csv('/home/xkchen/figs/ZLW_clust_match_low-rich.csv')
# out_data.to_csv('/home/xkchen/figs/ZLW_clust_match_low_BCG_star-Mass.csv')

keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'bcg_Mr', 'clus_zp', 'R200c', 'rich', 'N_mem', 'clust_name']

hi_mat_arr = [ bcg_ra[ idx_h[ dex_lim_h ] ], bcg_dec[ idx_h[ dex_lim_h ] ], bcg_zs[ idx_h[ dex_lim_h ] ], mr_bcg[ idx_h[ dex_lim_h ] ], 
				clus_zp[ idx_h[ dex_lim_h ] ], R_200[ idx_h[ dex_lim_h ] ], rich_200[ idx_h[ dex_lim_h ] ], Ng_200[ idx_h[ dex_lim_h ] ], 
				cat_name[ idx_h[ dex_lim_h ] ] ]

fill = dict( zip( keys, hi_mat_arr ) )
out_data = pds.DataFrame( fill )
out_data.to_csv('/home/xkchen/figs/ZLW_clust_match_high-age.csv')
# out_data.to_csv('/home/xkchen/figs/ZLW_clust_match_high-rich.csv')
# out_data.to_csv('/home/xkchen/figs/ZLW_clust_match_high_BCG_star-Mass.csv')

#. matched SDSS redMaPPer catalog information
keys = ['ra', 'dec', 'z', 'order']
values = [ lo_ra[dex_lim_l], lo_dec[dex_lim_l], lo_z[dex_lim_l], np.where(dex_lim_l)[0] ]
fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )
out_data.to_csv('/home/xkchen/figs/redMapper_matched_low-age.csv')
# out_data.to_csv('/home/xkchen/figs/redMapper_matched_low-rich.csv')
# out_data.to_csv('/home/xkchen/figs/redMapper_matched_low_BCG_star-Mass.csv')

keys = ['ra', 'dec', 'z', 'order']
values = [ hi_ra[dex_lim_h], hi_dec[dex_lim_h], hi_z[dex_lim_h], np.where(dex_lim_h)[0] ]
fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )
out_data.to_csv('/home/xkchen/figs/redMapper_matched_high-age.csv')
# out_data.to_csv('/home/xkchen/figs/redMapper_matched_high-rich.csv')
# out_data.to_csv('/home/xkchen/figs/redMapper_matched_high_BCG_star-Mass.csv')

raise
"""

"""
###.....### update catalog in Z.L. Wen et al. 2015
#. central galaxies
cen_dat = np.loadtxt('/home/xkchen/figs/sat_cat_ZLW/WH15_chenxk.dat')
cen_ra, cen_dec = cen_dat[:,1], cen_dat[:,2]
orin_dex = cen_dat[:,0]
clus_z = cen_dat[:,3]
r_bcg_m = cen_dat[:,4]
R_500c = cen_dat[:,5] # Mpc
clus_rich = cen_dat[:,6]
N_sat = cen_dat[:,7]
clus_z_flag = cen_dat[:,8]
samp_id = cen_dat[:,9]

#. tmp table for mag query
# ordex_arr = np.arange(0, len(cen_ra) )
# tmp_arr = np.array( [ np.r_[ 0, ordex_arr], np.r_[1, cen_ra], np.r_[ 2, cen_dec ] ] ).T
# np.savetxt('/home/xkchen/cen_galx_cat.dat', tmp_arr, fmt = ('%.0f', '%.5f', '%.5f'),)

#. satellite galaxies
sat_dat = np.loadtxt('/home/xkchen/figs/sat_cat_ZLW/clustmem_chenxk.dat')
sat_ra, sat_dec = sat_dat[:,1], sat_dat[:,2]
clust_id = sat_dat[:,0]
sat_z = sat_dat[:,3]
sat_rmag = sat_dat[:,4]
sat_rMag = sat_dat[:,5]
cen_dL = sat_dat[:,6]
sat_z_flag = sat_dat[:,7]

Ns = len( sat_ra )
ordex_arr = np.arange( 0, Ns )

N_step = 10000

###... sample division for SDSS photometric table query
# for tt in range( 24 ):
# 	print( 'tt = ', tt)
# 	if tt == 23:
# 		da0 = np.int( tt * N_step )
# 		da1 = Ns
# 	else:
# 		da0 = np.int( tt * N_step )
# 		da1 = np.int( (tt + 1) * N_step )
#	#. tmp table for mag query
# 	tmp_arr = np.array( [ np.r_[0, ordex_arr[da0: da1] ], np.r_[ 1, sat_ra[da0: da1] ], np.r_[ 2, sat_dec[da0: da1] ] ] ).T
# 	np.savetxt('/home/xkchen/sat_galx_cat_%d.dat' % tt, tmp_arr, fmt = ('%.0f', '%.5f', '%.5f'),)


###... combine color catalog with position, centric distance
sql_cen_dat = pds.read_csv('/home/xkchen/figs/cen_gax_mag.csv', skiprows = 1)

#. Wen 2015 params + sql_table of BCGs
keys = [ 'clust_id', 'clus_ra', 'clus_dec', 'clus_z', 'bcg_r_mag', 'R500c', 'rich', 'N500c', 'clus_z_flag', 'samp_id', ]
cen_out_arr = [ orin_dex, cen_ra, cen_dec, clus_z, r_bcg_m, R_500c, clus_rich, N_sat, clus_z_flag, samp_id, ]

kk_str = sql_cen_dat.columns
for jj in range( len(kk_str) ):

	keys.append( kk_str[ jj ] )
	cen_out_arr.append( sql_cen_dat[ '%s' % kk_str[jj] ] )

fill = dict( zip( keys, cen_out_arr ) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/figs/clust_sql_match_cat.csv' )


#. satellite tables combination
keys = ['clust_id', 'cat_ra', 'cat_dec', 'cat_z', 'cat_r_mag', 'cat_r_Mag', 'centric_R', 'z_flag' ]
sat_out_arr = [ clust_id, sat_ra, sat_dec, sat_z, sat_rmag, sat_rMag, cen_dL, sat_z_flag ]

sat_sql_dat = pds.read_csv('/home/xkchen/figs/sat_gax_mag_0.csv', skiprows = 1)
kk_str = sat_sql_dat.columns

_ii_arr = []
_ii_dex = np.array( [] )

for jj in range( len(kk_str) ):

	_ii_arr.append( sat_sql_dat[ '%s' % kk_str[jj] ] )

_sub_dex = sat_sql_dat['col0']
_sub_diff = np.diff( _sub_dex )

idx = _sub_diff > 1
if np.sum( idx ) > 0:

	id_lim = np.where( idx )[0]
	_ii_dex = np.r_[ _ii_dex, id_lim ]

for tt in range( 1,24 ):

	tt_dat = pds.read_csv('/home/xkchen/figs/sat_gax_mag_%d.csv' % tt, skiprows = 1)

	for jj in range( len(kk_str) ):
		_ii_arr[jj] = np.r_[ _ii_arr[jj], tt_dat[ '%s' % kk_str[jj] ] ]

	_sub_dex = tt_dat['col0']
	_sub_diff = np.diff( _sub_dex )

	idx = _sub_diff > 1
	if np.sum( idx ) > 0:

		id_lim = np.where( idx )[0]
		_ii_dex = np.r_[ _ii_dex, id_lim ]

#. rule out no query matched satellites 
for jj in range( len(keys) ):
	sat_out_arr[jj] = np.delete( sat_out_arr[jj], _ii_dex )


for jj in range( len(kk_str) ):

	keys.append( kk_str[jj] )
	sat_out_arr.append( _ii_arr[jj] )

fill = dict( zip( keys, sat_out_arr ) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/figs/sat_sql_match_cat.csv' )

raise
"""

###...### sample properties match
samp_dex = [1, 2] # 1 -- low-t_age; 2 -- high-t_age
cat_lis = [ 'low-age', 'hi-age' ]

fig_name = ['Low $t_{ \\mathrm{age} } \\mid M_{\\ast}^{\\mathrm{BCG}}$', 
			'High $t_{ \\mathrm{age} } \\mid M_{\\ast}^{\\mathrm{BCG}}$']

clus_dat = pds.read_csv( '/home/xkchen/figs/clust_sql_match_cat.csv' )
orin_dex = np.array( clus_dat['clust_id'] )
clus_ra, clus_dec, clus_z = np.array( clus_dat['clus_ra'] ), np.array( clus_dat['clus_dec'] ), np.array( clus_dat['clus_z'] )
clus_R500, clus_rich = np.array( clus_dat['R500c'] ), np.array( clus_dat['rich'] )
clus_Ng, clus_zf, clus_div = np.array( clus_dat['N500c'] ), np.array( clus_dat['clus_z_flag'] ), np.array( clus_dat['samp_id'] )


sat_dat = pds.read_csv( '/home/xkchen/figs/sat_sql_match_cat.csv' )
clus_id = np.array( sat_dat['clust_id'] )
sat_ra, sat_dec, sat_z = np.array( sat_dat['cat_ra'] ), np.array( sat_dat['cat_dec'] ), np.array( sat_dat['cat_z'] )
sat_zf, centric_L = np.array( sat_dat['z_flag'] ), np.array( sat_dat['centric_R'] )

sat_r_mag, sat_g_mag, sat_i_mag = np.array( sat_dat['modelMag_r'] ), np.array( sat_dat['modelMag_g'] ), np.array( sat_dat['modelMag_i'] )
sat_dered_rmag, sat_dered_gmag, sat_dered_imag = [ np.array( sat_dat['dered_r'] ), np.array( sat_dat['dered_g'] ), 
													np.array( sat_dat['dered_i'] ) ]

sat_rMag, sat_gMag, sat_iMag = np.array( sat_dat['absMagR'] ), np.array( sat_dat['absMagG'] ), np.array( sat_dat['absMagI'] )
sat_Lumi_r = absMag_to_Lumi_func( sat_rMag, 'r' )
sat_Lumi_g = absMag_to_Lumi_func( sat_gMag, 'g' )
sat_Lumi_i = absMag_to_Lumi_func( sat_iMag, 'i' )


id_div = clus_div == 1

N_Low = np.sum( id_div)
N_hig = np.sum( id_div == False)
N_tot = N_Low + N_hig
N_rep_dex = np.min( [ N_Low, N_hig ] )

'''
#. member match and color record ( for the low-t_age subsample)
low_F_tree = h5py.File( '/home/xkchen/figs/sat_cat_ZLW/ZLW_cat_%s_clus-sat_record.h5' % cat_lis[0], 'w')

dpx = []
for tt in range( N_Low ):

	ra_g, dec_g, z_g = clus_ra[ tt ], clus_dec[ tt ], clus_z[ tt ]

	if tt < N_rep_dex:
		cc_ra_g, cc_dec_g, cc_z_g = clus_ra[ tt + N_Low ], clus_ra[ tt + N_Low ], clus_ra[ tt + N_Low ]

	else:
		cc_ra_g, cc_dec_g, cc_z_g = -1000, -1000, -1000

	tag_dex = clus_id == tt
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
		sub_cen_R = np.array([])
		sub_r_mag, sub_g_mag, sub_i_mag = np.array([]), np.array([]), np.array([])
		sud_dered_rmag, sud_dered_gmag, sud_dered_imag = np.array([]), np.array([]), np.array([])
		sub_Lumi_r, sub_Lumi_g, sub_Lumi_i = np.array([]), np.array([]), np.array([])

		for pp in range(0, _n_bond -1, 2 ):

			da0, da1 = bond_dex[ pp ], bond_dex[ pp + 1 ] + 1

			mid_ra, mid_dec = sat_ra[ da0 : da1], sat_dec[ da0 : da1]
			mid_z = sat_z[ da0 : da1]
			mid_cen_dL = centric_L[ da0 : da1]

			mid_rmag, mid_gmag, mid_imag = sat_r_mag[da0 : da1], sat_g_mag[da0 : da1], sat_i_mag[da0 : da1]
			mid_dered_rmag, mid_dered_gmag = sat_dered_rmag[da0 : da1], sat_dered_gmag[da0 : da1]
			mid_dered_imag = sat_dered_imag[da0 : da1]

			mid_Lumi_r = sat_Lumi_r[da0 : da1]
			mid_Lumi_g = sat_Lumi_g[da0 : da1]
			mid_Lumi_i = sat_Lumi_i[da0 : da1]

			deta_ra_0 = np.mean( np.abs( mid_ra - ra_g ) )
			deta_ra_1 = np.mean( np.abs( mid_ra - cc_ra_g) )

			if deta_ra_0 < deta_ra_1:
				sub_ra = np.r_[ sub_ra, mid_ra ]
				sub_dec = np.r_[ sub_dec, mid_dec ]
				sub_z = np.r_[ sub_z, mid_z ]
				sub_cen_R = np.r_[ sub_cen_R, mid_cen_dL ]
				
				sub_r_mag = np.r_[ sub_r_mag, mid_rmag ]
				sub_g_mag = np.r_[ sub_g_mag, mid_gmag ]
				sub_i_mag = np.r_[ sub_i_mag, mid_imag ]

				sud_dered_rmag = np.r_[ sud_dered_rmag, mid_dered_rmag ]
				sud_dered_gmag = np.r_[ sud_dered_gmag, mid_dered_gmag ]
				sud_dered_imag = np.r_[ sud_dered_imag, mid_dered_imag ]

				sub_Lumi_r = np.r_[ sub_Lumi_r, mid_Lumi_r ]
				sub_Lumi_g = np.r_[ sub_Lumi_g, mid_Lumi_g ]
				sub_Lumi_i = np.r_[ sub_Lumi_i, mid_Lumi_i ]

			else:
				continue

		_sub_n = len( sub_cen_R )

	else:
		da0 = daa[ 0 ]
		da1 = daa[ -1 ] + 1

		sub_cen_R = centric_L[ da0: da1 ] # save in name 'centric_R(Mpc/h)', but the unit is Mpc

		sub_r_mag = sat_r_mag[ da0: da1 ]
		sub_g_mag = sat_g_mag[ da0: da1 ]
		sub_i_mag = sat_i_mag[ da0: da1 ]

		sud_dered_rmag = sat_dered_rmag[ da0: da1 ]
		sud_dered_gmag = sat_dered_gmag[ da0: da1 ]
		sud_dered_imag = sat_dered_imag[ da0: da1 ]

		sub_ra, sub_dec, sub_z = sat_ra[ da0: da1 ], sat_dec[ da0: da1 ], sat_z[ da0: da1 ]
		sub_Lumi_r, sub_Lumi_g, sub_Lumi_i = sat_Lumi_r[ da0: da1 ], sat_Lumi_g[ da0: da1 ], sat_Lumi_i[ da0: da1 ]

		_sub_n = len( sub_ra )

	sub_Pmem = np.ones( _sub_n, )
	sub_r_mag_err = np.ones( _sub_n, ) * 0.5
	sub_g_mag_err = np.ones( _sub_n, ) * 0.5
	sub_i_mag_err = np.ones( _sub_n, ) * 0.5

	sub_clus_z_arr = np.ones( _sub_n ) * z_g

	dpx.append( _sub_n )

	keys = [ 'centric_R(Mpc/h)', 'r_mags', 'g_mags', 'i_mags', 'P_member', 'r_mag_err', 'g_mag_err', 'i_mag_err', 
				'dered_r_mags', 'dered_g_mags', 'dered_i_mags', 'ra', 'dec', 'z', 'clus_z', 'L_r', 'L_g', 'L_i' ]
	values = [ sub_cen_R, sub_r_mag, sub_g_mag, sub_i_mag, sub_Pmem, sub_r_mag_err, sub_g_mag_err, sub_i_mag_err, 
				sud_dered_rmag, sud_dered_gmag, sud_dered_imag, sub_ra, sub_dec, sub_z, sub_clus_z_arr, 
				sub_Lumi_r, sub_Lumi_g, sub_Lumi_i ]

	fill = dict( zip( keys, values ) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/figs/sat_cat_ZLW/mem_match/' + 
		'ZLW_%s-band_ra%.3f_dec%.3f_z%.3f_members_mag.csv' % ('r', ra_g, dec_g, z_g), )

	#... catalog match
	out_arr = np.array( [ sub_ra, sub_dec, sub_z, sub_cen_R, 
						sub_g_mag - sub_r_mag, sub_g_mag - sub_i_mag, sub_r_mag - sub_i_mag, 
						sud_dered_gmag - sud_dered_rmag, sud_dered_gmag - sud_dered_imag, 
						sud_dered_rmag - sud_dered_imag, 
						sub_Lumi_r, sub_Lumi_g, sub_Lumi_i ] )

	gk = low_F_tree.create_group( "clust_%d/" % tt )
	dk0 = gk.create_dataset( "arr", data = out_arr )
low_F_tree.close()

print(' low sample matched')
'''

#. member match and color record ( for the high-t_age subsample)
hig_F_tree = h5py.File( '/home/xkchen/figs/sat_cat_ZLW/ZLW_cat_%s_clus-sat_record.h5' % cat_lis[1], 'w')
dpx = []
for tt in range( N_Low, N_tot ):

	ra_g, dec_g, z_g = clus_ra[ tt ], clus_dec[ tt ], clus_z[ tt ]

	sub_tt = np.int( tt - N_Low )

	if sub_tt < N_rep_dex:
		cc_ra_g, cc_dec_g, cc_z_g = clus_ra[ sub_tt ], clus_dec[ sub_tt ], clus_z[ sub_tt ]

	else:
		cc_ra_g, cc_dec_g, cc_z_g = -1000, -1000, -1000

	tag_dex = clus_id == sub_tt

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
		sub_cen_R = np.array([])
		sub_r_mag, sub_g_mag, sub_i_mag = np.array([]), np.array([]), np.array([])
		sud_dered_rmag, sud_dered_gmag, sud_dered_imag = np.array([]), np.array([]), np.array([])
		sub_Lumi_r, sub_Lumi_g, sub_Lumi_i = np.array([]), np.array([]), np.array([])

		for pp in range(0, _n_bond -1, 2 ):

			da0, da1 = bond_dex[ pp ], bond_dex[ pp + 1 ] + 1

			mid_ra, mid_dec = sat_ra[ da0 : da1], sat_dec[ da0 : da1]
			mid_z = sat_z[ da0 : da1]
			mid_cen_dL = centric_L[ da0 : da1]

			mid_rmag, mid_gmag, mid_imag = sat_r_mag[da0 : da1], sat_g_mag[da0 : da1], sat_i_mag[da0 : da1]
			mid_dered_rmag, mid_dered_gmag = sat_dered_rmag[da0 : da1], sat_dered_gmag[da0 : da1]
			mid_dered_imag = sat_dered_imag[da0 : da1]

			mid_Lumi_r = sat_Lumi_r[da0 : da1]
			mid_Lumi_g = sat_Lumi_g[da0 : da1]
			mid_Lumi_i = sat_Lumi_i[da0 : da1]

			deta_ra_0 = np.mean( np.abs( mid_ra - ra_g ) )
			deta_ra_1 = np.mean( np.abs( mid_ra - cc_ra_g) )

			if deta_ra_0 < deta_ra_1:
				sub_ra = np.r_[ sub_ra, mid_ra ]
				sub_dec = np.r_[ sub_dec, mid_dec ]
				sub_z = np.r_[ sub_z, mid_z ]
				sub_cen_R = np.r_[ sub_cen_R, mid_cen_dL ]
				
				sub_r_mag = np.r_[ sub_r_mag, mid_rmag ]
				sub_g_mag = np.r_[ sub_g_mag, mid_gmag ]
				sub_i_mag = np.r_[ sub_i_mag, mid_imag ]

				sud_dered_rmag = np.r_[ sud_dered_rmag, mid_dered_rmag ]
				sud_dered_gmag = np.r_[ sud_dered_gmag, mid_dered_gmag ]
				sud_dered_imag = np.r_[ sud_dered_imag, mid_dered_imag ]

				sub_Lumi_r = np.r_[ sub_Lumi_r, mid_Lumi_r ]
				sub_Lumi_g = np.r_[ sub_Lumi_g, mid_Lumi_g ]
				sub_Lumi_i = np.r_[ sub_Lumi_i, mid_Lumi_i ]

			else:
				continue

		_sub_n = len( sub_cen_R )

	else:
		da0 = daa[ 0 ]
		da1 = daa[ -1 ] + 1

		sub_cen_R = centric_L[ da0: da1 ] # save in name 'centric_R(Mpc/h)', but the unit is Mpc

		sub_r_mag = sat_r_mag[ da0: da1 ]
		sub_g_mag = sat_g_mag[ da0: da1 ]
		sub_i_mag = sat_i_mag[ da0: da1 ]

		sud_dered_rmag = sat_dered_rmag[ da0: da1 ]
		sud_dered_gmag = sat_dered_gmag[ da0: da1 ]
		sud_dered_imag = sat_dered_imag[ da0: da1 ]

		sub_ra, sub_dec, sub_z = sat_ra[ da0: da1 ], sat_dec[ da0: da1 ], sat_z[ da0: da1 ]
		sub_Lumi_r, sub_Lumi_g, sub_Lumi_i = sat_Lumi_r[ da0: da1 ], sat_Lumi_g[ da0: da1 ], sat_Lumi_i[ da0: da1 ]

		_sub_n = len( sub_ra )

	sub_Pmem = np.ones( _sub_n, )
	sub_r_mag_err = np.ones( _sub_n, ) * 0.5
	sub_g_mag_err = np.ones( _sub_n, ) * 0.5
	sub_i_mag_err = np.ones( _sub_n, ) * 0.5

	sub_clus_z_arr = np.ones( _sub_n ) * z_g

	dpx.append( _sub_n )

	keys = [ 'centric_R(Mpc/h)', 'r_mags', 'g_mags', 'i_mags', 'P_member', 'r_mag_err', 'g_mag_err', 'i_mag_err', 
				'dered_r_mags', 'dered_g_mags', 'dered_i_mags', 'ra', 'dec', 'z', 'clus_z', 'L_r', 'L_g', 'L_i' ]

	values = [ sub_cen_R, sub_r_mag, sub_g_mag, sub_i_mag, sub_Pmem, sub_r_mag_err, sub_g_mag_err, sub_i_mag_err, 
				sud_dered_rmag, sud_dered_gmag, sud_dered_imag, sub_ra, sub_dec, sub_z, sub_clus_z_arr, 
				sub_Lumi_r, sub_Lumi_g, sub_Lumi_i ]

	fill = dict( zip( keys, values ) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/figs/sat_cat_ZLW/mem_match/' + 
		'ZLW_%s-band_ra%.3f_dec%.3f_z%.3f_members_mag.csv' % ('r', ra_g, dec_g, z_g), )

	#... catalog match
	out_arr = np.array( [ sub_ra, sub_dec, sub_z, sub_cen_R, 
						sub_g_mag - sub_r_mag, sub_g_mag - sub_i_mag, sub_r_mag - sub_i_mag, 
						sud_dered_gmag - sud_dered_rmag, sud_dered_gmag - sud_dered_imag, 
						sud_dered_rmag - sud_dered_imag, 
						sub_Lumi_r, sub_Lumi_g, sub_Lumi_i ] )

	gk = hig_F_tree.create_group( "clust_%d/" % sub_tt )
	dk0 = gk.create_dataset( "arr", data = out_arr )
hig_F_tree.close()

print(' high sample matched')

