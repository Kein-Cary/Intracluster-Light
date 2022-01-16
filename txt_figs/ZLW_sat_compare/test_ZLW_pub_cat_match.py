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


##... Wen's cluster catalog
# wen_file = '/home/xkchen/figs/ZLW_cat_12/apjs425031t1_mrt.txt'  # Wen+2012
# bcg_ra, bcg_dec, bcg_zs, mr_bcg, clus_zp, R_200, rich_200, Ng_200, R_500, rich_500, Ng_500, cat_name = ZLW12_cat_read_func( wen_file, 30)


wen_file = '/home/xkchen/figs/ZLW_cat_15/apj515105t3_mrt.txt'   # Wen+2015 (the same as Wen+2012, but with new parameters)
bcg_ra, bcg_dec, bcg_zs, mr_bcg, clus_zp, R_200, rich_200, Ng_200, R_500, rich_500, Ng_500, cat_name = ZLW_cat_read_func( wen_file, 26)

# wen_file = '/home/xkchen/figs/ZLW_cat_15/apj515105t4_mrt.txt'  # new identified clusters in Wen+2015 
# bcg_ra, bcg_dec, bcg_zs, mr_bcg, clus_zp, R_200, rich_200, Ng_200, R_500, rich_500, Ng_500, cat_name = ZLW_cat_read_func( wen_file, 19)

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

raise


#. matched ZLWen catalog information
keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'bcg_Mr', 'clus_zp', 'R200c', 'rich', 'N_mem', 'clust_name']

lo_mat_arr = [ bcg_ra[ idx_l[ dex_lim_l ] ], bcg_dec[ idx_l[ dex_lim_l ] ], bcg_zs[ idx_l[ dex_lim_l ] ], mr_bcg[ idx_l[ dex_lim_l ] ], 
				clus_zp[ idx_l[ dex_lim_l ] ], R_200[ idx_l[ dex_lim_l ] ], rich_200[ idx_l[ dex_lim_l ] ], Ng_200[ idx_l[ dex_lim_l ] ],
				cat_name[ idx_l[ dex_lim_l ] ] ]

fill = dict( zip( keys, lo_mat_arr ) )

out_data = pds.DataFrame( fill )
out_data.to_csv('/home/xkchen/figs/ZLW_cat_15/ZLW_clust_match_low-age.csv')
# out_data.to_csv('/home/xkchen/figs/ZLW_cat_15/ZLW_clust_match_low-rich.csv')
# out_data.to_csv('/home/xkchen/figs/ZLW_cat_15/ZLW_clust_match_low_BCG_star-Mass.csv')



keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'bcg_Mr', 'clus_zp', 'R200c', 'rich', 'N_mem', 'clust_name']

hi_mat_arr = [ bcg_ra[ idx_h[ dex_lim_h ] ], bcg_dec[ idx_h[ dex_lim_h ] ], bcg_zs[ idx_h[ dex_lim_h ] ], mr_bcg[ idx_h[ dex_lim_h ] ], 
				clus_zp[ idx_h[ dex_lim_h ] ], R_200[ idx_h[ dex_lim_h ] ], rich_200[ idx_h[ dex_lim_h ] ], Ng_200[ idx_h[ dex_lim_h ] ], 
				cat_name[ idx_h[ dex_lim_h ] ] ]

fill = dict( zip( keys, hi_mat_arr ) )
out_data = pds.DataFrame( fill )
out_data.to_csv('/home/xkchen/figs/ZLW_cat_15/ZLW_clust_match_high-age.csv')
# out_data.to_csv('/home/xkchen/figs/ZLW_cat_15/ZLW_clust_match_high-rich.csv')
# out_data.to_csv('/home/xkchen/figs/ZLW_cat_15/ZLW_clust_match_high_BCG_star-Mass.csv')



#. matched SDSS redMaPPer catalog information
keys = ['ra', 'dec', 'z', 'order']
values = [ lo_ra[dex_lim_l], lo_dec[dex_lim_l], lo_z[dex_lim_l], np.where(dex_lim_l)[0] ]
fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )
out_data.to_csv('/home/xkchen/figs/ZLW_cat_15/redMapper_matched_low-age.csv')
# out_data.to_csv('/home/xkchen/figs/ZLW_cat_15/redMapper_matched_low-rich.csv')
# out_data.to_csv('/home/xkchen/figs/ZLW_cat_15/redMapper_matched_low_BCG_star-Mass.csv')



keys = ['ra', 'dec', 'z', 'order']
values = [ hi_ra[dex_lim_h], hi_dec[dex_lim_h], hi_z[dex_lim_h], np.where(dex_lim_h)[0] ]
fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )
out_data.to_csv('/home/xkchen/figs/ZLW_cat_15/redMapper_matched_high-age.csv')
# out_data.to_csv('/home/xkchen/figs/ZLW_cat_15/redMapper_matched_high-rich.csv')
# out_data.to_csv('/home/xkchen/figs/ZLW_cat_15/redMapper_matched_high_BCG_star-Mass.csv')

raise
