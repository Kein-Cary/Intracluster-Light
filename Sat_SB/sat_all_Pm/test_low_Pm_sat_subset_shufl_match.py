import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pds

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc

from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord

#.
from img_sat_clus_shuffle_list import shufl_list_map_func


##### cosmology model
Test_model = apcy.Planck15.clone( H0 = 67.74, Om0 = 0.311 )
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']


### === ### data load
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/cat/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_all_Pm/cat/'
shufl_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_all_Pm/shufl_list/'


bin_rich = [ 20, 30, 50, 210 ]

##. shuffle list order
list_order = 13     ###. use for test


###.. fixed scaled radius subsample
"""
R_bins = np.array( [0, 1e-1, 2e-1, 3e-1, 4.5e-1, 1] )   ### times R200m

for pp in range( 3 ):

	for tt in range( len(R_bins) - 1 ):

		sub_cat_file = ( out_path + 
                        'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_%.2f-%.2fR200m_mem_cat.csv'
                        % ( bin_rich[ pp ], bin_rich[ pp + 1], R_bins[tt], R_bins[tt + 1]),)[0]

		#.cluster catalog
		clust_file = cat_path + 'clust_rich_%d-%d_cat.csv' % (bin_rich[pp], bin_rich[pp + 1])

		for kk in range( 3 ):

			band_str = band[ kk ]

			N_shufl = 20

			shufl_list_file = ( shufl_path + 
                            'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_sat_%s-band_' % (bin_rich[pp], bin_rich[pp + 1], band_str) 
                            + 'sat-shuffle-%d_position.csv', )[0]

			orin_img_pos_file = ( out_path + 
                        'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_sat_%s-band_origin-img_position.csv'
                        % (bin_rich[pp], bin_rich[pp + 1], band_str), )[0]

			##.
			out_shufl_file = ( out_path + 'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_%.2f-%.2fR200m_mem_%s-band_' 
								% ( bin_rich[pp], bin_rich[pp + 1], R_bins[tt], R_bins[tt + 1], band_str) + 'sat-shufl-%d_cat.csv',)[0]

			out_pos_file = ( out_path + 'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_%.2f-%.2fR200m_mem_%s-band_'
								% ( bin_rich[pp], bin_rich[pp + 1], R_bins[tt], R_bins[tt + 1], band_str) + 'sat-shufl-%d_origin-img_position.csv',)[0]

			out_Ng_file = ( out_path + 'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_%.2f-%.2fR200m_mem_%s-band_' 
								% ( bin_rich[pp], bin_rich[pp + 1], R_bins[tt], R_bins[tt + 1], band_str) + 'sat-shufl-%d_shufl-Ng.csv',)[0]

			shufl_list_map_func( sub_cat_file, clust_file, N_shufl, shufl_list_file, out_shufl_file, 
								orin_img_pos_file, out_pos_file, out_Ng_file, list_idx = list_order )

raise
"""

###.. fixed physical radius subsample
R_bins = np.array( [ 0, 300, 400, 550, 5000] )

for pp in range( 3 ):

	for tt in range( len(R_bins) - 1 ):

		sub_cat_file = ( out_path + 'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_phyR_%d-%dkpc_mem_cat.csv' % 
								( bin_rich[pp], bin_rich[pp + 1], R_bins[tt], R_bins[tt + 1]),)[0]

		#.cluster catalog
		clust_file = ( cat_path + 'clust_rich_%d-%d_cat.csv' % (bin_rich[pp], bin_rich[pp + 1]),)[0]

		for kk in range( 3 ):

			band_str = band[ kk ]

			N_shufl = 20

			shufl_list_file = ( shufl_path + 
                            'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_sat_%s-band_' % (bin_rich[pp], bin_rich[pp + 1], band_str) 
                            + 'sat-shuffle-%d_position.csv', )[0]

			orin_img_pos_file = ( out_path + 
                        'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_sat_%s-band_origin-img_position.csv'
                        % (bin_rich[pp], bin_rich[pp + 1], band_str), )[0]

			##.
			out_shufl_file = ( out_path + 'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_phyR_%d-%dkpc_mem_%s-band_' 
								% ( bin_rich[pp], bin_rich[pp + 1], R_bins[tt], R_bins[tt + 1], band_str) + 'sat-shufl-%d_cat.csv',)[0]

			out_pos_file = ( out_path + 'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_phyR_%d-%dkpc_mem_%s-band_'
								% ( bin_rich[pp], bin_rich[pp + 1], R_bins[tt], R_bins[tt + 1], band_str) + 'sat-shufl-%d_origin-img_position.csv',)[0]

			out_Ng_file = ( out_path + 'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_phyR_%d-%dkpc_mem_%s-band_' 
								% ( bin_rich[pp], bin_rich[pp + 1], R_bins[tt], R_bins[tt + 1], band_str) + 'sat-shufl-%d_shufl-Ng.csv',)[0]

			shufl_list_map_func( sub_cat_file, clust_file, N_shufl, shufl_list_file, out_shufl_file, 
								orin_img_pos_file, out_pos_file, out_Ng_file, list_idx = list_order )

