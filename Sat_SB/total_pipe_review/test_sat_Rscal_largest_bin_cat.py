import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc

from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord

from img_sat_fast_stack import sat_img_fast_stack_func
from img_sat_fast_stack import sat_BG_fast_stack_func
#.
from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']



### === ### bulid random list for stacking~(also divid the outer most bin by scaled R_sat)
cat_path = '/home/xkchen/Pictures/BG_calib_SBs/fixR_bin/cat_Rbin/'
out_path = '/home/xkchen/Pictures/BG_calib_SBs/largest_Rs_compare/cat/'

R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

bin_rich = [ 20, 30, 50, 210 ]

"""
##... combine the member catalog
dat = pds.read_csv( cat_path + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_20-30_0.56-1.00R200m_mem_cat.csv') 

keys = dat.columns[1:]

N_ks = len( keys )

tmp_arr = []

#.
for dd in range( N_ks ):

	tmp_arr.append( np.array( dat['%s' % keys[dd] ] ) )

#.
for kk in range( 1, 3):

	pat = pds.read_csv( cat_path + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_0.56-1.00R200m_mem_cat.csv'
			% (bin_rich[kk], bin_rich[kk+1]),)

	for dd in range( N_ks ):

		tmp_arr[ dd ] = np.r_[ tmp_arr[ dd ], np.array( pat['%s' % keys[dd] ] ) ]

##.
fill = dict( zip( keys, tmp_arr) )
data = pds.DataFrame( fill )
data.to_csv( out_path + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_0.56-1.00R200m_mem_cat.csv',)


##... zref-pos cat. combine
dat = pds.read_csv( cat_path + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_20-30_0.56-1.00R200m_mem-r-band_pos-zref.csv') 

keys = dat.columns[1:]

N_ks = len( keys )

tmp_arr = []

#.
for dd in range( N_ks ):

	tmp_arr.append( np.array( dat['%s' % keys[dd] ] ) )

#.
for kk in range( 1, 3):

	pat = pds.read_csv( cat_path + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_0.56-1.00R200m_mem-r-band_pos-zref.csv'
			% (bin_rich[kk], bin_rich[kk+1]),)

	for dd in range( N_ks ):

		tmp_arr[ dd ] = np.r_[ tmp_arr[ dd ], np.array( pat['%s' % keys[dd] ] ) ]

##.
fill = dict( zip( keys, tmp_arr) )
data = pds.DataFrame( fill )
data.to_csv( out_path + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_0.56-1.00R200m_mem-r-band_pos-zref.csv',)


##... shufl_Ng cat. combine
dat = pds.read_csv( cat_path + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_20-30_0.56-1.00R200m_mem_r-band_sat_fixRs-shufl-13_shufl-Ng.csv') 

keys = dat.columns[1:]

N_ks = len( keys )

tmp_arr = []

#.
for dd in range( N_ks ):

	tmp_arr.append( np.array( dat['%s' % keys[dd] ] ) )

#.
for kk in range( 1, 3):

	pat = pds.read_csv( cat_path + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_0.56-1.00R200m_mem_r-band_sat_fixRs-shufl-13_shufl-Ng.csv'
			% (bin_rich[kk], bin_rich[kk+1]),)

	for dd in range( N_ks ):

		tmp_arr[ dd ] = np.r_[ tmp_arr[ dd ], np.array( pat['%s' % keys[dd] ] ) ]

##.
fill = dict( zip( keys, tmp_arr) )
data = pds.DataFrame( fill )
data.to_csv( out_path + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_0.56-1.00R200m_mem_r-band_sat_fixRs-shufl-13_shufl-Ng.csv',)

"""

### ... shuffle order and R binned
dat = pds.read_csv( out_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_0.56-1.00R200m_mem_cat.csv',)

bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

R_sat, R2Rv = np.array( dat['R_sat'] ), np.array( dat['R2Rv'] )

clus_ID = np.array( dat['clus_ID'] )
clus_ID = clus_ID.astype( int )

medi_Rs = np.median( R2Rv )

raise

id_x0 = R2Rv <= medi_Rs
id_x1 = R2Rv >= medi_Rs

##.
keys = dat.columns[1:]
values = [ bcg_ra[ id_x0 ], bcg_dec[ id_x0 ], bcg_z[ id_x0 ], sat_ra[ id_x0 ], sat_dec[ id_x0 ], 
			R_sat[ id_x0 ], R2Rv[ id_x0 ], clus_ID[ id_x0 ] ]

fill = dict( zip( keys, values ) )
data = pds.DataFrame( fill )
data.to_csv( out_path + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_0.56-1.00R200m_Rs-cut-0_mem_cat.csv',)

##.
keys = dat.columns[1:]
values = [ bcg_ra[ id_x1 ], bcg_dec[ id_x1 ], bcg_z[ id_x1 ], sat_ra[ id_x1 ], sat_dec[ id_x1 ], 
			R_sat[ id_x1 ], R2Rv[ id_x1 ], clus_ID[ id_x1 ] ]

fill = dict( zip( keys, values ) )
data = pds.DataFrame( fill )
data.to_csv( out_path + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_0.56-1.00R200m_Rs-cut-1_mem_cat.csv',)

##.
from list_shuffle import find_unique_shuffle_lists

N_galx = len( sat_ra )

dex_list = list( np.arange( N_galx ) )
n_times = 10

rand_dex = find_unique_shuffle_lists( dex_list, n_times )

np.savetxt( out_path + 'random_set_dex.txt', np.array( rand_dex ),)

raise
