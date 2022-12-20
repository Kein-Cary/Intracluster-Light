import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pds
import h5py

import scipy.stats as sts
from io import StringIO
import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable

#.
from img_sat_shuffle_map import zref_cut_func
from img_sat_BG_extract_tmp import zref_img_cut_func

#.
import time
from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


###... cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)

band = ['r', 'g', 'i']
z_ref = 0.25
pixel = 0.396
a_ref = 1 / (z_ref + 1)



### === ### data load
"""
## ... shuffle satellite fixed the alignment with the image frame
shufl_path = '/home/xkchen/data/SDSS/member_files/Mrich_binned_shufl_img/align_frame/shufl_cat/'
out_path = '/home/xkchen/data/SDSS/member_files/Mrich_binned_shufl_img/align_frame/zref_cut_cat/'

#.
bin_rich = [ 20, 30, 50, 210 ]

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

N_shufl = 60

R_cut = 320

#.
for pp in range( rank, rank + 1 ):

	for dd in range( 3 ):

		band_str = band[ dd ]

		for tt in range( 3 ):

			for kk in range( 2 ):

				##.
				shuffle_list = ( shufl_path + 
						'%s_clust_rich_%d-%d_%s-band_sat_fixRs-shuffle-%d_position.csv'
						% (cat_lis[kk], bin_rich[tt], bin_rich[tt + 1], band_str, pp), )[0]

				out_file = ( out_path + 
						'%s_clust_rich_%d-%d_%s-band_sat_fixRs-shuffle-%d_zref-img_cut-cat.csv'
						% (cat_lis[kk], bin_rich[tt], bin_rich[tt + 1], band_str, pp), )[0]

				zref_cut_func( shuffle_list, out_file, cut_size = R_cut, z_ref = z_ref, pixel = pixel)


				##.
				shuffle_list = ( shufl_path + 
						'%s_clust_rich_%d-%d_%s-band_sat-shuffle-%d_position.csv'
						% (cat_lis[kk], bin_rich[tt], bin_rich[tt + 1], band_str, pp), )[0]

				out_file = ( out_path + 
						'%s_clust_rich_%d-%d_%s-band_sat-shuffle-%d_zref-img_cut-cat.csv'
						% (cat_lis[kk], bin_rich[tt], bin_rich[tt + 1], band_str, pp), )[0]

				zref_cut_func( shuffle_list, out_file, cut_size = R_cut, z_ref = z_ref, pixel = pixel)

raise
"""


## ... shuffle satellite fixed the alignment with the major axis of BCG
shufl_path = '/home/xkchen/data/SDSS/member_files/Mrich_binned_shufl_img/align_BCG/shufl_cat/'
out_path = '/home/xkchen/data/SDSS/member_files/Mrich_binned_shufl_img/align_BCG/zref_cut_cat/'

#.
bin_rich = [ 20, 30, 50, 210 ]

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

N_shufl = 60

R_cut = 320

#.
for pp in range( rank, rank + 1 ):

	for dd in range( 3 ):

		band_str = band[ dd ]

		for tt in range( 3 ):

			for kk in range( 2 ):

				##.
				shuffle_list = ( shufl_path + 
						'%s_clust_rich_%d-%d_%s-band_sat_fixRs-shuffle-%d_position.csv'
						% (cat_lis[kk], bin_rich[tt], bin_rich[tt + 1], band_str, pp), )[0]

				out_file = ( out_path + 
						'%s_clust_rich_%d-%d_%s-band_sat_fixRs-shuffle-%d_zref-img_cut-cat.csv'
						% (cat_lis[kk], bin_rich[tt], bin_rich[tt + 1], band_str, pp), )[0]

				zref_cut_func( shuffle_list, out_file, cut_size = R_cut, z_ref = z_ref, pixel = pixel)


				##.
				shuffle_list = ( shufl_path + 
						'%s_clust_rich_%d-%d_%s-band_sat-shuffle-%d_position.csv'
						% (cat_lis[kk], bin_rich[tt], bin_rich[tt + 1], band_str, pp), )[0]

				out_file = ( out_path + 
						'%s_clust_rich_%d-%d_%s-band_sat-shuffle-%d_zref-img_cut-cat.csv'
						% (cat_lis[kk], bin_rich[tt], bin_rich[tt + 1], band_str, pp), )[0]

				zref_cut_func( shuffle_list, out_file, cut_size = R_cut, z_ref = z_ref, pixel = pixel)

print('%d-rank, Done!' % rank)

