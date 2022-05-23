"""
take 2000 satellite image to test
SB(r), compare with SDSS profMean
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pds
import h5py

import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
import scipy.interpolate as interp

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable

from img_sat_pros_stack import single_img_SB_func
from img_sat_pros_stack import aveg_SB_func
from img_sat_pros_stack import jack_aveg_SB_func
from img_sat_fig_out_mode import arr_jack_func
#.
from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


### === cosmology model
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


### === entire samples
"""
home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

cat_path = load + 'Extend_Mbcg_richbin_sat_cat/'
out_path = '/home/xkchen/fig_tmp/Extend_Mbcg_richbin_sat_stack/'


bin_rich = [ 20, 30, 50, 210 ]
sub_name = ['low-rich', 'medi-rich', 'high-rich']

#. physical raidii
r_bins = np.logspace( 0, 2.2, 25)
N_samples = 100


for ll in range( 3 ):

	for kk in range( 3 ):

		band_str = band[ kk ]

		##.
		dat = pds.read_csv( cat_path + 'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_%s-band_member_pos-zref.csv' % 
							(bin_rich[ll], bin_rich[ll + 1], band_str),)

		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

		pros_file = '/home/xkchen/data/SDSS/member_files/sat_profMean/' + 'sat_ra%.5f_dec%.5f_SDSS_prof.txt'

		sub_out_file = out_path + 'Extend_BCGM_gri-common_%s_%s-band' % (sub_name[ ll ], band_str) + 'jack-sub-%d_sdss-prof-SB.csv'
		aveg_out_file = out_path + 'Extend_BCGM_gri-common_%s_%s-band_aveg-sdss-prof-SB.csv' % (sub_name[ ll ], band_str)

		jack_aveg_SB_func( N_samples, sat_ra, sat_dec, band_str, bcg_z, pros_file, r_bins, sub_out_file, aveg_out_file, z_ref = z_ref )

"""


### === sub-samples
home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'


# cat_path = load + 'Extend_Mbcg_richbin_sat_cat/'
# out_path = '/home/xkchen/fig_tmp/Extend_Mbcg_richbin_sat_stack/'

#..
cat_path = load + 'Extend_Mbcg_rich_rebin_sat_cat/'
out_path = '/home/xkchen/fig_tmp/Extend_Mbcg_richbin_sat_stack/'


bin_rich = [ 20, 30, 50, 210 ]
sub_name = ['low-rich', 'medi-rich', 'high-rich']
cat_lis = ['inner', 'middle', 'outer']


#. physical raidii
r_bins = np.logspace( 0, 2.2, 25)
N_samples = 100


for ll in range( rank, rank + 1 ):

	for tt in range( 3 ):

		for kk in range( 3 ):

			band_str = band[ kk ]

			dat = pds.read_csv(cat_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR-%s-mem_%s-band_pos-zref.csv' 
							% ( bin_rich[ ll ], bin_rich[ll + 1], cat_lis[tt], band_str),)

			bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
			sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

			##.
			pros_file = '/home/xkchen/data/SDSS/member_files/sat_profMean/' + 'sat_ra%.5f_dec%.5f_SDSS_prof.txt'

			sub_out_file = ( out_path + 'Extend_BCGM_gri-common_%s_phyR-%s_%s-band' % (sub_name[ll], cat_lis[tt], band_str) + 
								'jack-sub-%d_sdss-prof-SB.csv',)[0]

			aveg_out_file = out_path + 'Extend_BCGM_gri-common_%s_phyR-%s_%s-band_aveg-sdss-prof-SB.csv' % (sub_name[ll], cat_lis[tt], band_str)

			jack_aveg_SB_func( N_samples, sat_ra, sat_dec, band_str, bcg_z, pros_file, r_bins, sub_out_file, aveg_out_file, z_ref = z_ref )

print('%d-rank, done!' % rank )

