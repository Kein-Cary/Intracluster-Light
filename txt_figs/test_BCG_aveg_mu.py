import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import time
import h5py
import numpy as np
import astropy.io.fits as fits

import mechanize
import pandas as pds
from io import StringIO
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

from BCG_SB_pro_stack import BCG_SB_pros_func
from fig_out_module import arr_jack_func
from fig_out_module import color_func

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

### constant
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']


### === load data
home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

band_str = band[ rank ]

color_s = [ 'r', 'g', 'b' ]
line_c = [ 'b', 'r' ]


##... low + high BCG Mstar catalog
cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']

lo_dat = pds.read_csv( load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_z-ref.csv' % (cat_lis[0], band_str),)
lo_ra, lo_dec, lo_z = np.array(lo_dat.ra), np.array(lo_dat.dec), np.array(lo_dat.z)
lo_imgx, lo_imgy = np.array(lo_dat.bcg_x), np.array(lo_dat.bcg_y)


hi_dat = pds.read_csv( load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_z-ref.csv' % (cat_lis[1], band_str),)
hi_ra, hi_dec, hi_z = np.array( hi_dat.ra ), np.array( hi_dat.dec ), np.array(hi_dat.z)
hi_imgx, hi_imgy = np.array( hi_dat.bcg_x ), np.array( hi_dat.bcg_y )


ra = np.r_[ lo_ra, hi_ra ]
dec = np.r_[ lo_dec, hi_dec ]
z = np.r_[ lo_z, hi_z ]


N_samples = 30
r_bins = np.logspace(0, 2.48, 25) # unit : kpc


##... directly mean of sample BCG profiles
pros_file = home + 'photo_files/BCG_profile/BCG_prof_Z%.3f_ra%.3f_dec%.3f.txt'
out_file = '/home/xkchen/figs/total_sample_%s-band_aveg_BCG_photo-SB_pros.csv' % band_str
BCG_SB_pros_func( band_str, z, ra, dec, pros_file, z_ref, out_file, r_bins)

print('N_sample = ', len(ra),)
print('band = %s' % band_str,)


##... divid sub-samples and estimate the average SB by Jackknife
zN = len( ra )
id_arr = np.arange(0, zN, 1)
id_group = id_arr % N_samples

lis_ra, lis_dec, lis_z = [], [], []

## sub-sample
for nn in range( N_samples ):

	id_xbin = np.where( id_group == nn )[0]

	lis_ra.append( ra[ id_xbin ] )
	lis_dec.append( dec[ id_xbin ] )
	lis_z.append( z[ id_xbin ] )

## jackknife sub-sample
for nn in range( N_samples ):

	id_arry = np.linspace( 0, N_samples - 1, N_samples )
	id_arry = id_arry.astype( int )
	jack_id = list( id_arry )
	jack_id.remove( jack_id[nn] )
	jack_id = np.array( jack_id )

	set_ra, set_dec, set_z = np.array([]), np.array([]), np.array([])

	for oo in ( jack_id ):
		set_ra = np.r_[ set_ra, lis_ra[oo] ]
		set_dec = np.r_[ set_dec, lis_dec[oo] ]
		set_z = np.r_[ set_z, lis_z[oo] ]

	pros_file = home + 'photo_files/BCG_profile/BCG_prof_Z%.3f_ra%.3f_dec%.3f.txt'  ## read SDSS photo_data
	out_file = '/home/xkchen/figs/total-sample_%s-band_jack-sub-%d_BCG_photo-SB_pros.csv' % (band_str, nn)

	BCG_SB_pros_func( band_str, set_z, set_ra, set_dec, pros_file, z_ref, out_file, r_bins)

## mean of jackknife sample
tmp_r, tmp_sb = [], []
for nn in range( N_samples ):

	pro_dat = pds.read_csv( '/home/xkchen/figs/total-sample_%s-band_jack-sub-%d_BCG_photo-SB_pros.csv' % (band_str, nn),)

	tt_r, tt_sb = np.array( pro_dat['R_ref'] ), np.array( pro_dat['SB_fdens'] )

	tmp_r.append( tt_r )
	tmp_sb.append( tt_sb )

mean_R, mean_sb, mean_sb_err, lim_R = arr_jack_func( tmp_sb, tmp_r, N_samples)

keys = [ 'R', 'aveg_sb', 'aveg_sb_err' ]
values = [ mean_R, mean_sb, mean_sb_err ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/figs/total-sample_%s-band_Mean-jack_BCG_photo-SB_pros.csv' % band_str,)

print( '%s band' % band_str, )

