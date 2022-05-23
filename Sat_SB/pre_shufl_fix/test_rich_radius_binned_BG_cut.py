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
from scipy.stats import binned_statistic as binned
import scipy.interpolate as interp
#.
from light_measure import light_measure_weit
from img_sat_BG_extract import BG_build_func
from img_sat_BG_extract import sat_BG_extract_func
from img_sat_BG_extract import origin_img_cut_func
from img_sat_resamp import resamp_func

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


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


### === tacking the stacking image of this cutout as background
load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'
cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_richbin_sat_cat/'

img_file = home + 'photo_files/pos_offset_correct_imgs/mask_img/photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

##. cluster subsamples
bin_rich = [ 20, 30, 50, 210 ]
sub_name = ['low-rich', 'medi-rich', 'high-rich']

"""
# for tt in range( len(bin_rich) - 1 ):
for tt in range( 1 ):

    #. target cluster (want to know background of satellites)
    dat = pds.read_csv( cat_path + 
            'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv' % ( bin_rich[tt], bin_rich[tt + 1]),)
    targ_IDs = np.array( dat['clus_ID'] )

    set_IDs = np.array( list( set( targ_IDs ) ) )
    set_IDs = set_IDs.astype( int )

    N_cc = len( set_IDs )


    #. shuffle cluster IDs
    R_cut = 320   ##. pixels
    out_file = home + 'member_files/rich_binned_shufl_img/mask_img/clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_img.fits'

    for kk in range( 3 ):

        band_str = band[ kk ]

        ##. satellite position record table
        # post_file = home + 'member_files/BG_tract_cat/Extend-BCGM_rgi-common_frame-limit_Pm-cut_exlu-BCG_Sat_%s-band_origin-img_position.csv'
        post_file = ( cat_path + 
                'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_' % (bin_rich[tt], bin_rich[tt + 1]) + 
                '%s-band_origin-img_position.csv',)[0]

        rand_IDs = np.loadtxt( cat_path +
            'clust_rich_%d-%d_frame-lim_Pm-cut_mem_%s-band_extra-shuffle-clus_cat.txt' % (bin_rich[tt], bin_rich[tt + 1], band_str),)

        rand_mp_IDs = rand_IDs.astype( int )

        #. shuffle cutout images
        m, n = divmod( N_cc, cpus )
        N_sub0, N_sub1 = m * rank, (rank + 1) * m
        if rank == cpus - 1:
            N_sub1 += n

        sub_clusID = set_IDs[N_sub0 : N_sub1]
        sub_rand_mp_ID = rand_mp_IDs[N_sub0 : N_sub1]

        origin_img_cut_func( post_file, img_file, band_str, sub_clusID, sub_rand_mp_ID, R_cut, pixel, out_file)

    print('%s, %d-rank, cut Done!' %(sub_name[tt], rank), )

"""


#. resampling... 
for tt in range( len(bin_rich) - 1 ):

    for kk in range( 3 ):

        band_str = band[ kk ]

        dat = pds.read_csv( cat_path + 
                'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_%s-band_member_pos.csv' % 
                (bin_rich[ tt ], bin_rich[tt+1], band_str),)

        bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
        sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
        sat_cx, sat_cy = np.array( dat['cut_cx'] ), np.array( dat['cut_cy'] )


        _Ns_ = len( sat_ra )

        m, n = divmod( _Ns_, cpus)
        N_sub0, N_sub1 = m * rank, (rank + 1) * m
        if rank == cpus - 1:
            N_sub1 += n

        sub_ra, sub_dec, sub_z = bcg_ra[N_sub0 : N_sub1], bcg_dec[N_sub0 : N_sub1], bcg_z[N_sub0 : N_sub1]
        ra_set, dec_set = sat_ra[N_sub0 : N_sub1], sat_dec[N_sub0 : N_sub1]
        img_x, img_y = sat_cx[N_sub0 : N_sub1], sat_cy[N_sub0 : N_sub1]


        id_dimm = True
        d_file = home + 'member_files/rich_binned_shufl_img/mask_img/clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_img.fits'
        out_file = home + 'member_files/rich_binned_shufl_img/resamp_img/clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp.fits'

        resamp_func( d_file, sub_z, sub_ra, sub_dec, ra_set, dec_set, img_x, img_y, band_str, out_file, z_ref, id_dimm = id_dimm )

    print( '%d rank, done!' % rank )

