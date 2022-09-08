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
from img_sat_clus_shuffle_list import sat_clust_shufl_func

#.
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
rad2arcsec = U.rad.to( U.arcsec )


### === data load
cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_rich_rebin_sat_cat/'
out_path = '/home/xkchen/data/SDSS/member_files/shufl_img_wBCG/shufl_cat/'

bin_rich = [ 20, 30, 50, 210 ]

sub_name = ['low-rich', 'medi-rich', 'high-rich']

N_shufl = 20       ###. shuffle 20 times


for dd in range( rank, rank + 1):

    band_str = band[ dd ]

    for tt in range( 3 ):

        for kk in range( N_shufl ):

            clus_cat = cat_path + 'clust_rich_%d-%d_cat.csv' % ( bin_rich[tt], bin_rich[tt + 1])

            member_cat = ( cat_path + 
                            'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_sat_%s-band_origin-img_position.csv' 
                            % (bin_rich[tt], bin_rich[tt + 1], band_str), )[0]

            out_file = ( out_path + 
                            'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_sat_%s-band_sat-shuffle-%d_position.csv'
                            % (bin_rich[tt], bin_rich[tt + 1], band_str, kk),)[0]

            sat_clust_shufl_func( clus_cat, member_cat, out_file )


##.. satellite number count
for dd in range( rank, rank + 1):

    band_str = band[ dd ]

    for tt in range( 3 ):

        for kk in range( N_shufl ):

            ##.
            dat = pds.read_csv( out_path + 
                        'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_sat_%s-band_sat-shuffle-%d_position.csv'
                        % (bin_rich[tt], bin_rich[tt + 1], band_str, kk), )

            bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
            sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

            orin_IDs = np.array( dat['orin_cID'] )
            rand_IDs = np.array( dat['shufl_cID'] )

            orin_IDs = orin_IDs.astype( int )
            rand_IDs = rand_IDs.astype( int )


            ##. entire all sample
            dat = pds.read_csv( cat_path + 'clust_rich_%d-%d_cat.csv' % ( bin_rich[tt], bin_rich[tt + 1]),)
            clus_IDs = np.array( dat['clust_ID'] )
            clus_IDs = clus_IDs.astype( int )

            N_w = len( clus_IDs )
            N_ss = len( bcg_ra )

            ##.
            pre_Ng = np.zeros( N_ss, )
            shufl_Ng = np.zeros( N_ss, )

            for mm in range( N_w ):

                sub_IDs = clus_IDs[ mm ]

                id_vx = orin_IDs == sub_IDs
                pre_Ng[ id_vx ] = np.sum( id_vx )

                id_ux = rand_IDs == sub_IDs
                shufl_Ng[ id_ux ] = np.sum( id_ux )

                print( np.sum( id_vx ) )

            ##. save
            keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'orin_Ng', 'shufl_Ng']
            values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, pre_Ng, shufl_Ng ]
            fill = dict( zip( keys, values ) )
            out_data = pds.DataFrame( fill )
            out_data.to_csv( out_path + 
                        'clust_rich_%d-%d_frame-lim_lower-Pm_sat_%s-band_sat-shuffle-%d_shufl-sat-Ng.csv'
                        % (bin_rich[tt], bin_rich[tt + 1], band_str, kk),)

