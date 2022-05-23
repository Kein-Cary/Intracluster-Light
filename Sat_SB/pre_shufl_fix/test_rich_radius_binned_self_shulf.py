import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Ellipse

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc
import scipy.stats as sts

from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from list_shuffle import find_unique_shuffle_lists
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
Omega_lambda = 1. - Omega_m
Omega_k = 1. - (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']


### === shuffle position of satellites but within their own host cluster
##. selecte the symmetry point of each cluster
def sate_position_check():

    bin_rich = [ 20, 30, 50, 210 ]
    sub_name = ['low-rich', 'medi-rich', 'high-rich']

    ##. SDSS origin image cut (for satellite background estimation)
    cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_richbin_sat_cat/'
    img_file = '/home/xkchen/data/SDSS/photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

    out_path = '/home/xkchen/data/SDSS/member_files/rich_binned_self_shufl/shufl_cat/'

    for tt in range( len(bin_rich) - 1 ):

        ##. origin image location of satellites
        dat = pds.read_csv( cat_path + 
            'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv' % ( bin_rich[tt], bin_rich[tt + 1]),)

        bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
        clus_IDs = np.array( dat['clus_ID'] )

        set_IDs = np.array( list( set( clus_IDs ) ) )
        set_IDs = set_IDs.astype( int )

        ## shuffle within the same cluster, rand IDs is the same as the cluster catalog
        rand_IDs = set_IDs.copy()

        N_ss = len( set_IDs )


        for mm in range( rank, rank + 1 ):

            band_str = band[ mm ]

            ##. record table
            sub_err_record_file = h5py.File( out_path + 
                'clust_rich_%d-%d_%s-band_err_in_symmetry-position.h5' % (bin_rich[tt], bin_rich[tt + 1], band_str), 'w')


            dat = pds.read_csv(cat_path + 
                'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_%s-band_origin-img_position.csv' % 
                    (bin_rich[tt], bin_rich[tt + 1], band_str ),)

            bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
            sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
            bcg_x, bcg_y = np.array( dat['bcg_x'] ), np.array( dat['bcg_y'] )
            sat_x, sat_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )
            tt_clus_ID = np.array( dat['clus_ID'] )


            err_ra, err_dec, err_z = np.array( [] ), np.array( [] ), np.array( [] )
            err_IDs, err_Ng = np.array( [] ), np.array( [] )
            err_sat_ra, err_sat_dec = np.array( [] ), np.array( [] )

            for kk in range( N_ss ):

                #. target cluster satellites
                id_vx = tt_clus_ID == set_IDs[ kk ]

                sub_ra, sub_dec = sat_ra[ id_vx ], sat_dec[ id_vx ]
                ra_g, dec_g, z_g = bcg_ra[ id_vx ][0], bcg_dec[ id_vx ][0], bcg_z[ id_vx ][0]

                x_cen, y_cen = bcg_x[ id_vx ][0], bcg_y[ id_vx ][0]
                x_sat, y_sat = sat_x[ id_vx ], sat_y[ id_vx ]

                sat_R = np.sqrt( (x_sat - x_cen)**2 + (y_sat - y_cen)**2 )
                sat_theta = np.arctan2( (y_sat - y_cen), (x_sat - x_cen) )

                off_x, off_y = sat_R * np.cos( sat_theta ), sat_R * np.sin( sat_theta )


                #. loop for satellites symmetry position adjust 
                N_pot = len( sat_R )

                tm_sx, tm_sy = np.zeros( N_pot,), np.zeros( N_pot,)
                id_err = np.zeros( N_pot,)  ##. records points cannot located in image frame

                for pp in range( N_pot ):

                    tm_phi = np.array( [ np.pi + sat_theta[ pp ], 
                                        np.pi - sat_theta[ pp ], np.pi * 2 - sat_theta[ pp ] ] )

                    tm_off_x = sat_R[ pp ] * np.cos( tm_phi )
                    tm_off_y = sat_R[ pp ] * np.sin( tm_phi )

                    tt_sx, tt_sy = x_cen + tm_off_x, y_cen + tm_off_y

                    id_ux = ( tt_sx >= 0 ) & ( tt_sx < 2047 )
                    id_uy = ( tt_sy >= 0 ) & ( tt_sy < 1488 )

                    id_up = id_ux & id_uy

                    if np.sum( id_up ) > 0:

                        tm_sx[ pp ] = tt_sx[ id_up ][0]
                        tm_sy[ pp ] = tt_sy[ id_up ][0]

                    else:
                        id_err[ pp ] = 1.

                #. 
                cp_sx, cp_sy = tm_sx + 0., tm_sy + 0.

                ##. if there is somepoint is always can not located in image frame
                ##. then take the symmetry points of entire satellites sample
                print( np.sum( id_err ) )

                if np.sum( id_err ) > 0:

                    err_IDs = np.r_[ err_IDs, set_IDs[ kk ] ]
                    err_ra = np.r_[ err_ra, ra_g ]
                    err_dec = np.r_[ err_dec, dec_g ]
                    err_z = np.r_[ err_z, z_g ]
                    err_Ng = np.r_[ err_Ng, np.sum( id_err ) ]

                    err_sat_ra = np.r_[ err_sat_ra,  sub_ra[ id_err > 0. ] ]
                    err_sat_dec = np.r_[ err_sat_dec, sub_dec[ id_err > 0.] ]

            ##..
            gk = sub_err_record_file.create_group( "err_points" )
            clus_info = np.array( [ err_ra, err_dec, err_z, err_IDs, err_Ng ] )
            dk_0 = gk.create_dataset("err_clusters", data = clus_info )

            sat_info = np.array( [ err_sat_ra, err_sat_dec ] )
            dk_1 = gk.create_dataset("err_sat", data = sat_info )

            sub_err_record_file.close()

    return

sate_position_check()

