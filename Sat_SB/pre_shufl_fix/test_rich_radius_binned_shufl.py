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


##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1. - Omega_m
Omega_k = 1. - (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']


### === shulffle order of cluster and images
def cluster_img_order_shuffle():

    bin_rich = [ 20, 30, 50, 210 ]

    out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/shufl_cat/'
    cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/cat/'

    sub_name = ['low-rich', 'medi-rich', 'high-rich'] 

    for kk in range( len(bin_rich) - 1 ):

        dat = pds.read_csv(cat_path + 'clust_rich_%d-%d_cat.csv' % ( bin_rich[kk], bin_rich[kk + 1]),)

        clus_ID = np.array( dat['clust_ID'] )
        set_IDs = clus_ID.astype( int )     ## the order is the same as the cluster ID list in the table

        N_cc = len( set_IDs )

        ##. shuffle cluster and image order
        tt0 = list( np.arange( N_cc ) )
        Nt = 20

        rand_lis = find_unique_shuffle_lists( tt0, Nt )
        rand_IDs = []

        for tt in range( Nt ):

            rand_IDs.append( set_IDs[ rand_lis[ tt ] ] )

        rand_IDs = np.array( rand_IDs )

        #. save
        np.savetxt( out_path + 
                'clust_rich_%d-%d_shuffle-clus_cat.txt' % ( bin_rich[kk], bin_rich[kk + 1]), rand_IDs )

    return


### === check image cutout( is any satellites outside the image frame)
def sate_position_check():

    bin_rich = [ 20, 30, 50, 210 ]
    sub_name = ['low-rich', 'medi-rich', 'high-rich']

    ##. SDSS origin image cut (for satellite background estimation)
    cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_richbin_sat_cat/'
    img_file = '/home/xkchen/data/SDSS/photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

    out_path = '/home/xkchen/data/SDSS/member_files/rich_binned_shufl_img/shufl_cat/'

    for tt in range( len(bin_rich) - 1 ):

        ##. origin image location of satellites
        dat = pds.read_csv( cat_path + 
            'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv' % ( bin_rich[tt], bin_rich[tt + 1]),)

        bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
        clus_IDs = np.array( dat['clus_ID'] )

        # sat_ra, sat_dec = np.array( dat['ra'] ), np.array( dat['dec'] )

        set_IDs = np.array( list( set( clus_IDs ) ) )
        set_IDs = set_IDs.astype( int )
        N_ss = len( set_IDs )


        ##. shuffle cluster IDs
        rand_IDs = np.loadtxt( cat_path + 'clust_rich_%d-%d_shuffle-clus_cat.txt' % ( bin_rich[tt], bin_rich[tt + 1]),)
        # rand_mp_IDs = rand_IDs[0].astype( int )

        #. shuffle order~( total is 20 )
        N_shfl = 20

        for mm in range( 3 ):

            band_str = band[ mm ]

            # for qq in range( N_shfl ):
            for qq in range( rank, rank+1 ):

                ##. record table
                sub_err_record_file = h5py.File( out_path + 
                    'clust_rich_%d-%d_%s-band_err_in_position_shuffle-%d.h5' % (bin_rich[tt], bin_rich[tt + 1], band_str, qq), 'w')

                rand_mp_IDs = rand_IDs[ qq ].astype( int )   ## order will have error ( treat as the fiducial order list)

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

                    # img = fits.open( img_file % (band_str, ra_g, dec_g, z_g),)
                    # img_arr = img[0].data

                    x_cen, y_cen = bcg_x[ id_vx ][0], bcg_y[ id_vx ][0]
                    x_sat, y_sat = sat_x[ id_vx ], sat_y[ id_vx ]

                    sat_R = np.sqrt( (x_sat - x_cen)**2 + (y_sat - y_cen)**2 )
                    sat_theta = np.arctan2( (y_sat - y_cen), (x_sat - x_cen) )

                    off_x, off_y = sat_R * np.cos( sat_theta ), sat_R * np.sin( sat_theta )


                    #. shuffle clusters and matched images
                    id_ux = clus_IDs == rand_mp_IDs[ kk ]
                    cp_ra_g, cp_dec_g, cp_z_g = bcg_ra[ id_ux ][0], bcg_dec[ id_ux ][0], bcg_z[ id_ux ][0]

                    cp_img = fits.open( img_file % (band_str, cp_ra_g, cp_dec_g, cp_z_g),)
                    cp_img_arr = cp_img[0].data

                    #. image center and BCG location
                    pix_cx, pix_cy = cp_img_arr.shape[1] / 2, cp_img_arr.shape[0] / 2

                    cp_cx, cp_cy = bcg_x[ id_ux ][0], bcg_y[ id_ux ][0]
                    cp_sx_1, cp_sy_1 = cp_cx + off_x, cp_cy + off_y


                    #. identify satellites beyond the image frame of shuffle cluster
                    # Lx, Ly = cp_img_arr.shape[1], cp_img_arr.shape[0]

                    id_x_lim = ( cp_sx_1 < 0 ) | ( cp_sx_1 >= 2047 )
                    id_y_lim = ( cp_sy_1 < 0 ) | ( cp_sy_1 >= 1488 )

                    id_lim = id_x_lim + id_y_lim

                    # tp_sx, tp_sy = cp_sx_1[ id_lim ], cp_sy_1[ id_lim ]
                    tp_sat_ra, tp_sat_dec = sub_ra[ id_lim ], sub_dec[ id_lim ]
                    tp_chi = sat_theta[ id_lim ]
                    tp_Rs = sat_R[ id_lim ]


                    #. loop for satellites position adjust 
                    N_pot = np.sum( id_lim )

                    tm_sx, tm_sy = np.zeros( N_pot,), np.zeros( N_pot,)
                    id_err = np.zeros( N_pot,)  ##. records points cannot located in image frame

                    for pp in range( N_pot ):

                        tm_phi = np.array( [ np.pi + tp_chi[ pp ], np.pi - tp_chi[ pp ], np.pi * 2 - tp_chi[ pp ] ] )

                        tm_off_x = tp_Rs[ pp ] * np.cos( tm_phi )
                        tm_off_y = tp_Rs[ pp ] * np.sin( tm_phi )

                        tt_sx, tt_sy = cp_cx + tm_off_x, cp_cy + tm_off_y

                        id_ux = ( tt_sx >= 0 ) & ( tt_sx < 2047 )
                        id_uy = ( tt_sy >= 0 ) & ( tt_sy < 1488 )

                        id_up = id_ux & id_uy

                        if np.sum( id_up ) > 0:

                            tm_sx[ pp ] = tt_sx[ id_up ][0]
                            tm_sy[ pp ] = tt_sy[ id_up ][0]

                        else:
                            id_err[ pp ] = 1.

                    #. 
                    cp_sx, cp_sy = cp_sx_1 + 0., cp_sy_1 + 0.
                    cp_sx[ id_lim ] = tm_sx
                    cp_sy[ id_lim ] = tm_sy


                    ##. if there is somepoint is always can not located in image frame
                    ##. then take the symmetry points of entire satellites sample
                    # print( np.sum( id_err ) )

                    if np.sum( id_err ) > 0:

                        # cp_sx, cp_sy = 2 * pix_cx - x_sat, 2 * pix_cy - y_sat
                        # _sm_cp_cx, _sm_cp_cy = 2 * pix_cx - x_cen, 2 * pix_cy - y_cen

                        err_IDs = np.r_[ err_IDs, set_IDs[ kk ] ]
                        err_ra = np.r_[ err_ra, ra_g ]
                        err_dec = np.r_[ err_dec, dec_g ]
                        err_z = np.r_[ err_z, z_g ]
                        err_Ng = np.r_[ err_Ng, np.sum( id_err ) ]

                        err_sat_ra = np.r_[ err_sat_ra, tp_sat_ra[ id_err > 0. ] ]
                        err_sat_dec = np.r_[ err_sat_dec, tp_sat_dec[ id_err > 0.] ]

                ##. record for row_0 in shuffle list
                # keys = [ 'ra', 'dec', 'z', 'clus_ID' ]
                # values = [ err_ra, err_dec, err_z, err_IDs ]
                # fill = dict( zip( keys, values ) )
                # out_data = pds.DataFrame( fill )
                # out_data.to_csv( '/home/xkchen/clust_rich_%d-%d_%s-band_err_in_position_shuffle.csv' % 
                #                 (bin_rich[tt], bin_rich[tt + 1], band_str),)


                gk = sub_err_record_file.create_group( "shufle_list_%d" % qq )
                clus_info = np.array( [ err_ra, err_dec, err_z, err_IDs, err_Ng ] )
                dk_0 = gk.create_dataset("err_clusters", data = clus_info )

                sat_info = np.array( [ err_sat_ra, err_sat_dec ] )
                dk_1 = gk.create_dataset("err_sat", data = sat_info )

                sub_err_record_file.close()

    return


# cluster_img_order_shuffle()
# sate_position_check()



### === image check
bin_rich = [ 20, 30, 50, 210 ]
sub_name = ['low-rich', 'medi-rich', 'high-rich']

out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/shufl_cat/'
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/cat/'
img_file = '/media/xkchen/My Passport/data/SDSS/photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

for tt in range( 2,3 ):

    ##. origin image location of satellites
    dat = pds.read_csv( cat_path + 
        'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv' % ( bin_rich[tt], bin_rich[tt + 1]),)

    bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
    sat_ra, sat_dec = np.array( dat['ra'] ), np.array( dat['dec'] )
    clus_IDs = np.array( dat['clus_ID'] )

    set_IDs = np.array( list( set( clus_IDs ) ) )
    set_IDs = set_IDs.astype( int )
    N_ss = len( set_IDs )


    ##. shuffle cluster IDs
    rand_IDs = np.loadtxt( out_path + 'clust_rich_%d-%d_shuffle-clus_cat.txt' % ( bin_rich[tt], bin_rich[tt + 1]),)
    rand_mp_IDs = rand_IDs[0].astype( int )   ## order will have error ( treat as the fiducial order list)
    band_str = band[ rank ]

    ##. error raise up ordex (use row_0 in shuffle list)
    band_str = 'i'

    err_dat = pds.read_csv( out_path + 
        'err_list/clust_rich_%d-%d_%s-band_err_in_position_shuffle.csv' % (bin_rich[tt], bin_rich[tt + 1], band_str),)
    err_IDs = np.array( err_dat['clus_ID'] )
    err_ra, err_dec, err_z = np.array( err_dat['ra'] ), np.array( err_dat['dec'] ), np.array( err_dat['z'] )

    ordex = []
    for pp in range( len(err_IDs) ):

        id_vx = np.where( set_IDs == err_IDs[ pp ] )[0][0]
        ordex.append( id_vx )

    ##. combine shuffle
    t_ids = 4
    t0_rand_arr = rand_IDs[ 0 ].astype( int )  ##. origin shuffle
    t1_rand_arr = rand_IDs[ t_ids ].astype( int )  ##. adjust

    # t1_rand_arr = np.loadtxt( out_path + 'clust_rich_50-210_frame-lim_Pm-cut_mem_g-band_extra-shuffle-clus_cat.txt')
    # t1_rand_arr = t1_rand_arr.astype( int )

    cp_rand_arr = t0_rand_arr + 0
    cp_rand_arr[ ordex ] = t1_rand_arr[ ordex ]


    ##. satellite position
    dat = pds.read_csv(cat_path + 
        'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_%s-band_origin-img_position.csv' % 
            (bin_rich[tt], bin_rich[tt + 1], band_str ),)

    bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
    sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
    bcg_x, bcg_y = np.array( dat['bcg_x'] ), np.array( dat['bcg_y'] )
    sat_x, sat_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )
    tt_clus_ID = np.array( dat['clus_ID'] )


    tt_N = []

    print( ( bin_rich[tt], bin_rich[tt + 1]) )
    print( band_str )

    for kk in range( len( ordex ) ):

        #. target cluster satellites
        # id_vx = tt_clus_ID == set_IDs[ kk ]
        id_vx = tt_clus_ID == set_IDs[ ordex[ kk ] ]

        sub_ra, sub_dec = sat_ra[ id_vx ], sat_dec[ id_vx ]
        ra_g, dec_g, z_g = bcg_ra[ id_vx ][0], bcg_dec[ id_vx ][0], bcg_z[ id_vx ][0]

        img = fits.open( img_file % (band_str, ra_g, dec_g, z_g),)
        img_arr = img[0].data


        x_cen, y_cen = bcg_x[ id_vx ][0], bcg_y[ id_vx ][0]
        x_sat, y_sat = sat_x[ id_vx ], sat_y[ id_vx ]

        sat_R = np.sqrt( (x_sat - x_cen)**2 + (y_sat - y_cen)**2 )
        sat_theta = np.arctan2( (y_sat - y_cen), (x_sat - x_cen) )

        off_x, off_y = sat_R * np.cos( sat_theta ), sat_R * np.sin( sat_theta )


        #. shuffle clusters and matched images
        # id_ux = clus_IDs == rand_mp_IDs[ kk ]
        id_ux = clus_IDs == t1_rand_arr[ ordex[ kk ] ]

        cp_ra_g, cp_dec_g, cp_z_g = bcg_ra[ id_ux ][0], bcg_dec[ id_ux ][0], bcg_z[ id_ux ][0]

        cp_img = fits.open( img_file % (band_str, cp_ra_g, cp_dec_g, cp_z_g),)
        cp_img_arr = cp_img[0].data

        #. image center
        pix_cx, pix_cy = cp_img_arr.shape[1] / 2, cp_img_arr.shape[0] / 2

        cp_cx, cp_cy = bcg_x[ id_ux ][0], bcg_y[ id_ux ][0]
        cp_sx_1, cp_sy_1 = cp_cx + off_x, cp_cy + off_y


        #. identify satellites beyond the image frame of shuffle cluster
        Lx, Ly = cp_img_arr.shape[1], cp_img_arr.shape[0]

        id_x_lim = ( cp_sx_1 < 0 ) | ( cp_sx_1 >= 2047 )
        id_y_lim = ( cp_sy_1 < 0 ) | ( cp_sy_1 >= 1488 )

        id_lim = id_x_lim + id_y_lim

        tp_sx, tp_sy = cp_sx_1[ id_lim ], cp_sy_1[ id_lim ]
        tp_chi = sat_theta[ id_lim ]
        tp_Rs = sat_R[ id_lim ]


        #. loop for satellites position adjust 
        N_pot = np.sum( id_lim )

        tm_sx, tm_sy = np.zeros( N_pot,), np.zeros( N_pot,)

        id_err = np.zeros( N_pot,)  ##. records points cannot located in image frame

        for pp in range( N_pot ):

            tm_phi = np.array( [ np.pi + tp_chi[ pp ], np.pi - tp_chi[ pp ], np.pi * 2 - tp_chi[ pp ] ] )

            tm_off_x = tp_Rs[ pp ] * np.cos( tm_phi )
            tm_off_y = tp_Rs[ pp ] * np.sin( tm_phi )

            tt_sx, tt_sy = cp_cx + tm_off_x, cp_cy + tm_off_y

            id_ux = ( tt_sx >= 0 ) & ( tt_sx < 2047 )
            id_uy = ( tt_sy >= 0 ) & ( tt_sy < 1488 )

            id_up = id_ux & id_uy

            if np.sum( id_up ) > 0:

                tm_sx[ pp ] = tt_sx[ id_up ][0]
                tm_sy[ pp ] = tt_sy[ id_up ][0]

            else:
                id_err[ pp ] = 1.

        #. 
        cp_sx, cp_sy = cp_sx_1 + 0., cp_sy_1 + 0.
        cp_sx[ id_lim ] = tm_sx
        cp_sy[ id_lim ] = tm_sy


        ##. if there is somepoint is always can not located in image frame
        ##. then take the symmetry points of entire satellites sample
        tt_N.append( np.sum( id_err ) )
        print( np.sum( id_err ) )

        if np.sum( id_err ) > 0:

            cp_sx, cp_sy = 2 * pix_cx - x_sat, 2 * pix_cy - y_sat
            _sm_cp_cx, _sm_cp_cy = 2 * pix_cx - x_cen, 2 * pix_cy - y_cen


        plt.figure( figsize = (10, 5),)
        ax0 = plt.subplot(121)
        ax1 = plt.subplot(122)

        ax0.imshow( img_arr, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e0, norm = mpl.colors.LogNorm(),)
        ax0.scatter( x_sat, y_sat, s = 10, marker = 'o', facecolors = 'none', edgecolors = 'r',)
        ax0.scatter( x_cen, y_cen, s = 5, marker = 's', facecolors = 'none', edgecolors = 'b',)

        ax1.imshow( cp_img_arr, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e0, norm = mpl.colors.LogNorm(),)
        ax1.scatter( cp_cx, cp_cy, s = 5, marker = 's', facecolors = 'none', edgecolors = 'r',)

        if np.sum( id_err ) > 0:
            ax1.scatter( _sm_cp_cx, _sm_cp_cy, s = 5, marker = 's', facecolors = 'none', edgecolors = 'b',)

        ax1.scatter( cp_sx_1, cp_sy_1, s = 12, marker = 'o', facecolors = 'none', edgecolors = 'g', label = 'relative offset')
        ax1.scatter( tp_sx, tp_sy, s = 8, marker = '*', facecolors = 'none', edgecolors = 'r',)
        ax1.scatter( tm_sx, tm_sy, s = 8, marker = 'd', facecolors = 'none', edgecolors = 'r',)

        plt.savefig('/home/xkchen/%s-band_random_sat-Pos_set_%d.png' % (band_str, kk), dpi = 300)
        plt.close()

    tt_N = np.array( tt_N )

    # ##. if there no exceed points save the shuffle order list
    # if np.sum( tt_N ) == 0:

    #     cp_rand_arr = t0_rand_arr + 0   ## origin shuffle list
    #     cp_rand_arr[ ordex ] = t1_rand_arr[ ordex ]   ## update shuffle list
    #     np.savetxt( out_path + 
    #         'clust_rich_%d-%d_frame-lim_Pm-cut_mem_%s-band_extra-shuffle-clus_cat.txt' % 
    #             (bin_rich[tt], bin_rich[tt + 1], band_str ), cp_rand_arr)

    # ##. save temporary list for compare
    # else:
    #     cp_rand_arr = t0_rand_arr + 0   ## origin shuffle list
    #     cp_rand_arr[ ordex ] = t1_rand_arr[ ordex ]   ## update shuffle list
    #     np.savetxt( '/home/xkchen/clust_rich_%d-%d_frame-lim_Pm-cut_mem_%s-band_tmp-shuffle-clus_cat.txt' % 
    #                 (bin_rich[tt], bin_rich[tt + 1], band_str ), cp_rand_arr)

