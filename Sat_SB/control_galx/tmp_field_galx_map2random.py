from math import isfinite
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pds
import h5py

import mechanize
from io import StringIO
import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable

#.
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


### === data load
home = '/home/xkchen/data/SDSS/'
cat_path = '/home/xkchen/data/SDSS/field_galx_redMap/redMap_compare/'

"""
galx_dat = fits.open( cat_path + 'sdss_redMaP-limt_control-map-galaxy_for_z-clus_0.2to0.3.fits')
galx_table = galx_dat[1].data

g_ra, g_dec = galx_table['ra'], galx_table['dec']
g_z = galx_table['z']

obj_ID = galx_table['objid']
obj_ID = obj_ID.astype( int )

g_coord = SkyCoord( ra = g_ra * U.deg, dec = g_dec * U.deg )


##. have applied image selected images of random points in redMapper
img_file = home + 'redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

for kk in range( 3 ):

    band_str = band[ kk ]

    dat = pds.read_csv('/home/xkchen/fig_tmp/random_cat/2_28/' + 
                'random_%s-band_tot_remain_cat_set_200-grid_6.0-sigma.csv' % band_str)

    c_ra, c_dec, c_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

    N_clus = len( c_ra )


    ##.
    m, n = divmod( N_clus, cpus )
    N_sub0, N_sub1 = m * rank, (rank + 1) * m
    if rank == cpus - 1:
        N_sub1 += n

    sub_ra, sub_dec, sub_z = c_ra[ N_sub0: N_sub1 ], c_dec[ N_sub0: N_sub1 ], c_z[ N_sub0: N_sub1 ]

    sub_N = len( sub_ra )

    ##.
    map_ra, map_dec, map_z = np.array( [] ), np.array( [] ), np.array( [] )
    map_sat_ra, map_sat_dec, map_sat_z = np.array( [] ), np.array( [] ), np.array( [] )
    map_sat_x, map_sat_y = np.array( [] ), np.array( [] )

    map_sat_objid = np.array( [], dtype = np.int64 )

    for dd in range( sub_N ):

        ra_x, dec_x, z_x = sub_ra[ dd ], sub_dec[ dd ], sub_z[ dd ]

        img_dat = fits.open( img_file % ( band_str, ra_x, dec_x, z_x),)
        img_arr = img_dat[0].data

        Head = img_dat[0].header
        wcs_lis = awc.WCS( Head )

        sat_x, sat_y = wcs_lis.all_world2pix( g_ra, g_dec, 0 )

        ##. galaxy located in the image frame
        id_vx = ( sat_x >= 0 ) & ( sat_x <= 2047 )
        id_vy = ( sat_y >= 0 ) & ( sat_y <= 1488 )

        id_lim = id_vx & id_vy
        n_sat = np.sum( id_lim )

        if n_sat > 0:

            map_ra = np.r_[ map_ra, np.ones( n_sat,) * ra_x ]
            map_dec = np.r_[ map_dec, np.ones( n_sat,) * dec_x ]
            map_z = np.r_[ map_z, np.ones( n_sat,) * z_x ]
            
            map_sat_ra = np.r_[ map_sat_ra, g_ra[ id_lim ] ]
            map_sat_dec = np.r_[ map_sat_dec, g_dec[ id_lim ] ]
            map_sat_z = np.r_[ map_sat_z, g_z[ id_lim ] ]

            map_sat_objid = np.r_[ map_sat_objid, obj_ID[ id_lim ] ]
            
            map_sat_x = np.r_[ map_sat_x, sat_x[ id_lim ] ] 
            map_sat_y = np.r_[ map_sat_y, sat_y[ id_lim ] ]

            # #. figs for location compare
            # if band_str == 'r':
            #     plt.figure()
            #     plt.imshow( img_arr, origin = 'lower', cmap = 'Greys', norm = mpl.colors.LogNorm(),)
            #     plt.scatter( sat_x[ id_lim ], sat_y[ id_lim ], marker = 'o', facecolors = 'none', edgecolors = 'r', s = 25,)
            #     plt.savefig('/home/xkchen/figs/random_galx_map_ra%.3f_dec%.3f_z%.3f_pos.png' % (ra_x, dec_x, z_x), dpi = 300)
            #     plt.close()

        else:
            continue

    ##. save the galaxy infor.
    keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_z', 'sat_objID', 'sat_x', 'sat_y' ]
    values = [ map_ra, map_dec, map_z, map_sat_ra, map_sat_dec, map_sat_z, map_sat_objid, map_sat_x, map_sat_y ]
    fill = dict( zip( keys, values ) )
    data = pds.DataFrame( fill )
    data.to_csv( '/home/xkchen/figs/' + 'random_field-galx_map_%s-band_%d-rank.csv' % (band_str, rank),)

commd.Barrier()

##. combine those tables
if rank == 0:

    for  kk in range( 3 ):

        band_str = band[ kk ]

        cat = pds.read_csv('/home/xkchen/figs/' + 'random_field-galx_map_%s-band_%d-rank.csv' % (band_str, 0),)

        keys = list( cat.columns[1:] )
        N_ks = len( keys )

        tmp_arr = []
        
        for dd in range( N_ks ):

            tmp_arr.append( np.array( cat[ keys[ dd ] ] ) )

        for pp in range( 1, cpus ):

            cat = pds.read_csv('/home/xkchen/figs/' + 'random_field-galx_map_%s-band_%d-rank.csv' % (band_str, pp),)

            for dd in range( N_ks ):
                
                tmp_arr[ dd ] = np.r_[ tmp_arr[ dd ], np.array( cat[ keys[ dd ] ] ) ]

        ##. 
        fill = dict( zip( keys, tmp_arr ) )
        data = pds.DataFrame( fill )
        data.to_csv('/home/xkchen/data/SDSS/member_files/redMap_contral_galx/control_cat/' + 
                    'random_field-galx_map_%s-band_cat.csv' % band_str )

raise
"""


### === mapping the properties of galaxy
galx_dat = fits.open( cat_path + 'sdss_redMaP-limt_control-map-galaxy_for_z-clus_0.2to0.3.fits')
galx_table = galx_dat[1].data

g_ra, g_dec = galx_table['ra'], galx_table['dec']
g_z = galx_table['z']

obj_ID = galx_table['objid']
obj_ID = obj_ID.astype( int )

g_coord = SkyCoord( ra = g_ra * U.deg, dec = g_dec * U.deg )

all_mag_u = np.array( galx_table['modelMag_u'] )
all_mag_g = np.array( galx_table['modelMag_g'] )
all_mag_r = np.array( galx_table['modelMag_r'] )
all_mag_i = np.array( galx_table['modelMag_i'] )
all_mag_z = np.array( galx_table['modelMag_z'] )

all_dered_u = np.array( galx_table['dered_u'] )
all_dered_g = np.array( galx_table['dered_g'] )
all_dered_r = np.array( galx_table['dered_r'] )
all_dered_i = np.array( galx_table['dered_i'] )
all_dered_z = np.array( galx_table['dered_z'] )

all_cmag_u = np.array( galx_table['cModelMag_u'] )
all_cmag_g = np.array( galx_table['cModelMag_g'] )
all_cmag_r = np.array( galx_table['cModelMag_r'] )
all_cmag_i = np.array( galx_table['cModelMag_i'] )
all_cmag_z = np.array( galx_table['cModelMag_z'] )

all_Exint_u = np.array( galx_table['extinction_u'] )
all_Exint_g = np.array( galx_table['extinction_g'] )
all_Exint_r = np.array( galx_table['extinction_r'] )
all_Exint_i = np.array( galx_table['extinction_i'] )
all_Exint_z = np.array( galx_table['extinction_z'] )


##.
for pp in range( 3 ):

    band_str = band[ pp ]

    mp_cat = pds.read_csv( '/home/xkchen/data/SDSS/member_files/redMap_contral_galx/control_cat/' + 
                        'random_field-galx_map_%s-band_cat.csv' % band_str )
    mp_ra, mp_dec, mp_z = np.array( mp_cat['sat_ra'] ), np.array( mp_cat['sat_dec'] ), np.array( mp_cat['sat_z'] )

    mp_IDs = np.array( mp_cat['sat_objID'] )
    mp_IDs = mp_IDs.astype( int )

    mp_coord = SkyCoord( ra = mp_ra * U.deg, dec = mp_dec * U.deg )

    id_gx, d2d, d3d = mp_coord.match_to_catalog_sky( g_coord )
    id_lim = d2d.value < 2.7e-4

    cp_cmag_u = all_cmag_u[ id_gx[ id_lim ] ]
    cp_cmag_g = all_cmag_g[ id_gx[ id_lim ] ]
    cp_cmag_r = all_cmag_r[ id_gx[ id_lim ] ]
    cp_cmag_i = all_cmag_i[ id_gx[ id_lim ] ]
    cp_cmag_z = all_cmag_z[ id_gx[ id_lim ] ]

    cp_dered_u = all_dered_u[ id_gx[ id_lim ] ]
    cp_dered_g = all_dered_g[ id_gx[ id_lim ] ]
    cp_dered_r = all_dered_r[ id_gx[ id_lim ] ]
    cp_dered_i = all_dered_i[ id_gx[ id_lim ] ]
    cp_dered_z = all_dered_z[ id_gx[ id_lim ] ]

    cp_mag_u = all_mag_u[ id_gx[ id_lim ] ]
    cp_mag_g = all_mag_g[ id_gx[ id_lim ] ]
    cp_mag_r = all_mag_r[ id_gx[ id_lim ] ]
    cp_mag_i = all_mag_i[ id_gx[ id_lim ] ]
    cp_mag_z = all_mag_z[ id_gx[ id_lim ] ]

    cp_Exint_u = all_Exint_u[ id_gx[ id_lim ] ]
    cp_Exint_g = all_Exint_g[ id_gx[ id_lim ] ]
    cp_Exint_r = all_Exint_r[ id_gx[ id_lim ] ]
    cp_Exint_i = all_Exint_i[ id_gx[ id_lim ] ]
    cp_Exint_z = all_Exint_z[ id_gx[ id_lim ] ]


    ##.
    keys = ['ra', 'dec', 'z', 'objid', 
            'cModelMag_u', 'cModelMag_g', 'cModelMag_r', 'cModelMag_i', 'cModelMag_z', 
            'modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z', 
            'dered_u', 'dered_g', 'dered_r', 'dered_i', 'dered_z', 
            'extinction_u', 'extinction_g', 'extinction_r', 'extinction_i', 'extinction_z']

    values = [ mp_ra, mp_dec, mp_z, mp_IDs, 
            cp_cmag_u, cp_cmag_g, cp_cmag_r, cp_cmag_i, cp_cmag_z, 
            cp_mag_u, cp_mag_g, cp_mag_r, cp_mag_i, cp_mag_z, 
            cp_dered_u, cp_dered_g, cp_dered_r, cp_dered_i, cp_dered_z, 
            cp_Exint_u, cp_Exint_g, cp_Exint_r, cp_Exint_i, cp_Exint_z ]

    fill = dict( zip( keys, values ) )
    data = pds.DataFrame( fill )
    data.to_csv('/home/xkchen/data/SDSS/member_files/redMap_contral_galx/control_cat/' + 
                'random_field-galx_map_%s-band_cat_params.csv' % band_str )

raise


### === map to galaxy catalog in random image frame~()
home = '/home/xkchen/data/SDSS/'
img_file = home + 'redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

pre_N = 50

for kk in range( 3 ):

    ##. random catalog
    band_str = band[ kk ]

    #.
    c_dat = pds.read_csv('/home/xkchen/fig_tmp/random_cat/2_28/' + 
                'random_%s-band_tot_remain_cat_set_200-grid_6.0-sigma.csv' % band_str)

    c_ra, c_dec, c_z = np.array( c_dat['ra'] ), np.array( c_dat['dec'] ), np.array( c_dat['z'] )

    N_clus = len( c_ra )    

    #.
    g_dat = pds.read_csv('/home/xkchen/data/SDSS/member_files/redMap_contral_galx/control_cat/' + 
                        'random_field-galx_map_%s-band_cat.csv' % band_str)

    bcg_ra, bcg_dec, bcg_z = np.array( g_dat['bcg_ra'] ), np.array( g_dat['bcg_dec'] ), np.array( g_dat['bcg_z'] )
    sat_ra, sat_dec, sat_z = np.array( g_dat['sat_ra'] ), np.array( g_dat['sat_dec'] ), np.array( g_dat['sat_z'] )

    sat_IDs = np.array( g_dat['sat_objID'] )
    sat_x, sat_y = np.array( g_dat['sat_x'] ), np.array( g_dat['sat_y'] )

    map_coord = SkyCoord( ra = bcg_ra * U.deg, dec = bcg_dec * U.deg )

    #.
    for dd in range( pre_N ):

        #. cluster image
        ra_x, dec_x, z_x = c_ra[ dd ], c_dec[ dd ], c_z[ dd ]

        img_dat = fits.open( img_file % ( band_str, ra_x, dec_x, z_x),)
        img_arr = img_dat[0].data

        Head = img_dat[0].header
        wcs_lis = awc.WCS( Head )

        dd_coord = SkyCoord( ra = ra_x * U.deg, dec = dec_x * U.deg )

        idx, d2d, d3d = dd_coord.match_to_catalog_sky( map_coord )
        id_lim = d2d.value < 2.7e-4

        if np.sum( id_lim ) == 0:
            continue
        
        else:
            if np.sum( id_lim ) == 1:

                dd_s_ra, dd_s_dec, dd_sat_z = sat_ra[ idx ], sat_dec[ idx ], sat_z[ idx ]
                mp_sx, mp_sy = sat_x[ idx ], sat_y[ idx ]

                dd_sx, dd_sy = wcs_lis.all_world2pix( dd_s_ra, dd_s_dec, 0 )

                fig = plt.figure()
                ax = fig.add_axes([0.11, 0.11, 0.80, 0.85])

                ax.set_title( '%s-band, ra%.3f, dec%.3f, z%.3f, sat,ra%.4f, dec%.4f' % (band_str, ra_x, dec_x, z_x, dd_s_ra, dd_s_dec),)
                ax.imshow( img_arr, origin = 'lower', cmap = 'Greys', norm = mpl.colors.SymLogNorm(linthresh = 0.005, linscale = 0.01, vmin = -1e-1, vmax = 1e0, base = 10),)
                ax.scatter( dd_sx, dd_sy, marker = 'o', s = 20, edgecolors = 'r', facecolors = 'none', )

                ax.set_xlim( dd_sx - 100, dd_sx + 100 )
                ax.set_ylim( dd_sy - 100, dd_sy + 100 )

                plt.savefig('/home/xkchen/figs/clus_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.3f_dec%.3f.png' % 
                            (band_str, ra_x, dec_x, z_x, dd_s_ra, dd_s_dec), dpi = 300)
                plt.close()

            if np.sum( id_lim ) > 1:

                dd_s_ra, dd_s_dec, dd_sat_z = sat_ra[ idx[ id_lim ] ], sat_dec[ idx[ id_lim ] ], sat_z[ idx[ id_lim ] ]
                mp_sx, mp_sy = sat_x[ idx[ id_lim ] ], sat_y[ idx[ id_lim ] ]

                dd_sx, dd_sy = wcs_lis.all_world2pix( dd_s_ra, dd_s_dec, 0 )

                for pp in range( len( dd_s_ra ) ):

                    fig = plt.figure()
                    ax = fig.add_axes([0.11, 0.11, 0.80, 0.85])

                    ax.set_title( '%s-band, ra%.3f, dec%.3f, z%.3f, sat,ra%.4f, dec%.4f' % (band_str, ra_x, dec_x, z_x, dd_s_ra[pp], dd_s_dec[pp]),)
                    ax.imshow( img_arr, origin = 'lower', cmap = 'Greys', norm = mpl.colors.SymLogNorm(linthresh = 0.005, linscale = 0.01, vmin = -1e-1, vmax = 1e0, base = 10),)
                    ax.scatter( dd_sx[ pp ], dd_sy[ pp ], marker = 'o', s = 20, edgecolors = 'r', facecolors = 'none', )

                    ax.set_xlim( dd_sx[ pp ] - 100, dd_sx[ pp ] + 100 )
                    ax.set_ylim( dd_sy[ pp ] - 100, dd_sy[ pp ] + 100 )

                    plt.savefig('/home/xkchen/figs/clus_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.3f_dec%.3f.png' % 
                                (band_str, ra_x, dec_x, z_x, dd_s_ra[pp], dd_s_dec[pp]), dpi = 300)
                    plt.close()

