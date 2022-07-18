import h5py
import numpy as np
import pandas as pds

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord


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


### === funcs
def mem_match_func( img_cat_file, mem_cat_file, out_sat_file ):

    ##. cluster cat
    dat = pds.read_csv( img_cat_file )

    ref_ra, ref_dec, ref_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
    ref_Rvir, ref_rich = np.array( dat['R_vir'] ), np.array( dat['rich'] )
    ref_clust_ID = np.array( dat['clust_ID'] )

    Ns = len( ref_ra )


    ##. satellite samples
    s_dat = pds.read_csv( mem_cat_file )

    bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
    p_ra, p_dec, p_zspec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] ), np.array( s_dat['z_spec'])
    R_sat, R_sat2Rv = np.array( s_dat['R_cen'] ), np.array( s_dat['Rcen/Rv'] )

    p_gr, p_ri, p_gi = np.array( s_dat['g-r'] ), np.array( s_dat['r-i'] ), np.array( s_dat['g-i'] )
    p_clus_ID = np.array( s_dat['clus_ID'] )
    p_clus_ID = p_clus_ID.astype( int )


    ##. member find
    sat_ra, sat_dec, sat_z = np.array([]), np.array([]), np.array([])
    sat_Rcen, sat_gr, sat_ri, sat_gi = np.array([]), np.array([]), np.array([]), np.array([])

    sat_R2Rv = np.array([])
    sat_host_ID = np.array([])

    cp_bcg_ra, cp_bcg_dec, cp_bcg_z = np.array([]), np.array([]), np.array([])

    ##. match
    for pp in range( Ns ):

        id_px = p_clus_ID == ref_clust_ID[ pp ]

        cut_ra, cut_dec, cut_z = p_ra[ id_px ], p_dec[ id_px ], p_zspec[ id_px ]

        cut_R2Rv = R_sat2Rv[ id_px ]

        cut_gr, cut_ri, cut_gi = p_gr[ id_px ], p_ri[ id_px ], p_gi[ id_px ]
        cut_Rcen = R_sat[ id_px ]

        #. record array
        sat_ra = np.r_[ sat_ra, cut_ra ]
        sat_dec = np.r_[ sat_dec, cut_dec ]
        sat_z = np.r_[ sat_z, cut_z ]

        sat_Rcen = np.r_[ sat_Rcen, cut_Rcen ]
        sat_R2Rv = np.r_[ sat_R2Rv, cut_R2Rv ]

        sat_gr = np.r_[ sat_gr, cut_gr ]
        sat_ri = np.r_[ sat_ri, cut_ri ]
        sat_gi = np.r_[ sat_gi, cut_gi ]

        sat_host_ID = np.r_[ sat_host_ID, np.ones( len(cut_ra),) * ref_clust_ID[pp] ]

        cp_bcg_ra = np.r_[ cp_bcg_ra, np.ones( len(cut_ra),) * ref_ra[pp] ]
        cp_bcg_dec = np.r_[ cp_bcg_dec, np.ones( len(cut_ra),) * ref_dec[pp] ]
        cp_bcg_z = np.r_[ cp_bcg_z, np.ones( len(cut_ra),) * ref_z[pp] ]

    #. save member infor
    keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'ra', 'dec', 'z_spec', 'Rcen/Rv', 'R_cen', 'g-r', 'r-i', 'g-i', 'clus_ID' ]
    values = [ cp_bcg_ra, cp_bcg_dec, cp_bcg_z, sat_ra, sat_dec, sat_z, sat_R2Rv, sat_Rcen, sat_gr, sat_ri, sat_gi, sat_host_ID ]
    fill = dict( zip( keys, values) )
    out_data = pds.DataFrame( fill )
    out_data.to_csv( out_sat_file )

    return


def sat_clust_shufl_func( clust_cat, member_cat, out_file ):

    ##. cluster catalog

    cat = pds.read_csv( clust_cat, )
    set_IDs = np.array( cat['clust_ID'] )
    set_IDs = set_IDs.astype( int )


    ##. cluster member catalog, satellite location table
    dat = pds.read_csv( member_cat, )

    bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
    sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

    bcg_x, bcg_y = np.array( dat['bcg_x'] ), np.array( dat['bcg_y'] )
    sat_x, sat_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )
    sat_PA2bcg = np.array( dat['sat_PA2bcg'] )

    clus_IDs = np.array( dat['clus_ID'] )
    clus_IDs = clus_IDs.astype( int )

    N_sat = len( sat_ra )


    pre_R_sat = np.zeros( N_sat, )

    #. record the finally mapped cluster ID and the location of satellite in that image
    shufl_IDs = np.zeros( N_sat, dtype = int )
    shufl_ra, shufl_dec, shufl_z = np.zeros( N_sat,), np.zeros( N_sat,), np.zeros( N_sat,)

    shufl_sx, shufl_sy = np.zeros( N_sat, ), np.zeros( N_sat, )
    shufl_Rpix, shufl_PA =  np.zeros( N_sat, ), np.zeros( N_sat, )
    shufl_R_phy = np.zeros( N_sat, )    ###. in unit kpc

    #. record these points is symmetry (id_symP = 1,2,3) or not ( id_symP = 0 )
    id_symP = np.zeros( N_sat, dtype = int )

    #.
    for qq in range( N_sat ):

        ra_g, dec_g, z_g = bcg_ra[ qq ], bcg_dec[ qq ], bcg_z[ qq ]
        sub_ra, sub_dec = sat_ra[ qq ], sat_dec[ qq ]

        x_cen, y_cen = bcg_x[ qq ], bcg_y[ qq ]
        x_sat, y_sat = sat_x[ qq ], sat_y[ qq ]

        sat_Rpix = np.sqrt( (x_sat - x_cen)**2 + (y_sat - y_cen)**2 )  ### in pixel number
        sat_theta = np.arctan2( (y_sat - y_cen), (x_sat - x_cen) )

        Da_obs = Test_model.angular_diameter_distance( z_g ).value
        qq_R_sat = sat_Rpix * pixel * Da_obs * 1e3 / rad2arcsec     ##. kpc
        pre_R_sat[ qq ] = qq_R_sat


        ##. random select cluster and do a map between satellite and the 'selected' cluster
        qq_ID = clus_IDs[ qq ]

        id_vx = set_IDs == qq_ID
        ID_dex = np.where( id_vx )[0][0]

        copy_IDs = np.delete( set_IDs, ID_dex )
        cp_Ns = len( copy_IDs )

        #. random order of index in the cluster array
        rand_arr = np.random.choice( cp_Ns, cp_Ns, replace = False )


        ##. try to map satellite and the 'selected' cluster
        id_loop = 0
        id_live = 1

        while id_live > 0:

            rand_ID = copy_IDs[ rand_arr[ id_loop ] ]

            id_ux = clus_IDs == rand_ID
            cp_ra_g, cp_dec_g, cp_z_g = bcg_ra[ id_ux ][0], bcg_dec[ id_ux ][0], bcg_z[ id_ux ][0]			

            cp_Da = Test_model.angular_diameter_distance( cp_z_g ).value

            cp_sat_Rpix = ( ( qq_R_sat / 1e3 / cp_Da ) * rad2arcsec ) / pixel

            off_x, off_y = cp_sat_Rpix * np.cos( sat_theta ), cp_sat_Rpix * np.sin( sat_theta )

            cp_cx, cp_cy = bcg_x[ id_ux ][0], bcg_y[ id_ux ][0]

            cp_sx, cp_sy = cp_cx + off_x, cp_cy + off_y   ## new satellite location

            id_bond_x = ( cp_sx >= 0 ) & ( cp_sx < 2048 )
            id_bond_y = ( cp_sy >= 0 ) & ( cp_sy < 1489 )
            id_bond = id_bond_x & id_bond_y

            if id_bond:

                shufl_IDs[ qq ] = rand_ID
                shufl_sx[ qq ] = cp_sx
                shufl_sy[ qq ] = cp_sy
                shufl_Rpix[ qq ] = cp_sat_Rpix
                shufl_PA[ qq ] = sat_theta

                shufl_ra[ qq ] = cp_ra_g
                shufl_dec[ qq ] = cp_dec_g
                shufl_z[ qq ] = cp_z_g

                cp_R_phy = ( cp_sat_Rpix * pixel * cp_Da * 1e3 ) / rad2arcsec

                shufl_R_phy[ qq ] = cp_R_phy
                id_symP[ qq ] = 0

                id_live = id_live - 1
                break

            else:

                tmp_phi = np.array( [ 	np.pi + sat_theta, 
                                        np.pi - sat_theta, 
                                        np.pi * 2 - sat_theta ] )

                tmp_off_x = cp_sat_Rpix * np.cos( tmp_phi )
                tmp_off_y = cp_sat_Rpix * np.sin( tmp_phi )

                tmp_sx, tmp_sy = cp_cx + tmp_off_x, cp_cy + tmp_off_y				

                id_lim_x = ( tmp_sx >= 0 ) & ( tmp_sx < 2048 )
                id_lim_y = ( tmp_sy >= 0 ) & ( tmp_sy < 1489 )
                id_lim = id_lim_x & id_lim_y

                if np.sum( id_lim ) < 1 :  ## no points located in image frame

                    id_loop = id_loop + 1
                    continue

                else:

                    rt0 = np.random.choice( np.sum( id_lim ), np.sum( id_lim), replace = False)

                    shufl_IDs[ qq ] = rand_ID
                    shufl_sx[ qq ] = tmp_sx[ id_lim ][ rt0[ 0 ] ]
                    shufl_sy[ qq ] = tmp_sy[ id_lim ][ rt0[ 0 ] ]

                    shufl_Rpix[ qq ] = cp_sat_Rpix
                    shufl_PA[ qq ] = tmp_phi[ id_lim ][ rt0[ 0 ] ]

                    shufl_ra[ qq ] = cp_ra_g
                    shufl_dec[ qq ] = cp_dec_g
                    shufl_z[ qq ] = cp_z_g

                    cp_R_phy = ( cp_sat_Rpix * pixel * cp_Da * 1e3 ) / rad2arcsec

                    shufl_R_phy[ qq ] = cp_R_phy

                    #. symmetry point record
                    ident_x0 = np.where( id_lim )[0]
                    ident_x1 = ident_x0[ rt0[ 0 ] ]

                    id_symP[ qq ] = ident_x1 + 1

                    id_live = id_live - 1
                    break


    ##. save table	
    keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'orin_cID', 'orin_Rsat_phy', 
                'shufl_cID', 'cp_sx', 'cp_sy', 'cp_PA', 'cp_Rpix', 'cp_Rsat_phy', 'is_symP',
                'cp_bcg_ra', 'cp_bcg_dec', 'cp_bcg_z' ]

    values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, clus_IDs, pre_R_sat, 
            shufl_IDs, shufl_sx, shufl_sy, shufl_PA, shufl_Rpix, shufl_R_phy, id_symP,
            shufl_ra, shufl_dec, shufl_z ]

    fill = dict( zip( keys, values ) )
    out_data = pds.DataFrame( fill )
    out_data.to_csv( out_file )

    return


### === subsample shuffle list mapping
def shufl_list_map_func( cat_file, clust_file, N_shufl, shufl_list_file, out_shufl_file, 
                        oirn_img_pos_file, out_pos_file, out_Ng_file, list_idx = None):

    #. satellite catalog
    dat = pds.read_csv( cat_file,)
    bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
    sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

    pre_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )

    N_sat = len( sat_ra )

    if list_idx is not None:
        da0 = list_idx
        da1 = list_idx + 1

    else:
        da0 = 0
        da1 = N_shufl


    for dd in range( da0, da1 ):

        #.
        pat = pds.read_csv( shufl_list_file % dd,)

        keys = pat.columns[1:]

        kk_ra, kk_dec = np.array( pat['sat_ra'] ), np.array( pat['sat_dec'] )
        kk_IDs = np.array( pat['orin_cID'] )

        kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

        idx, sep, d3d = pre_coord.match_to_catalog_sky( kk_coord )
        id_lim = sep.value < 2.7e-4

        N_ks = len( keys )

        #. record matched information
        tmp_arr = []
        N_ks = len( keys )

        for pp in range( N_ks ):

            sub_arr = np.array( pat['%s' % keys[pp] ] )
            tmp_arr.append( sub_arr[ idx[ id_lim ] ] )

        ##.save
        fill = dict( zip( keys, tmp_arr ) )
        data = pds.DataFrame( fill )
        data.to_csv( out_shufl_file % dd,)


    #. image cut information match
    for dd in range( da0, da1 ):

        ##. T200 catalog
        dat = pds.read_csv( out_shufl_file % dd,)
        bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
        sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

        pre_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )

        N_sat = len( sat_ra )

        host_IDs = np.array( dat['orin_cID'] )
        rand_IDs = np.array( dat['shufl_cID'] )

        host_IDs = host_IDs.astype( int )
        rand_IDs = rand_IDs.astype( int )


        ##.==. origin image location match
        pat = pds.read_csv( oirn_img_pos_file,)
        keys = pat.columns[1:]

        kk_ra, kk_dec = np.array( pat['sat_ra'] ), np.array( pat['sat_dec'] )
        kk_IDs = np.array( pat['clus_ID'] )

        kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

        idx, sep, d3d = pre_coord.match_to_catalog_sky( kk_coord )
        id_lim = sep.value < 2.7e-4

        mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]
        mp_IDs = kk_IDs[ idx[ id_lim ] ]

        #. record matched information
        tmp_arr = []
        N_ks = len( keys )

        for pp in range( N_ks ):

            sub_arr = np.array( pat['%s' % keys[pp] ] )
            tmp_arr.append( sub_arr[ idx[ id_lim ] ] )

        ##.save
        fill = dict( zip( keys, tmp_arr ) )
        data = pds.DataFrame( fill )
        data.to_csv( out_pos_file % dd,)


    #. count the number of satellite before and after shuffling
    for dd in range( da0, da1 ):

        ##. T200 catalog
        dat = pds.read_csv( out_shufl_file % dd,)
        bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
        sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

        pre_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )

        N_sat = len( sat_ra )

        host_IDs = np.array( dat['orin_cID'] )
        rand_IDs = np.array( dat['shufl_cID'] )

        host_IDs = host_IDs.astype( int )
        rand_IDs = rand_IDs.astype( int )


        ##. entire cluster catalog
        cat = pds.read_csv( clust_file,)
        set_IDs = np.array( cat['clust_ID'] )	
        set_IDs = set_IDs.astype( int )

        N_cs = len( set_IDs )

        ##. satellite number counts
        tmp_clus_Ng = np.zeros( N_sat, )
        tmp_shufl_Ng = np.zeros( N_sat, )

        for pp in range( N_cs ):

            sub_IDs = set_IDs[ pp ]

            id_vx = host_IDs == sub_IDs
            id_ux = rand_IDs == sub_IDs

            tmp_clus_Ng[ id_vx ] = np.sum( id_vx )
            tmp_shufl_Ng[ id_ux ] = np.sum( id_ux )

        #. save
        keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'orin_Ng', 'shufl_Ng']
        values = [bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, tmp_clus_Ng, tmp_shufl_Ng ]
        fill = dict( zip( keys, values ) )
        data = pds.DataFrame( fill )
        data.to_csv( out_Ng_file % dd,)

    return


### === ??? make sure the distribution relative to BCG is the same~(before and after shuffling)
##. 

