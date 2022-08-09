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


### === cosmology
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


### === ### scaled Radius bin

def sat_scaleR_binned():

    out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_all_Pm/cat/'

    ##. fixed R for all richness subsample
    R_bins = np.array( [0, 1e-1, 2e-1, 3e-1, 4.5e-1, 1] )   ### times R200m

    bin_rich = [ 20, 30, 50, 210 ]

    ##... radius binned satellite
    for kk in range( 3 ):

        ##.
        s_dat = pds.read_csv( out_path + 
                    'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_member-cat.csv' % ( bin_rich[kk], bin_rich[kk + 1]),)

        bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
        p_ra, p_dec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] )

        p_Rsat = np.array( s_dat['R_cen'] )
        p_R2Rv = np.array( s_dat['Rcen/Rv'] )
        clus_IDs = np.array( s_dat['clus_ID'] )

        ##. division
        for nn in range( len( R_bins ) - 1 ):

            if nn == len( R_bins ) - 2:
                sub_N = p_R2Rv >= R_bins[ nn ]
            else:
                sub_N = (p_R2Rv >= R_bins[ nn ]) & (p_R2Rv < R_bins[ nn + 1])

            ##. save
            out_c_ra, out_c_dec, out_c_z = bcg_ra[ sub_N ], bcg_dec[ sub_N ], bcg_z[ sub_N ]
            out_s_ra, out_s_dec = p_ra[ sub_N ], p_dec[ sub_N ]
            out_Rsat = p_Rsat[ sub_N ]
            out_R2Rv = p_R2Rv[ sub_N ]
            out_clus_ID = clus_IDs[ sub_N ]

            keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'R_sat', 'R2Rv', 'clus_ID']
            values = [ out_c_ra, out_c_dec, out_c_z, out_s_ra, out_s_dec, out_Rsat, out_R2Rv, out_clus_ID]
            fill = dict( zip( keys, values ) )
            data = pds.DataFrame( fill )

            data.to_csv( out_path + 
                        'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_%.2f-%.2fR200m_mem_cat.csv' 
                        % ( bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1]),)


    ##... match the P_mem information
    pre_cat = fits.open('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')
    pre_table = pre_cat[1].data
    pre_ra, pre_dec = np.array( pre_table['RA'] ), np.array( pre_table['DEC'] )
    pre_Pm = np.array( pre_table['P'] )

    pre_coord = SkyCoord( ra = pre_ra * U.deg, dec = pre_dec * U.deg )

    for kk in range( 3 ):

        for nn in range( len( R_bins ) - 1 ):

            dat = pds.read_csv( out_path + 
                        'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_%.2f-%.2fR200m_mem_cat.csv' 
                        % ( bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1]), )

            nn_bcg_ra, nn_bcg_dec = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] )
            nn_bcg_z = np.array( dat['bcg_z'] )
            p_ra, p_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

            ##. Pm match
            p_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

            idx, sep, d3d = p_coord.match_to_catalog_sky( pre_coord )
            id_lim = sep.value < 2.7e-4 

            mp_Pm = pre_Pm[ idx[ id_lim ] ]

            ##. save
            keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'P_mem']
            values = [ nn_bcg_ra, nn_bcg_dec, nn_bcg_z, p_ra, p_dec, mp_Pm ] 
            fill = dict( zip( keys, values ) )
            data = pds.DataFrame( fill )

            data.to_csv( out_path + 
                        'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_%.2f-%.2fR200m_mem_Pm_cat.csv' 
                        % ( bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1]),)

    raise

    ##... match with stacked information
    pos_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/'

    for pp in range( 3 ):

        for tt in range( len( R_bins ) - 1 ):

            s_dat = pds.read_csv( out_path + 'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_%.2f-%.2fR200m_mem_cat.csv' % 
                                ( bin_rich[pp], bin_rich[pp + 1], R_bins[tt], R_bins[tt + 1]), )

            bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
            p_ra, p_dec = np.array( s_dat['sat_ra'] ), np.array( s_dat['sat_dec'] )	

            pre_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

            ##.
            for kk in range( 3 ):

                #. z-ref position
                dat = pds.read_csv( pos_path + 'Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos_z-ref.csv' % band[kk])
                kk_ra, kk_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
                kk_imgx, kk_imgy = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

                kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

                idx, sep, d3d = pre_coord.match_to_catalog_sky( kk_coord )
                id_lim = sep.value < 2.7e-4

                mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]	
                mp_imgx, mp_imgy = kk_imgx[ idx[ id_lim ] ], kk_imgy[ idx[ id_lim ] ]

                keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_x', 'sat_y']
                values = [ bcg_ra, bcg_dec, bcg_z, p_ra, p_dec, mp_imgx, mp_imgy ]
                fill = dict( zip( keys, values ) )
                data = pds.DataFrame( fill )
                data.to_csv( out_path + 
                            'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_%.2f-%.2fR200m_mem-%s-band_pos-zref.csv' 
                            % (bin_rich[pp], bin_rich[pp + 1], R_bins[tt], R_bins[tt + 1], band[kk]),)

    return


### === physical radius bin
def sat_phyR_binned():

    out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_all_Pm/cat/'

    ##. fixed R for all richness subsample
    R_bins = np.array( [ 0, 300, 400, 550, 5000] )
    R_bins = [ R_bins ] * 3


    bin_rich = [ 20, 30, 50, 210 ]

    ##. radius binned satellite
    for kk in range( 3 ):

        ##.
        s_dat = pds.read_csv( out_path + 
                    'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_member-cat.csv' % ( bin_rich[kk], bin_rich[kk + 1]),)

        bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
        p_ra, p_dec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] )

        p_Rsat = np.array( s_dat['R_cen'] )
        p_R2Rv = np.array( s_dat['Rcen/Rv'] )
        clus_IDs = np.array( s_dat['clus_ID'] )

        a_obs = 1 / ( bcg_z + 1 )

        cp_Rsat = p_Rsat * 1e3 * a_obs / h  ##. physical radius


        ##. division
        for nn in range( len( R_bins[0] ) - 1 ):

            if nn == len( R_bins[0] ) - 2:
                sub_N = cp_Rsat >= R_bins[kk][ nn ]
            else:
                sub_N = (cp_Rsat >= R_bins[kk][ nn ]) & (cp_Rsat < R_bins[kk][ nn + 1])

            ##. save
            out_c_ra, out_c_dec, out_c_z = bcg_ra[ sub_N ], bcg_dec[ sub_N ], bcg_z[ sub_N ]
            out_s_ra, out_s_dec = p_ra[ sub_N ], p_dec[ sub_N ]
            out_Rsat = p_Rsat[ sub_N ]
            out_R2Rv = p_R2Rv[ sub_N ]
            out_clus_ID = clus_IDs[ sub_N ]

            keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'R_sat', 'R2Rv', 'clus_ID'] 
            values = [ out_c_ra, out_c_dec, out_c_z, out_s_ra, out_s_dec, out_Rsat, out_R2Rv, out_clus_ID]
            fill = dict( zip( keys, values ) )
            data = pds.DataFrame( fill )
            data.to_csv( out_path + 
                'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_phyR_%d-%dkpc_mem_cat.csv' % 
                ( bin_rich[kk], bin_rich[kk + 1], R_bins[kk][nn], R_bins[kk][nn + 1]),)


    ##... match the P_mem information
    pre_cat = fits.open('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')
    pre_table = pre_cat[1].data
    pre_ra, pre_dec = np.array( pre_table['RA'] ), np.array( pre_table['DEC'] )
    pre_Pm = np.array( pre_table['P'] )

    pre_coord = SkyCoord( ra = pre_ra * U.deg, dec = pre_dec * U.deg )

    for kk in range( 3 ):

        for nn in range( len( R_bins ) - 1 ):

            dat = pds.read_csv( out_path + 
                'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_phyR_%d-%dkpc_mem_cat.csv' % 
                ( bin_rich[kk], bin_rich[kk + 1], R_bins[kk][nn], R_bins[kk][nn + 1]), )

            nn_bcg_ra, nn_bcg_dec = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] )
            nn_bcg_z = np.array( dat['bcg_z'] )
            p_ra, p_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

            ##. Pm match
            p_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

            idx, sep, d3d = p_coord.match_to_catalog_sky( pre_coord )
            id_lim = sep.value < 2.7e-4 

            mp_Pm = pre_Pm[ idx[ id_lim ] ]

            ##. save
            keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'P_mem']
            values = [ nn_bcg_ra, nn_bcg_dec, nn_bcg_z, p_ra, p_dec, mp_Pm ]
            fill = dict( zip( keys, values ) )
            data = pds.DataFrame( fill )

            data.to_csv( out_path + 
                'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_phyR_%d-%dkpc_mem_Pm_cat.csv' % 
                ( bin_rich[kk], bin_rich[kk + 1], R_bins[kk][nn], R_bins[kk][nn + 1]), )

    raise

    ##... match with stacked information
    pos_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/'

    for pp in range( 3 ):

        for tt in range( len( R_bins[0] ) - 1 ):

            s_dat = pds.read_csv( out_path + 'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_phyR_%d-%dkpc_mem_cat.csv' % 
                                ( bin_rich[pp], bin_rich[pp + 1], R_bins[pp][tt], R_bins[pp][tt + 1]), )

            bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
            p_ra, p_dec = np.array( s_dat['sat_ra'] ), np.array( s_dat['sat_dec'] )	

            pre_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

            ##.
            for kk in range( 3 ):

                #. z-ref position
                dat = pds.read_csv( pos_path + 'Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos_z-ref.csv' % band[kk])
                kk_ra, kk_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
                kk_imgx, kk_imgy = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

                kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

                idx, sep, d3d = pre_coord.match_to_catalog_sky( kk_coord )
                id_lim = sep.value < 2.7e-4

                mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]	
                mp_imgx, mp_imgy = kk_imgx[ idx[ id_lim ] ], kk_imgy[ idx[ id_lim ] ]

                keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_x', 'sat_y']
                values = [ bcg_ra, bcg_dec, bcg_z, p_ra, p_dec, mp_imgx, mp_imgy ]
                fill = dict( zip( keys, values ) )
                data = pds.DataFrame( fill )
                data.to_csv( out_path + 
                            'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_phyR_%d-%dkpc_mem-%s-band_pos-zref.csv' 
                            % (bin_rich[pp], bin_rich[pp + 1], R_bins[pp][tt], R_bins[pp][tt + 1], band[kk]),)

    return

##.
# sat_scaleR_binned()

sat_phyR_binned()
