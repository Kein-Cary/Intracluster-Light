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


### === match func
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


### === data load
def clust_member_match():

    cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/cat/'
    out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_all_Pm/cat/'

    bin_rich = [ 20, 30, 50, 210 ]

    ##. cluster match to satellites
    for kk in range( len(bin_rich) - 1 ):

        img_cat_file = cat_path + 'clust_rich_%d-%d_cat.csv' % ( bin_rich[kk], bin_rich[kk + 1])
        mem_cat_file = out_path + 'Extend-BCGM_rgi-common_frame-lim_lower-Pm_member-cat.csv'
        out_sat_file = out_path + 'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_member-cat.csv' % ( bin_rich[kk], bin_rich[kk + 1])

        mem_match_func( img_cat_file, mem_cat_file, out_sat_file )


    ### === tables for Background stacking
    ##. cluster member match with satellites position
    for kk in range( len(bin_rich) - 1 ):

        dat = pds.read_csv( out_path + 
                    'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_member-cat.csv' % ( bin_rich[kk], bin_rich[kk + 1]),)

        bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
        sat_ra, sat_dec = np.array( dat['ra'] ), np.array( dat['dec'] )
        clus_IDs = np.array( dat['clus_ID'] )

        sub_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )

        for tt in range( 3 ):

            band_str = band[ tt ]

            pat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/shufle_test/img_tract_cat/' + 
                'Extend-BCGM_rgi-common_frame-limit_exlu-BCG_Sat_%s-band_origin-img_position.csv' % band_str )

            kk_ra, kk_dec = np.array( pat['sat_ra'] ), np.array( pat['sat_dec'] )
            kk_bcg_x, kk_bcg_y = np.array( pat['bcg_x'] ), np.array( pat['bcg_y'] )
            kk_sat_x, kk_sat_y = np.array( pat['sat_x'] ), np.array( pat['sat_y'] )
            kk_PA = np.array( pat['sat_PA2bcg'] )

            kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

            #. match P_mem cut sample
            idx, sep, d3d = sub_coord.match_to_catalog_sky( kk_coord )
            id_lim = sep.value < 2.7e-4

            #. satellite information (for check)
            mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]
            mp_bcg_x, mp_bcg_y = kk_bcg_x[ idx[ id_lim ] ], kk_bcg_y[ idx[ id_lim ] ]
            mp_sat_x, mp_sat_y = kk_sat_x[ idx[ id_lim ] ], kk_sat_y[ idx[ id_lim ] ]
            mp_sat_PA = kk_PA[ idx[ id_lim ] ]

            #. save
            keys = pat.columns[1:]
            mp_arr = [ bcg_ra, bcg_dec, bcg_z, mp_ra, mp_dec, mp_bcg_x, mp_bcg_y, 
                                        mp_sat_x, mp_sat_y, mp_sat_PA, clus_IDs ]
            fill = dict( zip( keys, mp_arr ) )
            out_data = pds.DataFrame( fill )
            out_data.to_csv( out_path + 
                    'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_sat_%s-band_origin-img_position.csv' 
                    % (bin_rich[kk], bin_rich[kk + 1], band_str),)


    ##. cluster member match with the cutout information at z_obs and z-ref
    for kk in range( len(bin_rich) - 1 ):

        dat = pds.read_csv( out_path + 
                    'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_member-cat.csv' % ( bin_rich[kk], bin_rich[kk + 1]),)

        bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
        sat_ra, sat_dec = np.array( dat['ra'] ), np.array( dat['dec'] )
        clus_IDs = np.array( dat['clus_ID'] )

        sub_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )

        for tt in range( 3 ):

            band_str = band[ tt ]

            ##. satellite location and cutout at z_obs
            dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/' + 
                    'Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos.csv' % band_str,)
            kk_ra, kk_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
            kk_imgx, kk_imgy = np.array( dat['cut_cx'] ), np.array( dat['cut_cy'] )

            kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

            idx, sep, d3d = sub_coord.match_to_catalog_sky( kk_coord )
            id_lim = sep.value < 2.7e-4

            mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]
            mp_imgx, mp_imgy = kk_imgx[ idx[ id_lim ] ], kk_imgy[ idx[ id_lim ] ]

            keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'cut_cx', 'cut_cy']
            values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, mp_imgx, mp_imgy ]
            fill = dict( zip( keys, values ) )
            data = pds.DataFrame( fill )
            data.to_csv( out_path + 
                    'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_sat_%s-band_member_pos.csv' 
                    % (bin_rich[kk], bin_rich[kk + 1], band_str),)


            ##. satellite location and cutout at z_ref
            dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/' + 
                    'Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos_z-ref.csv' % band[kk] )
            kk_ra, kk_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
            kk_imgx, kk_imgy = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

            kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

            idx, sep, d3d = sub_coord.match_to_catalog_sky( kk_coord )
            id_lim = sep.value < 2.7e-4

            mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]
            mp_imgx, mp_imgy = kk_imgx[ idx[ id_lim ] ], kk_imgy[ idx[ id_lim ] ]

            keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_x', 'sat_y']
            values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, mp_imgx, mp_imgy ]
            fill = dict( zip( keys, values ) )
            data = pds.DataFrame( fill )
            data.to_csv( out_path + 
                    'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_sat_%s-band_member_pos-zref.csv' 
                    % (bin_rich[kk], bin_rich[kk + 1], band_str),)

    return
    
# clust_member_match()
# raise


### === figs
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_all_Pm/cat/'

bin_rich = [ 20, 30, 50, 210 ]
fig_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']

line_c = ['b', 'g', 'r']


##. satellite properties
pre_cat = fits.open('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')
pre_table = pre_cat[1].data
pre_ra, pre_dec = np.array( pre_table['RA'] ), np.array( pre_table['DEC'] )
pre_Pm = np.array( pre_table['P'] )

pre_coord = SkyCoord( ra = pre_ra * U.deg, dec = pre_dec * U.deg )


fig = plt.figure( figsize = (12, 6),)
ax0 = plt.subplot( 121 )
ax1 = plt.subplot( 122 )

for kk in range( len(bin_rich) - 1 ):
    
    ##.
    s_dat = pds.read_csv( out_path + 
            'clust_rich_%d-%d_rgi-common_frame-lim_lower-Pm_member-cat.csv' % ( bin_rich[kk], bin_rich[kk + 1]),)

    bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
    p_ra, p_dec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] )
    clus_IDs = np.array( s_dat['clus_ID'] )

    p_Rsat = np.array( s_dat['R_cen'] )
    p_R2Rv = np.array( s_dat['Rcen/Rv'] )

    a_obs = 1 / (1 + bcg_z)

    p_Rsat = p_Rsat * 1e3 * a_obs / h    ##. physical radius

    ##. Pm match
    p_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

    idx, sep, d3d = p_coord.match_to_catalog_sky( pre_coord )
    id_lim = sep.value < 2.7e-4 

    mp_Pm = pre_Pm[ idx[ id_lim ] ]

    ##.
    ax0.hist( p_Rsat, bins = 55, density = True, color = line_c[kk], histtype = 'step', label = fig_name[kk],)
    ax0.hist( p_Rsat * mp_Pm, bins = 55, density = True, color = line_c[kk], ls = '--', histtype = 'step', 
            label = fig_name[kk] + ', with $P_{m}$ weight',)

    ax1.hist( p_R2Rv, bins = 55, density = True, color = line_c[kk], histtype = 'step', label = fig_name[kk],)
    ax1.hist( p_R2Rv * mp_Pm, bins = 55, density = True, color = line_c[kk], ls = '--', histtype = 'step',)

    ax1.hist( p_R2Rv, bins = 55, density = True, weights = mp_Pm, color = 'k', ls = '-', histtype = 'step', alpha = 0.5,)

ax0.legend( loc = 1, frameon = False, fontsize = 12,)
ax0.set_xlabel('$R_{sat} \;[kpc]$', fontsize = 12,)
ax0.set_xlim( 0, 1600 )

ax1.set_xlabel('$R_{sat} / R_{200m}$', fontsize = 12,)
ax1.legend( loc = 1, frameon = False, fontsize = 12, )
ax1.set_xlim( 0, 1.4 )

plt.savefig('/home/xkchen/low-Pm_sat_Rs_hist.png', dpi = 300)
plt.close()

