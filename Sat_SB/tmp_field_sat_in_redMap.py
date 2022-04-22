"""
for given random cluster in redMaPPer, using this file to find the member galaxies and
query or match their properties (i.e. Mstar, color, model and cModel magnitudes, centric distance,
location on image frame and cut region)
"""
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

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

#.
from Mass_rich_radius import rich2R_Simet
from img_sat_fig_out_mode import zref_sat_pos_func


### === cosmology model
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


### === 
def frame_limit_mem_match_func( img_cat_file, galx_cat_file, img_file, band_lis, out_galx_file):
    """
    img_cat_file : catalog('.csv') includes cluster information, need to find frame limited satellites
    galx_cat_file : all galaxies in the sky coverage of SDSS
    img_file : images corresponding to the random cluster catalog
    out_galx_file : image + galaxy matched catalog
    """

    #. random cluster catalog image cat
    dat = pds.read_csv( img_cat_file )
    ref_ra, ref_dec, ref_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

    #. order the random cluster
    Ns = len( ref_ra )

    list_ord = np.arange( Ns )
    ref_clust_ID = list_ord.astype( int )


    #. 'member' match~( here may not be member, but the galaxy within a given image frame)
    sat_ra, sat_dec, sat_z = np.array([]), np.array([]), np.array([])
    sat_gr, sat_ri, sat_gi = np.array([]), np.array([]), np.array([])
    sat_objID = np.array([])

    sat_host_ID = np.array([])  ##. here the ID is the order for cluster match
    cp_bcg_ra, cp_bcg_dec, cp_bcg_z = np.array([]), np.array([]), np.array([])


    for pp in range( Ns ):

        #. galaxy cat. load
        galx_dat = fits.open( galx_cat_file )
        pre_dat = galx_dat[1].data

        sub_ra, sub_dec, sub_z = np.array( pre_dat['ra'] ), np.array( pre_dat['dec'] ), np.array( pre_dat['z_spec'] )

        sub_objID = np.array( pre_dat['objID'] )
        sub_rmag = np.array( pre_dat['model_r'] )
        sub_gmag = np.array( pre_dat['model_g'] )
        sub_imag = np.array( pre_dat['model_i'] )

        sub_gr, sub_ri, sub_gi = sub_gmag - sub_rmag, sub_rmag - sub_imag, sub_gmag - sub_imag


        #. img frame load
        ra_g, dec_g, z_g = ref_ra[ pp ], ref_dec[ pp ], ref_z[ pp ]

        img_dat = fits.open( img_file % (band_lis, ra_g, dec_g, z_g), )
        Header = img_dat[0].header
        img_arr = img_dat[0].data

        wcs_lis = awc.WCS( Header )
        x_pos, y_pos = wcs_lis.all_world2pix( sub_ra, sub_dec, 0)

        id_x0 = ( x_pos >= 0 ) & ( x_pos <= 2047 )
        id_y0 = ( y_pos >= 0 ) & ( y_pos <= 1488 )
        id_lim = id_x0 & id_y0

        cut_ra, cut_dec, cut_z = sub_ra[ id_lim ], sub_dec[ id_lim ], sub_z[ id_lim ]
        cut_gr, cut_ri, cut_gi = sub_gr[ id_lim ], sub_ri[ id_lim ], sub_gi[ id_lim ]
        cut_objID = sub_objID[ id_lim ]


        #. record array
        sat_ra = np.r_[ sat_ra, cut_ra ]
        sat_dec = np.r_[ sat_dec, cut_dec ]
        sat_z = np.r_[ sat_z, cut_z ]

        sat_gr = np.r_[ sat_gr, cut_gr ]
        sat_ri = np.r_[ sat_ri, cut_ri ]
        sat_gi = np.r_[ sat_gi, cut_gi ]

        sat_objID = np.r_[ sat_objID, cut_objID ]

        sat_host_ID = np.r_[ sat_host_ID, np.ones( len(cut_ra),) * ref_clust_ID[pp] ]
        cp_bcg_ra = np.r_[ cp_bcg_ra, np.ones( len(cut_ra),) * ref_ra[pp] ]
        cp_bcg_dec = np.r_[ cp_bcg_dec, np.ones( len(cut_ra),) * ref_dec[pp] ]
        cp_bcg_z = np.r_[ cp_bcg_z, np.ones( len(cut_ra),) * ref_z[pp] ]

    #. save member infor
    keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'ra', 'dec', 'z_spec', 'g-r', 'r-i', 'g-i', 'clus_ID' ]
    values = [ cp_bcg_ra, cp_bcg_dec, cp_bcg_z, sat_ra, sat_dec, sat_z, sat_gr, sat_ri, sat_gi, sat_host_ID ]
    fill = dict( zip( keys, values) )
    out_data = pds.DataFrame( fill )
    out_data.to_csv( out_galx_file )

    return

def entire_sample_func():

    c_dat = fits.getdata('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')

    RA = np.array( c_dat.RA )
    DEC = np.array( c_dat.DEC )
    ID = np.array( c_dat.OBJID )

    rich = np.array( c_dat.LAMBDA )
    Z_photo = np.array( c_dat.Z_LAMBDA )
    Z_photo_err = np.array( c_dat.Z_LAMBDA_ERR )
    Z_spec = np.array( c_dat.Z_SPEC )

    clus_order = np.array( c_dat.ID )

    #. 0.2~z~0.3
    idx_lim = ( Z_photo >= 0.2 ) & ( Z_photo <= 0.3 )
    lim_ra, lim_dec, lim_z = RA[ idx_lim ], DEC[ idx_lim ], Z_photo[ idx_lim ]
    lim_ID = ID[ idx_lim ]

    lim_rich = rich[ idx_lim ]
    lim_zspec = Z_spec[ idx_lim ]
    lim_z_pho_err = Z_photo_err[ idx_lim ]
    lim_order = clus_order[ idx_lim ]


    #. approximation on radius (R200m)
    M_vir, R_vir = rich2R_Simet( lim_z, lim_rich )  ## M_sun, kpc
    M_vir, R_vir = M_vir * h, R_vir * h / 1e3       ## M_sun / h, Mpc / h
    lg_Mvir = np.log10( M_vir )


    ###... save the cluster (BCG) information
    keys = ['ra', 'dec', 'z_pho', 'z_pho_err', 'z_spec', 'objID', 'rich', 'lg_M200m', 'R200m', 'clus_ID']
    values = [ lim_ra, lim_dec, lim_z, lim_z_pho_err, lim_zspec, lim_ID, lim_rich, lg_Mvir, R_vir, lim_order ]
    fill = dict(zip( keys, values) )
    out_data = pds.DataFrame( fill )
    out_data.to_csv( '/home/xkchen/figs/field_sat_redMap/redMap_compare/redMaPPer_z-pho_0.2-0.3_clus-cat.csv',)


    ### === matched member for clusters above
    Nz = len( lim_z )

    m_dat = fits.getdata('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')
    m_ra, m_dec, m_z = np.array( m_dat.RA ), np.array( m_dat.DEC ), np.array( m_dat.Z_SPEC )

    clus_IDs = np.array( m_dat.ID )   ## the ID number of cluster in redMaPPer
    R_cen = np.array( m_dat.R )       ## Mpc / h
    P_mem = np.array( m_dat.P )
    m_objIDs = np.array( m_dat.OBJID )

    #. model mag for pre-color view
    m_mag_r = np.array( m_dat.MODEL_MAG_R )
    m_mag_g = np.array( m_dat.MODEL_MAG_G )
    m_mag_i = np.array( m_dat.MODEL_MAG_I )

    #.
    cop_ra, cop_dec, cop_z = np.array([]), np.array([]), np.array([])
    cop_r_mag, cop_g_mag, cop_i_mag = np.array([]), np.array([]), np.array([])
    cop_Pm, cop_Rcen, cop_objID = np.array([]), np.array([]), np.array([], dtype = int)

    for kk in range( Nz ):

        id_vx = clus_IDs == lim_order[ kk ]

        sub_ra, sub_dec, sub_z = m_ra[ id_vx ], m_dec[ id_vx ], m_z[ id_vx ]
        sub_Pm, sub_Rcen, sub_objIDs = P_mem[ id_vx ], R_cen[ id_vx ], m_objIDs[ id_vx ]
        sub_magr, sub_magg, sub_magi = m_mag_r[ id_vx ], m_mag_g[ id_vx ], m_mag_i[ id_vx ]

        ##.
        cop_ra = np.r_[ cop_ra, sub_ra ]
        cop_dec = np.r_[ cop_dec, sub_dec ]
        cop_z = np.r_[ cop_z, sub_z ]

        cop_r_mag = np.r_[ cop_r_mag, sub_magr ]
        cop_g_mag = np.r_[ cop_g_mag, sub_magg ]
        cop_i_mag = np.r_[ cop_i_mag, sub_magi ]

        cop_Pm = np.r_[ cop_Pm, sub_Pm ]
        cop_Rcen = np.r_[ cop_Rcen, sub_Rcen ]
        cop_objID = np.r_[ cop_objID, sub_objIDs ]

    ##... list member galaxy table
    keys = [ "ra", "dec", "z_spec", "Pm", "Rcen", "ObjID", "mod_mag_r", "mod_mag_g", "mod_mag_i" ]
    out_arr = [ cop_ra, cop_dec, cop_z, cop_Pm, cop_Rcen, cop_objID, cop_r_mag, cop_g_mag, cop_i_mag ]
    tab_file = Table( out_arr, names = keys )
    tab_file.write( '/home/xkchen/figs/field_sat_redMap/redMap_compare/' + 
                    'redMaPPer_z-pho_0.2-0.3_clus_member.fits', overwrite = True)

    return


### === data load
cat_path = '/home/xkchen/data/SDSS/field_galx_redMap/galx_cat/'

##. match to Zhiwei's table~(applied survey masks)
def pre_select_galax():

    zw_dat = fits.open( cat_path + 
        'redmapper_notclean_objid_ugriz_BLENDED_or_notNODEBLEND_ALL_i_err_nocollision_nocenterpost_zhiwei.fits')
    zw_table = zw_dat[1].data

    zw_ra = np.array( zw_table['ra'] )
    zw_dec = np.array( zw_table['dec'] )

    zw_coord = SkyCoord( ra = zw_ra * U.deg, dec = zw_dec * U.deg )
    print( 'N_zw = ', len(zw_ra) )


    ##... pre-match cluster catalog
    data = fits.open( cat_path + 
            'redmapper_notclean_objid_photoobj_ugriz_BLENDED_or_notNODEBLEND_ALL_i_err_xkchen.fit')
    table_lis = data[1].data

    keys = ['RA', 'DEC', 'z', 'zErr', 'objid', 
            'cmodel_u', 'cmodel_g', 'cmodel_r', 'cmodel_i', 'cmodel_z', 
            'cmodelerr_u', 'cmodelerr_g', 'cmodelerr_r', 'cmodelerr_i', 'cmodelerr_z', 
            'model_u', 'model_g', 'model_r', 'model_i', 'model_z', 
            'modelerr_u', 'modelerr_g', 'modelerr_r', 'modelerr_i', 'modelerr_z', 
            'dered_u', 'dered_g', 'dered_r', 'dered_i', 'dered_z',]

    t_ra = np.array( table_lis['RA'] )
    t_dec = np.array( table_lis['DEC'] )

    t_coord = SkyCoord( ra = t_ra * U.deg, dec = t_dec * U.deg )
    print( 'N_t = ', len(t_ra) )


    #. match the two cataklog
    idx, sep, d3d = t_coord.match_to_catalog_sky( zw_coord )
    id_lim = sep.value < 2.7e-4  ## match within 1arcsec


    z_arr = np.array( data[1].data['z'] )

    ##. z_photo limit based on the z_photo error of cluster
    delt_z0 = ( 1 + 0.2 ) * 0.006
    lim_z0 = 0.2 - delt_z0

    delt_z1 = ( 1 + 0.3 ) * 0.006
    lim_z1 = 0.3 + delt_z1

    id_vx = ( z_arr >= lim_z0 ) & ( z_arr <= lim_z1 )


    Ns = len( keys )
    tmp_arr = []

    for kk in range( Ns ):

        sub_arr = np.array( table_lis[ keys[kk] ] )

        id_px = id_vx & id_lim
        tmp_arr.append( sub_arr[ id_px ] )

    tab_file = Table( tmp_arr, names = keys )
    tab_file.write( '/home/xkchen/' + 'sdss_redMaP-limt_galaxy_for_z-clus_0.2to0.3.fits', overwrite = True)

    return

# pre_select_galax()



##... color and luminosity match between galaxies and satellite galaxies
all_galx = fits.open('/home/xkchen/sdss_redMaP-limt_galaxy_for_z-clus_0.2to0.3.fits')
all_table = all_galx[1].data

keys_0 = ['RA', 'DEC', 'z', 'zErr', 'objid', 
        'cmodel_u', 'cmodel_g', 'cmodel_r', 'cmodel_i', 'cmodel_z', 
        'cmodelerr_u', 'cmodelerr_g', 'cmodelerr_r', 'cmodelerr_i', 'cmodelerr_z', 
        'model_u', 'model_g', 'model_r', 'model_i', 'model_z', 
        'modelerr_u', 'modelerr_g', 'modelerr_r', 'modelerr_i', 'modelerr_z', 
        'dered_u', 'dered_g', 'dered_r', 'dered_i', 'dered_z' ]

Ns_0 = len( keys_0 )

all_ra, all_dec, all_z = np.array( all_table['RA'] ), np.array( all_table['DEC'] ), np.array( all_table['z'] )
all_objID = np.array( all_table['objid'] )

all_i_cmag = np.array( all_table['cmodel_i'] )
all_r_cmag = np.array( all_table['cmodel_r'] )

all_extin_i = np.array( all_table['model_i'] ) - np.array( all_table['dered_i'] )
all_extin_r = np.array( all_table['model_r'] ) - np.array( all_table['dered_r'] )

all_coord = SkyCoord( ra = all_ra * U.deg, dec = all_dec * U.deg )


#. members in redMaPPer cluster~( z_phot=0.2~0.3)
# entire_sample_func()  ##. selection on cluster and members

red_dat = fits.open('/home/xkchen/figs/field_sat_redMap/redMap_compare/redMaPPer_z-pho_0.2-0.3_clus_member.fits')
# red_dat = fits.open('/home/xkchen/data/SDSS/field_galx_redMap/redMap_compare/redMaPPer_z-pho_0.2-0.3_clus_member.fits')
red_table = red_dat[1].data

keys_1 = [ 'ra', 'dec', 'z_spec', 'Pm', 'Rcen', 'ObjID', 'mod_mag_r', 'mod_mag_g', 'mod_mag_i' ]

Ns_1 = len( keys_1 )

red_ra, red_dec = np.array( red_table['ra'] ), np.array( red_table['dec'] )
red_objID = np.array( red_table['objID'] )

red_coord = SkyCoord( ra = red_ra * U.deg, dec = red_dec * U.deg )


N_sat = len( red_ra )

tmp_extin_r, tmp_extin_i = np.array( [] ), np.array( [] )
tmp_z_pho = np.array( [] )
tmp_i_cmag, tmp_r_cmag = np.array( [] ), np.array( [] )

for tt in range( N_sat ):

    id_vx = all_objID == red_objID[ tt ]

    tmp_extin_r = np.r_[ tmp_extin_r, all_extin_r[ id_vx ] ]
    tmp_extin_i = np.r_[ tmp_extin_i, all_extin_i[ id_vx ] ]
    tmp_z_pho = np.r_[ tmp_z_pho, all_z[ id_vx ] ]
    tmp_i_cmag = np.r_[ tmp_i_cmag, all_i_cmag[ id_vx ] ]
    tmp_r_cmag = np.r_[ tmp_r_cmag, all_r_cmag[ id_vx ] ]

##. 
cp_keys = ['extin_r', 'extin_i', 'i_cmag', 'r_cmag', 'z_photo']

for tt in range( len(cp_keys) ):
    keys_1.append( cp_keys[ tt ] )

cp_arr = [ tmp_extin_r, tmp_extin_i, tmp_i_cmag, tmp_r_cmag, tmp_z_pho ]

out_arr = []

for tt in range( Ns_1 ):
    out_arr.append( np.array( red_table[ keys_1[ tt ] ] ) )

for tt in range( len(cp_keys) ):
    out_arr.append( cp_arr[ tt ] )

##.
tab_file = Table( out_arr, names = keys_1 )
tab_file.write( '/home/xkchen/sdss_redMaP-limt_galaxy_for_z-clus_0.2to0.3_params-match.fits', overwrite = True)


raise

##.. the total selected image in random catalog~(have applied image selection)


