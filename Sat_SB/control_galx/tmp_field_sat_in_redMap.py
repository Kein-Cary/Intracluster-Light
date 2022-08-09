"""
for given random cluster in redMaPPer, using this file to find the member galaxies and
query or match their properties (i.e. Mstar, color, model and cModel magnitudes, centric distance,
location on image frame and cut region)
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import numpy as np
import pandas as pds
import h5py

import mechanize
import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C

from io import StringIO
from sklearn.neighbors import KDTree
from sklearn import metrics

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

import time


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


### === redMapper member properties mapping of field galaxy ( 0.2-0.3 cluster pre-match catalog )
"""
##. redMapper cluster catalog
pre_dat = fits.open('/home/xkchen/data/SDSS/redmapper/' + 
                'redmapper_dr8_public_v6.3_catalog.fits')

pre_table = pre_dat[1].data

clus_ID = pre_table['ID']
clus_ID = clus_ID.astype( int )

zc, zc_err = pre_table['Z_LAMBDA'], pre_table['Z_LAMBDA_ERR']

##, 0.2~0.3 is cluster z_photo limitation
id_zx = ( zc >= 0.2 ) & ( zc <= 0.3 )

lim_ID = clus_ID[ id_zx ]
lim_zc = zc[ id_zx ]
lim_zc_err = zc_err[ id_zx ]
N_clus = len( lim_ID )


##. redMapper member catalog
cat = fits.open('/home/xkchen/data/SDSS/redmapper/' + 
                'redmapper_dr8_public_v6.3_members.fits')

sat_table = cat[1].data

host_ID = sat_table['ID']  
sat_ra, sat_dec = sat_table['RA'], sat_table['DEC']

sat_objID = sat_table['OBJID']
sat_objID = sat_objID.astype( int )

#.( these mag are deredden-applied )
sat_mag_r = sat_table['MODEL_MAG_R']
sat_mag_g = sat_table['MODEL_MAG_G']
sat_mag_i = sat_table['MODEL_MAG_I']
sat_mag_u = sat_table['MODEL_MAG_U']
sat_mag_z = sat_table['MODEL_MAG_Z']


lim_sat_dex = np.array([ ])

##. focus on satellites in range of (0.2 ~ zc ~ 0.3)
for dd in range( N_clus ):

    id_vx = host_ID == lim_ID[ dd ]

    dd_arr = np.where( id_vx )[0]

    lim_sat_dex = np.r_[ lim_sat_dex, dd_arr ]

lim_sat_dex = lim_sat_dex.astype( int )

lim_sat_ra, lim_sat_dec = sat_ra[ lim_sat_dex ], sat_dec[ lim_sat_dex ]

lim_objID = sat_objID[ lim_sat_dex ]

lim_mag_r = sat_mag_r[ lim_sat_dex ]
lim_mag_g = sat_mag_g[ lim_sat_dex ]
lim_mag_i = sat_mag_i[ lim_sat_dex ]
lim_mag_u = sat_mag_u[ lim_sat_dex ]
lim_mag_z = sat_mag_z[ lim_sat_dex ]

lim_coord = SkyCoord( ra = lim_sat_ra * U.deg, dec = lim_sat_dec * U.deg )


##. member galaxy information catalog~( absolute magnitude)
pat = fits.open('/home/xkchen/data/SDSS/extend_Zphoto_cat/zphot_01_033_cat/' + 
                'redMaPPer_z-phot_0.1-0.33_member_params.fit')

cp_table = pat[1].data
cp_ra, cp_dec = cp_table['ra'], cp_table['dec']
cp_z, cp_zErr = cp_table['z'], cp_table['zErr']

cp_cmag_u = cp_table['cModelMag_u']
cp_cmag_g = cp_table['cModelMag_g']
cp_cmag_r = cp_table['cModelMag_r']
cp_cmag_i = cp_table['cModelMag_i']
cp_cmag_z = cp_table['cModelMag_z']

cp_coord = SkyCoord( ra = cp_ra * U.deg, dec = cp_dec * U.deg )

idx, d2d, d3d = lim_coord.match_to_catalog_sky( cp_coord )
id_lim = d2d.value < 2.7e-4

lim_cmag_u = cp_cmag_u[ idx[id_lim] ]
lim_cmag_g = cp_cmag_g[ idx[id_lim] ]
lim_cmag_r = cp_cmag_r[ idx[id_lim] ]
lim_cmag_i = cp_cmag_i[ idx[id_lim] ]
lim_cmag_z = cp_cmag_z[ idx[id_lim] ]

lim_z = cp_z[ idx[id_lim] ]
lim_zErr = cp_zErr[ idx[id_lim] ]


##. save the member properties
keys = ['ra', 'dec', 'z', 'zErr', 'objid', 
      'cModelMag_u', 'cModelMag_g', 'cModelMag_r', 'cModelMag_i', 'cModelMag_z', 
      'modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z']

values = [ lim_sat_ra, lim_sat_dec, lim_z, lim_zErr, lim_objID, 
        lim_cmag_u, lim_cmag_g, lim_cmag_r, lim_cmag_i, lim_cmag_z, 
        lim_mag_u, lim_mag_g, lim_mag_r, lim_mag_i, lim_mag_z ]

tab_file = Table( values, names = keys )
tab_file.write( '/home/xkchen/data/SDSS/field_galx_redMap/redMap_compare/' + 
                'sdss_redMap_member-mag_of_clus_z0.2to0.3.fits', overwrite = True )

"""

##.
lim_data = fits.open( '/home/xkchen/data/SDSS/field_galx_redMap/redMap_compare/' + 
                    'sdss_redMap_member-mag_of_clus_z0.2to0.3.fits')

lim_table = lim_data[1].data

lim_ra, lim_dec = lim_table['ra'], lim_table['dec']
lim_z = lim_table['z']

lim_cmag_u = lim_table['cModelMag_u']
lim_cmag_g = lim_table['cModelMag_g']
lim_cmag_r = lim_table['cModelMag_r']
lim_cmag_i = lim_table['cModelMag_i']
lim_cmag_z = lim_table['cModelMag_z']

lim_mag_u = lim_table['modelMag_u']
lim_mag_g = lim_table['modelMag_g']
lim_mag_r = lim_table['modelMag_r']
lim_mag_i = lim_table['modelMag_i']
lim_mag_z = lim_table['modelMag_z']

lim_ug = lim_mag_u - lim_mag_g
lim_gr = lim_mag_g - lim_mag_r
lim_ri = lim_mag_r - lim_mag_i
lim_iz = lim_mag_i - lim_mag_z
lim_gi = lim_mag_g - lim_mag_i

##. mags
# lim_arr = np.array( [ lim_cmag_u, lim_cmag_g, lim_cmag_r, lim_cmag_i, lim_cmag_z ] ).T

##. mags + colors
lim_arr = np.array( [ lim_cmag_u, lim_cmag_g, lim_cmag_r, lim_cmag_i, lim_cmag_z, lim_gr, lim_ri, lim_ug ] ).T
print('member read Done!')


### === galaxy information of all galaxy catalog
# keys = ['ra', 'dec', 'z', 'zErr', 'objid', 
#       'cModelMag_u', 'cModelMag_g', 'cModelMag_r', 'cModelMag_i', 'cModelMag_z', 
#       'modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z', 
#       'dered_u', 'dered_g', 'dered_r', 'dered_i', 'dered_z', 
#       'extinction_u', 'extinction_g', 'extinction_r', 'extinction_i', 'extinction_z']

all_cat = fits.open( '/home/xkchen/data/SDSS/field_galx_redMap/galx_cat/' + 
                    'sdss_galaxy_i-cmag_to_21mag.fits' )

all_arr = all_cat[1].data

all_ra, all_dec = np.array( all_arr['RA'] ), np.array( all_arr['DEC'] )
all_z, all_z_err = np.array( all_arr['z'] ), np.array( all_arr['zErr'] )

all_objID = np.array( all_arr['objid'] )
all_objID = all_objID.astype( int )

all_mag_u = np.array( all_arr['modelMag_u'] )
all_mag_g = np.array( all_arr['modelMag_g'] )
all_mag_r = np.array( all_arr['modelMag_r'] )
all_mag_i = np.array( all_arr['modelMag_i'] )
all_mag_z = np.array( all_arr['modelMag_z'] )

all_dered_u = np.array( all_arr['dered_u'] )
all_dered_g = np.array( all_arr['dered_g'] )
all_dered_r = np.array( all_arr['dered_r'] )
all_dered_i = np.array( all_arr['dered_i'] )
all_dered_z = np.array( all_arr['dered_z'] )

all_cmag_u = np.array( all_arr['cModelMag_u'] )
all_cmag_g = np.array( all_arr['cModelMag_g'] )
all_cmag_r = np.array( all_arr['cModelMag_r'] )
all_cmag_i = np.array( all_arr['cModelMag_i'] )
all_cmag_z = np.array( all_arr['cModelMag_z'] )

all_Exint_u = np.array( all_arr['extinction_u'] )
all_Exint_g = np.array( all_arr['extinction_g'] )
all_Exint_r = np.array( all_arr['extinction_r'] )
all_Exint_i = np.array( all_arr['extinction_i'] )
all_Exint_z = np.array( all_arr['extinction_z'] )

all_coord = SkyCoord( ra = all_ra * U.deg, dec = all_dec * U.deg )

all_gr = all_dered_g - all_dered_r
all_ri = all_dered_r - all_dered_i
all_ug = all_dered_u - all_dered_g

##. KDTree mapping
tt0 = time.time()

# cp_arr = np.array( [ all_cmag_u, all_cmag_g, all_cmag_r, all_cmag_i, all_cmag_z ] ).T
cp_arr = np.array( [ all_cmag_u, all_cmag_g, all_cmag_r, all_cmag_i, all_cmag_z, all_gr, all_ri, all_ug ] ).T
cp_Tree = KDTree( cp_arr )

# map_tree, map_idex = cp_Tree.query( lim_arr, k = 6 )
# map_tree, map_idex = cp_Tree.query( lim_arr, k = 17 )
# map_tree, map_idex = cp_Tree.query( lim_arr, k = 21 )
map_tree, map_idex = cp_Tree.query( lim_arr, k = 26 )

tt1 = time.time()
print( tt1 - tt0 )


map_cmag_u = all_cmag_u[ map_idex ].flatten()
map_cmag_g = all_cmag_g[ map_idex ].flatten()
map_cmag_r = all_cmag_r[ map_idex ].flatten()
map_cmag_i = all_cmag_i[ map_idex ].flatten()
map_cmag_z = all_cmag_z[ map_idex ].flatten()

map_dered_u = all_dered_u[ map_idex ].flatten()
map_dered_g = all_dered_g[ map_idex ].flatten()
map_dered_r = all_dered_r[ map_idex ].flatten()
map_dered_i = all_dered_i[ map_idex ].flatten()
map_dered_z = all_dered_z[ map_idex ].flatten()

map_mag_u = all_mag_u[ map_idex ].flatten()
map_mag_g = all_mag_g[ map_idex ].flatten()
map_mag_r = all_mag_r[ map_idex ].flatten()
map_mag_i = all_mag_i[ map_idex ].flatten()
map_mag_z = all_mag_z[ map_idex ].flatten()

map_Exint_u = all_Exint_u[ map_idex ].flatten()
map_Exint_g = all_Exint_g[ map_idex ].flatten()
map_Exint_r = all_Exint_r[ map_idex ].flatten()
map_Exint_i = all_Exint_i[ map_idex ].flatten()
map_Exint_z = all_Exint_z[ map_idex ].flatten()

map_ra, map_dec = all_ra[ map_idex ].flatten(), all_dec[ map_idex ].flatten()
map_z, map_zErr = all_z[ map_idex ].flatten(), all_z_err[ map_idex ].flatten()
map_objID = all_objID[ map_idex ].flatten()

tt2 = time.time()
print( tt2 - tt1 )


##. save selected catalog
keys = ['ra', 'dec', 'z', 'zErr', 'objid', 
      'cModelMag_u', 'cModelMag_g', 'cModelMag_r', 'cModelMag_i', 'cModelMag_z', 
      'modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z', 
      'dered_u', 'dered_g', 'dered_r', 'dered_i', 'dered_z', 
      'extinction_u', 'extinction_g', 'extinction_r', 'extinction_i', 'extinction_z']

values = [ map_ra, map_dec, map_z, map_zErr, map_objID, 
        map_cmag_u, map_cmag_g, map_cmag_r, map_cmag_i, map_cmag_z, 
        map_mag_u, map_mag_g, map_mag_r, map_mag_i, map_mag_z, 
        map_dered_u, map_dered_g, map_dered_r, map_dered_i, map_dered_z, 
        map_Exint_u, map_Exint_g, map_Exint_r, map_Exint_i, map_Exint_z ]

tab_file = Table( values, names = keys )
tab_file.write( '/home/xkchen/data/SDSS/field_galx_redMap/redMap_compare/' + 
                'sdss_redMaP-limt_control-map-galaxy_for_z-clus_0.2to0.3.fits', overwrite = True )

map_ug = map_dered_u - map_dered_g
map_gr = map_dered_g - map_dered_r
map_gi = map_dered_g - map_dered_i
map_ri = map_dered_r - map_dered_i
map_iz = map_dered_i - map_dered_z

print('Finished mapping!')


##. figs
bins_mag_u = np.linspace( np.median( lim_cmag_u ) - 5 * np.std( lim_cmag_u ), np.median( lim_cmag_u ) + 5 * np.std( lim_cmag_u ), 65)
bins_mag_g = np.linspace( np.median( lim_cmag_g ) - 5 * np.std( lim_cmag_g ), np.median( lim_cmag_g ) + 5 * np.std( lim_cmag_g ), 65)
bins_mag_r = np.linspace( np.median( lim_cmag_r ) - 5 * np.std( lim_cmag_r ), np.median( lim_cmag_r ) + 5 * np.std( lim_cmag_r ), 65)
bins_mag_i = np.linspace( np.median( lim_cmag_i ) - 5 * np.std( lim_cmag_i ), np.median( lim_cmag_i ) + 5 * np.std( lim_cmag_i ), 65)
bins_mag_z = np.linspace( np.median( lim_cmag_z ) - 5 * np.std( lim_cmag_z ), np.median( lim_cmag_z ) + 5 * np.std( lim_cmag_z ), 65)

bins_ug = np.linspace( np.median( lim_ug ) - 5 * np.std( lim_ug ), np.median( lim_ug ) + 5 * np.std( lim_ug ), 65)
bins_gr = np.linspace( np.median( lim_gr ) - 5 * np.std( lim_gr ), np.median( lim_gr ) + 5 * np.std( lim_gr ), 65)
bins_gi = np.linspace( np.median( lim_gi ) - 5 * np.std( lim_gi ), np.median( lim_gi ) + 5 * np.std( lim_gi ), 65)
bins_ri = np.linspace( np.median( lim_ri ) - 5 * np.std( lim_ri ), np.median( lim_ri ) + 5 * np.std( lim_ri ), 65)
bins_iz = np.linspace( np.median( lim_iz ) - 5 * np.std( lim_iz ), np.median( lim_iz ) + 5 * np.std( lim_iz ), 65)

bins_z = np.linspace( np.median( lim_z ) - 5 * np.std( lim_z ), np.median( lim_z ) + 5 * np.std( lim_z ), 65 )


plt.figure()
plt.hist( lim_z, bins = bins_z, density = True, color = 'r', alpha = 0.75, label = 'RedMapper Satellites')
plt.hist( map_z, bins = bins_z, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
plt.legend( loc = 2, frameon = False)
plt.xlabel('$z_{photo}$')
plt.savefig('/home/xkchen/contrl_sat_z_compare.png', dpi = 300)
plt.close()


fig = plt.figure( figsize = (20, 4) )
axs = gridspec.GridSpec( 1, 5, figure = fig, width_ratios = [1,1,1,1,1],)

gax = fig.add_subplot( axs[0] )
gax.hist( lim_cmag_u, bins = bins_mag_u, density = True, color = 'b', alpha = 0.75, label = 'RedMapper Satellites')
gax.hist( map_cmag_u, bins = bins_mag_u, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
gax.set_xlabel('cMag_u')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[1] )
gax.hist( lim_cmag_g, bins = bins_mag_g, density = True, color = 'g', alpha = 0.75, label = 'RedMapper Satellites')
gax.hist( map_cmag_g, bins = bins_mag_g, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
gax.set_xlabel('cMag_g')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[2] )
gax.hist( lim_cmag_r, bins = bins_mag_r, density = True, color = 'r', alpha = 0.75, label = 'RedMapper Satellites')
gax.hist( map_cmag_r, bins = bins_mag_r, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
gax.set_xlabel('cMag_r')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[3] )
gax.hist( lim_cmag_i, bins = bins_mag_i, density = True, color = 'm', alpha = 0.75, label = 'RedMapper Satellites')
gax.hist( map_cmag_i, bins = bins_mag_i, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
gax.set_xlabel('cMag_i')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[4] )
gax.hist( lim_cmag_z, bins = bins_mag_z, density = True, color = 'c', alpha = 0.75, label = 'RedMapper Satellites')
gax.hist( map_cmag_z, bins = bins_mag_z, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
gax.set_xlabel('cMag_z')
gax.legend( loc = 2, frameon = False)

plt.savefig('/home/xkchen/SDSS_redMapper_mem_mag.png', dpi = 300)
plt.close()


fig = plt.figure( figsize = (20, 4) )
axs = gridspec.GridSpec( 1, 5, figure = fig, width_ratios = [1,1,1,1,1],)

gax = fig.add_subplot( axs[0] )
gax.hist( lim_ug, bins = bins_ug, density = True, color = 'b', alpha = 0.75, label = 'RedMapper Satellites')
gax.hist( map_ug, bins = bins_ug, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
gax.set_xlabel('u - g')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[1] )
gax.hist( lim_gr, bins = bins_gr, density = True, color = 'g', alpha = 0.75, label = 'RedMapper Satellites')
gax.hist( map_gr, bins = bins_gr, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
gax.set_xlabel('g - r')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[2] )
gax.hist( lim_gi, bins = bins_gi, density = True, color = 'r', alpha = 0.75, label = 'RedMapper Satellites')
gax.hist( map_gi, bins = bins_gi, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
gax.set_xlabel('g - i')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[3] )
gax.hist( lim_ri, bins = bins_ri, density = True, color = 'm', alpha = 0.75, label = 'RedMapper Satellites')
gax.hist( map_ri, bins = bins_ri, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
gax.set_xlabel('r - i')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[4] )
gax.hist( lim_iz, bins = bins_iz, density = True, color = 'c', alpha = 0.75, label = 'RedMapper Satellites')
gax.hist( map_iz, bins = bins_iz, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
gax.set_xlabel('i - z')
gax.legend( loc = 2, frameon = False)

plt.savefig('/home/xkchen/SDSS_redMapper_mem_color.png', dpi = 300)
plt.close()

