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


### === redMagic catalog
##. bright
b_data = fits.open('/home/xkchen/figs/field_sat_redMap/galx_cat/' + 
			'redmagic_dr8_public_v6.3_bright.fits.gz')
b_table = b_data[1].data

keys = ['OBJID', 'RA', 'DEC', 'IMAG', 'IMAG_ERR', 'Z_SPEC', 'ILUM', 'ZREDMAGIC', 'ZREDMAGIC_ERR', 
		'MODEL_MAG_U', 'MODEL_MAG_G', 'MODEL_MAG_R', 'MODEL_MAG_I', 'MODEL_MAG_Z',
		'MODEL_MAGERR_U', 'MODEL_MAGERR_G', 'MODEL_MAGERR_R', 'MODEL_MAGERR_I', 'MODEL_MAGERR_Z',
		'MABS_U', 'MABS_G', 'MABS_R', 'MABS_I', 'MABS_Z',
		'MABS_ERR_U', 'MABS_ERR_G', 'MABS_ERR_R', 'MABS_ERR_I', 'MABS_ERR_Z' ]

ref_ra_0 = np.array( b_table['RA'] )
ref_dec_0 = np.array( b_table['DEC'] )
ref_objID_0 = np.array( b_table['OBJID'] )
ref_z_0 = np.array( b_table['ZREDMAGIC'] )

ref_coord_0 = SkyCoord( ra = ref_ra_0 * U.deg, dec = ref_dec_0 * U.deg )


#. save for z_photo in DR8 query
p_keys = ['id', 'ra', 'dec']
values = [ ref_objID_0, ref_ra_0, ref_dec_0 ]
fill = dict(zip( p_keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/redmagic_bright_query_cat.csv', index = None )


N_s = len( keys )

tmp_arr = []

for tt in range( N_s ):

	tmp_arr.append( b_table[ keys[ tt ] ] )


##. faint
f_data = fits.open('/home/xkchen/figs/field_sat_redMap/galx_cat/' + 
			'redmagic_dr8_public_v6.3_faint.fits.gz')
f_table = f_data[1].data

ref_ra_1 = np.array( f_table['RA'] )
ref_dec_1 = np.array( f_table['DEC'] )
ref_objID_1 = np.array( f_table['OBJID'] )
ref_z_1 = np.array( f_table['ZREDMAGIC'] )

ref_coord_1 = SkyCoord( ra = ref_ra_1 * U.deg, dec = ref_dec_1 * U.deg )
N_s1 = np.int( len( ref_z_1 ) / 2 )


#. save for z_photo in DR8 query
p_keys = [ 'objid', 'ra', 'dec' ]
values = [ ref_objID_1[:N_s1], ref_ra_1[:N_s1], ref_dec_1[:N_s1] ]
fill = dict(zip( p_keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/redmagic_faint_query_cat_p0.csv', index = None)

p_keys = [ 'objid', 'ra', 'dec' ]
values = [ ref_objID_1[N_s1:], ref_ra_1[N_s1:], ref_dec_1[N_s1:] ]
fill = dict(zip( p_keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/redmagic_faint_query_cat_p1.csv', index = None)


#. entire sample
for tt in range( N_s ):
	tmp_arr[ tt ] = np.r_[ tmp_arr[ tt ], f_table[ keys[ tt ] ] ]


##. entire redMagic catalog
fill = dict(zip( keys, tmp_arr) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/redmagic_faint_bright_galaxy_cat.csv')


##. SDSS redMapper member galaxy
red_dat = fits.open('/home/xkchen/figs/field_sat_redMap/redMap_compare/redMaPPer_z-pho_0.2-0.3_clus_member.fits')
red_table = red_dat[1].data

keys_1 = [ 'ra', 'dec', 'z_spec', 'Pm', 'Rcen', 'ObjID', 'mod_mag_r', 'mod_mag_g', 'mod_mag_i' ]

Ns_1 = len( keys_1 )

red_ra, red_dec = np.array( red_table['ra'] ), np.array( red_table['dec'] )
red_objID = np.array( red_table['objID'] )

red_coord = SkyCoord( ra = red_ra * U.deg, dec = red_dec * U.deg )


idx, sep, d3d = red_coord.match_to_catalog_sky( ref_coord_0 )
id_lim = sep.value < 2.7e-4  ## match within 1arcsec

mp_ra_0 = ref_ra_0[ idx[ id_lim ] ]
mp_dec_0 = ref_dec_0[ idx[ id_lim ] ]

mp_objID_0 = ref_objID_0[ idx[ id_lim ] ]

cp_objID_0 = red_objID[ id_lim ]


idx, sep, d3d = red_coord.match_to_catalog_sky( ref_coord_1 )
id_lim = sep.value < 2.7e-4  ## match within 1arcsec

mp_ra_1 = ref_ra_1[ idx[ id_lim ] ]
mp_dec_1 = ref_dec_1[ idx[ id_lim ] ]

mp_objID_1 = ref_objID_1[ idx[ id_lim ] ]
cp_objID_1 = red_objID[ id_lim ]


print( np.sum( cp_objID_0 - mp_objID_0 ) )
print( np.sum( cp_objID_1 - mp_objID_1 ) )

print( 'in brighter', cp_objID_0.shape )
print( 'in faint', cp_objID_1.shape )


##.


