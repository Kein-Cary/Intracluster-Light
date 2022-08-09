import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc
from astropy.table import Table

from scipy import spatial
from sklearn.neighbors import KDTree

import scipy.signal as signal
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord


##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
a_ref = 1 / (z_ref + 1)

band = ['u', 'g', 'r', 'i', 'z']


### === data load
def redMap_limit_cat():

	### === magnitude catalog
	import glob

	# file_s = glob.glob('/home/xkchen/Downloads/*.fit')
	file_s = glob.glob('/home/xkchen/figs/*.fit')

	N_s = len( file_s )

	keys = ['ra', 'dec', 'z', 'zErr', 'objid', 
			'cModelMag_u', 'cModelMag_g', 'cModelMag_r', 'cModelMag_i', 'cModelMag_z', 
			'modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z', 
			'dered_u', 'dered_g', 'dered_r', 'dered_i', 'dered_z', 
			'extinction_u', 'extinction_g', 'extinction_r', 'extinction_i', 'extinction_z']

	N_ks = len( keys )

	tmp_arr = []

	data = fits.open( file_s[ 0 ] )
	table = data[1].data

	for dd in range( N_ks ):

		tmp_arr.append( table[ keys[ dd ] ] )

	for ll in range( 1, N_s ):

		data_s = fits.open( file_s[ ll ] )
		table_s = data_s[1].data

		for dd in range( N_ks ):

			tmp_arr[ dd ] = np.r_[ tmp_arr[ dd ], table_s[ keys[ dd ] ] ]

	##. save the catalog
	tab_file = Table( tmp_arr, names = keys )
	tab_file.write( '/home/xkchen/data/SDSS/field_galx_redMap/galx_cat/' + 
					'sdss_galaxy_i-cmag_to_21mag.fits', overwrite = True )

def redMap_cat_mag_check():

	##. redMapper cluster catalog
	pre_dat = fits.open('/home/xkchen/mywork/ICL/data/redmapper/' + 
	                'redmapper_dr8_public_v6.3_catalog.fits')
	pre_table = pre_dat[1].data

	clus_ID = pre_table['ID']
	clus_ID = clus_ID.astype( int )

	zc, zc_err = pre_table['Z_LAMBDA'], pre_table['Z_LAMBDA_ERR']

	##. 0.2~0.3 is cluster z_photo limitation
	id_zx = ( zc >= 0.2 ) & ( zc <= 0.3 )

	lim_ID = clus_ID[ id_zx ]
	lim_zc = zc[ id_zx ]
	lim_zc_err = zc_err[ id_zx ]
	N_clus = len( lim_ID )


	##. redMapper member catalog
	cat = fits.open('/home/xkchen/mywork/ICL/data/redmapper/' + 
	                'redmapper_dr8_public_v6.3_members.fits')
	sat_table = cat[1].data

	host_ID = sat_table['ID']  
	sat_ra, sat_dec = sat_table['RA'], sat_table['DEC']

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

	lim_mag_r = sat_mag_r[ lim_sat_dex ]
	lim_mag_g = sat_mag_g[ lim_sat_dex ]
	lim_mag_i = sat_mag_i[ lim_sat_dex ]
	lim_mag_u = sat_mag_u[ lim_sat_dex ]
	lim_mag_z = sat_mag_z[ lim_sat_dex ]

	lim_coord = SkyCoord( ra = lim_sat_ra * U.deg, dec = lim_sat_dec * U.deg )


	##.
	ref_dat = fits.open('/home/xkchen/figs/extend_Zphoto_cat/zphot_01_033_cat/' + 
				'redMaPPer_z-phot_0.1-0.33_member_params.fit')
	ref_table = ref_dat[1].data

	ref_ra, ref_dec = ref_table['ra'], ref_table['dec']

	ref_mag_u = ref_table['modelMag_u']
	ref_mag_g = ref_table['modelMag_g']
	ref_mag_r = ref_table['modelMag_r']
	ref_mag_i = ref_table['modelMag_i']
	ref_mag_z = ref_table['modelMag_z']

	ref_dered_u = ref_table['dered_u']
	ref_dered_g = ref_table['dered_g']
	ref_dered_r = ref_table['dered_r']
	ref_dered_i = ref_table['dered_i']
	ref_dered_z = ref_table['dered_z']

	ref_coord = SkyCoord( ra = ref_ra * U.deg, dec = ref_dec * U.deg )

	idx, d2d, d3d = lim_coord.match_to_catalog_sky( ref_coord )
	id_lim = d2d.value < 2.7e-4

	mp_ra, mp_dec = ref_ra[ idx[ id_lim ] ], ref_dec[ idx[ id_lim ] ]

	# map_mag_u = ref_mag_u[ idx[ id_lim ] ]
	# map_mag_g = ref_mag_g[ idx[ id_lim ] ]
	# map_mag_r = ref_mag_r[ idx[ id_lim ] ]
	# map_mag_i = ref_mag_i[ idx[ id_lim ] ]
	# map_mag_z = ref_mag_z[ idx[ id_lim ] ]

	map_mag_u = ref_dered_u[ idx[ id_lim ] ]
	map_mag_g = ref_dered_g[ idx[ id_lim ] ]
	map_mag_r = ref_dered_r[ idx[ id_lim ] ]
	map_mag_i = ref_dered_i[ idx[ id_lim ] ]
	map_mag_z = ref_dered_z[ idx[ id_lim ] ]


	fig = plt.figure( figsize = (20, 4) )
	axs = gridspec.GridSpec( 1, 5, figure = fig, width_ratios = [1,1,1,1,1],)

	gax = fig.add_subplot( axs[0] )
	gax.plot( lim_mag_u, map_mag_u, 'r.', )
	gax.plot( lim_mag_u, lim_mag_u, 'k-', )
	gax.set_xlabel('m_u in redMapper catalog')
	gax.set_ylabel('m_u in My query')

	gax = fig.add_subplot( axs[1] )
	gax.plot( lim_mag_g, map_mag_g, 'r.', )
	gax.plot( lim_mag_g, lim_mag_g, 'k-', )
	gax.set_xlabel('m_g in redMapper catalog')
	gax.set_ylabel('m_g in My query')

	gax = fig.add_subplot( axs[2] )
	gax.plot( lim_mag_r, map_mag_r, 'r.', )
	gax.plot( lim_mag_r, lim_mag_r, 'k-', )
	gax.set_xlabel('m_r in redMapper catalog')
	gax.set_ylabel('m_r in My query')

	gax = fig.add_subplot( axs[3] )
	gax.plot( lim_mag_i, map_mag_i, 'r.', )
	gax.plot( lim_mag_i, lim_mag_i, 'k-', )
	gax.set_xlabel('m_i in redMapper catalog')
	gax.set_ylabel('m_i in My query')

	gax = fig.add_subplot( axs[4] )
	gax.plot( lim_mag_z, map_mag_z, 'r.', )
	gax.plot( lim_mag_z, lim_mag_z, 'k-', )
	gax.set_xlabel('m_z in redMapper catalog')
	gax.set_ylabel('m_z in My query')

	plt.savefig('/home/xkchen/redMapper_member_mag_check.png', dpi = 300)
	plt.close()

	return


# redMap_limit_cat()
# redMap_cat_mag_check()


### === 0.2-0.3 cluster pre-match catalog

##. redMapper cluster catalog
# pre_dat = fits.open('/home/xkchen/mywork/ICL/data/redmapper/' + 
#                 'redmapper_dr8_public_v6.3_catalog.fits')

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
# cat = fits.open('/home/xkchen/mywork/ICL/data/redmapper/' + 
#                 'redmapper_dr8_public_v6.3_members.fits')

cat = fits.open('/home/xkchen/data/SDSS/redmapper/' + 
                'redmapper_dr8_public_v6.3_members.fits')

sat_table = cat[1].data

host_ID = sat_table['ID']  
sat_ra, sat_dec = sat_table['RA'], sat_table['DEC']

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

lim_mag_r = sat_mag_r[ lim_sat_dex ]
lim_mag_g = sat_mag_g[ lim_sat_dex ]
lim_mag_i = sat_mag_i[ lim_sat_dex ]
lim_mag_u = sat_mag_u[ lim_sat_dex ]
lim_mag_z = sat_mag_z[ lim_sat_dex ]

lim_ug = lim_mag_u - lim_mag_g
lim_gr = lim_mag_g - lim_mag_r
lim_ri = lim_mag_r - lim_mag_i
lim_iz = lim_mag_i - lim_mag_z
lim_gi = lim_mag_g - lim_mag_i

lim_coord = SkyCoord( ra = lim_sat_ra * U.deg, dec = lim_sat_dec * U.deg )


##. member galaxy information catalog~( absolute magnitude)
# pat = fits.open('/home/xkchen/figs/extend_Zphoto_cat/zphot_01_033_cat/' + 
#                 'redMaPPer_z-phot_0.1-0.33_member_params.fit')

pat = fits.open('/home/xkchen/data/SDSS/extend_Zphoto_cat/zphot_01_033_cat/' + 
                'redMaPPer_z-phot_0.1-0.33_member_params.fit')

cp_table = pat[1].data
cp_ra, cp_dec = cp_table['ra'], cp_table['dec']

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

print( lim_cmag_r.shape )


##. galaxy information of all galaxy catalog
# keys = ['ra', 'dec', 'z', 'zErr', 'objid', 
# 		'cModelMag_u', 'cModelMag_g', 'cModelMag_r', 'cModelMag_i', 'cModelMag_z', 
# 		'modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z', 
# 		'dered_u', 'dered_g', 'dered_r', 'dered_i', 'dered_z', 
# 		'extinction_u', 'extinction_g', 'extinction_r', 'extinction_i', 'extinction_z']

all_cat = fits.open( '/home/xkchen/data/SDSS/field_galx_redMap/galx_cat/' + 
					'sdss_galaxy_i-cmag_to_21mag.fits' )

all_arr = all_cat[1].data

all_ra, all_dec = np.array( all_arr['RA'] ), np.array( all_arr['DEC'] )
all_z, all_z_err = np.array( all_arr['z'] ), np.array( all_arr['zErr'] )

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

idx, d2d, d3d = lim_coord.match_to_catalog_sky( all_coord )
id_lim = d2d.value < 2.7e-4 

mp_z = all_z[ idx[id_lim] ]
mp_ra, mp_dec = all_ra[ idx[id_lim] ], all_dec[ idx[id_lim] ]

mp_cmag_u = all_cmag_u[ idx[id_lim] ]
mp_cmag_g = all_cmag_g[ idx[id_lim] ]
mp_cmag_r = all_cmag_r[ idx[id_lim] ]
mp_cmag_i = all_cmag_i[ idx[id_lim] ]
mp_cmag_z = all_cmag_z[ idx[id_lim] ]

mp_Exint_u = all_Exint_u[ idx[id_lim] ]
mp_Exint_g = all_Exint_g[ idx[id_lim] ]
mp_Exint_r = all_Exint_r[ idx[id_lim] ]
mp_Exint_i = all_Exint_i[ idx[id_lim] ]
mp_Exint_z = all_Exint_z[ idx[id_lim] ]

print( mp_ra.shape )


### === 
plt.figure()
plt.plot( all_Exint_r, all_mag_r - all_dered_r, 'r.')
plt.plot( all_Exint_r, all_Exint_r, 'k-')
plt.savefig('/home/xkchen/Extinction_r_check.png', dpi = 300)
plt.close()

plt.figure()
plt.plot( all_Exint_g, all_mag_g - all_dered_g, 'r.')
plt.plot( all_Exint_g, all_Exint_g, 'k-')
plt.savefig('/home/xkchen/Extinction_g_check.png', dpi = 300)
plt.close()

plt.figure()
plt.plot( all_Exint_i, all_mag_i - all_dered_i, 'r.')
plt.plot( all_Exint_i, all_Exint_i, 'k-')
plt.savefig('/home/xkchen/Extinction_i_check.png', dpi = 300)
plt.close()

plt.figure()
plt.plot( all_Exint_u, all_mag_u - all_dered_u, 'r.')
plt.plot( all_Exint_u, all_Exint_u, 'k-')
plt.savefig('/home/xkchen/Extinction_u_check.png', dpi = 300)
plt.close()

plt.figure()
plt.plot( all_Exint_z, all_mag_z - all_dered_z, 'r.')
plt.plot( all_Exint_z, all_Exint_z, 'k-')
plt.savefig('/home/xkchen/Extinction_z_check.png', dpi = 300)
plt.close()


fig = plt.figure()
plt.plot( lim_cmag_r, mp_cmag_r, 'r.',)
plt.plot( lim_cmag_r, lim_cmag_r, 'k-',)
plt.savefig('/home/xkchen/r_cmag_compare.png', dpi = 300)
plt.close()

fig = plt.figure()
plt.plot( lim_cmag_g, mp_cmag_g, 'r.',)
plt.plot( lim_cmag_g, lim_cmag_g, 'k-',)
plt.savefig('/home/xkchen/g_cmag_compare.png', dpi = 300)
plt.close()

fig = plt.figure()
plt.plot( lim_cmag_i, mp_cmag_i, 'r.',)
plt.plot( lim_cmag_i, lim_cmag_i, 'k-',)
plt.savefig('/home/xkchen/i_cmag_compare.png', dpi = 300)
plt.close()

fig = plt.figure()
plt.plot( lim_cmag_u, mp_cmag_u, 'r.',)
plt.plot( lim_cmag_u, lim_cmag_u, 'k-',)
plt.savefig('/home/xkchen/u_cmag_compare.png', dpi = 300)
plt.close()

fig = plt.figure()
plt.plot( lim_cmag_z, mp_cmag_z, 'r.',)
plt.plot( lim_cmag_z, lim_cmag_z, 'k-',)
plt.savefig('/home/xkchen/z_cmag_compare.png', dpi = 300)
plt.close()


fig = plt.figure()
plt.plot( lim_cmag_r, mp_cmag_r - mp_Exint_r, 'r.',)
plt.plot( lim_cmag_r, lim_cmag_r, 'k-',)
plt.savefig('/home/xkchen/r_dered_cmag_compare.png', dpi = 300)
plt.close()

fig = plt.figure()
plt.plot( lim_cmag_g, mp_cmag_g - mp_Exint_g, 'r.',)
plt.plot( lim_cmag_g, lim_cmag_g, 'k-',)
plt.savefig('/home/xkchen/g_dered_cmag_compare.png', dpi = 300)
plt.close()

fig = plt.figure()
plt.plot( lim_cmag_i, mp_cmag_i - mp_Exint_i, 'r.',)
plt.plot( lim_cmag_i, lim_cmag_i, 'k-',)
plt.savefig('/home/xkchen/i_dered_cmag_compare.png', dpi = 300)
plt.close()

fig = plt.figure()
plt.plot( lim_cmag_u, mp_cmag_u - mp_Exint_u, 'r.',)
plt.plot( lim_cmag_u, lim_cmag_u, 'k-',)
plt.savefig('/home/xkchen/u_dered_cmag_compare.png', dpi = 300)
plt.close()

fig = plt.figure()
plt.plot( lim_cmag_z, mp_cmag_z - mp_Exint_z, 'r.',)
plt.plot( lim_cmag_z, lim_cmag_z, 'k-',)
plt.savefig('/home/xkchen/z_dered_cmag_compare.png', dpi = 300)
plt.close()

