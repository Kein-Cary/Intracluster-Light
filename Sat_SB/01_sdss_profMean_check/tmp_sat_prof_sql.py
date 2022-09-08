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
import scipy.interpolate as interp

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable

#.
from img_sat_pros_stack import single_img_SB_func


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
pixel = 0.396
z_ref = 0.25



"""
### === sat. cat. match
s_dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/' + 
						'Extend-BCGM_rgi-common_frame-lim_exlu-BCG_member-cat.csv')

bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
p_ra, p_dec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] )

p_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )
print( len( p_ra ) )


#. sat. cat. with objID
dat = fits.open( '/home/xkchen/figs/extend_Zphoto_cat/zphot_01_033_cat/redMaPPer_z-phot_0.1-0.33_member_params.fit')

ref_ra, ref_dec, ref_IDs = dat[1].data['ra'], dat[1].data['dec'], dat[1].data['objID']
ref_coord = SkyCoord( ra = ref_ra * U.deg, dec = ref_dec * U.deg )


idx, sep, d3d = p_coord.match_to_catalog_sky( ref_coord )
id_lim = sep.value < 2.7e-4

mp_ra, mp_dec, mp_IDs = ref_ra[ idx[ id_lim ] ], ref_dec[ idx[ id_lim ] ], ref_IDs[ idx[ id_lim ] ]

cp_ra, cp_dec = p_ra[ id_lim ], p_dec[ id_lim ]

print( len( mp_ra ) )


keys = [ 'ra', 'dec', 'obj_IDs' ]
values = [ cp_ra, cp_dec, mp_IDs ]
fill = dict( zip( keys, values) )
out_data = pds.DataFrame( fill )

# out_data.to_csv( '/home/xkchen/sat_SDSS-profMean_sql_cat.csv')
out_data.to_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/prof_Mean_sql/' + 'sat_SDSS-profMean_sql_cat.csv' )

raise
"""


### === frame-limited and excluded BCGs catalog match
"""
dat = pds.read_csv('/home/xkchen/data/SDSS/member_files/prof_Mean_sql/' + 'sat_SDSS-profMean_sql_cat.csv')
ra, dec, IDs = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['obj_IDs'] )
ref_coord = SkyCoord( ra = ra * U.deg, dec = dec * U.deg )

Ns = len( ra )
m, n = divmod( Ns, cpus)
N_sub0, N_sub1 = m * rank, (rank + 1) * m
if rank == cpus - 1:
	N_sub1 += n

sub_ra, sub_dec = ra[N_sub0 : N_sub1], dec[N_sub0 : N_sub1]
sub_IDs = IDs[N_sub0 : N_sub1]


##.. Table from CASJob
cat = fits.open('/home/xkchen/data/SDSS/extend_Zphoto_cat/sat_profMean/redMaP_Sat_profMean_sql.fit')
data = cat[1].data

c_ra, c_dec = np.array(data['ra'] ), np.array(data['dec'] )

prof_ID = np.array( data['objID'] )
prof_bin = np.array( data['bin'] )
prof_Mean = np.array( data['profMean'] )
prof_band = np.array( data['band'] )
prof_Err = np.array( data['profErr'] )


#... map-to-SDSS table
N_sat = len( sub_ra )

for pp in range( N_sat ):

	ra_g, dec_g = sub_ra[ pp ], sub_dec[ pp ]
	pp_ID = sub_IDs[ pp ]

	#. match by objIDs
	id_lim = prof_ID == pp_ID

	mp_ra, mp_dec = c_ra[ id_lim ], c_dec[ id_lim ]

	mp_IDs = prof_ID[ id_lim ]
	mp_prof_bin = prof_bin[ id_lim ]
	mp_prof_Mean = prof_Mean[ id_lim ]
	mp_prof_band = prof_band[ id_lim ]
	mp_prof_Err = prof_Err[ id_lim ]

	#.
	_pp_Ns = len( mp_ra )

	out_doc = open( '/home/xkchen/data/SDSS/member_files/sat_profMean/' + 
					'sat_ra%.5f_dec%.5f_SDSS_prof.txt' % (ra_g, dec_g), 'w')

	# keys = [ 'objID', 'bin', 'band', 'profMean', 'profErr' ]
	print( 'objID,bin,band,profMean,profErr', file = out_doc )

	for tt in range( _pp_Ns ):

		out_str = '%d,%d,%d,%.8f,%.8f' % ( mp_IDs[tt], mp_prof_bin[tt], mp_prof_band[tt], 
											mp_prof_Mean[tt], mp_prof_Err[tt])

		print( out_str, file = out_doc )

	out_doc.close()

print('%d-rank, done!' % rank )

raise
"""


"""
##... prosfile check
dat = pds.read_csv( '/home/xkchen/data/SDSS/member_files/sat_cat_z02_03/' + 
					'Extend-BCGM_rgi-common_frame-lim_exlu-BCG_member-cat.csv')

prof_file = '/home/xkchen/data/SDSS/member_files/sat_profMean/sat_ra%.5f_dec%.5f_SDSS_prof.txt'

bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
sat_ra, sat_dec = np.array( dat['ra'] ), np.array( dat['dec'] )


n_skip = 0
r_bins = np.logspace( 0, 2.48, 27 )  ##. kpc

N_ss = len( sat_ra )

doc = open('/home/xkchen/prof_check_sat-cat.txt', 'w')
doc_1 = open('/home/xkchen/prof_no-map_sat-cat.txt', 'w')

for ii in range( N_ss ):

	ra_g, dec_g, z_g = sat_ra[ii], sat_dec[ii], bcg_z[ii]

	Da_g = Test_model.angular_diameter_distance( z_g ).value # unit Mpc
	Dl_g = Test_model.luminosity_distance( z_g ).value # unit Mpc

	band_str = 'r'

	try:
		tt_rbins, tt_fdens = single_img_SB_func( band_str, z_g, ra_g, dec_g, prof_file, r_bins, n_skip = n_skip)

	except ValueError:

		out_str = '%.10f, %.10f' % (ra_g, dec_g)
		print( out_str, file = doc )

	except FileNotFoundError:

		out_str = '%.10f, %.10f' % (ra_g, dec_g)
		print( out_str, file = doc_1 )

doc.close()
doc_1.close()

"""


##.. remeasure Match profMean
dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/prof_Mean_sql/sat_SDSS-profMean_sql_cat.csv')
ra, dec, IDs = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['obj_IDs'] )
ref_coord = SkyCoord( ra = ra * U.deg, dec = dec * U.deg )


pat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/prof_Mean_sql/prof_check_sat-cat.txt') ## doulble record satellites
p_ra, p_dec = np.array( pat['ra']), np.array( pat['dec'])

p_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

idx, sep, d3d = p_coord.match_to_catalog_sky( ref_coord )
id_lim = sep.value < 2.7e-4

sub_ra, sub_dec, sub_IDs = ra[ idx[ id_lim ] ], dec[ idx[ id_lim ] ], IDs[ idx[ id_lim ] ]


##.. Table from CASJob
cat = fits.open('/home/xkchen/figs/extend_Zphoto_cat/sat_profMean/redMaP_Sat_profMean_sql.fit')
data = cat[1].data

c_ra, c_dec = np.array(data['ra'] ), np.array(data['dec'] )

prof_ID = np.array( data['objID'] )
prof_bin = np.array( data['bin'] )
prof_Mean = np.array( data['profMean'] )
prof_band = np.array( data['band'] )
prof_Err = np.array( data['profErr'] )


#... map-to-SDSS table
N_sat = len( sub_ra )

for pp in range( N_sat ):

	ra_g, dec_g = sub_ra[ pp ], sub_dec[ pp ]
	pp_ID = sub_IDs[ pp ]

	#. match by objIDs
	id_lim = prof_ID == pp_ID

	mp_ra, mp_dec = c_ra[ id_lim ], c_dec[ id_lim ]

	mp_IDs = prof_ID[ id_lim ]
	mp_prof_bin = prof_bin[ id_lim ]
	mp_prof_Mean = prof_Mean[ id_lim ]
	mp_prof_band = prof_band[ id_lim ]
	mp_prof_Err = prof_Err[ id_lim ]


	#.
	_pp_Ns = np.int( len( mp_ra ) / 2 )

	out_doc = open( '/home/xkchen/Downloads/prof_Mean/' + 
					'sat_ra%.5f_dec%.5f_SDSS_prof.txt' % (ra_g, dec_g), 'w')

	# keys = [ 'objID', 'bin', 'band', 'profMean', 'profErr' ]
	print( 'objID,bin,band,profMean,profErr', file = out_doc )

	for tt in range( _pp_Ns ):

		out_str = '%d,%d,%d,%.8f,%.8f' % ( mp_IDs[tt], mp_prof_bin[tt], mp_prof_band[tt], 
											mp_prof_Mean[tt], mp_prof_Err[tt])

		print( out_str, file = out_doc )

	out_doc.close()

print('%d-rank, done!' % rank )

