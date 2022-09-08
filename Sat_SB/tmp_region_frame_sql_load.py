"""
This file use to query image frames of SDSS for given target (ra, dec) pairs.
these pairs are BCG locations indetified by SDSS redMaPPer
"""
import h5py
import random
import numpy as np
import pandas as pds
import scipy.stats as sts

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.wcs as awc

from scipy import interpolate as interp
from astropy import cosmology as apcy
from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.coordinates import Angle

from io import StringIO
import subprocess as subpro
import wget
import glob
import time
import os


#.
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpathes


###. cosmology
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)



### === func.s
def region_sql_func( bcg_ra, bcg_dec, bcg_z, R_sq, band_str, record_file, out_path ):
	"""
	bcg_ra, bcg_dec, bcg_z : catalog information
	R_sq : circle radius for image query, in units of arcsec ~ (details can be found in astroquery.SDSS )
	band_str : filter information ~ ('r', 'g', 'i')
	[ https://astroquery.readthedocs.io/en/latest/api/astroquery.sdss.SDSSClass.html#astroquery.sdss.SDSSClass.query_region ]
	---------------------------------------------------------------

	record_file : files for records during download~( like some file may loss and cannot download)
	out_path : where to save imag frames
	"""

	import sys
	# sys.path.append('/home/xkchen/tool/Conda/Tools/normitems')
	sys.path.append('/home/xkchen/Tools/normitems')

	import changds

	N_g = len( bcg_ra )

	for nn in range( N_g ):

		##.
		R_A = R_sq[ nn ] * U.arcsec

		##.
		pos = coords.SkyCoord( '%fd %fd' % ( bcg_ra[ nn ], bcg_dec[ nn ] ), frame = 'icrs')
		xid = SDSS.query_region( pos, spectro = False, radius = R_A, timeout = None,)

		run_ID = xid['run']
		rerun_ID = xid['rerun']
		camcol_ID = xid['camcol']
		field_ID = xid['field']

		##. if there is no Extended_path, build it, or just write
		extend_path = out_path + 'ra%.3fdec%.3fz%.3f/' % (bcg_ra[ nn ], bcg_dec[ nn ], bcg_z[ nn ])

		id_path = os.path.exists( extend_path )

		if id_path is False:

			cmd = 'mkdir %s' % extend_path
			APRO = subpro.Popen( cmd, shell = True )
			APRO.wait()

		else:

			pass

		##. map sorround images
		run_set = np.array( list( set( run_ID ) ) )
		rerun_set = np.array( list( set( rerun_ID ) ) )
		camcol_set = np.array( list( set( camcol_ID ) ) )
		field_set = np.array( list( set( field_ID ) ) )

		N0 = len( run_set )
		N1 = len( rerun_set )
		N2 = len( camcol_set )
		N3 = len( field_set )

		for kk in range( N0 ):

			for pp in range( N1 ):

				for dd in range( N2 ):

					for mm in range( N3 ):

						try:
							##. download surrounding fields
							url_lod = ( 'https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/%d/%d/%d/frame-%s-%s-%d-%s.fits.bz2' % 
										( rerun_set[ pp ], run_set[ kk ], camcol_set[ dd ], band_str, str(run_set[ kk ]).zfill( 6 ), camcol_set[ dd ], str(field_set[ mm ]).zfill(4) ),)[0]

							out_lis = ( extend_path + 'frame-%s-%s-%d-%s.fits.bz2' % ( band_str, str(run_set[ kk ]).zfill( 6 ), camcol_set[ dd ], str(field_set[ mm ]).zfill(4) ),)[0]

							wget.download( url_lod, out_lis )

						except:
							##. save the "exception" case
							doc = open( record_file, 'w')
							s = '\n rerun = %d, run = %d, camcol=%d, field=%d \n' % ( rerun_set[ pp ], run_set[ kk ], camcol_set[ dd ], field_set[ mm ] )
							print(s, file = doc)
							doc.close()

							continue

	return


### === ### catalog of cluster 0.2~z~0.3
"""
##. specifically, the same cluster calatog as used for ICL measurement
cat_path = '/home/xkchen/mywork/ICL/data/photo_cat/'
out_path = '/home/xkchen/figs/extend_Zphoto_cat/region_sql_cat/'

band = ['r', 'g', 'i']


##. SDSS redMaPPer catalog
c_dat = fits.getdata('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')

RA = np.array( c_dat.RA )
DEC = np.array( c_dat.DEC )
ID = np.array( c_dat.OBJID )

rich = np.array( c_dat.LAMBDA )
Z_photo = np.array( c_dat.Z_LAMBDA )

ord_dex = np.array( c_dat.ID )

ref_coord = SkyCoord( ra = RA * U.deg, dec = DEC * U.deg )


##. RedMaPPer member catalog
m_dat = fits.getdata('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')

clus_IDs = np.array( m_dat.ID )   ## the ID number of cluster in redMaPPer
R_cen = np.array( m_dat.R )       ## Mpc / h
P_mem = np.array( m_dat.P )
m_objIDs = np.array( m_dat.OBJID )


for kk in range( 3 ):

	#.
	#. keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
	dat = pds.read_csv( cat_path + 
				'photo-z_%s-band_tot_remain_cat_set_200-grid_6.0-sigma.csv' % band[ kk ],)

	kk_ra, kk_dec, kk_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

	kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

	idx, sep, d3d = kk_coord.match_to_catalog_sky( ref_coord )
	id_lim = sep.value < 2.7e-4

	##.
	mp_ra, mp_dec, mp_z = RA[ idx[ id_lim ] ], DEC[ idx[ id_lim ] ], Z_photo[ idx[ id_lim ] ]
	mp_IDs = ord_dex[ idx[ id_lim ] ]
	mp_rich = rich[ idx[ id_lim ] ]

	N_z = len( mp_ra )

	##. member map
	#. record the maximum of satellite centric distance
	Rs_max = np.zeros( N_z,)
	Rs08_max = np.zeros( N_z,)  ##. record of Pm_cut = 0.8

	for dd in range( N_z ):

		id_vx = clus_IDs == mp_IDs[ dd ]
		sub_Pm, sub_Rcen = P_mem[ id_vx ], R_cen[ id_vx ]
		R_x1 = sub_Rcen.max()

		id_m = sub_Pm >= 0.8
		R_x2 = sub_Rcen[ id_m ].max()

		Rs_max[ dd ] = R_x1 + 0.
		Rs08_max[ dd ] = R_x2 + 0.

	##. save
	keys = [ 'ra', 'dec', 'z', 'clus_ID', 'rich', 'Rs_max', 'Rs_0.8_max' ]
	values = [ mp_ra, mp_dec, mp_z, mp_IDs, mp_rich, Rs_max, Rs08_max ]
	fill = dict( zip( keys, values ) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv('/home/xkchen/figs/extend_Zphoto_cat/region_sql_cat/' + 
					'redMaPPer_z-phot_0.2-0.3_%s-band_selected_clust_sql_size.csv' % band[ kk ],)

raise
"""


### === ### image load
# cat_path = '/home/xkchen/figs/extend_Zphoto_cat/region_sql_cat/'
cat_path = '/home/xkchen/data/SDSS/extend_Zphoto_cat/region_sql_cat/'
out_path = '/home/xkchen/data/SDSS/photo_RQ_data/'

band = ['r', 'g', 'i']

##.
for kk in range( 3 ):

	band_str = band[ kk ]

	##. location and z_obs
	dat = pds.read_csv( cat_path + 
				'redMaPPer_z-phot_0.2-0.3_%s-band_selected_clust_sql_size.csv' % band_str,)

	bcg_ra, bcg_dec, bcg_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

	##. Mpc / h ~(Rykoff et al. 2014)
	# R_cut = np.array( dat['Rs_max'] )     ##. use all member
	R_cut = np.array( dat['Rs_0.8_max'] )   ##. use all member with Pm >= 0.8

	##. 
	Da_x = Test_model.angular_diameter_distance( bcg_z ).value
	a_x = 1 / ( 1 + bcg_z )

	A_sx = ( (R_cut / h) * a_x / Da_x ) * rad2arcsec

	record_file = '/home/xkchen/img_load_err.txt'

	region_sql_func( bcg_ra, bcg_dec, bcg_z, A_sx, band_str, record_file, out_path )

