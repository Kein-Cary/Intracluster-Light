"""
Testing for query surrounding image frame for given target galaxy in SDSS catalog
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


### === ### image query test
cat_path = '/home/xkchen/data/SDSS/extend_Zphoto_cat/region_sql_cat/'
# out_path = '/home/xkchen/data/SDSS/photo_RQ_data/'
out_path = '/home/xkchen/figs/'

band = ['r', 'g', 'i']

##.
for kk in range( 1 ):

	band_str = band[ kk ]

	##. location and z_obs
	dat = pds.read_csv( cat_path + 'redMaPPer_z-phot_0.1-0.33_clust_sql_size.csv',)

	bcg_ra, bcg_dec, bcg_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

	##. Mpc / h ~(Rykoff et al. 2014)
	# R_cut = np.array( dat['Rs_max'] )     ##. use all member
	R_cut = np.array( dat['Rs_0.8_max'] )   ##. use all member with Pm >= 0.8

	##. 
	Da_x = Test_model.angular_diameter_distance( bcg_z ).value
	a_x = 1 / ( 1 + bcg_z )

	A_sx = ( (R_cut / h) * a_x / Da_x ) * rad2arcsec

	##. test for large A_sx cluster
	record_file = '/home/xkchen/img_load_err.txt'

	region_sql_func( bcg_ra[35:37], bcg_dec[35:37], bcg_z[35:37], A_sx[35:37], band_str, record_file, out_path )


raise


### === ### special cluster view
##.
dat = pds.read_csv('/home/xkchen/data/SDSS/extend_Zphoto_cat/region_sql_cat/redMaPPer_z-phot_0.1-0.33_clust_sql_size.csv',)

bcg_ra, bcg_dec, bcg_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

sub_IDs = np.array( dat['clus_ID'] )
sub_IDs = sub_IDs.astype( int )

N_g = len( bcg_ra )

band_str = 'r'


##.
m_dat = fits.getdata('/home/xkchen/data/SDSS/redmapper/redmapper_dr8_public_v6.3_members.fits')

clus_IDs = np.array( m_dat.ID )   ## the ID number of cluster in redMaPPer
P_mem = np.array( m_dat.P )
p_ra, p_dec = np.array( m_dat.RA ), np.array( m_dat.DEC )


##.
for nn in range( 35, 36 ):

	##. ref_point images
	ra_g, dec_g, z_g = bcg_ra[ nn ], bcg_dec[ nn ], bcg_z[ nn ]

	ref_file = '/home/xkchen/Downloads/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

	data_ref = fits.open( ref_file % (band_str, ra_g, dec_g, z_g),)
	ref_img = data_ref[0].data
	Head = data_ref[0].header
	wcs_ref = awc.WCS( Head )
	N_x, N_y = ref_img.shape[1], ref_img.shape[0]

	Lx = np.int( N_x / 2 )
	Ly = np.int( N_y / 2 )


	##. member location
	id_vx = clus_IDs == sub_IDs[ nn ]
	mp_s_ra, mp_s_dec = p_ra[ id_vx ], p_dec[ id_vx ]
	mp_Pm = P_mem[ id_vx ]

	sx_0, sy_0 = wcs_ref.all_world2pix( mp_s_ra, mp_s_dec, 0 )


	##.
	fig = plt.figure()
	ax = fig.add_axes([0.10, 0.10, 0.85, 0.80])

	ax.imshow( ref_img, origin = 'lower', cmap = 'Greys', norm = mpl.colors.LogNorm( vmin = 1e-4, vmax = 1e0), )
	ax.scatter( sx_0, sy_0, marker = 's', facecolors = 'none', edgecolors = 'r',)
	plt.savefig('/home/xkchen/clust-%d_member_located.png' % nn, dpi = 300)
	plt.close()

