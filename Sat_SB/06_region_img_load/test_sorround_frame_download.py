"""
Testing for query surrounding image frame for given target galaxy in SDSS catalog
"""
import sys 
sys.path.append('/home/xkchen/tool/Conda/Tools/normitems')

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

#.
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpathes

#.
import changds


###. cosmology
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)


### === ### record the cluster image and satellite centric distance
"""
#. cluster ~ (0.1~0.33 )
c_dat = fits.getdata('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')

RA = np.array( c_dat.RA )
DEC = np.array( c_dat.DEC )
ID = np.array( c_dat.OBJID )

rich = np.array( c_dat.LAMBDA )
Z_photo = np.array( c_dat.Z_LAMBDA )

ord_dex = np.array( c_dat.ID )

#. 0.1~z~0.33
idx_lim = ( Z_photo >= 0.1 ) & ( Z_photo <= 0.33 )
lim_ra, lim_dec, lim_z = RA[ idx_lim ], DEC[ idx_lim ], Z_photo[ idx_lim ]
lim_ID = ID[ idx_lim ]

lim_rich = rich[ idx_lim ]
lim_order = ord_dex[ idx_lim ]

N_z = len( lim_z )


#. member catalog
m_dat = fits.getdata('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')

clus_IDs = np.array( m_dat.ID )   ## the ID number of cluster in redMaPPer
R_cen = np.array( m_dat.R )       ## Mpc / h
P_mem = np.array( m_dat.P )
m_objIDs = np.array( m_dat.OBJID )
m_ra, m_dec = np.array( m_dat.RA ), np.array( m_dat.DEC )


#. record the maximum of satellite centric distance
Rs_max = np.zeros( N_z,)
Rs08_max = np.zeros( N_z,)  ##. record of Pm_cut = 0.8
sep_R = np.zeros( N_z, )

for kk in range( N_z ):

	id_vx = clus_IDs == lim_order[ kk ]
	sub_Pm, sub_Rcen = P_mem[ id_vx ], R_cen[ id_vx ]
	sub_ra, sub_dec = m_ra[ id_vx ], m_dec[ id_vx ]

	R_x1 = sub_Rcen.max()

	id_m = sub_Pm >= 0.8
	R_x2 = sub_Rcen[ id_m ].max()

	Rs_max[ kk ] = R_x1 + 0.
	Rs08_max[ kk ] = R_x2 + 0.

	##. record the angle separation on sky coordinate
	kk_pos = SkyCoord( ra = lim_ra[ kk ] * U.deg, dec = lim_dec[ kk ] * U.deg )
	mp_pos = SkyCoord( ra = sub_ra * U.deg, dec = sub_dec * U.deg )

	dp_aR = kk_pos.separation( mp_pos )
	sep_R[ kk ] = dp_aR.max().to(U.arcsec).value


##. save
keys = [ 'ra', 'dec', 'z', 'clus_ID', 'rich', 'Rs_max', 'Rs_0.8_max', 'angl_sep' ]
values = [ lim_ra, lim_dec, lim_z, lim_order, lim_rich, Rs_max, Rs08_max, sep_R ]
fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )
out_data.to_csv('/home/xkchen/figs_cp/SDSS_img_load/cat_file/' + 
				'redMaPPer_z-phot_0.1-0.33_clust_sql_size.csv',)

raise
"""


### === ### image query based on given coordinate points

##. location and z_obs
dat = pds.read_csv('/home/xkchen/figs_cp/SDSS_img_load/cat_file/' + 
				'redMaPPer_z-phot_0.1-0.33_clust_sql_size.csv',)

bcg_ra, bcg_dec, bcg_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

##. Mpc / h ~(Rykoff et al. 2014)
R_cut_0 = np.array( dat['Rs_max'] )
R_cut_1 = np.array( dat['Rs_0.8_max'] )

##. 
Da_x = Test_model.angular_diameter_distance( bcg_z ).value
a_x = 1 / ( 1 + bcg_z )

A_sx_0 = ( (R_cut_0 / h) * a_x / Da_x ) * rad2arcsec
A_sx_1 = ( (R_cut_1 / h) * a_x / Da_x ) * rad2arcsec

N_g = len( bcg_ra )

band_str = 'r'

# for nn in range( N_g ):
for nn in range( 4, 5 ):

	##.
	# R_A = A_sx_0[ nn ] * U.arcsec
	R_A = A_sx_1[ nn ] * U.arcsec

	##.
	pos = coords.SkyCoord( '%fd %fd' % ( bcg_ra[ nn ], bcg_dec[ nn ] ), frame = 'icrs', unit = 'deg')
	xid = SDSS.query_region( pos, spectro = False, radius = R_A, timeout = None,)

	run_ID = xid['run']
	rerun_ID = xid['rerun']
	camcol_ID = xid['camcol']
	field_ID = xid['field']

	##. find ref_frame ~ (the closed one)
	aa = np.array( [ xid['ra'], xid['dec'] ] )
	da = np.sqrt( ( aa[0,:] - bcg_ra[ nn ] )**2 + ( aa[1,:] - bcg_dec[ nn ])**2 )

	##.
	dl = da.tolist()
	pl = dl.index( np.min( da ) )
	s_bgn = changds.chidas_int( xid[pl][3],6 )
	s_end = changds.chidas_int( xid[pl][6],4 )

	##.
	url_road = ( 'http://data.sdss.org/sas/dr12/boss/photoObj/frames/%d/%d/%d/frame-%s-%s-%d-%s.fits.bz2' %
					( xid[pl][4], xid[pl][3], xid[pl][5], band_str, s_bgn, xid[pl][5], s_end),)[0]

	out_lis = ( '/home/xkchen/figs_cp/SDSS_img_load/tmp_file/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % 
					(band_str, bcg_ra[ nn ], bcg_dec[ nn ], bcg_z[ nn ]),)[0]

	wget.download( url_road, out_lis )

	##. also download psfField for BCG modelling
	psf_url = ( 'https://data.sdss.org/sas/dr12/boss/photo/redux/%d/%d/objcs/%d/psField-%s-%s-%s.fit' % 
					( xid[pl][4], xid[pl][3], xid[pl][5], s_bgn, xid[pl][5], s_end),)[0]

	out_link = ( '/home/xkchen/figs_cp/SDSS_img_load/tmp_file/psfField_ra%.3f_dec%.3f_z%.3f.fit' % 
					(bcg_ra[ nn ], bcg_dec[ nn ], bcg_z[ nn ]),)[0]

	wget.download( psf_url, out_link )


	##. map sorround images
	run_set = np.array( list( set( run_ID ) ) )
	rerun_set = np.array( list( set( rerun_ID ) ) )
	camcol_set = np.array( list( set( camcol_ID ) ) )
	field_set = np.array( list( set( field_ID ) ) )

	N0 = len( run_set )
	N1 = len( rerun_set )
	N2 = len( camcol_set )
	N3 = len( field_set )

	print( '*' * 20 )
	print( N0 * N1 * N2 * N3 )

	for kk in range( N0 ):

		for pp in range( N1 ):

			for dd in range( N2 ):

				for mm in range( N3 ):

					try:
						##. download surrounding fields
						url_lod = ( 'https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/%d/%d/%d/frame-%s-%s-%d-%s.fits.bz2' % 
									( rerun_set[ pp ], run_set[ kk ], camcol_set[ dd ], band_str, str(run_set[ kk ]).zfill( 6 ), camcol_set[ dd ], str(field_set[ mm ]).zfill(4) ),)[0]

						out_lis = ( '/home/xkchen/figs_cp/SDSS_img_load/frame-%s-%s-%d-%s.fits.bz2' % 
									( band_str, str(run_set[ kk ]).zfill( 6 ), camcol_set[ dd ], str(field_set[ mm ]).zfill(4) ),)[0]

						wget.download( url_lod, out_lis )

					except:
						##. save the "exception" case
						doc = open('/home/xkchen/Downloads/no_match_files.txt', 'w')
						s = '\n rerun = %d, run = %d, camcol=%d, field=%d \n' % ( rerun_set[ pp ], run_set[ kk ], camcol_set[ dd ], field_set[ mm ] )
						print(s, file = doc)
						doc.close()

						continue

raise


