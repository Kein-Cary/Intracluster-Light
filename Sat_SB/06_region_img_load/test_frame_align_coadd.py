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

#.
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
from reproject.mosaicking import reproject_and_coadd
import reproject
from drizzle import *
from drizzle import drizzle


###. cosmology
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)



### === ### (co-add images without weight)
def drizzle_demo_two( reference, outfile, infiles ):

	reflist = fits.open(reference)

	reference_wcs = awc.WCS( reflist[0].header)

	driz = drizzle.Drizzle( outwcs = reference_wcs )

	N_l = len( infiles )

	##.
	for pp in tqdm( range( N_l ) ):

		imlist = fits.open( infiles[ pp ] )

		image = imlist[0].data

		image_wcs = awc.WCS( imlist[0].header )

		driz.add_image( image, image_wcs )

		##.
		del image
		del image_wcs

	##. save
	driz.write( outfile )

	return


### === image align_ment compare
def pre_align_compare():

	dat = pds.read_csv('/home/xkchen/figs_cp/SDSS_img_load/cat_file/' + 
					'redMaPPer_z-phot_0.1-0.33_clust_sql_size.csv',)

	bcg_ra, bcg_dec, bcg_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

	band_str = 'r'

	ref_file = '/home/xkchen/figs_cp/SDSS_img_load/tmp_file/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

	for nn in range( 4, 5 ):

		ra_g, dec_g, z_g = bcg_ra[ nn ], bcg_dec[ nn ], bcg_z[ nn ]

		#.
		input_file = ref_file % (band_str, ra_g, dec_g, z_g)
		frame_files = glob.glob('/home/xkchen/figs_cp/SDSS_img_load/*.fits.bz2')
		out_file = '/home/xkchen/drizzle_test.fits'

		drizzle_demo_two( input_file, out_file, frame_files )

		#.
		data = fits.open( input_file )
		ref_wcs = awc.WCS( data[0].header )

		array_bgmatch, _ = reproject_and_coadd( frame_files, ref_wcs, shape_out = (1489, 2048), 
							combine_function = 'mean', match_background = True,
							reproject_function = reproject.reproject_exact, hdu_in = 0,)

		#.
		out_data = fits.open( out_file )
		out_img = out_data[1].data

		#.
		fig = plt.figure( figsize = (12.8, 4.8),)
		ax0 = fig.add_axes([0.08, 0.10, 0.40, 0.85])
		ax1 = fig.add_axes([0.53, 0.10, 0.40, 0.85])

		ax0.set_title('Drizzle')
		ax0.imshow( out_img, origin = 'lower', cmap = 'Greys', norm = mpl.colors.LogNorm(),)

		ax1.set_title('Reproject_coadd')
		ax1.imshow( array_bgmatch, origin = 'lower', cmap = 'Greys', norm = mpl.colors.LogNorm(),)

		plt.savefig('/home/xkchen/align_test.png', dpi = 300)
		plt.close()

	return


### === 
dat = pds.read_csv('/home/xkchen/figs_cp/SDSS_img_load/cat_file/' + 
				'redMaPPer_z-phot_0.1-0.33_clust_sql_size.csv',)

bcg_ra, bcg_dec, bcg_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

sub_IDs = np.array( dat['clus_ID'] )
sub_IDs = sub_IDs.astype( int )

N_g = len( bcg_ra )

band_str = 'r'

##.
for nn in range( 4, 5 ):

	##. ref_point images
	ra_g, dec_g, z_g = bcg_ra[ nn ], bcg_dec[ nn ], bcg_z[ nn ]

	ref_file = '/home/xkchen/figs_cp/SDSS_img_load/tmp_file/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

	frame_files = glob.glob('/home/xkchen/figs_cp/SDSS_img_load/*.fits.bz2')

	data_ref = fits.open( ref_file % (band_str, ra_g, dec_g, z_g),)
	ref_img = data_ref[0].data
	Head = data_ref[0].header
	wcs_ref = awc.WCS( Head )
	N_x, N_y = ref_img.shape[1], ref_img.shape[0]


	##. set the out frame arround the ref_img
	Nx0 = N_x + 0
	n_wid = 5

	out_arr = np.zeros( (Nx0 * n_wid, Nx0 * n_wid), dtype = np.float32 )
	cx0, cy0 = np.int( out_arr.shape[1] / 2 ), np.int( out_arr.shape[0] / 2 )


	##. 
	hdu = fits.PrimaryHDU()
	hdu.data = out_arr

	Head_0 = Head.copy()

	Head_0['CRPIX1'] = cx0
	Head_0['CRPIX2'] = cy0

	Head_0['NAXIS1'] = out_arr.shape[1]
	Head_0['NAXIS2'] = out_arr.shape[0]

	hdu.header = Head_0
	hdu.writeto( '/home/xkchen/img_comb_test.fits', overwrite = True)

	##.
	frame_files = glob.glob('/home/xkchen/figs_cp/SDSS_img_load/*.fits.bz2')
	out_file = '/home/xkchen/figs_cp/SDSS_img_load/drizzle_frame_r-band_test.fits'
	input_file = '/home/xkchen/img_comb_test.fits'

	drizzle_demo_two( input_file, out_file, frame_files )

	##.
	cmd = 'rm -r /home/xkchen/img_comb_test.fits'
	pa = subpro.Popen(cmd, shell = True)
	pa.wait()


### === ### source location
m_dat = fits.getdata('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')

clus_IDs = np.array( m_dat.ID )   ## the ID number of cluster in redMaPPer
P_mem = np.array( m_dat.P )
p_ra, p_dec = np.array( m_dat.RA ), np.array( m_dat.DEC )

for nn in range( 4, 5 ):

	##. ref_point images
	ra_g, dec_g, z_g = bcg_ra[ nn ], bcg_dec[ nn ], bcg_z[ nn ]

	ref_file = '/home/xkchen/figs_cp/SDSS_img_load/tmp_file/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

	data_ref = fits.open( ref_file % (band_str, ra_g, dec_g, z_g),)
	ref_img = data_ref[0].data
	Head = data_ref[0].header
	wcs_ref = awc.WCS( Head )
	N_x, N_y = ref_img.shape[1], ref_img.shape[0]

	Lx = np.int( N_x / 2 )
	Ly = np.int( N_y / 2 )


	##. combined image
	data = fits.open('/home/xkchen/figs_cp/SDSS_img_load/drizzle_frame_r-band_test.fits')
	comb_img = data[1].data
	wcs_lis = awc.WCS( data[1].header )

	cx0, cy0 = data[1].header['CRPIX1'], data[1].header['CRPIX1']

	dx0 = cx0 - Lx
	dy0 = cy0 - Ly


	##. member location
	id_vx = clus_IDs == sub_IDs[ nn ]
	mp_s_ra, mp_s_dec = p_ra[ id_vx ], p_dec[ id_vx ]
	mp_Pm = P_mem[ id_vx ]

	sx_0, sy_0 = wcs_ref.all_world2pix( mp_s_ra, mp_s_dec, 0 )
	sx_0, sy_0 = sx_0 + (cx0 - Lx), sy_0 + (cy0 - Ly)

	sx_1, sy_1 = wcs_lis.all_world2pix( mp_s_ra, mp_s_dec, 0 )

	d_R = np.sqrt( (sx_0 - cx0)**2 + (sy_0 - cy0)**2 )
	dR_max = np.ceil( d_R.max() + 50 )


	##.
	fig = plt.figure( )
	ax0 = fig.add_axes([0.08, 0.10, 0.80, 0.85])

	ax0.imshow( comb_img, origin = 'lower', cmap = 'Greys', 
				norm = mpl.colors.LogNorm( vmin = 1e-4, vmax = 1e0),)

	rect = mpathes.Rectangle( (dx0, dy0), N_x, N_y, ec = 'r', fc = 'none',)
	ax0.add_patch( rect )

	plt.savefig('/home/xkchen/adjust_frame_test.png', dpi = 300)
	plt.close()


	fig = plt.figure( )
	ax0 = fig.add_axes([0.08, 0.10, 0.80, 0.85])

	ax0.imshow( comb_img, origin = 'lower', cmap = 'Greys', 
				norm = mpl.colors.LogNorm( vmin = 1e-4, vmax = 1e0),)

	rect = mpathes.Rectangle( (dx0, dy0), N_x, N_y, ec = 'r', fc = 'none',)
	ax0.add_patch( rect )


	id_px = mp_Pm >= 0.8

	ax0.scatter( sx_0[ id_px ], sy_0[ id_px ], marker = 'o', facecolors = 'none', edgecolors = 'r', alpha = 0.75)

	ax0.scatter( sx_0[ id_px == False ], sy_0[ id_px == False ], 
				marker = 's', facecolors = 'none', edgecolors = 'b', alpha = 0.75)

	ax0.scatter( sx_1, sy_1, marker = 's', facecolors = 'none', edgecolors = 'k', ls = ':', alpha = 0.75 )

	ax0.set_xlim( cx0 - dR_max, cx0 + dR_max )
	ax0.set_ylim( cy0 - dR_max, cy0 + dR_max )

	plt.savefig('/home/xkchen/adjust_frame_sat_test.png', dpi = 300)
	plt.close()

