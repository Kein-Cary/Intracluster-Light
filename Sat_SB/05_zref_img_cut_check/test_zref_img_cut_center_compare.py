"""
this file use to check and unstanding repeated IDs in field galaxy query catalog
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
import scipy.stats as sts

from scipy import interpolate as interp
from astropy import cosmology as apcy
from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy.table import Table

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
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator



### === ### catalog check
"""
shufl_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/shufl_list/tables/'

bin_rich = [ 20, 30, 50, 210 ]

##.
for kk in range( 1,2 ):

	##.
	# cat = pds.read_csv( shufl_path + 
	# 	'clust_rich_%d-%d_r-band_sat-shuffle-13_position.csv' % (bin_rich[kk], bin_rich[kk + 1]),)

	cat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/zref_imgcut_check/cat/' + 
		'clust_rich_30-50_r-band_sat-shuffle-13_zref-img_cut-cat.csv')

	clus_ID = np.array( cat['orin_cID'] )
	clus_ID = clus_ID.astype( int )

	bcg_ra, bcg_dec, bcg_z = np.array( cat['bcg_ra'] ), np.array( cat['bcg_dec'] ), np.array( cat['bcg_z'] )

	sat_ra, sat_dec = np.array( cat['sat_ra'] ), np.array( cat['sat_dec'] )

	rand_IDs = np.array( cat['shufl_cID'] )


	##. previous
	dat = pds.read_csv(
		'/home/xkchen/clust_rich_%d-%d_r-band_sat-shuffle-13_position.csv' % (bin_rich[kk], bin_rich[kk + 1]),)

	cp_clus_ID = np.array( dat['orin_cID'] )
	cp_clus_ID = cp_clus_ID.astype( int )

	cp_bcg_ra, cp_bcg_dec, cp_bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )

	cp_sat_ra, cp_sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

	cp_rand_IDs = np.array( dat['shufl_cID'] )


	##.
	sum_0 = np.sum( bcg_ra - cp_bcg_ra )
	print( sum_0 )

	sum_1 = np.sum( bcg_dec - cp_bcg_dec )
	print( sum_1 )

	sum_2 = np.sum( sat_ra - cp_sat_ra )
	print( sum_2 )

	sum_3 = np.sum( sat_dec - cp_sat_dec )
	print( sum_3 )

	sum_4 = np.sum( clus_ID - cp_clus_ID )
	print( sum_4 )

	sum_5 = np.sum( rand_IDs - cp_rand_IDs )
	print( sum_5 )

	print('*' * 10)

raise
"""


### === ### image cut check
home = '/home/xkchen/data/SDSS/'
cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_richbin_sat_cat/'

bin_rich = [ 20, 30, 50, 210 ]

band_str = 'r'

list_order = 13

for kk in range( 1,2 ):

	#. ref_cluster catalog
	dat = pds.read_csv( cat_path + 'clust_rich_%d-%d_cat.csv' % (bin_rich[kk], bin_rich[kk + 1]),)
	ref_ra, ref_dec, ref_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

	ref_IDs = np.array( dat['clust_ID'] )
	ref_IDs = ref_IDs.astype( int )


	#. orin_img cut table
	img_file = home + 'photo_files/pos_offset_correct_imgs/mask_img/photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

	rand_cat = pds.read_csv( '/home/xkchen/data/SDSS/member_files/shufl_img_wBCG/shufl_cat/' + 
				'clust_rich_%d-%d_%s-band_sat-shuffle-%d_position.csv' % (bin_rich[kk], bin_rich[kk+1], band_str, list_order),)

	bcg_ra, bcg_dec, bcg_z = np.array( rand_cat['bcg_ra'] ), np.array( rand_cat['bcg_dec'] ), np.array( rand_cat['bcg_z'] )
	sat_ra, sat_dec = np.array( rand_cat['sat_ra'] ), np.array( rand_cat['sat_dec'] )

	shufl_sx, shufl_sy = np.array( rand_cat['cp_sx'] ), np.array( rand_cat['cp_sy'] )

	set_IDs = np.array( rand_cat['orin_cID'] )
	rand_IDs = np.array( rand_cat['shufl_cID'] )

	set_IDs = set_IDs.astype( int )
	rand_mp_IDs = rand_IDs.astype( int )

	R_cut = 320   ##. pixels


	#. zref_img cut table
	zref_img_file = home + 'photo_files/pos_offset_correct_imgs/resamp_img/photo-z_resamp_%s_ra%.3f_dec%.3f_z%.3f.fits'

	cp_cat = pds.read_csv( '/home/xkchen/img_zref_cut_test/cat/' + 
				'clust_rich_%d-%d_%s-band_sat-shuffle-%d_zref-img_cut-cat.csv' % (bin_rich[kk], bin_rich[kk + 1], band_str, list_order),)

	cp_bcg_ra, cp_bcg_dec, cp_bcg_z = np.array( cp_cat['bcg_ra'] ), np.array( cp_cat['bcg_dec'] ), np.array( cp_cat['bcg_z'] )
	cp_sat_ra, cp_sat_dec = np.array( cp_cat['sat_ra'] ), np.array( cp_cat['sat_dec'] )

	cp_shufl_sx, cp_shufl_sy = np.array( cp_cat['cp_sx'] ), np.array( cp_cat['cp_sy'] )

	cp_set_IDs = np.array( cp_cat['orin_cID'] )
	cp_rand_IDs = np.array( cp_cat['shufl_cID'] )

	cp_set_IDs = set_IDs.astype( int )
	cp_rand_mp_IDs = rand_IDs.astype( int )

	cp_R_cut = np.array( cp_cat['cut_size'] )


	##. check with top_20
	da0, da1 = 50, 80

	for pp in range( da0, da1 ):

		##. orin_img cut
		kk_px_0, kk_py_0 = shufl_sx[ pp ], shufl_sy[ pp ]
		kk_ra_0, kk_dec_0 = sat_ra[ pp ], sat_dec[ pp ]

		orin_ra_0, orin_dec_0, orin_z_0 = bcg_ra[ pp ], bcg_dec[ pp ], bcg_z[ pp ]

		#. shuffle mapped cluster
		id_ux = ref_IDs == rand_mp_IDs[ pp ]
		ra_g_0, dec_g_0, z_g_0 = ref_ra[ id_ux ][0], ref_dec[ id_ux ][0], ref_z[ id_ux ][0]

		cp_img = fits.open( img_file % (band_str, ra_g_0, dec_g_0, z_g_0),)
		cp_img_arr = cp_img[0].data


		##. zref_img cut
		kk_px_1, kk_py_1 = cp_shufl_sx[ pp ], cp_shufl_sy[ pp ]
		kk_ra_1, kk_dec_1 = cp_sat_ra[ pp ], cp_sat_dec[ pp ]

		orin_ra_1, orin_dec_1, orin_z_1 = cp_bcg_ra[ pp ], cp_bcg_dec[ pp ], cp_bcg_z[ pp ]

		#. shuffle mapped cluster
		id_ux = ref_IDs == cp_rand_mp_IDs[ pp ]
		ra_g_1, dec_g_1, z_g_1 = ref_ra[ id_ux ][0], ref_dec[ id_ux ][0], ref_z[ id_ux ][0]

		zref_img = fits.open( zref_img_file % (band_str, ra_g_1, dec_g_1, z_g_1),)
		zref_img_arr = zref_img[0].data


		##. cut out image
		#. cut and resampling ~ (corrected)
		BG_file = ( '/home/xkchen/data/SDSS/member_files/shufl_img_wBCG/resamp_img/' + 
					'clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp.fits',)[0]

		#. cut with BCGs~( corrected )
		# BG_file = ( '/home/xkchen/data/SDSS/member_files/shufl_img_wBCG/mask_img/' + 
		# 				'clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_img.fits',)[0]

		#. no-BCG cut~( corrected )
		# BG_file = ( '/home/xkchen/data/SDSS/member_files/shufl_img_woBCG/resamp_img/' + 
		# 			'clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp.fits',)[0]


		##.
		cp_BG_file = ( '/home/xkchen/img_zref_cut_test/imgs/' + 
					'clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_img.fits',)[0]

		#.
		orin_cut = fits.open( BG_file % (band_str, orin_ra_0, orin_dec_0, orin_z_0, kk_ra_0, kk_dec_0),)
		cut_img = orin_cut[0].data

		#.
		cp_cut = fits.open( cp_BG_file % (band_str, orin_ra_1, orin_dec_1, orin_z_1, kk_ra_1, kk_dec_1),)
		cp_cut_img = cp_cut[0].data


		#. cutout images
		fig = plt.figure( figsize = (13.12, 9.84) )
		ax0 = fig.add_axes([0.05, 0.55, 0.40, 0.45])
		ax1 = fig.add_axes([0.55, 0.55, 0.40, 0.45])

		ax2 = fig.add_axes([0.05, 0.05, 0.40, 0.45])
		ax3 = fig.add_axes([0.55, 0.05, 0.40, 0.45])

		#.
		ax0.set_title('ra%.3f,dec%.3f,z%.3f; ra%.4f,dec%.4f' % 
					(orin_ra_0, orin_dec_0, orin_z_0, kk_ra_0, kk_dec_0),)

		ax0.imshow( cp_img_arr, origin = 'lower', cmap = 'Greys', vmin = -1e-2, vmax = 1e-2,)

		rect = mpathes.Circle( (kk_px_0, kk_py_0), radius = R_cut, ec = 'r', fc = 'none',)
		ax0.add_patch( rect )


		ax1.set_title('ra%.3f,dec%.3f,z%.3f; ra%.4f,dec%.4f' % 
					(orin_ra_1, orin_dec_1, orin_z_1, kk_ra_1, kk_dec_1),)

		ax1.imshow( zref_img_arr, origin = 'lower', cmap = 'Greys', vmin = -1e-2, vmax = 1e-2,)

		rect = mpathes.Circle( (kk_px_1, kk_py_1), radius = cp_R_cut[ pp ], ec = 'r', fc = 'none',)
		ax1.add_patch( rect )

		#.
		ax2.set_title('orin_cut : ra%.3f,dec%.3f,z%.3f' % (ra_g_0, dec_g_0, z_g_0),)
		ax2.imshow( cut_img, origin = 'lower', cmap = 'Greys', vmin = -1e-2, vmax = 1e-2,)

		ax3.set_title('zref_cut : ra%.3f,dec%.3f,z%.3f' % (ra_g_1, dec_g_1, z_g_1),)
		ax3.imshow( cp_cut_img, origin = 'lower', cmap = 'Greys', vmin = -1e-2, vmax = 1e-2,)

		plt.savefig('/home/xkchen/figs/check_%d_img.png' % pp, dpi = 300)
		plt.close()


### === ### center points of cut out image

