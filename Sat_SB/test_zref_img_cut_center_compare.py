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


### === catalog check
"""
##. images with BCG
img_file = '/home/xkchen/data/SDSS/photo_files/pos_offset_correct_imgs/resamp_img/photo-z_resamp_%s_ra%.3f_dec%.3f_z%.3f.fits'

list_order = 13

band_str = 'r'

tt = 1  ## median richness subsample

R_bins = np.array( [ 0, 300, 400, 550, 5000] )
bin_rich = [ 20, 30, 50, 210 ]


##. cluster catalog
all_clus = pds.read_csv( '/home/xkchen/fig_tmp/Extend_Mbcg_richbin_sat_cat/' + 
			'clust_rich_%d-%d_cat.csv' % (bin_rich[tt], bin_rich[tt + 1]),)

ref_ra, ref_dec, ref_z = np.array( all_clus['ra'] ), np.array( all_clus['dec'] ), np.array( all_clus['z'] )

ref_IDs = np.array( all_clus['clust_ID'] )
ref_IDs = ref_IDs.astype( int )



##. pre cut table
rand_cat = pds.read_csv( '/home/xkchen/data/SDSS/member_files/rich_binned_shufl_img/shufl_cat/' + 
			'clust_rich_%d-%d_%s-band_sat-shuffle-%d_position.csv' % (bin_rich[tt], bin_rich[tt+1], band_str, list_order),)

bcg_ra, bcg_dec, bcg_z = np.array( rand_cat['bcg_ra'] ), np.array( rand_cat['bcg_dec'] ), np.array( rand_cat['bcg_z'] )
sat_ra, sat_dec = np.array( rand_cat['sat_ra'] ), np.array( rand_cat['sat_dec'] )

shufl_sx, shufl_sy = np.array( rand_cat['cp_sx'] ), np.array( rand_cat['cp_sy'] )

set_IDs = np.array( rand_cat['orin_cID'] )
rand_IDs = np.array( rand_cat['shufl_cID'] )

set_IDs = set_IDs.astype( int )
rand_mp_IDs = rand_IDs.astype( int )



##. zref_img cut
cp_rand_cat = pds.read_csv( '/home/xkchen/img_zref_cut_test/cat/' + 
			'clust_rich_%d-%d_%s-band_sat-shuffle-%d_zref-img_cut-cat.csv' % (bin_rich[tt], bin_rich[tt + 1], band_str, list_order),)

cp_bcg_ra, cp_bcg_dec, cp_bcg_z = np.array( cp_rand_cat['bcg_ra'] ), np.array( cp_rand_cat['bcg_dec'] ), np.array( cp_rand_cat['bcg_z'] )
cp_sat_ra, cp_sat_dec = np.array( cp_rand_cat['sat_ra'] ), np.array( cp_rand_cat['sat_dec'] )

cp_shufl_sx, cp_shufl_sy = np.array( cp_rand_cat['cp_sx'] ), np.array( cp_rand_cat['cp_sy'] )

cp_set_IDs = np.array( cp_rand_cat['orin_cID'] )
cp_rand_IDs = np.array( cp_rand_cat['shufl_cID'] )

cp_set_IDs = cp_set_IDs.astype( int )
cp_rand_mp_IDs = cp_rand_IDs.astype( int )



da0, da1 = 0, 30

for dd in range( da0, da1 ):

	dc_ra, dc_dec, dc_z = bcg_ra[ dd ], bcg_dec[ dd ], bcg_z[ dd ]
	ds_ra, ds_dec = sat_ra[ dd ], sat_dec[ dd ]

	#.
	id_ux = ref_IDs == rand_mp_IDs[ dd ]
	ra_g, dec_g, z_g = ref_ra[ id_ux ][0], ref_dec[ id_ux ][0], ref_z[ id_ux ][0]

	c0_img = fits.open( img_file % (band_str, ra_g, dec_g, z_g),)
	c0_img_arr = c0_img[0].data


	#. 
	id_vx = ref_IDs == cp_rand_mp_IDs[ dd ]
	cp_ra_g, cp_dec_g, cp_z_g = ref_ra[ id_vx ][0], ref_dec[ id_vx ][0], ref_z[ id_vx ][0]

	c1_img = fits.open( img_file % (band_str, cp_ra_g, cp_dec_g, cp_z_g),)
	c1_img_arr = c1_img[0].data


	#.
	fig = plt.figure( figsize = (12, 5) )
	ax0 = fig.add_axes( [0.10, 0.10, 0.40, 0.80] )
	ax1 = fig.add_axes( [0.55, 0.10, 0.40, 0.80] )

	ax0.imshow( c0_img_arr, origin = 'lower', cmap = 'Greys',)# norm = mpl.colors.LogNorm(),)

	ax1.set_title('ra, dec, z, ra, dec = %.3f, %.3f, %.3f, %.4f, %.4f' % 
				(cp_bcg_ra[ dd ], cp_bcg_dec[ dd ], cp_bcg_z[ dd ], cp_sat_ra[ dd ], cp_sat_dec[ dd ]),)
	ax1.imshow( c1_img_arr, origin = 'lower', cmap = 'Greys',)# norm = mpl.colors.LogNorm(),)

	plt.savefig('/home/xkchen/figs/' + 
		'%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_BG-img.png' % 
		(band_str, dc_ra, dc_dec, dc_z, ds_ra, ds_dec), dpi = 300)
	plt.close()

"""


### === data load
cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_rich_rebin_sat_cat/'

img_path = '/home/xkchen/data/SDSS/member_files/rich_binned_sat_wBCG/resamp_img/'
d_file = img_path + 'Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits'

##. 
BG_file = ('/home/xkchen/data/SDSS/member_files/rich_binned_shufl_img/resamp_img/' + 
			'clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp.fits',)[0]

'/home/xkchen/data/SDSS/member_files/rich_binned_shufl_img'

##.
cp_file = ( '/home/xkchen/img_zref_cut_test/imgs/' + 
			'clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_img.fits',)[0]


##... original image location check
band_str = 'r'

bin_rich = [ 20, 30, 50, 210 ]
sub_name = ['low-rich', 'medi-rich', 'high-rich']

R_bins = np.array( [ 0, 300, 400, 550, 5000] )

list_order = 13

ll = 1

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( cat_path + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_mem-%s-band_pos-zref.csv'
			% ( bin_rich[ ll ], bin_rich[ll + 1], R_bins[tt], R_bins[tt + 1], band_str),)

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

	img_x, img_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

	N_sub = 20

	##.
	for dd in range( N_sub ):

		ra_g, dec_g, z_g = bcg_ra[ dd ], bcg_dec[ dd ], bcg_z[ dd ]

		ds_ra, ds_dec = sat_ra[ dd ], sat_dec[ dd ]

		ds_x, ds_y = img_x[ dd ], img_y[ dd ]

		data = fits.open( d_file % (band_str, ra_g, dec_g, z_g, ds_ra, ds_dec),)
		img = data[0].data

		##. BG_cut files
		bg_data = fits.open( BG_file % (band_str, ra_g, dec_g, z_g, ds_ra, ds_dec),)
		bg_img = bg_data[0].data


		##. zref_img cut test
		cp_data = fits.open( cp_file % (band_str, ra_g, dec_g, z_g, ds_ra, ds_dec),)
		cp_img = cp_data[0].data

		cp_x, cp_y = cp_img.shape[1] / 2, cp_img.shape[0] / 2


		# #. sat_img
		# plt.figure()
		# plt.imshow( img, origin = 'lower', cmap = 'Greys', norm = mpl.colors.LogNorm(),)

		# plt.scatter( ds_x, ds_y, s = 10, marker = 'o', facecolors = 'none', edgecolors = 'r',)
		# plt.scatter( cp_x, cp_y, s = 10, marker = 's', facecolors = 'none', edgecolors = 'b',)

		# plt.xlim( ds_x - 25, ds_x + 25 )
		# plt.ylim( ds_y - 25, ds_y + 25 )

		# plt.savefig('/home/xkchen/figs/' + 
		# 	'%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f.png' % 
		# 	(band_str, ra_g, dec_g, z_g, ds_ra, ds_dec), dpi = 300)
		# plt.close()


		#. BG_img
		fig = plt.figure( figsize = (12, 5) )
		ax0 = fig.add_axes( [0.10, 0.10, 0.40, 0.80] )
		ax1 = fig.add_axes( [0.55, 0.10, 0.40, 0.80] )

		ax0.imshow( bg_img, origin = 'lower', cmap = 'Greys',)# norm = mpl.colors.LogNorm(),)
		ax0.scatter( ds_x, ds_y, s = 10, marker = 'o', facecolors = 'none', edgecolors = 'r',)
		ax0.scatter( cp_x, cp_y, s = 10, marker = 's', facecolors = 'none', edgecolors = 'b',)

		ax1.set_title('zref_img cut')
		ax1.imshow( cp_img, origin = 'lower', cmap = 'Greys',)# norm = mpl.colors.LogNorm(),)
		ax1.scatter( ds_x, ds_y, s = 10, marker = 'o', facecolors = 'none', edgecolors = 'r',)
		ax1.scatter( cp_x, cp_y, s = 10, marker = 's', facecolors = 'none', edgecolors = 'b',)

		plt.savefig('/home/xkchen/figs/' + 
			'%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_BG-img.png' % 
			(band_str, ra_g, dec_g, z_g, ds_ra, ds_dec), dpi = 300)
		plt.close()

	raise

