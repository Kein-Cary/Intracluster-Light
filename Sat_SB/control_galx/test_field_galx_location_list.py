import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Ellipse

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc
import scipy.stats as sts

from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy.stats import binned_statistic as binned

#.
from Mass_rich_radius import rich2R_Simet
# from img_sat_fig_out_mode import zref_sat_pos_func
from img_sat_fig_out_mode import arr_zref_pos_func

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1. - Omega_m
Omega_k = 1. - (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']


### === data load
home = '/home/xkchen/data/SDSS/'
"""
img_file = home + 'member_files/redMap_contral_galx/mask_img/ctrl-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img.fits'

for pp in range( 1 ):

	band_str = band[ pp ]

	#.
	dat = pds.read_csv( '/home/xkchen/data/SDSS/member_files/redMap_contral_galx/control_cat/' + 
						'random_field-galx_map_%s-band_cat.csv' % band_str )

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec, sat_z = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] ), np.array( dat['sat_z'] )

	#.
	Ns = len( sat_ra )

	tmp_m_x, tmp_m_y = [], []
	tmp_pk_x, tmp_pk_y = [], []

	for kk in range( Ns ):

		ra_g, dec_g, z_g = bcg_ra[ kk ], bcg_dec[ kk ], bcg_z[ kk]

		kk_ra, kk_dec = sat_ra[ kk ], sat_dec[ kk ]

		img_data = fits.open( img_file % ( band_str, ra_g, dec_g, z_g, kk_ra, kk_dec),)

		pp_mx, pp_my = img_data[0].header['CENTER_X'], img_data[0].header['CENTER_Y']
		pp_pkx, pp_pky = img_data[0].header['PEAK_X'], img_data[0].header['PEAK_Y']

		tmp_m_x.append( pp_mx )
		tmp_m_y.append( pp_my )

		tmp_pk_x.append( pp_pkx )
		tmp_pk_y.append( pp_pky )

	tmp_m_x, tmp_m_y = np.array( tmp_m_x ), np.array( tmp_m_y )
	tmp_pk_x, tmp_pk_y = np.array( tmp_pk_x ), np.array( tmp_pk_y )

	#. save location list
	keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'mx', 'my', 'peak_x', 'peak_y']
	values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, tmp_m_x, tmp_m_y, tmp_pk_x, tmp_pk_y ]

	fill = dict( zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/fig_tmp/Extend_Mbcg_ctrlGalx_cat/' + 
					'random_field-galx_map_%s-band_frame-limit_pos-compare.csv' % band_str,)

"""


### === estimate galaxy location in img_cut after resampling
##... position at z_ref

for kk in range( 1 ):

	band_str = band[ kk ]

	##.
	dat = pds.read_csv('/home/xkchen/fig_tmp/Extend_Mbcg_ctrlGalx_cat/' + 
					'random_field-galx_map_%s-band_frame-limit_pos-compare.csv' % band_str,)

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

	pre_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )

	mx, my = np.array( dat['mx'] ), np.array( dat['my'] )
	pk_x, pk_y = np.array( dat['peak_x'] ), np.array( dat['peak_y'] )

	_off_cx, _off_cy = mx - 1, my - 1    #. position adjust.( based on SDSS_check )

	# off_R = np.sqrt( (mx - pk_x)**2 + (my - pk_y)**2 )

	##.
	cat = pds.read_csv('/home/xkchen/fig_tmp/Extend_Mbcg_ctrlGalx_cat/' + 
						'redMap_map_control_galx_%s-band_cut_pos.csv' % band_str,)

	sat_z = np.array( cat['sat_z'] )

	##.
	keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_z', 'cut_cx', 'cut_cy']
	values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, sat_z, _off_cx, _off_cy ]
	fill = dict( zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/fig_tmp/Extend_Mbcg_ctrlGalx_cat/' + 
					'random_field-galx_map_frame-limit_%s-band_pos.csv' % band_str,)


	ref_satx, ref_saty = arr_zref_pos_func( _off_cx, _off_cy, sat_z, z_ref, pixel )

	##.
	keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_z', 'sat_x', 'sat_y' ]
	values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, sat_z, ref_satx, ref_saty ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( '/home/xkchen/fig_tmp/Extend_Mbcg_ctrlGalx_cat/' + 
				'random_field-galx_map_frame-limit_%s-band_pos_z-ref.csv' % band_str,)

