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
from img_sat_fig_out_mode import zref_sat_pos_func


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


"""
### === satellite peak_xy and centric_xy compare (based on the cutout satellite images)
home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

band_str = band[ rank ]

for ll in range( 2 ):

	if ll == 0:
		img_file = home + 'member_files/cutsize_test/mask_img/Sat_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img_wider.fits'
	
	if ll == 1:
		img_file = home + 'member_files/cutsize_test/mask_img/Sat_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img_small.fits'


	dat = pds.read_csv(home + 'member_files/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv')
	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec = np.array( dat['ra'] ), np.array( dat['dec'] )


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

	if ll == 0:
		out_data.to_csv( '/home/xkchen/frame-lim_Pm-cut_exlu-BCG_member_%s-band_wider-cut_pos-compare.csv' % band_str,)

	if ll == 1:
		out_data.to_csv( '/home/xkchen/frame-lim_Pm-cut_exlu-BCG_member_%s-band_small-cut_pos-compare.csv' % band_str,)

"""


### === position at zref
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/cutsize_test/pos_cat/'

"""
for ll in range( 2 ):

	for kk in range( 3 ):

		band_str = band[ kk ]

		if ll == 0:
			dat = pds.read_csv( cat_path + 'frame-lim_Pm-cut_exlu-BCG_member_%s-band_small-cut_pos-compare.csv' % band_str )
		
		if ll == 1:
			dat = pds.read_csv( cat_path + 'frame-lim_Pm-cut_exlu-BCG_member_%s-band_wider-cut_pos-compare.csv' % band_str )


		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

		mx, my = np.array( dat['mx'] ), np.array( dat['my'] )
		pk_x, pk_y = np.array( dat['peak_x'] ), np.array( dat['peak_y'] )

		_off_cx, _off_cy = mx - 1, my - 1    #. position adjust


		#. location at z_obs
		keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'cut_cx', 'cut_cy']
		values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, _off_cx, _off_cy ]
		fill = dict( zip( keys, values) )
		out_data = pds.DataFrame( fill )

		if ll == 0:
			out_data.to_csv(cat_path + 'frame-lim_Pm-cut_exlu-BCG_member_%s-band_small-cut_member_pos.csv' % band_str,)

		if ll == 1:
			out_data.to_csv(cat_path + 'frame-lim_Pm-cut_exlu-BCG_member_%s-band_wider-cut_member_pos.csv' % band_str,)

		#.
		if ll == 0:
			cat_file = cat_path + 'frame-lim_Pm-cut_exlu-BCG_member_%s-band_small-cut_member_pos.csv' % band_str
			out_file = cat_path + 'frame-lim_Pm-cut_exlu-BCG_member_%s-band_small-cut_member_pos_z-ref.csv' % band_str

		if ll == 1:
			cat_file = cat_path + 'frame-lim_Pm-cut_exlu-BCG_member_%s-band_wider-cut_member_pos.csv' % band_str
			out_file = cat_path + 'frame-lim_Pm-cut_exlu-BCG_member_%s-band_wider-cut_member_pos_z-ref.csv' % band_str

		zref_sat_pos_func( cat_file, z_ref, out_file, pixel )

"""

##.. subsample match
sub_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/cat/'

id_size = 1  ## 0, 1

cat_lis = ['inner', 'middle', 'outer']

for mm in range( 3 ):

	s_dat = pds.read_csv( sub_path + 'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_member.csv' % cat_lis[mm],)

	bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
	p_ra, p_dec = np.array( s_dat['sat_ra'] ), np.array( s_dat['sat_dec'] )	

	pre_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

	for kk in range( 3 ):

		#. z-ref position
		if id_size == 0:
			dat = pds.read_csv( cat_path + 'frame-lim_Pm-cut_exlu-BCG_member_%s-band_small-cut_member_pos_z-ref.csv' % band[ kk ])

		if id_size == 1:
			dat = pds.read_csv( cat_path + 'frame-lim_Pm-cut_exlu-BCG_member_%s-band_wider-cut_member_pos_z-ref.csv' % band[ kk ])


		kk_ra, kk_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
		kk_imgx, kk_imgy = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

		kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

		idx, sep, d3d = pre_coord.match_to_catalog_sky( kk_coord )
		id_lim = sep.value < 2.7e-4

		mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]	
		mp_imgx, mp_imgy = kk_imgx[ idx[ id_lim ] ], kk_imgy[ idx[ id_lim ] ]

		keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_x', 'sat_y']
		values = [ bcg_ra, bcg_dec, bcg_z, p_ra, p_dec, mp_imgx, mp_imgy ]
		fill = dict( zip( keys, values ) )
		data = pds.DataFrame( fill )

		if id_size == 0:
			data.to_csv( cat_path + 'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_member_small-cut_%s-band_pos-zref.csv' % (cat_lis[mm], band[kk]),)

		if id_size == 1:
			data.to_csv( cat_path + 'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_member_wider-cut_%s-band_pos-zref.csv' % (cat_lis[mm], band[kk]),)

