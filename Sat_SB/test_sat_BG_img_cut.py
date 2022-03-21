import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import h5py
import numpy as np
import pandas as pds

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc

from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy.stats import binned_statistic as binned
import scipy.interpolate as interp

from light_measure import light_measure_weit
from img_sat_BG_extract import BG_build_func
from img_sat_BG_extract import sat_BG_extract_func


from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


##### cosmology model
Test_model = apcy.Planck15.clone( H0 = 67.74, Om0 = 0.311 )
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']


#*********************************#
### === ### read resampled satellite images and record the image size (use to background image cut)
"""
home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

img_file = home + 'member_files/resamp_imgs/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp-img.fits'
obs_file = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

##... position catalog build

cat_lis = cat_lis = ['inner-mem', 'outer-mem']

for ll in range( 2 ):

	dat = pds.read_csv( home + 'member_files/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_%s_cat.csv' % cat_lis[ll],)
	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

	R_sat = np.array( dat['R_sat'] )   # Mpc / h
	R_sat = R_sat * 1e3 / h   # kpc

	N_ss = len( bcg_ra )

	m, n = divmod( N_ss, cpus )
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	sub_ra, sub_dec, sub_z = bcg_ra[N_sub0 : N_sub1], bcg_dec[N_sub0 : N_sub1], bcg_z[N_sub0 : N_sub1]
	ra_set, dec_set = sat_ra[N_sub0 : N_sub1], sat_dec[N_sub0 : N_sub1]	
	R_set = R_sat[N_sub0 : N_sub1]

	N_sub = len( sub_ra )

	for kk in range( 3 ):

		tmp_cut_Nx, tmp_cut_Ny = np.array([]), np.array([])
		tmp_lx, tmp_rx = np.array([]), np.array([])
		tmp_ly, tmp_uy = np.array([]), np.array([])
		tmp_PA = np.array([])

		band_str = band[ kk ]

		for pp in range( N_sub ):

			ra_g, dec_g, z_g = sub_ra[pp], sub_dec[pp], sub_z[pp]
			_kk_ra, _kk_dec = ra_set[ pp ], dec_set[pp]


			#. record position relative to BCG
			img_arry = fits.open( obs_file % (band_str, ra_g, dec_g, z_g),)
			Head = img_arry[0].header
			_kk_img = img_arry[0].data

			wcs_lis = awc.WCS( Head )

			cen_x, cen_y = wcs_lis.all_world2pix( ra_g, dec_g, 0 )
			sx, sy = wcs_lis.all_world2pix( _kk_ra, _kk_dec, 0 )

			sat_theta = np.arctan2( (sy - cen_y), (sx - cen_x) )  ## in units of rad


			#. record the cut-image information
			img_data = fits.open( img_file % (band_str, ra_g, dec_g, z_g, _kk_ra, _kk_dec), )
			img_arr = img_data[0].data

			xn, yn = np.int( img_arr.shape[1] / 2 ), np.int( img_arr.shape[0] / 2 )

			la0, la1 = xn, img_arr.shape[1] - xn
			lb0, lb1 = yn, img_arr.shape[0] - yn

			tmp_cut_Nx = np.r_[ tmp_cut_Nx, img_arr.shape[1] ]
			tmp_cut_Ny = np.r_[ tmp_cut_Ny, img_arr.shape[0] ]

			tmp_lx = np.r_[ tmp_lx, la0 ]
			tmp_rx = np.r_[ tmp_rx, la1 ]

			tmp_ly = np.r_[ tmp_ly, lb0 ]
			tmp_uy = np.r_[ tmp_uy, lb1 ]

			tmp_PA = np.r_[ tmp_PA, sat_theta ]

		#.
		keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'R_sat', 'sat_PA', 
				'cut_Nx', 'cut_Ny', 'low_dx', 'up_dx', 'low_dy', 'up_dy']
		values = [ sub_ra, sub_dec, sub_z, ra_set, dec_set, R_set, tmp_PA, 
					tmp_cut_Nx, tmp_cut_Ny, tmp_lx, tmp_rx, tmp_ly, tmp_uy ]

		fill = dict( zip( keys, values ) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv('/home/xkchen/project/tmp_obj_cat/' + 
				'Extend-BCGM_rgi-common_frame-lim_Pm-cut_%s_%s-band_BG-tract_cat_%d.csv' % (cat_lis[ll], band_str, rank),)

commd.Barrier()


if rank == 0:

	# N_pbs = cpus
	N_pbs = 280

	for ll in range( 2 ):

		for kk in range( 3 ):

			band_str = band[kk]

			dat = pds.read_csv('/home/xkchen/project/tmp_obj_cat/' + 
				'Extend-BCGM_rgi-common_frame-lim_Pm-cut_%s_%s-band_BG-tract_cat_0.csv' % (cat_lis[ll], band_str),)

			keys = dat.columns[1:]
			Nks = len( keys )

			tmp_arr = []

			for tt in range( Nks ):

				sub_arr = np.array( dat['%s' % keys[ tt ] ] )
				tmp_arr.append( sub_arr )

			for pp in range( 1, N_pbs ):
				dat = pds.read_csv('/home/xkchen/project/tmp_obj_cat/' + 
					'Extend-BCGM_rgi-common_frame-lim_Pm-cut_%s_%s-band_BG-tract_cat_%d.csv' % (cat_lis[ll], band_str, pp),)

				for tt in range( Nks ):

					sub_arr = np.array( dat['%s' % keys[ tt ] ] )
					tmp_arr[ tt ] = np.r_[ tmp_arr[ tt ], sub_arr ]

			#.
			fill = dict( zip( keys, tmp_arr ) )
			out_data = pds.DataFrame( fill )
			out_data.to_csv('/home/xkchen/' + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_%s_%s-band_BG-tract_cat.csv' % (cat_lis[ll], band_str),)
			# out_data.to_csv('/home/xkchen/data/SDSS/member_files/BG_tract_cat/' + 
			# 				'Extend-BCGM_rgi-common_frame-lim_Pm-cut_%s_%s-band_BG-tract_cat.csv' % (cat_lis[ll], band_str),)

print('Done!')
"""


#*********************************#
### === BG_img cutout (based on mocked 2D image of BCG + ICL)
"""
load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'

cat_lis = ['inner-mem', 'outer-mem']

for mm in range( 2 ):

	for kk in range( 3 ):

		band_str = band[ kk ]

		dat = pds.read_csv( home + 'member_files/BG_tract_cat/'
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_%s_%s-band_BG-tract_cat.csv' % (cat_lis[mm], band_str),)

		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

		R_sat = np.array( dat['R_sat'] )   ## in units of kpc
		sat_PA = np.array( dat['sat_PA'] )  ## position angle (relative to BCG)

		lim_dx0, lim_dx1 = np.array( dat['low_dx'] ), np.array( dat['up_dx'] )
		lim_dy0, lim_dy1 = np.array( dat['low_dy'] ), np.array( dat['up_dy'] )
		cut_Nx, cut_Ny = np.array( dat['cut_Nx'] ), np.array( dat['cut_Ny'] )


		N_ss = len( sat_ra )

		m, n = divmod( N_ss, cpus )
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		sub_ra, sub_dec, sub_z = bcg_ra[N_sub0 : N_sub1], bcg_dec[N_sub0 : N_sub1], bcg_z[N_sub0 : N_sub1]
		ra_set, dec_set = sat_ra[N_sub0 : N_sub1], sat_dec[N_sub0 : N_sub1]

		R_set = R_sat[N_sub0 : N_sub1]
		chi_set = sat_PA[N_sub0 : N_sub1]

		set_dx0, set_dx1 = lim_dx0[N_sub0 : N_sub1], lim_dx1[N_sub0 : N_sub1]
		set_dy0, set_dy1 = lim_dy0[N_sub0 : N_sub1], lim_dy1[N_sub0 : N_sub1]
		set_Nx, set_Ny = cut_Nx[N_sub0 : N_sub1], cut_Ny[N_sub0 : N_sub1]

		N_sub = len( sub_ra )

		for pp in range( N_sub ):

			ra_g, dec_g, z_g = sub_ra[ pp ], sub_dec[ pp ], sub_z[ pp ]
			_kk_ra, _kk_dec = ra_set[ pp ], dec_set[ pp ]

			_kk_Rs = R_set[ pp ]
			_kk_phi = chi_set[ pp ]

			_kk_dx0, _kk_dx1 = set_dx0[ pp ], set_dx1[ pp ]
			_kk_dy0, _kk_dy1 = set_dy0[ pp ], set_dy1[ pp ]
			_kk_Nx, _kk_Ny = set_Nx[ pp ], set_Ny[ pp ]

			BG_file = home + 'member_files/BG_2D_file/%s_%s-band_BG_img.fits' % (cat_lis[mm], band_str)
			out_file = home + 'member_files/BG_imgs/clus_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_BG.fits'

			sat_BG_extract_func( ra_g, dec_g, z_g, _kk_ra, _kk_dec, _kk_Rs, _kk_phi, band_str, z_ref, 
								_kk_dx0, _kk_dx1, _kk_dy0, _kk_dy1, pixel, BG_file, out_file )

print('rank = %d' % rank)

"""


#*********************************# fixed i_Mag_10 case
### === satellite location at z_ref mapping
# load = '/home/xkchen/fig_tmp/'
# home = '/home/xkchen/data/SDSS/'

# cat_lis = ['inner', 'middle', 'outer']

"""
for kk in range( 3 ):

	band_str = band[ kk ]

	##. load the tract information
	dat = pds.read_csv( home + 'member_files/BG_tract_cat/'
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_inner-mem_%s-band_BG-tract_cat.csv' % band_str,)

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

	R_sat = np.array( dat['R_sat'] )   ## in units of kpc
	sat_PA = np.array( dat['sat_PA'] )  ## position angle (relative to BCG)

	lim_dx0, lim_dx1 = np.array( dat['low_dx'] ), np.array( dat['up_dx'] )
	lim_dy0, lim_dy1 = np.array( dat['low_dy'] ), np.array( dat['up_dy'] )
	cut_Nx, cut_Ny = np.array( dat['cut_Nx'] ), np.array( dat['cut_Ny'] )

	cat = pds.read_csv( home + 'member_files/sat_cat_z02_03/' + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_inner-mem_cat.csv')
	clus_ID = np.array( cat['clus_ID'] )


	dat = pds.read_csv( home + 'member_files/BG_tract_cat/'
		'Extend-BCGM_rgi-common_frame-lim_Pm-cut_outer-mem_%s-band_BG-tract_cat.csv' % band_str,)

	bcg_ra = np.r_[ bcg_ra, np.array( dat['bcg_ra'] ) ]
	bcg_dec = np.r_[ bcg_dec, np.array( dat['bcg_dec'] ) ]
	bcg_z = np.r_[ bcg_z, np.array( dat['bcg_z'] ) ]

	sat_ra = np.r_[ sat_ra, np.array( dat['sat_ra'] ) ]
	sat_dec = np.r_[ sat_dec, np.array( dat['sat_dec'] ) ]

	R_sat = np.r_[ R_sat, np.array( dat['R_sat'] ) ]
	sat_PA = np.r_[ sat_PA, np.array( dat['sat_PA'] ) ]

	lim_dx0 = np.r_[ lim_dx0, np.array( dat['low_dx'] ) ]
	lim_dx1 = np.r_[ lim_dx1, np.array( dat['up_dx'] ) ]

	lim_dy0 = np.r_[ lim_dy0, np.array( dat['low_dy'] ) ]
	lim_dy1 = np.r_[ lim_dy1, np.array( dat['up_dy'] ) ]

	cut_Nx = np.r_[ cut_Nx, np.array( dat['cut_Nx'] ) ]
	cut_Ny = np.r_[ cut_Ny, np.array( dat['cut_Ny'] ) ]

	cat = pds.read_csv( home + 'member_files/sat_cat_z02_03/' + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_outer-mem_cat.csv')
	clus_ID = np.r_[ clus_ID, np.array( cat['clus_ID'] ) ]


	#. mock_2D_BG tract catalog
	keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'R_sat', 'sat_PA', 
						'cut_Nx', 'cut_Ny', 'low_dx', 'up_dx', 'low_dy', 'up_dy', 'clus_ID' ]
	values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, R_sat, sat_PA, 
						cut_Nx, cut_Ny, lim_dx0, lim_dx1, lim_dy0, lim_dy1, clus_ID ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( home + 'member_files/BG_tract_cat/' + 
				'Extend-BCGM_rgi-common_frame-lim_Pm-cut_%s-band_BG-tract_cat.csv' % band_str,)


for kk in range( 3 ):

	band_str = band[ kk ]

	#. map table
	dat = pds.read_csv( home + 'member_files/BG_tract_cat/' + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_%s-band_BG-tract_cat.csv' % band_str,)

	kk_bcg_ra, kk_bcg_dec, kk_bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	kk_s_ra, kk_s_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
	kk_clus_ID = np.array( dat['clus_ID'] )

	kk_R_sat = np.array( dat['R_sat'] )
	kk_sat_PA = np.array( dat['sat_PA'] )

	kk_lim_dx0 = np.array( dat['low_dx'] )
	kk_lim_dx1 = np.array( dat['up_dx'] )

	kk_lim_dy0 = np.array( dat['low_dy'] )
	kk_lim_dy1 = np.array( dat['up_dy'] )

	kk_cut_Nx = np.array( dat['cut_Nx'] )
	kk_cut_Ny = np.array( dat['cut_Ny'] )


	for mm in range( 3 ):

		#. subsample need to match
		s_dat = pds.read_csv( home + 'member_files/sat_cat_z02_03/' + 
					'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_member.csv' % cat_lis[ mm ],)

		bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
		sat_ra, sat_dec = np.array( s_dat['sat_ra'] ), np.array( s_dat['sat_dec'] )
		clus_IDs = np.array( s_dat['clus_ID'] )

		print( len(bcg_ra) )

		set_IDs = np.array( list( set(clus_IDs) ) )
		set_IDs = set_IDs.astype( int )
		N_cs = len( set_IDs )


		#. image tract information
		mp_bcg_ra, mp_bcg_dec, mp_bcg_z = np.array([]), np.array([]), np.array([])
		mp_sat_ra, mp_sat_dec = np.array([]), np.array([])

		mp_Rsat, mp_PA = np.array([]), np.array([])
		mp_lim_x0, mp_lim_x1 = np.array([]), np.array([])
		mp_lim_y0, mp_lim_y1 = np.array([]), np.array([])
		mp_cut_Nx, mp_cut_Ny = np.array([]), np.array([])


		for tt in range( N_cs ):

			id_x = clus_IDs == set_IDs[ tt ]

			tt_ra, tt_dec = sat_ra[ id_x ], sat_dec[ id_x ]
			tt_cen_ra, tt_cen_dec, tt_cen_z = bcg_ra[ id_x ][0], bcg_dec[ id_x ][0], bcg_z[ id_x ][0]

			sub_coord = SkyCoord( ra = tt_ra * U.deg, dec = tt_dec * U.deg)


			id_y = kk_clus_ID == set_IDs[ tt ]

			cp_ra, cp_dec = kk_s_ra[ id_y ], kk_s_dec[ id_y ]

			cp_R_sat, cp_sat_PA = kk_R_sat[ id_y ], kk_sat_PA[ id_y ]
			cp_lim_x0, cp_lim_x1 = kk_lim_dx0[ id_y ], kk_lim_dx1[ id_y ]
			cp_lim_y0, cp_lim_y1 = kk_lim_dy0[ id_y ], kk_lim_dy1[ id_y ]
			cp_cut_Nx, cp_cut_Ny = kk_cut_Nx[ id_y ], kk_cut_Ny[ id_y ]

			kk_coord = SkyCoord( ra = cp_ra * U.deg, dec = cp_dec * U.deg )


			#. map
			idx, sep, d3d = kk_coord.match_to_catalog_sky( sub_coord )
			id_lim = sep.value < 2.7e-4

			_mm_ra, _mm_dec = cp_ra[ id_lim ], cp_dec[ id_lim ]  ## for mapping check
			_N_sat = len( _mm_ra )


			mp_bcg_ra = np.r_[ mp_bcg_ra, np.ones( _N_sat,) * tt_cen_ra ]
			mp_bcg_dec = np.r_[ mp_bcg_dec, np.ones( _N_sat,) * tt_cen_dec ]
			mp_bcg_z = np.r_[ mp_bcg_z, np.ones( _N_sat,) * tt_cen_z ]
			
			mp_sat_ra = np.r_[ mp_sat_ra, tt_ra[ idx[ id_lim ] ] ]
			mp_sat_dec = np.r_[ mp_sat_dec, tt_dec[ idx[ id_lim ] ] ]

			mp_Rsat = np.r_[ mp_Rsat, cp_R_sat[ id_lim ] ]
			mp_PA = np.r_[ mp_PA, cp_sat_PA[ id_lim ] ]

			mp_lim_x0 = np.r_[ mp_lim_x0, cp_lim_x0[ id_lim ] ]
			mp_lim_x1 = np.r_[ mp_lim_x1, cp_lim_x1[ id_lim ] ]

			mp_lim_y0 = np.r_[ mp_lim_y0, cp_lim_y0[ id_lim ] ]
			mp_lim_y1 = np.r_[ mp_lim_y1, cp_lim_y1[ id_lim ] ]

			mp_cut_Nx = np.r_[ mp_cut_Nx, cp_cut_Nx[ id_lim ] ]
			mp_cut_Ny = np.r_[ mp_cut_Ny, cp_cut_Ny[ id_lim ] ]

		print( len( mp_PA ) )
		##. save catalog
		keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'R_sat', 'sat_PA', 
							'cut_Nx', 'cut_Ny', 'low_dx', 'up_dx', 'low_dy', 'up_dy' ]
		values = [ mp_bcg_ra, mp_bcg_dec, mp_bcg_z, mp_sat_ra, mp_sat_dec, mp_Rsat, mp_PA, 
							mp_cut_Nx, mp_cut_Ny, mp_lim_x0, mp_lim_x1, mp_lim_y0, mp_lim_y1 ]

		fill = dict( zip( keys, values ) )
		data = pds.DataFrame( fill )
		data.to_csv( home + 'member_files/BG_tract_cat/' + 
					'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_member_%s-band_BG-tract_cat.csv' % (cat_lis[mm], band_str),)

"""


### === BG_img cutout (based on mocked 2D image of BCG+ICL)
"""
load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'
cat_lis = ['inner', 'middle', 'outer']

for kk in range( 3 ):

	band_str = band[ kk ]

	for mm in range( 3 ):

		dat = pds.read_csv( home + 'member_files/BG_tract_cat/' + 
				'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_member_%s-band_BG-tract_cat.csv' % (cat_lis[mm], band_str),)

		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

		R_sat = np.array( dat['R_sat'] )   ## in units of kpc
		sat_PA = np.array( dat['sat_PA'] )  ## position angle (relative to BCG)

		lim_dx0, lim_dx1 = np.array( dat['low_dx'] ), np.array( dat['up_dx'] )
		lim_dy0, lim_dy1 = np.array( dat['low_dy'] ), np.array( dat['up_dy'] )
		cut_Nx, cut_Ny = np.array( dat['cut_Nx'] ), np.array( dat['cut_Ny'] )


		#. BG_img cutout
		N_ss = len( sat_ra )

		m, n = divmod( N_ss, cpus )
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		sub_ra, sub_dec, sub_z = bcg_ra[N_sub0 : N_sub1], bcg_dec[N_sub0 : N_sub1], bcg_z[N_sub0 : N_sub1]
		ra_set, dec_set = sat_ra[N_sub0 : N_sub1], sat_dec[N_sub0 : N_sub1]

		R_set = R_sat[N_sub0 : N_sub1]
		chi_set = sat_PA[N_sub0 : N_sub1]

		set_dx0, set_dx1 = lim_dx0[N_sub0 : N_sub1], lim_dx1[N_sub0 : N_sub1]
		set_dy0, set_dy1 = lim_dy0[N_sub0 : N_sub1], lim_dy1[N_sub0 : N_sub1]
		set_Nx, set_Ny = cut_Nx[N_sub0 : N_sub1], cut_Ny[N_sub0 : N_sub1]


		N_sub = len( sub_ra )

		for pp in range( N_sub ):

			ra_g, dec_g, z_g = sub_ra[ pp ], sub_dec[ pp ], sub_z[ pp ]
			_kk_ra, _kk_dec = ra_set[ pp ], dec_set[ pp ]

			_kk_Rs = R_set[ pp ]
			_kk_phi = chi_set[ pp ]

			_kk_dx0, _kk_dx1 = set_dx0[ pp ], set_dx1[ pp ]
			_kk_dy0, _kk_dy1 = set_dy0[ pp ], set_dy1[ pp ]
			_kk_Nx, _kk_Ny = set_Nx[ pp ], set_Ny[ pp ]

			BG_file = home + 'member_files/BG_2D_file/fix_iMag-10_%s_%s-band_BG_img.fits' % (cat_lis[mm], band_str)
			out_file = home + 'member_files/BG_imgs/clus_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_BG.fits'

			sat_BG_extract_func( ra_g, dec_g, z_g, _kk_ra, _kk_dec, _kk_Rs, _kk_phi, band_str, z_ref, 
								_kk_dx0, _kk_dx1, _kk_dy0, _kk_dy1, pixel, BG_file, out_file )

print('rank = %d' % rank)

"""


### === BG_img cutout (based on stacked cluster image, which is BCG+ICL)
"""
load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'

cat_lis = ['inner', 'middle', 'outer']

for kk in range( 3 ):

	band_str = band[ kk ]

	for mm in range( 3 ):

		dat = pds.read_csv( home + 'member_files/BG_tract_cat/' + 
				'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_member_%s-band_BG-tract_cat.csv' % (cat_lis[mm], band_str),)

		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

		R_sat = np.array( dat['R_sat'] )   ## in units of kpc
		sat_PA = np.array( dat['sat_PA'] )  ## position angle (relative to BCG)

		lim_dx0, lim_dx1 = np.array( dat['low_dx'] ), np.array( dat['up_dx'] )
		lim_dy0, lim_dy1 = np.array( dat['low_dy'] ), np.array( dat['up_dy'] )
		cut_Nx, cut_Ny = np.array( dat['cut_Nx'] ), np.array( dat['cut_Ny'] )


		#. BG_img cutout
		N_ss = len( sat_ra )

		m, n = divmod( N_ss, cpus )
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		sub_ra, sub_dec, sub_z = bcg_ra[N_sub0 : N_sub1], bcg_dec[N_sub0 : N_sub1], bcg_z[N_sub0 : N_sub1]
		ra_set, dec_set = sat_ra[N_sub0 : N_sub1], sat_dec[N_sub0 : N_sub1]

		R_set = R_sat[N_sub0 : N_sub1]
		chi_set = sat_PA[N_sub0 : N_sub1]

		set_dx0, set_dx1 = lim_dx0[N_sub0 : N_sub1], lim_dx1[N_sub0 : N_sub1]
		set_dy0, set_dy1 = lim_dy0[N_sub0 : N_sub1], lim_dy1[N_sub0 : N_sub1]
		set_Nx, set_Ny = cut_Nx[N_sub0 : N_sub1], cut_Ny[N_sub0 : N_sub1]


		N_sub = len( sub_ra )

		for pp in range( N_sub ):

			ra_g, dec_g, z_g = sub_ra[ pp ], sub_dec[ pp ], sub_z[ pp ]
			_kk_ra, _kk_dec = ra_set[ pp ], dec_set[ pp ]

			_kk_Rs = R_set[ pp ]
			_kk_phi = chi_set[ pp ]

			_kk_dx0, _kk_dx1 = set_dx0[ pp ], set_dx1[ pp ]
			_kk_dy0, _kk_dy1 = set_dy0[ pp ], set_dy1[ pp ]
			_kk_Nx, _kk_Ny = set_Nx[ pp ], set_Ny[ pp ]

			BG_file = home + 'member_files/BG_2D_file/stacked_cluster_%s_%s-band_img.fits' % (cat_lis[mm], band_str)  ##. stacked image with Ng weight
			# BG_file = home + 'member_files/BG_2D_file/stacked_all_cluster_%s-band_img.fits' % band_str              ##. stacked image without Ng weight

			out_file = home + 'member_files/BG_imgs_nomock/clus_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_BG.fits'

			sat_BG_extract_func( ra_g, dec_g, z_g, _kk_ra, _kk_dec, _kk_Rs, _kk_phi, band_str, z_ref, 
								_kk_dx0, _kk_dx1, _kk_dy0, _kk_dy1, pixel, BG_file, out_file = out_file)

print('rank = %d' % rank)

"""

#*********************************#
### === shuffle the order of BCG and images, random select satellite images
from img_sat_BG_extract import origin_img_cut_func
from img_sat_resamp import resamp_func

##... tacking the stacking image of this cutout as background
load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'


# #. BCG position file
# post_file = home + 'member_files/BG_tract_cat/Extend-BCGM_rgi-common_frame-limit_Pm-cut_exlu-BCG_Sat_%s-band_origin-img_position.csv'
# img_file = home + 'photo_files/pos_offset_correct_imgs/mask_img/photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'


# #. target cluster (want to know background of satellites)
# dat = pds.read_csv( home + 'member_files/sat_cat_z02_03/' + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv')

# targ_IDs = np.array( dat['clus_ID'] )

# set_IDs = np.array( list( set( targ_IDs ) ) )
# set_IDs = set_IDs.astype( int )

# N_cc = len( set_IDs )


# #. shuffle cluster IDs
# R_cut = 320
# out_file = home + 'member_files/shuffl_cut_img/mask_img/clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_img.fits'

# for kk in range( 1,3 ):

# 	band_str = band[ kk ]

# 	# rand_IDs = np.loadtxt( home + 'member_files/BG_tract_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_mem_shuffle-clus_cat.txt')
# 	rand_IDs = np.loadtxt( home + 'member_files/BG_tract_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_mem_%s-band_extra-shuffle-clus_cat.txt' % band_str)
# 	rand_mp_IDs = rand_IDs.astype( int )

# 	#. shuffle cutout images
# 	m, n = divmod( N_cc, cpus )
# 	N_sub0, N_sub1 = m * rank, (rank + 1) * m
# 	if rank == cpus - 1:
# 		N_sub1 += n

# 	sub_clusID = set_IDs[N_sub0 : N_sub1]
# 	sub_rand_mp_ID = rand_mp_IDs[N_sub0 : N_sub1]

# 	origin_img_cut_func( post_file, img_file, band_str, sub_clusID, sub_rand_mp_ID, R_cut, pixel, out_file)

# print('%d-rank, cut Done!' % rank)


#. resampling... 
for kk in range( 1,3 ):

	band_str = band[ kk ]

	dat = pds.read_csv( load + 'Extend_Mbcg_sat_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_member_%s-band_pos.csv' % band_str )
	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
	sat_cx, sat_cy = np.array( dat['cut_cx'] ), np.array( dat['cut_cy'] )


	_Ns_ = len( sat_ra )

	m, n = divmod( _Ns_, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	sub_ra, sub_dec, sub_z = bcg_ra[N_sub0 : N_sub1], bcg_dec[N_sub0 : N_sub1], bcg_z[N_sub0 : N_sub1]
	ra_set, dec_set = sat_ra[N_sub0 : N_sub1], sat_dec[N_sub0 : N_sub1]
	img_x, img_y = sat_cx[N_sub0 : N_sub1], sat_cy[N_sub0 : N_sub1]


	id_dimm = True
	d_file = home + 'member_files/shuffl_cut_img/mask_img/clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_img.fits'
	out_file = home + 'member_files/shuffl_cut_img/resamp_img/clus_shufl-tract_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_resamp.fits'

	resamp_func( d_file, sub_z, sub_ra, sub_dec, ra_set, dec_set, img_x, img_y, band_str, out_file, z_ref, id_dimm = id_dimm )

print( '%d rank, done!' % rank )
raise
