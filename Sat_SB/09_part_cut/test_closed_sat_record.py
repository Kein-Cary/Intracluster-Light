"""
record the most closed galaxy and its size (~3 * R_kron from source detection)
PS : for comparison, 
the most closed ===> R_pix or R_phy~( given by redMaPPer)
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes

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

rad2arcsec = U.rad.to( U.arcsec )

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']



### === data load
"""
#.
home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

#.
img_file = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

#. source detection catalog
gal_file = home + 'photo_files/detect_source_cat/photo-z_img_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'


#. member galaxy
dat = pds.read_csv( home + 'member_files/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_exlu-BCG_member-cat.csv')

bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
sat_ra, sat_dec = np.array( dat['ra'] ), np.array( dat['dec'] )

R_sat = np.array( dat['R_cen'] )    #. Mpc / h
sR_sat = np.array( dat['Rcen/Rv'] )

clus_ID = np.array( dat['clus_ID'] )

#. cluster IDs
IDs, N_peat = sts.find_repeats( clus_ID )
IDs = IDs.astype( int )

Ns = len( IDs )


#.
tmp_bcg_ra, tmp_bcg_dec, tmp_bcg_z = np.array( [] ), np.array( [] ), np.array( [] )
tmp_c_IDs = np.array( [] )
tmp_xc, tmp_yc = np.array( [] ), np.array( [] )


#. min( R_pix )
tmp_xs_0, tmp_ys_0 = np.array( [] ), np.array( [] )
tmp_PAs_0 = np.array( [] )
tmp_sat_ra_0, tmp_sat_dec_0 = np.array( [] ), np.array( [] )

tmp_Rsat_0 = np.array( [] )
tmp_sR_sat_0 = np.array( [] )

tmp_sat_chi_0 = np.array( [] )
tmp_sat_a_0 = np.array( [] )
tmp_sat_b_0 = np.array( [] )


#. min( R_phy )
tmp_xs_1, tmp_ys_1 = np.array( [] ), np.array( [] )
tmp_PAs_1 = np.array( [] )
tmp_sat_ra_1, tmp_sat_dec_1 = np.array( [] ), np.array( [] )

tmp_Rsat_1 = np.array( [] )
tmp_sR_sat_1 = np.array( [] )

tmp_sat_chi_1 = np.array( [] )
tmp_sat_a_1 = np.array( [] )
tmp_sat_b_1 = np.array( [] )

#.
band_str = band[ rank ]

#.
for kk in range( Ns ):

	#.
	sub_ID = IDs[ kk ]

	id_vx = clus_ID == sub_ID

	sub_ra, sub_dec = sat_ra[ id_vx ], sat_dec[ id_vx ]
	sub_Rsat = R_sat[ id_vx ]
	sub_sR_sat = sR_sat[ id_vx ]

	ra_g, dec_g, z_g = bcg_ra[ id_vx ], bcg_dec[ id_vx ], bcg_z[ id_vx ]


	##. img file
	img_data = fits.open( img_file % (band_str, ra_g[0], dec_g[0], z_g[0]),)
	Head = img_data[0].header
	wcs_lis = awc.WCS( Head )

	x_cen, y_cen = wcs_lis.all_world2pix( ra_g[0], dec_g[0], 0 )
	x_cen, y_cen = x_cen.min(), y_cen.min()

	x_sat, y_sat = wcs_lis.all_world2pix( sub_ra, sub_dec, 0 )


	##. source detection cat.
	source = asc.read( gal_file % (band_str, ra_g[0], dec_g[0], z_g[0]),)
	det_A = np.array(source['A_IMAGE'])
	det_B = np.array(source['B_IMAGE'])
	det_chi = np.array(source['THETA_IMAGE'])

	det_sy = np.array(source['Y_IMAGE'])
	det_sx = np.array(source['X_IMAGE'])


	##. in units of rad
	sat_theta = np.arctan2( (y_sat - y_cen), (x_sat - x_cen) )


	##. 
	dR_pix = np.sqrt( (x_sat - x_cen)**2 + (y_sat - y_cen)**2 )
	id_ux = np.where( dR_pix == dR_pix.min() )[0][0]

	##. min( R_pix )
	tmp_xs_0 = np.r_[ tmp_xs_0, x_sat[ id_ux ] ]
	tmp_ys_0 = np.r_[ tmp_ys_0, y_sat[ id_ux ] ]
	tmp_PAs_0 = np.r_[ tmp_PAs_0, sat_theta[ id_ux ] ]

	tmp_sat_ra_0 = np.r_[ tmp_sat_ra_0, sub_ra[ id_ux ] ]
	tmp_sat_dec_0 = np.r_[ tmp_sat_dec_0, sub_dec[ id_ux ] ]

	tmp_Rsat_0 = np.r_[ tmp_Rsat_0, sub_Rsat[ id_ux ] ]
	tmp_sR_sat_0 = np.r_[ tmp_sR_sat_0, sub_sR_sat[ id_ux ] ]

	#.
	dR_0 = np.sqrt( (det_sx - x_sat[ id_ux ])**2 + (det_sy - y_sat[ id_ux ])**2 )
	id_px_0 = np.where( dR_0 == dR_0.min() )[0][0]

	tmp_sat_chi_0 = np.r_[ tmp_sat_chi_0, det_chi[ id_px_0 ] ]
	tmp_sat_a_0 = np.r_[ tmp_sat_a_0, det_A[ id_px_0 ] ]
	tmp_sat_b_0 = np.r_[ tmp_sat_b_0, det_B[ id_px_0 ] ]


	##. min( R_phy )
	id_nx = np.where( sub_Rsat == sub_Rsat.min() )[0][0]

	sub_Rsat = R_sat[ id_vx ]
	sub_sR_sat = sR_sat[ id_vx ]

	tmp_xs_1 = np.r_[ tmp_xs_1, x_sat[ id_nx ] ]
	tmp_ys_1 = np.r_[ tmp_ys_1, y_sat[ id_nx ] ]
	tmp_PAs_1 = np.r_[ tmp_PAs_1, sat_theta[ id_nx ] ]

	tmp_sat_ra_1 = np.r_[ tmp_sat_ra_1, sub_ra[ id_nx ] ]
	tmp_sat_dec_1 = np.r_[ tmp_sat_dec_1, sub_dec[ id_nx ] ]

	tmp_Rsat_1 = np.r_[ tmp_Rsat_1, sub_Rsat[ id_nx ] ]
	tmp_sR_sat_1 = np.r_[ tmp_sR_sat_1, sub_sR_sat[ id_nx ] ]

	#.
	dR_1 = np.sqrt( (det_sx - x_sat[ id_nx ])**2 + (det_sy - y_sat[ id_nx ])**2 )
	id_px_1 = np.where( dR_1 == dR_1.min() )[0][0]

	tmp_sat_chi_1 = np.r_[ tmp_sat_chi_1, det_chi[ id_px_1 ] ]
	tmp_sat_a_1 = np.r_[ tmp_sat_a_1, det_A[ id_px_1 ] ]
	tmp_sat_b_1 = np.r_[ tmp_sat_b_1, det_B[ id_px_1 ] ]


	##. mapped BCG
	tmp_xc = np.r_[ tmp_xc, x_cen ]
	tmp_yc = np.r_[ tmp_yc, y_cen ]

	tmp_c_IDs = np.r_[ tmp_c_IDs, sub_ID ]

	tmp_bcg_ra = np.r_[ tmp_bcg_ra, ra_g[0] ]
	tmp_bcg_dec = np.r_[ tmp_bcg_dec, dec_g[0] ]
	tmp_bcg_z = np.r_[ tmp_bcg_z, z_g[0] ]

##.
keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'bcg_x', 'bcg_y', 'clus_ID', 

		 'min_Rpix_sat_x', 'min_Rpix_sat_y', 'min_Rpix_sat_ra', 'min_Rpix_sat_dec', 
		 'min_Rpix_Rsat', 'min_Rpix_sR_sat', 'min_Rpix_PA2bcg',
		 'min_Rpix_sat_ar', 'min_Rpix_sat_br', 'min_Rpix_sat_chi',

		 'min_Rphy_sat_x', 'min_Rphy_sat_y', 'min_Rphy_sat_ra', 'min_Rphy_sat_dec', 
		 'min_Rphy_Rsat', 'min_Rphy_sR_sat', 'min_Rphy_PA2bcg', 
		 'min_Rphy_sat_ar', 'min_Rphy_sat_br', 'min_Rphy_sat_chi' ]

values = [ tmp_bcg_ra, tmp_bcg_dec, tmp_bcg_z, tmp_xc, tmp_yc, tmp_c_IDs, 

			tmp_xs_0, tmp_ys_0, tmp_sat_ra_0, tmp_sat_dec_0, 
			tmp_Rsat_0, tmp_sR_sat_0, tmp_PAs_0, 
			tmp_sat_a_0, tmp_sat_b_0, tmp_sat_chi_0, 

			tmp_xs_1, tmp_ys_1, tmp_sat_ra_1, tmp_sat_dec_1, 
			tmp_Rsat_1, tmp_sR_sat_1, tmp_PAs_1, 
			tmp_sat_a_1, tmp_sat_b_1, tmp_sat_chi_1 ]

fill = dict( zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/data/SDSS/member_files/BCG_part_cut_cat/' + 
				'Extend-BCGM_rgi-common_frame-limit_exlu-BCG_Sat_%s-band_closed-mem_position.csv' % band_str,)

raise
"""


### === ### information for image cut at z_ref
band_str = band[ rank ]

##. BCG Position Angle table
pat = pds.read_csv('/home/xkchen/data/SDSS/member_files/BCG_part_cut_cat/' + 'BCG_located-params_%s-band.csv' % band_str,)
all_ra, all_dec, all_z = np.array( pat['ra'] ), np.array( pat['dec'] ), np.array( pat['z'] )
all_IDs = np.array( pat['clus_ID'] )

#. -90 ~ 90
all_PA = np.array( pat['PA'] )
#. change to rad
all_PA = all_PA * np.pi / 180

coord_all = SkyCoord( ra = all_ra * U.deg, dec = all_dec * U.deg )


##.
dat = pds.read_csv('/home/xkchen/data/SDSS/member_files/BCG_part_cut_cat/' + 
			'Extend-BCGM_rgi-common_frame-limit_exlu-BCG_Sat_%s-band_closed-mem_position.csv' % band_str,)

bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )

bcg_x = np.array( dat['bcg_x'] )
bcg_y = np.array( dat['bcg_y'] )
clus_ID = np.array( dat['clus_ID'] )

coord_dat = SkyCoord( ra = bcg_ra * U.deg, dec = bcg_dec * U.deg )


##. sate ~ (min_Rpix_sat)
x_sat = np.array( dat['min_Rpix_sat_x'] )
y_sat = np.array( dat['min_Rpix_sat_y'] )

sat_ar = np.array( dat['min_Rpix_sat_ar'] )
sat_PA = np.array( dat['min_Rpix_sat_chi'] )

##.
Da_z = Test_model.angular_diameter_distance( bcg_z ).value
Da_ref = Test_model.angular_diameter_distance( z_ref ).value

L_ref = Da_ref * pixel / rad2arcsec
L_z = Da_z * pixel / rad2arcsec
eta = L_ref / L_z

ref_sx = x_sat / eta
ref_sy = y_sat / eta

ref_bcg_x = bcg_x / eta
ref_bcg_y = bcg_y / eta

R_fact = 2.
ref_sat_ar = R_fact * sat_ar / eta


##.
idx, d2d, d3d = coord_dat.match_to_catalog_sky( coord_all )
id_lim = d2d.value < 2.7e-4

mp_bcg_PA = all_PA[ idx[ id_lim ] ] 


##.
keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'bcg_x', 'bcg_y', 'bcg_PA', 'sat_x', 'sat_y', 'sat_PA', 'sat_ar']
values = [ bcg_ra, bcg_dec, bcg_z, ref_bcg_x, ref_bcg_y, mp_bcg_PA, ref_sx, ref_sy, sat_PA, ref_sat_ar ]
fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/data/SDSS/member_files/BCG_part_cut_cat/' + 'BCG_%s-band_part-cut_zref-pos.csv' % band_str,)

