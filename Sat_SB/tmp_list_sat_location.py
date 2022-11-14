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


### === satellite peak_xy and centric_xy compare (relative position in the origin SDSS image frame)
home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

band_str = band[ rank ]

img_file = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

dat = pds.read_csv( home + 'member_files/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_exlu-BCG_member-cat.csv')

bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
sat_ra, sat_dec = np.array( dat['ra'] ), np.array( dat['dec'] )

R_sat = np.array( dat['R_cen'] )
clus_ID = np.array( dat['clus_ID'] )

#. cluster IDs
IDs, N_peat = sts.find_repeats( clus_ID )

IDs = IDs.astype( int )

Ns = len( IDs )


tmp_bcg_ra, tmp_bcg_dec, tmp_bcg_z = np.array( [] ), np.array( [] ), np.array( [] )
tmp_c_IDs = np.array( [] )

tmp_PAs = np.array( [] )
tmp_xc, tmp_yc = np.array( [] ), np.array( [] )
tmp_xs, tmp_ys = np.array( [] ), np.array( [] )
tmp_sat_ra, tmp_sat_dec = np.array( [] ), np.array( [] )


for kk in range( Ns ):

	sub_ID = IDs[ kk ]

	id_vx = clus_ID == sub_ID

	sub_ra, sub_dec = sat_ra[ id_vx ], sat_dec[ id_vx ]
	ra_g, dec_g, z_g = bcg_ra[ id_vx ], bcg_dec[ id_vx ], bcg_z[ id_vx ]

	#.img file

	img_data = fits.open( img_file % (band_str, ra_g[0], dec_g[0], z_g[0]),)
	Head = img_data[0].header
	wcs_lis = awc.WCS( Head )

	x_cen, y_cen = wcs_lis.all_world2pix( ra_g[0], dec_g[0], 0 )
	x_sat, y_sat = wcs_lis.all_world2pix( sub_ra, sub_dec, 0 )

	sat_theta = np.arctan2( (y_sat - y_cen), (x_sat - x_cen) )  ## in units of rad

	tmp_xs = np.r_[ tmp_xs, x_sat ]
	tmp_ys = np.r_[ tmp_ys, y_sat ]

	tmp_xc = np.r_[ tmp_xc, np.ones( np.sum(id_vx), ) * x_cen ]
	tmp_yc = np.r_[ tmp_yc, np.ones( np.sum(id_vx), ) * y_cen ]
	tmp_c_IDs = np.r_[ tmp_c_IDs, np.ones( np.sum(id_vx), ) * sub_ID ]

	tmp_PAs = np.r_[ tmp_PAs, sat_theta ]

	tmp_bcg_ra = np.r_[ tmp_bcg_ra, ra_g ]
	tmp_bcg_dec = np.r_[ tmp_bcg_dec, dec_g ]
	tmp_bcg_z = np.r_[ tmp_bcg_z, z_g ]

	tmp_sat_ra = np.r_[ tmp_sat_ra, sub_ra ]
	tmp_sat_dec = np.r_[ tmp_sat_dec, sub_dec ]

#. save
keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'bcg_x', 'bcg_y', 'sat_x', 'sat_y', 'sat_PA2bcg', 'clus_ID' ]
values = [ tmp_bcg_ra, tmp_bcg_dec, tmp_bcg_z, tmp_sat_ra, tmp_sat_dec, tmp_xc, tmp_yc, tmp_xs, tmp_ys, tmp_PAs, tmp_c_IDs ]
fill = dict( zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/Extend-BCGM_rgi-common_frame-limit_exlu-BCG_Sat_%s-band_origin-img_position.csv' % band_str,)


raise


### === satellite peak_xy and centric_xy compare (based on the cutout satellite images, before pixel resampling)
img_file = home + 'member_files/mask_imgs/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img.fits'

band_str = band[ rank ]

dat = pds.read_csv(home + 'member_files/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-limit_member-cat.csv')
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
out_data.to_csv( '/home/xkchen/Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos-compare.csv' % band_str,)

