import matplotlib as mpl
import matplotlib.pyplot as plt

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

from img_sat_resamp import resamp_func
from img_sat_resamp import BG_resamp_func
from img_sat_BG_extract_tmp import origin_img_cut_func

from img_sat_fast_stack import sat_img_fast_stack_func
from img_sat_fast_stack import sat_BG_fast_stack_func

#.
import time
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
rad2arcsec = U.rad.to( U.arcsec )



### === ### data load
cat_path = '/home/xkchen/data/SDSS/member_files/redMap_contral_galx/control_cat/'
out_path = '/home/xkchen/data/SDSS/member_files/redMap_contral_galx/BG_shufl_cat/'

N_shufl = 20

for kk in range( 3 ):

	band_str = band[ kk ]

	##.
	dat = pds.read_csv( cat_path + 'random_field-galx_map_%s-band_cat.csv' % band_str )

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec, sat_z = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] ), np.array( dat['sat_z'] )

	sat_x, sat_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

	N_s = len( sat_ra )

	for dd in range( N_shufl ):

		##. 
		rand_dex = np.random.choice( N_s, N_s, replace = False )

		cp_ra, cp_dec, cp_z = bcg_ra[ rand_dex ], bcg_dec[ rand_dex ], bcg_z[ rand_dex ]

		cp_gx = 2048 - sat_x
		cp_gy = 1489 - sat_y


		##.
		keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_z', 'orin_x', 'orin_y',
				'shfl_bcg_ra', 'shfl_bcg_dec', 'shfl_bcg_z', 'shfl_x', 'shfl_y' ]
		values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, sat_z, sat_x, sat_y, 
				cp_ra, cp_dec, cp_z, cp_gx, cp_gy ]

		fill = dict( zip( keys, values ) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_path + 
						'random_field-galx_map_%s-band_sat-shuffle-%d_position.csv' % (band_str, dd),)

