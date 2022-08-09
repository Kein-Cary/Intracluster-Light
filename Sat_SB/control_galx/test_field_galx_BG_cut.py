from math import isfinite
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pds
import h5py

import mechanize
from io import StringIO
import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable

#.
from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

#.
from img_sat_BG_extract import contrl_galx_BGcut_func
from img_sat_resamp import contrl_galx_resamp_func


###... cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)

band = ['r', 'g', 'i']
pixel = 0.396
z_ref = 0.25


### === use the masked random image for test
##.
home = '/home/xkchen/data/SDSS/'

R_cut = 320

# for kk in range( 3 ):
for kk in range( 1 ):  ##. r-band for test

	band_str = band[ kk ]

	##.
	# keys : ,bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, sat_z, ori_imgx, ori_imgy, cut_cx, cut_cy
	dat = pds.read_csv( '/home/xkchen/fig_tmp/Extend_Mbcg_ctrlGalx_cat/' + 
						'redMap_map_control_galx_%s-band_cut_pos.csv' % band_str,)

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
	sat_cx, sat_cy = np.array( dat['ori_imgx'] ), np.array( dat['ori_imgy'] )

	sat_z = np.array( dat['sat_z'] )

	##.
	_Ns_ = len( sat_ra )

	m, n = divmod( _Ns_, cpus)

	#.
	for mm in range( rank, rank + 1):

		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		sub_ra, sub_dec = bcg_ra[N_sub0 : N_sub1], bcg_dec[N_sub0 : N_sub1]

		ra_set, dec_set = sat_ra[N_sub0 : N_sub1], sat_dec[N_sub0 : N_sub1]

		img_x, img_y = sat_cx[N_sub0 : N_sub1], sat_cy[N_sub0 : N_sub1]

		sub_z = bcg_z[N_sub0 : N_sub1]

		z_set = sat_z[N_sub0 : N_sub1]

		N_pp = len( sub_ra )

		#.
		for pp in range( N_pp ):

			p_ra, p_dec, p_z = sub_ra[ pp ], sub_dec[ pp ], sub_z[ pp ]
			ps_ra, ps_dec, ps_z = ra_set[ pp ], dec_set[ pp ], z_set[ pp ]

			##.
			d_file = home + 'bcg_mask_img/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

			out_file = home + 'member_files/redMap_contral_galx/BG_img/ctrl-tract_BG_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img.fits'

			offset_file = home + 'photo_files/pos_offset_correct_imgs/offset/random_%s-band_ra%.3f_dec%.3f_z%.3f_star-pos-offset.csv'

			gal_file = home + 'source_detect_cat/random_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'

			contrl_galx_BGcut_func( d_file, p_ra, p_dec, p_z, ps_ra, ps_dec, ps_z, band_str, gal_file, out_file, R_cut, offset_file = offset_file )


print('%d rank, Done!' % rank)

raise

### ... image resampling


