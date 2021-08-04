import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from scipy.ndimage import gaussian_filter
from scipy import interpolate as interp
# mode 
from img_pre_selection import cat_match_func, get_mu_sigma
from fig_out_module import cc_grid_img, grid_img
from fig_out_module import zref_BCG_pos_func
from img_mask import source_detect_func, mask_func
from img_mask_adjust import adjust_mask_func
from img_condition_select import img_condition_select_func

from img_block_select import diffuse_identi_func
from img_pre_selection import map_mu_sigma_func
from img_resample import resamp_func

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

# constant
rad2asec = U.rad.to(U.arcsec)
band = ['r', 'g', 'i',]
mag_add = np.array([0, 0, 0])
pixel = 0.396

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'
z_ref = 0.25

##### random images
'''
### random imgs (selection based on source detection)
for kk in range( 3 ):

	dat = pds.read_csv( home + 'selection/rand_%s_band_catalog.csv' % band[ kk ],)
	ra, dec, z = np.array( dat.ra), np.array( dat.dec), np.array( dat.z)

	zN = len( z )
	m, n = divmod(zN, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	cat_file = home + 'source_detect_cat/random_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
	data_file = home + 'redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

	rule_file = '/home/xkchen/project/tmp/random_tot-%s-band_bad-img_cat_%d-rank.csv' % ( band[kk], rank)
	remain_file = '/home/xkchen/project/tmp/random_tot-%s-band_norm-img_cat_%d-rank.csv' % ( band[kk], rank)

	img_condition_select_func(band[kk], ra[N_sub0:N_sub1], dec[N_sub0:N_sub1], z[N_sub0:N_sub1],
		data_file, cat_file, rule_file, remain_file,)

	commd.Barrier()
	print('finished, random imgs, sub-select')

	if rank == 0:

		norm_ra, norm_dec, norm_z = np.array([]), np.array([]), np.array([])
		norm_imgx, norm_imgy = np.array([]), np.array([])

		bad_ra, bad_dec, bad_z = np.array([]), np.array([]), np.array([])
		bad_imgx, bad_imgy = np.array([]), np.array([]) 

		for mm in range( cpus ):

			p_dat = pds.read_csv('/home/xkchen/project/tmp/random_tot-%s-band_norm-img_cat_%d-rank.csv' % (band[kk],mm),)
			p_ra, p_dec, p_z = np.array(p_dat['ra']), np.array(p_dat['dec']), np.array(p_dat['z'])
			p_bcgx, p_bcgy = np.array(p_dat['bcg_x']), np.array(p_dat['bcg_y'])

			norm_ra = np.r_[ norm_ra, p_ra ]
			norm_dec = np.r_[ norm_dec, p_dec ]
			norm_z = np.r_[ norm_z, p_z ]
			norm_imgx = np.r_[ norm_imgx, p_bcgx ]
			norm_imgy = np.r_[ norm_imgy, p_bcgy ]

			p_dat = pds.read_csv('/home/xkchen/project/tmp/random_tot-%s-band_bad-img_cat_%d-rank.csv' % (band[kk],mm),)
			p_ra, p_dec, p_z = np.array(p_dat['ra']), np.array(p_dat['dec']), np.array(p_dat['z'])
			p_bcgx, p_bcgy = np.array(p_dat['bcg_x']), np.array(p_dat['bcg_y'])

			bad_ra = np.r_[ bad_ra, p_ra ]
			bad_dec = np.r_[ bad_dec, p_dec ]
			bad_z = np.r_[ bad_z, p_z ]
			bad_imgx = np.r_[ bad_imgx, p_bcgx ]
			bad_imgy = np.r_[ bad_imgy, p_bcgy ]

		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
		values = [ norm_ra, norm_dec, norm_z, norm_imgx, norm_imgy ]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv( '/home/xkchen/random_tot-%s-band_norm-img_cat.csv' % band[kk] )

		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
		values = [ bad_ra, bad_dec, bad_z, bad_imgx, bad_imgy ]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv( '/home/xkchen/random_tot-%s-band_bad-img_cat.csv' % band[kk] )

	print('finished, pre-selection!')
'''

'''
### random imgs (image mean flux and scatter)
for kk in range( 3 ):
	dat = pds.read_csv('/home/xkchen/random_tot-%s-band_norm-img_cat.csv' % band[ kk ],)
	ra, dec, z = np.array( dat.ra), np.array( dat.dec), np.array( dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	zN = len( z )
	m, n = divmod(zN, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	L_cen = 500
	N_step = 100 # 200

	img_file = home + 'tmp_stack/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
	out_file = '/home/xkchen/project/tmp/random_%s-band_%d-grid-img_mu-sigma_%d-rank.csv' % (band[kk], N_step, rank)

	map_mu_sigma_func(ra[N_sub0:N_sub1], dec[N_sub0:N_sub1], z[N_sub0:N_sub1], clus_x[N_sub0:N_sub1], clus_y[N_sub0:N_sub1],
		img_file, band[kk], L_cen, N_step, out_file,)

	commd.Barrier()
	print('finished, sub-collection!')

	if rank == 0:

		tt_ra, tt_dec, tt_z = np.array([]), np.array([]), np.array([])
		tt_imgx, tt_imgy = np.array([]), np.array([])
		tt_cen_mu, tt_cen_sigm = np.array([]), np.array([])
		tt_img_mu, tt_img_sigm = np.array([]), np.array([])

		for mm in range( cpus ):

			p_dat = pds.read_csv(
				'/home/xkchen/project/tmp/random_%s-band_%d-grid-img_mu-sigma_%d-rank.csv' % (band[kk], N_step, mm),)
			p_ra, p_dec, p_z = np.array(p_dat['ra']), np.array(p_dat['dec']), np.array(p_dat['z'])
			p_bcgx, p_bcgy = np.array(p_dat['bcg_x']), np.array(p_dat['bcg_y'])
			p_cen_mu, p_cen_sigm = np.array(p_dat['cen_mu']), np.array(p_dat['cen_sigma'])
			p_img_mu, p_img_sigm = np.array(p_dat['img_mu']), np.array(p_dat['img_sigma'])

			tt_ra = np.r_[ tt_ra, p_ra ]
			tt_dec = np.r_[ tt_dec, p_dec ]
			tt_z = np.r_[ tt_z, p_z ]
			tt_imgx = np.r_[ tt_imgx, p_bcgx ]
			tt_imgy = np.r_[ tt_imgy, p_bcgy ]
			tt_cen_mu = np.r_[ tt_cen_mu, p_cen_mu ]
			tt_cen_sigm = np.r_[ tt_cen_sigm, p_cen_sigm ]
			tt_img_mu = np.r_[ tt_img_mu, p_img_mu ]
			tt_img_sigm = np.r_[ tt_img_sigm, p_img_sigm ]

		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'cen_mu', 'cen_sigma', 'img_mu', 'img_sigma',]
		values = [ tt_ra, tt_dec, tt_z, tt_imgx, tt_imgy, tt_cen_mu, tt_cen_sigm, tt_img_mu, tt_img_sigm ]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv('/home/xkchen/random_%s-band_%d-grid-img_mu-sigma.csv' % (band[kk], N_step),)

print('finished random-img part!')
'''

### img selection, based on 2D flux hist.
for kk in range( 3 ):

	dat = pds.read_csv('/home/xkchen/random_tot-%s-band_norm-img_cat.csv' % band[ kk ],)
	ra, dec, z = np.array( dat.ra), np.array( dat.dec), np.array( dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	L_cen = 500
	N_step = 200 # 100

	d_file = home + 'tmp_stack/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
	mu_sigm_file = '/home/xkchen/random_%s-band_%d-grid-img_mu-sigma.csv' % (band[kk], N_step)

	thres_S0, thres_S1 = 3, 5
	sigma = 6

	rule_file = '/home/xkchen/random_%s-band_tot_rule-out_cat_set_%d-grid_%.1f-sigma.csv' % (band[kk], N_step, sigma)
	remain_file = '/home/xkchen/random_%s-band_tot_remain_cat_set_%d-grid_%.1f-sigma.csv' % (band[kk], N_step, sigma)

	diffuse_identi_func( band[kk], ra, dec, z, d_file, rule_file, remain_file, thres_S0, thres_S1, sigma, mu_sigm_file,
		L_cen, N_step, id_single = False, id_mode = True, bin_w = 10, id_mean = False,)

print('random img select finished!')

##### cluster images
'''
### cluster imgs (selection based on source detection)
for kk in range( 3 ):

	dat = pds.read_csv( home + 'selection/%s_band_sky_catalog.csv' % band[ kk ],)
	ra, dec, z = np.array( dat.ra), np.array( dat.dec), np.array( dat.z)

	zN = len( z )
	m, n = divmod(zN, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	cat_file = home + 'source_detect_cat/cluster_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
	data_file = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

	rule_file = '/home/xkchen/project/tmp/cluster_tot-%s-band_bad-img_cat_%d-rank.csv' % ( band[kk], rank)
	remain_file = '/home/xkchen/project/tmp/cluster_tot-%s-band_norm-img_cat_%d-rank.csv' % ( band[kk], rank)

	img_condition_select_func(band[kk], ra[N_sub0:N_sub1], dec[N_sub0:N_sub1], z[N_sub0:N_sub1],
		data_file, cat_file, rule_file, remain_file,)

	commd.Barrier()
	print('finished, clusetr imgs, sub-select')

	if rank == 0:

		norm_ra, norm_dec, norm_z = np.array([]), np.array([]), np.array([])
		norm_imgx, norm_imgy = np.array([]), np.array([])

		bad_ra, bad_dec, bad_z = np.array([]), np.array([]), np.array([])
		bad_imgx, bad_imgy = np.array([]), np.array([]) 

		for mm in range( cpus ):

			p_dat = pds.read_csv('/home/xkchen/project/tmp/cluster_tot-%s-band_norm-img_cat_%d-rank.csv' % (band[kk],mm),)
			p_ra, p_dec, p_z = np.array(p_dat['ra']), np.array(p_dat['dec']), np.array(p_dat['z'])
			p_bcgx, p_bcgy = np.array(p_dat['bcg_x']), np.array(p_dat['bcg_y'])

			norm_ra = np.r_[ norm_ra, p_ra ]
			norm_dec = np.r_[ norm_dec, p_dec ]
			norm_z = np.r_[ norm_z, p_z ]
			norm_imgx = np.r_[ norm_imgx, p_bcgx ]
			norm_imgy = np.r_[ norm_imgy, p_bcgy ]

			p_dat = pds.read_csv('/home/xkchen/project/tmp/cluster_tot-%s-band_bad-img_cat_%d-rank.csv' % (band[kk],mm),)
			p_ra, p_dec, p_z = np.array(p_dat['ra']), np.array(p_dat['dec']), np.array(p_dat['z'])
			p_bcgx, p_bcgy = np.array(p_dat['bcg_x']), np.array(p_dat['bcg_y'])

			bad_ra = np.r_[ bad_ra, p_ra ]
			bad_dec = np.r_[ bad_dec, p_dec ]
			bad_z = np.r_[ bad_z, p_z ]
			bad_imgx = np.r_[ bad_imgx, p_bcgx ]
			bad_imgy = np.r_[ bad_imgy, p_bcgy ]

		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
		values = [ norm_ra, norm_dec, norm_z, norm_imgx, norm_imgy ]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv( '/home/xkchen/cluster_tot-%s-band_norm-img_cat.csv' % band[kk] )

		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
		values = [ bad_ra, bad_dec, bad_z, bad_imgx, bad_imgy ]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv( '/home/xkchen/cluster_tot-%s-band_bad-img_cat.csv' % band[kk] )

	print('finished, pre-selection!')
'''

'''
### cluster imgs (image mean flux and scatter)
for kk in range( 3 ):

	dat = pds.read_csv('/home/xkchen/cluster_tot-%s-band_norm-img_cat.csv' % band[ kk ],)
	ra, dec, z = np.array( dat.ra), np.array( dat.dec), np.array( dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	zN = len( z )
	m, n = divmod(zN, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	L_cen = 500
	N_step = 100 # 200

	img_file = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
	out_file = '/home/xkchen/project/tmp/cluster_%s-band_%d-grid-img_mu-sigma_%d-rank.csv' % (band[kk], N_step, rank)

	map_mu_sigma_func(ra[N_sub0:N_sub1], dec[N_sub0:N_sub1], z[N_sub0:N_sub1], clus_x[N_sub0:N_sub1], clus_y[N_sub0:N_sub1],
		img_file, band[kk], L_cen, N_step, out_file,)

	commd.Barrier()
	print('finished, sub-collection!')

	if rank == 0:

		tt_ra, tt_dec, tt_z = np.array([]), np.array([]), np.array([])
		tt_imgx, tt_imgy = np.array([]), np.array([])
		tt_cen_mu, tt_cen_sigm = np.array([]), np.array([])
		tt_img_mu, tt_img_sigm = np.array([]), np.array([])

		for mm in range( cpus ):

			p_dat = pds.read_csv(
				'/home/xkchen/project/tmp/cluster_%s-band_%d-grid-img_mu-sigma_%d-rank.csv' % (band[kk], N_step, mm),)
			p_ra, p_dec, p_z = np.array(p_dat['ra']), np.array(p_dat['dec']), np.array(p_dat['z'])
			p_bcgx, p_bcgy = np.array(p_dat['bcg_x']), np.array(p_dat['bcg_y'])
			p_cen_mu, p_cen_sigm = np.array(p_dat['cen_mu']), np.array(p_dat['cen_sigma'])
			p_img_mu, p_img_sigm = np.array(p_dat['img_mu']), np.array(p_dat['img_sigma'])

			tt_ra = np.r_[ tt_ra, p_ra ]
			tt_dec = np.r_[ tt_dec, p_dec ]
			tt_z = np.r_[ tt_z, p_z ]
			tt_imgx = np.r_[ tt_imgx, p_bcgx ]
			tt_imgy = np.r_[ tt_imgy, p_bcgy ]
			tt_cen_mu = np.r_[ tt_cen_mu, p_cen_mu ]
			tt_cen_sigm = np.r_[ tt_cen_sigm, p_cen_sigm ]
			tt_img_mu = np.r_[ tt_img_mu, p_img_mu ]
			tt_img_sigm = np.r_[ tt_img_sigm, p_img_sigm ]

		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'cen_mu', 'cen_sigma', 'img_mu', 'img_sigma',]
		values = [ tt_ra, tt_dec, tt_z, tt_imgx, tt_imgy, tt_cen_mu, tt_cen_sigm, tt_img_mu, tt_img_sigm ]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv('/home/xkchen/cluster_%s-band_%d-grid-img_mu-sigma.csv' % (band[kk], N_step),)

print('finished cluster-img part!')
'''

### img selection, based on 2D flux hist.
for kk in range( 3 ):

	dat = pds.read_csv('/home/xkchen/cluster_tot-%s-band_norm-img_cat.csv' % band[ kk ],)
	ra, dec, z = np.array( dat.ra), np.array( dat.dec), np.array( dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	L_cen = 500
	N_step = 200 # 100

	d_file = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
	mu_sigm_file = '/home/xkchen/cluster_%s-band_%d-grid-img_mu-sigma.csv' % (band[kk], N_step)

	thres_S0, thres_S1 = 3, 5
	sigma = 6

	rule_file = '/home/xkchen/cluster_%s-band_tot_rule-out_cat_set_%d-grid_%.1f-sigma.csv' % (band[kk], N_step, sigma)
	remain_file = '/home/xkchen/cluster_%s-band_tot_remain_cat_set_%d-grid_%.1f-sigma.csv' % (band[kk], N_step, sigma)

	diffuse_identi_func( band[kk], ra, dec, z, d_file, rule_file, remain_file, thres_S0, thres_S1, sigma, mu_sigm_file,
		L_cen, N_step, id_single = False, id_mode = True, bin_w = 10, id_mean = False,)

print('cluster img select finished!')

