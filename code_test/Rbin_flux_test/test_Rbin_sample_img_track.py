import h5py
import pandas as pds
import numpy as np

import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
import astropy.io.fits as fits

import scipy.stats as sts
import scipy.special as special
from scipy import optimize
from scipy.stats import binned_statistic as binned
## mode
from img_Rbin_pix_get import pix_flux_set_func
from img_Rbin_pix_get import Rbin_flux_track
from img_Rbin_pix_get import radi_bin_flux_set_func

from light_measure_tmp import SB_measure_Z0_weit_func
from light_measure_tmp import light_measure_weit

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

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

load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'
z_ref = 0.25

###****************###
##### stacking in physical coordinate
R0 = 2840
id_cen = 0
n_rbins = 110
N_bin = 30

R_bins = np.logspace(0, np.log10(R0), n_rbins)

Da0 = Test_model.angular_diameter_distance( z_ref ).value
phy_R = Da0 * 1e3 * R_bins * pixel / rad2asec
'''
if rank == 0:
	# random img, BCG-stack
	with h5py.File( load + 'stack/random_r-band_BCG-stack_Mean_jack_img_z-ref.h5', 'r') as f:
		aveg_img = np.array( f['a'] )

	with h5py.File( load + 'stack/random_r-band_BCG-stack_Mean_jack_pix-cont_z-ref.h5', 'r') as f:
		aveg_cont = np.array( f['a'] )

	xn, yn = np.int( aveg_img.shape[1] / 2), np.int(aveg_img.shape[0] / 2)

	bin_flux_file = '/home/xkchen/project/stack/phyR_stack/random_BCG-stack_%.3f-kpc_flux-arr_z-ref.h5'

	Intns, phy_r, Intns_err, N_pix, nsum_ratio = light_measure_weit(aveg_img, aveg_cont, pixel, xn, yn, z_ref, R_bins, bin_flux_file,)

	keys = ['f_mean', 'R_kpc', 'N_pix']
	values = [ Intns, phy_r, N_pix]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame(fill)
	out_data.to_csv('/home/xkchen/project/stack/random_BCG-stack_Mean_f_pros_z-ref.csv')

	# cluster img, BCG-stack
	with h5py.File( load + 'stack/com-BCG-star-Mass_r-band_Mean_jack_img_z-ref_with-selection.h5', 'r') as f:
		aveg_img = np.array( f['a'] )

	with h5py.File( load + 'stack/com-BCG-star-Mass_r-band_Mean_jack_pix-cont_z-ref_with-selection.h5', 'r') as f:
		aveg_cont = np.array( f['a'] )

	xn, yn = np.int( aveg_img.shape[1] / 2), np.int(aveg_img.shape[0] / 2)

	bin_flux_file = '/home/xkchen/project/stack/phyR_stack/com-BCG-star-Mass_%.3f-kpc_flux-arr_z-ref.h5'

	Intns, phy_r, Intns_err, N_pix, nsum_ratio = light_measure_weit(aveg_img, aveg_cont, pixel, xn, yn, z_ref, R_bins, bin_flux_file,)

	keys = ['f_mean', 'R_kpc', 'N_pix']
	values = [ Intns, phy_r, N_pix]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame(fill)
	out_data.to_csv('/home/xkchen/project/stack/com-BCG-star-Mass_Mean_f_pros_z-ref.csv')

	# random img, random center
	for ll in range( 10 ):

		with h5py.File( load + 'stack/random_r-band_rand-stack_Mean_jack_img_z-ref_%d-rank.h5' % ll, 'r') as f:
			aveg_img = np.array( f['a'] )

		with h5py.File( load + 'stack/random_r-band_rand-stack_Mean_jack_pix-cont_z-ref_%d-rank.h5' % ll, 'r') as f:
			aveg_cont = np.array( f['a'] )

		xn, yn = np.int( aveg_img.shape[1] / 2), np.int(aveg_img.shape[0] / 2)

		bin_flux_file = '/home/xkchen/project/stack/phyR_stack/random_rand-stack_%.3f-kpc_flux-arr_z-ref' + '_%d-rank.h5' % ll

		Intns, phy_r, Intns_err, N_pix, nsum_ratio = light_measure_weit(aveg_img, aveg_cont, pixel, xn, yn, z_ref, R_bins, bin_flux_file,)

		keys = ['f_mean', 'R_kpc', 'N_pix']
		values = [ Intns, phy_r, N_pix]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame(fill)
		out_data.to_csv('/home/xkchen/project/stack/random_rand-stack_Mean_f_pros_z-ref_%d-rank.csv' % ll)

commd.Barrier()
'''

### trace sample imgs
# cluster
lo_dat = pds.read_csv(load + 'img_cat/low_BCG_star-Mass_r-band_remain_cat_resamp_BCG-pos.csv',)
lo_ra, lo_dec, lo_z = np.array(lo_dat.ra), np.array(lo_dat.dec), np.array(lo_dat.z)
lo_imgx, lo_imgy = np.array(lo_dat.bcg_x), np.array(lo_dat.bcg_y)

hi_dat = pds.read_csv(load + 'img_cat/high_BCG_star-Mass_r-band_remain_cat_resamp_BCG-pos.csv',)
hi_ra, hi_dec, hi_z = np.array(hi_dat.ra), np.array(hi_dat.dec), np.array(hi_dat.z)
hi_imgx, hi_imgy = np.array(hi_dat.bcg_x), np.array(hi_dat.bcg_y)

ra = np.r_[ lo_ra, hi_ra ]
dec = np.r_[ lo_dec, hi_dec ]
z = np.r_[ lo_z, hi_z ]

clus_x = np.r_[ lo_imgx, hi_imgx ]
clus_y = np.r_[ lo_imgy, hi_imgy ]

cdat = pds.read_csv('/home/xkchen/project/stack/com-BCG-star-Mass_Mean_f_pros_z-ref.csv')
phy_r = np.array( cdat['R_kpc'] )
flux_r = np.array( cdat['f_mean'] )
npix = np.array( cdat['N_pix'] )

#idNul = np.isnan( flux_r )
#phy_r = phy_r[idNul == False]
idNul = npix > 0
phy_r = phy_r[ idNul ]
Nr = len( phy_r )

m, n = divmod( Nr, cpus)
N_sub0, N_sub1 = m * rank, (rank + 1) * m
if rank == cpus - 1:
	N_sub1 += n

tmp_R = phy_r[N_sub0 : N_sub1]

stack_img = load + 'stack/com-BCG-star-Mass_r-band_Mean_jack_img_z-ref_with-selection.h5'

with h5py.File( stack_img, 'r') as f:
	aveg_img = np.array( f['a'] )
xn, yn = np.int( aveg_img.shape[1] / 2), np.int(aveg_img.shape[0] / 2)

out_file = '/home/xkchen/project/stack/phyR_stack/com-BCG-star-Mass_%.3f-kpc_sample-img_flux_z-ref.h5'
d_file = home + 'pix_resamp_imgs/z_ref_0.25/resamp-%s-ra%.3f-dec%.3f-redshift%.3f' + '30-FWHM-ov2.fits'

for mm in range( len(tmp_R) ):

	targ_R = tmp_R[mm]

	radi_bin_flux_set_func(targ_R, phy_R, stack_img, xn, yn, R_bins, ra, dec, z, clus_x, clus_y, d_file, pixel, band[0], out_file)

print('cluster img finished')

#commd.Barrier()


# random, bcg-stack
dat = pds.read_csv(load + 'random_cat/12_21/random_r-band_tot_remain_zref_BCG-pos.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

cdat = pds.read_csv('/home/xkchen/project/stack/random_BCG-stack_Mean_f_pros_z-ref.csv')
phy_r = np.array( cdat['R_kpc'] )
flux_r = np.array( cdat['f_mean'] )
npix = np.array( cdat['N_pix'] )

#idNul = np.isnan( flux_r )
#phy_r = phy_r[idNul == False]
idNul = npix > 0
phy_r = phy_r[ idNul ]
Nr = len( phy_r )

m, n = divmod( Nr, cpus)
N_sub0, N_sub1 = m * rank, (rank + 1) * m
if rank == cpus - 1:
	N_sub1 += n

tmp_R = phy_r[N_sub0 : N_sub1]

stack_img = load + 'stack/random_r-band_BCG-stack_Mean_jack_img_z-ref.h5'

with h5py.File( stack_img, 'r') as f:
	aveg_img = np.array( f['a'] )
xn, yn = np.int( aveg_img.shape[1] / 2), np.int(aveg_img.shape[0] / 2)

out_file = '/home/xkchen/project/stack/phyR_stack/random_BCG-stack_%.3f-kpc_sample-img_flux_z-ref.h5'
d_file = home + 'tmp_stack/pix_resample/random_resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'

for mm in range( len(tmp_R) ):

	targ_R = tmp_R[mm]

	radi_bin_flux_set_func(targ_R, phy_R, stack_img, xn, yn, R_bins, ra, dec, z, clus_x, clus_y, d_file, pixel, band[0], out_file)

print('random img finished')

#commd.Barrier()


# random, rand-stack
for tt in range( 10 ):

	dat = pds.read_csv(
		'/home/xkchen/project/rand_center/random_r-band_tot_remain_mock-BCG-pos_zref_%d-rank.csv' % tt)
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	cdat = pds.read_csv('/home/xkchen/project/stack/random_rand-stack_Mean_f_pros_z-ref_%d-rank.csv' % tt)
	phy_r = np.array( cdat['R_kpc'] )
	flux_r = np.array( cdat['f_mean'] )
	npix = np.array( cdat['N_pix'] )

	#idNul = np.isnan( flux_r )
	#phy_r = phy_r[idNul == False]
	idNul = npix > 0
	phy_r = phy_r[ idNul ]
	Nr = len( phy_r )

	m, n = divmod( Nr, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	tmp_R = phy_r[N_sub0 : N_sub1]

	stack_img = load + 'stack/random_r-band_rand-stack_Mean_jack_img_z-ref_%d-rank.h5' % tt

	with h5py.File( stack_img, 'r') as f:
		aveg_img = np.array( f['a'] )
	xn, yn = np.int( aveg_img.shape[1] / 2), np.int(aveg_img.shape[0] / 2)

	out_file = '/home/xkchen/project/stack/phyR_stack/random_rand-stack_%.3f-kpc_sample-img_flux_z-ref' + '_%d-rank.h5' % tt
	d_file = home + 'tmp_stack/pix_resample/random_resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'

	for mm in range( len(tmp_R) ):

		targ_R = tmp_R[mm]

		radi_bin_flux_set_func(targ_R, phy_R, stack_img, xn, yn, R_bins, ra, dec, z, clus_x, clus_y, d_file, pixel, band[0], out_file)

	print('%d rank, random img finished' % tt)

print('%d finished all !' % rank)


####### stacking in angle coordinate
"""
###pix flux trace for random imgs (BCG-stack case) and cluster imgs
id_cen = 0
n_rbins = 100
N_bin = 30
R_bins = np.logspace(0, np.log10(2520), n_rbins)

### random img case
if rank == 0:
	with h5py.File( load + 'stack/random_r-band_BCG-stack_Mean_jack_img.h5', 'r') as f:
		aveg_rnd_img = np.array( f['a'] )

	with h5py.File( load + 'stack/random_r-band_BCG-stack_Mean_jack_pix-cont.h5', 'r') as f:
		aveg_rnd_cont = np.array( f['a'] )

	xn, yn = np.int( aveg_rnd_img.shape[1] / 2), np.int(aveg_rnd_img.shape[0] / 2)

	bin_flux_file = '/home/xkchen/project/stack/angle_stack/random_BCG-stack_%.3f-arcsec_flux-arr.h5'

	mean_intens, Angl_r, intens_err, N_pix, nsum_ratio = SB_measure_Z0_weit_func(aveg_rnd_img, aveg_rnd_cont, pixel, xn, yn, R_bins, bin_flux_file,)

	keys = ['f_mean', 'R_arcsec',]
	values = [ mean_intens, Angl_r ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame(fill)
	out_data.to_csv('/home/xkchen/project/stack/random_BCG-stack_Mean_f_pros.csv')

commd.Barrier()

### trace the sample imgs
dat = pds.read_csv(load + 'random_cat/12_21/random_r-band_tot_remain_cat.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

cdat = pds.read_csv('/home/xkchen/project/stack/random_BCG-stack_Mean_f_pros.csv')
Angl_r = np.array( cdat['R_arcsec'] )
flux_r = np.array( cdat['f_mean'] )

idNul = np.isnan( flux_r )
angl_r = Angl_r[idNul == False]
Nr = len(angl_r)

m, n = divmod( Nr, cpus)
N_sub0, N_sub1 = m * rank, (rank + 1) * m
if rank == cpus - 1:
	N_sub1 += n

tmp_R = angl_r[N_sub0 : N_sub1]

stack_img = load + 'stack/random_r-band_BCG-stack_Mean_jack_img.h5'

with h5py.File( stack_img, 'r') as f:
	aveg_rnd_img = np.array( f['a'] )
xn, yn = np.int( aveg_rnd_img.shape[1] / 2), np.int(aveg_rnd_img.shape[0] / 2)

out_file = '/home/xkchen/project/stack/random_BCG-stack_%.3f-arcsec_sample-img_flux.h5'
d_file = home + 'tmp_stack/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'

for mm in range( len(tmp_R) ):

	targ_R = tmp_R[mm]

	radi_bin_flux_set_func(targ_R, stack_img, xn, yn, R_bins, ra, dec, z, clus_x, clus_y,
		d_file, pixel, band[0], out_file,)

print('radnom img finished')


### cluster img case
if rank == 0:

	with h5py.File( load + 'stack/cluster_r-band_BCG-stack_Mean_jack_img.h5', 'r') as f:
		aveg_clus_img = np.array( f['a'] )

	with h5py.File( load + 'stack/cluster_r-band_BCG-stack_Mean_jack_pix-cont.h5', 'r') as f:
		aveg_clus_cont = np.array( f['a'] )

	xn, yn = np.int( aveg_clus_img.shape[1] / 2), np.int(aveg_clus_img.shape[0] / 2)

	bin_flux_file = '/home/xkchen/project/stack/cluster_BCG-stack_%.3f-arcsec_flux-arr.h5'

	mean_intens, Angl_r, intens_err, N_pix, nsum_ratio = SB_measure_Z0_weit_func(aveg_clus_img, aveg_clus_cont, pixel, xn, yn, R_bins, bin_flux_file,)

	keys = ['f_mean', 'R_arcsec',]
	values = [ mean_intens, Angl_r ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame(fill)
	out_data.to_csv('/home/xkchen/project/stack/cluster_BCG-stack_Mean_f_pros.csv')

commd.Barrier()

### trace the sample imgs
lo_dat = pds.read_csv(load + 'img_cat/low_BCG_star-Mass_r-band_remain_cat.csv',)
lo_ra, lo_dec, lo_z = np.array(lo_dat.ra), np.array(lo_dat.dec), np.array(lo_dat.z)
lo_imgx, lo_imgy = np.array(lo_dat.bcg_x), np.array(lo_dat.bcg_y)

hi_dat = pds.read_csv(load + 'img_cat/high_BCG_star-Mass_r-band_remain_cat.csv',)
hi_ra, hi_dec, hi_z = np.array(hi_dat.ra), np.array(hi_dat.dec), np.array(hi_dat.z)
hi_imgx, hi_imgy = np.array(hi_dat.bcg_x), np.array(hi_dat.bcg_y)

ra = np.r_[ lo_ra, hi_ra ]
dec = np.r_[ lo_dec, hi_dec ]
z = np.r_[ lo_z, hi_z ]

clus_x = np.r_[ lo_imgx, hi_imgx ]
clus_y = np.r_[ lo_imgy, hi_imgy ]


cdat = pds.read_csv('/home/xkchen/project/stack/cluster_BCG-stack_Mean_f_pros.csv')
Angl_r = np.array( cdat['R_arcsec'] )
flux_r = np.array( cdat['f_mean'] )

idNul = np.isnan( flux_r )
angl_r = Angl_r[idNul == False]
Nr = len(angl_r)

m, n = divmod( Nr, cpus)
N_sub0, N_sub1 = m * rank, (rank + 1) * m
if rank == cpus - 1:
	N_sub1 += n

tmp_R = angl_r[N_sub0 : N_sub1]

stack_img = load + 'stack/cluster_r-band_BCG-stack_Mean_jack_img.h5'

with h5py.File( stack_img, 'r') as f:
	aveg_rnd_img = np.array( f['a'] )
xn, yn = np.int( aveg_rnd_img.shape[1] / 2), np.int(aveg_rnd_img.shape[0] / 2)

out_file = '/home/xkchen/project/stack/cluster_BCG-stack_%.3f-arcsec_sample-img_flux.h5'
d_file = home + 'mask_imgs/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_30-FWHM-ov2.fits'

for mm in range( len(tmp_R) ):

	targ_R = tmp_R[mm]

	radi_bin_flux_set_func(targ_R, stack_img, xn, yn, R_bins, ra, dec, z, clus_x, clus_y,
		d_file, pixel, band[0], out_file,)

print('radnom img finished')
"""

