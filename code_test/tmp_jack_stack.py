import time

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from scipy import ndimage
from astropy import cosmology as apcy

from img_stack import stack_func
from img_sky_stack import sky_stack_func
from img_edg_cut_stack import cut_stack_func
from light_measure import light_measure_Z0_weit

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

### constants transform
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
kpc2m = U.kpc.to(U.m)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Msun2kg = U.M_sun.to(U.kg)
G = C.G.value

### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

### observation params
pixel = 0.396
band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])
mag_add = np.array([0, 0, 0, -0.04, 0.02])

def medi_stack_img(N_sample, data_file, out_file):

	tt_img = []
	for nn in range(N_sample):
		with h5py.File(data_file % nn, 'r') as f:
			tmp_img = np.array(f['a'])
		tt_img.append(tmp_img)

	tt_img = np.array(tt_img)
	medi_img = np.nanmedian(tt_img, axis = 0)
	with h5py.File(out_file, 'w') as f:
		f['a'] = np.array(medi_img)

	return

def aveg_stack_img(N_sample, data_file, out_file):

	tt = 0
	with h5py.File(data_file % (tt), 'r') as f:
		tmp_img = np.array(f['a'])

	Nx, Ny = tmp_img.shape[1], tmp_img.shape[0]
	mean_img = np.zeros((Ny, Nx), dtype = np.float32)
	mean_pix_cont = np.zeros((Ny, Nx), dtype = np.float32)

	for nn in range( N_sample ):

		with h5py.File(data_file % nn, 'r') as f:
			tmp_img = np.array(f['a'])
		idnn = np.isnan(tmp_img)
		mean_img[idnn == False] = mean_img[idnn == False] + tmp_img[idnn == False]
		mean_pix_cont[idnn == False] = mean_pix_cont[idnn == False] + 1.

	idzero = mean_pix_cont == 0.
	mean_pix_cont[idzero] = np.nan
	mean_img[idzero] = np.nan
	mean_img = mean_img / mean_pix_cont

	with h5py.File(out_file, 'w') as f:
		f['a'] = np.array( mean_img )

	return

def jack_samp_stack(d_file, id_set, out_file):

	tt = 0
	with h5py.File(d_file % (tt), 'r') as f:
		tmp_img = np.array(f['a'])
	Nx, Ny = tmp_img.shape[1], tmp_img.shape[0]

	sum_array_A = np.zeros( (Ny,Nx), dtype = np.float32)
	count_array_A = np.ones( (Ny,Nx), dtype = np.float32) * np.nan
	p_count_A = np.zeros( (Ny,Nx), dtype = np.float32)

	for jj in id_set:

		with h5py.File(d_file % ( jj ), 'r') as f:
			sub_img = np.array(f['a'])

		id_nn = np.isnan(sub_img)
		idv = id_nn == False
		sum_array_A[idv] = sum_array_A[idv] + sub_img[idv]
		count_array_A[idv] = sub_img[idv]
		id_nan = np.isnan(count_array_A)
		id_fals = id_nan == False
		p_count_A[id_fals] = p_count_A[id_fals] + 1.
		count_array_A[idv] = np.nan

	id_zero = p_count_A == 0
	p_count_A[id_zero] = np.nan
	sum_array_A[id_zero] = np.nan

	stack_img = sum_array_A / p_count_A
	where_are_inf = np.isinf(stack_img)
	stack_img[where_are_inf] = np.nan

	with h5py.File(out_file, 'w') as f:
		f['a'] = np.array(stack_img)

	return

home = '/home/xkchen/data/SDSS/'

sigma = np.array([3.5, 4, 4.5, 5, 5.5, 6,]) # sub-patch limit is 3

#thres_S = np.array([2, 4])
#sigma = 3.5

N_bin = 30
out_path = home + 'tmp_stack/jack/'

### set multiprocess for stacking
m, n = divmod( len(sigma), cpus)
#m, n = divmod( len(thres_S), cpus)
N_sub0, N_sub1 = m * rank, (rank + 1) * m
if rank == cpus - 1:
	N_sub1 += n

### too bright imgs
cat_brit = pds.read_csv(home + 'selection/tmp/cluster_to_bright_cat.csv')
set_ra, set_dec, set_z = np.array(cat_brit.ra), np.array(cat_brit.dec), np.array(cat_brit.z)
out_ra = ['%.3f' % ll for ll in set_ra]
out_dec = ['%.3f' % ll for ll in set_dec]

### selected imgs
dat = pds.read_csv(home + 'selection/tmp/tot_clust_remain_cat_%.1f-sigma.csv' % (sigma[N_sub0: N_sub1][0]),)
#dat = pds.read_csv( home + 'selection/tmp/tot_clust_remain_cat_%d-patch_%.1f-sigm.csv' % (thres_S[N_sub0: N_sub1][0], sigma),)

ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

lis_ra, lis_dec, lis_z = [], [], []
lis_x, lis_y = [], []
for ll in range( len(z) ):
	identi = ('%.3f' % ra[ll] in out_ra) & ('%.3f' % dec[ll] in out_dec)
	if identi == True:
		continue
	else:
		lis_ra.append( ra[ll] )
		lis_dec.append( dec[ll] )
		lis_z.append( z[ll] )
		lis_x.append( clus_x[ll] )
		lis_y.append( clus_y[ll] )

lis_ra = np.array(lis_ra)
lis_dec = np.array(lis_dec)
lis_z = np.array(lis_z)
lis_x = np.array(lis_x)
lis_y = np.array(lis_y)

zN = len(lis_z)
n_step = zN // N_bin
id_arr = np.linspace(0, zN - 1, zN)
id_arr = id_arr.astype(int)

id_cen = 0 # BCG-stacking
d_file = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'
"""
for nn in range(N_bin):

	if nn == N_bin - 1:
		dot = id_arr[nn * n_step:]
	else:
		dot = id_arr[nn * n_step: (nn + 1) * n_step]

	set_z = lis_z[dot]
	set_ra = lis_ra[dot]
	set_dec = lis_dec[dot]

	set_x = lis_x[dot]
	set_y = lis_y[dot]
	'''
	### change sigma limit
	sub_img_file = out_path + 'clust_BCG-stack_%.1f-sigma_sub-%d.h5' % (sigma[N_sub0: N_sub1][0], nn)
	sub_cont_file = out_path + 'clust_BCG-stack_pix-cont_%.1f-sigma_sub-%d.h5' % (sigma[N_sub0: N_sub1][0], nn)
	'''
	### change blocks
	sub_img_file = out_path + 'clust_BCG-stack_%.1f-sigma_%d-patch_sub-%d.h5' % (sigma, thres_S[N_sub0: N_sub1][0], nn)
	sub_cont_file = out_path + 'clust_BCG-stack_pix-cont_%.1f-sigma_%d-patch_sub-%d.h5' % (sigma, thres_S[N_sub0: N_sub1][0], nn)

	stack_func(d_file, sub_img_file, set_z, set_ra, set_dec, band[0], set_x, set_y, id_cen, rms_file = None, pix_con_file = sub_cont_file,)

for nn in range(N_bin):

	id_arry = np.linspace(0, N_bin -1, N_bin)
	id_arry = id_arry.astype(int)
	jack_id = list(id_arry)
	jack_id.remove(jack_id[nn])
	jack_id = np.array(jack_id)
	'''
	### change sigma limit
	d_file = out_path + 'clust_BCG-stack_%.1f-sigma_' % (sigma[N_sub0: N_sub1][0],) + 'sub-%d.h5'
	jack_img_file = out_path + 'clust_BCG-stack_%.1f-sigma_jack-%d.h5' % (sigma[N_sub0: N_sub1][0], nn)
	jack_samp_stack(d_file, jack_id, jack_img_file)

	d_file = out_path + 'clust_BCG-stack_pix-cont_%.1f-sigma_' % (sigma[N_sub0: N_sub1][0],) + 'sub-%d.h5'
	jack_cont_file = out_path + 'clust_BCG-stack_pix-cont_%.1f-sigma_jack-%d.h5' % (sigma[N_sub0: N_sub1][0], nn)
	jack_samp_stack(d_file, jack_id, jack_cont_file)
	'''
	### change blocks
	d_file = out_path + 'clust_BCG-stack_%.1f-sigma_%d-patch_' % (sigma, thres_S[N_sub0: N_sub1][0],) + 'sub-%d.h5'
	jack_img_file = out_path + 'clust_BCG-stack_%.1f-sigma_%d-patch_jack-%d.h5' % (sigma, thres_S[N_sub0: N_sub1][0], nn)
	jack_samp_stack(d_file, jack_id, jack_img_file)

	d_file = out_path + 'clust_BCG-stack_pix-cont_%.1f-sigma_%d-patch_' % (sigma, thres_S[N_sub0: N_sub1][0],) + 'sub-%d.h5'
	jack_cont_file = out_path + 'clust_BCG-stack_pix-cont_%.1f-sigma_%d-patch_jack-%d.h5' % (sigma, thres_S[N_sub0: N_sub1][0], nn)
	jack_samp_stack(d_file, jack_id, jack_cont_file)
"""
### find the max Radius limit
lim_r0 = 0
lim_r1 = 0
for nn in range(N_bin):

	with h5py.File(out_path + 'CC_clust_BCG-stack_%.1f-sigma_sub-%d.h5' % (sigma[N_sub0: N_sub1][0], nn), 'r') as f:
	#with h5py.File(out_path + 'clust_BCG-stack_%.1f-sigma_%d-patch_sub-%d.h5' % (sigma, thres_S[N_sub0: N_sub1][0], nn), 'r') as f:
		tmp_img = np.array(f['a'])
	xn, yn = np.int(tmp_img.shape[1] / 2), np.int(tmp_img.shape[0] / 2)
	id_nn = np.isnan(tmp_img)
	eff_y, eff_x = np.where(id_nn == False)
	dR = np.sqrt((eff_y - yn)**2 + (eff_x - xn)**2)
	dR_max = np.int( dR.max() ) + 1
	lim_r0 = np.max([lim_r0, dR_max])

	with h5py.File(out_path + 'CC_clust_BCG-stack_%.1f-sigma_jack-%d.h5' % (sigma[N_sub0: N_sub1][0], nn), 'r') as f:
	#with h5py.File(out_path + 'clust_BCG-stack_%.1f-sigma_%d-patch_jack-%d.h5' % (sigma, thres_S[N_sub0: N_sub1][0], nn), 'r') as f:
		sub_jk_img = np.array(f['a'])
	xn, yn = np.int(sub_jk_img.shape[1] / 2), np.int(sub_jk_img.shape[0] / 2)
	id_nn = np.isnan(sub_jk_img)
	eff_y, eff_x = np.where(id_nn == False)
	dR = np.sqrt((eff_y - yn)**2 + (eff_x - xn)**2)
	dR_max = np.int( dR.max() ) + 1
	lim_r1 = np.max([lim_r1, dR_max])

lim_r0 = commd.gather(lim_r0, root = 0)
lim_r1 = commd.gather(lim_r1, root = 0)
if rank == 0:
	max_r0 = np.max(lim_r0)
	lim_r0 = np.array(lim_r0)
	lim_r0[:] = max_r0

	max_r1 = np.max(lim_r1)
	lim_r1 = np.array(lim_r1)
	lim_r1[:] = max_r1

commd.Barrier()
lim_r0 = commd.scatter(lim_r0, root = 0)
lim_r1 = commd.scatter(lim_r1, root = 0)

### SB profile measurement and save
r_bins_0 = np.logspace(0, np.log10(lim_r0), 110) #95, 110)
r_bins_1 = np.logspace(0, np.log10(lim_r1), 110)

for nn in range(N_bin):

	# individual samples
	with h5py.File(out_path + 'CC_clust_BCG-stack_%.1f-sigma_sub-%d.h5' % (sigma[N_sub0: N_sub1][0], nn), 'r') as f:
	#with h5py.File(out_path + 'clust_BCG-stack_%.1f-sigma_%d-patch_sub-%d.h5' % (sigma, thres_S[N_sub0: N_sub1][0], nn), 'r') as f:
		tmp_img = np.array(f['a'])

	with h5py.File(out_path + 'CC_clust_BCG-stack_pix-cont_%.1f-sigma_sub-%d.h5' % (sigma[N_sub0: N_sub1][0], nn), 'r') as f:
	#with h5py.File(out_path + 'clust_BCG-stack_pix-cont_%.1f-sigma_%d-patch_sub-%d.h5' % (sigma, thres_S[N_sub0: N_sub1][0], nn), 'r') as f:
		tmp_cont = np.array(f['a'])

	xn, yn = np.int(tmp_img.shape[1] / 2), np.int(tmp_img.shape[0] / 2)
	Intns, Angl_r, Intns_err, npix, nratio = light_measure_Z0_weit(tmp_img, tmp_cont, pixel, xn, yn, r_bins_0)
	sb_arr, sb_err_arr = Intns / pixel**2, Intns_err / pixel**2
	r_arr = Angl_r

	with h5py.File(out_path + 'CC_clust_BCG-stack_SB_%.1f-sigma_sub-%d.h5' % (sigma[N_sub0: N_sub1][0], nn), 'w') as f:
	#with h5py.File(out_path + 'clust_BCG-stack_SB_%.1f-sigma_%d-patch_sub-%d.h5' % (sigma, thres_S[N_sub0: N_sub1][0], nn), 'w') as f:
		f['r'] = np.array(r_arr)
		f['sb'] = np.array(sb_arr)
		f['sb_err'] = np.array(sb_err_arr)
		f['nratio'] = np.array(nratio)
		f['npix'] = np.array(npix)

	# jack sub-sample
	with h5py.File(out_path + 'CC_clust_BCG-stack_%.1f-sigma_jack-%d.h5' % (sigma[N_sub0: N_sub1][0], nn), 'r') as f:
	#with h5py.File(out_path + 'clust_BCG-stack_%.1f-sigma_%d-patch_jack-%d.h5' % (sigma, thres_S[N_sub0: N_sub1][0], nn), 'r') as f:
		sub_jk_img = np.array(f['a'])

	with h5py.File(out_path + 'CC_clust_BCG-stack_pix-cont_%.1f-sigma_jack-%d.h5' % (sigma[N_sub0: N_sub1][0], nn), 'r') as f:
	#with h5py.File(out_path + 'clust_BCG-stack_pix-cont_%.1f-sigma_%d-patch_jack-%d.h5' % (sigma, thres_S[N_sub0: N_sub1][0], nn), 'r') as f:
		sub_jk_cont = np.array(f['a'])

	xn, yn = np.int(sub_jk_img.shape[1] / 2), np.int(sub_jk_img.shape[0] / 2)
	Intns, Angl_r, Intns_err, npix, nratio = light_measure_Z0_weit(sub_jk_img, sub_jk_cont, pixel, xn, yn, r_bins_1)
	sb_arr, sb_err_arr = Intns / pixel**2, Intns_err / pixel**2
	r_arr = Angl_r

	with h5py.File(out_path + 'CC_clust_BCG-stack_SB_%.1f-sigma_jk-sub-%d.h5' % (sigma[N_sub0: N_sub1][0], nn), 'w') as f:
	#with h5py.File(out_path + 'clust_BCG-stack_SB_%.1f-sigma_%d-patch_jk-sub-%d.h5' % (sigma, thres_S[N_sub0: N_sub1][0], nn), 'w') as f:
		f['r'] = np.array(r_arr)
		f['sb'] = np.array(sb_arr)
		f['sb_err'] = np.array(sb_err_arr)
		f['nratio'] = np.array(nratio)
		f['npix'] = np.array(npix)

### compute the average img of sub-jackknife samples
# change sigma limit
d_file = out_path + 'CC_clust_BCG-stack_%.1f-sigma' % (sigma[N_sub0: N_sub1][0],) + '_jack-%d.h5'
out_file = out_path + 'CC_clust_BCG-stack_%.1f-sigma_Mean-jk_img.h5' % (sigma[N_sub0: N_sub1][0],)
aveg_stack_img(N_bin, d_file, out_file)

d_file = out_path + 'CC_clust_BCG-stack_pix-cont_%.1f-sigma' % (sigma[N_sub0: N_sub1][0],) + '_jack-%d.h5'
out_file = out_path + 'CC_clust_BCG-stack_pix-cont_%.1f-sigma_Mean-jk.h5' % (sigma[N_sub0: N_sub1][0],)
aveg_stack_img(N_bin, d_file, out_file)
'''
# change blocks
d_file = out_path + 'clust_BCG-stack_%.1f-sigma_%d-patch' % (sigma, thres_S[N_sub0: N_sub1][0],) + '_jack-%d.h5'
out_file = out_path + 'clust_BCG-stack_%.1f-sigma_%d-patch_Mean-jk_img.h5' % (sigma, thres_S[N_sub0: N_sub1][0],)
aveg_stack_img(N_bin, d_file, out_file)

d_file = out_path + 'clust_BCG-stack_pix-cont_%.1f-sigma_%d-patch' % (sigma, thres_S[N_sub0: N_sub1][0],) + '_jack-%d.h5'
out_file = out_path + 'clust_BCG-stack_pix-cont_%.1f-sigma_%d-patch_Mean-jk.h5' % (sigma, thres_S[N_sub0: N_sub1][0],)
aveg_stack_img(N_bin, d_file, out_file)
'''
print('finished !')
