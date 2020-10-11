import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from img_stack import stack_func
from img_sky_stack import sky_stack_func
from img_edg_cut_stack import cut_stack_func
from list_shuffle import find_unique_shuffle_lists as find_list
from img_jack_stack import jack_main_func

# cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel, z_ref = 0.396, 0.250
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # (erg/s)/cm^-2
R0 = 1 # Mpc
Angu_ref = (R0 / Da_ref)*rad2asec
Rpp = Angu_ref / pixel

band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])

home = '/media/xkchen/My Passport/data/SDSS/'
load = '/media/xkchen/My Passport/data/SDSS/'

################## test jackknife stacking code.
dat = pds.read_csv('/home/xkchen/mywork/ICL/code/SEX/result/test_1000-to-250_cat.csv')
#dat = pds.read_csv('/home/xkchen/mywork/ICL/code/SEX/result/test_1000-to-98_cat.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

size_arr = np.array([5, 10, 15, 20, 25])
for mm in range(5):

	id_cen = 0
	n_rbins = 110
	N_bin = 30
	d_file = home + '20_10_test/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_' + '%d-FWHM-ov2.fits' % (size_arr[mm])

	sub_img = load + '20_10_test/jack_test/clust_BCG-stack_sub-%d_img' + '_%d-FWHM-ov2.h5' % (size_arr[mm])
	sub_pix_cont = load + '20_10_test/jack_test/clust_BCG-stack_sub-%d_pix-cont' + '_%d-FWHM-ov2.h5' % (size_arr[mm])
	sub_sb = load + '20_10_test/jack_test/clust_BCG-stack_sub-%d_SB-pro' + '_%d-FWHM-ov2.h5' % (size_arr[mm])

	J_sub_img = load + '20_10_test/jack_test/clust_BCG-stack_jack-sub-%d_img' + '_%d-FWHM-ov2.h5' % (size_arr[mm])
	J_sub_pix_cont = load + '20_10_test/jack_test/clust_BCG-stack_jack-sub-%d_pix-cont' + '_%d-FWHM-ov2.h5' % (size_arr[mm])
	J_sub_sb = load + '20_10_test/jack_test/clust_BCG-stack_jack-sub-%d_SB-pro' + '_%d-FWHM-ov2.h5' % (size_arr[mm])

	jack_SB_file = load + '20_10_test/jack_test/clust_BCG-stack_Mean_jack_SB-pro' + '_%d-FWHM-ov2.h5' % (size_arr[mm])
	jack_img = load + '20_10_test/jack_test/clust_BCG-stack_Mean_jack_img' + '_%d-FWHM-ov2.h5' % (size_arr[mm])
	jack_cont_arr = load + '20_10_test/jack_test/clust_BCG-stack_Mean_jack_pix-cont' + '_%d-FWHM-ov2.h5' % (size_arr[mm])

	jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[0], sub_img,
		sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,)

raise

"""
################## test edge pixel out
#dat = pds.read_csv('/home/xkchen/Downloads/new_1000_test/clust_test-1000_remain_cat.csv')
dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/re_clust-1000-select_remain_test.csv')
#dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/clust-1000-select_remain_test.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

cat_brit = pds.read_csv('/home/xkchen/tmp/00_tot_select/cluster_to_bright_cat.csv')
set_ra, set_dec, set_z = np.array(cat_brit.ra), np.array(cat_brit.dec), np.array(cat_brit.z)
out_ra = ['%.3f' % ll for ll in set_ra]
out_dec = ['%.3f' % ll for ll in set_dec]

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

N_edg = 500 # 500, 200

d_file = '/media/xkchen/My Passport/data/SDSS/tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'
#d_file = '/media/xkchen/My Passport/data/SDSS/tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

id_cen = 0

out_file = '/home/xkchen/Downloads/new_1000_test/clust_test-1000_BCG-stack_N_edg-%d_correct.h5' % ( N_edg )
rms_file = '/home/xkchen/Downloads/new_1000_test/clust_test-1000_BCG-stack_var_N_edg-%d_correct.h5' % ( N_edg )
cont_file = '/home/xkchen/Downloads/new_1000_test/clust_test-1000_BCG-stack_pix-cont_N_edg-%d_correct.h5' % ( N_edg )

#out_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_N_edg-%d_correct.h5' % ( N_edg ) # _N_edg-%d.h5' % ( N_edg )
#rms_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_var_N_edg-%d_correct.h5' % ( N_edg ) # _N_edg-%d.h5' % ( N_edg ) 
#cont_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_pix-cont_N_edg-%d_correct.h5' % ( N_edg ) # _N_edg-%d.h5' % ( N_edg )
#cut_stack_func(d_file, out_file, z, ra, dec, band[0], clus_x, clus_y, id_cen, N_edg, rms_file, cont_file,)
cut_stack_func(d_file, out_file, lis_z, lis_ra, lis_dec, band[0], lis_x, lis_y, id_cen, N_edg, rms_file, cont_file)


#dat = pds.read_csv('/home/xkchen/Downloads/new_1000_test/random_clus-1000-match_remain_cat.csv')
dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/re_random_clus-1000-match_remain_test.csv')
#dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/random_clus-1000-match_remain_test.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

cat_brit = pds.read_csv('/home/xkchen/tmp/00_tot_select/random_to_bright_cat.csv')
set_ra, set_dec, set_z = np.array(cat_brit.ra), np.array(cat_brit.dec), np.array(cat_brit.z)
out_ra = ['%.3f' % ll for ll in set_ra]
out_dec = ['%.3f' % ll for ll in set_dec]

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

d_file = '/media/xkchen/My Passport/data/SDSS/tmp_stack/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'
#d_file = '/media/xkchen/My Passport/data/SDSS/tmp_stack/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

id_cen = 0

out_file = '/home/xkchen/Downloads/new_1000_test/random_test-1000_BCG-stack_N_edg-%d_correct.h5' % ( N_edg )
rms_file = '/home/xkchen/Downloads/new_1000_test/random_test-1000_BCG-stack_var_N_edg-%d_correct.h5' % ( N_edg )
cont_file = '/home/xkchen/Downloads/new_1000_test/random_test-1000_BCG-stack_pix-cont_N_edg-%d_correct.h5' % ( N_edg )

#out_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_N_edg-%d_correct.h5' % ( N_edg ) # _N_edg-%d.h5' % ( N_edg )
#rms_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_var_N_edg-%d_correct.h5' % ( N_edg ) # _N_edg-%d.h5' % ( N_edg )
#cont_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_pix-cont_N_edg-%d_correct.h5' % ( N_edg ) # _N_edg-%d.h5' % ( N_edg )
#cut_stack_func(d_file, out_file, z, ra, dec, band[0], clus_x, clus_y, id_cen, N_edg, rms_file, cont_file,)
cut_stack_func(d_file, out_file, lis_z, lis_ra, lis_dec, band[0], lis_x, lis_y, id_cen, N_edg, rms_file, cont_file)

raise
"""
################## part 1
dat = pds.read_csv('/home/xkchen/tmp/00_tot_select/tot_clust_remain_cat.csv')
## test-1000
#dat = pds.read_csv('/home/xkchen/Downloads/new_1000_test/clust_test-1000_remain_cat.csv')
#dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/re_clust-1000-select_remain_test.csv')
#dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/clust-1000-select_remain_test.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

cat_brit = pds.read_csv('/home/xkchen/tmp/00_tot_select/cluster_to_bright_cat.csv')
set_ra, set_dec, set_z = np.array(cat_brit.ra), np.array(cat_brit.dec), np.array(cat_brit.z)

out_ra = ['%.3f' % ll for ll in set_ra]
out_dec = ['%.3f' % ll for ll in set_dec]

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

d_file = '/media/xkchen/My Passport/data/SDSS/tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'
#d_file = '/media/xkchen/My Passport/data/SDSS/tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
#d_file = '/media/xkchen/My Passport/data/SDSS/tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_add-photo-G.fits'
'''
id_cen = 1 # center-stacking
out_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_center-stack_test-train_correct.h5' # _add-photo-G.h5'# _test-train.h5' #
rms_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_center-stack_var-train_correct.h5' # _add-photo-G.h5'# _var-train.h5' #
cont_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_center-stack_pix-cont-train_correct.h5' # _add-photo-G.h5'# _pix-cont-train.h5' #
stack_func(d_file, out_file, lis_z, lis_ra, lis_dec, band[0], lis_x, lis_y, id_cen, rms_file, cont_file)
'''

id_cen = 0 # BCG-stacking

out_file = '/media/xkchen/My Passport/data/SDSS/tmp_stack/jack/clust_tot_BCG-stack_correct.h5'
rms_file = '/media/xkchen/My Passport/data/SDSS/tmp_stack/jack/clust_tot_BCG-stack_var_correct.h5' #
cont_file = '/media/xkchen/My Passport/data/SDSS/tmp_stack/jack/clust_tot_BCG-stack_pix-cont_correct.h5' #

#out_file = '/home/xkchen/Downloads/new_1000_test/clust_test-1000_BCG-stack_test-train_correct.h5'
#rms_file = '/home/xkchen/Downloads/new_1000_test/clust_test-1000_BCG-stack_var-train_correct.h5'
#cont_file = '/home/xkchen/Downloads/new_1000_test/clust_test-1000_BCG-stack_pix-cont-train_correct.h5'

#out_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_test-train_correct.h5' # _add-photo-G.h5'# _test-train.h5' #
#rms_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_var-train_correct.h5' # _add-photo-G.h5'# _var-train.h5' #
#cont_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_pix-cont-train_correct.h5' # _add-photo-G.h5'# _pix-cont-train.h5' #
stack_func(d_file, out_file, lis_z, lis_ra, lis_dec, band[0], lis_x, lis_y, id_cen, rms_file, cont_file)
raise

"""
### stacking sky img
d_file = '/media/xkchen/My Passport/data/SDSS/sky/origin_sky/sky-ra%.3f-dec%.3f-z%.3f-%s-band.fits'

id_mean = 0 # stacking sky-img
id_cen = 0 # 1
out_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_stack_sky-train.h5'#center-stack_sky-train.h5'
rms_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_stack_sky_var-train.h5'#center-stack_sky_var-train.h5'
cont_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_stack_sky_pix-cont-train.h5'#center-stack_sky_pix-cont-train.h5'
sky_stack_func(d_file, out_file, z, ra, dec, band[0], clus_x, clus_y, id_cen, id_mean, rms_file, cont_file)

id_mean = 2 # stacking sky-img minus np.median(sky-img)
id_cen = 0
out_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_sky-train.h5'
rms_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_sky_var-train.h5'
cont_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_sky_pix-cont-train.h5'
sky_stack_func(d_file, out_file, z, ra, dec, band[0], clus_x, clus_y, id_cen, id_mean, rms_file, cont_file)

'''
rnx = np.zeros((10, len(z)), dtype = np.float)
rny = np.zeros((10, len(z)), dtype = np.float)
for nn in range(10):
	tmpx = np.random.choice(2048, len(z), replace = True)
	tmpy = np.random.choice(1489, len(z), replace = True)
	rnx[nn,:] = tmpx + 0.
	rny[nn,:] = tmpy + 0.
with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_rand-img_pos-x.h5', 'w') as f:
	f['a'] = np.array(rnx)
with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_rand-img_pos-y.h5', 'w') as f:
	f['a'] = np.array(rny)
'''
with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_rand-img_pos-x.h5', 'r') as f:
	rnx = np.array(f['a'])
with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_rand-img_pos-y.h5', 'r') as f:
	rny = np.array(f['a'])

for nn in range( 10 ):
	tmpx = rnx[nn,:]
	tmpy = rny[nn,:]

	id_cen = 2
	out_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_rand-stack_sky-train_%d.h5' % nn
	rms_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_rand-stack_sky_var-train_%d.h5' % nn
	cont_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_rand-stack_sky_pix-cont-train_%d.h5' % nn
	sky_stack_func(d_file, out_file, z, ra, dec, band[0], tmpx, tmpy, id_cen, id_mean, rms_file, cont_file)
"""

dat = pds.read_csv('/home/xkchen/tmp/00_tot_select/tot_random_remain_cat.csv')

#dat = pds.read_csv('/home/xkchen/Downloads/new_1000_test/random_clus-1000-match_remain_cat.csv')
#dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/re_random_clus-1000-match_remain_test.csv')
#dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/random_clus-1000-match_remain_test.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

cat_brit = pds.read_csv('/home/xkchen/tmp/00_tot_select/random_to_bright_cat.csv')
set_ra, set_dec, set_z = np.array(cat_brit.ra), np.array(cat_brit.dec), np.array(cat_brit.z)

out_ra = ['%.3f' % ll for ll in set_ra]
out_dec = ['%.3f' % ll for ll in set_dec]

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

d_file = '/media/xkchen/My Passport/data/SDSS/tmp_stack/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'
#d_file = '/media/xkchen/My Passport/data/SDSS/tmp_stack/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
#d_file = '/media/xkchen/My Passport/data/SDSS/tmp_stack/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f_add-photo-G.fits'
'''
id_cen = 1 # center-stacking
out_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_center-stack_test-train_correct.h5' # _add-photo-G.h5'# _test-train.h5' #
rms_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_center-stack_var-train_correct.h5' # _add-photo-G.h5'# _var-train.h5' #
cont_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_center-stack_pix-cont-train_correct.h5' # _add-photo-G.h5'# _pix-cont-train.h5' #
stack_func(d_file, out_file, lis_z, lis_ra, lis_dec, band[0], lis_x, lis_y, id_cen, rms_file, cont_file)
'''
id_cen = 0 # BCG-stacking
out_file = '/media/xkchen/My Passport/data/SDSS/tmp_stack/jack/random_tot_BCG-stack_correct.h5'
rms_file = '/media/xkchen/My Passport/data/SDSS/tmp_stack/jack/random_tot_BCG-stack_var_correct.h5'
cont_file = '/media/xkchen/My Passport/data/SDSS/tmp_stack/jack/random_tot_BCG-stack_pix-cont_correct.h5'

#out_file = '/home/xkchen/Downloads/new_1000_test/random_test-1000_BCG-stack_test-train_correct.h5'
#rms_file = '/home/xkchen/Downloads/new_1000_test/random_test-1000_BCG-stack_Var-train_correct.h5'
#cont_file = '/home/xkchen/Downloads/new_1000_test/random_test-1000_BCG-stack_pix-cont-train_correct.h5'

#out_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_test-train_correct.h5' # _add-photo-G.h5'# _test-train.h5'
#rms_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_Var-train_correct.h5' # _add-photo-G.h5'# _Var-train.h5'
#cont_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_pix-cont-train_correct.h5' # _add-photo-G.h5'# _pix-cont-train.h5'
stack_func(d_file, out_file, lis_z, lis_ra, lis_dec, band[0], lis_x, lis_y, id_cen, rms_file, cont_file)

"""
### stacking sky img
d_file = '/media/xkchen/My Passport/data/SDSS/random_cat/sky_img/random_sky-ra%.3f-dec%.3f-z%.3f-%s-band.fits'

id_mean = 0 # stacking sky-img
id_cen = 0 # 1
out_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_stack_sky-train.h5'#center-stack_sky-train.h5'
rms_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_stack_sky_var-train.h5'#center-stack_sky_var-train.h5'
cont_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_stack_sky_pix-cont-train.h5'#center-stack_sky_pix-cont-train.h5'
sky_stack_func(d_file, out_file, z, ra, dec, band[0], clus_x, clus_y, id_cen, id_mean, rms_file, cont_file)

id_mean = 2 # stacking sky-img minus np.median(sky-img)
id_cen = 0
out_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_sky-train.h5'
rms_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_sky_var-train.h5'
cont_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_sky_pix-cont-train.h5'
sky_stack_func(d_file, out_file, z, ra, dec, band[0], clus_x, clus_y, id_cen, id_mean, rms_file, cont_file)
'''
rnx = np.zeros((10, len(z)), dtype = np.float)
rny = np.zeros((10, len(z)), dtype = np.float)
for nn in range(10):
	tmpx = np.random.choice(2048, len(z), replace = True)
	tmpy = np.random.choice(1489, len(z), replace = True)
	rnx[nn,:] = tmpx + 0.
	rny[nn,:] = tmpy + 0.
with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_rand-img_pos-x.h5', 'w') as f:
	f['a'] = np.array(rnx)
with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_rand-img_pos-y.h5', 'w') as f:
	f['a'] = np.array(rny)
'''
with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_rand-img_pos-x.h5', 'r') as f:
	rnx = np.array(f['a'])
with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_rand-img_pos-y.h5', 'r') as f:
	rny = np.array(f['a'])

for nn in range( 10 ):
	tmpx = rnx[nn,:]
	tmpy = rny[nn,:]

	id_cen = 2
	out_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_rand-stack_sky-train_%d.h5' % nn
	rms_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_rand-stack_sky_var-train_%d.h5' % nn
	cont_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_rand-stack_sky_pix-cont-train_%d.h5' % nn
	sky_stack_func(d_file, out_file, z, ra, dec, band[0], tmpx, tmpy, id_cen, id_mean, rms_file, cont_file)
"""
