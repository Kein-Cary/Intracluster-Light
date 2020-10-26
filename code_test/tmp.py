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
## stacking imgs at reference redshift
dat = pds.read_csv('test_1000-to-AB_resamp_BCG-pos.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

set_ra = ra[:250]
set_dec = dec[:250]
set_z = z[:250]
set_x, set_y = clus_x[:250], clus_y[:250]

id_cen = 0
n_rbins = 100
N_bin = 25

size_arr = np.array([5, 30])
'''
for mm in range(2):
	if mm == 0:
		d_file = home + '20_10_test/resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'
	if mm == 1:
		d_file = home + 'tmp_stack/pix_resample/resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'

	sub_img = load + '20_10_test/jack_test/A_clust_BCG-stack_sub-%d_img' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])
	sub_pix_cont = load + '20_10_test/jack_test/A_clust_BCG-stack_sub-%d_pix-cont' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])
	sub_sb = load + '20_10_test/jack_test/A_clust_BCG-stack_sub-%d_SB-pro' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])

	J_sub_img = load + '20_10_test/jack_test/A_clust_BCG-stack_jack-sub-%d_img' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])
	J_sub_pix_cont = load + '20_10_test/jack_test/A_clust_BCG-stack_jack-sub-%d_pix-cont' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])
	J_sub_sb = load + '20_10_test/jack_test/A_clust_BCG-stack_jack-sub-%d_SB-pro' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])

	jack_SB_file = load + '20_10_test/jack_test/A_clust_BCG-stack_Mean_jack_SB-pro' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])
	jack_img = load + '20_10_test/jack_test/A_clust_BCG-stack_Mean_jack_img' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])
	jack_cont_arr = load + '20_10_test/jack_test/A_clust_BCG-stack_Mean_jack_pix-cont' + '_%d-FWHM-ov2_z-ref.h5' % (size_arr[mm])

	#jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[0], sub_img,
	#	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
	#	id_cut = False, N_edg = None, id_Z0 = False, z_ref = 0.25,)

	jack_main_func(id_cen, N_bin, n_rbins, set_ra, set_dec, set_z, set_x, set_y, d_file, band[0], sub_img,
		sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
		id_cut = False, N_edg = None, id_Z0 = False, z_ref = 0.25,)
'''
print('obs. finish!')
'''
load = '/home/xkchen/mywork/ICL/data/tmp_img/'
d_file = load + 'resamp_mock/resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'

sub_img = load + 'stack_mock/mock-A_BCG-stack_sub-%d_img_z-ref.h5'
sub_pix_cont = load + 'stack_mock/mock-A_BCG-stack_sub-%d_pix-cont_z-ref.h5'
sub_sb = load + 'stack_mock/mock-A_BCG-stack_sub-%d_SB-pro_z-ref.h5'

J_sub_img = load + 'stack_mock/mock-A_BCG-stack_jack-sub-%d_img_z-ref.h5'
J_sub_pix_cont = load + 'stack_mock/mock-A_BCG-stack_jack-sub-%d_pix-cont_z-ref.h5'
J_sub_sb = load + 'stack_mock/mock-A_BCG-stack_jack-sub-%d_SB-pro_z-ref.h5'

jack_SB_file = load + 'stack_mock/mock-A_BCG-stack_Mean_jack_SB-pro_z-ref.h5'
jack_img = load + 'stack_mock/mock-A_BCG-stack_Mean_jack_img_z-ref.h5'
jack_cont_arr = load + 'stack_mock/mock-A_BCG-stack_Mean_jack_pix-cont_z-ref.h5'

z_ref = 0.254 # mean of the z

jack_main_func(id_cen, N_bin, n_rbins, set_ra, set_dec, set_z, set_x, set_y, d_file, band[0], sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
	id_cut = False, N_edg = None, id_Z0 = False, z_ref = 0.254,)
'''
print('mock finish!')

'''
## test for star masking size
dat = pds.read_csv('/home/xkchen/mywork/ICL/code/SEX/result/test_1000-to-250_cat.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

Bdat = pds.read_csv('/home/xkchen/mywork/ICL/code/SEX/result/test_1000-to-98_cat.csv')
Bra, Bdec, Bz = np.array(Bdat.ra), np.array(Bdat.dec), np.array(Bdat.z)
Bclus_x, Bclus_y = np.array(Bdat.bcg_x), np.array(Bdat.bcg_y)

ra = np.r_[ ra, Bra ]
dec = np.r_[ dec, Bdec ]
z = np.r_[ z, Bz ]
clus_x = np.r_[ clus_x, Bclus_x ]
clus_y = np.r_[ clus_y, Bclus_y ]

size_arr = np.array([5, 10, 15, 20, 25])
for mm in range(5):

	id_cen = 0
	n_rbins = 110
	N_bin = 30
	d_file = home + '20_10_test/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_' + '%d-FWHM-ov2.fits' % (size_arr[mm])

	sub_img = load + '20_10_test/jack_test/AB_clust_BCG-stack_sub-%d_img' + '_%d-FWHM-ov2.h5' % (size_arr[mm])
	sub_pix_cont = load + '20_10_test/jack_test/AB_clust_BCG-stack_sub-%d_pix-cont' + '_%d-FWHM-ov2.h5' % (size_arr[mm])
	sub_sb = load + '20_10_test/jack_test/AB_clust_BCG-stack_sub-%d_SB-pro' + '_%d-FWHM-ov2.h5' % (size_arr[mm])

	J_sub_img = load + '20_10_test/jack_test/AB_clust_BCG-stack_jack-sub-%d_img' + '_%d-FWHM-ov2.h5' % (size_arr[mm])
	J_sub_pix_cont = load + '20_10_test/jack_test/AB_clust_BCG-stack_jack-sub-%d_pix-cont' + '_%d-FWHM-ov2.h5' % (size_arr[mm])
	J_sub_sb = load + '20_10_test/jack_test/AB_clust_BCG-stack_jack-sub-%d_SB-pro' + '_%d-FWHM-ov2.h5' % (size_arr[mm])

	jack_SB_file = load + '20_10_test/jack_test/AB_clust_BCG-stack_Mean_jack_SB-pro' + '_%d-FWHM-ov2.h5' % (size_arr[mm])
	jack_img = load + '20_10_test/jack_test/AB_clust_BCG-stack_Mean_jack_img' + '_%d-FWHM-ov2.h5' % (size_arr[mm])
	jack_cont_arr = load + '20_10_test/jack_test/AB_clust_BCG-stack_Mean_jack_pix-cont' + '_%d-FWHM-ov2.h5' % (size_arr[mm])

	jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[0], sub_img,
		sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,)

print('start point!')
id_cen = 0
n_rbins = 110
N_bin = 30
d_file = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits' ## 30 (FWHM/2) case

sub_img = load + '20_10_test/jack_test/AB_clust_BCG-stack_sub-%d_img_30-FWHM-ov2.h5'
sub_pix_cont = load + '20_10_test/jack_test/AB_clust_BCG-stack_sub-%d_pix-cont_30-FWHM-ov2.h5'
sub_sb = load + '20_10_test/jack_test/AB_clust_BCG-stack_sub-%d_SB-pro_30-FWHM-ov2.h5'

J_sub_img = load + '20_10_test/jack_test/AB_clust_BCG-stack_jack-sub-%d_img_30-FWHM-ov2.h5'
J_sub_pix_cont = load + '20_10_test/jack_test/AB_clust_BCG-stack_jack-sub-%d_pix-cont_30-FWHM-ov2.h5'
J_sub_sb = load + '20_10_test/jack_test/AB_clust_BCG-stack_jack-sub-%d_SB-pro_30-FWHM-ov2.h5'

jack_SB_file = load + '20_10_test/jack_test/AB_clust_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2.h5'
jack_img = load + '20_10_test/jack_test/AB_clust_BCG-stack_Mean_jack_img_30-FWHM-ov2.h5'
jack_cont_arr = load + '20_10_test/jack_test/AB_clust_BCG-stack_Mean_jack_pix-cont_30-FWHM-ov2.h5'

jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[0], sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,)
raise
'''
'''
## 30 (FWHM/2) case, cut img edge pixels test
id_cen = 0
n_rbins = 110
N_bin = 30
d_file = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'

dat = pds.read_csv('/home/xkchen/mywork/ICL/code/SEX/result/test_1000-to-250_cat.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

N_cut = np.array([200, 500])

for ll in range(3):

	sub_img = load + '20_10_test/jack_test/A_clust_BCG-stack_sub-%d_img_30-FWHM-ov2' + '_cut-%d.h5' % N_cut[ll]
	sub_pix_cont = load + '20_10_test/jack_test/A_clust_BCG-stack_sub-%d_pix-cont_30-FWHM-ov2' + '_cut-%d.h5' % N_cut[ll]
	sub_sb = load + '20_10_test/jack_test/A_clust_BCG-stack_sub-%d_SB-pro_30-FWHM-ov2' + '_cut-%d.h5' % N_cut[ll]

	J_sub_img = load + '20_10_test/jack_test/A_clust_BCG-stack_jack-sub-%d_img_30-FWHM-ov2' + '_cut-%d.h5' % N_cut[ll]
	J_sub_pix_cont = load + '20_10_test/jack_test/A_clust_BCG-stack_jack-sub-%d_pix-cont_30-FWHM-ov2' + '_cut-%d.h5' % N_cut[ll]
	J_sub_sb = load + '20_10_test/jack_test/A_clust_BCG-stack_jack-sub-%d_SB-pro_30-FWHM-ov2' + '_cut-%d.h5' % N_cut[ll]

	jack_SB_file = load + '20_10_test/jack_test/A_clust_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2' + '_cut-%d.h5' % N_cut[ll]
	jack_img = load + '20_10_test/jack_test/A_clust_BCG-stack_Mean_jack_img_30-FWHM-ov2' + '_cut-%d.h5' % N_cut[ll]
	jack_cont_arr = load + '20_10_test/jack_test/A_clust_BCG-stack_Mean_jack_pix-cont_30-FWHM-ov2' + '_cut-%d.h5' % N_cut[ll]

	jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[0], sub_img,
		sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
		id_cut = True, N_edg = N_cut[ll],)
raise
'''

##### decals imgs
print('DECaLS')

dat = pds.read_csv('/home/xkchen/mywork/ICL/code/SEX/result/test_1000-to-250_cat.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

id_cen = 0
n_rbins = 110
N_bin = 30

d_file = '/media/xkchen/My Passport/data/BASS/A_250_mask/ap_sdss_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
#d_file = '/media/xkchen/My Passport/data/BASS/A_250_mask/comb_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

sub_img = load + '20_10_test/jack_test/decals_A-250_BCG-stack_sub-%d_img.h5'
sub_pix_cont = load + '20_10_test/jack_test/decals_A-250_BCG-stack_sub-%d_pix-cont.h5'
sub_sb = load + '20_10_test/jack_test/decals_A-250_BCG-stack_sub-%d_SB-pro.h5'
'''
J_sub_img = load + '20_10_test/jack_test/decals_A-250_BCG-stack_jack-sub-%d_img.h5'
J_sub_pix_cont = load + '20_10_test/jack_test/decals_A-250_BCG-stack_jack-sub-%d_pix-cont.h5'
J_sub_sb = load + '20_10_test/jack_test/decals_A-250_BCG-stack_jack-sub-%d_SB-pro.h5'

jack_SB_file = load + '20_10_test/jack_test/decals_A-250_BCG-stack_Mean_jack_SB-pro.h5'
jack_img = load + '20_10_test/jack_test/decals_A-250_BCG-stack_Mean_jack_img.h5'
jack_cont_arr = load + '20_10_test/jack_test/decals_A-250_BCG-stack_Mean_jack_pix-cont.h5'
'''
J_sub_img = load + '20_10_test/jack_test/decals_A-250_BCG-stack_jack-sub-%d_img_sdss-mask.h5'
J_sub_pix_cont = load + '20_10_test/jack_test/decals_A-250_BCG-stack_jack-sub-%d_pix-cont_sdss-mask.h5'
J_sub_sb = load + '20_10_test/jack_test/decals_A-250_BCG-stack_jack-sub-%d_SB-pro_sdss-mask.h5'

jack_SB_file = load + '20_10_test/jack_test/decals_A-250_BCG-stack_Mean_jack_SB-pro_sdss-mask.h5'
jack_img = load + '20_10_test/jack_test/decals_A-250_BCG-stack_Mean_jack_img_sdss-mask.h5'
jack_cont_arr = load + '20_10_test/jack_test/decals_A-250_BCG-stack_Mean_jack_pix-cont_sdss-mask.h5'

jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[0], sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,)

raise

def cov_MX_func(radius, pros, id_jack = True,):

	flux_array = np.array(pros)
	r_array = np.array(radius)
	Nt = len(flux_array)

	r_min, r_max = 0, np.nanmax(r_array)
	for ll in range(Nt):
		idnn = np.isnan( flux_array[ll] )
		lim_in_r = r_array[ll][ idnn == False]
		r_min = np.max([ r_min, np.min(lim_in_r) ])
		r_max = np.min([ r_max, np.max(lim_in_r) ])

	SB_value = []
	R_value = []
	for ll in range(Nt):
		idux = (r_array[ll] >= r_min) & (r_array[ll] <= r_max)
		set_flux = flux_array[ll][ idux ]
		set_r = r_array[ll][ idux ]
		SB_value.append( set_flux )
		R_value.append( set_r )

	SB_value = np.array(SB_value)
	R_value = np.array(R_value)
	R_mean = np.nanmean(R_value, axis = 0)

	mean_lit = np.nanmean(SB_value, axis = 0)
	std_lit = np.nanstd(SB_value, axis = 0)
	nx, ny = SB_value.shape[1], SB_value.shape[0]

	cov_tt = np.zeros((nx, nx), dtype = np.float)
	cor_tt = np.zeros((nx, nx), dtype = np.float)

	for qq in range(nx):
		for tt in range(nx):
			cov_tt[qq, tt] = np.sum( (SB_value[:,qq] - mean_lit[qq]) * (SB_value[:,tt] - mean_lit[tt]) ) / ny

	for qq in range(nx):
		for tt in range(nx):
			cor_tt[qq, tt] = cov_tt[qq, tt] / (std_lit[qq] * std_lit[tt])
	if id_jack == True:
		cov_MX = cov_tt * (ny - 1.) ## jackknife factor
	else:
		cov_MX = cov_tt * 1.
	cor_MX = cor_tt * 1.

	return R_mean, cov_MX, cor_MX

## covariance & correlation
#load = '/media/xkchen/My Passport/data/SDSS/'
#J_sub_sb = load + '20_10_test/jack_test/clust_BCG-stack_jack-sub-%d_SB-pro_30-FWHM-ov2.h5'
#J_sub_sb = load + '20_10_test/jack_test/A_clust_BCG-stack_jack-sub-%d_SB-pro_30-FWHM-ov2_z-ref.h5'

load = '/home/xkchen/mywork/ICL/data/tmp_img/'
#J_sub_sb = load + 'stack_mock/mock-A_BCG-stack_jack-sub-%d_SB-pro.h5'
J_sub_sb = load + 'stack_mock/mock-A_BCG-stack_jack-sub-%d_SB-pro_z-ref.h5'

tt_r, tt_sb = [], []

for mm in range(25):#30):
	with h5py.File(J_sub_sb % mm, 'r') as f:
		r_arr = np.array(f['r'])
		sb_arr = np.array(f['sb'])
		nratio = np.array(f['nratio'])
		npix = np.array(f['npix'])

	idnun = npix < 1.
	r_arr[idnun] = np.nan
	sb_arr[idnun] = np.nan

	tt_r.append(r_arr)
	tt_sb.append(sb_arr)

R_mean, cov_Mx, cor_Mx = cov_MX_func(tt_r, tt_sb,)

fig = plt.figure( figsize = (13.12, 4.8) )
ax0 = fig.add_axes([0.05, 0.09, 0.40, 0.85])
ax1 = fig.add_axes([0.55, 0.09, 0.40, 0.85])

#ticks = [0, 20, 40, 60, 65,]
#ticks = [0, 10, 20, 30, 40, 50, 55]
ticks = [0, 20, 40, 60, 65, 70, 80]

ax0.set_title('Covariance Matrix')
tf = ax0.imshow(cov_Mx, origin = 'lower', cmap = 'rainbow', vmin = 1e-8, vmax = 2e-3, norm = mpl.colors.LogNorm(),)
cbr = plt.colorbar(tf, ax = ax0, fraction = 0.040, pad = 0.01, label = '$ [nanomaggies \, / \, arcsec^2]^2 $')
ax0.set_xticks(ticks)
ax0.set_xticklabels([],)
ax0.set_yticks(ticks)
ax0.set_yticklabels(['%.1f' % ll for ll in R_mean[ticks] ])
ax0.set_xlabel('R [kpc]')
ax0.set_ylabel('R [kpc]')

ax1.set_title('Correlation Matrix')
tg = ax1.imshow(cor_Mx, origin = 'lower', cmap = 'seismic', vmin = -1, vmax = 1,)
plt.colorbar(tg, ax = ax1, fraction = 0.040, pad = 0.01,)
ax1.set_xticks(ticks)
ax1.set_xticklabels([],)
ax1.set_yticks(ticks)
ax1.set_yticklabels(['%.1f' % ll for ll in R_mean[ticks] ])
'''
ax1.set_xlabel('R [arcsec]')
ax1.set_ylabel('R [arcsec]')
'''
ax1.set_xlabel('R [kpc]')
ax1.set_ylabel('R [kpc]')

plt.subplots_adjust(left = 0.05, right = 0.95,)
plt.savefig('cov_Mx_test.png', dpi = 300)
plt.close()


raise

