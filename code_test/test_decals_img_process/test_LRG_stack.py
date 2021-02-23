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

from list_shuffle import find_unique_shuffle_lists as find_list
from img_jack_stack import jack_main_func
from light_measure import cc_grid_img, grid_img
from light_measure import light_measure_Z0_weit, light_measure_rn_Z0_weit

# cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
band = ['r', 'g', 'i', 'u', 'z']

#**********#
home = '/home/xkchen/mywork/ICL/data/tmp_img/'
load = '/media/xkchen/My Passport/data/SDSS/'

#dat = pds.read_csv('A250_LRG-img-pos.csv')
dat = pds.read_csv('A250_LRG_norm-img_pos.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

id_cen = 0
n_rbins = 100
N_bin = 30

'''
sub_img = load + '20_10_test_jack/A250-match_LRG_sub-%d_img.h5'
sub_pix_cont = load + '20_10_test_jack/A250-match_LRG_sub-%d_pix-cont.h5'
sub_sb = load + '20_10_test_jack/A250-match_LRG_sub-%d_SB-pro.h5'

### SDSS imgs
d_file = home + 'LRG_mask/sdss-img_%s_ra%.3f_dec%.3f_z%.3f.fits'

J_sub_img = load + '20_10_test_jack/sdss_A250-match_LRG_jack-sub-%d_img.h5'
J_sub_pix_cont = load + '20_10_test_jack/sdss_A250-match_LRG_jack-sub-%d_pix-cont.h5'
J_sub_sb = load + '20_10_test_jack/sdss_A250-match_LRG_jack-sub-%d_SB-pro.h5'

jack_SB_file = load + '20_10_test_jack/sdss_A250-match_LRG_Mean_jack_SB-pro.h5'
jack_img = load + '20_10_test_jack/sdss_A250-match_LRG_Mean_jack_img.h5'
jack_cont_arr = load + '20_10_test_jack/sdss_A250-match_LRG_Mean_jack_pix-cont.h5'

jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[0], sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,)

### DECaLS
d_file = home + 'LRG_mask/decals-img_%s_ra%.3f_dec%.3f_z%.3f.fits'

J_sub_img = load + '20_10_test_jack/decals_A250-match_LRG_jack-sub-%d_img.h5'
J_sub_pix_cont = load + '20_10_test_jack/decals_A250-match_LRG_jack-sub-%d_pix-cont.h5'
J_sub_sb = load + '20_10_test_jack/decals_A250-match_LRG_jack-sub-%d_SB-pro.h5'

jack_SB_file = load + '20_10_test_jack/decals_A250-match_LRG_Mean_jack_SB-pro.h5'
jack_img = load + '20_10_test_jack/decals_A250-match_LRG_Mean_jack_img.h5'
jack_cont_arr = load + '20_10_test_jack/decals_A250-match_LRG_Mean_jack_pix-cont.h5'

jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[0], sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,)
'''

with h5py.File(load + '20_10_test_jack/sdss_A250-match_LRG_Mean_jack_SB-pro.h5', 'r') as f:
	sdss_r = np.array(f['r'])
	sdss_sb = np.array(f['sb'])
	sdss_sb_err = np.array(f['sb_err'])

with h5py.File(load + '20_10_test_jack/sdss_A250-match_LRG_Mean_jack_img.h5', 'r') as f:
	sdss_img = np.array(f['a'])

id_nan = np.isnan(sdss_img)
idvx = id_nan == False
idy, idx = np.where(idvx == True)
x_low, x_up = np.min(idx), np.max(idx)
y_low, y_up = np.min(idy), np.max(idy)

dpt_sdss = sdss_img[y_low: y_up+1, x_low: x_up + 1]
sdss_patch = cc_grid_img(dpt_sdss, 100, 100)[0]

with h5py.File(load + '20_10_test_jack/decals_A250-match_LRG_Mean_jack_SB-pro.h5', 'r') as f:
	desi_r = np.array(f['r'])
	desi_sb = np.array(f['sb'])
	desi_sb_err = np.array(f['sb_err'])

with h5py.File(load + '20_10_test_jack/decals_A250-match_LRG_Mean_jack_img.h5', 'r') as f:
	desi_img = np.array(f['a'])

dpt_desi = desi_img[y_low: y_up+1, x_low: x_up + 1]
desi_patch = cc_grid_img(dpt_desi, 100, 100)[0]

diffi_img = dpt_sdss - dpt_desi
diff_block = cc_grid_img(diffi_img, 100, 100)[0]

## BCG-stacking ( DECaLS case)
with h5py.File(load + '20_10_test_jack/decals_A-250_BCG-stack_Mean_jack_SB-pro_sdss-mask.h5', 'r') as f:
	pre_desi_r = np.array(f['r'])
	pre_desi_sb = np.array(f['sb'])
	pre_desi_sb_err = np.array(f['sb_err'])

fig = plt.figure( figsize = (19.84, 4.8) )
ax0 = fig.add_axes([0.02, 0.09, 0.28, 0.85])
ax1 = fig.add_axes([0.35, 0.09, 0.28, 0.85])
ax2 = fig.add_axes([0.68, 0.09, 0.28, 0.85])

ax0.set_title('SDSS')
tg = ax0.imshow(sdss_patch / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -2e-2, vmax = 2e-2,)
cb = plt.colorbar(tg, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
cb.formatter.set_powerlimits( (0,0) )

ax1.set_title('DECaLS')
tg = ax1.imshow(desi_patch / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -2e-2, vmax = 2e-2,)
cb = plt.colorbar(tg, ax = ax1, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
cb.formatter.set_powerlimits( (0,0) )

ax2.set_title('SDSS - DECaLS')
tg = ax2.imshow(diff_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -2e-2, vmax = 2e-2,)
cb = plt.colorbar(tg, ax = ax2, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
cb.formatter.set_powerlimits( (0,0) )

plt.savefig('comapre_stacking_img.png', dpi = 300)
plt.close()


plt.figure()
ax = plt.subplot(111)
ax.set_title('LRG stacking test')

ax.plot(sdss_r, sdss_sb, ls = '-', color = 'r', alpha = 0.8, label = 'LRG [SDSS]',)
ax.fill_between(sdss_r, y1 = sdss_sb - sdss_sb_err, y2 = sdss_sb + sdss_sb_err, color = 'r', alpha = 0.2,)

ax.plot(desi_r, desi_sb, ls = '-', color = 'b', alpha = 0.8, label = 'LRG [DECaLS]',)
ax.fill_between(desi_r, y1 = desi_sb - desi_sb_err, y2 = desi_sb + desi_sb_err, color = 'b', alpha = 0.2,)

ax.plot(pre_desi_r, pre_desi_sb, ls = '-', color = 'g', alpha = 0.8, label = 'cluster [DECaLS]',)
ax.fill_between(pre_desi_r, y1 = pre_desi_sb - pre_desi_sb_err, y2 = pre_desi_sb + pre_desi_sb_err, color = 'g', alpha = 0.2,)

ax.set_ylim(1e-4, 2e-2)
ax.set_yscale('log')

ax.set_xlim(1e1, 1e3)
ax.set_xlabel('R [arcsec]')
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 1, frameon = False, fontsize = 8)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in',)

plt.savefig('grid_2D_and_SB_compare.png', dpi = 300)
plt.close()


