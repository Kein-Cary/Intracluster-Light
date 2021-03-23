import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.io.ascii as asc

import subprocess as subpro
import scipy.stats as sts
import astropy.units as U
import astropy.constants as C
from scipy.optimize import curve_fit
from astropy import cosmology as apcy

from img_jack_stack import jack_main_func
from light_measure import light_measure_Z0_weit
from light_measure import jack_SB_func
from fig_out_module import cc_grid_img, grid_img

rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
band = ['r', 'g', 'i', 'u', 'z']
mag_add = np.array([0, 0, 0, -0.04, 0.02])

#****************************#
from cc_block_select import diffuse_identi_func
#mu_sigm_file = '/home/xkchen/mywork/ICL/code/SEX/result/img_test-1000_mean_sigm.csv'
mu_sigm_file = '/home/xkchen/mywork/ICL/code/SEX/result/img_A-250_mean_sigm.csv'

home = '/media/xkchen/My Passport/data/SDSS/'
d_file = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'

thres_S0, thres_S1 = 3, 5
sigma = np.array([3.5, 4, 4.5, 5, 5.5, 6,])
n_main = np.array([250, 98, 193, 459])
'''
### Bro-mode selection for different sigma
for kk in range( 4 ):

	dat = pds.read_csv('SEX/result/test_1000-to-%d_cat.csv' % n_main[kk],)
	set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

	for mm in range( 6 ):

		#'SEX/result/differ_sigma_with_T1000/'

		rule_file = 'SEX/result/differ_sigma_with_A250/Bro-mode-select_1000-to-%d_rule-out_cat_%.1f-sigma.csv' % ( n_main[kk], sigma[mm] )
		remain_file = 'SEX/result/differ_sigma_with_A250/Bro-mode-select_1000-to-%d_remain_cat_%.1f-sigma.csv' % ( n_main[kk], sigma[mm] )
		diffuse_identi_func(band[0], set_ra, set_dec, set_z, d_file, rule_file, remain_file, thres_S0, thres_S1, sigma[mm],
							mu_sigm_file, id_single = False, id_mode = True,)
'''
tmp_ra = []
tmp_eta = []

for mm in range( 6 ):

	tt_ra = np.array([])
	tt_eta = np.array([])

	for kk in range( 4 ):

		dat = pds.read_csv('SEX/result/test_1000-to-%d_cat.csv' % (n_main[kk]) )
		set_ra = np.array(dat.ra)

		##### select based on T1000
		pre_dat = pds.read_csv('SEX/result/differ_sigma_with_T1000/Bro-mode-select_1000-to-%d_remain_cat_%.1f-sigma.csv' % ( n_main[kk], sigma[mm] ),)
		#pre_dat = pds.read_csv('SEX/result/differ_sigma_with_A250/Bro-mode-select_1000-to-%d_remain_cat_%.1f-sigma.csv' % ( n_main[kk], sigma[mm] ),)
		pre_ra = np.array(pre_dat.ra)
		eta = pre_ra.shape[0] / set_ra.shape[0]

		tt_eta = np.r_[ tt_eta, eta ]
		tt_ra = np.r_[ tt_ra, np.array(pre_ra) ]

	tmp_ra.append( tt_ra )
	tmp_eta.append( tt_eta )

xpot = np.array([1,2,3,4])
Len = [len(ll) for ll in tmp_ra]

plt.figure()
ax=  plt.subplot(111)
ax.set_title('T1000 img selection $[Mode(\\sigma), Mode(\\mu)]$')
#ax.set_title('A250 img selection $[Mode(\\sigma), Mode(\\mu)]$')

for mm in range( 6 ):

	ax.plot(xpot, tmp_eta[mm], '*', color = mpl.cm.rainbow(mm / 5), alpha = 0.5, label = '%.1f $ \\sigma $' % sigma[mm],)

ax.legend(loc = 3, frameon = False,)
ax.set_ylim(-0.1, 1.1)
ax.set_xlim(0.5, 4.5)
ax.set_xticks(xpot)
ax.set_xticklabels(['A', 'B', 'C', 'D'])
ax.set_xlabel('sample type')
ax.set_ylabel('$ N_{normal} / N_{total} $')

plt.savefig('img_select_compare_T1000.png', dpi = 300)
#plt.savefig('img_select_compare_A250.png', dpi = 300)
plt.close()

tmp_A_eta = []
tmp_tot_eta = []

for mm in range( 6 ):

	dat0 = pds.read_csv('SEX/result/differ_sigma_with_T1000/T1000_Bro-mode-select_BCG-pos_%.1f.csv' % sigma[mm],)
	ra0 = np.array(dat0['ra'])

	tmp_tot_eta.append( len(ra0) / 1000 )

	dat1 = pds.read_csv('SEX/result/differ_sigma_with_A250/A250_Bro-mode-select_BCG-pos_%.1f.csv' % sigma[mm],)
	ra1 = np.array(dat1['ra'])

	tmp_A_eta.append( len(ra1) / 1000 )

plt.figure()
plt.plot(sigma, tmp_A_eta, 'b*-', alpha = 0.5, label = 'selected by A250',)
plt.plot(sigma, tmp_tot_eta, 'ro-', alpha = 0.5, label = 'selected by T1000',)
plt.ylim(0.4, 0.75)
plt.grid( axis = 'both', which = 'both', alpha = 0.1,)
plt.xlabel('$\\sigma$')
plt.ylabel('Cluster Number')
plt.legend( loc = 4,)
plt.savefig('selected_number_compare.png', dpi = 100)
plt.close()


raise

## stacking test
load = '/media/xkchen/My Passport/data/SDSS/'

id_cen = 0
n_rbins = 100
N_bin = 30

"""
## on physical coordinate
for mm in range( 6 ):

	#dat = pds.read_csv('SEX/result/differ_sigma_with_T1000/T1000_Bro-mode-select_resamp_BCG-pos_%.1f.csv' %  sigma[mm] )
	dat = pds.read_csv('SEX/result/differ_sigma_with_A250/A250_Bro-mode-select_resamp_BCG-pos_%.1f.csv' %  sigma[mm] )

	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	sub_img = load + '20_10_test_jack/T1000_select-test_sub-%d_img_diffi-sigma.h5'
	sub_pix_cont = load + '20_10_test_jack/T1000_select-test_sub-%d_pix-cont_diffi-sigma.h5'
	sub_sb = load + '20_10_test_jack/T1000_select-test_sub-%d_SB-pro_diffi-sigma.h5'

	d_file = home + 'tmp_stack/pix_resample/resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'
	'''
	## selected by tot-1000
	J_sub_img = load + '20_10_test_jack/Bro-mode-select_jack-sub-%d_img_selected-by-tot' + '_%.1f-sigma_z-ref.h5' % sigma[mm]
	J_sub_pix_cont = load + '20_10_test_jack/Bro-mode-select_jack-sub-%d_pix-cont_selected-by-tot' + '_%.1f-sigma_z-ref.h5' % sigma[mm]
	J_sub_sb = load + '20_10_test_jack/Bro-mode-select_jack-sub-%d_SB-pro_selected-by-tot' + '_%.1f-sigma_z-ref.h5' % sigma[mm]

	jack_SB_file = load + '20_10_test_jack/Bro-mode-select_Mean_jack_SB-pro_selected-by-tot' + '_%.1f-sigma_z-ref.h5' % sigma[mm]
	jack_img = load + '20_10_test_jack/Bro-mode-select_Mean_jack_img_selected-by-tot' + '_%.1f-sigma_z-ref.h5' % sigma[mm]
	jack_cont_arr = load + '20_10_test_jack/Bro-mode-select_Mean_jack_pix-cont_selected-by-tot' + '_%.1f-sigma_z-ref.h5' % sigma[mm]
	'''
	## selected by A250
	J_sub_img = load + '20_10_test_jack/Bro-mode-select_jack-sub-%d_img' + '_%.1f-sigma_z-ref.h5' % sigma[mm]
	J_sub_pix_cont = load + '20_10_test_jack/Bro-mode-select_jack-sub-%d_pix-cont' + '_%.1f-sigma_z-ref.h5' % sigma[mm]
	J_sub_sb = load + '20_10_test_jack/Bro-mode-select_jack-sub-%d_SB-pro' + '_%.1f-sigma_z-ref.h5' % sigma[mm]

	jack_SB_file = load + '20_10_test_jack/Bro-mode-select_Mean_jack_SB-pro' + '_%.1f-sigma_z-ref.h5' % sigma[mm]
	jack_img = load + '20_10_test_jack/Bro-mode-select_Mean_jack_img' + '_%.1f-sigma_z-ref.h5' % sigma[mm]
	jack_cont_arr = load + '20_10_test_jack/Bro-mode-select_Mean_jack_pix-cont' + '_%.1f-sigma_z-ref.h5' % sigma[mm]

	jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[0], sub_img,
		sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
		id_cut = False, N_edg = None, id_Z0 = False, z_ref = 0.25, id_S2N = True, S2N = 5,)

print('to here 2')
"""

### SB profile compare
load = '/media/xkchen/My Passport/data/SDSS/'

## Zibetti et al.,2005
SB_tt = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/r_band_BCG_ICL.csv')
R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
R_obs = R_obs**4
flux_obs = 10**( (22.5 - SB_obs + 2.5 * np.log10(pixel**2) ) / 2.5 ) / pixel**2

"""
phy_r, phy_sb, phy_sb_err = [], [], []

for kk in range( 6 ):

	## at z_ref
	with h5py.File( load + '20_10_test_jack/Bro-mode-select_Mean_jack_SB-pro_selected-by-tot_%.1f-sigma_z-ref.h5' % sigma[kk], 'r') as f:
	#with h5py.File( load + '20_10_test_jack/Bro-mode-select_Mean_jack_SB-pro_%.1f-sigma_z-ref.h5' % sigma[kk], 'r') as f:
		c_r_arr = np.array(f['r'])
		c_sb_arr = np.array(f['sb'])
		c_sb_err = np.array(f['sb_err'])

	phy_r.append(c_r_arr)
	phy_sb.append(c_sb_arr)
	phy_sb_err.append(c_sb_err)

	with h5py.File( load + '20_10_test_jack/Bro-mode-select_Mean_jack_img_selected-by-tot_%.1f-sigma_z-ref.h5' % sigma[kk], 'r') as f:
	#with h5py.File( load + '20_10_test_jack/Bro-mode-select_Mean_jack_img_%.1f-sigma_z-ref.h5' % sigma[kk], 'r') as f:
		tt_img = np.array(f['a'])

	id_nan = np.isnan(tt_img)
	idvx = id_nan == False
	idy, idx = np.where(idvx == True)
	x_low, x_up = np.min(idx), np.max(idx)
	y_low, y_up = np.min(idy), np.max(idy)

	dpt_img = tt_img[y_low: y_up+1, x_low: x_up + 1]
	img_block = cc_grid_img(dpt_img, 100, 100,)[0]
	'''
	fig = plt.figure( figsize = (13.12, 4.8) )
	ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
	ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

	ax0.set_title('%.1f $\\sigma$ [stacking at $ z_{ref} $]' % sigma[kk],)
	tg = ax0.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits( (0,0) )

	ax1.plot(c_r_arr, c_sb_arr, ls = '-', color = 'r', alpha = 0.8, label = '%.1f $\\sigma$' % sigma[kk],)
	ax1.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'r', alpha = 0.2,)

	ax1.plot(alt_A250_r, alt_A250_sb, ls = '-', color = 'k', alpha = 0.8, label = 'A-250',)
	ax1.fill_between(alt_A250_r, y1 = alt_A250_sb - alt_A250_sb_err, y2 = alt_A250_sb + alt_A250_sb_err, color = 'k', alpha = 0.2,)

	ax1.plot(alt_tot_r, alt_tot_sb, ls = '-', color = 'c', alpha = 0.8, label = 'tot-1000',)
	ax1.fill_between(alt_tot_r, y1 = alt_tot_sb - alt_tot_sb_err, y2 = alt_tot_sb + alt_tot_sb_err, color = 'c', alpha = 0.2,)

	ax1.set_ylim(1e-3, 3e-2)
	ax1.set_yscale('log')
	ax1.set_xlim(5e1, 4e3)
	ax1.set_xlabel('R [kpc]')
	ax1.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax1.set_xscale('log')
	ax1.legend(loc = 3, frameon = False,)
	ax1.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax1.tick_params(axis = 'both', which = 'both', direction = 'in',)
	tick_form = mpl.ticker.LogFormatter(labelOnlyBase = False)
	ax1.get_yaxis().set_minor_formatter(tick_form)

	plt.savefig('2D-grd_SB_%.1f-sigma_selected_z-ref.png' % sigma[kk], dpi = 300)
	plt.close()
	'''

plt.figure()
ax = plt.subplot(111)
ax.set_title('selected by tot-1000')
#ax.set_title('selected by A250')

for kk in range( 6 ):

	ax.plot(phy_r[kk], phy_sb[kk], ls = '-', color = mpl.cm.plasma( kk / 6), alpha = 0.8, label = '%.1f $\\sigma$' % sigma[kk],)
	#ax.fill_between(phy_r[kk], y1 = phy_sb[kk] - phy_sb_err[kk], 
	#	y2 = phy_sb[kk] + phy_sb_err[kk], color = mpl.cm.plasma( kk / 6), alpha = 0.2,)

	idr = phy_r[kk] > 1e3
	idsb = np.nanmin( phy_sb[kk][idr] )
	devi_sb = phy_sb[kk] - idsb

	#ax.axhline(y = idsb, ls = ':', color = mpl.cm.plasma( kk / 6), alpha = 0.5,)
	ax.plot(phy_r[kk], devi_sb, ls = '--', color = mpl.cm.plasma( kk / 6), alpha = 0.8,)
	#ax.fill_between(phy_r[kk], y1 = devi_sb - phy_sb[kk], y2 = devi_sb + phy_sb[kk], color = mpl.cm.plasma( kk / 6), alpha = 0.2,)

ax.plot(R_obs, flux_obs, ls = '-.', color = 'g', alpha = 0.5, label = 'Z05',)

ax.plot(alt_A250_r, alt_A250_sb, ls = '-', color = 'k', alpha = 0.8, label = 'A-250',)
ax.fill_between(alt_A250_r, y1 = alt_A250_sb - alt_A250_sb_err, 
	y2 = alt_A250_sb + alt_A250_sb_err, color = 'k', alpha = 0.2,)

idr = np.abs(alt_A250_r - 2e3)
idrx = np.where(idr == idr.min() )[0][0]
idsb = np.nanmin( alt_A250_sb[idrx] )
devi_sb = alt_A250_sb - idsb

#ax.axhline(y = idsb, ls = ':', color = 'k', alpha = 0.5,)
ax.plot(alt_A250_r, devi_sb, ls = '--', color = 'k', alpha = 0.8,)
ax.fill_between(alt_A250_r, y1 = devi_sb - alt_A250_sb_err, y2 = devi_sb + alt_A250_sb_err, color = 'k', alpha = 0.2,)


ax.plot(alt_tot_r, alt_tot_sb, ls = '-', color = 'c', alpha = 0.8, label = 'tot-1000',)
ax.fill_between(alt_tot_r, y1 = alt_tot_sb - alt_tot_sb_err, 
	y2 = alt_tot_sb + alt_tot_sb_err, color = 'c', alpha = 0.2,)

idr = alt_tot_r > 1e3
idsb = np.nanmin( alt_tot_sb[idr] )
devi_sb = alt_tot_sb - idsb

#ax.axhline(y = idsb, ls = ':', color = 'c', alpha = 0.5,)
ax.plot(alt_tot_r, devi_sb, ls = '--', color = 'c', alpha = 0.8,)
ax.fill_between(alt_tot_r, y1 = devi_sb - alt_tot_sb_err, y2 = devi_sb + alt_tot_sb_err, color = 'c', alpha = 0.2,)

ax.set_xlim(5e1, 4e3)
ax.set_xlabel('R [kpc]')

ax.set_ylim(1e-4, 3e-2)
ax.set_yscale('log')
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 3, frameon = False, fontsize = 9)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in',)

plt.savefig('T1000_sigma-selected_SB-compare_z-ref.png', dpi = 300)
plt.close()
"""

##### compare different mask size (for normal stars)
size_arr = np.array([10, 20, 30])

tt_sigma = np.array([4, 5, 6])

tot_sb_pros = ['/home/xkchen/jupyter/stack/alt_T1000-tot_Mean_jack_SB-pro_10-FWHM-ov2_z-ref.h5',
				'/home/xkchen/jupyter/stack/alt_T1000-tot_Mean_jack_SB-pro_20-FWHM-ov2_z-ref.h5',
				'tmp_test/T1000-tot_R-bin_SB_test.h5']

A250_sb_pros = ['/home/xkchen/jupyter/stack/alt_T1000-to-A250_Mean_jack_SB-pro_10-FWHM-ov2_z-ref.h5',
				'/home/xkchen/jupyter/stack/alt_T1000-to-A250_Mean_jack_SB-pro_20-FWHM-ov2_z-ref.h5',
				'tmp_test/A250_R-bin_SB_test.h5']
'''
tot_sb_pros = [	'/home/xkchen/jupyter/stack/T1000-tot_Mean_jack_SB-pro_10-FWHM-ov2_z-ref.h5',
				'/home/xkchen/jupyter/stack/T1000-tot_Mean_jack_SB-pro_20-FWHM-ov2_z-ref.h5',
				'/home/xkchen/jupyter/stack/T1000-tot_Mean_jack_SB-pro_30-FWHM-ov2_z-ref.h5']

A250_sb_pros = ['/home/xkchen/jupyter/stack/T1000-to-A250_Mean_jack_SB-pro_10-FWHM-ov2_z-ref.h5',
				'/home/xkchen/jupyter/stack/T1000-to-A250_Mean_jack_SB-pro_20-FWHM-ov2_z-ref.h5',
				'/home/xkchen/jupyter/stack/T1000-to-A250_Mean_jack_SB-pro_30-FWHM-ov2_z-ref.h5']
'''
R_set, SB_set, err_set = [], [], []
tot_R, tot_SB, tot_err = [], [], []
A_R, A_SB, A_err = [], [], []

for kk in range( 3 ):

	with h5py.File( tot_sb_pros[kk], 'r') as f:
		alt_tot_r = np.array(f['r'])
		alt_tot_sb = np.array(f['sb'])
		alt_tot_sb_err = np.array(f['sb_err'])

		idnn = np.isnan(alt_tot_sb)
		idNul = alt_tot_r > 0
		idv = (idnn == False) & idNul
		alt_tot_r, alt_tot_sb_err, alt_tot_sb = alt_tot_r[idv], alt_tot_sb_err[idv], alt_tot_sb[idv]

	tot_R.append( alt_tot_r)
	tot_SB.append( alt_tot_sb)
	tot_err.append( alt_tot_sb_err)

	with h5py.File( A250_sb_pros[kk], 'r') as f:
		alt_A250_r = np.array(f['r'])
		alt_A250_sb = np.array(f['sb'])
		alt_A250_sb_err = np.array(f['sb_err'])

		idnn = np.isnan(alt_A250_sb)
		idNul = alt_A250_r > 0
		idv = (idnn == False) & idNul
		alt_A250_r, alt_A250_sb_err, alt_A250_sb = alt_A250_r[idv], alt_A250_sb_err[idv], alt_A250_sb[idv]

	A_R.append( alt_A250_r)
	A_SB.append( alt_A250_sb)
	A_err.append( alt_A250_sb_err)

	tmp_r, tmp_sb, tmp_err = [], [], []

	for mm in range( 3 ):

		if kk == 2:

			with h5py.File('/home/xkchen/jupyter/stack/Bro-mode-select_Mean_jack_SB-pro_selected-by-tot_%.1f-sigma_z-ref.h5' % tt_sigma[mm], 'r') as f:
				c_r_arr = np.array(f['r'])
				c_sb_arr = np.array(f['sb'])
				c_sb_err = np.array(f['sb_err'])

			idnn = np.isnan(c_sb_arr)
			idNul = c_r_arr > 0
			idv = (idnn == False) & idNul
			c_r_arr, c_sb_arr, c_sb_err = c_r_arr[idv], c_sb_arr[idv], c_sb_err[idv]

			tmp_r.append(c_r_arr)
			tmp_sb.append(c_sb_arr)
			tmp_err.append(c_sb_err)

			with h5py.File(
			load + '20_10_test_jack/Bro-mode-select_Mean_jack_img_selected-by-tot_%.1f-sigma_z-ref.h5' % tt_sigma[mm], 'r') as f:
				tt_img = np.array(f['a'])

		else:
			with h5py.File(
			'/home/xkchen/jupyter/stack/Bro-mode-select_Mean_jack_img_%.1f-sigma_%d-FWHM-ov2_z-ref.h5' % (tt_sigma[mm], size_arr[kk]), 'r') as f:
				tt_img = np.array(f['a'])

			#with h5py.File('/home/xkchen/jupyter/stack/Bro-mode-select_Mean_jack_SB-pro_%.1f-sigma_%d-FWHM-ov2_z-ref.h5' % (tt_sigma[mm], size_arr[kk]), 'r') as f:
			with h5py.File('/home/xkchen/jupyter/stack/alt_Bro-mode-select_Mean_jack_SB-pro_%.1f-sigma_%d-FWHM-ov2_z-ref.h5' % (tt_sigma[mm], size_arr[kk]), 'r') as f:
				c_r_arr = np.array(f['r'])
				c_sb_arr = np.array(f['sb'])
				c_sb_err = np.array(f['sb_err'])

			idnn = np.isnan(c_sb_arr)
			idNul = c_r_arr > 0
			idv = (idnn == False) & idNul
			c_r_arr, c_sb_arr, c_sb_err = c_r_arr[idv], c_sb_arr[idv], c_sb_err[idv]

			tmp_r.append(c_r_arr)
			tmp_sb.append(c_sb_arr)
			tmp_err.append(c_sb_err)

		id_nan = np.isnan(tt_img)
		idvx = id_nan == False
		idy, idx = np.where(idvx == True)
		x_low, x_up = np.min(idx), np.max(idx)
		y_low, y_up = np.min(idy), np.max(idy)

		dpt_img = tt_img[y_low: y_up+1, x_low: x_up + 1]
		grid_patch = cc_grid_img(dpt_img, 100, 100,)
		img_block = grid_patch[0]
		block_Var = grid_patch[2]
		lx, ly = grid_patch[-2], grid_patch[-1]
		'''
		fig = plt.figure( figsize = (13.12, 4.8) )
		ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
		ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

		ax0.set_title('%.1f $\\sigma$, %d (FWHM/2)' % (tt_sigma[mm], size_arr[kk]),)
		tg = ax0.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
		cb = plt.colorbar(tg, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
		cb.formatter.set_powerlimits( (0,0) )

		ax1.plot(c_r_arr, c_sb_arr, ls = '-', color = 'r', alpha = 0.8,)
		ax1.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'r', alpha = 0.2,)

		ax1.plot(alt_A250_r, alt_A250_sb, ls = '-', color = 'k', alpha = 0.8, label = 'A-250',)
		ax1.fill_between(alt_A250_r, y1 = alt_A250_sb - alt_A250_sb_err, y2 = alt_A250_sb + alt_A250_sb_err, color = 'k', alpha = 0.2,)

		ax1.plot(alt_tot_r, alt_tot_sb, ls = '-', color = 'c', alpha = 0.8, label = 'tot-1000',)
		ax1.fill_between(alt_tot_r, y1 = alt_tot_sb - alt_tot_sb_err, y2 = alt_tot_sb + alt_tot_sb_err, color = 'c', alpha = 0.2,)

		ax1.set_ylim(1e-3, 3e-2)
		ax1.set_yscale('log')
		ax1.set_xlim(5e1, 4e3)
		ax1.set_xlabel('R [kpc]')
		ax1.set_ylabel('SB [nanomaggies / $arcsec^2$]')
		ax1.set_xscale('log')
		ax1.legend(loc = 3, frameon = False,)
		ax1.grid(which = 'both', axis = 'both', alpha = 0.25)
		ax1.tick_params(axis = 'both', which = 'both', direction = 'in',)
		tick_form = mpl.ticker.LogFormatter(labelOnlyBase = False)
		ax1.get_yaxis().set_minor_formatter(tick_form)

		plt.savefig('2D-grd_SB_%.1f-sigma_%d-FWHM-ov2.png' % (tt_sigma[mm], size_arr[kk]), dpi = 300)
		plt.close()
		'''
	R_set.append( tmp_r )
	SB_set.append( tmp_sb )
	err_set.append( tmp_err )
'''
lsy = [':', '-.', '-',]
lcy = ['b', 'g', 'r']

for kk in range( 3 ):

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('%d(FWHM/2) mask-size' % size_arr[kk])

	for mm in range( 3 ):

		tt_r = R_set[kk][mm]
		tt_sb = SB_set[kk][mm]
		tt_err = err_set[kk][mm]

		ax.plot(tt_r, tt_sb, ls = lsy[mm], color = lcy[mm], alpha = 0.5, label = '%.1f$\\sigma$' % tt_sigma[mm],)
		ax.fill_between(tt_r, y1 = tt_sb - tt_err, y2 = tt_sb + tt_err, color = lcy[mm], alpha = 0.2,)

		idr = (tt_r > 1e3) & (tt_r <= 3e3)
		idsb = np.nanmin( tt_sb[idr] )
		devi_sb = tt_sb - idsb

		ax.plot(tt_r, devi_sb, ls = '--', color = lcy[mm], alpha = 0.5,)

	ax.plot(R_obs, flux_obs, ls = '-', color = 'm', alpha = 0.5, label = 'Z05',)

	ax.plot(A_R[kk], A_SB[kk], ls = '-', color = 'k', alpha = 0.8, label = 'A-250',)
	ax.fill_between(A_R[kk], y1 = A_SB[kk] - A_err[kk], y2 = A_SB[kk] + A_err[kk], color = 'k', alpha = 0.2,)

	idr = (A_R[kk] > 1e3) & (A_R[kk] <= 3e3)
	idsb = np.nanmin( A_SB[kk][idr] )
	devi_sb = A_SB[kk] - idsb

	ax.plot(A_R[kk], devi_sb, ls = '--', color = 'k', alpha = 0.8,)
	ax.fill_between(A_R[kk], y1 = devi_sb - A_err[kk], y2 = devi_sb + A_err[kk], color = 'k', alpha = 0.2,)


	ax.plot(tot_R[kk], tot_SB[kk], ls = '-', color = 'c', alpha = 0.8, label = 'tot-1000',)
	ax.fill_between(tot_R[kk], y1 = tot_SB[kk] - tot_err[kk], y2 = tot_SB[kk] + tot_err[kk], color = 'c', alpha = 0.2,)

	idr = (tot_R[kk] > 1e3) & (tot_R[kk] <= 3e3)
	idsb = np.nanmin( tot_SB[kk][idr] )
	devi_sb = tot_SB[kk] - idsb

	ax.plot(tot_R[kk], devi_sb, ls = '--', color = 'c', alpha = 0.8,)
	#ax.fill_between(tot_R[kk], y1 = devi_sb - tot_err[kk], y2 = devi_sb + tot_err[kk], color = 'c', alpha = 0.2,)

	ax.set_xlim(5e1, 4e3)
	ax.set_xlabel('R [kpc]')
	ax.set_ylim(1e-4, 3e-2)
	ax.set_yscale('log')
	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax.set_xscale('log')
	ax.legend(loc = 3, frameon = False,)
	ax.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in',)

	plt.savefig('sigma-selection_SB_compare_%d-FWHM-ov2.png' % size_arr[kk], dpi = 300)
	plt.close()

for kk in range( 3 ):

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('%d(FWHM/2) mask-size' % size_arr[kk])

	for mm in range( 3 ):

		tt_r = R_set[kk][mm]
		tt_sb = SB_set[kk][mm]
		tt_err = err_set[kk][mm]

		ax.plot(tt_r, tt_err, ls = lsy[mm], color = lcy[mm], alpha = 0.5, label = '%.1f$\\sigma$' % tt_sigma[mm],)

	ax.plot(A_R[kk], A_err[kk], ls = '-', color = 'k', alpha = 0.5, label = 'A-250',)

	ax.plot(tot_R[kk], tot_err[kk], ls = '-', color = 'c', alpha = 0.5, label = 'tot-1000',)

	ax.set_xlim(5e1, 4e3)
	ax.set_xlabel('R [kpc]')
	ax.set_ylim(1e-5, 3e-3)
	ax.set_yscale('log')
	ax.set_ylabel('SB err [nanomaggies / $arcsec^2$]')
	ax.set_xscale('log')
	ax.legend(loc = 3, frameon = False,)
	ax.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in',)

	plt.savefig('sigma-selection_SB-err_compare_%d-FWHM-ov2.png' % size_arr[kk], dpi = 300)
	plt.close()

for mm in range( 3 ):

	cc_r = R_set[0][mm]
	cc_err = err_set[0][mm]

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('%.1f$\\sigma$' % tt_sigma[mm])

	plot_lines = []

	for kk in range( 3 ):

		tt_r = R_set[kk][mm]
		tt_sb = SB_set[kk][mm]
		tt_err = err_set[kk][mm]

		l1, = ax.plot(tt_r, tt_err, ls = lsy[kk], color = 'r', alpha = 0.5, label = '%d(FWHM/2) mask-size' % size_arr[kk],)

		l2, = ax.plot(A_R[kk], A_err[kk], ls = lsy[kk], color = 'k', alpha = 0.5, label = 'A-250',)

		l3, = ax.plot(tot_R[kk], tot_err[kk], ls = lsy[kk], color = 'c', alpha = 0.5, label = 'tot-1000',)

		plot_lines.append([l1, l2, l3])

	legend1 = plt.legend( plot_lines[0][1:], ['A-250', 'tot-1000'], loc = 4, frameon = False,)
	plt.legend( [plot_lines[0][0], plot_lines[1][0], plot_lines[2][0],], 
				['10(FWHM/2)', '20(FWHM/2)', '30(FWHM/2)', 'tot-1000', 'A-250',], loc = 3, frameon = False,)
	plt.gca().add_artist(legend1)

	ax.set_xlim(5e1, 4e3)
	ax.set_xlabel('R [kpc]')
	ax.set_ylim(1e-5, 3e-3)
	ax.set_yscale('log')
	ax.set_ylabel('SB err [nanomaggies / $arcsec^2$]')
	ax.set_xscale('log')
	ax.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in',)

	plt.savefig('sigma-selection_SB-err_compare_%.1f-sigma.png' % tt_sigma[mm], dpi = 300)
	plt.close()
'''

### allpied SN_mask case
lsy = [':', '-.', '-',]
lcy = ['b', 'g', 'r']

lim_sb = np.array([4., 8.]) ## *1e-3
sigma = np.array([4, 5, 6])
star_size = np.array([10, 20, 30])

limd_tot_pros = '/home/xkchen/jupyter/stack/' + 'T1000-tot_Mean_jack_SB-pro_%d-FWHM-ov2_z-ref_%de-3-lim.h5'

limd_A250_pros = '/home/xkchen/jupyter/stack/' + 'T1000-to-A250_Mean_jack_SB-pro_%d-FWHM-ov2_z-ref_%de-3-lim.h5'

comp_sb = [	'/home/xkchen/jupyter/stack/' + 'alt_Bro-mode-select_Mean_jack_SB-pro_6.0-sigma_10-FWHM-ov2_z-ref.h5',
			'/home/xkchen/jupyter/stack/' + 'alt_Bro-mode-select_Mean_jack_SB-pro_6.0-sigma_20-FWHM-ov2_z-ref.h5',
			'/home/xkchen/jupyter/stack/' + 'Bro-mode-select_Mean_jack_SB-pro_selected-by-tot_6.0-sigma_z-ref.h5']

pros_lis = '/home/xkchen/jupyter/stack/' + 'SN_masked_Bro-mode-select_Mean_jack_SB-pro_6.0-sigma_%d-FWHM-ov2_z-ref_%de-3-lim.h5'


for mm in range( 3 ):

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('%d(FWHM/2) mask-size' % star_size[mm])

	ax.plot(A_R[mm], A_SB[mm], ls = '-', color = 'k', alpha = 0.8, label = 'A-250',)
	ax.fill_between(A_R[mm], y1 = A_SB[mm] - A_err[mm], y2 = A_SB[mm] + A_err[mm], color = 'k', alpha = 0.2,)

	ax.plot(tot_R[mm], tot_SB[mm], ls = '-', color = 'c', alpha = 0.8, label = 'tot-1000',)
	ax.fill_between(tot_R[mm], y1 = tot_SB[mm] - tot_err[mm], y2 = tot_SB[mm] + tot_err[mm], color = 'c', alpha = 0.2,)

	idr = (A_R[mm] > 1e3) & (A_R[mm] <= 3e3)
	idsb = np.nanmin( A_SB[mm][idr] )
	devi_sb = A_SB[mm] - idsb

	ax.plot(A_R[mm], devi_sb, ls = '--', color = 'k', alpha = 0.8,)
	ax.fill_between(A_R[mm], y1 = devi_sb - A_err[mm], y2 = devi_sb + A_err[mm], color = 'k', alpha = 0.2,)

	idr = (tot_R[mm] > 1e3) & (tot_R[mm] <= 3e3)
	idsb = np.nanmin( tot_SB[mm][idr] )
	devi_sb = tot_SB[mm] - idsb

	ax.plot(tot_R[mm], devi_sb, ls = '--', color = 'c', alpha = 0.8,)
	#ax.fill_between(tot_R[mm], y1 = devi_sb - tot_err[mm], y2 = devi_sb + tot_err[mm], color = 'c', alpha = 0.2,)

	with h5py.File( comp_sb[mm], 'r') as f:
		com_r = np.array(f['r'])
		com_sb = np.array(f['sb'])
		com_sb_err = np.array(f['sb_err'])

	idnn = np.isnan(com_sb)
	idNul = com_r > 0
	idv = (idnn == False) & idNul
	com_r, com_sb, com_sb_err = com_r[idv], com_sb[idv], com_sb_err[idv]

	ax.plot(com_r, com_sb, ls = '-', color = 'r', alpha = 0.8, label = 'use all blocks',)
	ax.fill_between(com_r, y1 = com_sb - com_sb_err, y2 = com_sb + com_sb_err, color = 'r', alpha = 0.2,)

	idr = (com_r > 1e3) & (com_r <= 3e3)
	idsb = np.nanmin( com_sb[idr] )
	devi_sb = com_sb - idsb

	ax.plot(com_r, devi_sb, ls = '--', color = 'r', alpha = 0.8,)

	## use 6-sigma only
	for kk in range( 2 ):

		with h5py.File( limd_tot_pros % (star_size[mm], lim_sb[kk]), 'r') as f:
			lim_tot_r = np.array(f['r'])
			lim_tot_sb = np.array(f['sb'])
			lim_tot_err = np.array(f['sb_err'])

		idnn = np.isnan(lim_tot_sb)
		idNul = lim_tot_r > 0
		idv = (idnn == False) & idNul
		lim_tot_r, lim_tot_sb, lim_tot_err = lim_tot_r[idv], lim_tot_sb[idv], lim_tot_err[idv]

		with h5py.File( limd_A250_pros % (star_size[mm], lim_sb[kk]), 'r') as f:
			lim_A_r = np.array(f['r'])
			lim_A_sb = np.array(f['sb'])
			lim_A_err = np.array(f['sb_err'])

		idnn = np.isnan(lim_A_sb)
		idNul = lim_A_r > 0
		idv = (idnn == False) & idNul
		lim_A_r, lim_A_sb, lim_A_err = lim_A_r[idv], lim_A_sb[idv], lim_A_err[idv]

		ax.plot(lim_A_r, lim_A_sb, ls = lsy[kk], color = 'k', alpha = 0.8,)
		#ax.fill_between(lim_A_r, y1 = lim_A_sb - lim_A_err, y2 = lim_A_sb + lim_A_err, color = 'k', alpha = 0.2,)

		ax.plot(lim_tot_r, lim_tot_sb, ls = lsy[kk], color = 'c', alpha = 0.8,)
		#ax.fill_between(lim_tot_r, y1 = lim_tot_sb - lim_tot_err, y2 = lim_tot_sb + lim_tot_err, color = 'c', alpha = 0.2,)

		with h5py.File( pros_lis % (star_size[mm], lim_sb[kk]), 'r') as f:
			c_r_arr = np.array(f['r'])
			c_sb_arr = np.array(f['sb'])
			c_sb_err = np.array(f['sb_err'])

		idnn = np.isnan(c_sb_arr)
		idNul = c_r_arr > 0
		idv = (idnn == False) & idNul
		c_r_arr, c_sb_arr, c_sb_err = c_r_arr[idv], c_sb_arr[idv], c_sb_err[idv]

		ax.plot(c_r_arr, c_sb_arr, ls = lsy[kk], color = lcy[kk], alpha = 0.8, label = 'use blocks:$\\sigma$ <= %de-3' % lim_sb[kk],)
		ax.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = lcy[kk], alpha = 0.2,)

		idr = (c_r_arr > 1e3) & (c_r_arr <= 3e3)
		idsb = np.nanmin( c_sb_arr[idr] )
		devi_sb = c_sb_arr - idsb

		ax.plot(c_r_arr, devi_sb, ls = '--', color = lcy[kk], alpha = 0.5,)
		#ax.fill_between(c_r_arr, y1 = devi_sb - c_sb_err, y2 = devi_sb + c_sb_err, color = lcy[kk], alpha = 0.2,)

	ax.set_ylim(1e-4, 3e-2)
	ax.set_yscale('log')
	ax.set_xlim(5e1, 4e3)
	ax.set_xlabel('R [kpc]')
	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax.set_xscale('log')
	ax.legend(loc = 3, frameon = False,)
	ax.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in',)

	plt.savefig('SN_masked_SB_%de-3-lim_%d-FWHM-ov2.png' % (lim_sb[kk], star_size[mm]), dpi = 300,)
	plt.close()

raise
"""
### 2D img compare
for mm in range( 3 ):

	with h5py.File(
		'/home/xkchen/jupyter/stack/Bro-mode-select_Mean_jack_img_%.1f-sigma_%d-FWHM-ov2_z-ref.h5' % (tt_sigma[mm], size_arr[0]), 'r') as f:
		tt_img = np.array(f['a'])

	id_nan = np.isnan(tt_img)
	idvx = id_nan == False
	idy, idx = np.where(idvx == True)
	x_low, x_up = np.min(idx), np.max(idx)
	y_low, y_up = np.min(idy), np.max(idy)

	dpt_img = tt_img[y_low: y_up+1, x_low: x_up + 1]
	img_block = cc_grid_img(dpt_img, 100, 100,)[0]

	block_arr = []

	for kk in range( 1,3):

		if kk == 2:
			with h5py.File(
			load + '20_10_test_jack/Bro-mode-select_Mean_jack_img_selected-by-tot_%.1f-sigma_z-ref.h5' % tt_sigma[mm], 'r') as f:
				tk_img = np.array(f['a'])
		else:
			with h5py.File(
			'/home/xkchen/jupyter/stack/Bro-mode-select_Mean_jack_img_%.1f-sigma_%d-FWHM-ov2_z-ref.h5' % (tt_sigma[mm], size_arr[kk]), 'r') as f:
				tk_img = np.array(f['a'])

		dpk_img = tk_img[y_low: y_up+1, x_low: x_up + 1]
		patch_mean = cc_grid_img(dpk_img, 100, 100,)[0]
		block_arr.append( patch_mean )

	fig = plt.figure( figsize = (19.84, 4.8) )
	ax0 = fig.add_axes([0.02, 0.09, 0.28, 0.85])
	ax1 = fig.add_axes([0.35, 0.09, 0.28, 0.85])
	ax2 = fig.add_axes([0.68, 0.09, 0.28, 0.85])

	ax0.set_title('10 (FWHM/2)')
	tg = ax0.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits( (0,0) )

	ax1.set_title('20 (FWHM/2)')
	tg = ax1.imshow(block_arr[0] / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = ax1, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits( (0,0) )

	ax2.set_title('30 (FWHM/2)')
	tg = ax2.imshow(block_arr[1] / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = ax2, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits( (0,0) )

	plt.savefig('2D-img_compare_%.1f-sigma.png' % (tt_sigma[mm]), dpi = 300)
	plt.close()

tot_img_lis = [ '/home/xkchen/jupyter/stack/T1000-tot_Mean_jack_img_10-FWHM-ov2_z-ref.h5',
				'/home/xkchen/jupyter/stack/T1000-tot_Mean_jack_img_20-FWHM-ov2_z-ref.h5',
				load + '20_10_test_jack/T1000-tot_BCG-stack_Mean_jack_img_z-ref.h5']

A250_img_lis = ['/home/xkchen/jupyter/stack/T1000-to-A250_Mean_jack_img_10-FWHM-ov2_z-ref.h5',
				'/home/xkchen/jupyter/stack/T1000-to-A250_Mean_jack_img_20-FWHM-ov2_z-ref.h5',
				load + '20_10_test_jack/A_clust_BCG-stack_Mean_jack_img_30-FWHM-ov2_z-ref.h5']

tot_patch = []
Asub_patch = []

with h5py.File(tot_img_lis[0], 'r') as f:
	tt_img = np.array(f['a'])

id_nan = np.isnan(tt_img)
idvx = id_nan == False
idy, idx = np.where(idvx == True)
x_low, x_up = np.min(idx), np.max(idx)
y_low, y_up = np.min(idy), np.max(idy)

dpt_img = tt_img[y_low: y_up+1, x_low: x_up + 1]
img_block = cc_grid_img(dpt_img, 100, 100,)[0]
tot_patch.append( img_block )

with h5py.File(A250_img_lis[0], 'r') as f:
	tt_img = np.array(f['a'])
dpt_img = tt_img[y_low: y_up+1, x_low: x_up + 1]
img_block = cc_grid_img(dpt_img, 100, 100,)[0]
Asub_patch.append( img_block )

### 2D img compare
for mm in range( 1,3 ):

	with h5py.File(tot_img_lis[mm], 'r') as f:
		tk_img = np.array(f['a'])

	dpk_img = tk_img[y_low: y_up+1, x_low: x_up + 1]
	patch_mean = cc_grid_img(dpk_img, 100, 100,)[0]
	tot_patch.append( patch_mean )	

	with h5py.File(A250_img_lis[mm], 'r') as f:
		tk_img = np.array(f['a'])

	dpk_img = tk_img[y_low: y_up+1, x_low: x_up + 1]
	patch_mean = cc_grid_img(dpk_img, 100, 100,)[0]
	Asub_patch.append( patch_mean )		

fig = plt.figure( figsize = (19.84, 4.8) )
#fig.suptitle('A-250')
fig.suptitle('tot-1000')

ax0 = fig.add_axes([0.02, 0.09, 0.28, 0.80])
ax1 = fig.add_axes([0.35, 0.09, 0.28, 0.80])
ax2 = fig.add_axes([0.68, 0.09, 0.28, 0.80])

ax0.set_title('10 (FWHM/2)')
#tg = ax0.imshow(Asub_patch[0] / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
tg = ax0.imshow(tot_patch[0] / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
cb = plt.colorbar(tg, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
cb.formatter.set_powerlimits( (0,0) )

ax1.set_title('20 (FWHM/2)')
#tg = ax1.imshow(Asub_patch[1] / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
tg = ax1.imshow(tot_patch[1] / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
cb = plt.colorbar(tg, ax = ax1, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
cb.formatter.set_powerlimits( (0,0) )

ax2.set_title('30 (FWHM/2)')
#tg = ax2.imshow(Asub_patch[2] / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
tg = ax2.imshow(tot_patch[2] / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
cb = plt.colorbar(tg, ax = ax2, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
cb.formatter.set_powerlimits( (0,0) )

#plt.savefig('A250_2D-img_compare.png', dpi = 300)
plt.savefig('T1000_2D-img_compare.png', dpi = 300)
plt.close()
"""

#### masking size for bright stars [25, 50, 75(default) ]
tot_test_img = ['/home/xkchen/jupyter/stack/T1000-tot_Mean_jack_img_25-FWHM-ov2_bri-star.h5', 
				'/home/xkchen/jupyter/stack/T1000-tot_Mean_jack_img_50-FWHM-ov2_bri-star.h5',
				load + '20_10_test_jack/T1000-tot_BCG-stack_Mean_jack_img_z-ref.h5']

tot_test_sb = [	'/home/xkchen/jupyter/stack/T1000-tot_Mean_jack_SB-pro_25-FWHM-ov2_bri-star.h5', 
				'/home/xkchen/jupyter/stack/T1000-tot_Mean_jack_SB-pro_50-FWHM-ov2_bri-star.h5',
				'tmp_test/T1000-tot_R-bin_SB_test.h5']

Asub_imgs = [	'/home/xkchen/jupyter/stack/T1000-to-A250_Mean_jack_img_25-FWHM-ov2_bri-star.h5', 
				'/home/xkchen/jupyter/stack/T1000-to-A250_Mean_jack_img_50-FWHM-ov2_bri-star.h5', 
				load + '20_10_test_jack/A_clust_BCG-stack_Mean_jack_img_30-FWHM-ov2_z-ref.h5']

Asub_pros = [	'/home/xkchen/jupyter/stack/T1000-to-A250_Mean_jack_SB-pro_25-FWHM-ov2_bri-star.h5', 
				'/home/xkchen/jupyter/stack/T1000-to-A250_Mean_jack_SB-pro_50-FWHM-ov2_bri-star.h5', 
				'tmp_test/A250_R-bin_SB_test.h5']

size_arr = np.array([25, 50, 75])

patch_means = []
R_set, SB_set, err_set = [], [], []

for kk in range( 3 ):

	#with h5py.File( tot_test_img[kk], 'r') as f:
	with h5py.File( Asub_imgs[kk], 'r') as f:
		tt_img = np.array(f['a'])

	#with h5py.File( tot_test_sb[kk], 'r') as f:
	with h5py.File( Asub_pros[kk], 'r') as f:
		c_r_arr = np.array(f['r'])
		c_sb_arr = np.array(f['sb'])
		c_sb_err = np.array(f['sb_err'])

	idnn = np.isnan(c_sb_arr)
	idNul = c_r_arr > 0
	idv = (idnn == False) & idNul
	c_r_arr, c_sb_arr, c_sb_err = c_r_arr[idv], c_sb_arr[idv], c_sb_err[idv]

	id_nan = np.isnan(tt_img)
	idvx = id_nan == False
	idy, idx = np.where(idvx == True)
	x_low, x_up = np.min(idx), np.max(idx)
	y_low, y_up = np.min(idy), np.max(idy)

	dpt_img = tt_img[y_low: y_up+1, x_low: x_up + 1]
	img_block = cc_grid_img(dpt_img, 100, 100,)[0]

	patch_means.append(img_block)
	R_set.append( c_r_arr)
	SB_set.append( c_sb_arr)
	err_set.append( c_sb_err)

	fig = plt.figure( figsize = (13.12, 4.8) )
	ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
	ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

	ax0.set_title('%d (FWHM/2)' % size_arr[kk],)
	tg = ax0.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits( (0,0) )

	ax1.plot(c_r_arr, c_sb_arr, ls = '-', color = 'r', alpha = 0.8,)
	ax1.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'r', alpha = 0.2,)

	ax1.set_ylim(1e-3, 3e-2)
	ax1.set_yscale('log')
	ax1.set_xlim(5e1, 4e3)
	ax1.set_xlabel('R [kpc]')
	ax1.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax1.set_xscale('log')
	ax1.legend(loc = 3, frameon = False,)
	ax1.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax1.tick_params(axis = 'both', which = 'both', direction = 'in',)
	tick_form = mpl.ticker.LogFormatter(labelOnlyBase = False)
	ax1.get_yaxis().set_minor_formatter(tick_form)

	#plt.savefig('tot-1000_2D-grd_SB_%d-FWHM-ov2_bri-star.png' % size_arr[kk], dpi = 300)
	plt.savefig('A250_2D-grd_SB_%d-FWHM-ov2_bri-star.png' % size_arr[kk], dpi = 300)
	plt.close()


fig = plt.figure( figsize = (19.84, 4.8) )
#fig.suptitle('tot-1000')
fig.suptitle('A-250')

ax0 = fig.add_axes([0.02, 0.09, 0.28, 0.80])
ax1 = fig.add_axes([0.35, 0.09, 0.28, 0.80])
ax2 = fig.add_axes([0.68, 0.09, 0.28, 0.80])

ax0.set_title('25 (FWHM/2)')
tg = ax0.imshow(patch_means[0] / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
cb = plt.colorbar(tg, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
cb.formatter.set_powerlimits( (0,0) )

ax1.set_title('50 (FWHM/2)')
tg = ax1.imshow(patch_means[1] / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
cb = plt.colorbar(tg, ax = ax1, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
cb.formatter.set_powerlimits( (0,0) )

ax2.set_title('75 (FWHM/2)')
tg = ax2.imshow(patch_means[2] / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
cb = plt.colorbar(tg, ax = ax2, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
cb.formatter.set_powerlimits( (0,0) )

#plt.savefig('T1000_2D-img_compare.png', dpi = 300)
plt.savefig('A250_2D-img_compare.png', dpi = 300)
plt.close()


lsy = [':', '-.', '-',]
lcy = ['b', 'g', 'r']

plt.figure()
ax = plt.subplot(111)
#ax.set_title('tot-1000')
ax.set_title('A-250')

for kk in range( 3 ):

	tt_r = R_set[kk]
	tt_sb = SB_set[kk]
	tt_err = err_set[kk]

	ax.plot(tt_r, tt_sb, ls = '-', color = lcy[kk], alpha = 0.5, label = '%d (FWHM/2)' % size_arr[kk],)
	ax.fill_between(tt_r, y1 = tt_sb - tt_err, y2 = tt_sb + tt_err, color = lcy[kk], alpha = 0.2,)

	idr = (tt_r > 1e3) & (tt_r <= 3e3)
	idsb = np.nanmin( tt_sb[idr] )
	devi_sb = tt_sb - idsb

	ax.plot(tt_r, devi_sb, ls = '--', color = lcy[kk], alpha = 0.5,)

ax.set_xlim(5e1, 4e3)
ax.set_xlabel('R [kpc]')
ax.set_ylim(1e-4, 3e-2)
ax.set_yscale('log')
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 3, frameon = False,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in',)

#plt.savefig('bri-star-mask_SB_compare_tot-1000.png', dpi = 300)
plt.savefig('bri-star-mask_SB_compare_A-250.png', dpi = 300)
plt.close()

## err compare
plt.figure()
ax = plt.subplot(111)
#ax.set_title('tot-1000')
ax.set_title('A-250')

for kk in range( 3 ):

	tt_r = R_set[kk]
	tt_sb = SB_set[kk]
	tt_err = err_set[kk]

	ax.plot(tt_r, tt_err, ls = '-', color = lcy[kk], alpha = 0.5, label = '%d (FWHM/2)' % size_arr[kk],)

ax.set_xlim(5e1, 4e3)
ax.set_xlabel('R [kpc]')
ax.set_ylim(1e-4, 3e-3)
ax.set_yscale('log')
ax.set_ylabel('SB err [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 3, frameon = False,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in',)

#plt.savefig('bri-star-mask_SB-err_compare_tot-1000.png', dpi = 300)
plt.savefig('bri-star-mask_SB-err_compare_A-250.png', dpi = 300)
plt.close()

