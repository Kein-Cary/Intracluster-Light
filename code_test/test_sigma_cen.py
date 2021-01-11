import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

import glob
import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

import scipy.stats as sts
import astropy.units as U
import subprocess as subpro
import astropy.constants as C

from scipy.optimize import curve_fit
from astropy import cosmology as apcy
from fig_out_module import cc_grid_img, grid_img

### cosmology model
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

############## sample selection test
#from img_block_select import diffuse_identi_func
from cc_block_select import diffuse_identi_func

thres_S0, thres_S1 = 3, 5
sigma = np.array([3.5, 4, 4.5, 5, 5.5, 6,]) ## sigma as limit

home = '/media/xkchen/My Passport/data/SDSS/'
d_file = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'
id_sigm = 1

mu_sigm_file = '/home/xkchen/mywork/ICL/code/SEX/img_test-1000_mean_sigm.csv'
#mu_sigm_file = '/home/xkchen/mywork/ICL/code/SEX/img_A-250_mean_sigm.csv'

n_main = np.array([250, 98, 193, 459]) ## (A, B, C, D)
"""
for kk in range( 4 ):

	dat = pds.read_csv('SEX/result/test_1000-to-%d_cat.csv' % n_main[kk],)
	set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	'''
	rule_file = 'mode-select_1000-to-%d_rule-out_cat_%.1f-sigma.csv' % ( n_main[kk], sigma[id_sigm] )
	remain_file = 'mode-select_1000-to-%d_remain_cat_%.1f-sigma.csv' % ( n_main[kk], sigma[id_sigm] )
	diffuse_identi_func(band[0], set_ra, set_dec, set_z, d_file, rule_file, remain_file, thres_S0, thres_S1, sigma[id_sigm],
						mu_sigm_file, id_single = False, id_mode = True,)

	rule_file = 'mean-select_1000-to-%d_rule-out_cat_%.1f-sigma.csv' % ( n_main[kk], sigma[id_sigm] )
	remain_file = 'mean-select_1000-to-%d_remain_cat_%.1f-sigma.csv' % ( n_main[kk], sigma[id_sigm] )
	diffuse_identi_func(band[0], set_ra, set_dec, set_z, d_file, rule_file, remain_file, thres_S0, thres_S1, sigma[id_sigm],
							mu_sigm_file, id_single = False, id_mode = False,)
	'''
	rule_file = 'Bro-mean-select_1000-to-%d_rule-out_cat_%.1f-sigma.csv' % ( n_main[kk], sigma[id_sigm] )
	remain_file = 'Bro-mean-select_1000-to-%d_remain_cat_%.1f-sigma.csv' % ( n_main[kk], sigma[id_sigm] )
	diffuse_identi_func(band[0], set_ra, set_dec, set_z, d_file, rule_file, remain_file, thres_S0, thres_S1, sigma[id_sigm],
						mu_sigm_file, id_single = False, id_mode = False,)

	rule_file = 'Bro-mode-select_1000-to-%d_rule-out_cat_%.1f-sigma.csv' % ( n_main[kk], sigma[id_sigm] )
	remain_file = 'Bro-mode-select_1000-to-%d_remain_cat_%.1f-sigma.csv' % ( n_main[kk], sigma[id_sigm] )
	diffuse_identi_func(band[0], set_ra, set_dec, set_z, d_file, rule_file, remain_file, thres_S0, thres_S1, sigma[id_sigm],
						mu_sigm_file, id_single = False, id_mode = True,)
"""

## PS:
## 		mean-select*_cat -- select imgs based on mean of (mu_cen, sigma_cen) for given img sample + 
##							further brighter subpatches
##		mode-select*_cat -- select imgs based on mode point of (mu_cen, sigma_cen) for given img sample + 
##							further brighter subpatches

## 		Bro-mean-select*_cat -- select imgs based on mean of (mu_cen, sigma_cen) for given img sample, 
## 		Bro-mode-select*_cat -- select imgs based on mode point of (mu_cen, sigma_cen) for given img sample, 

"""
n_eta = np.zeros(4, dtype = np.float32)
p_eta = np.zeros(4, dtype = np.float32)
a_eta = np.zeros(4, dtype = np.float32)
m_eta = np.zeros(4, dtype = np.float32)

m_ra = np.array([])
mod_ra = np.array([])
B_m_ra = np.array([])
B_mod_ra = np.array([])

for mm in range(4):

	dat = pds.read_csv('SEX/result/test_1000-to-%d_cat.csv' % (n_main[mm]) )
	set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	##### select_based_on_A250
	pre_dat = pds.read_csv('SEX/result/select_based_on_T1000/mean-select_1000-to-%d_remain_cat_%.1f-sigma.csv' % (n_main[mm], sigma[id_sigm] ), )
	pre_ra = np.array(pre_dat.ra)
	p_eta[mm] = pre_ra.shape[0] / set_ra.shape[0]
	m_ra = np.r_[ m_ra, np.array(pre_dat.ra) ]

	alt_dat = pds.read_csv('SEX/result/select_based_on_T1000/mode-select_1000-to-%d_remain_cat_%.1f-sigma.csv' % (n_main[mm], sigma[id_sigm] ), )
	alt_ra = np.array(alt_dat.ra)
	a_eta[mm] = alt_ra.shape[0] / set_ra.shape[0]
	mod_ra = np.r_[ mod_ra, np.array(alt_dat.ra) ]

	code_dat = pds.read_csv('SEX/result/select_based_on_T1000/Bro-mean-select_1000-to-%d_remain_cat_%.1f-sigma.csv' % (n_main[mm], sigma[id_sigm] ), )
	cod_ra = np.array(code_dat.ra)
	n_eta[mm] = cod_ra.shape[0] / set_ra.shape[0]
	B_m_ra = np.r_[ B_m_ra, np.array(code_dat.ra) ]


	mm_dat = pds.read_csv('SEX/result/select_based_on_T1000/Bro-mode-select_1000-to-%d_remain_cat_%.1f-sigma.csv' % (n_main[mm], sigma[id_sigm] ), )
	mm_ra = np.array(mm_dat.ra)
	m_eta[mm] = mm_ra.shape[0] / set_ra.shape[0]
	B_mod_ra = np.r_[ B_mod_ra, np.array(mm_dat.ra) ]

xpot = np.array([1,2,3,4])

name_lis = ['$\\bar{\\sigma}, \\bar{\\mu}$ + brighter sub-patches', 
			'$Mode(\\sigma), Mode(\\mu)$ + brighter sub-patches', 
			'$\\bar{\\sigma}, \\bar{\\mu}$', 
			'$Mode(\\sigma), Mode(\\mu)$']

plt.figure()
ax = plt.subplot(111)

#ax.set_title('selection based on A-250')
ax.set_title('selection based on test-1000')
ax.plot(xpot, p_eta, 'bs', alpha = 0.5, label = name_lis[0], )
ax.plot(xpot, a_eta, 'g^', alpha = 0.5, label = name_lis[1], )
ax.plot(xpot, n_eta, 'ro', alpha = 0.5, label = name_lis[2], )
ax.plot(xpot, m_eta, 'c*', alpha = 0.5, label = name_lis[3], )

ax.legend(loc = 3, frameon = False,)
ax.set_ylim(-0.1, 1.1)
ax.set_xlim(0.5, 4.5)
ax.set_xticks(xpot)
ax.set_xticklabels(['A', 'B', 'C', 'D'])
ax.set_xlabel('sample type')
ax.set_ylabel('$ N_{normal} / N_{total} $')
plt.savefig('img_select_compare.png', dpi = 300)
plt.close()

Nx = [m_ra.shape[0], mod_ra.shape[0], B_m_ra.shape[0], B_mod_ra.shape[0] ]
plt.figure()
ax = plt.subplot(111)
#ax.set_title('sample size [selection based on A250]')
ax.set_title('sample size [selection based on T1000]')

ax.plot([5, 15, 25, 30], Nx, 'b*',)
ax.set_xticks([5, 15, 25, 30])
ax.set_xticklabels(['$\\bar{\\sigma}, \\bar{\\mu}$'+ '\n' +'brighter sub-patches', 
			'$Mode(\\sigma), Mode(\\mu)$' + '\n' + 'brighter sub-patches', 
			'$\\bar{\\sigma}, \\bar{\\mu}$', '$Mode(\\sigma), Mode(\\mu)$'])
#plt.savefig('img_number_selected_by_A250.png', dpi = 100)
plt.savefig('img_number_selected_by_T1000.png', dpi = 100)
plt.close()
"""

### A, B, C, D sub-sample stacking test
load = '/media/xkchen/My Passport/data/SDSS/'
"""
pro_lis = [	load + '20_10_test_jack/T1000_total_Mean_jack_SB-pro.h5', 
			load + '20_10_test_jack/clust_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2.h5', 
			load + '20_10_test_jack/AB_clust_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2.h5', 
			load + '20_10_test_jack/T1000_No-C_Mean_jack_SB-pro.h5', 
			load + '20_10_test_jack/T1000_No-D_Mean_jack_SB-pro.h5']

img_lis = [	load + '20_10_test_jack/T1000_total_Mean_jack_img.h5', 
			load + '20_10_test_jack/clust_BCG-stack_Mean_jack_img_30-FWHM-ov2.h5', 
			load + '20_10_test_jack/AB_clust_BCG-stack_Mean_jack_img_30-FWHM-ov2.h5', 
			load + '20_10_test_jack/T1000_No-C_Mean_jack_img.h5', 
			load + '20_10_test_jack/T1000_No-D_Mean_jack_img.h5']

line_label = ['tot-1000', 'A-250', 'A+B', 'A+B+D', 'A+B+C']
line_color = ['k', 'r', 'g', 'b', 'm']

plt.figure()
ax = plt.subplot(111)

for mm in range( 5 ):
	with h5py.File(img_lis[mm], 'r') as f:
		tt_img = np.array(f['a'])

	with h5py.File(pro_lis[mm], 'r') as f:
		c_r_arr = np.array(f['r'])
		c_sb_arr = np.array(f['sb'])
		c_sb_err = np.array(f['sb_err'])

	id_nan = np.isnan(tt_img)
	idvx = id_nan == False
	idy, idx = np.where(idvx == True)
	x_low, x_up = np.min(idx), np.max(idx)
	y_low, y_up = np.min(idy), np.max(idy)

	dpt_img = tt_img[y_low: y_up+1, x_low: x_up + 1]
	img_block = cc_grid_img(dpt_img, 100, 100,)[0]

	fig = plt.figure( figsize = (13.12, 4.8) )
	ax1 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
	ax2 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

	tg = ax1.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = ax1, fraction = 0.035, pad = 0.01,)
	cb.formatter.set_powerlimits((0,0))

	ax2.plot(c_r_arr, c_sb_arr, ls = '-', color = 'r', alpha = 0.8, label = line_label[mm],)
	ax2.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'r', alpha = 0.2,)

	ax2.set_ylim(1e-3, 1e-2)
	ax2.set_yscale('log')
	ax2.set_xlim(5e1, 1e3)
	ax2.set_xlabel('R [arcsec]')
	ax2.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax2.set_xscale('log')
	ax2.legend(loc = 1, frameon = False, fontsize = 8)
	ax2.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax2.tick_params(axis = 'both', which = 'both', direction = 'in',)
	tick_form = mpl.ticker.LogFormatter(labelOnlyBase = False)
	ax2.get_yaxis().set_minor_formatter(tick_form)

	plt.savefig('grid_2D_img_SB_%d.png' % mm, dpi = 300)
	plt.close()

	ax.plot(c_r_arr, c_sb_arr, ls = '-', color = line_color[mm], alpha = 0.8, label = line_label[mm],)
	ax.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = line_color[mm], alpha = 0.2,)

	if mm != 3:
		idr = np.abs( c_r_arr - 600)
		idrx = np.where( idr == idr.min() )[0]
		idsb = c_sb_arr[idrx]
		devi_sb = c_sb_arr - idsb
		ax.axhline(y = idsb, ls = ':', color = line_color[mm], alpha = 0.5,)
		ax.plot(c_r_arr, devi_sb, ls = '--', color = line_color[mm], alpha = 0.8,)
		ax.fill_between(c_r_arr, y1 = devi_sb - c_sb_err, y2 = devi_sb + c_sb_err, color = line_color[mm], alpha = 0.2,)

ax.set_ylim(1e-4, 3e-2)
ax.set_yscale('log')
ax.set_xlim(1e1, 1e3)
ax.set_xlabel('R [arcsec]')
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 3, frameon = False, fontsize = 8)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in',)
tick_form = mpl.ticker.LogFormatter(labelOnlyBase = False)
ax.get_yaxis().set_minor_formatter(tick_form)

plt.savefig('sample_SB_compare.png', dpi = 300)
plt.close()
"""

