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

from img_stack import stack_func
from light_measure import light_measure_Z0_weit
from light_measure import jack_SB_func
from light_measure import cc_grid_img, grid_img

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

def err_mean(err_array, R_array, N_sample,):

	L_arr = [len(ll) for ll in err_array]
	min_L = np.min(L_arr)

	dx_r, dy_err = [], []
	for mm in range(N_sample):
		dx_r.append( R_array[mm][-min_L:] )
		dy_err.append( err_array[mm][-min_L:] )
	dx_r = np.array(dx_r)
	dy_err = np.array(dy_err)

	Len = np.zeros( min_L, dtype = np.float32)
	for nn in range( min_L ):
		tmp_I = dy_err[:,nn]
		idnn = np.isnan(tmp_I)
		Len[nn] = N_sample - np.sum(idnn)

	Stack_R = np.nanmean(dx_r, axis = 0)
	Stack_err = np.nanmean(dy_err, axis = 0)
	std_Stack_err = np.nanstd(dy_err, axis = 0)

	### limit the radius bin contribution at least 1/3 * N_sample
	id_one = Len > 1
	Stack_R = Stack_R[ id_one ]
	Stack_err = Stack_err[ id_one ]
	std_Stack_err = std_Stack_err[ id_one ]
	N_img = Len[ id_one ]
	jk_Stack_err = np.sqrt(N_img - 1) * std_Stack_err

	id_min = N_img >= np.int(N_sample / 3)
	lim_r = Stack_R[id_min]
	lim_R = np.nanmax(lim_r)

	return Stack_R, Stack_err, jk_Stack_err

######### sample selection test
from img_block_select import diffuse_identi_func
#from cc_block_select import diffuse_identi_func

thres_S0, thres_S1 = 3, 5
sigma = np.array([3.5, 4, 4.5, 5, 5.5, 6,]) ## sigma as limit
home = '/media/xkchen/My Passport/data/SDSS/'
d_file = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'
id_sigm = 1

'''
mu_sigm_file = pds.read_csv('/home/xkchen/mywork/ICL/code/SEX/result/img_test-1000_mean_sigm.csv')

# A type
dat = pds.read_csv('SEX/result/test_1000-to-250_cat-match.csv')
set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

rule_file = 'test_1000-to-250_rule-out_cat_%.1f-sigma.csv' % ( sigma[id_sigm] )
remain_file = 'test_1000-to-250_remain_cat_%.1f-sigma.csv' % ( sigma[id_sigm] )
diffuse_identi_func(band[0], set_ra, set_dec, set_z, d_file, rule_file, remain_file, thres_S0, thres_S1, sigma[id_sigm],
					mu_sigm_file, id_single = True,)

# B type
dat = pds.read_csv('SEX/result/test_1000-to-98_cat.csv')
set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

rule_file = 'test_1000-to-98_rule-out_cat_%.1f-sigma.csv' % ( sigma[id_sigm] )
remain_file = 'test_1000-to-98_remain_cat_%.1f-sigma.csv' % ( sigma[id_sigm] )
diffuse_identi_func(band[0], set_ra, set_dec, set_z, d_file, rule_file, remain_file, thres_S0, thres_S1, sigma[id_sigm],
					mu_sigm_file, id_single = True,)

# C type
dat = pds.read_csv('SEX/result/test_1000-to-193_cat.csv')
set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

rule_file = 'test_1000-to-193_rule-out_cat_%.1f-sigma.csv' % ( sigma[id_sigm] )
remain_file = 'test_1000-to-193_remain_cat_%.1f-sigma.csv' % ( sigma[id_sigm] )
diffuse_identi_func(band[0], set_ra, set_dec, set_z, d_file, rule_file, remain_file, thres_S0, thres_S1, sigma[id_sigm],
					mu_sigm_file, id_single = True,)

# D type
dat = pds.read_csv('SEX/result/test_1000-to-459_cat.csv')
set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

rule_file = 'test_1000-to-459_rule-out_cat_%.1f-sigma.csv' % ( sigma[id_sigm] )
remain_file = 'test_1000-to-459_remain_cat_%.1f-sigma.csv' % ( sigma[id_sigm] )
diffuse_identi_func(band[0], set_ra, set_dec, set_z, d_file, rule_file, remain_file, thres_S0, thres_S1, sigma[id_sigm],
					mu_sigm_file, id_single = True,)
raise
'''
## PS:
## 		test*_cat -- select imgs based on (mu_cen, sigma_cen) of single img
## 		alt_test*_cat -- select imgs based on mean of (mu_cen, sigma_cen) for given img sample
## 		CC_test*_cat -- select imgs based on mean of (mu_cen, sigma_cen) for given img sample, 
## 						and no limit on sub-patches 2D hist (broadly case)
## 		MM_test*_cat -- select imgs based on mode point of (mu_cen, sigma_cen) for given img sample, 
## 						and no limit on sub-patches 2D hist (broadly case)
'''
n_main = np.array([250, 98, 193, 459])

n_eta = np.zeros(4, dtype = np.float32)
p_eta = np.zeros(4, dtype = np.float32)
a_eta = np.zeros(4, dtype = np.float32)
m_eta = np.zeros(4, dtype = np.float32)

for mm in range(4):
	if mm == 0:
		dat = pds.read_csv('SEX/result/test_1000-to-250_cat-match.csv')
		set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	else:
		dat = pds.read_csv('SEX/result/test_1000-to-%d_cat.csv' % (n_main[mm]) )
		set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

	pre_dat = pds.read_csv('SEX/result/4sigma/test_1000-to-%d_remain_cat_%.1f-sigma.csv' % (n_main[mm], sigma[id_sigm] ), )
	pre_ra = np.array(pre_dat.ra)
	p_eta[mm] = pre_ra.shape[0] / set_ra.shape[0]

	alt_dat = pds.read_csv('SEX/result/4sigma/alt_test_1000-to-%d_remain_cat_%.1f-sigma.csv' % (n_main[mm], sigma[id_sigm] ), )
	alt_ra = np.array(alt_dat.ra)
	a_eta[mm] = alt_ra.shape[0] / set_ra.shape[0]

	code_dat = pds.read_csv('SEX/result/4sigma/CC_test_1000-to-%d_remain_cat_%.1f-sigma.csv' % (n_main[mm], sigma[id_sigm] ), )
	cod_ra = np.array(code_dat.ra)
	n_eta[mm] = cod_ra.shape[0] / set_ra.shape[0]

	mm_dat = pds.read_csv('SEX/result/4sigma/MM_test_1000-to-%d_remain_cat_%.1f-sigma.csv' % (n_main[mm], sigma[id_sigm] ), )
	mm_ra = np.array(mm_dat.ra)
	m_eta[mm] = mm_ra.shape[0] / set_ra.shape[0]

xpot = np.array([1,2,3,4])
plt.figure()
ax = plt.subplot(111)
#ax.set_title('search on total image region')
ax.set_title('search on image edge(500pixel) region')

ax.plot(xpot, p_eta, 'bs', alpha = 0.5, label = 'single img[$\\sigma_{cen}, \\mu_{cen}$]')
ax.plot(xpot, a_eta, 'g^', alpha = 0.5, label = 'sample imgs[$\\bar{\\sigma}_{cen}, \\bar{\\mu}_{cen}$]')
ax.plot(xpot, n_eta, 'ro', alpha = 0.5, label = 'sample imgs[$\\bar{\\sigma}_{cen}, \\bar{\\mu}_{cen}$] + broadly case')
ax.plot(xpot, m_eta, 'c*', alpha = 0.5, label = 'sample imgs[$Mode(\\sigma_{cen}), Mode(\\mu_{cen})$] + broadly case')

ax.legend(loc = 3, frameon = False,)
ax.set_ylim(-0.1, 1.1)
ax.set_xlim(0.5, 4.5)
ax.set_xticks(xpot)
ax.set_xticklabels(['A', 'B', 'C', 'D'])
ax.set_xlabel('sample type')
ax.set_ylabel('$ N_{normal} / N_{total} $')
plt.savefig('img_select_compare.png', dpi = 300)
plt.show()
'''

'''
## test the stacking SB pros.
n_main = np.array([250, 98, 193, 459])
home = '/media/xkchen/My Passport/data/SDSS/'
dfile = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'
id_cen = 0 # BCG-stacking

com_ra, com_dec, com_z = np.array([0]), np.array([0]), np.array([0])
com_imgx, com_imgy = np.array([0]), np.array([0])
for mm in range(4):
	if mm == 2:
		continue
	else:
		#dat = pds.read_csv('test_1000-to-%d_remain_cat_%.1f-sigma.csv' % (n_main[mm], sigma[id_sigm] ),)
		#dat = pds.read_csv('alt_test_1000-to-%d_remain_cat_%.1f-sigma.csv' % (n_main[mm], sigma[id_sigm] ),)
		#dat = pds.read_csv('CC_test_1000-to-%d_remain_cat_%.1f-sigma.csv' % (n_main[mm], sigma[id_sigm] ),)
		dat = pds.read_csv('MM_test_1000-to-%d_remain_cat_%.1f-sigma.csv' % (n_main[mm], sigma[id_sigm] ),)

		t_ra, t_dec, t_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
		t_imgx, t_imgy = np.array(dat.bcg_x), np.array(dat.bcg_y)

		com_ra = np.r_[ com_ra, t_ra]
		com_dec = np.r_[ com_dec, t_dec]
		com_z = np.r_[ com_z, t_z]
		com_imgx = np.r_[ com_imgx, t_imgx]
		com_imgy = np.r_[ com_imgy, t_imgy]

com_ra = com_ra[1:]
com_dec = com_dec[1:]
com_z = com_z[1:]
com_imgx = com_imgx[1:]
com_imgy = com_imgy[1:]

tot_file = 'trained_ABCD_%d.h5' % (len(com_ra),)
tot_pix_cont = 'trained_pix-cont_%d.h5' % (len(com_ra),)
stack_func(dfile, tot_file, com_z, com_ra, com_dec, band[0], com_imgx, com_imgy, id_cen, rms_file = None, pix_con_file = tot_pix_cont,)
raise
'''

"""
N_lis = [700, 703, 667, 290]
#N_lis = [760, 743, 712, 345]
legend_lis = ['single img[$\\sigma_{cen}, \\mu_{cen}$]', 'sample imgs[$\\bar{\\sigma}_{cen}, \\bar{\\mu}_{cen}$]',
'sample imgs[$\\bar{\\sigma}_{cen}, \\bar{\\mu}_{cen}$] + broadly case', 'sample imgs[$Mode(\\sigma_{cen}), Mode(\\mu_{cen})$] + broadly case']
'''
for nn in range(4):
	with h5py.File('SEX/result/trained_ABCD_%d.h5' % N_lis[nn], 'r') as f:
		tmp_img = np.array(f['a'])
	with h5py.File('SEX/result/trained_pix-cont_%d.h5' % N_lis[nn], 'r') as f:
		tmp_cont = np.array(f['a'])

	xn, yn = np.int(tmp_img.shape[1] / 2), np.int(tmp_img.shape[0] / 2)

	id_nn = np.isnan(tmp_img)
	eff_y, eff_x = np.where(id_nn == False)
	dR = np.sqrt((eff_y - yn)**2 + (eff_x - xn)**2)
	dR_max = np.int( dR.max() ) + 1
	rbins = np.logspace(0, np.log10(dR_max), 110)

	c_sb_arr, c_r_arr, c_sb_err_arr, c_npix, c_nratio = light_measure_Z0_weit(tmp_img, tmp_cont, pixel, xn, yn, rbins)
	idzo = c_npix < 1
	c_r_arr, c_sb_arr, c_sb_err_arr = c_r_arr[idzo == False], c_sb_arr[idzo == False], c_sb_err_arr[idzo == False]
	c_sb_arr, c_sb_err_arr = c_sb_arr / pixel**2, c_sb_err_arr / pixel**2

	with h5py.File('SB_trained_ABCD_%d.h5' % N_lis[nn], 'w') as f:
		f['r'] = np.array(c_r_arr)
		f['sb'] = np.array(c_sb_arr)
		f['sb_err'] = np.array(c_sb_err_arr)
		f['nratio'] = np.array(c_nratio[idzo == False])
		f['npix'] = np.array(c_npix[idzo == False])
'''
plt.figure()
ax = plt.subplot(111)

for nn in range(4):

	with h5py.File('SEX/result/trained_ABCD_%d.h5' % N_lis[nn], 'r') as f:
		tmp_img = np.array(f['a'])
	block_m = grid_img(tmp_img, 100, 100)[0]

	with h5py.File('SEX/result/SB_trained_ABCD_%d.h5' % N_lis[nn], 'r') as f:
		c_r_arr = np.array(f['r'])
		c_sb_arr = np.array(f['sb'])
		c_sb_err = np.array(f['sb_err'])

	fig = plt.figure( figsize = (13.12, 4.8) )
	ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
	ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

	tg = ax0.imshow(block_m / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits((0,0))

	ax1.plot(c_r_arr, c_sb_arr, ls = '-', alpha = 0.8, color = 'r', label = legend_lis[nn],)
	ax1.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'r', alpha = 0.2,)
	ax1.set_ylim(1e-3, 7e-3)
	ax1.set_xlim(5e1, 1e3)
	ax1.set_xscale('log')
	ax1.set_xlabel('$ R[arcsec] $')
	ax1.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax1.legend(loc = 1, frameon = False,)
	ax1.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax1.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.savefig('trained_2D_img_%d.png' % N_lis[nn], dpi = 300)
	plt.close()

	ax.plot(c_r_arr, c_sb_arr, ls = '-', alpha = 0.8, color = mpl.cm.rainbow(nn / 4), label = legend_lis[nn],)
	ax.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = mpl.cm.rainbow(nn / 4), alpha = 0.2,)

	idr = np.abs( c_r_arr - 500)
	idrx = np.where( idr == idr.min() )[0]
	idsb = c_sb_arr[idrx + 1]
	devi_sb = c_sb_arr - idsb

	ax.axhline(y = idsb, ls = ':', alpha = 0.5, color = mpl.cm.rainbow(nn / 4),)
	ax.plot(c_r_arr, devi_sb, ls = '--', alpha = 0.8, color = mpl.cm.rainbow(nn / 4),)
	ax.fill_between(c_r_arr, y1 = devi_sb - c_sb_err, y2 = devi_sb + c_sb_err, color = mpl.cm.rainbow(nn / 4), alpha = 0.2,)
'''
# sub-samples without selection
#with h5py.File('SEX/result/test_BCG-stack_SB-250.h5', 'r') as f:
#with h5py.File('SEX/result/combine_A-B_SB_348.h5', 'r') as f:
with h5py.File('SEX/result/combine_A-B-C_SB_541.h5', 'r') as f:
	t_r = np.array(f['r'])
	t_sb = np.array(f['sb'])
	t_sb_err = np.array(f['sb_err'])

#with h5py.File('SEX/result/sub-BG_BCG-stack_SB-250.h5', 'r') as f:
#with h5py.File('SEX/result/sub-BG_combine_A-B_SB_348.h5', 'r') as f:
with h5py.File('SEX/result/sub-BG_combine_A-B-C_SB_541.h5', 'r') as f:
	d_r = np.array(f['r'])
	d_sb = np.array(f['sb'])
	d_sb_err = np.array(f['sb_err'])
	d_bg = np.array(f['BG'])

ax.plot(t_r, t_sb, ls = '-', color = 'k', alpha = 0.8, label = 'sample-541[A+B+C]',)
ax.fill_between(t_r, y1 = t_sb - t_sb_err, y2 = t_sb + t_sb_err, color = 'k', alpha = 0.2,)

ax.plot(d_r, d_sb, ls = '--', color = 'k', alpha = 0.8,)
ax.fill_between(d_r, y1 = d_sb - d_sb_err, y2 = d_sb + d_sb_err, color = 'k', alpha = 0.2,)
ax.axhline(y = d_bg[0], ls = ':', color = 'k', alpha = 0.5,)
'''
ax.set_ylim(1e-4, 3e-2)
ax.set_yscale('log')
ax.set_xlim(1e1, 1e3)
ax.set_xlabel('$ R[arcsec] $')
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 3, frameon = False, fontsize = 8)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in')

plt.subplots_adjust(left = 0.15, right = 0.95,)
plt.savefig('img_select_SB_test.png', dpi = 300)
plt.close()
raise

## applied to sample-3100
pro_lis = [home + '20_10_test/jack_test/tot_clust_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2.h5',
			home + '20_10_test/jack_test/T1000_clust_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2.h5']
img_lis = [home + '20_10_test/jack_test/tot_clust_BCG-stack_Mean_jack_img_30-FWHM-ov2.h5',
			home + '20_10_test/jack_test/T1000_clust_BCG-stack_Mean_jack_img_30-FWHM-ov2.h5']
label_lis = ['total[1122/3100]', 'test 1000[290/1000]']
jk_r, jk_sb, jk_err = [], [], []

for nn in range(2):

	with h5py.File(img_lis[nn], 'r') as f:
		tt_img = np.array(f['a'])
	tt_block =  grid_img(tt_img, 100, 100)[0]

	with h5py.File(pro_lis[nn], 'r') as f:
		tt_jk_r = np.array(f['r'])
		tt_jk_sb = np.array(f['sb'])
		tt_jk_err = np.array(f['sb_err'])
	jk_r.append(tt_jk_r)
	jk_sb.append(tt_jk_sb)
	jk_err.append(tt_jk_err)

	fig = plt.figure( figsize = (13.12, 4.8) )
	a_ax0 = fig.add_axes([0.05, 0.09, 0.40, 0.80])
	a_ax1 = fig.add_axes([0.55, 0.09, 0.40, 0.80])

	tg = a_ax0.imshow(tt_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = a_ax0, fraction = 0.035, pad = 0.01,)
	cb.formatter.set_powerlimits((0,0))

	a_ax1.errorbar(tt_jk_r, tt_jk_sb, yerr = tt_jk_err, xerr = None, color = 'r', 
		marker = 'None', ls = '-', ecolor = 'r', alpha = 0.5, label = label_lis[nn],)
	a_ax1.set_ylim(1e-3, 7e-3)
	a_ax1.set_xlim(5e1, 1e3)
	a_ax1.set_xlabel('$ R[arcsec] $')
	a_ax1.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	a_ax1.set_xscale('log')
	a_ax1.legend(loc = 1, frameon = False, fontsize = 8)
	a_ax1.grid(which = 'both', axis = 'both', alpha = 0.25)
	a_ax1.tick_params(axis = 'both', which = 'both', direction = 'in',)
	a_ax1.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)

	plt.savefig('grid_2D_SB_%d.png' % nn, dpi = 300)
	plt.close()

plt.figure()
ax = plt.subplot(111)

ax.plot(jk_r[0], jk_sb[0], color = 'r', ls = '-', alpha = 0.8, label = label_lis[0],)
ax.fill_between(jk_r[0], y1 = jk_sb[0] - jk_err[0], y2 = jk_sb[0] + jk_err[0], color = 'r', alpha = 0.2,)

idr = np.abs( jk_r[0] - 400 )
idrx = np.where( idr == idr.min() )[0]
idsb = jk_sb[0][idrx + 1]
devi_sb = jk_sb[0] - idsb

ax.plot(jk_r[0], devi_sb, color = 'r', ls= '--', alpha = 0.8, )
ax.fill_between(jk_r[0], y1 = devi_sb - jk_err[0], y2 = devi_sb + jk_err[0], color = 'r', alpha = 0.2,)
ax.axhline(y = idsb, ls = ':', color = 'r', alpha = 0.5,)

ax.plot(jk_r[1], jk_sb[1], color = 'b', ls = '-', alpha = 0.8, label = label_lis[1],)
ax.fill_between(jk_r[1], y1 = jk_sb[1] - jk_err[1], y2 = jk_sb[1] + jk_err[1], color = 'b', alpha = 0.2,)

idr = np.abs( jk_r[1] - 400 )
idrx = np.where( idr == idr.min() )[0]
idsb = jk_sb[1][idrx + 2]
devi_sb = jk_sb[1] - idsb

ax.plot(jk_r[1], devi_sb, color = 'b', ls= '--', alpha = 0.8, )
ax.fill_between(jk_r[1], y1 = devi_sb - jk_err[1], y2 = devi_sb + jk_err[1], color = 'b', alpha = 0.2,)
ax.axhline(y = idsb, ls = ':', color = 'b', alpha = 0.5,)

ax.set_ylim(1e-4, 3e-2)
ax.set_yscale('log')
ax.set_xlim(1e1, 1e3)
ax.set_xlabel('$ R[arcsec] $')
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 3, frameon = False,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in')

plt.subplots_adjust(left = 0.15, right = 0.95,)
plt.savefig('selection_img_SB_test.png', dpi = 300)
plt.close()
"""

########################## pre-select sub-samples and SB pros. compare
load = '/media/xkchen/My Passport/data/SDSS/'

### A or A+B type imgs (at z_ref)
size_arr = np.array([5, 30])
'''
jack_SB_file = load + '20_10_test/jack_test/AB_clust_BCG-stack_Mean_jack_SB-pro_%d-FWHM-ov2_z-ref.h5'
jack_img = load + '20_10_test/jack_test/AB_clust_BCG-stack_Mean_jack_img_%d-FWHM-ov2_z-ref.h5'
jack_cont_arr = load + '20_10_test/jack_test/AB_clust_BCG-stack_Mean_jack_pix-cont_%d-FWHM-ov2_z-ref.h5'
'''
jack_SB_file = load + '20_10_test/jack_test/A_clust_BCG-stack_Mean_jack_SB-pro_%d-FWHM-ov2_z-ref.h5'
jack_img = load + '20_10_test/jack_test/A_clust_BCG-stack_Mean_jack_img_%d-FWHM-ov2_z-ref.h5'
jack_cont_arr = load + '20_10_test/jack_test/A_clust_BCG-stack_Mean_jack_pix-cont_%d-FWHM-ov2_z-ref.h5'

plt.figure()
ax = plt.subplot(111)

r_meth, sb_meth = [], []
err0_meth, err1_meth = [], []

for mm in range( 2 ):
	with h5py.File( jack_img % size_arr[mm], 'r') as f:
		tt_img = np.array(f['a'])
	img_block, grd_pix = grid_img(tt_img, 100, 100)[:2]

	with h5py.File( jack_SB_file % (size_arr[mm]), 'r') as f:
		c_r_arr = np.array(f['r'])
		c_sb_arr = np.array(f['sb'])
		c_sb_err = np.array(f['sb_err'])

	fig = plt.figure( figsize = (19.84, 4.8) )
	a_ax0 = fig.add_axes([0.03, 0.09, 0.30, 0.85])
	a_ax1 = fig.add_axes([0.38, 0.09, 0.30, 0.85])
	a_ax2 = fig.add_axes([0.73, 0.09, 0.25, 0.85])

	tf = a_ax0.imshow(grd_pix, origin = 'lower', cmap = 'rainbow', vmin = 1, vmax = grd_pix.max(),)
	cb = plt.colorbar(tf, ax = a_ax0, fraction = 0.035, pad = 0.01, label = 'pixel count',)
	tf.cmap.set_under('white')
	cb.formatter.set_powerlimits((0,0))

	idux = (grd_pix <= 500) & (grd_pix > 0)
	poy, pox = np.where(idux == True)
	for ll in range( np.sum(idux)):

		a_ax0.text(pox[ll], poy[ll], s = '%d' % (grd_pix[ poy[ll],pox[ll] ]), fontsize = 4, color = 'w', ha = 'center',)

	tg = a_ax1.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = a_ax1, fraction = 0.035, pad = 0.01,)
	cb.formatter.set_powerlimits((0,0))

	a_ax2.plot(c_r_arr, c_sb_arr, ls = '-', color = 'r', alpha = 0.8, label = '%d $ FWHM / 2 $' % size_arr[mm],)
	a_ax2.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'r', alpha = 0.2,)

	a_ax2.set_ylim(1e-3, 3e-2)
	a_ax2.set_yscale('log')
	a_ax2.set_xlim(5e1, 4e3)
	a_ax2.set_xlabel('R [kpc]')
	a_ax2.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	a_ax2.set_xscale('log')
	a_ax2.legend(loc = 1, frameon = False, fontsize = 8)
	a_ax2.grid(which = 'both', axis = 'both', alpha = 0.25)
	a_ax2.tick_params(axis = 'both', which = 'both', direction = 'in',)
	#a_ax2.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)
	tick_form = mpl.ticker.LogFormatter(labelOnlyBase = False)
	a_ax2.get_yaxis().set_minor_formatter(tick_form)

	plt.savefig('grid_2D_SB_%d-FWHM-ov2.png' % size_arr[mm], dpi = 300)
	plt.close()

	ax.plot(c_r_arr, c_sb_arr, ls = '-', alpha = 0.8, color = mpl.cm.rainbow(mm / 5), label = '%d $ FWHM / 2 $' % size_arr[mm],)
	ax.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = mpl.cm.rainbow(mm / 5), alpha = 0.2,)

	idr = np.abs( c_r_arr - 2000 )
	idrx = np.where( idr == idr.min() )[0]
	idsb = c_sb_arr[idrx] #[idrx + 2]
	devi_sb = c_sb_arr - idsb

	ax.axhline(y = idsb, ls = ':', alpha = 0.5, color = mpl.cm.rainbow(mm / 5),)
	ax.plot(c_r_arr, devi_sb, ls = '--', alpha = 0.8, color = mpl.cm.rainbow(mm / 5),)
	ax.fill_between(c_r_arr, y1 = devi_sb - c_sb_err, y2 = devi_sb + c_sb_err, color = mpl.cm.rainbow(mm / 5), alpha = 0.2,)

	## for compare observation
	SB_devi = 22.5 - 2.5 * np.log10(devi_sb) + mag_add[0]
	dSB0 = 22.5 - 2.5 * np.log10(devi_sb + c_sb_err) + mag_add[0]
	dSB1 = 22.5 - 2.5 * np.log10(devi_sb - c_sb_err) + mag_add[0]
	err0 = SB_devi - dSB0
	err1 = dSB1 - SB_devi

	id_nan = np.isnan(SB_devi)
	SB_out, R_out = SB_devi[id_nan == False], c_r_arr[id_nan == False]
	out_err0, out_err1 = err0[id_nan == False], err1[id_nan == False]
	dSB0, dSB1 = dSB0[id_nan == False], dSB1[id_nan == False]
	idx_nan = np.isnan(dSB1)
	out_err1[idx_nan] = 100.

	r_meth.append(R_out)
	sb_meth.append(SB_out)
	err0_meth.append(out_err0)
	err1_meth.append(out_err1)

ax.axvline(x = 1000 / h, ls = '--', color = 'k', alpha = 0.5, label = '1 Mpc / h')

ax.set_ylim(1e-4, 3e-2)
ax.set_yscale('log')
ax.set_xlim(5e1, 4e3)
ax.set_xlabel('R [kpc]')
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 3, frameon = False,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in')

plt.subplots_adjust(left = 0.15, right = 0.95,)
plt.savefig('SB_mask_size_test.png', dpi = 300)
plt.close()

SB_tt = pds.read_csv('/home/xkchen/mywork/ICL/Zibetti_SB/r_band_BCG_ICL.csv')
R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
R_obs = R_obs**4

plt.figure()
ax = plt.subplot(111)
for mm in range(2):
	ax.plot(r_meth[mm], sb_meth[mm], ls = '-', alpha = 0.8, color = mpl.cm.rainbow(mm / 5), label = '%d $ FWHM / 2 $' % size_arr[mm],)
	ax.fill_between(r_meth[mm], y1 = sb_meth[mm] - err0_meth[mm], y2 = sb_meth[mm] + err1_meth[mm], color = mpl.cm.rainbow(mm / 5), alpha = 0.2,)

ax.plot(R_obs, SB_obs, color = 'r', ls = '-', alpha = 0.5, label = 'Zibetti',)
ax.set_xlabel('$R[kpc]$')
ax.set_ylabel('$SB[mag / arcsec^2]$')
ax.set_xscale('log')
ax.set_ylim(25, 34)
ax.set_xlim(5e1, 2e3)
ax.legend(loc = 1, frameon = False)
ax.invert_yaxis()
ax.grid(which = 'both', axis = 'both')
ax.tick_params(axis = 'both', which = 'both', direction = 'in')

plt.subplots_adjust(left = 0.15, right = 0.95,)
plt.savefig('Z05_compare.png', dpi = 300)
plt.close()

raise

'''
## masking size test (A or A+B type imgs)
size_arr = np.array([5, 10, 15, 20, 25, 30])

plt.figure()
ax = plt.subplot(111)

for mm in range(6):

	with h5py.File(load + '20_10_test/jack_test/clust_BCG-stack_Mean_jack_img_%d-FWHM-ov2.h5' % (size_arr[mm]), 'r') as f:
	#with h5py.File(load + '20_10_test/jack_test/AB_clust_BCG-stack_Mean_jack_img_%d-FWHM-ov2.h5' % (size_arr[mm]), 'r') as f:
		tt_img = np.array(f['a'])
	img_block, grd_pix = grid_img(tt_img, 100, 100)[:2]

	with h5py.File(load + '20_10_test/jack_test/clust_BCG-stack_Mean_jack_SB-pro_%d-FWHM-ov2.h5' % (size_arr[mm]), 'r') as f:
	#with h5py.File(load + '20_10_test/jack_test/AB_clust_BCG-stack_Mean_jack_SB-pro_%d-FWHM-ov2.h5' % (size_arr[mm]), 'r') as f:
		c_r_arr = np.array(f['r'])
		c_sb_arr = np.array(f['sb'])
		c_sb_err = np.array(f['sb_err'])

	fig = plt.figure( figsize = (19.84, 4.8) )
	a_ax0 = fig.add_axes([0.03, 0.09, 0.30, 0.85])
	a_ax1 = fig.add_axes([0.38, 0.09, 0.30, 0.85])
	a_ax2 = fig.add_axes([0.73, 0.09, 0.25, 0.85])

	tf = a_ax0.imshow(grd_pix, origin = 'lower', cmap = 'rainbow', vmin = 1, vmax = grd_pix.max(),)
	cb = plt.colorbar(tf, ax = a_ax0, fraction = 0.035, pad = 0.01, label = 'pixel count',)
	tf.cmap.set_under('white')
	cb.formatter.set_powerlimits((0,0))

	idux = (grd_pix <= 500) & (grd_pix > 0)
	poy, pox = np.where(idux == True)
	for ll in range( np.sum(idux)):

		a_ax0.text(pox[ll], poy[ll], s = '%d' % (grd_pix[ poy[ll],pox[ll] ]), fontsize = 4, color = 'w', ha = 'center',)

	tg = a_ax1.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = a_ax1, fraction = 0.035, pad = 0.01,)
	cb.formatter.set_powerlimits((0,0))

	a_ax2.plot(c_r_arr, c_sb_arr, ls = '-', color = 'r', alpha = 0.8, label = '%d $ FWHM / 2 $' % size_arr[mm],)
	a_ax2.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'r', alpha = 0.2,)

	a_ax2.set_ylim(1e-3, 7e-3)
	a_ax2.set_xlim(5e1, 1e3)
	a_ax2.set_xlabel('$ R[arcsec] $')
	a_ax2.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	a_ax2.set_xscale('log')
	a_ax2.legend(loc = 1, frameon = False, fontsize = 8)
	a_ax2.grid(which = 'both', axis = 'both', alpha = 0.25)
	a_ax2.tick_params(axis = 'both', which = 'both', direction = 'in',)
	a_ax2.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)

	plt.savefig('grid_2D_SB_%d-FWHM-ov2.png' % size_arr[mm], dpi = 300)
	plt.close()

	ax.plot(c_r_arr, c_sb_arr, ls = '-', alpha = 0.8, color = mpl.cm.rainbow(mm / 5), label = '%d $ FWHM / 2 $' % size_arr[mm],)
	ax.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = mpl.cm.rainbow(mm / 5), alpha = 0.2,)

	idr = np.abs( c_r_arr - 600 )
	idrx = np.where( idr == idr.min() )[0]
	idsb = c_sb_arr[idrx]
	devi_sb = c_sb_arr - idsb

	ax.axhline(y = idsb, ls = ':', alpha = 0.5, color = mpl.cm.rainbow(mm / 5),)
	ax.plot(c_r_arr, devi_sb, ls = '--', alpha = 0.8, color = mpl.cm.rainbow(mm / 5),)
	ax.fill_between(c_r_arr, y1 = devi_sb - c_sb_err, y2 = devi_sb + c_sb_err, color = mpl.cm.rainbow(mm / 5), alpha = 0.2,)

ax.set_ylim(1e-4, 3e-2)
ax.set_yscale('log')
ax.set_xlim(1e1, 1e3)
ax.set_xlabel('$ R[arcsec] $')
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 3, frameon = False,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in')

plt.subplots_adjust(left = 0.15, right = 0.95,)
plt.savefig('SB_mask_size_test.png', dpi = 300)
plt.close()

err_R, err_trac, err_trac_std = [], [], []
for mm in range(6):

	#sub_err_file = load + '20_10_test/jack_test/clust_BCG-stack_jack-sub-%d_SB-pro' + '_%d-FWHM-ov2.h5' % (size_arr[mm])
	sub_err_file = load + '20_10_test/jack_test/AB_clust_BCG-stack_jack-sub-%d_SB-pro' + '_%d-FWHM-ov2.h5' % (size_arr[mm])
	tmp_r, tmp_err = [], []
	for nn in range(30):
		with h5py.File(sub_err_file % nn, 'r') as f:
			sub_r = np.array(f['r'])
			sub_err = np.array(f['sb_err'])
			sub_npix = np.array(f['npix'])

			idv = sub_npix > 1
			sub_r = sub_r[idv]
			sub_err = sub_err[idv]
			sub_npix = sub_npix[idv]

		tmp_r.append(sub_r)
		tmp_err.append(sub_err)

	jk_r, jk_M_err, jk_err_std = err_mean(tmp_err, tmp_r, 30,)

	err_R.append(jk_r)
	err_trac.append(jk_M_err)
	err_trac_std.append(jk_err_std)

R_size = np.array([5, 10, 15, 20, 25, 30])

plt.figure()
gs = gridspec.GridSpec(2,1, height_ratios = [4,1])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

n_min = len(err_R[-1])
for mm in range(6):

	ax0.plot(err_R[mm], err_trac[mm], color = mpl.cm.rainbow(mm / 6), alpha = 0.8, label = '%d $ FWHM / 2 $' % R_size[mm],)
	ax0.fill_between(err_R[mm], y1 = err_trac[mm] - err_trac_std[mm], y2 = err_trac[mm] + err_trac_std[mm], 
		color = mpl.cm.rainbow(mm / 6), alpha = 0.2 )

	ax1.plot(err_R[mm][-n_min:], err_trac[mm][-n_min:] / err_trac[-1], color = mpl.cm.rainbow(mm / 6), alpha = 0.8, )
	ax1.fill_between(err_R[mm][-n_min:], y1 = (err_trac[mm][-n_min:] - err_trac_std[mm][-n_min:]) / err_trac[-1], 
		y2 = (err_trac[mm][-n_min:] + err_trac_std[mm][-n_min:]) / err_trac[-1], color = mpl.cm.rainbow(mm / 6), alpha = 0.2 )	

ax0.set_xlim(1e1, 1e3)
ax0.set_xscale('log')
ax0.set_ylim(4e-5, 7e-3)
ax0.set_yscale('log')
ax0.set_ylabel('SB err [nanomaggies / arcsec^2]')
ax0.legend(loc = 'upper center', frameon = False,)
ax0.grid(which = 'both', axis = 'both', alpha = 0.25)
ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

ax1.set_xlim(ax0.get_xlim() )
ax1.set_ylim(0.75, 1.25)
ax1.set_ylabel('err / $ err_{30 (FWHM / 2)} $',)
ax1.set_xscale('log')
ax1.set_xlabel('$ R[arcsec] $')
ax1.grid(which = 'both', axis = 'both', alpha = 0.25)
ax1.tick_params(axis = 'both', which = 'both', direction = 'in')
ax0.set_xticks([])

plt.subplots_adjust(left = 0.15, right = 0.95, hspace = 0.02)
plt.savefig('size_test_err_compare.png', dpi = 300)
plt.show()
'''

"""
## cut stacking img test
#		individual sub-samples
N_cut = np.array([100, 200, 300, 400, 500])
lim_r1 = 2460
r_bins_1 = np.logspace(0, np.log10(lim_r1), 110)
Angl_r = r_bins_1 * pixel
medi_R = 0.5 * (Angl_r[1:] + Angl_r[:-1])

sub_img = load + '20_10_test/jack_test/clust_BCG-stack_sub-%d_img_30-FWHM-ov2.h5'
sub_pix_cont = load + '20_10_test/jack_test/clust_BCG-stack_sub-%d_pix-cont_30-FWHM-ov2.h5'
sub_sb = load + '20_10_test/jack_test/clust_BCG-stack_sub-%d_SB-pro_30-FWHM-ov2.h5'

with h5py.File(load + '20_10_test/jack_test/clust_BCG-stack_Mean_jack_img_30-FWHM-ov2.h5', 'r') as f:
	tot_img = np.array(f['a'])

with h5py.File(load + '20_10_test/jack_test/clust_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2.h5', 'r') as f:
	tot_r_arr = np.array(f['r'])
	tot_sb_arr = np.array(f['sb'])
	tot_sb_err = np.array(f['sb_err'])

idnn = np.isnan(tot_img)
idvx = idnn == False
idy, idx = np.where(idvx == True)
x_lft, x_rit = np.min(idx), np.max(idx)
y_botm, y_top = np.min(idy), np.max(idy)

dpt_tot = tot_img[y_botm: y_top+1, x_lft: x_rit + 1]
tot_block = cc_grid_img(dpt_tot, 100, 100,)[0]

for pp in range( 6 ):

	cc_stack_img = np.zeros((tot_img.shape[0], tot_img.shape[1]), dtype = np.float32)
	cc_pix_cont = np.zeros((tot_img.shape[0], tot_img.shape[1]), dtype = np.float32)

	for nn in range( pp * 5, (pp + 1) * 5 ):

		with h5py.File(sub_img % nn, 'r') as f:
			tmp_img = np.array(f['a'])
		with h5py.File(sub_pix_cont % nn, 'r') as f:
			tmp_cont = np.array(f['a'])

		id_nn = np.isnan( tmp_img )
		weit_img = tmp_img * tmp_cont
		cc_stack_img[id_nn == False] = cc_stack_img[id_nn == False] + weit_img[id_nn == False]
		cc_pix_cont[id_nn == False] = cc_pix_cont[id_nn == False] + tmp_cont[id_nn == False]

	id_zero = cc_pix_cont < 1.
	cc_stack_img[id_zero] = np.nan
	cc_pix_cont[id_zero] = np.nan
	cc_stack_img = cc_stack_img / cc_pix_cont

	with h5py.File('test/clust_BGC-stack_combin_sub-%d_img.h5' % pp, 'w') as f:
		f['a'] = np.array(cc_stack_img)
	with h5py.File('test/clust_BGC-stack_combin_sub-%d_pix-cont.h5' % pp, 'w') as f:
		f['a'] = np.array(cc_pix_cont)
	xn, yn = np.int(cc_stack_img.shape[1] / 2), np.int(cc_stack_img.shape[0] / 2)

	id_nan = np.isnan(cc_stack_img)
	idvx = id_nan == False
	idy, idx = np.where(idvx == True)
	x_low, x_up = np.min(idx), np.max(idx)
	y_low, y_up = np.min(idy), np.max(idy)

	dpt_img = cc_stack_img[y_low: y_up+1, x_low: x_up + 1]
	cen_x, cen_y = xn - x_low, yn - y_low
	test_grd = cc_grid_img(dpt_img, 100, 100,)
	img_block, grd_pix = test_grd[0], test_grd[1]
	Nlx, Nly = img_block.shape[1], img_block.shape[0]
	grd_x, grd_y = test_grd[-2], test_grd[-1]
	dpt_cont = cc_pix_cont[y_low: y_up+1, x_low: x_up + 1]	

	## compare with total imgs case
	Intns, Intns_r, Intns_err, npix, nratio = light_measure_Z0_weit(cc_stack_img, cc_pix_cont, pixel, xn, yn, r_bins_1)
	sb_arr, sb_err_arr = Intns / pixel**2, Intns_err / pixel**2
	r_arr = Intns_r * 1

	with h5py.File('test/clust_BGC-stack_combin_sub-%d_SB.h5' % pp, 'w') as f:
		f['r'] = np.array(r_arr)
		f['sb'] = np.array(sb_arr)
		f['sb_err'] = np.array(sb_err_arr)
		f['nratio'] = np.array(nratio)
		f['npix'] = np.array(npix)

	idvx = npix < 1.
	c_sb_arr = sb_arr[idvx == False]
	c_r_arr = r_arr[idvx == False]
	c_sb_err = sb_err_arr[idvx == False]

	diffi = tot_img - cc_stack_img
	dpt_diffi = diffi[y_low: y_up+1, x_low: x_up + 1]
	diffi_block = cc_grid_img(dpt_diffi, 100, 100,)[0]

	fig = plt.figure( figsize = (13.12, 9.84) )
	ax0 = fig.add_axes([0.05, 0.55, 0.40, 0.47])
	ax1 = fig.add_axes([0.55, 0.55, 0.40, 0.47])
	ax2 = fig.add_axes([0.05, 0.09, 0.40, 0.39])
	ax3 = fig.add_axes([0.55, 0.05, 0.40, 0.47])

	ax0.set_title('total')
	tg = ax0.imshow(tot_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = ax0, fraction = 0.036, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits((0,0))
	ax0.set_xlim(-2, tot_block.shape[1] + 1)
	ax0.set_ylim(-2, tot_block.shape[0] + 1)

	ax1.set_title('sub sample [%d]' % pp)
	tg = ax1.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = ax1, fraction = 0.036, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits((0,0))
	ax1.set_xlim(-2, img_block.shape[1] + 1)
	ax1.set_ylim(-2, img_block.shape[0] + 1)

	ax3.set_title('total minus sub-sample')
	tg = ax3.imshow(diffi_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-3, vmax = 4e-3,)
	cb = plt.colorbar(tg, ax = ax3, fraction = 0.036, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits((0,0))
	ax3.set_xlim(-2, diffi_block.shape[1] + 1)
	ax3.set_ylim(-2, diffi_block.shape[0] + 1)

	ax2.set_title('SB comparison')
	ax2.plot(c_r_arr, c_sb_arr, ls = '-', color = 'r', alpha = 0.8, label = '$ N_{cut} = 0$')
	ax2.plot(tot_r_arr, tot_sb_arr, ls = '-', color = 'b', alpha = 0.8, label = 'total',)
	ax2.fill_between(tot_r_arr, y1 = tot_sb_arr - tot_sb_err, y2 = tot_sb_arr + tot_sb_err, color = 'b', alpha = 0.2,)

	ax2.set_ylim(-2.5e-3, 1e-2)
	ax2.set_xlim(5e1, 1e3)
	ax2.set_xlabel('$ R[arcsec] $')
	ax2.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax2.set_xscale('log')
	ax2.legend(loc = 3, frameon = False, fontsize = 8)
	ax2.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax2.tick_params(axis = 'both', which = 'both', direction = 'in',)
	ax2.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)

	plt.savefig('total_to_sub-%d_SB_compare.png' % pp, dpi = 300)
	plt.close()

	for ll in range(5):

		edg_xl, edg_xr = grd_x[ll + 1], grd_x[-(ll + 2)]
		edg_yb, edg_yu = grd_y[ll + 1], grd_y[-(ll + 2)]

		dpt_img[:edg_yb, :] = np.nan
		dpt_img[edg_yu:, :] = np.nan
		dpt_img[:, :edg_xl] = np.nan
		dpt_img[:, edg_xr:] = np.nan

		dpt_cont[:edg_yb, :] = np.nan
		dpt_cont[edg_yu:, :] = np.nan
		dpt_cont[:, :edg_xl] = np.nan
		dpt_cont[:, edg_xr:] = np.nan

		cc_stack_img[y_low: y_up+1, x_low: x_up + 1] = dpt_img.copy()
		cc_pix_cont[y_low: y_up+1, x_low: x_up + 1] = dpt_cont.copy()

		img_block, grd_pix = cc_grid_img(dpt_img, 100, 100)[:2]

		Intns, Intns_r, Intns_err, npix, nratio = light_measure_Z0_weit(cc_stack_img, cc_pix_cont, pixel, xn, yn, r_bins_1)
		sb_arr, sb_err_arr = Intns / pixel**2, Intns_err / pixel**2
		r_arr = Intns_r * 1

		with h5py.File('test/clust_BGC-stack_combin_sub-%d_cut-%d_SB.h5' % (pp, N_cut[ll]), 'w') as f:
			f['r'] = np.array(r_arr)
			f['sb'] = np.array(sb_arr)
			f['sb_err'] = np.array(sb_err_arr)
			f['nratio'] = np.array(nratio)
			f['npix'] = np.array(npix)

		idvx = npix < 1.
		c_sb_arr = sb_arr[idvx == False]
		c_r_arr = r_arr[idvx == False]
		c_sb_err = sb_err_arr[idvx == False]

		## compare to total sample
		with h5py.File('test/clust_Mean_jack_img_cut-%d.h5' % N_cut[ll], 'r') as f:
			cut_tot_img = np.array(f['a'])

		with h5py.File('clust_jack_SB-pro_30-FWHM-ov2_cut-Mean-img_%d.h5' % (N_cut[ll]), 'r') as f:
			cut_r_arr = np.array(f['r'])
			cut_sb_arr = np.array(f['sb'])
			cut_sb_err = np.array(f['sb_err'])

		dpt_cut_tot = cut_tot_img[y_botm: y_top+1, x_lft: x_rit + 1]
		cut_block = cc_grid_img(dpt_cut_tot, 100, 100,)[0]

		diffi = cut_tot_img - cc_stack_img
		dpt_diffi = diffi[y_low: y_up+1, x_low: x_up + 1]
		diffi_block = cc_grid_img(dpt_diffi, 100, 100,)[0]

		fig = plt.figure( figsize = (13.12, 9.84) )
		ax0 = fig.add_axes([0.05, 0.55, 0.40, 0.47])
		ax1 = fig.add_axes([0.55, 0.55, 0.40, 0.47])
		ax2 = fig.add_axes([0.05, 0.09, 0.40, 0.39])
		ax3 = fig.add_axes([0.55, 0.05, 0.40, 0.47])

		ax0.set_title('total')
		tg = ax0.imshow(cut_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
		cb = plt.colorbar(tg, ax = ax0, fraction = 0.036, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
		cb.formatter.set_powerlimits((0,0))
		ax0.set_xlim(-2, cut_block.shape[1] + 1)
		ax0.set_ylim(-2, cut_block.shape[0] + 1)

		ax1.set_title('sub sample [%d]' % pp)
		tg = ax1.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
		cb = plt.colorbar(tg, ax = ax1, fraction = 0.036, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
		cb.formatter.set_powerlimits((0,0))
		ax1.set_xlim(-2, img_block.shape[1] + 1)
		ax1.set_ylim(-2, img_block.shape[0] + 1)

		ax3.set_title('total minus sub-sample')
		tg = ax3.imshow(diffi_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-3, vmax = 4e-3,)
		cb = plt.colorbar(tg, ax = ax3, fraction = 0.036, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
		cb.formatter.set_powerlimits((0,0))
		ax3.set_xlim(-2, diffi_block.shape[1] + 1)
		ax3.set_ylim(-2, diffi_block.shape[0] + 1)

		ax2.set_title('SB comparison')
		ax2.plot(c_r_arr, c_sb_arr, ls = '-', color = 'r', alpha = 0.8, label = '$ N_{cut} = 0$')
		ax2.plot(cut_r_arr, cut_sb_arr, ls = '-', color = 'b', alpha = 0.8, label = 'total',)
		ax2.fill_between(cut_r_arr, y1 = cut_sb_arr - cut_sb_err, y2 = cut_sb_arr + cut_sb_err, color = 'b', alpha = 0.2,)

		ax2.set_ylim(-2.5e-3, 1e-2)
		ax2.set_xlim(5e1, 1e3)
		ax2.set_xlabel('$ R[arcsec] $')
		ax2.set_ylabel('SB [nanomaggies / $arcsec^2$]')
		ax2.set_xscale('log')
		ax2.legend(loc = 3, frameon = False, fontsize = 8)
		ax2.grid(which = 'both', axis = 'both', alpha = 0.25)
		ax2.tick_params(axis = 'both', which = 'both', direction = 'in',)
		ax2.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)

		plt.savefig('total_to_sub-%d_SB_compare_cut-%d.png' % (pp, N_cut[ll]), dpi = 300)
		plt.close()


		fig = plt.figure( figsize = (13.12, 9.84) )
		ax0 = fig.add_axes([0.05, 0.55, 0.40, 0.47])
		ax1 = fig.add_axes([0.55, 0.55, 0.40, 0.47])
		ax2 = fig.add_axes([0.05, 0.09, 0.40, 0.39])
		ax3 = fig.add_axes([0.55, 0.05, 0.40, 0.47])

		tf = ax0.imshow(grd_pix, origin = 'lower', cmap = 'rainbow', vmin = 1, vmax = grd_pix.max(),)
		cb = plt.colorbar(tf, ax = ax0, fraction = 0.036, pad = 0.01, label = 'pixel count',)
		tf.cmap.set_under('white')
		cb.formatter.set_powerlimits((0,0))
		ax0.set_xlim(-2, img_block.shape[1] + 1)
		ax0.set_ylim(-2, img_block.shape[0] + 1)

		tg = ax1.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
		cb = plt.colorbar(tg, ax = ax1, fraction = 0.036, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
		cb.formatter.set_powerlimits((0,0))
		ax1.set_xlim(-2, img_block.shape[1] + 1)
		ax1.set_ylim(-2, img_block.shape[0] + 1)

		ax2.plot(c_r_arr, c_sb_arr, ls = '-', color = 'r', alpha = 0.8, label = '$ N_{cut} = %d$' % N_cut[ll])
		#ax2.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'r', alpha = 0.2,)
		for mm in range( 5 ):
			devi_R = np.abs(medi_R - c_r_arr[-(mm + 1)])
			idvx = devi_R == devi_R.min()
			bins_id = np.where(idvx == True)[0][0]
			ax2.axvline(x = medi_R[ bins_id], ls = '--', color = mpl.cm.rainbow(mm / 5), alpha = 0.5,)

		ax2.set_ylim(-2.5e-3, 1e-2)
		ax2.set_xlim(5e1, 1e3)
		ax2.set_xlabel('$ R[arcsec] $')
		ax2.set_ylabel('SB [nanomaggies / $arcsec^2$]')
		ax2.set_xscale('log')
		ax2.legend(loc = 3, frameon = False, fontsize = 8)
		ax2.grid(which = 'both', axis = 'both', alpha = 0.25)
		ax2.tick_params(axis = 'both', which = 'both', direction = 'in',)
		ax2.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)

		tg = ax3.imshow(dpt_img / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -2e0, vmax = 2e0,)
		cb = plt.colorbar(tg, ax = ax3, fraction = 0.036, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
		cb.formatter.set_powerlimits((0,0))

		for mm in range( 5 ):
			devi_R = np.abs(medi_R - c_r_arr[-(mm + 1)])
			idvx = devi_R == devi_R.min() 
			bins_id = np.where(idvx == True)[0][0]

			pix_low = Angl_r[ bins_id ] / pixel
			pix_hig = Angl_r[ bins_id + 1] / pixel

			clust = Circle(xy = (cen_x, cen_y), radius = pix_hig, fill = False, ec = 'b', ls = '-', linewidth = 0.75, alpha = 0.5,)
			ax3.add_patch(clust)
			if mm == 4:
				clust = Circle(xy = (cen_x, cen_y), radius = pix_low, fill = False, ec = 'b', ls = '-', linewidth = 0.75, alpha = 0.5,)
				ax3.add_patch(clust)
			clust = Circle(xy = (cen_x, cen_y), radius = medi_R[ bins_id ] / pixel, fill = False, ec = mpl.cm.rainbow(mm / 5), 
				ls = '--', linewidth = 1, alpha = 0.5,)
			ax3.add_patch(clust)

		for kk in range(Nly):
			for mm in range(Nlx):
					a0, a1 = grd_x[mm], grd_x[mm + 1]
					b0, b1 = grd_y[kk], grd_y[kk + 1]
					idnn = np.isnan( img_block[kk,mm] )
					if idnn == False:
						region = Rectangle(xy = (a0, b0), width = a1 - a0, height = b1 - b0, fill = False, ec = 'g', 
							linewidth = 0.5, alpha = 0.5,)
						ax3.add_patch(region)

		ax3.set_xlim(-200, dpt_img.shape[1] + 200)
		ax3.set_ylim(-200, dpt_img.shape[0] + 200)

		plt.savefig('grid_2D_SB_sub-%d_cut-%d.png' % (pp, N_cut[ll]), dpi = 300)
		plt.close()

raise
"""
## cut stacking img test
#		jackknife sub-samples
J_sub_img = load + '20_10_test/jack_test/clust_BCG-stack_jack-sub-%d_img_30-FWHM-ov2.h5'
J_sub_pix_cont = load + '20_10_test/jack_test/clust_BCG-stack_jack-sub-%d_pix-cont_30-FWHM-ov2.h5'

n_rbins = 110 # 60 # 
N_bin = 30
lim_r1 = 0
N_cut = np.array([100, 200, 300, 400, 500])

for nn in range( 30 ):

	with h5py.File(J_sub_img % nn, 'r') as f:
		sub_jk_img = np.array(f['a'])
	xn, yn = np.int(sub_jk_img.shape[1] / 2), np.int(sub_jk_img.shape[0] / 2)
	id_nn = np.isnan(sub_jk_img)
	eff_y, eff_x = np.where(id_nn == False)
	dR = np.sqrt((eff_y - yn)**2 + (eff_x - xn)**2)
	dR_max = np.int( dR.max() ) + 1
	lim_r1 = np.max([lim_r1, dR_max])

r_bins_1 = np.logspace(0, np.log10(lim_r1), n_rbins)

for nn in range( 30 ):

	with h5py.File(J_sub_img % nn, 'r') as f:
		tmp_img = np.array(f['a'])
	with h5py.File(J_sub_pix_cont % nn, 'r') as f:
		tmp_cont = np.array(f['a'])

	xn, yn = np.int(tmp_img.shape[1] / 2), np.int(tmp_img.shape[0] / 2)

	id_nan = np.isnan(tmp_img)
	idvx = id_nan == False
	idy, idx = np.where(idvx == True)
	x_low, x_up = np.min(idx), np.max(idx)
	y_low, y_up = np.min(idy), np.max(idy)

	dpt_img = tmp_img[y_low: y_up+1, x_low: x_up + 1]
	test_grd = cc_grid_img(dpt_img, 100, 100,)
	grd_x, grd_y = test_grd[-2], test_grd[-1]

	dpt_cont = tmp_cont[y_low: y_up+1, x_low: x_up + 1]

	for ll in range( 5 ):

		edg_xl, edg_xr = grd_x[ll + 1], grd_x[-(ll + 2)]
		edg_yb, edg_yu = grd_y[ll + 1], grd_y[-(ll + 2)]

		dpt_img[:edg_yb, :] = np.nan
		dpt_img[edg_yu:, :] = np.nan
		dpt_img[:, :edg_xl] = np.nan
		dpt_img[:, edg_xr:] = np.nan

		dpt_cont[:edg_yb, :] = np.nan
		dpt_cont[edg_yu:, :] = np.nan
		dpt_cont[:, :edg_xl] = np.nan
		dpt_cont[:, edg_xr:] = np.nan

		tmp_img[y_low: y_up+1, x_low: x_up + 1] = dpt_img.copy()
		tmp_cont[y_low: y_up+1, x_low: x_up + 1] = dpt_cont.copy()

		Intns, Intns_r, Intns_err, npix, nratio = light_measure_Z0_weit(tmp_img, tmp_cont, pixel, xn, yn, r_bins_1)
		sb_arr, sb_err_arr = Intns / pixel**2, Intns_err / pixel**2
		r_arr = Intns_r

		with h5py.File('test/clust_jack-sub-%d_SB-pro_30-FWHM-ov2_cut-Mean-img_%d.h5' % (nn, N_cut[ll]), 'w') as f:
			f['r'] = np.array(r_arr)
			f['sb'] = np.array(sb_arr)
			f['sb_err'] = np.array(sb_err_arr)
			f['nratio'] = np.array(nratio)
			f['npix'] = np.array(npix)

for ll in range( 5 ):
	tmp_sb = []
	tmp_r = []
	for nn in range( 30 ):
		with h5py.File('test/clust_jack-sub-%d_SB-pro_30-FWHM-ov2_cut-Mean-img_%d.h5' % (nn, N_cut[ll]), 'r') as f:
			r_arr = np.array(f['r'])[:-1]
			sb_arr = np.array(f['sb'])[:-1]
			sb_err = np.array(f['sb_err'])[:-1]
			npix = np.array(f['npix'])[:-1]
			nratio = np.array(f['nratio'])[:-1]

			idvx = npix < 1.
			sb_arr[idvx] = np.nan
			r_arr[idvx] = np.nan

			tmp_sb.append(sb_arr)
			tmp_r.append(r_arr)

	## only save the sb result in unit " nanomaggies / arcsec^2 "
	tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(tmp_sb, tmp_r, 0, 30,)[4:]
	sb_lim_r = np.ones( len(tt_jk_R) ) * lim_R

	with h5py.File('clust_jack_SB-pro_30-FWHM-ov2_cut-Mean-img_%d.h5' % (N_cut[ll]), 'w') as f:
	#with h5py.File('clust_jack_SB-pro_30-FWHM-ov2_cut-Mean-img_%d_wide-rbin.h5' % (N_cut[ll]), 'w') as f:

	#with h5py.File('clust_jack_SB-pro_30-FWHM-ov2_H-cut-Mean-img_%d.h5' % (N_cut[ll]), 'w') as f:
	#with h5py.File('clust_jack_SB-pro_30-FWHM-ov2_V-cut-Mean-img_%d.h5' % (N_cut[ll]), 'w') as f:
		f['r'] = np.array(tt_jk_R)
		f['sb'] = np.array(tt_jk_SB)
		f['sb_err'] = np.array(tt_jk_err)
		f['lim_r'] = np.array(sb_lim_r)

raise


## jack_SB compare (cut stacking imgs)
N_cut = np.array([100, 200, 300, 400, 500])
pro_lis = ['$ N_{cut} $ = 0', '$ N_{cut} $ = 100', '$ N_{cut} $ = 200', 
			'$ N_{cut} $ = 300', '$ N_{cut} $ = 400', '$ N_{cut} $ = 500',]

with h5py.File(load + '20_10_test/jack_test/clust_BCG-stack_Mean_jack_img_30-FWHM-ov2.h5', 'r') as f:
	tt_img = np.array(f['a'])

id_nan = np.isnan(tt_img)
idvx = id_nan == False
idy, idx = np.where(idvx == True)
x_low, x_up = np.min(idx), np.max(idx)
y_low, y_up = np.min(idy), np.max(idy)

xn, yn = np.int(tt_img.shape[1] / 2), np.int(tt_img.shape[0] / 2)
dpt_img = tt_img[y_low: y_up+1, x_low: x_up + 1]
cen_x, cen_y = xn - x_low, yn - y_low

#img_block, grd_pix = grid_img(dpt_img, 100, 100)[:2]
test_grd = cc_grid_img(dpt_img, 100, 100,)
grd_x, grd_y = test_grd[-2], test_grd[-1]
img_block = test_grd[0]
Nlx, Nly = img_block.shape[1], img_block.shape[0]

lim_r1 = 2460
r_bins_1 = np.logspace(0, np.log10(lim_r1), 110)
Angl_r = r_bins_1 * pixel
medi_R = 0.5 * (Angl_r[1:] + Angl_r[:-1])

plt.figure()
ax = plt.subplot(111)

for ll in range( 6 ):
	if ll == 0:

		img_block, grd_pix = cc_grid_img(dpt_img, 100, 100)[:2]

		with h5py.File(load + '20_10_test/jack_test/clust_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2.h5', 'r') as f:
		#with h5py.File('clust_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2_wide-rbin.h5', 'r') as f:
			c_r_arr = np.array(f['r'])
			c_sb_arr = np.array(f['sb'])
			c_sb_err = np.array(f['sb_err'])

	else:
		edg_xl, edg_xr = grd_x[ll], grd_x[-(ll + 1)]
		edg_yb, edg_yu = grd_y[ll], grd_y[-(ll + 1)]

		dpt_img[:edg_yb, :] = np.nan
		dpt_img[edg_yu:, :] = np.nan
		dpt_img[:, :edg_xl] = np.nan
		dpt_img[:, edg_xr:] = np.nan

		img_block, grd_pix = cc_grid_img(dpt_img, 100, 100)[:2]

		copy_img = tt_img.copy()
		copy_img[y_low: y_up+1, x_low: x_up + 1] = dpt_img * 1.
		with h5py.File('test/clust_Mean_jack_img_cut-%d.h5' % N_cut[ll-1], 'w') as f:
			f['a'] = np.array(copy_img)

		with h5py.File('clust_jack_SB-pro_30-FWHM-ov2_cut-Mean-img_%d.h5' % (N_cut[ll - 1]), 'r') as f:
		#with h5py.File('clust_jack_SB-pro_30-FWHM-ov2_cut-Mean-img_%d_wide-rbin.h5' % (N_cut[ll - 1]), 'r') as f:

		#with h5py.File('clust_jack_SB-pro_30-FWHM-ov2_H-cut-Mean-img_%d.h5' % (N_cut[ll - 1]), 'r') as f:
		#with h5py.File('clust_jack_SB-pro_30-FWHM-ov2_V-cut-Mean-img_%d.h5' % (N_cut[ll - 1]), 'r') as f:
			c_r_arr = np.array(f['r'])
			c_sb_arr = np.array(f['sb'])
			c_sb_err = np.array(f['sb_err'])

	fig = plt.figure( figsize = (13.12, 9.84) )
	ax0 = fig.add_axes([0.05, 0.55, 0.40, 0.47])
	ax1 = fig.add_axes([0.55, 0.55, 0.40, 0.47])
	ax2 = fig.add_axes([0.05, 0.09, 0.40, 0.39])
	ax3 = fig.add_axes([0.55, 0.05, 0.40, 0.47])

	tf = ax0.imshow(grd_pix, origin = 'lower', cmap = 'rainbow', vmin = 1, vmax = grd_pix.max(),)
	cb = plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = 'pixel count',)
	tf.cmap.set_under('white')
	cb.formatter.set_powerlimits((0,0))

	idux = (grd_pix <= 500) & (grd_pix > 0)
	poy, pox = np.where(idux == True)
	for qq in range( np.sum(idux)):
		ax0.text(pox[qq], poy[qq], s = '%d' % (grd_pix[ poy[qq],pox[qq] ]), fontsize = 4, color = 'w', ha = 'center',)

	tg = ax1.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = ax1, fraction = 0.035, pad = 0.01,)
	cb.formatter.set_powerlimits((0,0))

	ax2.plot(c_r_arr, c_sb_arr, ls = '-', color = 'r', alpha = 0.8, label = pro_lis[ll],)
	ax2.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'r', alpha = 0.2,)

	for pp in range( 5 ):
		devi_R = np.abs(medi_R - c_r_arr[-(pp+1)])
		idvx = devi_R == devi_R.min() 
		bins_id = np.where(idvx == True)[0][0]
		ax2.axvline(x = medi_R[ bins_id], ls = '--', color = mpl.cm.rainbow(pp / 5), alpha = 0.5,)

	ax2.set_ylim(4e-4, 7e-3)
	ax2.set_xlim(5e1, 1e3)
	ax2.set_xlabel('$ R[arcsec] $')
	ax2.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax2.set_xscale('log')
	ax2.legend(loc = 3, frameon = False, fontsize = 8)
	ax2.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax2.tick_params(axis = 'both', which = 'both', direction = 'in',)
	ax2.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)

	tg = ax3.imshow(dpt_img / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -5e-1, vmax = 5e-1,)
	cb = plt.colorbar(tg, ax = ax3, fraction = 0.036, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits((0,0))

	for mm in range( 5 ):
		devi_R = np.abs(medi_R - c_r_arr[-(mm + 1)])
		idvx = devi_R == devi_R.min() 
		bins_id = np.where(idvx == True)[0][0]

		pix_low = Angl_r[ bins_id ] / pixel
		pix_hig = Angl_r[ bins_id + 1] / pixel

		clust = Circle(xy = (cen_x, cen_y), radius = pix_hig, fill = False, ec = 'b', ls = '-', linewidth = 0.75, alpha = 0.5,)
		ax3.add_patch(clust)
		if mm == 4:
			clust = Circle(xy = (cen_x, cen_y), radius = pix_low, fill = False, ec = 'b', ls = '-', linewidth = 0.75, alpha = 0.5,)
			ax3.add_patch(clust)
		clust = Circle(xy = (cen_x, cen_y), radius = medi_R[ bins_id ] / pixel, fill = False, ec = mpl.cm.rainbow(mm / 5), 
			ls = '--', linewidth = 1, alpha = 0.5,)
		ax3.add_patch(clust)

	for kk in range(Nly):
		for mm in range(Nlx):
				a0, a1 = grd_x[mm], grd_x[mm + 1]
				b0, b1 = grd_y[kk], grd_y[kk + 1]
				idnn = np.isnan( img_block[kk,mm] )
				if idnn == False:
					region = Rectangle(xy = (a0, b0), width = a1 - a0, height = b1 - b0, fill = False, ec = 'g', 
						linewidth = 0.5, alpha = 0.5,)
					ax3.add_patch(region)

	ax3.set_xlim(-200, dpt_img.shape[1] + 200)
	ax3.set_ylim(-200, dpt_img.shape[0] + 200)

	plt.savefig('grid_2D_SB_cut_test_%d.png' % ll, dpi = 300)
	plt.close()

	ax.plot(c_r_arr, c_sb_arr, ls = '-', alpha = 0.8, color = mpl.cm.rainbow(ll / 6), label = pro_lis[ll],)
	ax.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = mpl.cm.rainbow(ll / 6), alpha = 0.2,)

ax.set_ylim(4e-4, 7e-3)
ax.set_yscale('log')
ax.set_xlim(5e1, 1e3)
ax.set_xlabel('$ R[arcsec] $')
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 3, frameon = False,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in')

plt.subplots_adjust(left = 0.15, right = 0.95,)
plt.savefig('SB_cut-size_test.png', dpi = 300)
plt.close()

raise

## cut single img test
N_cut = np.array([200, 500])
pro_lis = ['$ N_{cut} $ = 0', '$ N_{cut} $ = 200', '$ N_{cut} $ = 500',]

plt.figure()
ax = plt.subplot(111)

for ll in range(3):
	if ll == 0:
		with h5py.File(load + '20_10_test/jack_test/clust_BCG-stack_Mean_jack_img_30-FWHM-ov2.h5', 'r') as f:
			tt_img = np.array(f['a'])
		img_block, grd_pix = grid_img(tt_img, 100, 100)[:2]

		with h5py.File(load + '20_10_test/jack_test/clust_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2.h5', 'r') as f:
			c_r_arr = np.array(f['r'])
			c_sb_arr = np.array(f['sb'])
			c_sb_err = np.array(f['sb_err'])
	else:
		with h5py.File(load + '20_10_test/jack_test/A_clust_BCG-stack_Mean_jack_img_30-FWHM-ov2_cut-%d.h5' % N_cut[ll-1], 'r') as f:
			tt_img = np.array(f['a'])
		img_block, grd_pix = grid_img(tt_img, 100, 100)[:2]

		with h5py.File(load + '20_10_test/jack_test/A_clust_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2_cut-%d.h5' % N_cut[ll-1], 'r') as f:
			c_r_arr = np.array(f['r'])
			c_sb_arr = np.array(f['sb'])
			c_sb_err = np.array(f['sb_err'])

	fig = plt.figure( figsize = (19.84, 4.8) )
	a_ax0 = fig.add_axes([0.03, 0.09, 0.30, 0.85])
	a_ax1 = fig.add_axes([0.38, 0.09, 0.30, 0.85])
	a_ax2 = fig.add_axes([0.73, 0.09, 0.25, 0.85])

	tf = a_ax0.imshow(grd_pix, origin = 'lower', cmap = 'rainbow', vmin = 1, vmax = grd_pix.max(),)
	cb = plt.colorbar(tf, ax = a_ax0, fraction = 0.035, pad = 0.01, label = 'pixel count',)
	tf.cmap.set_under('white')
	cb.formatter.set_powerlimits((0,0))

	idux = (grd_pix <= 500) & (grd_pix > 0)
	poy, pox = np.where(idux == True)
	for qq in range( np.sum(idux)):

		a_ax0.text(pox[qq], poy[qq], s = '%d' % (grd_pix[ poy[qq],pox[qq] ]), fontsize = 4, color = 'w', ha = 'center',)

	tg = a_ax1.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = a_ax1, fraction = 0.035, pad = 0.01,)
	cb.formatter.set_powerlimits((0,0))

	a_ax2.plot(c_r_arr, c_sb_arr, ls = '-', color = 'r', alpha = 0.8, label = pro_lis[ll],)
	a_ax2.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'r', alpha = 0.2,)

	a_ax2.set_ylim(4e-4, 7e-3)
	a_ax2.set_xlim(5e1, 1e3)
	a_ax2.set_xlabel('$ R[arcsec] $')
	a_ax2.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	a_ax2.set_xscale('log')
	a_ax2.legend(loc = 1, frameon = False, fontsize = 8)
	a_ax2.grid(which = 'both', axis = 'both', alpha = 0.25)
	a_ax2.tick_params(axis = 'both', which = 'both', direction = 'in',)
	a_ax2.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)

	plt.savefig('grid_2D_SB_cut_test_%d.png' % ll, dpi = 300)
	plt.close()

	ax.plot(c_r_arr, c_sb_arr, ls = '-', alpha = 0.8, color = mpl.cm.plasma(ll / 3), label = pro_lis[ll],)
	ax.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = mpl.cm.plasma(ll / 3), alpha = 0.2,)

ax.set_ylim(1e-4, 7e-3)
ax.set_yscale('log')
ax.set_xlim(5e1, 1e3)
ax.set_xlabel('$ R[arcsec] $')
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 3, frameon = False,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in')

plt.subplots_adjust(left = 0.15, right = 0.95,)
plt.savefig('SB_origin-img_cut-size_test.png', dpi = 300)
plt.close()

### compare with sky-img stacking
J_sub_img = load + '20_10_test/jack_test/clust_BCG-stack_jack-sub-%d_img_30-FWHM-ov2.h5'
J_sub_pix_cont = load + '20_10_test/jack_test/clust_BCG-stack_jack-sub-%d_pix-cont_30-FWHM-ov2.h5'
#J_sub_img = load + '20_10_test/jack_test/A_sky_BCG-stack_jack-sub-%d_img_d-medi.h5' # _img.h5'
sub_img = load + '20_10_test/jack_test/A_sky_BCG-stack_sub-%d_img_d-medi.h5'

n_rbins = 110
N_bin = 30
lim_r1 = 0
for nn in range( 30 ):

	with h5py.File(J_sub_img % nn, 'r') as f:
		sub_jk_img = np.array(f['a'])
	xn, yn = np.int(sub_jk_img.shape[1] / 2), np.int(sub_jk_img.shape[0] / 2)
	id_nn = np.isnan(sub_jk_img)
	eff_y, eff_x = np.where(id_nn == False)
	dR = np.sqrt((eff_y - yn)**2 + (eff_x - xn)**2)
	dR_max = np.int( dR.max() ) + 1
	lim_r1 = np.max([lim_r1, dR_max])

r_bins_1 = np.logspace(0, np.log10(lim_r1), n_rbins)

for nn in range( 30 ):

	with h5py.File(load + '20_10_test/jack_test/clust_BCG-stack_jack-sub-%d_img_30-FWHM-ov2.h5' % nn, 'r') as f:
		tmp_img = np.array(f['a'])
	with h5py.File(load + '20_10_test/jack_test/clust_BCG-stack_jack-sub-%d_pix-cont_30-FWHM-ov2.h5' % nn, 'r') as f:
		tmp_cont = np.array(f['a'])

	#with h5py.File(load + '20_10_test/jack_test/A_sky_BCG-stack_jack-sub-%d_img.h5' % nn, 'r') as f:
	with h5py.File(load + '20_10_test/jack_test/A_sky_BCG-stack_jack-sub-%d_img_d-medi.h5' % nn, 'r') as f:
		sky_img = np.array(f['a'])
	comb_img = tmp_img + sky_img

	xn, yn = np.int(tmp_img.shape[1] / 2), np.int(tmp_img.shape[0] / 2)

	Intns, Angl_r, Intns_err, npix, nratio = light_measure_Z0_weit(comb_img, tmp_cont, pixel, xn, yn, r_bins_1)
	sb_arr, sb_err_arr = Intns / pixel**2, Intns_err / pixel**2
	r_arr = Angl_r

	with h5py.File('test/clust_jack-sub-%d_SB-pro_30-FWHM-ov2_add-resi-sky.h5' % nn, 'w') as f:
		f['r'] = np.array(r_arr)
		f['sb'] = np.array(sb_arr)
		f['sb_err'] = np.array(sb_err_arr)
		f['nratio'] = np.array(nratio)
		f['npix'] = np.array(npix)

tmp_sb = []
tmp_r = []
for nn in range( 30 ):
	with h5py.File('test/clust_jack-sub-%d_SB-pro_30-FWHM-ov2_add-resi-sky.h5' % nn, 'r') as f:
		r_arr = np.array(f['r'])[:-1]
		sb_arr = np.array(f['sb'])[:-1]
		sb_err = np.array(f['sb_err'])[:-1]
		npix = np.array(f['npix'])[:-1]
		nratio = np.array(f['nratio'])[:-1]

		idvx = npix < 1.
		sb_arr[idvx] = np.nan
		r_arr[idvx] = np.nan

		tmp_sb.append(sb_arr)
		tmp_r.append(r_arr)

## only save the sb result in unit " nanomaggies / arcsec^2 "
tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(tmp_sb, tmp_r, 0, 30)[4:]
sb_lim_r = np.ones( len(tt_jk_R) ) * lim_R

with h5py.File('clust_jack_SB-pro_30-FWHM-ov2_add-resi-sky.h5', 'w') as f:
	f['r'] = np.array(tt_jk_R)
	f['sb'] = np.array(tt_jk_SB)
	f['sb_err'] = np.array(tt_jk_err)
	f['lim_r'] = np.array(sb_lim_r)

with h5py.File(load + '20_10_test/jack_test/clust_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2.h5', 'r') as f:
	img_sb_r = np.array(f['r'])
	img_sb = np.array(f['sb'])
	img_sb_err = np.array(f['sb_err'])

with h5py.File('clust_jack_SB-pro_30-FWHM-ov2_add-resi-sky.h5', 'r') as f:
	com_sb_r = np.array(f['r'])
	com_sb = np.array(f['sb'])
	com_sb_err = np.array(f['sb_err'])

plt.figure()
ax = plt.subplot(111)

ax.plot(img_sb_r, img_sb, color = 'r', ls = '-', alpha = 0.8, label = 'stacking cluster img',)
ax.fill_between(img_sb_r, y1 = img_sb - img_sb_err, y2 = img_sb + img_sb_err, color = 'r', alpha = 0.2,)
ax.plot(com_sb_r, com_sb, color = 'b', ls = '-', alpha = 0.8, label = 'stacking [cluster + median subtracted sky]',)
ax.fill_between(com_sb_r, y1 = com_sb - com_sb_err, y2 = com_sb + com_sb_err, color = 'b', alpha = 0.2,)

ax.set_xlim(5e1, 1e3)
ax.set_xscale('log')
ax.set_ylim(7e-4, 7e-3)
ax.set_yscale('log')
ax.set_ylabel('SB err [nanomaggies / arcsec^2]')
ax.legend(loc = 3, frameon = False,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in')

plt.subplots_adjust(left = 0.15, right = 0.95,)
plt.savefig('add_resi-sky_SB_compare.png', dpi = 300)
plt.show()

raise

with h5py.File(load + '20_10_test/jack_test/A_sky_BCG-stack_Mean_jack_img_d-medi.h5', 'r') as f:
#with h5py.File(load + '20_10_test/jack_test/A_sky_BCG-stack_Mean_jack_img.h5', 'r') as f:
	sky_img = np.array(f['a'])

with h5py.File(load + '20_10_test/jack_test/A_sky_BCG-stack_Mean_jack_SB-pro_d-medi.h5', 'r') as f:
#with h5py.File(load + '20_10_test/jack_test/A_sky_BCG-stack_Mean_jack_SB-pro.h5', 'r') as f:
	sky_pro_r = np.array(f['r'])
	sky_pro_sb = np.array(f['sb'])
	sky_pro_err = np.array(f['sb_err'])

fig = plt.figure( figsize = (13.12, 4.8) )
a_ax0 = fig.add_axes([0.05, 0.09, 0.40, 0.80])
a_ax1 = fig.add_axes([0.55, 0.09, 0.40, 0.80])

a_ax0.set_title('median subtracted case')
tg = a_ax0.imshow(sky_img / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -2.5e-2, vmax = 2.5e-2,)
#tg = a_ax0.imshow(sky_img / pixel**2, origin = 'lower',)
cb = plt.colorbar(tg, ax = a_ax0, fraction = 0.035, pad = 0.01,)
cb.formatter.set_powerlimits((0,0))

cen_x, cen_y = np.int(sky_img.shape[1] / 2), np.int(sky_img.shape[0] / 2)
clust = Circle(xy = (cen_x, cen_y), radius = 700 / pixel, fill = False, ec = 'k', ls = '--', linewidth = 1, alpha = 0.5,)
a_ax0.add_patch(clust)
clust = Circle(xy = (cen_x, cen_y), radius = 800 / pixel, fill = False, ec = 'k', ls = '-.', linewidth = 1, alpha = 0.5,)
a_ax0.add_patch(clust)
clust = Circle(xy = (cen_x, cen_y), radius = 900 / pixel, fill = False, ec = 'k', ls = '-', linewidth = 1, alpha = 0.5,)
a_ax0.add_patch(clust)

a_ax1.plot(sky_pro_r, sky_pro_sb, color = 'r', ls = '-', alpha = 0.8, )
a_ax1.fill_between(sky_pro_r, y1 = sky_pro_sb - sky_pro_err, y2 = sky_pro_sb + sky_pro_err, alpha = 0.2, color = 'r',)
a_ax1.axvline(x = 700, ls = '--', color = 'k', alpha = 0.5,)
a_ax1.axvline(x = 800, ls = '-.', color = 'k', alpha = 0.5,)
a_ax1.axvline(x = 900, ls = '-', color = 'k', alpha = 0.5,)

a_ax1.set_ylim(-3e-3, 3e-3)
a_ax1.set_xlim(5e1, 1e3)
a_ax1.set_xlabel('$ R[arcsec] $')
a_ax1.set_ylabel('SB [nanomaggies / $arcsec^2$]')
a_ax1.set_xscale('log')
a_ax1.legend(loc = 1, frameon = False, fontsize = 8)
a_ax1.grid(which = 'both', axis = 'both', alpha = 0.25)
a_ax1.tick_params(axis = 'both', which = 'both', direction = 'in',)
a_ax1.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)

plt.savefig('2D_sky_img_SB.png', dpi = 300)
plt.close()

