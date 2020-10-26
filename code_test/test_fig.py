import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle

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
from light_measure import light_measure_Z0_weit

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
mag_add = np.array([0, 0, 0, -0.04, 0.02])

########
home = '/home/xkchen/data/SDSS/'

sigma = np.array([3.5, 4, 4.5, 5, 5.5, 6,])
#thres_S = np.array([2, 3, 4])
#sigma = 3.5

### fig the result
def slop2r(r_data, sb_data, r_pix, id_log = True):
	"""
	r_data : radius bins
	sb_data : surface brightness, in linear unit -- nanomaggy/arcsec^2
	r_pix : pixel number in radius bins
	"""
	tmpr = 0.5 * (r_data[1:] + r_data[:-1])
	dmpr = r_data[1:] - r_data[:-1]
	tmpk = np.zeros( len(tmpr), dtype = np.float32)
	rpix = r_pix.astype(np.int)

	for kk in range( len(tmpr) ):
		nk_0 = rpix[kk]
		nk_1 = rpix[kk + 1]

		if (nk_0 == 0) | (nk_1 == 0):
			tmpk[kk] = np.nan
		else:
			if id_log == True:
				tmpk[kk] = ( (sb_data[kk + 1] - sb_data[kk]) / dmpr[kk] ) * ( tmpr[kk] * np.log(10) )
			else:
				tmpk[kk] = ( (sb_data[kk + 1] - sb_data[kk]) / dmpr[kk] )

	return tmpr, tmpk

def jack_slop2r(sub_r, sub_slop, N_sample,):

	dx_r = np.array(sub_r)
	dy_k = np.array(sub_slop)
	n_r = dx_r.shape[1]

	Len = np.zeros( n_r, dtype = np.float32)
	for nn in range( n_r ):
		tmp_k = dy_k[:,nn]
		idnn = np.isnan(tmp_k)
		Len[nn] = N_sample - np.sum(idnn)

	stack_r = np.nanmean(dx_r, axis = 0)
	stack_k = np.nanmean(dy_k, axis = 0)
	std_stack_k = np.nanstd(dy_k, axis = 0)

	# limit on sub-sample contribution
	id_one = Len > 1
	stack_r = stack_r[id_one]
	stack_k = stack_k[id_one]
	std_stack_k = std_stack_k[id_one]
	N_img = Len[id_one]
	jk_k_err = np.sqrt(N_img - 1) * std_stack_k

	id_min = N_img >= np.int(N_sample / 3)
	lim_r = stack_r[id_min]
	lim_R = np.nanmax(lim_r)

	return stack_r, stack_k, jk_k_err, lim_R

def jack_SB(SB_array, R_array, band_id, N_sample,):
	"""
	stacking profile based on surface brightness,
	SB_array : list of surface brightness profile, in unit of " nanomaggies / arcsec^2 "
	"""
	dx_r = np.array(R_array)
	dy_sb = np.array(SB_array)

	n_r = dx_r.shape[1]
	Len = np.zeros( n_r, dtype = np.float32)
	for nn in range( n_r ):
		tmp_I = dy_sb[:,nn]
		idnn = np.isnan(tmp_I)
		Len[nn] = N_sample - np.sum(idnn)

	Stack_R = np.nanmean(dx_r, axis = 0)
	Stack_SB = np.nanmean(dy_sb, axis = 0)
	std_Stack_SB = np.nanstd(dy_sb, axis = 0)

	### limit the radius bin contribution at least 1/3 * N_sample
	id_one = Len > 1
	Stack_R = Stack_R[ id_one ]
	Stack_SB = Stack_SB[ id_one ]
	std_Stack_SB = std_Stack_SB[ id_one ]
	N_img = Len[ id_one ]
	jk_Stack_err = np.sqrt(N_img - 1) * std_Stack_SB

	id_min = N_img >= np.int(N_sample / 3)
	lim_r = Stack_R[id_min]
	lim_R = np.nanmax(lim_r)

	## change flux to magnitude
	jk_Stack_SB = 22.5 - 2.5 * np.log10(Stack_SB) + mag_add[band_id]
	dSB0 = 22.5 - 2.5 * np.log10(Stack_SB + jk_Stack_err) + mag_add[band_id]
	dSB1 = 22.5 - 2.5 * np.log10(Stack_SB - jk_Stack_err) + mag_add[band_id]
	err0 = jk_Stack_SB - dSB0
	err1 = dSB1 - jk_Stack_SB
	id_nan = np.isnan(jk_Stack_SB)
	jk_Stack_SB, jk_Stack_R = jk_Stack_SB[id_nan == False], Stack_R[id_nan == False]
	jk_Stack_err0, jk_Stack_err1 = err0[id_nan == False], err1[id_nan == False]
	dSB0, dSB1 = dSB0[id_nan == False], dSB1[id_nan == False]
	idx_nan = np.isnan(dSB1)
	jk_Stack_err1[idx_nan] = 100.

	return jk_Stack_SB, jk_Stack_R, jk_Stack_err0, jk_Stack_err1, Stack_R, Stack_SB, jk_Stack_err, lim_R

def grid_img(img_data, N_stepx, N_stepy):

	ly = np.arange(0, img_data.shape[0], N_stepy)
	ly = np.r_[ly, img_data.shape[0] - N_stepy, img_data.shape[0] ]
	lx = np.arange(0, img_data.shape[1], N_stepx)
	lx = np.r_[lx, img_data.shape[1] - N_stepx, img_data.shape[1] ]

	lx = np.delete(lx, -1)
	lx = np.delete(lx, -2)
	ly = np.delete(ly, -1)
	ly = np.delete(ly, -2)

	patch_mean = np.zeros( (len(ly), len(lx) ), dtype = np.float )
	patch_pix = np.zeros( (len(ly), len(lx) ), dtype = np.float )
	patch_Var = np.zeros( (len(ly), len(lx) ), dtype = np.float )
	for nn in range( len(ly) ):
		for tt in range( len(lx) ):

			sub_flux = img_data[ly[nn]: ly[nn] + N_stepy, lx[tt]: lx[tt] + N_stepx]
			id_nn = np.isnan(sub_flux)
			patch_mean[nn,tt] = np.mean( sub_flux[id_nn == False] )
			patch_pix[nn,tt] = len( sub_flux[id_nn == False] )
			patch_Var[nn,tt] = np.std( sub_flux[id_nn == False] )

	id_zeros = patch_pix == 0.
	patch_mean[id_zeros] = np.nan
	patch_Var[id_zeros] = np.nan

	## limit for pixels in sub-patch (much less is unreliable)
	id_min = patch_pix <= 5
	patch_Var[id_min] = np.nan

	return patch_mean, patch_Var
"""
dat_0 = pds.read_csv(home + 'selection/tmp/cluster_tot-r-band_norm-img_cat.csv')
N_tot = dat_0.ra.shape[0]

N_ratio = []
sb_arr = []
r_arr = []
com_eta = []
#for nn in range( len(thres_S) ):
for nn in range( len(sigma) ):
	dat = pds.read_csv(home + 'selection/tmp/CC_tot_clust_remain_cat_%.1f-sigma.csv' % (sigma[nn]),)
	#dat = pds.read_csv(home + 'selection/tmp/tot_clust_remain_cat_%d-patch_3.5-sigm.csv' % (thres_S[nn]),)
	nsize = dat.ra.shape[0]
	eta = nsize / N_tot
	N_ratio.append( eta )

	dat = pds.read_csv(home + 'selection/tmp/tot_clust_remain_cat_%.1f-sigma.csv' % (sigma[nn]),)
	nsize = dat.ra.shape[0]
	com_eta.append( nsize / N_tot )

N_ratio = np.array(N_ratio)

plt.figure()
ax = plt.subplot(111)
ax.plot(sigma, N_ratio, 'bo-', alpha = 0.5, label = 'new test',)
ax.plot(sigma, com_eta, 'ro-', alpha = 0.5, label = 'old selection',)
ax.legend(loc = 4, frameon = False,)
#ax.plot(thres_S, N_ratio, 'ro-', alpha = 0.5,)

ax.set_xlabel('Thresh value of $ \\sigma $')
#ax.set_xlabel('Thresh value of patch number')

ax.set_ylabel('$ N_{sample} / N_{tot} $')
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in')

ax.set_xlim(3.4, 6.1)
ax.set_ylim(0.6, 0.9)

plt.subplots_adjust(left = 0.15, right = 0.95, hspace = 0.05)
plt.savefig('sample_size_ratio.png', dpi = 300)
plt.close()
"""
N_bin = 30

#out_path = home + 'tmp_stack/jack/'
out_path = '/home/xkchen/tmp/jk_test/SBs/'

#for nn in range( len(thres_S) ):
for nn in range( len(sigma) ):

	tmp_sb = []
	tmp_r = []
	up2sbr = []

	tmp_slop = []
	tmp_slop_r = []

	fig, axes = plt.subplots(1, 2, sharex = False, sharey = True, figsize=(12, 6))
	ax0 = axes[0]
	ax1 = axes[1]
	ax0.set_title('sub-sample SB')
	ax1.set_title('jack sub-sample SB')

	for ll in range( N_bin ):
		'''
		#with h5py.File(out_path + 'CC_clust_BCG-stack_%.1f-sigma_sub-%d.h5' % (sigma[nn], ll), 'r') as f:
		with h5py.File(out_path + 'clust_BCG-stack_%.1f-sigma_sub-%d.h5' % (sigma[nn], ll), 'r') as f:
		#with h5py.File(out_path + 'clust_BCG-stack_4-sigma_%d-patch_sub-%d.h5' % (thres_S[nn], ll), 'r') as f:
			tmp_img = np.array(f['a'])
		block_m = grid_img(tmp_img, 100, 100)[0]
		'''
		#with h5py.File(out_path + 'CC_clust_BCG-stack_SB_%.1f-sigma_sub-%d.h5' % (sigma[nn], ll), 'r') as f:
		with h5py.File(out_path + 'clust_BCG-stack_SB_%.1f-sigma_sub-%d.h5' % (sigma[nn], ll), 'r') as f:
		#with h5py.File(out_path + 'clust_BCG-stack_SB_%.1f-sigma_%d-patch_sub-%d.h5' % (sigma, thres_S[nn], ll), 'r') as f:
			r_arr = np.array(f['r'])[:-1]
			sb_arr = np.array(f['sb'])[:-1]
			sb_err = np.array(f['sb_err'])[:-1]
			npix = np.array(f['npix'])[:-1]
			nratio = np.array(f['nratio'])[:-1]

		idvx = nratio >= ( np.max(nratio) * (1 / 20) ) # np.e, 5, 10, 20
		up_r = np.max( r_arr[idvx] )

		ax0.plot(r_arr, sb_arr, ls = '-', color = mpl.cm.rainbow(ll / N_bin), alpha = 0.45,)
		ax0.axvline(x = up_r, ls = '--', color = mpl.cm.rainbow(ll / N_bin), alpha = 0.45,)
		'''
		fig = plt.figure( figsize = (13.12, 4.8) )
		a_ax0 = fig.add_axes([0.05, 0.09, 0.40, 0.80])
		a_ax1 = fig.add_axes([0.55, 0.09, 0.40, 0.80])

		tg = a_ax0.imshow(block_m / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
		cb = plt.colorbar(tg, ax = a_ax0, fraction = 0.035, pad = 0.01,)
		cb.formatter.set_powerlimits((0,0))

		idzo = npix < 1
		a_ax1.errorbar(r_arr[idzo == False], sb_arr[idzo == False], yerr = sb_err[idzo == False], xerr = None, color = 'r', 
			marker = 'None', ls = '-', ecolor = 'r', alpha = 0.5, label = '$3.5 \\sigma$',)
		a_ax1.set_ylim(1e-3, 7e-3)
		a_ax1.set_xlim(5e1, 1e3)
		a_ax1.set_xlabel('$ R[arcsec] $')
		a_ax1.set_ylabel('SB [nanomaggies / $arcsec^2$]')
		a_ax1.set_xscale('log')
		a_ax1.legend(loc = 1, frameon = False, fontsize = 8)
		a_ax1.grid(which = 'both', axis = 'both', alpha = 0.25)
		a_ax1.tick_params(axis = 'both', which = 'both', direction = 'in',)
		a_ax1.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)

		plt.savefig('/home/xkchen/fig_tmp/grid_2D_%.1f-sigma_sub-%d.png' % (sigma[nn], ll), dpi = 300)
		plt.close()
		'''
		#with h5py.File(out_path + 'CC_clust_BCG-stack_SB_%.1f-sigma_jk-sub-%d.h5' % (sigma[nn], ll), 'r') as f:
		with h5py.File(out_path + 'clust_BCG-stack_SB_%.1f-sigma_jk-sub-%d.h5' % (sigma[nn], ll), 'r') as f:
		#with h5py.File(out_path + 'clust_BCG-stack_SB_%.1f-sigma_%d-patch_jk-sub-%d.h5' % (sigma, thres_S[nn], ll), 'r') as f:
			r_arr = np.array(f['r'])[:-1]
			sb_arr = np.array(f['sb'])[:-1]
			sb_err = np.array(f['sb_err'])[:-1]
			npix = np.array(f['npix'])[:-1]
			nratio = np.array(f['nratio'])[:-1]

		idvx = nratio > ( np.max(nratio) * (1 / 20) )
		up_r = np.max( r_arr[idvx] )
		up2sbr.append(up_r)

		# gradient
		mid_r, sb_k = slop2r(r_arr, sb_arr, npix, id_log = True,)

		ax1.plot(r_arr, sb_arr, ls = '-', color = mpl.cm.rainbow(ll / N_bin), alpha = 0.45, label = '%d' % ll)
		ax1.axvline(x = up_r, ls = '--', color = mpl.cm.rainbow(ll / N_bin), alpha = 0.45,)

		idvx = npix < 1.
		sb_arr[idvx] = np.nan
		r_arr[idvx] = np.nan

		tmp_sb.append(sb_arr)
		tmp_r.append(r_arr)

		tmp_slop.append(sb_k)
		tmp_slop_r.append(mid_r)

	ax0.set_ylim(2e-3, 3e-2)
	ax0.set_yscale('log')
	ax0.set_xlim(1e1, 1e3)
	ax0.set_xlabel('$ R[arcsec] $')
	ax0.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax0.set_xscale('log')
	ax0.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

	ax1.set_xlim(1e1, 1e3)
	ax1.set_xlabel('$ R[arcsec] $')
	ax1.set_xscale('log')
	ax1.legend(loc = 3, frameon = False, fontsize = 8, ncol = 5,)
	ax1.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax1.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.subplots_adjust(left = 0.1, right = 0.95, wspace = 0.,)
	plt.savefig('sub-sample_SB_%.1f-sigma.png' % (sigma[nn],), dpi = 300)
	#plt.savefig('sub-sample_SB_%d-patch_%.1f-sigm.png' % (thres_S[nn], sigma), dpi = 300)
	plt.close()

	tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB(tmp_sb, tmp_r, 0, N_bin)[4:]
	#sb_lim_r = np.ones( len(tt_jk_R) ) * lim_R

	tt_slop_r, tt_slop, tt_slop_err, slop_lim_R = jack_slop2r(tmp_slop_r, tmp_slop, N_bin,)
	#slop_lim_r = np.ones( len(tt_slop_r) ) * slop_lim_R

	up2sbr = np.array(up2sbr)
	sb_lim_r = np.ones( len(tt_jk_R) ) * np.mean(up2sbr)
	slop_lim_r = np.ones( len(tt_slop_r) ) * np.mean(up2sbr)
	### save the mean jack SB profile
	#with h5py.File('/home/xkchen/tmp/jk_test/CC_clust_BCG-stack_mean-jk-SB_%.1f-sigma.h5' % (sigma[nn]), 'w') as f:
	with h5py.File('/home/xkchen/tmp/jk_test/clust_BCG-stack_mean-jk-SB_%.1f-sigma.h5' % (sigma[nn]), 'w') as f:
	#with h5py.File('/home/xkchen/tmp/jk_test/clust_BCG-stack_mean-jk-SB_%.1f-sigma_%d-patch.h5' % (sigma, thres_S[nn]), 'w') as f:
		f['r'] = np.array(tt_jk_R)
		f['sb'] = np.array(tt_jk_SB)
		f['sb_err'] = np.array(tt_jk_err)
		f['lim_r'] = np.array(sb_lim_r)

	#with h5py.File('/home/xkchen/tmp/jk_test/CC_clust_BCG-stack_mean-jk-SB_slop_%.1f-sigma.h5' % (sigma[nn]), 'w') as f:
	with h5py.File('/home/xkchen/tmp/jk_test/clust_BCG-stack_mean-jk-SB_slop_%.1f-sigma.h5' % (sigma[nn]), 'w') as f:
	#with h5py.File('/home/xkchen/tmp/jk_test/clust_BCG-stack_mean-jk-SB_slop_%.1f-sigma_%d-patch.h5' % (sigma, thres_S[nn]), 'w') as f:
		f['r'] = np.array(tt_slop_r)
		f['slop'] = np.array(tt_slop)
		f['slop_err'] = np.array(tt_slop_err)
		f['lim_r'] = np.array(slop_lim_r)


jk_r, jk_sb, jk_err, sb_lim_r = [], [], [], []
jk_kr, jk_k, jk_kerr, k_lim_r = [], [], [], []

### individual sigma case
#for nn in range( len(thres_S) ):
for nn in range( len(sigma) ):
	"""
	if nn == 1:
		with h5py.File('/home/xkchen/tmp/jk_test/clust_BCG-stack_mean-jk-SB_%.1f-sigma.h5' % (sigma), 'r') as f:
			tt_jk_R = np.array(f['r'])
			tt_jk_SB = np.array(f['sb'])
			tt_jk_err = np.array(f['sb_err'])
			lim_r0 = np.array(f['lim_r'])[0]

		jk_r.append(tt_jk_R)
		jk_sb.append(tt_jk_SB)
		jk_err.append(tt_jk_err)
		sb_lim_r.append(lim_r0)

		with h5py.File('/home/xkchen/tmp/jk_test/clust_BCG-stack_mean-jk-SB_slop_%.1f-sigma.h5' % (sigma), 'r') as f:
			tt_k_R = np.array(f['r'])
			tt_k = np.array(f['slop'])
			tt_k_err = np.array(f['slop_err'])
			lim_r1 = np.array(f['lim_r'])[0]

		jk_kr.append(tt_k_R)
		jk_k.append(tt_k)
		jk_kerr.append(tt_k_err)
		k_lim_r.append(lim_r1)

	else:
	"""
	#with h5py.File('/home/xkchen/tmp/jk_test/CC_clust_BCG-stack_mean-jk-SB_%.1f-sigma.h5' % (sigma[nn]), 'r') as f:
	with h5py.File('/home/xkchen/tmp/jk_test/clust_BCG-stack_mean-jk-SB_%.1f-sigma.h5' % (sigma[nn]), 'r') as f:
	#with h5py.File('/home/xkchen/tmp/jk_test/clust_BCG-stack_mean-jk-SB_%.1f-sigma_%d-patch.h5' % (sigma, thres_S[nn]), 'r') as f:
		tt_jk_R = np.array(f['r'])
		tt_jk_SB = np.array(f['sb'])
		tt_jk_err = np.array(f['sb_err'])
		lim_r0 = np.array(f['lim_r'])[0]

	jk_r.append(tt_jk_R)
	jk_sb.append(tt_jk_SB)
	jk_err.append(tt_jk_err)
	sb_lim_r.append(lim_r0)

	#with h5py.File('/home/xkchen/tmp/jk_test/CC_clust_BCG-stack_mean-jk-SB_slop_%.1f-sigma.h5' % (sigma[nn]), 'r') as f:
	with h5py.File('/home/xkchen/tmp/jk_test/clust_BCG-stack_mean-jk-SB_slop_%.1f-sigma.h5' % (sigma[nn]), 'r') as f:
	#with h5py.File('/home/xkchen/tmp/jk_test/clust_BCG-stack_mean-jk-SB_slop_%.1f-sigma_%d-patch.h5' % (sigma, thres_S[nn]), 'r') as f:
		tt_k_R = np.array(f['r'])
		tt_k = np.array(f['slop'])
		tt_k_err = np.array(f['slop_err'])
		lim_r1 = np.array(f['lim_r'])[0]

	jk_kr.append(tt_k_R)
	jk_k.append(tt_k)
	jk_kerr.append(tt_k_err)
	k_lim_r.append(lim_r1)

	with h5py.File('/home/xkchen/tmp/jk_test/CC_clust_BCG-stack_%.1f-sigma_Mean-jk_img.h5' % (sigma[nn]), 'r') as f:
	#with h5py.File('/home/xkchen/tmp/jk_test/clust_BCG-stack_%.1f-sigma_%d-patch_Mean-jk_img.h5' % (sigma, thres_S[nn]), 'r') as f:
		tt_img = np.array(f['a'])
	xn, yn = np.int(tt_img.shape[1] / 2), np.int(tt_img.shape[0] / 2)
	block_m = grid_img(tt_img, 100, 100)[0]
'''
	fig = plt.figure( figsize = (13.12, 9.84) )
	fig.suptitle('stacking result [%.1f $\\sigma$]' % (sigma[nn]),)
	#fig.suptitle('stacking result [%.1f $\\sigma, %d patch$]' % (sigma,thres_S[nn]),)
	ax0 = fig.add_axes([0.05, 0.50, 0.40, 0.40])
	ax1 = fig.add_axes([0.55, 0.50, 0.40, 0.40])
	ax2 = fig.add_axes([0.05, 0.05, 0.40, 0.40])
	ax3 = fig.add_axes([0.55, 0.05, 0.40, 0.40])

	ax0.set_title('stacking image')
	tf = ax0.imshow(tt_img / pixel**2, cmap = 'seismic', origin = 'lower', vmin = -0.1, vmax = 0.1,)
	cb = plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = 'flux [nanomaggies / $arcsec^2$]')
	cb.formatter.set_powerlimits((0,0))
	clust = Circle(xy = (xn, yn), radius = lim_r0 / pixel, fill = False, ec = 'b', ls = '--', alpha = 0.5,)
	ax0.add_patch(clust)
	ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

	ax1.set_title('block means [100*100 grid]')
	tg = ax1.imshow(block_m / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = ax1, fraction = 0.035, pad = 0.01, label = 'flux [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits((0,0))
	ax1.tick_params(axis = 'both', which = 'both', direction = 'in')

	ax2.set_title('SB profile')
	ax2.plot(tt_jk_R, tt_jk_SB, ls = '-', color = 'r', alpha = 0.8, label = '%.1f$\\sigma$' % (sigma[nn]),)
	#ax2.plot(tt_jk_R, tt_jk_SB, ls = '-', color = 'r', alpha = 0.8, label = '%d-patch' % (thres_S[nn]),)
	ax2.fill_between(tt_jk_R, y1 = tt_jk_SB - tt_jk_err, y2 = tt_jk_SB + tt_jk_err, color = 'r', alpha = 0.2,)
	ax2.axvline(x = lim_r0, ls = '--', color = 'b', alpha = 0.5,)
	ax2.set_ylim(3.4e-3, 5.0e-3)
	ax2.set_xlim(1e2, 1e3)
	ax2.set_xlabel('$ R[arcsec] $')
	ax2.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax2.set_xscale('log')
	ax2.legend(loc = 3, frameon = False,)
	ax2.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax2.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax2.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)

	ax3.set_title('grident of SB profile')
	ax3.plot(tt_k_R, tt_k, color = 'r', ls = '-', alpha = 0.8, label = '%.1f $\\sigma$' % sigma[nn],)
	#ax3.plot(tt_k_R, tt_k, color = 'r', ls = '-', alpha = 0.8, label = '%d-patch' % (thres_S[nn]),)
	ax3.fill_between(tt_k_R,  y1 = tt_k - tt_k_err, y2 = tt_k + tt_k_err, color = 'r', alpha = 0.2,)
	ax3.axvline(x = lim_r1, ls = '--', color = 'b', alpha = 0.5,)
	ax3.set_ylim(-1e-2, 1e-2)
	ax3.set_xlim(5e1, 1e3)
	ax3.set_xlabel('$ R[arcsec] $')
	ax3.set_ylabel('$\\Delta$SB / $\\Delta logR$')
	ax3.set_xscale('log')
	ax3.legend(loc = 2, frameon = False,)
	ax3.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax3.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax3.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)

	plt.savefig('stack_result_%.1f-sigma.png' % sigma[nn], dpi = 300)
	#plt.savefig('stack_result_%.1f-sigma_%d-patch.png' % (sigma,thres_S[nn]), dpi = 300)
	plt.close()
'''
jk_r = np.array(jk_r)
jk_sb = np.array(jk_sb)
jk_err = np.array(jk_err)

jk_kr = np.array(jk_kr)
jk_k = np.array(jk_k)
jk_kerr = np.array(jk_kerr)


### SB as function of sigma
plt.figure()
gs = gridspec.GridSpec(2,1, height_ratios = [4,1])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

#for nn in range( len(thres_S) ):
for nn in range( len(sigma) ):

	ax0.plot(jk_r[nn], jk_sb[nn], ls = '-', color = mpl.cm.rainbow(nn/len(sigma) ), alpha = 0.8, label = '%.1f$\\sigma$' % (sigma[nn]),)
	ax0.fill_between(jk_r[nn], y1 = jk_sb[nn] - jk_err[nn], y2 = jk_sb[nn] + jk_err[nn], color = mpl.cm.rainbow(nn/len(sigma) ), alpha = 0.2,)
	ax0.axvline(x = sb_lim_r[nn], ls = '--', color = mpl.cm.rainbow(nn/len(sigma) ), alpha = 0.5,)

	ax1.plot(jk_r[0], jk_sb[nn][:len(jk_r[0])] - jk_sb[0], ls = '-', color = mpl.cm.rainbow(nn/len(sigma) ), alpha = 0.8,)
	ax1.fill_between(jk_r[0], 
		y1 = jk_sb[nn][:len(jk_r[0])] - jk_err[nn][:len(jk_r[0])] - jk_sb[0], y2 = jk_sb[nn][:len(jk_r[0])] + jk_err[nn][:len(jk_r[0])] - jk_sb[0], 
		color = mpl.cm.rainbow(nn/len(sigma) ), alpha = 0.2,)
	ax1.axvline(x = sb_lim_r[nn], ls = '--', color = mpl.cm.rainbow(nn/len(sigma) ), alpha = 0.5,)
	'''
	ax0.plot(jk_r[nn], jk_sb[nn], ls = '-', color = mpl.cm.rainbow(nn / (len(thres_S) + 2) ), alpha = 0.8, label = '%d-patch' % (thres_S[nn]),)
	ax0.fill_between(jk_r[nn], y1 = jk_sb[nn] - jk_err[nn], y2 = jk_sb[nn] + jk_err[nn], 
		color = mpl.cm.rainbow(nn / (len(thres_S) + 2) ), alpha = 0.2,)
	ax0.axvline(x = sb_lim_r[nn], ls = '--', color = mpl.cm.rainbow(nn / (len(thres_S) + 2) ), alpha = 0.5,)

	ax1.plot(jk_r[nn], jk_sb[nn] - jk_sb[0], ls = '-', color = mpl.cm.rainbow(nn / (len(thres_S) + 2) ), alpha = 0.8,)
	ax1.fill_between(jk_r[nn], y1 = jk_sb[nn] - jk_err[nn] - jk_sb[0], y2 = jk_sb[nn] + jk_err[nn] - jk_sb[0], 
		color = mpl.cm.rainbow(nn / (len(thres_S) + 2) ), alpha = 0.2,)
	ax1.axvline(x = sb_lim_r[nn], ls = '--', color = mpl.cm.rainbow(nn / (len(thres_S) + 2) ), alpha = 0.5,)
	'''
ax0.set_ylim(3.4e-3, 5.0e-3)
ax0.set_xlim(1e2, 1e3)
ax0.set_xlabel('$ R[arcsec] $')
ax0.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax0.set_xscale('log')
ax0.legend(loc = 3, frameon = False,)
ax0.grid(which = 'both', axis = 'both', alpha = 0.25)
ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

ax1.set_ylim(-3e-4, 3e-4)
ax1.set_xlim(ax0.get_xlim() )
ax1.set_ylabel('$SB - SB_{3.5 \\sigma}$', fontsize = 8.5,)
#ax1.set_ylabel('$SB - SB_{2-patch}$', fontsize = 8.5,)

ax1.set_xscale('log')
ax1.set_xlabel('$ R[arcsec] $')
ax1.grid(which = 'both', axis = 'both', alpha = 0.25)
ax1.tick_params(axis = 'both', which = 'both', direction = 'in')
ax0.set_xticks([])

plt.subplots_adjust(left = 0.15, right = 0.95, hspace = 0.)

plt.savefig('SB_sigma_relation.png', dpi = 300)
#plt.savefig('SB_patch_relation.png', dpi = 300)
plt.close()
raise

### test the grident
plt.figure()
ax = plt.subplot(111)

#for nn in range( len(thres_S) ):
for nn in range( len(sigma) ):

	ax.plot(jk_kr[nn], jk_k[nn], color = mpl.cm.rainbow(nn / len(sigma)), ls = '-', alpha = 0.8, label = '%.1f $\\sigma$' % sigma[nn],)
	ax.fill_between(jk_kr[nn],  y1 = jk_k[nn] - jk_kerr[nn], y2 = jk_k[nn] + jk_kerr[nn], color = mpl.cm.rainbow(nn/len(sigma) ), alpha = 0.2,)
	ax.axvline(	x = k_lim_r[nn], ls = '--', color = mpl.cm.rainbow(nn/len(sigma) ), alpha = 0.5,)
	'''
	ax.plot(jk_kr[nn], jk_k[nn], color = mpl.cm.rainbow(nn / (len(thres_S) + 2) ), ls = '-', alpha = 0.8, label = '%.1f $\\sigma$' % thres_S[nn],)
	ax.fill_between(jk_kr[nn],  y1 = jk_k[nn] - jk_kerr[nn], y2 = jk_k[nn] + jk_kerr[nn], color = mpl.cm.rainbow(nn/(len(thres_S) + 2) ), alpha = 0.2,)
	ax.axvline(	x = k_lim_r[nn], ls = '--', color = mpl.cm.rainbow(nn/(len(thres_S) + 2) ), alpha = 0.5,)
	'''
ax.set_ylim(-1e-2, 1e-2)
ax.set_xlim(5e1, 1e3)
ax.set_xlabel('$ R[arcsec] $')
ax.set_ylabel('$\\Delta$SB / $\\Delta logR$')
ax.set_xscale('log')
ax.legend(loc = 2, frameon = False,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in')
ax.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)

plt.subplots_adjust(left = 0.15, right = 0.95,)
plt.savefig('slop_test.png', dpi = 300)
plt.close()

