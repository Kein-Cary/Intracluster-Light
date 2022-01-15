import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.gridspec as gridspec

import astropy.units as U
import astropy.constants as C

import h5py
import time
import numpy as np
import pandas as pds
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.fits as fits
import scipy.stats as sts

from scipy import ndimage
from astropy import cosmology as apcy
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic as binned
from light_measure import light_measure, light_measure_Z0, light_measure_Z0_weit

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
kpc2m = U.kpc.to(U.m)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Msun2kg = U.M_sun.to(U.kg)
Lsun = C.L_sun.value*10**7
G = C.G.value

# cosmology model
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

#home = '/mnt/ddnfs/data_users/cxkttwl/ICL/'
home = '/media/xkchen/My Passport/data/SDSS/'
band = ['r', 'g', 'i', 'u', 'z']
mag_add = np.array([0, 0, 0, -0.04, 0.02])

def gau_func(x, mu, sigma):
	return sts.norm.pdf(x, mu, sigma)

def jack_SB(SB_array, R_array, band_id, N_bins,):

	## stacking profile based on flux
	dx_r = np.array(R_array)
	dy_sb = np.array(SB_array)

	n_r = dx_r.shape[1]
	Len = np.zeros( n_r, dtype = np.float32)
	for nn in range( n_r ):
		tmp_I = dy_sb[:,nn]
		idnn = np.isnan(tmp_I)
		Len[nn] = N_bins - np.sum(idnn)

	Stack_R = np.nanmean(dx_r, axis = 0)
	Stack_SB = np.nanmean(dy_sb, axis = 0)
	std_Stack_SB = np.nanstd(dy_sb, axis = 0)

	### limit the radius bin contribution at least 1/3 * N_bins
	id_min = Len >= np.int(N_bins / 3)
	Stack_R = Stack_R[ id_min ]
	Stack_SB = Stack_SB[ id_min ]
	std_Stack_SB = std_Stack_SB[ id_min ]

	N_img = Len[ id_min ]
	jk_Stack_err = np.sqrt( N_img - 1) * std_Stack_SB

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

	return jk_Stack_SB, jk_Stack_R, jk_Stack_err0, jk_Stack_err1, Stack_R, Stack_SB, jk_Stack_err

def SB_pro(img, R_bins, R_min, R_max, Cx, Cy, pix_size, zg, band_id):
    kk = band_id
    Intns, Intns_r, Intns_err = light_measure(img, R_bins, R_min, R_max, Cx, Cy, pix_size, zg)
    SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pix_size**2) + mag_add[kk]
    flux0 = Intns + Intns_err
    flux1 = Intns - Intns_err
    dSB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pix_size**2) + mag_add[kk]
    dSB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pix_size**2) + mag_add[kk]
    err0 = SB - dSB0
    err1 = dSB1 - SB
    id_nan = np.isnan(SB)
    SB_out, R_out, out_err0, out_err1 = SB[id_nan == False], Intns_r[id_nan == False], err0[id_nan == False], err1[id_nan == False]
    dSB0, dSB1 = dSB0[id_nan == False], dSB1[id_nan == False]
    idx_nan = np.isnan(dSB1)
    out_err1[idx_nan] = 100.

    return R_out, SB_out, out_err0, out_err1, Intns, Intns_r, Intns_err

def SB_pro_z0(img, pix_size, r_lim, R_pix, cx, cy, R_bins, band_id):
    kk = band_id
    Intns, Angl_r, Intns_err = light_measure_Z0(img, pix_size, r_lim, R_pix, cx, cy, R_bins)
    SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pix_size**2) + mag_add[kk]
    flux0 = Intns + Intns_err
    flux1 = Intns - Intns_err
    dSB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pix_size**2) + mag_add[kk]
    dSB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pix_size**2) + mag_add[kk]
    err0 = SB - dSB0
    err1 = dSB1 - SB
    id_nan = np.isnan(SB)

    SB_out, R_out, out_err0, out_err1 = SB[id_nan == False], Angl_r[id_nan == False], err0[id_nan == False], err1[id_nan == False]
    dSB0, dSB1 = dSB0[id_nan == False], dSB1[id_nan == False]
    idx_nan = np.isnan(dSB1)
    out_err1[idx_nan] = 100.

    return R_out, SB_out, out_err0, out_err1, Intns, Angl_r, Intns_err

def SB_pro_z0_weit(img, weit_img, pix_size, r_lim, R_pix, cx, cy, R_bins, band_id):

    kk = band_id
    Intns, Angl_r, Intns_err = light_measure_Z0_weit(img, weit_img, pix_size, r_lim, R_pix, cx, cy, R_bins)[:3]
    SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pix_size**2) + mag_add[kk]
    flux0 = Intns + Intns_err
    flux1 = Intns - Intns_err
    dSB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pix_size**2) + mag_add[kk]
    dSB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pix_size**2) + mag_add[kk]
    err0 = SB - dSB0
    err1 = dSB1 - SB
    id_nan = np.isnan(SB)

    SB_out, R_out, out_err0, out_err1 = SB[id_nan == False], Angl_r[id_nan == False], err0[id_nan == False], err1[id_nan == False]
    dSB0, dSB1 = dSB0[id_nan == False], dSB1[id_nan == False]
    idx_nan = np.isnan(dSB1)
    out_err1[idx_nan] = 100.

    return R_out, SB_out, out_err0, out_err1, Intns, Angl_r, Intns_err

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

def sub_figs(img_data, fig_title, file_name, SB, SB_r, SB_err, xlim_1d, ylim_1d,
	color_lo, color_hi, lo_mean, hi_mean, scale_x = 'linear', scale_y = 'linear',delt_N = 0):

	x0, y0 = np.int(img_data.shape[1] / 2), np.int(img_data.shape[0] / 2)

	### grid the stacking image
	N_step = 100
	patch_mean = grid_img(img_data, N_step, N_step)[0]
	patch_mean = patch_mean / pixel**2

	fig = plt.figure( figsize = (19.84, 4.8) )
	ax0 = fig.add_axes([0.03, 0.09, 0.30, 0.85])
	ax1 = fig.add_axes([0.38, 0.09, 0.30, 0.85])
	ax2 = fig.add_axes([0.73, 0.09, 0.25, 0.85])

	ax0.set_title(fig_title)
	clust0 = Circle(xy = (x0 - delt_N, y0 - delt_N), radius = img_data.shape[0] / 2, fill = False, ec = 'k', ls = '-', alpha = 0.75,)
	clust1 = Circle(xy = (x0 - delt_N, y0 - delt_N), radius = img_data.shape[1] / 2, fill = False, ec = 'k', ls = '--', alpha = 0.75,)
	ax0.add_patch(clust0)
	ax0.add_patch(clust1)

	tf = ax0.imshow(img_data / pixel**2, cmap = 'seismic', origin = 'lower', vmin = color_lo, vmax = color_hi,)
	cb0 = plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, spacing = 2.5e-3, label = 'SB[nanomaggies / arcsec^2]')
	cb0.formatter.set_powerlimits((0,0))

	ax0.axis('scaled')
	ax0.set_xlim(0, img_data.shape[1])
	ax0.set_ylim(0, img_data.shape[0])

	ax1.set_title('$2D \, \\bar{f} $ histogram [%d * %d $pixel^2$ / block]' % (N_step, N_step) )
	tg = ax1.imshow(patch_mean, origin = 'lower', cmap = 'rainbow', vmin = lo_mean, vmax = hi_mean,)
	cb1 = plt.colorbar(tg, ax = ax1, fraction = 0.035, pad = 0.01, label = 'SB[nanomaggies / arcsec^2]')
	cb1.formatter.set_powerlimits((0,0))

	ax2.errorbar(SB_r, SB, yerr = SB_err, xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1.25, 
		ecolor = 'r', elinewidth = 1.25, alpha = 0.5, )
	ax2.axvline(x = pixel * (img_data.shape[0] / 2), color = 'k', linestyle = '-', linewidth = 1, label = '$ H_{img} / 2$')
	ax2.axvline(x = pixel * (img_data.shape[1] / 2), color = 'k', linestyle = '--', linewidth = 1, label = '$ W_{img} / 2$')

	ax2.set_ylim(ylim_1d[0], ylim_1d[1])
	ax2.set_yscale( scale_y )
	ax2.set_xlim(xlim_1d[0], xlim_1d[1])
	ax2.set_xlabel('$ R[arcsec] $')
	ax2.set_xscale( scale_x )
	ax2.legend(loc = 3, frameon = False)
	ax2.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax2.tick_params(axis = 'both', which = 'both', direction = 'in')

	if scale_y == 'linear':
		ax2.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)
	if scale_y == 'log':
		tick_form = mpl.ticker.LogFormatter(labelOnlyBase = False)
		ax2.get_yaxis().set_minor_formatter(tick_form)

	plt.savefig(file_name, dpi = 300)
	plt.close()

def fig_2D(stack_img, rms_img, pix_cont_img, clim_00, clim_01, clim_10, clim_11, file_name,):

	fig = plt.figure( figsize = (19.84, 4.8) )
	ax0 = fig.add_axes([0.03, 0.09, 0.27, 0.85])
	ax1 = fig.add_axes([0.36, 0.09, 0.27, 0.85])
	ax2 = fig.add_axes([0.69, 0.09, 0.27, 0.85])

	ax0.set_title('stacking image')
	tf = ax0.imshow(stack_img, cmap = 'seismic', origin = 'lower', vmin = clim_00, vmax = clim_01,)
	cb = plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = 'flux [nanomaggies]')
	cb.formatter.set_powerlimits((0,0))
	ax0.axis('scaled')
	ax0.set_xlim(0, stack_img.shape[1])
	ax0.set_ylim(0, stack_img.shape[0])

	ax1.set_title('$ \\sqrt{sub-patch \; Variance} $ [100 * 100 $ pixel^2 $ / patch]')
	tg = ax1.imshow(rms_img, origin = 'lower', cmap = 'rainbow', vmin = clim_10, vmax = clim_11,)
	cb = plt.colorbar(tg, ax = ax1, fraction = 0.034, pad = 0.01, label = 'flux [nanomaggies]')
	cb.formatter.set_powerlimits((0,0))
	ax1.axis('scaled')
	ax1.set_xlim(0, rms_img.shape[1] - 1)
	ax1.set_ylim(0, rms_img.shape[0] - 1)

	ax2.set_title('pixel counts')
	th = ax2.imshow(pix_cont_img, origin = 'lower', cmap = 'rainbow', vmin = 1, vmax = np.nanmax(pix_cont_img),)
	cb = plt.colorbar(th, ax = ax2, fraction = 0.035, pad = 0.01, label = 'pixel counts')
	cb.formatter.set_powerlimits((0,0))
	ax2.axis('scaled')
	ax2.set_xlim(0, pix_cont_img.shape[1])
	ax2.set_ylim(0, pix_cont_img.shape[0])

	plt.tight_layout()
	plt.savefig(file_name, dpi = 300)
	plt.close()

##### stacking imgs
def re_mask():

	bins, R_smal, R_max = 95, 1, 3.0e3 ## for sky ICL

	for kk in range(1):
		"""
		##### center stacking
		with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_center-stack_test-train_correct.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_center-stack_test-train_correct.h5', 'r') as f:
			cen_stack_img = np.array(f['a'])

		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_center-stack_test-train.h5', 'r') as f:
		with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_center-stack_test-train_add-photo-G.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_center-stack_test-train.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_center-stack_test-train_add-photo-G.h5', 'r') as f:
			tmp_cen_img = np.array(f['a'])

		block_0 = grid_img(tmp_cen_img, 100, 100)[0]
		block_1 = grid_img(cen_stack_img, 100, 100)[0]

		fig = plt.figure( figsize = (19.84, 4.8) )
		ax0 = fig.add_axes([0.03, 0.09, 0.27, 0.85])
		ax1 = fig.add_axes([0.36, 0.09, 0.27, 0.85])
		ax2 = fig.add_axes([0.69, 0.09, 0.27, 0.85])

		ax0.set_title('previous')
		tf = ax0.imshow(block_0 / pixel**2, origin = 'lower', cmap = 'rainbow', vmin = 2e-3, vmax = 6e-3,)
		cb = plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = 'flux [nanomaggies / $arcsec^2$]')
		cb.formatter.set_powerlimits((0,0))
		ax0.axis('scaled')

		ax1.set_title('now [re-select out-liers]')
		tg = ax1.imshow(block_1 / pixel**2, origin = 'lower', cmap = 'rainbow', vmin = 2e-3, vmax = 6e-3,)
		cb = plt.colorbar(tg, ax = ax1, fraction = 0.034, pad = 0.01, label = 'flux [nanomaggies / $arcsec^2$]')
		cb.formatter.set_powerlimits((0,0))
		ax1.axis('scaled')

		ax2.set_title('previous - now')
		th = ax2.imshow( (block_0 - block_1) / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -2e-4, vmax = 2e-4,)
		cb = plt.colorbar(th, ax = ax2, fraction = 0.035, pad = 0.01, label = 'pixel counts')
		cb.formatter.set_powerlimits((0,0))
		ax2.axis('scaled')

		plt.tight_layout()
		plt.savefig('cluster_correct_test.png', dpi = 300)
		#plt.savefig('random_correct_test.png', dpi = 300)
		plt.close()

		with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_center-stack_pix-cont-train_correct.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_center-stack_pix-cont-train_correct.h5', 'r') as f:

		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_center-stack_pix-cont-train.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_center-stack_pix-cont-train_add-photo-G.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_center-stack_pix-cont-train.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_center-stack_pix-cont-train_add-photo-G.h5', 'r') as f:
			cen_pix_cont = np.array(f['a'])

		xn, yn = np.int(cen_stack_img.shape[1] / 2), np.int(cen_stack_img.shape[0] / 2)
		weit_cen_I, weit_cen_I_r, weit_cen_I_err = SB_pro_z0_weit(cen_stack_img, cen_pix_cont, pixel, 1, 1270, xn, yn, bins, kk)[4:]
		weit_cen_I, weit_cen_I_err = weit_cen_I / pixel**2, weit_cen_I_err / pixel**2

		with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_center-stack_SB_correct.h5', 'w') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_center-stack_SB_correct.h5', 'w') as f:

		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_center-stack_weit-SB.h5', 'w') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_center-stack_weit-SB_add-photo-G.h5', 'w') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_center-stack_weit-SB.h5', 'w') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_center-stack_weit-SB_add-photo-G.h5', 'w') as f:
			f['r'] = np.array(weit_cen_I_r)
			f['sb'] = np.array(weit_cen_I)
			f['sb_err'] = np.array(weit_cen_I_err)

		idux = weit_cen_I_r >= 100
		weit_mean_sb = np.nanmean( weit_cen_I[idux] )

		tit_str = 'masking source only [no out-lier]'
		fil_str = 'center_stack_SB.png'

		xlim_set = (1e1, 1e3)
		ylim_set = (2e-3, 7e-3)
		#sub_figs(cen_stack_img, tit_str, fil_str, weit_cen_I, weit_cen_I_r, weit_cen_I_err, xlim_set, ylim_set, -0.15, 0.15, 2e-3, 6e-3, 'log', 'log',)
		grd_Var = grid_img(cen_stack_img, 100, 100)[1]
		#fig_2D(cen_stack_img, grd_Var, cen_pix_cont, -0.02, 0.02, 0, 5e-3, file_name = 'center-stack_2D_fig.png')

		with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_center-stack_weit-SB.h5', 'r') as f:
			cen_I_r = np.array(f['r'])
			cen_I = np.array(f['sb'])
			cen_I_err = np.array(f['sb_err'])

		with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_center-stack_weit-SB_add-photo-G.h5', 'r') as f:
			cen_I_r_addG = np.array(f['r'])
			cen_I_addG = np.array(f['sb'])
			cen_I_err_addG = np.array(f['sb_err'])

		with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_center-stack_weit-SB.h5', 'r') as f:
			rnd_I_r = np.array(f['r'])
			rnd_I = np.array(f['sb'])
			rnd_I_err = np.array(f['sb_err'])

		with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_center-stack_weit-SB_add-photo-G.h5', 'r') as f:
			rnd_I_r_addG = np.array(f['r'])
			rnd_I_addG = np.array(f['sb'])
			rnd_I_err_addG = np.array(f['sb_err'])

		with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_center-stack_SB_correct.h5', 'r') as f:
			cen_I_r_cor = np.array(f['r'])
			cen_I_cor = np.array(f['sb'])
			cen_I_err_cor = np.array(f['sb_err'])

		with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_center-stack_SB_correct.h5', 'r') as f:
			rnd_I_r_cor = np.array(f['r'])
			rnd_I_cor = np.array(f['sb'])
			rnd_I_err_cor = np.array(f['sb_err'])

		plt.figure()
		gs = gridspec.GridSpec(2,1, height_ratios = [4,1])
		ax0 = plt.subplot(gs[0])
		ax1 = plt.subplot(gs[1])
		'''
		ax0.errorbar(cen_I_r_addG, cen_I_addG, yerr = cen_I_err_addG, xerr = None, color = 'm', marker = 'None', ls = '-.',
			ecolor = 'm', alpha = 0.5, label = 'cluster img [edge galaxy correction]')
		ax0.errorbar(cen_I_r, cen_I, yerr = cen_I_err, xerr = None, color = 'r', marker = 'None', ls = '-', 
			ecolor = 'r', alpha = 0.5, label = 'cluster img')

		ax0.errorbar(rnd_I_r_addG, rnd_I_addG, yerr = rnd_I_err_addG, xerr = None, color = 'c', marker = 'None', ls = '-.', 
			ecolor = 'c', alpha = 0.5, label = 'random img [edge galaxy correction]')
		ax0.errorbar(rnd_I_r, rnd_I, yerr = rnd_I_err, xerr = None, color = 'b', marker = 'None', ls = '-', linewidth = 1.25, 
			ecolor = 'b', elinewidth = 1.25, alpha = 0.5, label = 'random img')
		'''
		ax0.errorbar(cen_I_r_addG, cen_I_addG, yerr = cen_I_err_addG, xerr = None, color = 'r', marker = 'None', ls = '-',
			ecolor = 'r', alpha = 0.5, label = 'cluster img')
		ax0.errorbar(cen_I_r_cor, cen_I_cor, yerr = cen_I_err_cor, xerr = None, color = 'm', marker = 'None', ls = '-.',
			ecolor = 'm', alpha = 0.5, label = 'cluster img[re-select out-lier]')
		ax0.errorbar(rnd_I_r_addG, rnd_I_addG, yerr = rnd_I_err_addG, xerr = None, color = 'b', marker = 'None', ls = '-', 
			ecolor = 'b', alpha = 0.5, label = 'random img')
		ax0.errorbar(rnd_I_r_cor, rnd_I_cor, yerr = rnd_I_err_cor, xerr = None, color = 'c', marker = 'None', ls = '-.', 
			ecolor = 'c', alpha = 0.5, label = 'random img [re-select out-lier]')

		ax0.axvline(x = pixel * (cen_stack_img.shape[0] / 2), color = 'k', linestyle = '-', linewidth = 1, label = '$ H_{img} / 2$')
		ax0.axvline(x = pixel * (cen_stack_img.shape[1] / 2), color = 'k', linestyle = '--', linewidth = 1, label = '$ W_{img} / 2$')

		ax0.set_ylim(2e-3, 7e-3)
		ax0.set_yscale('log')
		ax0.set_xlim(1e1, 1e3)
		ax0.set_xscale('log')
		ax0.set_xlabel('$ R[arcsec] $')
		ax0.set_ylabel('SB [nanomaggies / $ arcsec^2 $]')
		ax0.legend(loc = 3, frameon = False)
		ax0.grid(which = 'both', axis = 'both', alpha = 0.25)
		ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

		#ax1.plot(cen_I_r, cen_I / cen_I_addG, ls = '-', color = 'r', alpha = 0.5, label = 'cluster img')
		#ax1.plot(rnd_I_r, rnd_I / rnd_I_addG, ls = '-', color = 'b', alpha = 0.5, label = 'random img')
		ax1.plot(cen_I_r, cen_I_addG / cen_I_cor, ls = '-', color = 'r', alpha = 0.5, label = 'cluster img')
		ax1.plot(rnd_I_r, rnd_I_addG / rnd_I_cor, ls = '-', color = 'b', alpha = 0.5, label = 'random img')

		ax1.set_ylim(0.9, 1.1)
		ax1.set_xlim(ax0.get_xlim() )
		ax1.set_ylabel('$SB / SB_{corrected}$', fontsize = 8.5,)
		ax1.set_xscale('log')
		ax1.set_xlabel('$ R[arcsec] $')
		ax1.legend(loc = 3, fontsize = 8.5,)
		ax1.grid(which = 'both', axis = 'both', alpha = 0.25)
		ax1.tick_params(axis = 'both', which = 'both', direction = 'in')
		ax0.set_xticks([])

		plt.subplots_adjust(left = 0.15, right = 0.95, hspace = 0.05)
		plt.savefig('cen-stack_SB_compare.png', dpi = 300)
		plt.close()
		"""

		##### BCG-stack
		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_test-train.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_test-train_add-photo-G.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_test-train.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_test-train_add-photo-G.h5', 'r') as f:

		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_test-train_correct.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/new_1000_test/clust_test-1000_BCG-stack_test-train_correct.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_test-train_correct.h5', 'r') as f:
		with h5py.File('/home/xkchen/Downloads/new_1000_test/random_test-1000_BCG-stack_test-train_correct.h5', 'r') as f:

		## cut-stack
		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_N_edg-500.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_N_edg-500.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_N_edg-500_correct.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_N_edg-500_correct.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/new_1000_test/clust_test-1000_BCG-stack_N_edg-500_correct.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/new_1000_test/random_test-1000_BCG-stack_N_edg-500_correct.h5', 'r') as f:
			stack_img = np.array(f['a'])


		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_pix-cont-train.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_pix-cont-train_add-photo-G.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_pix-cont-train.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_pix-cont-train_add-photo-G.h5', 'r') as f:

		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_pix-cont-train_correct.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/new_1000_test/clust_test-1000_BCG-stack_pix-cont-train_correct.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_pix-cont-train_correct.h5', 'r') as f:
		with h5py.File('/home/xkchen/Downloads/new_1000_test/random_test-1000_BCG-stack_pix-cont-train_correct.h5', 'r') as f:

		## cut-stack
		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_pix-cont_N_edg-500.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_pix-cont_N_edg-500.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_pix-cont_N_edg-500_correct.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_pix-cont_N_edg-500_correct.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/new_1000_test/clust_test-1000_BCG-stack_pix-cont_N_edg-500_correct.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/new_1000_test/random_test-1000_BCG-stack_pix-cont_N_edg-500_correct.h5', 'r') as f:
			bcg_pix_cont = np.array(f['a'])

		xn, yn = np.int(stack_img.shape[1] / 2), np.int(stack_img.shape[0] / 2)
		weit_bcg_I, weit_bcg_I_r, weit_bcg_I_err = SB_pro_z0_weit(stack_img, bcg_pix_cont, pixel, 1, 3000, xn, yn, np.int(1.22 * bins), kk)[4:]
		weit_bcg_I, weit_bcg_I_err = weit_bcg_I / pixel**2, weit_bcg_I_err / pixel**2


		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_train_SB.h5', 'w') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_train_SB_add-photo-G.h5', 'w') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_train_SB.h5', 'w') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_train_SB_add-photo-G.h5', 'w') as f:

		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_SB_correct.h5', 'w') as f:
		#with h5py.File('/home/xkchen/Downloads/new_1000_test/clust_test-1000_BCG-stack_SB_correct.h5', 'w') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_SB_correct.h5', 'w') as f:
		with h5py.File('/home/xkchen/Downloads/new_1000_test/random_test-1000_BCG-stack_SB_correct.h5', 'w') as f:

		## cut-stack
		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_SB_N-edg-500.h5', 'w') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_SB_N-edg-500.h5', 'w') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_SB_N-edg-500_correct.h5', 'w') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_SB_N-edg-500_correct.h5', 'w') as f:
		#with h5py.File('/home/xkchen/Downloads/new_1000_test/clust_test-1000_BCG-stack_SB_N-edg-500_correct.h5', 'w') as f:
		#with h5py.File('/home/xkchen/Downloads/new_1000_test/random_test-1000_BCG-stack_SB_N-edg-500_correct.h5', 'w') as f:
			f['r'] = np.array(weit_bcg_I_r)
			f['sb'] = np.array(weit_bcg_I)
			f['sb_err'] = np.array(weit_bcg_I_err)

		### mark position
		r_mark_0 = 50 / pixel ## 50", assuming BCG with 200kpc size and z~0.25
		r_mark_1 = 100 / pixel ## beyond 100", the SB profile is more flat

		tit_str = 'masking source only [no out-lier]'
		fil_str = 'BCG-stack_test.png'

		xlim_set = (1e1, 1e3)
		#ylim_set = (3e-3, 3e-2)
		ylim_set = (2e-3, 7e-3)
		#sub_figs(stack_img, tit_str, fil_str, weit_bcg_I, weit_bcg_I_r, weit_bcg_I_err, xlim_set, ylim_set, -0.15, 0.15, 2e-3, 6e-3, 'log', 'log',)
		grd_Var = grid_img(stack_img, 100, 100)[1]
		#fig_2D(stack_img, grd_Var, bcg_pix_cont, -0.02, 0.02, 0, 4e-2, file_name = 'BCG-stack_2D_fig.png')


		### SB comparison
		with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_train_SB.h5', 'r') as f:
			bcg_I_r = np.array(f['r'])
			bcg_I = np.array(f['sb'])
			bcg_I_err = np.array(f['sb_err'])

		with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_train_SB_add-photo-G.h5', 'r') as f:
			ext_bcg_I_r = np.array(f['r'])
			ext_bcg_I = np.array(f['sb'])
			ext_bcg_I_err = np.array(f['sb_err'])

		with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_train_SB.h5', 'r') as f:
			rand_I_r = np.array(f['r'])
			rand_I = np.array(f['sb'])
			rand_I_err = np.array(f['sb_err'])

		with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_train_SB_add-photo-G.h5','r') as f:
			ext_rand_I_r = np.array(f['r'])
			ext_rand_I = np.array(f['sb'])
			ext_rand_I_err = np.array(f['sb_err'])

		with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_SB_N-edg-500.h5', 'r') as f:
			cut_clus_I_r = np.array(f['r'])
			cut_clus_I = np.array(f['sb'])
			cut_clus_I_err = np.array(f['sb_err'])

		with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_SB_N-edg-500.h5', 'r') as f:
			cut_rand_I_r = np.array(f['r'])
			cut_rand_I = np.array(f['sb'])
			cut_rand_I_err = np.array(f['sb_err'])

		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_SB_N-edg-500_correct.h5', 'r') as f:
		with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_SB_correct.h5', 'r') as f:
			bcg_I_r_cor = np.array(f['r'])
			bcg_I_cor = np.array(f['sb'])
			bcg_I_err_cor = np.array(f['sb_err'])

		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_SB_N-edg-500_correct.h5', 'r') as f:
		with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_SB_correct.h5', 'r') as f:
			rand_I_r_cor = np.array(f['r'])
			rand_I_cor = np.array(f['sb'])
			rand_I_err_cor = np.array(f['sb_err'])

		#with h5py.File('/home/xkchen/Downloads/new_1000_test/random_test-1000_BCG-stack_SB_N-edg-500_correct.h5', 'r') as f:
		with h5py.File('/home/xkchen/Downloads/new_1000_test/random_test-1000_BCG-stack_SB_correct.h5', 'r') as f:
			new_rand_I_r_cor = np.array(f['r'])
			new_rand_I_cor = np.array(f['sb'])
			new_rand_I_err_cor = np.array(f['sb_err'])

		#with h5py.File('/home/xkchen/Downloads/new_1000_test/clust_test-1000_BCG-stack_SB_N-edg-500_correct.h5', 'r') as f:
		with h5py.File('/home/xkchen/Downloads/new_1000_test/clust_test-1000_BCG-stack_SB_correct.h5', 'r') as f:
			new_bcg_I_r_cor = np.array(f['r'])
			new_bcg_I_cor = np.array(f['sb'])
			new_bcg_I_err_cor = np.array(f['sb_err'])

		plt.figure()
		ax = plt.subplot(111)
		'''
		ax.errorbar(bcg_I_r, bcg_I, yerr = bcg_I_err, xerr = None, color = 'r', marker = 'None', ls = '-', ecolor = 'r', 
			alpha = 0.5, label = 'cluster img')
		ax.errorbar(rand_I_r, rand_I, yerr = rand_I_err, xerr = None, color = 'b', marker = 'None', ls = '-', ecolor = 'b', 
			alpha = 0.5, label = 'random img')
		'''
		'''
		ax.plot(bcg_I_r, bcg_I, color = 'r', ls = '-', alpha = 0.5, label = 'cluster img')
		ax.plot(cut_clus_I_r, cut_clus_I, color = 'm', ls = '-.', alpha = 0.5, label = 'cluster img [cut edge pixels]')
		#ax.plot(ext_bcg_I_r, ext_bcg_I, color = 'm', ls = '-.', alpha = 0.5, label = 'cluster img [edge galaxy correction]')

		ax.plot(rand_I_r, rand_I, color = 'b', ls = '-', alpha = 0.5, label = 'random img')
		ax.plot(cut_rand_I_r, cut_rand_I, color = 'c', ls = '-.', alpha = 0.5, label = 'random img [cut edge pixels]')
		#ax.plot(ext_rand_I_r, ext_rand_I, color = 'c', ls = '-.', alpha = 0.5, label = 'random img [edge galaxy correction]')
		'''
		'''
		ax.plot(bcg_I_r, bcg_I, color = 'm', ls = '-', alpha = 0.7, label = 'cluster',)
		ax.fill_between(bcg_I_r, y1 = bcg_I - bcg_I_err, y2 = bcg_I + bcg_I_err, color = 'm', alpha = 0.30,)
		ax.plot(rand_I_r, rand_I, color = 'c', ls = '-', alpha = 0.7, label = 'random',)
		ax.fill_between(rand_I_r, y1 = rand_I - rand_I_err, y2 = rand_I + rand_I_err, color = 'c', alpha = 0.30,)
		'''
		ax.plot(bcg_I_r_cor, bcg_I_cor, color = 'r', ls = '-', alpha = 0.7, label = 'cluster img [old sample]')
		ax.fill_between(bcg_I_r_cor, y1 = bcg_I_cor - bcg_I_err_cor, y2 = bcg_I_cor + bcg_I_err_cor, color = 'r', alpha = 0.30,)
		ax.plot(rand_I_r_cor, rand_I_cor, color = 'b', ls = '-', alpha = 0.7, label = 'random img [old sample]')
		ax.fill_between(rand_I_r_cor, y1 = rand_I_cor - rand_I_err_cor, y2 = rand_I_cor + rand_I_err_cor, color = 'b', alpha = 0.30,)

		ax.plot(new_bcg_I_r_cor, new_bcg_I_cor, color = 'm', ls = '-', alpha = 0.7, label = 'cluster img [new sample]')
		ax.fill_between(new_bcg_I_r_cor, y1 = new_bcg_I_cor - new_bcg_I_err_cor, 
			y2 = new_bcg_I_cor + new_bcg_I_err_cor, color = 'm', alpha = 0.3,)
		ax.plot(new_rand_I_r_cor, new_rand_I_cor, color = 'c', ls = '-', alpha = 0.7, label = 'random img [new sample]')
		ax.fill_between(new_rand_I_r_cor, y1 = new_rand_I_cor - new_rand_I_err_cor, 
			y2 = new_rand_I_cor + new_rand_I_err_cor, color = 'c', alpha = 0.3,)

		ax.axvline(x = pixel * (stack_img.shape[0] / 2), color = 'k', linestyle = '-', linewidth = 1, label = '$ H_{img} / 2$', alpha = 0.5,)
		ax.axvline(x = pixel * (stack_img.shape[1] / 2), color = 'k', linestyle = '--', linewidth = 1, label = '$ W_{img} / 2$', alpha = 0.5,)
		ax.set_ylim(3e-3, 3e-2)
		ax.set_yscale('log')
		ax.set_xlim(1e1, 1e3)
		ax.set_xlabel('$ R[arcsec] $')
		ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
		ax.set_xscale('log')
		ax.legend(loc = 'upper center', frameon = False, fontsize = 8)
		ax.grid(which = 'both', axis = 'both', alpha = 0.25)
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')

		plt.subplots_adjust(left = 0.15, right = 0.95,)
		plt.savefig('BCG-stack_SB_compare.png', dpi = 300)
		plt.close()

	raise

def sky_stack():

	bins, R_smal, R_max = 95, 1, 3.0e3

	for kk in range(1):

		##### center stacking sky-img
		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_center-stack_sky-train.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_center-stack_sky-train.h5', 'r') as f:

		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_stack_sky-train.h5', 'r') as f:
		with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_stack_sky-train.h5', 'r') as f:
			mean_sky = np.array(f['a'])

		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_center-stack_sky_pix-cont-train.h5', 'r') as f:
		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_center-stack_sky_pix-cont-train.h5', 'r') as f:

		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_stack_sky_pix-cont-train.h5', 'r') as f:
		with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_stack_sky_pix-cont-train.h5', 'r') as f:
			pix_conts = np.array(f['a'])

		xn, yn = np.int(mean_sky.shape[1] / 2), np.int(mean_sky.shape[0] / 2)
		#sky_cen_I, sky_cen_I_r, sky_cen_I_err = SB_pro_z0_weit(mean_sky, pix_conts, pixel, 1, 1270, xn, yn, bins, kk)[4:]
		sky_cen_I, sky_cen_I_r, sky_cen_I_err = SB_pro_z0_weit(mean_sky, pix_conts, pixel, 1, 3000, xn, yn, np.int(1.22 * bins), kk)[4:]
		sky_cen_I, sky_cen_I_err = sky_cen_I / pixel**2, sky_cen_I_err / pixel**2

		#with h5py.File('/home/xkchen/Downloads/random_test-1000_center-stack_sky-SB.h5', 'w') as f:
		#with h5py.File('/home/xkchen/Downloads/cluster_test-1000_center-stack_sky-SB.h5', 'w') as f:

		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_stack_sky-SB.h5', 'w') as f:
		with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_stack_sky-SB.h5', 'w') as f:
			f['r'] = np.array(sky_cen_I_r)
			f['sb'] = np.array(sky_cen_I)
			f['sb_err'] = np.array(sky_cen_I_err)

		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_center-stack_sky-train.h5', 'r') as f:
		with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_stack_sky-train.h5', 'r') as f:
			rnd_sky = np.array(f['a'])

		#with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_center-stack_sky-SB.h5', 'r') as f:
		with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_stack_sky-SB.h5', 'r') as f:
			rnd_cen_r = np.array(f['r'])
			rnd_cen_I = np.array(f['sb'])
			rnd_cen_I_err = np.array(f['sb_err'])	

		plt.figure()
		gs = gridspec.GridSpec(2,1, height_ratios = [3,2],)
		ax = plt.subplot(gs[0])
		ax1 = plt.subplot(gs[1])

		ax.set_title('sky SB profile comparison')
		ax.errorbar(sky_cen_I_r, sky_cen_I + 0.214, yerr = sky_cen_I_err, xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1.25, 
			ecolor = 'r', elinewidth = 1.25, alpha = 0.5, label = 'cluster + 0.214')
		ax.errorbar(rnd_cen_r, rnd_cen_I, yerr = rnd_cen_I_err, xerr = None, color = 'b', marker = 'None', ls = '-', linewidth = 1.25, 
			ecolor = 'b', elinewidth = 1.25, alpha = 0.5, label = 'random')

		ax.set_ylim(4.0, 4.6)
		ax.set_xlim(1e1, 1e3)
		ax.set_xlabel('R [arcsec]')
		ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
		ax.set_xscale('log')
		ax.legend(loc = 2, frameon = False)
		ax.grid(which = 'both', axis = 'both', alpha = 0.25)
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')

		ax1.plot(rnd_cen_r, sky_cen_I - rnd_cen_I, color = 'r', ls= '-', alpha = 0.5,)
		ax1.set_ylim(-0.35, 0.1)
		ax1.set_xlim(ax.get_xlim())
		ax1.set_xlabel('R [arcsec]')
		ax1.set_xscale('log')
		ax1.grid(which = 'both', axis = 'both', alpha = 0.25)
		ax1.tick_params(axis = 'both', which = 'both', direction = 'in')
		ax.set_xticks([])

		plt.subplots_adjust(hspace = 0.03)
		plt.savefig('sky_SB_compare.png', dpi = 300)
		plt.close()

		fig = plt.figure( figsize = (19.84, 4.8) )
		ax0 = fig.add_axes([0.03, 0.09, 0.27, 0.85])
		ax1 = fig.add_axes([0.36, 0.09, 0.27, 0.85])
		ax2 = fig.add_axes([0.69, 0.09, 0.27, 0.85])

		ax0.set_title('cluster')
		tf = ax0.imshow(mean_sky / pixel**2, origin = 'lower', cmap = 'rainbow', )#vmin = 4.15, vmax = 4.35,)
		plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]')
		ax0.set_xticks([])
		ax0.set_yticks([])

		ax1.set_title('random')
		tf = ax1.imshow(rnd_sky / pixel**2, origin = 'lower', cmap = 'rainbow', )#vmin = 4.15, vmax = 4.35,)
		plt.colorbar(tf, ax = ax1, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]')
		ax1.set_xticks([])
		ax1.set_yticks([])

		diffi_img = mean_sky - rnd_sky
		ax2.set_title('cluster - random')
		tf = ax2.imshow(diffi_img / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
		plt.colorbar(tf, ax = ax2, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]')
		ax2.set_xticks([])
		ax2.set_yticks([])

		plt.savefig('sky_img_differ.png', dpi = 300)	
		plt.close()

		raise
		'''
		##### cluster sky residual img -- random stacking
		N_sample = 10
		#d_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_rand-stack_sky-train_%d.h5'
		#out_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_rand-stack_Mean_res-sky-img.h5'
		d_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_rand-stack_sky-train_%d.h5'
		out_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_rand-stack_Mean_res-sky-img.h5'
		aveg_stack_img(N_sample, d_file, out_file)

		#d_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_rand-stack_sky_pix-cont-train_%d.h5'
		#out_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_rand-stack_Mean_res-sky-pix-cont.h5'
		d_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_rand-stack_sky_pix-cont-train_%d.h5'
		out_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_rand-stack_Mean_res-sky-pix-cont.h5'
		aveg_stack_img(N_sample, d_file, out_file)

		#d_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_rand-stack_sky_var-train_%d.h5'
		#out_file = '/home/xkchen/Downloads/test_imgs/clust_test-1000_rand-stack_Mean_res-sky_Var.h5'
		d_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_rand-stack_sky_var-train_%d.h5'
		out_file = '/home/xkchen/Downloads/test_imgs/random_test-1000_rand-stack_Mean_res-sky_Var.h5'
		aveg_stack_img(N_sample, d_file, out_file)
		'''
		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_rand-stack_Mean_res-sky-img.h5', 'r') as f:
		with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_rand-stack_Mean_res-sky-img.h5', 'r') as f:
			M_rnd_sky = np.array(f['a'])
		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_rand-stack_Mean_res-sky-pix-cont.h5', 'r') as f:
		with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_rand-stack_Mean_res-sky-pix-cont.h5', 'r') as f:
			M_pix_cont = np.array(f['a'])
		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_rand-stack_Mean_res-sky_Var.h5', 'r') as f:
		with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_rand-stack_Mean_res-sky_Var.h5', 'r') as f:
			M_rnd_sky_Var = np.array(f['a'])

		xn, yn = np.int(M_rnd_sky.shape[1] / 2), np.int(M_rnd_sky.shape[0] / 2)
		M_sky_I, M_sky_I_r, M_sky_I_err = SB_pro_z0_weit(M_rnd_sky, M_pix_cont, pixel, 1, 3000, xn, yn, np.int(1.22 * bins), kk)[4:]
		M_sky_I, M_sky_I_err = M_sky_I / pixel**2, M_sky_I_err / pixel**2

		#tit_str = 'cluster residual sky stacking [random center]'
		#fil_str = 'cluster_rnd-sky-stack_test.png'
		tit_str = 'random residual sky stacking [random center]'
		fil_str = 'random_rnd-sky-stack_test.png'
		xlim_set = (1e1, 1e3)
		ylim_set = (-2e-4, 5e-4)
		sub_figs(M_rnd_sky, tit_str, fil_str, M_sky_I, M_sky_I_r, M_sky_I_err, xlim_set, ylim_set, -3e-3, 3e-3, 1e-4, 1e-3, 'linear', 'linear',)
		grd_Var = grid_img(M_rnd_sky, 100, 100)[1]
		#fig_2D(M_rnd_sky, grd_Var, M_pix_cont, -2e-4, 2e-4, 0, 4e-4, file_name = 'cluster_rand-sky-stack_2D_fig.png')
		fig_2D(M_rnd_sky, grd_Var, M_pix_cont, -2e-4, 2e-4, 0, 4e-4, file_name = 'random_rand-sky-stack_2D_fig.png')

		##### BCG-stacking case
		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_sky-train.h5', 'r') as f:
		with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_sky-train.h5', 'r') as f:
			bcg_sky = np.array(f['a'])
		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_sky_pix-cont-train.h5', 'r') as f:
		with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_sky_pix-cont-train.h5', 'r') as f:
			bcg_sky_cont = np.array(f['a'])
		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_BCG-stack_sky_var-train.h5', 'r') as f:
		with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_BCG-stack_sky_var-train.h5', 'r') as f:
			bcg_sky_rms = np.array(f['a'])

		xn, yn = np.int(bcg_sky.shape[1] / 2), np.int(bcg_sky.shape[0] / 2)
		bcg_sky_I, bcg_sky_I_r, bcg_sky_I_err = SB_pro_z0_weit(bcg_sky, bcg_sky_cont, pixel, 1, 3000, xn, yn, np.int(1.22 * bins), kk)[4:]
		bcg_sky_I, bcg_sky_I_err = bcg_sky_I / pixel**2, bcg_sky_I_err / pixel**2

		#tit_str = 'cluster residual sky stacking [centered on BCGs]'
		#fil_str = 'cluster_bcg-sky-stack_test.png'
		tit_str = 'random residual sky stacking [centered on BCGs]'
		fil_str = 'random_bcg-sky-stack_test.png'
		xlim_set = (1e1, 1e3)
		ylim_set = (-2e-4, 5e-4)
		sub_figs(bcg_sky, tit_str, fil_str, bcg_sky_I, bcg_sky_I_r, bcg_sky_I_err, xlim_set, ylim_set, -3e-3, 3e-3, 1e-4, 1e-3, 'linear', 'linear',)
		grd_Var = grid_img(bcg_sky, 100, 100)[1]
		#fig_2D(bcg_sky, grd_Var, bcg_sky_cont, -2e-4, 2e-4, 0, 4e-4, file_name = 'cluster_bcg-sky-stack_2D_fig.png')
		fig_2D(bcg_sky, grd_Var, bcg_sky_cont, -2e-4, 2e-4, 0, 4e-4, file_name = 'random_bcg-sky-stack_2D_fig.png')

		#with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_res-sky-SB.h5', 'w') as f:
		with h5py.File('/home/xkchen/Downloads/test_imgs/random_test-1000_res-sky-SB.h5', 'w') as f:
			f['r'] = np.array(M_sky_I_r)
			f['rnd_sb'] = np.array(M_sky_I)
			f['rnd_sb_err'] = np.array(M_sky_I_err)
			f['bcg_sb'] = np.array(bcg_sky_I)
			f['bcg_sb_err'] = np.array(bcg_sky_I_err)

		fig = plt.figure( figsize = (13.12, 9.84) )
		fig.suptitle('stacking residual sky img [cluster]')
		ax0 = fig.add_axes([0.05, 0.50, 0.40, 0.40])
		ax1 = fig.add_axes([0.55, 0.50, 0.40, 0.40])
		ax2 = fig.add_axes([0.05, 0.05, 0.40, 0.40])
		ax3 = fig.add_axes([0.55, 0.05, 0.40, 0.40])

		ax0.set_title('centered on BCGs')
		tf = ax0.imshow(bcg_sky / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -3e-3, vmax = 3e-3,)
		cb = plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = 'pixel SB [nanomaggies / $pixel^2$]')
		cb.formatter.set_powerlimits((0,-3))
		clust0 = Circle(xy = (xn, yn), radius = bcg_sky.shape[0] / 2, fill = False, ec = 'k', ls = '-', alpha = 0.75,)
		clust1 = Circle(xy = (xn, yn), radius = bcg_sky.shape[1] / 2, fill = False, ec = 'k', ls = '--', alpha = 0.75,)
		ax0.add_patch(clust0)
		ax0.add_patch(clust1)

		ax1.set_title('random center')
		tg = ax1.imshow(M_rnd_sky / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -3e-3, vmax = 3e-3,)
		cb = plt.colorbar(tg, ax = ax1, fraction = 0.035, pad = 0.01, label = 'pixel SB [nanomaggies / $pixel^2$]')
		cb.formatter.set_powerlimits((0,-3))
		clust0 = Circle(xy = (xn, yn), radius = bcg_sky.shape[0] / 2, fill = False, ec = 'k', ls = '-', alpha = 0.75,)
		clust1 = Circle(xy = (xn, yn), radius = bcg_sky.shape[1] / 2, fill = False, ec = 'k', ls = '--', alpha = 0.75,)
		ax1.add_patch(clust0)
		ax1.add_patch(clust1)

		ax2.set_title('centered on BCGs - random center')
		th = ax2.imshow( (bcg_sky - M_rnd_sky) / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -3e-3, vmax = 3e-3,)
		cb = plt.colorbar(tg, ax = ax2, fraction = 0.035, pad = 0.01, label = 'pixel SB [nanomaggies / $pixel^2$]')
		cb.formatter.set_powerlimits((0,-3))
		clust0 = Circle(xy = (xn, yn), radius = bcg_sky.shape[0] / 2, fill = False, ec = 'k', ls = '-', alpha = 0.75,)
		clust1 = Circle(xy = (xn, yn), radius = bcg_sky.shape[1] / 2, fill = False, ec = 'k', ls = '--', alpha = 0.75,)
		ax2.add_patch(clust0)
		ax2.add_patch(clust1)

		ax3.set_title('SB profile')
		ax3.plot(bcg_sky_I_r, bcg_sky_I, color = 'r', ls = '-', alpha = 0.5, label = 'centered on BCGs')
		ax3.plot(M_sky_I_r, M_sky_I, color = 'b', ls = '-', alpha = 0.5, label = 'random center')
		ax3.plot(bcg_sky_I_r, bcg_sky_I - M_sky_I, color = 'g', ls = '-', alpha = 0.5, label = 'centered on BCGs - random center')

		ax3.axvline(x = pixel * (bcg_sky.shape[0] / 2), color = 'k', linestyle = '-', linewidth = 1, label = '$ H_{img} / 2$')
		ax3.axvline(x = pixel * (bcg_sky.shape[1] / 2), color = 'k', linestyle = '--', linewidth = 1, label = '$ W_{img} / 2$')

		ax3.set_ylim(-2e-4, 5e-4)
		ax3.set_xlim(1e1, 1e3)
		ax3.set_xlabel('R [arcsec]')
		ax3.set_ylabel('SB [nanomaggies / $arcsec^2$]')
		ax3.set_xscale('log')
		ax3.legend(loc = 2, frameon = False)
		ax3.grid(which = 'both', axis = 'both', alpha = 0.25)
		ax3.tick_params(axis = 'both', which = 'both', direction = 'in')
		ax3.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,-4),)

		#plt.savefig('cluster_sky_stack_test.png', dpi = 300)
		plt.savefig('random_sky_stack_test.png', dpi = 300)
		plt.close()

		with h5py.File('/home/xkchen/Downloads/test_imgs/clust_test-1000_res-sky-SB.h5', 'r') as f:
			clus_r = np.array(f['r'])
			clus_sky_I = np.array(f['bcg_sb'])
			clus_rnd_sky_I = np.array(f['rnd_sb'])

		plt.figure()
		ax = plt.subplot(111)
		ax.set_title('residual sky SB profile comparison')
		ax.plot(clus_r, clus_sky_I - clus_rnd_sky_I, color = 'r', ls = '-', alpha = 0.5, label = 'cluster')
		ax.plot(bcg_sky_I_r, bcg_sky_I - M_sky_I, color = 'b', ls = '-', alpha = 0.5, label = 'random')

		ax.axvline(x = pixel * (bcg_sky.shape[0] / 2), color = 'k', linestyle = '-', linewidth = 1, label = '$ H_{img} / 2$')
		ax.axvline(x = pixel * (bcg_sky.shape[1] / 2), color = 'k', linestyle = '--', linewidth = 1, label = '$ W_{img} / 2$')

		ax.set_ylim(-2e-4, 5e-4)
		ax.set_xlim(1e1, 1e3)
		ax.set_xlabel('R [arcsec]')
		ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
		ax.set_xscale('log')
		ax.legend(loc = 2, frameon = False)
		ax.grid(which = 'both', axis = 'both', alpha = 0.25)
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')
		ax.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,-4),)

		plt.savefig('residual_sky_SB_compare.png', dpi = 300)
		plt.close()

		raise

def jack_stack():

	bins, R_smal, R_max = 95, 1, 3.0e3 ## for sky ICL
	out_path = '/media/xkchen/My Passport/data/SDSS/tmp_stack/jack/'
	N_edg = 500
	N_bin = 30 #18

	kk = 0
	"""
	### individual sub-sample imgs & SB
	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('cluster sub sample SB')

	for nn in range( N_bin ):

		with h5py.File(out_path + 'clust_BCG-stack_sub-%d.h5' % (nn), 'r') as f:
			sub_img = np.array(f['a'])
		sub_block_m = grid_img(sub_img, 100, 100)[0]

		with h5py.File(out_path + 'clust_BCG-stack_pix-cont_sub-%d.h5' % (nn), 'r') as f:
			sub_cont = np.array(f['a'])
		'''
		plt.figure( figsize = (12, 6) )
		ax0 = plt.subplot(121)
		ax1 = plt.subplot(122)

		ax0.set_title('stacking img')
		tf = ax0.imshow(sub_img, origin = 'lower', cmap = 'seismic', vmin = -0.02, vmax = 0.02,)
		cb = plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = 'flux [nanomaggies]',)
		cb.formatter.set_powerlimits((0,0))

		ax1.set_title('100*100 grid blocks means')
		tg = ax1.imshow(sub_block_m, origin = 'lower', cmap = 'seismic', vmin = -4e-3, vmax = 4e-3,)
		cb = plt.colorbar(tg, ax = ax1, fraction = 0.035, pad = 0.01, label = 'flux [nanomaggies]',)
		cb.formatter.set_powerlimits((0,0))

		plt.tight_layout()
		plt.savefig('2D_img_%d.png' % nn, dpi = 300)
		plt.close()
		'''
		'''
		xn, yn = np.int(sub_img.shape[1] / 2), np.int(sub_img.shape[0] / 2)
		sb_arr, r_arr, sb_err_arr = SB_pro_z0_weit(sub_img, sub_cont, pixel, 1, 3000, xn, yn, np.int(1.22 * bins), kk)[4:]
		sb_arr, sb_err_arr = sb_arr / pixel**2, sb_err_arr / pixel**2
		'''
		with h5py.File(out_path + 'clust_BCG-stack_sub-%d_SB.h5' % (nn), 'r') as f:
			r_arr = np.array(f['r'])
			sb_arr = np.array(f['sb'])
			sb_err_arr = np.array(f['sb_err'])

		ax.plot(r_arr, sb_arr, ls = '-', color = mpl.cm.rainbow( nn / N_bin ), alpha = 0.5, label = '%d' % nn)

	ax.set_ylim(3e-4, 3e-2)
	ax.set_yscale('log')
	ax.set_xlim(1e1, 1e3)
	ax.set_xlabel('$ R[arcsec] $')
	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax.set_xscale('log')
	ax.legend(loc = 3, frameon = False, fontsize = 8, ncol = 5,)
	ax.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.subplots_adjust(left = 0.15, right = 0.95,)
	plt.savefig('sub-SB_compare_%d.png' % N_edg, dpi = 300)
	plt.close()

	'''
	## sub-jack sample imgs & SB
	for nn in range(N_bin):

		#with h5py.File(out_path + 'random_BCG-stack_N-edg-%d_jack-%d.h5' % ( N_edg, nn), 'r') as f:
		#with h5py.File(out_path + 'random_BCG-stack_jack-%d.h5' % ( nn ), 'r') as f:
		#with h5py.File(out_path + 'clust_BCG-stack_N-edg-%d_jack-%d.h5' % ( N_edg, nn), 'r') as f:
		with h5py.File(out_path + 'clust_BCG-stack_jack-%d.h5' % ( nn ), 'r') as f:
			stack_img = np.array(f['a'])

		#with h5py.File(out_path + 'random_BCG-stack_pix-cont_N-edg-%d_jack-%d.h5' % ( N_edg, nn), 'r') as f:
		#with h5py.File(out_path + 'random_BCG-stack_pix-cont_jack-%d.h5' % ( nn ), 'r') as f:
		#with h5py.File(out_path + 'clust_BCG-stack_pix-cont_N-edg-%d_jack-%d.h5' % ( N_edg, nn), 'r') as f:
		with h5py.File(out_path + 'clust_BCG-stack_pix-cont_jack-%d.h5' % ( nn ), 'r') as f:
			pix_cont = np.array(f['a'])

		xn, yn = np.int(stack_img.shape[1] / 2), np.int(stack_img.shape[0] / 2)
		weit_bcg_I, weit_bcg_I_r, weit_bcg_I_err = SB_pro_z0_weit(stack_img, pix_cont, pixel, 1, 3000, xn, yn, np.int(1.22 * bins), kk)[4:]
		weit_bcg_I, weit_bcg_I_err = weit_bcg_I / pixel**2, weit_bcg_I_err / pixel**2

		#with h5py.File(out_path +'random_BCG-stack_N-edg-%d_jack-%d_SB.h5' % ( N_edg, nn), 'w') as f:
		#with h5py.File(out_path + 'random_BCG-stack_jack-%d_SB.h5' % ( nn ), 'w') as f:
		#with h5py.File(out_path + 'clust_BCG-stack_N-edg-%d_jack-%d_SB.h5' % ( N_edg, nn), 'w') as f:
		with h5py.File(out_path + 'clust_BCG-stack_jack-%d_SB.h5' % ( nn ), 'w') as f:
			f['r'] = np.array(weit_bcg_I_r)
			f['sb'] = np.array(weit_bcg_I)
			f['sb_err'] = np.array(weit_bcg_I_err)
	'''
	## aveg jack SB profiles
	tmp_sb = []
	tmp_r = []

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('cluster sub-jack sample SB')

	for nn in range( N_bin ):

		#with h5py.File(out_path + 'random_BCG-stack_N-edg-%d_jack-%d_SB.h5' % ( N_edg, nn), 'r') as f:
		with h5py.File(out_path + 'random_BCG-stack_jack-%d_SB.h5' % ( nn ), 'r') as f:
		#with h5py.File(out_path + 'clust_BCG-stack_N-edg-%d_jack-%d_SB.h5' % ( N_edg, nn), 'r') as f:
		#with h5py.File(out_path + 'clust_BCG-stack_jack-%d_SB.h5' % ( nn ), 'r') as f:
			sb_arr = np.array(f['sb'])
			r_arr = np.array(f['r'])

		tmp_sb.append(sb_arr)
		tmp_r.append(r_arr)

		ax.plot(r_arr, sb_arr, ls = '-', color = mpl.cm.rainbow( nn / N_bin ), alpha = 0.5, label = '%d' % nn)

	ax.set_ylim(3e-3, 3e-2)
	ax.set_yscale('log')
	ax.set_xlim(1e1, 1e3)
	ax.set_xlabel('$ R[arcsec] $')
	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax.set_xscale('log')
	ax.legend(loc = 1, frameon = False, fontsize = 8, ncol = 5,)
	ax.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.subplots_adjust(left = 0.15, right = 0.95,)
	plt.savefig('sub_jack-SB_compare_%d.png' % N_edg, dpi = 300)
	plt.close()

	jk_R, jk_SB, jk_err = jack_SB(tmp_sb, tmp_r, kk, N_bin)[4:]

	#with h5py.File(out_path + 'random_BCG-stack_N-edg-%d_jack_SB.h5' % ( N_edg ), 'w') as f:
	#with h5py.File(out_path + 'random_BCG-stack_jack_SB.h5', 'w') as f:
	#with h5py.File(out_path + 'clust_BCG-stack_N-edg-%d_jack_SB.h5' % ( N_edg ), 'w') as f:
	#with h5py.File(out_path + 'clust_BCG-stack_jack_SB.h5', 'w') as f:

	##median adjust
	with h5py.File(out_path + 'random_BCG-stack_median_jack_SB.h5', 'w') as f:
	#with h5py.File(out_path + 'clust_BCG-stack_median_jack_SB.h5', 'w') as f:
		f['jack_r'] = np.array(jk_R)
		f['jack_sb'] = np.array(jk_SB)
		f['jack_err'] = np.array(jk_err)
	"""
	'''
	## average / median stack img
	d_file = out_path + 'random_BCG-stack_N-edg-%d' % (N_edg) + '_jack-%d.h5'
	out_file = out_path + 'random_BCG-stack_N-edg-%d_jack_Mean.h5' % ( N_edg )
	aveg_stack_img(N_bin, d_file, out_file)

	d_file = out_path + 'clust_BCG-stack_N-edg-%d' % (N_edg) + '_jack-%d.h5'
	out_file = out_path + 'clust_BCG-stack_N-edg-%d_jack_Mean.h5' % ( N_edg )
	aveg_stack_img(N_bin, d_file, out_file)

	d_file = out_path + 'random_BCG-stack_jack-%d.h5'
	out_file = out_path + 'random_BCG-stack_jack_Mean.h5'
	aveg_stack_img(N_bin, d_file, out_file)

	d_file = out_path + 'clust_BCG-stack_jack-%d.h5'
	out_file = out_path + 'clust_BCG-stack_jack_Mean.h5'
	aveg_stack_img(N_bin, d_file, out_file)

	d_file = out_path + 'random_BCG-stack_jack-%d.h5'
	out_file = out_path + 'random_BCG-stack_jack_Median.h5'
	medi_stack_img(N_bin, d_file, out_file)

	d_file = out_path + 'clust_BCG-stack_jack-%d.h5'
	out_file = out_path + 'clust_BCG-stack_jack_Median.h5'
	medi_stack_img(N_bin, d_file, out_file)

	d_file = out_path + 'random_BCG-stack_pix-cont_jack-%d.h5'
	out_file = out_path + 'random_BCG-stack_pix-cont_jack_Mean.h5'
	aveg_stack_img(N_bin, d_file, out_file)

	d_file = out_path + 'clust_BCG-stack_pix-cont_jack-%d.h5'
	out_file = out_path + 'clust_BCG-stack_pix-cont_jack_Mean.h5'
	aveg_stack_img(N_bin, d_file, out_file)
	'''
	## SB compare
	with h5py.File(out_path + 'random_BCG-stack_N-edg-%d_jack_Mean.h5' % ( N_edg ), 'r') as f:
		cut_rand_jk_m = np.array(f['a'])

	with h5py.File(out_path + 'clust_BCG-stack_N-edg-%d_jack_Mean.h5' % ( N_edg ), 'r') as f:
		cut_clus_jk_m = np.array(f['a'])

	with h5py.File(out_path + 'random_BCG-stack_jack_Mean.h5', 'r') as f:
		rand_m_jk = np.array(f['a'])

	with h5py.File(out_path + 'clust_BCG-stack_jack_Mean.h5', 'r') as f:
		clus_m_jk = np.array(f['a'])

	with h5py.File(out_path + 'random_BCG-stack_jack_Median.h5', 'r') as f:
		rand_medi_jk = np.array(f['a'])

	with h5py.File(out_path + 'clust_BCG-stack_jack_Median.h5', 'r') as f:
		clus_medi_jk = np.array(f['a'])

	cut_grd_img_clus = grid_img(cut_clus_jk_m, 100, 100)[0]
	cut_grd_img_rand = grid_img(cut_rand_jk_m, 100, 100)[0]
	grd_img_rand = grid_img(rand_m_jk, 100, 100)[0]
	grd_img_clus = grid_img(clus_m_jk, 100, 100)[0]
	grd_medi_clus = grid_img(clus_medi_jk, 100, 100)[0]
	grd_medi_rand = grid_img(rand_medi_jk, 100, 100)[0]

	fig = plt.figure( figsize = (19.84, 4.8) )
	ax0 = fig.add_axes([0.03, 0.09, 0.27, 0.85])
	ax1 = fig.add_axes([0.36, 0.09, 0.27, 0.85])
	ax2 = fig.add_axes([0.69, 0.09, 0.27, 0.85])

	#ax0.set_title('random img [mean of sub-jack img]')
	ax0.set_title('cluster img [mean of sub-jack img]')
	#tf = ax0.imshow(grd_img_rand / pixel**2, origin = 'lower', cmap = 'rainbow', vmin = 2e-3, vmax = 6e-3,)
	tf = ax0.imshow(grd_img_clus / pixel**2, origin = 'lower', cmap = 'rainbow', vmin = 2e-3, vmax = 6e-3,)
	cb = plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]')
	cb.formatter.set_powerlimits((0,0))

	#ax1.set_title('random img [median of sub-jack img]',)
	ax1.set_title('cluster img [median of sub-jack img]')
	#tf = ax1.imshow(grd_medi_rand / pixel**2, origin = 'lower', cmap = 'rainbow', vmin = 2e-3, vmax = 6e-3,)
	tf = ax1.imshow(grd_medi_clus / pixel**2, origin = 'lower', cmap = 'rainbow', vmin = 2e-3, vmax = 6e-3,)
	cb = plt.colorbar(tf, ax = ax1, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]')
	cb.formatter.set_powerlimits((0,0))

	ax2.set_title('[mean of sub-jack img] - [median of sub-jack img]',)
	#diff_img = grd_img_rand - grd_medi_rand
	diff_img = grd_img_clus - grd_medi_clus

	tf = ax2.imshow(diff_img / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -5e-5, vmax = 5e-5,)
	cb = plt.colorbar(tf, ax = ax2, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]')
	cb.formatter.set_powerlimits((0,0))

	plt.tight_layout()
	#plt.savefig('random_2D_grid_mean_compare_%d.png' % ( N_edg ), dpi = 300)
	plt.savefig('clust_2D_grid_mean_compare_%d.png' % ( N_edg ), dpi = 300)
	plt.close()
	raise

	### just stack the imgs
	#with h5py.File(out_path + 'clust_tot_BCG-stack_correct.h5', 'r') as f:
	#with h5py.File(out_path + 'random_tot_BCG-stack_correct.h5', 'r') as f:

	#with h5py.File(out_path + 'clust_BCG-stack_jack_Median.h5', 'r') as f:
	#with h5py.File(out_path + 'random_BCG-stack_jack_Median.h5', 'r') as f:
	with h5py.File(out_path + 'clust_BCG-stack_jack_Mean.h5', 'r') as f:
	#with h5py.File(out_path + 'random_BCG-stack_jack_Mean.h5', 'r') as f:
		D_stack_img = np.array(f['a'])

	#with h5py.File(out_path + 'clust_tot_BCG-stack_pix-cont_correct.h5', 'r') as f:
	#with h5py.File(out_path + 'random_tot_BCG-stack_pix-cont_correct.h5', 'r') as f:

	with h5py.File(out_path + 'clust_BCG-stack_pix-cont_jack_Mean.h5', 'r') as f:
	#with h5py.File(out_path + 'random_BCG-stack_pix-cont_jack_Mean.h5', 'r') as f:
		D_pix_cont = np.array(f['a'])

	xn, yn = np.int(D_stack_img.shape[1] / 2), np.int(D_stack_img.shape[0] / 2)
	weit_bcg_I, weit_bcg_I_r, weit_bcg_I_err = SB_pro_z0_weit(D_stack_img, D_pix_cont, pixel, 1, 3000, xn, yn, np.int(1.22 * bins), kk)[4:]
	weit_bcg_I, weit_bcg_I_err = weit_bcg_I / pixel**2, weit_bcg_I_err / pixel**2

	#with h5py.File(out_path + 'clust_tot_BCG-stack_correct_SB.h5', 'w') as f:
	#with h5py.File(out_path + 'random_tot_BCG-stack_correct_SB.h5', 'w') as f:
	#with h5py.File(out_path + 'clust_tot_BCG-stack_median-jk-img_SB.h5', 'w') as f:
	#with h5py.File(out_path + 'random_tot_BCG-stack_median-jk-img_SB.h5', 'w') as f:
	with h5py.File(out_path + 'clust_tot_BCG-stack_mean-jk-img_SB.h5', 'w') as f:
	#with h5py.File(out_path + 'random_tot_BCG-stack_mean-jk-img_SB.h5', 'w') as f:	
		f['r'] = np.array(weit_bcg_I_r)
		f['sb'] = np.array(weit_bcg_I)
		f['sb_err'] = np.array(weit_bcg_I_err)

	### SB compare
	with h5py.File(out_path + 'random_BCG-stack_N-edg-%d_jack_SB.h5' % ( N_edg ), 'r') as f:
		cut_rand_jk_sb = np.array(f['jack_sb'])
		cut_rand_jk_r = np.array(f['jack_r'])
		cut_rand_jk_err = np.array(f['jack_err'])

	with h5py.File(out_path + 'random_BCG-stack_jack_SB.h5', 'r') as f:
		rand_jk_sb = np.array(f['jack_sb'])
		rand_jk_r = np.array(f['jack_r'])
		rand_jk_err = np.array(f['jack_err'])

	with h5py.File(out_path + 'clust_BCG-stack_N-edg-%d_jack_SB.h5' % ( N_edg ), 'r') as f:
		cut_clus_jk_sb = np.array(f['jack_sb'])
		cut_clus_jk_r = np.array(f['jack_r'])
		cut_clus_jk_err = np.array(f['jack_err'])

	with h5py.File(out_path + 'clust_BCG-stack_jack_SB.h5', 'r') as f:
		clus_jk_sb = np.array(f['jack_sb'])
		clus_jk_r = np.array(f['jack_r'])
		clus_jk_err = np.array(f['jack_err'])
	### median jack SB
	with h5py.File(out_path + 'random_BCG-stack_median_jack_SB.h5', 'r') as f:
		rand_mid_jk_sb = np.array(f['jack_sb'])
		rand_mid_jk_r = np.array(f['jack_r'])
		rand_mid_jk_err = np.array(f['jack_err'])

	with h5py.File(out_path + 'clust_BCG-stack_median_jack_SB.h5', 'r') as f:
		clus_mid_jk_sb = np.array(f['jack_sb'])
		clus_mid_jk_r = np.array(f['jack_r'])
		clus_mid_jk_err = np.array(f['jack_err'])

	with h5py.File(out_path + 'clust_tot_BCG-stack_median-jk-img_SB.h5', 'r') as f:
		clus_mid_jk_img_sb = np.array(f['sb'])
		clus_mid_jk_img_r = np.array(f['r'])
		clus_mid_jk_img_err = np.array(f['sb_err'])

	with h5py.File(out_path + 'random_tot_BCG-stack_median-jk-img_SB.h5', 'r') as f:
		rand_mid_jk_img_sb = np.array(f['sb'])
		rand_mid_jk_img_r = np.array(f['r'])
		rand_mid_jk_img_err = np.array(f['sb_err'])

	with h5py.File(out_path + 'clust_tot_BCG-stack_mean-jk-img_SB.h5', 'r') as f:
		clus_men_jk_img_sb = np.array(f['sb'])
		clus_men_jk_img_r = np.array(f['r'])
		clus_men_jk_img_err = np.array(f['sb_err'])

	with h5py.File(out_path + 'random_tot_BCG-stack_mean-jk-img_SB.h5', 'r') as f:
		rand_men_jk_img_sb = np.array(f['sb'])
		rand_men_jk_img_r = np.array(f['r'])
		rand_men_jk_img_err = np.array(f['sb_err'])

	### just stacking the images
	with h5py.File(out_path + 'clust_tot_BCG-stack_correct_SB.h5', 'r') as f:
		clus_I = np.array(f['sb'])
		clus_I_r = np.array(f['r'])
		clus_I_err = np.array(f['sb_err'])

	with h5py.File(out_path + 'random_tot_BCG-stack_correct_SB.h5', 'r') as f:
		rand_I = np.array(f['sb'])
		rand_I_r = np.array(f['r'])
		rand_I_err = np.array(f['sb_err'])

	with h5py.File(out_path + 'clust_tot_BCG-stack_SB_tmp.h5', 'r') as f:
		tmp_sb = np.array(f['sb'])
		tmp_r = np.array(f['r'])
		tmp_err = np.array(f['sb_err'])

	plt.figure()
	gs = gridspec.GridSpec(2,1, height_ratios = [4,1])
	ax = plt.subplot(gs[0])
	ax1 = plt.subplot(gs[1])

	#ax = plt.subplot(111)
	'''
	ax.plot(cut_clus_jk_r, cut_clus_jk_sb, color = 'm', ls = '-', alpha = 0.5, label = 'cluster img [cut %d edgs pixels]' % N_edg,)
	ax.fill_between(cut_clus_jk_r, y1 = cut_clus_jk_sb - cut_clus_jk_err, y2 = cut_clus_jk_sb + cut_clus_jk_err, color = 'm', alpha = 0.45,)
	ax.plot(cut_rand_jk_r, cut_rand_jk_sb, color = 'c', ls = '-', alpha = 0.5, label = 'random img [cut %d edgs pixels]' % N_edg,)
	ax.fill_between(cut_rand_jk_r, y1 = cut_rand_jk_sb - cut_rand_jk_err, y2 = cut_rand_jk_sb + cut_rand_jk_err, color = 'c', alpha = 0.45,)
	'''
	ax.plot(clus_jk_r, clus_jk_sb, color = 'r', ls = '-', alpha = 0.7, label = 'cluster [mean of sub-jack SB]',)
	ax.fill_between(clus_jk_r, y1 = clus_jk_sb - clus_jk_err, y2 = clus_jk_sb + clus_jk_err, color = 'r', alpha = 0.3,)

	#ax.errorbar(clus_men_jk_img_r, clus_men_jk_img_sb, yerr = clus_men_jk_img_err, xerr = None, color = 'm', marker = 'None', ls = '-', 
	#	ecolor = 'm', alpha = 0.5, label = 'cluster [mean of sub-jack imgs]')
	#ax.errorbar(clus_mid_jk_img_r, clus_mid_jk_img_sb, yerr = clus_mid_jk_img_err, xerr = None, color = 'm', marker = 'None', ls = '-', 
	#	ecolor = 'm', alpha = 0.5, label = 'cluster [median of sub-jack imgs]')
	#ax.errorbar(clus_mid_jk_r, clus_mid_jk_sb, yerr = clus_mid_jk_err, xerr = None, color = 'm', marker = 'None', ls = '-', ecolor = 'm',
	#	alpha = 0.5, label = 'cluster [median of sub-jack SB]')
	ax.errorbar(clus_I_r, clus_I, yerr = clus_I_err, xerr = None, color = 'm', marker = 'None', ls = '-', ecolor = 'm',
		alpha = 0.5, label = 'cluster [Direct stacking imgs]')

	ax.plot(rand_jk_r, rand_jk_sb, color = 'b', ls = '-', alpha = 0.7, label = 'random [mean of sub-jack SB]',)
	ax.fill_between(rand_jk_r, y1 = rand_jk_sb - rand_jk_err, y2 = rand_jk_sb + rand_jk_err, color = 'b', alpha = 0.3,)

	#ax.errorbar(rand_men_jk_img_r, rand_men_jk_img_sb, yerr = rand_men_jk_img_err, xerr = None, color = 'c', marker = 'None', ls = '-', 
	#	ecolor = 'c', alpha = 0.5, label = 'random [mean of sub-jakc imgs]')
	#ax.errorbar(rand_mid_jk_img_r, rand_mid_jk_img_sb, yerr = rand_mid_jk_img_err, xerr = None, color = 'c', marker = 'None', ls = '-', 
	#	ecolor = 'c', alpha = 0.5, label = 'random [median of sub-jakc imgs]')
	#ax.errorbar(rand_mid_jk_r, rand_mid_jk_sb, yerr = rand_mid_jk_err, xerr = None, color = 'c', marker = 'None', ls = '-', ecolor = 'c',
	#	alpha = 0.5, label = 'random [median of sub-jakc SB]')
	ax.errorbar(rand_I_r, rand_I, yerr = rand_I_err, xerr = None, color = 'c', marker = 'None', ls = '-', ecolor = 'c',
		alpha = 0.5, label = 'random [Direct stacking imgs]')

	#ax.plot(rand_jk_r, rand_jk_sb - 5e-4, color = 'g', ls = '--', alpha = 0.5,)
	ax.axvline(x = pixel * (clus_m_jk.shape[0] / 2), color = 'k', linestyle = '-', linewidth = 1, label = '$ H_{img} / 2$', alpha = 0.5,)
	ax.axvline(x = pixel * (clus_m_jk.shape[1] / 2), color = 'k', linestyle = '--', linewidth = 1, label = '$ W_{img} / 2$', alpha = 0.5,)
	ax.set_ylim(3e-3, 3e-2)
	ax.set_yscale('log')
	ax.set_xlim(1e1, 1e3)
	ax.set_xlabel('$ R[arcsec] $')
	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax.set_xscale('log')
	ax.legend(loc = 1, frameon = False, fontsize = 8)
	ax.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')

	ax1.plot(clus_I_r, clus_jk_sb / clus_I, ls = '-', color = 'r', alpha = 0.5, label = 'cluster')
	ax1.plot(rand_I_r, rand_jk_sb / rand_I, ls = '-', color = 'b', alpha = 0.5, label = 'random')

	ax1.set_ylim(0.99, 1.01)
	ax1.set_xlim(ax.get_xlim() )
	ax1.set_ylabel('$SB_{j} / SB_{D}$', fontsize = 8.5,)
	#ax1.set_ylabel('$ \\frac{SB_{j-mean}} {SB_{j-median}} \\qquad$',)
	#ax1.set_ylabel('$ \\frac{SB_{j-mean}} {SB_{j-median}} \\qquad $',)
	#ax1.set_ylabel('$ \\frac{SB_{mean \, of \, profile}} {SB_{mean \, of \, image}} \\qquad $',)

	ax1.set_xscale('log')
	ax1.set_xlabel('$ R[arcsec] $')
	ax1.legend(loc = 3, fontsize = 8.5,)
	ax1.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax1.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax.set_xticks([])

	plt.subplots_adjust(left = 0.15, right = 0.95, hspace = 0.05)
	plt.savefig('BCG_stack_jk-Dstack_SB_compare.png', dpi = 300)
	#plt.savefig('BCG_stack_jk-jk_SB_compare.png', dpi = 300)
	#plt.savefig('BCG_stack_jk-medi-jk-img_SB_compare.png', dpi = 300)
	#plt.savefig('BCG_stack_jk-mean-jk-img_SB_compare.png', dpi = 300)

	plt.close()

	raise

def main():

	#re_mask()
	#sky_stack()
	jack_stack()

if __name__ == "__main__":
	main()
