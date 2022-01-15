import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc

from astropy.coordinates import SkyCoord
from astropy import cosmology as apcy
from scipy.optimize import curve_fit
from scipy import signal
from scipy import interpolate as interp
from scipy.ndimage import gaussian_filter

from light_measure import jack_SB_func
from fig_out_module import arr_jack_func, arr_slope_func
from fig_out_module import cc_grid_img, grid_img

from BCG_SB_pro_stack import BCG_SB_pros_func
from img_pre_selection import cat_match_func, get_mu_sigma
from fig_out_module import zref_BCG_pos_func
from img_cat_param_match import match_func
from img_stack import stack_func

from img_mask import source_detect_func, mask_func
from img_mask_adjust import adjust_mask_func
##### constant
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
pc2cm = U.pc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)

##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

## check for special sample

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

pixel = 0.396
band = ['r', 'g', 'i']

### BCG-star-Mass sample
def deV_fit(r, I_e, r_e,):
	"""
	SB profile : Mo. galaxy evolution and evolution, Chapter 2, eq. 2.23
	"""
	ndex = 4
	belta_n = 2 * ndex - 0.324
	f_n = - belta_n * ( r / r_e)**(1 / ndex) + belta_n
	I_r = I_e * np.exp( f_n )

	return I_r

def sersic_func(r, Ie, re, ndex,):

	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

def power_fit(r, I_0, alpha):

	I_r = I_0 * r**( alpha )

	return I_r

def SB_fit(r, Ie0, re0, ndex0, Ie1, re1, ndex1, Ie2, re2, ndex2,):

	mu0 = sersic_func( r, Ie0, re0, ndex0,)
	mu1 = sersic_func( r, Ie1, re1, ndex1,)
	mu2 = sersic_func( r, Ie2, re2, ndex2,)

	mu = mu0 + mu1 + mu2

	return mu

def BG_sub_SB(N_sample, jk_sub_sb, sb_out_put, band_str,):

	tmp_r, tmp_sb = [], []

	for kk in range( N_sample ):

		with h5py.File( jk_sub_sb % kk, 'r') as f:
			c_r_arr = np.array(f['r'])
			c_sb_arr = np.array(f['sb'])
			c_sb_err = np.array(f['sb_err'])
			npix = np.array(f['npix'])

		id_Nul = npix < 1
		c_r_arr[id_Nul ] = np.nan
		c_sb_arr[id_Nul ] = np.nan
		c_sb_err[id_Nul ] = np.nan

		# minus the minimum of SB in range(1Mpc ~ 2Mpc)
		idr = ( c_r_arr >= 1e3 ) & ( c_r_arr <= 2e3 )
		idsb = np.nanmin(c_sb_arr[idr])
		devi_sb = c_sb_arr - idsb

		tmp_r.append( c_r_arr )
		tmp_sb.append( devi_sb )

	tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(tmp_sb, tmp_r, band_str, N_sample,)[4:]

	## save BG-subtracted pros
	with h5py.File( sb_out_put, 'w') as f:
		f['r'] = np.array(tt_jk_R)
		f['sb'] = np.array(tt_jk_SB)
		f['sb_err'] = np.array(tt_jk_err)

	return

def new_BG_sub_sb(N_sample, jk_sub_sb, sb_out_put, band_str, BG_file,):

	tmp_r, tmp_sb = [], []

	bg_cat = pds.read_csv(BG_file)
	r_bg = np.array( bg_cat['R_kpc'] )
	bg_sb = np.array( bg_cat['BG_sb'] )
	r_lim0, r_lim1 = np.min(r_bg), np.max(r_bg)

	intep_func = interp.interp1d(r_bg, bg_sb, kind = 'cubic',)

	for kk in range( N_sample ):

		with h5py.File( jk_sub_sb % kk, 'r') as f:
			c_r_arr = np.array(f['r'])
			c_sb_arr = np.array(f['sb'])
			c_sb_err = np.array(f['sb_err'])
			npix = np.array(f['npix'])

		id_Nul = npix < 1
		c_r_arr[ id_Nul ] = np.nan
		c_sb_arr[ id_Nul ] = np.nan
		c_sb_err[ id_Nul ] = np.nan

		## interpl1 get the BG in c_r_arr region (?????)
		idr = ( c_r_arr >= r_lim0 ) & ( c_r_arr <= r_lim1 )
		lim_r = c_r_arr[ idr ]
		lim_BG = intep_func( lim_r )

		devi_sb = c_sb_arr[ idr ] - lim_BG

		tmp_r.append( lim_r )
		tmp_sb.append( devi_sb )

	tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(tmp_sb, tmp_r, band_str, N_sample,)[4:]

	## save BG-subtracted pros
	with h5py.File( sb_out_put, 'w') as f:
		f['r'] = np.array(tt_jk_R)
		f['sb'] = np.array(tt_jk_SB)
		f['sb_err'] = np.array(tt_jk_err)

	return

load = '/home/xkchen/fig_tmp/'

id_cen = 0
n_rbins = 100
N_bin = 30
z_ref = 0.25

pre_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$',]

BG_file = '/home/xkchen/project/ICL/BG_pros.csv'

for ll in range( 1 ):

	for mm in range( 2 ):

		jk_sub_sb = load + 'stack/' + pre_lis[mm] + '_%s-band' % band[ll] + '_jack-sub-%d_SB-pro_z-ref_with-selection.h5'
		sb_out_put = '/home/xkchen/project/tmp/' + pre_lis[mm] + '_%s-band_Mean-jack_BG-sub_SB-pro_z-ref_with-selection.h5' % band[ll]
		#BG_sub_SB(N_bin, jk_sub_sb, sb_out_put, band[ll],)
		new_BG_sub_sb(N_bin, jk_sub_sb, sb_out_put, band[ll], BG_file,)

tmp_r, tmp_sb, tmp_err = [], [], []

for kk in range( 1 ):

	for ll in range( 2 ):

		with h5py.File(load + 'stack/' + pre_lis[ll] + '_%s-band_Mean_jack_img_z-ref_with-selection.h5' % band[kk], 'r') as f:
			tt_img = np.array(f['a'])

		xn, yn = np.int(tt_img.shape[1] / 2), np.int(tt_img.shape[0] / 2)

		id_nan = np.isnan(tt_img)
		idvx = id_nan == False
		idy, idx = np.where(idvx == True)
		x_low, x_up = np.min(idx), np.max(idx)
		y_low, y_up = np.min(idy), np.max(idy)

		dpt_img = tt_img[y_low: y_up+1, x_low: x_up + 1]
		img_block = cc_grid_img(dpt_img, 100, 100,)[0]

		with h5py.File(load + 'stack/' + pre_lis[ll] + '_%s-band_Mean_jack_SB-pro_z-ref_with-selection.h5' % band[kk], 'r') as f:
			c_r_arr = np.array(f['r'])
			c_sb_arr = np.array(f['sb'])
			c_sb_err = np.array(f['sb_err'])

		id_Nul = c_r_arr > 0
		c_r_arr = c_r_arr[id_Nul]
		c_sb_arr = c_sb_arr[id_Nul]
		c_sb_err = c_sb_err[id_Nul]

		tmp_r.append( c_r_arr )
		tmp_sb.append( c_sb_arr )
		tmp_err.append( c_sb_err )

tmp_sub_r, tmp_sub_sb, tmp_sub_err = [], [], []

for kk in range( 1 ):

	for mm in range( 2 ):

		with h5py.File('/home/xkchen/project/tmp/' + pre_lis[mm] + '_%s-band_Mean-jack_BG-sub_SB-pro_z-ref_with-selection.h5' % band[kk], 'r') as f:
			tt_jk_r = np.array( f['r'] )
			tt_jk_sb = np.array( f['sb'] )
			tt_jk_err = np.array( f['sb_err'] )

		tmp_sub_r.append( tt_jk_r )
		tmp_sub_sb.append( tt_jk_sb )
		tmp_sub_err.append( tt_jk_err )

line_name = ['r low $M_{\\ast}$', 'r high $M_{\\ast}$',
			'g low $M_{\\ast}$', 'g high $M_{\\ast}$', 
			'i low $M_{\\ast}$', 'i high $M_{\\ast}$']

line_c = ['r', 'r', 'g', 'g', 'b', 'b']

line_s = ['--', '-', '--', '-', '--', '-']

bg_cat = pds.read_csv(BG_file)
r_bg = np.array( bg_cat['R_kpc'] )
bg_sb = np.array( bg_cat['BG_sb'] )

plt.figure()
ax = plt.subplot(111)

color_s = ['b', 'r']

for pp in range( 2 ):

	ax.plot(tmp_r[pp], tmp_sb[pp], ls = '-', color = color_s[pp], alpha = 0.5, label = line_name[pp],)
	ax.fill_between(tmp_r[pp], y1 = tmp_sb[pp] - tmp_err[pp], y2 = tmp_sb[pp] + tmp_err[pp], color = color_s[pp], alpha = 0.15,)

	ax.plot(tmp_sub_r[pp], tmp_sub_sb[pp], ls = '--', color = color_s[pp], alpha = 0.5,)
	ax.fill_between(tmp_sub_r[pp], y1 = tmp_sub_sb[pp] - tmp_sub_err[pp], y2 = tmp_sub_sb[pp] + tmp_sub_err[pp],
		color = color_s[pp], alpha = 0.15,)

ax.plot(r_bg, bg_sb, ls = '-', color = 'k', alpha = 0.5, label = 'BackGround',)

ax.set_ylim(1e-5, 1e0)
ax.set_yscale('log')
ax.set_xlim(1e1, 4e3)
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 3, frameon = False, fontsize = 8.0,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25,)
ax.tick_params(axis = 'both', which = 'both', direction = 'in',)
ax.set_xlabel('R [kpc]')

plt.savefig(load + 'figs/BCG-star-Mass_gri-SB_pros.png', dpi = 300)
plt.close()

'''
plt.figure()
ax = plt.subplot(111)

#for pp in (0, 2, 4):
for pp in (1, 3, 5):
	ax.plot(tmp_r[pp], tmp_sb[pp], ls = '-', color = line_c[pp], alpha = 0.8, label = line_name[pp],)
	ax.fill_between(tmp_r[pp], y1 = tmp_sb[pp] - tmp_err[pp], y2 = tmp_sb[pp] + tmp_err[pp], color = line_c[pp], alpha = 0.2,)

	ax.plot(tmp_sub_r[pp], tmp_sub_sb[pp], ls = '--', color = line_c[pp], alpha = 0.8,)
	ax.fill_between(tmp_sub_r[pp], y1 = tmp_sub_sb[pp] - tmp_sub_err[pp], y2 = tmp_sub_sb[pp] + tmp_sub_err[pp], color = line_c[pp], alpha = 0.2,)

ax.set_ylim(1e-5, 2e0)
ax.set_yscale('log')
ax.set_xlim(1e1, 4e3)
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 3, frameon = False, fontsize = 8.0,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25,)
ax.tick_params(axis = 'both', which = 'both', direction = 'in',)
ax.set_xlabel('R [kpc]')

if pp % 2 == 0:
	plt.savefig(load + 'figs/low_BCG-star-Mass_gri-SB_pros.png', dpi = 300)
if pp % 2 == 1:
	plt.savefig(load + 'figs/high_BCG-star-Mass_gri-SB_pros.png', dpi = 300)
plt.close()


plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2])
ax = plt.subplot(gs[0])
bx = plt.subplot(gs[1])

for pp in (0, 2, 4):

	ax.plot(tmp_sub_r[pp], tmp_sub_sb[pp], ls = line_s[pp], color = line_c[pp], alpha = 0.5, label = line_name[pp],)
	ax.fill_between(tmp_sub_r[pp], y1 = tmp_sub_sb[pp] - tmp_sub_err[pp], y2 = tmp_sub_sb[pp] + tmp_sub_err[pp], 
		color = line_c[pp], alpha = 0.15,)

	ax.plot(tmp_sub_r[ pp + 1 ], tmp_sub_sb[ pp + 1 ], ls = line_s[ pp + 1 ], color = line_c[ pp + 1 ], alpha = 0.5, label = line_name[ pp + 1 ],)
	ax.fill_between(tmp_sub_r[ pp + 1 ], y1 = tmp_sub_sb[ pp + 1 ] - tmp_sub_err[ pp + 1 ], y2 = tmp_sub_sb[ pp + 1 ] + tmp_sub_err[ pp + 1 ], 
		color = line_c[ pp + 1 ], alpha = 0.15,)

	idnn = np.isnan( tmp_sub_sb[ pp + 1 ] )
	cc_sb = tmp_sub_sb[ pp + 1 ][idnn == False]
	cc_r = tmp_sub_r[ pp + 1 ][idnn == False]
	interp_F = interp.interp1d( cc_r, cc_sb, kind = 'cubic',)

	idmx = ( tmp_sub_r[ pp ] >= cc_r.min() ) & ( tmp_sub_sb[ pp ] <= cc_r.max() )
	cop_r = tmp_sub_r[ pp ][idmx]
	cop_hi_sb = interp_F( cop_r )

	bx.plot(tmp_sub_r[pp], tmp_sub_sb[pp] / tmp_sub_sb[pp], ls = line_s[pp], color = line_c[pp], alpha = 0.5,)
	bx.fill_between(tmp_sub_r[pp], y1 = (tmp_sub_sb[pp] - tmp_sub_err[pp]) / tmp_sub_sb[pp], y2 = (tmp_sub_sb[pp] + tmp_sub_err[pp]) / tmp_sub_sb[pp], 
		color = line_c[pp], alpha = 0.15,)

	bx.plot(cop_r, cop_hi_sb / tmp_sub_sb[ pp ][idmx], ls = line_s[pp + 1], color = line_c[pp + 1], alpha = 0.5,)

ax.set_ylim(1e-4, 2e0)
ax.set_yscale('log')
ax.set_xlim(1e1, 4e3)
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 1, frameon = False, fontsize = 8.0,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in',)

bx.set_ylim(0, 2)
bx.set_xlim( ax.get_xlim() )
bx.set_xscale( 'log' )
bx.set_xlabel('R[kpc]')
bx.set_ylabel('$ SB / SB_{low \; M_{\\ast} } $')
bx.grid(which = 'both', axis = 'both', alpha = 0.25)
bx.tick_params(axis = 'both', which = 'both', direction = 'in',)
ax.set_xticklabels( labels = [], fontsize = 0.005,)

plt.subplots_adjust( hspace = 0.05,)
plt.savefig(load + 'figs/BCG-star-Mass_gri-SB_pros.png', dpi = 300)
plt.close()
'''


### combine sample (low BCG-star-Mass + high BCG-star-Mass)
tmp_r, tmp_sb, tmp_err = [], [], []
tmp_sub_r, tmp_sub_sb, tmp_sub_err = [], [], []

for ll in range( 1 ):

	jk_sub_sb = load + 'stack/com-BCG-star-Mass_%s-band' % band[ll] + '_jack-sub-%d_SB-pro_z-ref_with-selection.h5'
	sb_out_put = '/home/xkchen/project/tmp/' + 'com-BCG-star-Mass_%s-band_Mean-jack_BG-sub_SB-pro_z-ref_with-selection.h5' % band[ll]
	#BG_sub_SB(N_bin, jk_sub_sb, sb_out_put, band[ll],)
	new_BG_sub_sb(N_bin, jk_sub_sb, sb_out_put, band[ll], BG_file,)

for ll in range( 1 ):

	## 2D imgs and SB
	with h5py.File(load + 'stack/com-BCG-star-Mass_%s-band_Mean_jack_img_z-ref_with-selection.h5' % band[ll], 'r') as f:
		tt_img = np.array(f['a'])

	xn, yn = np.int(tt_img.shape[1] / 2), np.int(tt_img.shape[0] / 2)

	id_nan = np.isnan(tt_img)
	idvx = id_nan == False
	idy, idx = np.where(idvx == True)
	x_low, x_up = np.min(idx), np.max(idx)
	y_low, y_up = np.min(idy), np.max(idy)

	dpt_img = tt_img[y_low: y_up+1, x_low: x_up + 1]
	img_block = cc_grid_img(dpt_img, 100, 100,)[0]

	with h5py.File(load + 'stack/com-BCG-star-Mass_%s-band_Mean_jack_SB-pro_z-ref_with-selection.h5' % band[ll], 'r') as f:
		c_r_arr = np.array(f['r'])
		c_sb_arr = np.array(f['sb'])
		c_sb_err = np.array(f['sb_err'])

	id_Nul = c_sb_arr > 0
	c_r_arr = c_r_arr[id_Nul]
	c_sb_arr = c_sb_arr[id_Nul]
	c_sb_err = c_sb_err[id_Nul]

	tmp_r.append( c_r_arr )
	tmp_sb.append( c_sb_arr )
	tmp_err.append( c_sb_err )
'''
	cen_L = 1260 # pixels(~ 2 Mpc/h)
	dnoise = 45
	cen_img = tt_img[yn - cen_L: yn + cen_L, xn - cen_L:xn + cen_L] / pixel**2
	kernl_img = gaussian_filter(cen_img, sigma = dnoise,  mode = 'nearest')
	f_lels = np.array([3e-3, 4e-3, 5e-3, 1e-2, 2e-2, 3e-2, 2e-1, 3e-1])

	D_ref = Test_model.angular_diameter_distance( z_ref ).value
	L_pix = pixel * D_ref * 1e3 / rad2asec

	R_100 = 100 / L_pix
	R_500 = 500 / L_pix
	R_1000 = 1000 / L_pix

	fig = plt.figure()
	ax0 = plt.subplot(111)
	ax0.set_title( '%s band' % band[ll] )

	ax0.imshow(cen_img, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm(),)
	tf = ax0.contour(kernl_img, origin = 'lower', cmap = 'rainbow', levels = f_lels, alpha = 0.5,)
	plt.clabel(tf, inline = True, fontsize = 6.5, fmt = '%.5f',)

	clust = Circle(xy = (cen_L, cen_L), radius = R_100, fill = False, ec = 'r', ls = '--', linewidth = 1.25, alpha = 0.5,)
	ax0.add_patch(clust)
	clust = Circle(xy = (cen_L, cen_L), radius = R_500, fill = False, ec = 'r', ls = '-.', linewidth = 1.25, alpha = 0.5,)
	ax0.add_patch(clust)
	clust = Circle(xy = (cen_L, cen_L), radius = R_1000, fill = False, ec = 'r', ls = '-', linewidth = 1.25, alpha = 0.5,)
	ax0.add_patch(clust)

	plt.savefig(load + 'figs/com-BCG-star-Mass_%s-band_BCG_region_check.png' % band[ll], dpi = 300)
	plt.close()
'''
for ll in range( 1 ):

	with h5py.File('/home/xkchen/project/tmp/' + 'com-BCG-star-Mass_%s-band_Mean-jack_BG-sub_SB-pro_z-ref_with-selection.h5' % band[ll], 'r') as f:
		tt_jk_r = np.array( f['r'] )
		tt_jk_sb = np.array( f['sb'] )
		tt_jk_err = np.array( f['sb_err'] )

	tmp_sub_r.append( tt_jk_r )
	tmp_sub_sb.append( tt_jk_sb )
	tmp_sub_err.append( tt_jk_err )

com_line_c = ['r', 'g', 'b']

plt.figure()
ax = plt.subplot(111)

for pp in range( 1 ):

	ax.plot(tmp_r[pp], tmp_sb[pp], ls = '-', color = com_line_c[pp], alpha = 0.8, label = '%s band' % band[pp],)
	ax.fill_between(tmp_r[pp], y1 = tmp_sb[pp] - tmp_err[pp], y2 = tmp_sb[pp] + tmp_err[pp], color = com_line_c[pp], alpha = 0.2,)

	ax.plot(tmp_sub_r[pp], tmp_sub_sb[pp], ls = '--', color = com_line_c[pp], alpha = 0.8,)
	ax.fill_between(tmp_sub_r[pp], y1 = tmp_sub_sb[pp] - tmp_sub_err[pp], y2 = tmp_sub_sb[pp] + tmp_sub_err[pp], color = com_line_c[pp], alpha = 0.2,)

ax.set_ylim(1e-5, 1e0)
ax.set_yscale('log')
ax.set_xlim(1e1, 4e3)
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 3, frameon = False, fontsize = 8.0,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in',)
ax.set_xlabel('R [kpc]')

plt.savefig(load + 'figs/com-BCG-star-Mass_gri-SB_pros.png', dpi = 300)
plt.close()


"""
### fit SB profile

for ll in range( 3 ):

	#SB_tt = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/%s_band_BCG_ICL.csv' % band[ll],)
	SB_tt = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/%s_band_sub_unmask.csv' % band[ll],)
	R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
	R_obs = R_obs**4
	flux_obs = 10**( (22.5 - SB_obs + 2.5 * np.log10(pixel**2) ) / 2.5 ) / pixel**2

	if ll < 2:
		Z_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/Zhang_SB/Zhang_18_%s-band_SB.txt' % band[ll],)
		Z_r, Z_SB = np.array(Z_dat['R_kpc']), np.array(Z_dat['SB_mag-arcsec2'])
		Z_fdens = 10**( (22.5 - Z_SB + 2.5 * np.log10(pixel**2) ) / 2.5 ) / pixel**2

	with h5py.File('/home/xkchen/mywork/ICL/com-BCG-star-Mass_%s-band_Mean-jack_BG-sub_SB-pro_z-ref_with-selection.h5' % band[ll], 'r') as f:
		obs_r = np.array( f['r'] )
		obs_fdens = np.array( f['sb'] )
		obs_ferr = np.array( f['sb_err'] )

	#obs_r = tmp_sub_r[ll]
	#obs_fdens = tmp_sub_sb[pp]
	#obs_ferr = tmp_sub_err[pp]

	idnn = np.isnan( obs_fdens )
	obs_r = obs_r[ idnn == False ]
	obs_fdens = obs_fdens[ idnn == False ]
	obs_ferr = obs_ferr[ idnn == False ]

	idmx = obs_r <= 1.2e3 # ~1 Mpc / h
	obs_r = obs_r[ idmx ]
	obs_fdens = obs_fdens[ idmx ]
	obs_ferr = obs_ferr[ idmx ]

	# <~ 1 Mpc
	idux = obs_r <= 1000
	idR = obs_r[idux]
	idSB = obs_fdens[idux]
	idSB_err = obs_ferr[idux]

	# r, i band
	if ll != 1:
		re0, mu_e0, ndex0 = 4, 5e0, 1.5
		re1, mu_e1, ndex1 = 50, 5e-1, 3
		re2, mu_e2, ndex2 = 500, 3e-3, 2

		po = np.array([mu_e0, re0, ndex0, mu_e1, re1, ndex1, mu_e2, re2, ndex2])
		popt, pcov = curve_fit( SB_fit, idR, idSB, p0 = po, bounds = ([1, 1, 1, 1e-1, 20, 1, 1e-3, 300, 1], [10, 10, 3.5, 1e0, 70, 7, 7e-3, 2000, 6]),
			sigma = idSB_err, method = 'trf')

	# g band
	if ll == 1:
		re0, mu_e0, ndex0 = 4, 5e-1, 1.5
		re1, mu_e1, ndex1 = 50, 5e-2, 3
		re2, mu_e2, ndex2 = 650, 5e-5, 3

		po = np.array([mu_e0, re0, ndex0, mu_e1, re1, ndex1, mu_e2, re2, ndex2])
		popt, pcov = curve_fit( SB_fit, idR, idSB, p0 = po, bounds = ([1e-1, 1, 1,  1e-2, 20, 1,  1e-5, 400, 2], [1, 10, 3.5,  1e-1, 70, 7,  7e-4, 2000, 8]),
			sigma = idSB_err, method = 'trf')

	Ie0, Re0, Ne0, Ie1, Re1, Ne1, Ie2, Re2, Ne2 = popt
	fit_line = SB_fit(obs_r, Ie0, Re0, Ne0, Ie1, Re1, Ne1, Ie2, Re2, Ne2)

	fit_line_0 = sersic_func(obs_r, Ie0, Re0, Ne0)
	fit_line_1 = sersic_func(obs_r, Ie1, Re1, Ne1)
	fit_line_2 = sersic_func(obs_r, Ie2, Re2, Ne2)

	'''
	## single power law (10 ~ 1000 kpc)
	# r, i band
	if ll != 1:
		idux = (obs_r >= 10) & (obs_r <= 1000)
		idR = obs_r[idux]
		idSB = obs_fdens[idux]
		idSB_err = obs_ferr[idux]

		I0 = 60
		alpha = -1.8

		po = np.array([I0, alpha])
		popt, pcov = curve_fit( power_fit, idR, idSB, p0 = po, bounds = ([10, -3], [100, 0]), sigma = idSB_err, method = 'trf')

	# g band
	if ll == 1:
		idux = (obs_r >= 100) & (obs_r <= 1000)
		idR = obs_r[idux]
		idSB = obs_fdens[idux]
		idSB_err = obs_ferr[idux]

		I0 = 50
		alpha = -1.5

		po = np.array([I0, alpha])
		popt, pcov = curve_fit( power_fit, idR, idSB, p0 = po, bounds = ([10, -3], [100, -1]), sigma = idSB_err, method = 'trf')	

	fit_I0, fit_alpha = popt
	fit_line = power_fit(obs_r, fit_I0, fit_alpha)
	'''

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('%s band' % band[ll])

	ax.plot(obs_r, obs_fdens, ls = '-', color = 'k', alpha = 0.5,)
	ax.fill_between( obs_r, y1 = obs_fdens - obs_ferr, y2 = obs_fdens + obs_ferr, color = 'k', alpha = 0.15,)
	ax.plot(R_obs, flux_obs, ls = '-', color = 'c', alpha = 0.5, label = 'Z05')

	if ll < 2:
		#ax.plot(Z_r, Z_fdens, ls = '-', color = 'm', alpha = 0.5, label = 'Zhang 2018')
		dev_R = np.abs(Z_r - 1e3)
		idr = dev_R == np.min( dev_R )
		off_SB = Z_fdens[ idr ]
		ax.plot(Z_r, Z_fdens - off_SB, color = 'm', ls = '-', alpha = 0.5, label = 'Zhang 2018')

	ax.plot(obs_r, fit_line, ls = '--', color = 'r', alpha = 0.5,)
	ax.plot(obs_r, fit_line_0, ls = ':', color = 'c', alpha = 0.5, label = '$n = %.3f, R_{e} = %.3f$' % (Ne0, Re0),)
	ax.plot(obs_r, fit_line_1, ls = ':', color = 'b', alpha = 0.5, label = '$n = %.3f, R_{e} = %.3f$' % (Ne1, Re1),)
	ax.plot(obs_r, fit_line_2, ls = ':', color = 'g', alpha = 0.5, label = '$n = %.3f, R_{e} = %.3f$' % (Ne2, Re2),)

	#ax.plot(obs_r, fit_line, ls = '--', color = 'r', alpha = 0.5, label = 'power law,$\\alpha = %.3f$' % fit_alpha,)

	ax.set_ylim(1e-5, 2e1)
	ax.set_yscale('log')
	ax.set_xlim(1e0, 2e3)
	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax.set_xscale('log')
	ax.legend(loc = 3, frameon = False,)
	ax.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in',)
	ax.set_xlabel('R [kpc]')

	plt.savefig('com-BCG-star-Mass_%s-band_SB_pros-fit.png' % band[ll], dpi = 300)
	plt.close()
"""


