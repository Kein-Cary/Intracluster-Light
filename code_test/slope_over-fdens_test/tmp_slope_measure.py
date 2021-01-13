import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

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
from scipy.optimize import curve_fit
from scipy import signal
from scipy import interpolate as interp
from scipy.ndimage import gaussian_filter

from light_measure import jack_SB_func
from fig_out_module import arr_jack_func, arr_slope_func
from fig_out_module import cc_grid_img, grid_img

# cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
band = ['r', 'g', 'i']

def SB_fit(r, I_e, r_e,):
	"""
	SB profile : Mo. galaxy evolution and evolution, Chapter 2, eq. 2.23
	"""
	ndex = 4
	belta_n = 2 * ndex - 0.324
	f_n = - belta_n * ( r / r_e)**(1 / ndex) + belta_n
	I_r = I_e * np.exp( f_n )

	return I_r

def power_fit(r, I_0, alpha):

	I_r = I_0 * r**( alpha )

	return I_r

load = '/home/xkchen/mywork/ICL/'

#*************************************
"""
from light_measure import over_dens_sb_func

id_cen = 0
n_rbins = 100
N_bin = 30
z_ref = 0.25

pre_lis = ['low-BCG-star-Mass', 'high-BCG-star-Mass']
line_c = ['b', 'r']
line_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$']
'''
for kk in range( 2 ):

	lim_r = 0

	for ll in range( N_bin ):
		with h5py.File('/home/xkchen/jupyter/stack/' + pre_lis[kk] + '_jack-sub-%d_img_z-ref.h5' % ll, 'r') as f:
			tmp_img = np.array(f['a'])

		xn, yn = np.int(tmp_img.shape[1] / 2), np.int(tmp_img.shape[0] / 2)

		id_nn = np.isnan(tmp_img)
		eff_y, eff_x = np.where(id_nn == False)
		dR = np.sqrt((eff_y - yn)**2 + (eff_x - xn)**2)
		dR_max = np.int( dR.max() ) + 1
		lim_r = np.max( [lim_r, dR_max] )

	R_bins = np.logspace(0, np.log10(lim_r), n_rbins)

	for ll in range( N_bin ):

		with h5py.File('/home/xkchen/jupyter/stack/' + pre_lis[kk] + '_jack-sub-%d_img_z-ref.h5' % ll, 'r') as f:
			tt_img = np.array(f['a'])

		with h5py.File('/home/xkchen/jupyter/stack/' + pre_lis[kk] + '_jack-sub-%d_pix-cont_z-ref.h5' % ll, 'r') as f:
			tt_cont = np.array(f['a'])

		xn, yn = np.int(tt_img.shape[1] / 2), np.int(tt_img.shape[0] / 2)
		Intns_r, Intns, Intns_err, over_fdens, N_pix, nsum_ratio = over_dens_sb_func(tt_img, tt_cont, pixel, xn, yn, z_ref, R_bins,)

		with h5py.File('tmp_test/' + pre_lis[kk] + '_jack-sub-%d_over_fdens_z-ref.h5' % ll, 'w') as f:
			f['sb'] = np.array( Intns )
			f['sb_err'] = np.array( Intns_err )
			f['r'] = np.array( Intns_r )
			f['over_fdens'] = np.array( over_fdens )

	print( 'kk = ', kk)

	tmp_fdens, tmp_r = [], []

	for ll in range( N_bin ):

		with h5py.File('tmp_test/' + pre_lis[kk] + '_jack-sub-%d_over_fdens_z-ref.h5' % ll, 'r') as f:
			over_R = np.array(f['r'])
			over_fdens = np.array(f['over_fdens'])
		
		tmp_fdens.append( over_fdens )
		tmp_r.append( over_R )

	jk_R, jk_fdens, jk_fdens_err, lim_R = arr_jack_func(tmp_fdens, tmp_r, N_bin,)

	with h5py.File('tmp_test/' + pre_lis[kk] + '_jack-mean_over_fdens_z-ref.h5', 'w') as f:
		f['r'] = np.array( jk_R )
		f['over_fdens'] = np.array( jk_fdens )
		f['fdens_err'] = np.array( jk_fdens_err )
'''

plt.figure()
ax = plt.subplot(111)
ax.set_title('$\\Delta \\mu(r)$')

for kk in range( 2 ):

	with h5py.File('tmp_test/' + pre_lis[kk] + '_jack-mean_over_fdens_z-ref.h5', 'r') as f:
		jk_R = np.array(f['r'])
		jk_fdens = np.array(f['over_fdens'])
		jk_fdens_err = np.array(f['fdens_err'])

	ax.plot(jk_R, jk_fdens, color = line_c[kk], ls = '-', alpha = 0.5, label = line_name[kk],)
	ax.fill_between(jk_R, y1 = jk_fdens - jk_fdens_err, y2 = jk_fdens + jk_fdens_err, color = line_c[kk], alpha = 0.2,)

ax.set_ylim(1e-5, 2e0)
ax.set_yscale('log')
ax.set_xlim(1e1, 4e3)
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 1, frameon = False,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in',)
ax.set_xlabel('R [kpc]')

plt.savefig('over-fdens_compare.png', dpi = 300)
plt.close()
"""

id_cen = 0
n_rbins = 100
N_bin = 30
z_ref = 0.25

pre_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$',]

wind_len = 9
poly_order = 3
'''
for ll in range( 3 ):

	for mm in range( 2 ):

		tmp_r, tmp_sb = [], []

		for kk in range( N_bin ):

			with h5py.File('/home/xkchen/jupyter/stack_12_10/' + pre_lis[mm] + 
				'_%s-band_jack-sub-%d_SB-pro_z-ref_with-selection_gri-cat.h5' % (band[ll], kk), 'r') as f:
				c_r_arr = np.array(f['r'])
				c_sb_arr = np.array(f['sb'])
				c_sb_err = np.array(f['sb_err'])

			with h5py.File('/home/xkchen/jupyter/stack/' + pre_lis[mm] + 
				'_%s-band_jack-sub-%d_img_z-ref_with-selection_gri-cat.h5' % (band[ll], kk), 'r') as f:
				sub_img = np.array(f['a'])
			xn, yn = np.int(sub_img.shape[1] / 2), np.int(sub_img.shape[0] / 2)

			id_Nul = c_r_arr > 0
			c_r_arr[id_Nul == False] = np.nan
			c_sb_arr[id_Nul == False] = np.nan
			c_sb_err[id_Nul == False] = np.nan

			idr = ( c_r_arr > 1e3 ) & ( c_r_arr < 2e3 )
			idsb = np.nanmin(c_sb_arr[idr])
			devi_sb = c_sb_arr - idsb

			tmp_r.append( c_r_arr )
			tmp_sb.append( devi_sb )

		dt_arr = np.array( tmp_r )
		medi_R = np.nanmedian( dt_arr, axis = 0 )
		cc_tmp_r = []
		cc_tmp_sb = []

		for kk in range( N_bin ):

			xx_R = tmp_r[ kk ] + 0.
			xx_sb = tmp_sb[ kk ] + 0.

			deviR = np.abs( xx_R - medi_R )

			idmx = deviR > 0
			xx_sb[ idmx ] = np.nan

			cc_tmp_sb.append( xx_sb )
			cc_tmp_r.append( medi_R )

		tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(cc_tmp_sb, cc_tmp_r, band[ll], N_bin,)[4:]

		## slope estimate (with filter smoothing)
		tmp_kr, tmp_slope = [], []

		id_nn = np.isnan(medi_R)

		for kk in range( N_bin ):

			xx_R = tmp_r[ kk ][id_nn == False]
			xx_sb = tmp_sb[ kk ][id_nn == False]

			deviR = np.abs( xx_R - medi_R[id_nn == False] )

			idnx = deviR > 0
			xx_sb[idnx] = tt_jk_SB[idnx]

			id_break = np.isnan( xx_sb )
			xx_sb[ id_break ] = tt_jk_SB[ id_break ]

			slopr, df_sign = arr_slope_func(xx_sb, medi_R[id_nn == False], wind_len, poly_order, id_log = True,)
			tmp_kr.append( slopr )
			tmp_slope.append( df_sign )

		m_slo_r, m_slope, m_slope_err = arr_jack_func(tmp_slope, tmp_kr, N_bin,)[:3]

		## save BG-subtracted pros
		with h5py.File( pre_lis[mm] + '_%s-band_Mean_jack_SB-pro_z-ref_with-selection_gri-cat.h5' % band[ll], 'w') as f:
			f['r'] = np.array(tt_jk_R)
			f['sb'] = np.array(tt_jk_SB)
			f['sb_err'] = np.array(tt_jk_err)

		keys = ['R', 'dlogsb_dlogr', 'slop_err']
		values = [m_slo_r, m_slope, m_slope_err]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv( pre_lis[mm] + '_%s-band_mean_BG-sub_SB-pros_slope.csv' % band[ll] )
'''
tmp_r, tmp_sb, tmp_err = [], [], []

for kk in range( 3 ):

	for ll in range( 2 ):

		with h5py.File('/home/xkchen/jupyter/stack_12_10/' + pre_lis[ll] + '_%s-band_Mean_jack_img_z-ref_with-selection_gri-cat.h5' % band[kk], 'r') as f:
			tt_img = np.array(f['a'])

		xn, yn = np.int(tt_img.shape[1] / 2), np.int(tt_img.shape[0] / 2)

		id_nan = np.isnan(tt_img)
		idvx = id_nan == False
		idy, idx = np.where(idvx == True)
		x_low, x_up = np.min(idx), np.max(idx)
		y_low, y_up = np.min(idy), np.max(idy)

		dpt_img = tt_img[y_low: y_up + 1, x_low: x_up + 1]
		img_block = cc_grid_img(dpt_img, 100, 100,)[0]

		with h5py.File('/home/xkchen/jupyter/stack_12_10/' + pre_lis[ll] + '_%s-band_Mean_jack_SB-pro_z-ref_with-selection_gri-cat.h5' % band[kk], 'r') as f:
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


		cen_L = 2428 #670
		dnoise = 75
		#cen_img = tt_img[yn - cen_L: yn + cen_L, xn - cen_L:xn + cen_L] / pixel**2
		cen_img = tt_img.copy()
		id_Nan = np.isnan(cen_img)
		cen_img[id_Nan] = 0
		cen_img = cen_img / pixel**2

		kernl_img = gaussian_filter(cen_img, sigma = dnoise,  mode = 'nearest')

		f_lels = np.array([3e-3, 4e-3, 1e-2, 5e-2, 1e-1, 5e-1])
		color_lis = ['r', 'g', 'b', 'orange', 'k', 'm']
		#for pp in range( 6 ):
		#	color_lis.append( mpl.cm.rainbow(pp / 6) )

		D_ref = Test_model.angular_diameter_distance( z_ref ).value
		L_pix = pixel * D_ref * 1e3 / rad2asec

		R_100 = 100 / L_pix
		R_500 = 500 / L_pix
		R_1000 = 1000 / L_pix

		fig = plt.figure()
		ax0 = plt.subplot(111)
		ax0.set_title( fig_name[ll] + ' %s band' % band[kk] )

		Nlx = np.linspace(0, cen_img.shape[1] - 1, cen_img.shape[1])
		Nly = np.linspace(0, cen_img.shape[0] - 1, cen_img.shape[0])
		NlX, NlY = np.meshgrid(Nlx, Nly)

		tg = ax0.imshow(cen_img, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm(),)
		tf = ax0.contour(Nlx, Nly, kernl_img, origin = 'lower', levels = f_lels, colors = color_lis, alpha = 0.75, linewidths = 1,)
		cb = plt.clabel(tf, inline = False, fontsize = 4, fmt = '%.3f', colors = 'r',)

		clust = Circle(xy = (xn, yn), radius = R_100, fill = False, ec = 'r', ls = ':', linewidth = 1.25, alpha = 0.5,)
		ax0.add_patch(clust)
		clust = Circle(xy = (xn, yn), radius = R_500, fill = False, ec = 'r', ls = '--', linewidth = 1.25, alpha = 0.5,)
		ax0.add_patch(clust)
		clust = Circle(xy = (xn, yn), radius = R_1000, fill = False, ec = 'r', ls = '-.', linewidth = 1.25, alpha = 0.5,)
		ax0.add_patch(clust)
		clust = Circle(xy = (xn, yn), radius = 2 * R_1000, fill = False, ec = 'r', ls = '-', linewidth = 1.25, alpha = 0.5,)
		ax0.add_patch(clust)
		ax0.set_xlim(x_low, x_up)
		ax0.set_ylim(y_low, y_up)

		plt.savefig('%s_%s-band_BCG_region_check.png' % (pre_lis[ll], band[kk] ), dpi = 300)
		plt.close()

		raise
		'''
		fig = plt.figure( figsize = (13.12, 4.8) )
		ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
		ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

		ax0.set_title( pre_lis[ll] + ' %s band' % band[kk] )
		tg = ax0.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
		cb = plt.colorbar(tg, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
		cb.formatter.set_powerlimits( (0,0) )

		ax1.plot(c_r_arr, c_sb_arr, ls = '-', color = 'r', alpha = 0.8,)
		ax1.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'r', alpha = 0.2,)

		ax1.set_ylim(2e-3, 2e0)
		ax1.set_yscale('log')
		ax1.set_xlim(1e1, 4e3)
		ax1.set_xlabel('R [kpc]')
		ax1.set_ylabel('SB [nanomaggies / $arcsec^2$]')
		ax1.set_xscale('log')
		ax1.legend(loc = 3, frameon = False,)
		ax1.grid(which = 'both', axis = 'both', alpha = 0.25)
		ax1.tick_params(axis = 'both', which = 'both', direction = 'in',)
		tick_form = mpl.ticker.LogFormatter(labelOnlyBase = False)
		ax1.get_yaxis().set_minor_formatter(tick_form)

		plt.savefig('%s_%s-band_2D-grd_SB.png' % (pre_lis[ll], band[kk]), dpi = 300)
		plt.close()
		'''
raise

tmp_sub_r, tmp_sub_sb, tmp_sub_err = [], [], []
tmp_slop_r, tmp_slop, tmp_slop_err = [], [], []

for kk in range( 3 ):

	for mm in range( 2 ):

		with h5py.File( pre_lis[mm] + '_%s-band_Mean_jack_SB-pro_z-ref_with-selection_gri-cat.h5' % band[kk], 'r') as f:
			tt_jk_r = np.array( f['r'] )
			tt_jk_sb = np.array( f['sb'] )
			tt_jk_err = np.array( f['sb_err'] )

		tmp_sub_r.append( tt_jk_r )
		tmp_sub_sb.append( tt_jk_sb )
		tmp_sub_err.append( tt_jk_err )

		k_dat = pds.read_csv( pre_lis[mm] + '_%s-band_mean_BG-sub_SB-pros_slope.csv' % band[kk] )
		k_r, k_v, k_err = np.array(k_dat['R']), np.array(k_dat['dlogsb_dlogr']), np.array(k_dat['slop_err'])

		tmp_slop_r.append( k_r )
		tmp_slop.append( k_v )
		tmp_slop_err.append( k_err )

line_name = ['r low $M_{\\ast}$', 'r high $M_{\\ast}$',
			'g low $M_{\\ast}$', 'g high $M_{\\ast}$', 
			'i low $M_{\\ast}$', 'i high $M_{\\ast}$']

line_c = ['r', 'r', 'g', 'g', 'b', 'b']

line_s = ['--', '-', '--', '-', '--', '-']

a_ref = 1 / (1 + z_ref)


plt.figure()
ax = plt.subplot(111)

for pp in (0, 2, 4):
#for pp in (1, 3, 5):
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
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in',)
ax.set_xlabel('R [kpc]')

if pp % 2 == 0:
	plt.savefig('low_BCG-star-Mass_gri-SB_pros.png', dpi = 300)
if pp % 2 == 1:
	plt.savefig('high_BCG-star-Mass_gri-SB_pros.png', dpi = 300)
plt.close()


plt.figure()
ax = plt.subplot(111)

for pp in range( 6 ):

	#ax.plot(tmp_r[pp], tmp_sb[pp], ls = line_s[pp], color = line_c[pp], alpha = 0.5, label = line_name[pp],)
	#ax.fill_between(tmp_r[pp], y1 = tmp_sb[pp] - tmp_err[pp], y2 = tmp_sb[pp] + tmp_err[pp], 
	#	color = line_c[pp], alpha = 0.15,)

	ax.plot(tmp_sub_r[pp], tmp_sub_sb[pp], ls = line_s[pp], color = line_c[pp], alpha = 0.5, label = line_name[pp],)
	ax.fill_between(tmp_sub_r[pp], y1 = tmp_sub_sb[pp] - tmp_sub_err[pp], y2 = tmp_sub_sb[pp] + tmp_sub_err[pp], 
		color = line_c[pp], alpha = 0.15,)

ax.set_ylim(1e-5, 2e0)
ax.set_yscale('log')
ax.set_xlim(1e1, 4e3)
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 1, frameon = False, fontsize = 8.0,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in',)
ax.set_xlabel('R [kpc]')

plt.savefig('BCG-star-Mass_gri-SB_pros.png', dpi = 300)
plt.close()

raise
## slope, comoving

plt.figure()
ax = plt.subplot(111)

#for pp in (1, 3, 5):
for pp in (0, 2, 4):

	#ax.errorbar(tmp_slop_r[pp], tmp_slop[pp], yerr = tmp_slop_err[pp], xerr = None, color = line_c[pp], marker = '.', ms = 3, 
	#	mec = line_c[pp], mfc = 'none', ls = '-', ecolor = line_c[pp], elinewidth = 1, alpha = 0.5, label = line_name[pp],)

	ax.fill_between(tmp_slop_r[pp] * h / a_ref, y1 = tmp_slop[pp] - tmp_slop_err[pp], y2 = tmp_slop[pp] + tmp_slop_err[pp], 
		color = line_c[pp], alpha = 0.2,)
	ax.plot(tmp_slop_r[pp] * h / a_ref, tmp_slop[pp], ls = '-', color = line_c[pp], alpha = 0.8, label = line_name[pp],)

	#ax.plot(tmp_slop_r[pp] * h / a_ref, tmp_slop[pp], ls = line_s[pp], color = line_c[pp], alpha = 0.8, label = line_name[pp],)

ax.set_xlabel('$R_{c}[kpc / h]$')
ax.set_ylabel('d(lgSB) / d(lgR)')
ax.set_xscale('log')
ax.set_xlim(1e1, 4e2)
ax.set_ylim(-3, 0)
ax.legend(loc = 2,)
ax.grid(which = 'both', axis = 'both', alpha = 0.20)
ax.tick_params(axis = 'both', which = 'both', direction = 'in')
if pp % 2 == 0:
	plt.savefig('low_BCG-star-Mass_gri-SB_slope.png', dpi = 300)
if pp % 2 == 1:
	plt.savefig('high_BCG-star-Mass_gri-SB_slope.png', dpi = 300)
plt.close()


plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
ax = plt.subplot(gs[0])
bx = plt.subplot(gs[1])

#for pp in ( 0, 2, 4 ): # low BCG star Mass
#for pp in ( 1, 3, 5 ): # high BCG star Mass
for pp in range( 6 ):

	obs_k_r = (tmp_slop_r[pp] / a_ref) * h
	obs_k = tmp_slop[pp]
	obs_k_err = tmp_slop_err[pp]

	obs_r = (tmp_sub_r[pp] / a_ref) * h
	obs_fdens = tmp_sub_sb[pp]
	obs_ferr = tmp_sub_err[pp]

	idvx = (obs_r >= 12.7) & ( obs_r <= 17)
	idR = obs_r[idvx]
	idSB = obs_fdens[idvx]
	idSB_err = obs_ferr[idvx]

	Re = 13
	mu_e = 5e-1

	po = np.array([mu_e, Re])
	popt, pcov = curve_fit(SB_fit, idR, idSB, p0 = po, bounds = ([1e-1, 11], [1e0, 15]), sigma = idSB_err, method = 'trf')

	Ie, Re = popt
	fit_line = SB_fit(obs_r, Ie, Re,)

	icl_resi = obs_fdens - fit_line
	icl_res_R, icl_res_slope = arr_slope_func(icl_resi, obs_r, wind_len, poly_order, id_log = True,)


	#ax.plot(obs_r, obs_fdens, ls = '-', color = line_c[pp], alpha = 0.5, label = line_name[pp],)
	#ax.plot(obs_r, fit_line, ls = '-.', color = line_c[pp], alpha = 0.5,)
	ax.plot(obs_r, obs_fdens, ls = line_s[pp], color = line_c[pp], alpha = 0.5, label = line_name[pp],)

	#ax.plot(obs_r, icl_resi, ls = '-', color = line_c[pp], alpha = 0.5, label = line_name[pp],)
	#ax.plot(obs_r, icl_resi, ls = line_s[pp], color = line_c[pp], alpha = 0.5, label = line_name[pp],)

	#bx.plot( obs_k_r, obs_k, ls = '-', color = line_c[pp], alpha = 0.5,)
	#bx.fill_between( obs_k_r, y1 = obs_k - obs_k_err, y2 = obs_k + obs_k_err, color = line_c[pp], alpha = 0.2,)
	bx.plot( obs_k_r, obs_k, ls = line_s[pp], color = line_c[pp], alpha = 0.5,)

	#bx.plot(icl_res_R, icl_res_slope, ls = '-', color = line_c[pp], alpha = 0.5,)
	#bx.plot(icl_res_R, icl_res_slope, ls = line_s[pp], color = line_c[pp], alpha = 0.5,)

ax.set_xlabel('$R_{c}[kpc / h]$')
ax.set_ylabel('$SB [nanomaggies / arcsec^2]$')
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_ylim(1e-3, 3e-1)
#ax.set_ylim(1e-3, 4e-2)

ax.set_xlim(2e1, 2e2)
ax.legend(loc = 1, fontsize = 8,)
ax.grid(which = 'both', axis = 'both', alpha = 0.20)
ax.tick_params(axis = 'both', which = 'both', direction = 'in')

bx.set_xlabel('$R_{c}[kpc / h]$')
bx.set_ylabel('d(lgSB) / d(lgR)')
bx.set_xscale('log')
bx.set_xlim( ax.get_xlim() )

bx.set_ylim(-3, -1)
#bx.set_ylim(-3, 3)

bx.grid(which = 'both', axis = 'both', alpha = 0.20)
bx.tick_params(axis = 'both', which = 'both', direction = 'in')
ax.set_xticklabels( labels = [], minor = True, fontsize = 0.001)

plt.subplots_adjust(left = 0.15, right = 0.9, hspace = 0.02 )
'''
if pp % 2 == 0:
	plt.savefig('%s_grident_compare.png' % pre_lis[0], dpi = 300)
else:
	plt.savefig('%s_grident_compare.png' % pre_lis[1], dpi = 300)
'''
plt.savefig('tot_SB-slope_compare.png', dpi = 300)
plt.close()


