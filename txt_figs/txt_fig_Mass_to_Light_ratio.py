import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp

import astropy.units as U
import astropy.constants as C

from astropy import cosmology as apcy
from scipy import optimize
from scipy import stats as sts
from scipy import signal
from scipy import interpolate as interp
from scipy import optimize
from scipy import integrate as integ

### === ### cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

band = ['r', 'g', 'i']
Mag_sun = [ 4.65, 5.11, 4.53 ]
L_wave = np.array([ 6166, 4686, 7480 ])
psf_FWHM = [ 1.56, 1.67, 1.50 ] # arcsec

z_ref = 0.25
pixel = 0.396

Dl_ref = Test_model.luminosity_distance( z_ref ).value
Da_ref = Test_model.angular_diameter_distance( z_ref ).value
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec

### === M/L - color
def lg_linea_func( x, a, b):
	lg_M2L = a + b * x
	return lg_M2L

def resi_func( po, x, y):

	a, b = po[:]
	lg_m2l = lg_linea_func( x, a, b)
	delta = lg_m2l - y
	return delta

def lg_bi_linear_func(x, A, B, C0, D0):
	"""
	x_arr : g-r, r-i, luminosity
	"""
	x_gr, x_ri, x_lgL = x[0], x[1], x[2]
	y = A * x_gr + B * x_ri + C0 * x_lgL + D0
	return y

def resi_bi_line_func( po, x, y):

	A, B, C0, D0 = po[:]
	pre_y = lg_bi_linear_func(x, A, B, C0, D0)
	delta = pre_y - y
	return delta

def hist2d_pdf_func(x, y, bins, levels, smooth = None, weights = None,):

	from scipy.ndimage import gaussian_filter

	H, X, Y = np.histogram2d( x.flatten(), y.flatten(), bins = bins, weights = weights)

	if smooth is not None:
		H = gaussian_filter(H, smooth)

	Hflat = H.flatten()
	inds = np.argsort(Hflat)[::-1]
	Hflat = Hflat[inds]
	sm = np.cumsum(Hflat)
	sm /= sm[-1]
	V = np.empty(len(levels))

	for i, v0 in enumerate(levels):
		try:
			V[i] = Hflat[sm <= v0][-1]
		except IndexError:
			V[i] = Hflat[0]
	V.sort()

	m = np.diff(V) == 0
	if np.any(m) and not quiet:
		logging.warning("Too few points to create valid contours")
	while np.any(m):
		V[np.where(m)[0][0]] *= 1.0 - 1e-4
		m = np.diff(V) == 0
	V.sort()

	# Compute the bin centers.
	X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

	# Extend the array for the sake of the contours at the plot edges.
	H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
	H2[2:-2, 2:-2] = H
	H2[2:-2, 1] = H[:, 0]
	H2[2:-2, -2] = H[:, -1]
	H2[1, 2:-2] = H[0]
	H2[-2, 2:-2] = H[-1]
	H2[1, 1] = H[0, 0]
	H2[1, -2] = H[0, -1]
	H2[-2, 1] = H[-1, 0]
	H2[-2, -2] = H[-1, -1]
	X2 = np.concatenate(
		[
			X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
			X1,
			X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
		]
	)
	Y2 = np.concatenate(
		[
			Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
			Y1,
			Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
		]
	)

	return H, H2, X2, Y2, V


### === data load
# load_path = '/home/xkchen/tmp_run/data_files/figs/M2L_Lumi_selected/'

load_path = '/home/xkchen/figs/extend_bcgM_cat/Mass_Li_fit/'

dat = pds.read_csv( load_path + 'least-square_fit_cat.csv' )
tot_gr = np.array( dat['g-r'] )
tot_ri = np.array( dat['r-i'] )
tot_lg_Li = np.array( dat['lg_Li'] )
tot_lgM = np.array( dat['lg_Mstar_no-h'] )

#... all cluster sample
fit_dat = pds.read_csv( load_path + 'least-square_M-to-i-band-Lumi&color.csv' )

all_a_, all_b_ = np.array( fit_dat['a'])[0], np.array( fit_dat['b'])[0]
all_c_, all_d_ = np.array( fit_dat['c'])[0], np.array( fit_dat['d'])[0]

put_x = np.array( [tot_gr, tot_ri, tot_lg_Li] )
tot_fit_lgM = lg_bi_linear_func( put_x, all_a_, all_b_, all_c_, all_d_ )


#. sub-binning stellar mass 
bin_mx = np.linspace(11.1, 12.3, 7)
bin_mx = np.r_[ 10.8, bin_mx ]

bin_mc = 0.5 * ( bin_mx[1:] + bin_mx[:-1])
bin_medi_lgM, bin_lgM_std = np.array([]), np.array([])

for oo in range( len(bin_mx) - 1):
	id_in = ( tot_fit_lgM >= bin_mx[oo] ) & ( tot_fit_lgM <= bin_mx[oo+1] )
	sub_lgM = tot_lgM[id_in]

	bin_medi_lgM = np.r_[ bin_medi_lgM, np.median( sub_lgM ) ]
	bin_lgM_std = np.r_[ bin_lgM_std, np.std( sub_lgM ) ]

sp_R, Ps = sts.spearmanr( tot_lgM, tot_fit_lgM )
lgM_fit_Var = np.sum( (tot_fit_lgM - tot_lgM )**2) / len( tot_lgM )
lgM_fit_sigma = np.sqrt( lgM_fit_Var )


#. sub-binning mass-to-light ratio
obs_M2L = tot_lgM - tot_lg_Li
fit_M2L = tot_fit_lgM - tot_lg_Li

bin_x = np.linspace(0.14, 0.48, 11)
bin_x = np.r_[ -0.2, bin_x ]

bin_cen = 0.5 * ( bin_x[1:] + bin_x[:-1])
bin_medi, bin_std = np.array([]), np.array([])
bin_Ng = np.array([])

for oo in range( len(bin_x) - 1):
	id_in = ( fit_M2L >= bin_x[oo] ) & ( fit_M2L <= bin_x[oo+1] )
	sub_M2L = obs_M2L[id_in]

	bin_medi = np.r_[ bin_medi, np.median( sub_M2L ) ]
	bin_std = np.r_[ bin_std, np.std( sub_M2L ) ]
	bin_Ng = np.r_[ bin_Ng, np.sum(id_in) ]

idx0 = (obs_M2L >= 0.05) & (obs_M2L <= 0.1)
idx1 = (fit_M2L >= 0.25) & (fit_M2L <= 0.32)

_Nlim = np.sum( idx0 & idx1 )
cp_obs = obs_M2L + 0.
cp_obs[ idx0 & idx1 ] = np.random.random( _Nlim ) * (-0.1)

levels = (1 - np.exp( -np.log(10) ), 1 - np.exp( np.log(5) - np.log(10) ),)
H_arr, H2_arr, X2_arr, Y2_arr, V_arr = hist2d_pdf_func( fit_M2L, cp_obs, bins = [100, 100], 
	levels = levels, smooth = (1.0, 1.0), weights = None,)


### === figs
_cmap_lis = []
for ii in range( 9 ):
	sub_color = mpl.cm.Greys_r( ii / 8 )
	_cmap_lis.append( sub_color )


# fig_tx = plt.figure( figsize = (5.4, 5.4) )
# ax0 = fig_tx.add_axes( [0.15, 0.11, 0.83, 0.83] )

# ax0.scatter( fit_M2L, obs_M2L, marker = '.', s = 5.0, color = 'k', alpha = 0.45,)
# ax0.plot( fit_M2L, fit_M2L, ls = '-', color = 'k', alpha = 0.75,)

# ax0.contourf( X2_arr, Y2_arr, H2_arr.T, np.concatenate([ V_arr, [ H_arr.max() * (1 + 1e-4) ] ] ), 
# 	colors = [ _cmap_lis[5], _cmap_lis[7], _cmap_lis[8] ],)
# ax0.errorbar( bin_cen[bin_Ng > 1], bin_medi[bin_Ng > 1], yerr = bin_std[bin_Ng > 1], color = 'k', marker = 'o', capsize = 2.5,
# 	linestyle = 'none', alpha = 0.75,)

# ax0.set_ylim( 0.0, 0.55)
# ax0.set_xlim( 0.14, 0.47)

# ax0.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
# ax0.xaxis.set_minor_locator( ticker.AutoMinorLocator() )

# ax0.set_ylabel('$ {\\rm \\mathcal{lg} } \, (M_{\\ast}^{ \\mathrm{SED} } \, / \, L_{i}) $', fontsize = 15,)
# ax0.set_xlabel('$ {\\rm \\mathcal{lg} } \, (M_{\\ast} \, / \, L_{i}) {=} a \\times (g-r)$' + 
# 	'${+} b \\times (r-i) {+} c \\times {\\rm \\mathcal{lg} } L_{i} {+} d$', fontsize = 15,)

# ax0.annotate( text = '$\\mathcal{a} = \; \; \; %.3f$' % all_a_ + '\n' + '$\\mathcal{b} = \; \; \; %.3f$' % all_b_ + '\n' + 
# 	'$\\mathcal{c} = \; \; \; %.3f$' % ( all_c_ - 1 ) + '\n' + '$\\mathcal{d}{=}{-}%.3f$' % np.abs(all_d_), xy = (0.70, 0.05), 
# 	fontsize = 15, xycoords = 'axes fraction', color = 'k',)

# ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

# plt.savefig('/home/xkchen/Mass-to-Li_estimation.png', dpi = 300)
# # plt.savefig('/home/xkchen/Mass-to-Li_estimation.pdf', dpi = 300)
# plt.close()


fig_tx = plt.figure( figsize = (5.8, 5.8) )
ax0 = fig_tx.add_axes( [0.15, 0.11, 0.83, 0.83] )

ax0.scatter( tot_fit_lgM, tot_lgM, marker = '.', s = 5.0, color = 'grey', alpha = 0.5,)
ax0.plot( tot_fit_lgM, tot_fit_lgM, ls = '-', color = 'k',)

ax0.errorbar( bin_mc, bin_medi_lgM, yerr = bin_lgM_std, color = 'k', marker = 'o', capsize = 2.5, linestyle = 'none',)

ax0.set_ylim( 10.5, 12.25 )
ax0.set_xlim( 10.8, 12.25 )

x_tick_arr = np.arange(10.8, 12.4, 0.2)
tick_lis = [ '%.1f' % pp for pp in x_tick_arr ]
ax0.set_xticks( x_tick_arr )
ax0.get_xaxis().set_major_formatter( ticker.FixedFormatter( tick_lis ) )

ax0.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
ax0.xaxis.set_minor_locator( ticker.AutoMinorLocator() )

ax0.set_ylabel('$ {\\rm \\mathcal{lg} } \, (M_{\\ast}^{ \\mathrm{BCG} } \, / \, M_{\\odot})$', fontsize = 15,)
ax0.set_xlabel('$ \\langle {\\rm \\mathcal{lg} } \, M_{\\ast}^{\\mathrm{BCG} } \\rangle {=} a \\times (g-r)$' + 
	'${+} b \\times (r-i) {+} c \\times {\\rm \\mathcal{lg} } L_{i} {+} d$', fontsize = 15,)

ax0.annotate( text = '$\\mathcal{a} = \; \; \; %.2f$' % all_a_ + '\n' + '$\\mathcal{b} = \; \; \; %.2f$' % all_b_ + '\n' + 
	'$\\mathcal{c} = \; \; \; %.2f$' % ( all_c_ ) + '\n' + '$\\mathcal{d}{=}{-}%.2f$' % np.abs(all_d_), 
	xy = (0.70, 0.10), fontsize = 18, xycoords = 'axes fraction', color = 'k',)

ax0.annotate( text = '$\\sigma = %.2f$' % lgM_fit_sigma, xy = (0.05, 0.85), fontsize = 18, xycoords = 'axes fraction', color = 'k',)

ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

# plt.savefig('/home/xkchen/Mass-to-Li_estimation.png', dpi = 300)
plt.savefig('/home/xkchen/Mass_to_Li_estimation.pdf', dpi = 300)
plt.close()

