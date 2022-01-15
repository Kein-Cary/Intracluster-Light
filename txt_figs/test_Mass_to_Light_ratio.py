import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

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

### === data load
cat_lis = [ 'low-age', 'hi-age' ]
fig_name = [ 'Low $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 
			'High $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $' ]
file_s = 'age_bin_fixed_BCG_M'

cat_path = '/home/xkchen/figs/'
out_path = '/home/xkchen/figs/M2L_fit_age_bin_fixed_bcgM/'

#... relation estimation
based_str = 'i'

dd_gr, dd_ri, dd_Li = [], [], []
dd_lgM = []
dd_fit_p = []
dd_sigma, dd_sp_R = [], []

for mm in range( 2 ):

	#... lg_Mstar
	l_dat = pds.read_csv( cat_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[mm] )
	l_obs_z, l_rich  = np.array( l_dat['z']), np.array( l_dat['rich'])
	l_lgM, l_age = np.array( l_dat['lg_Mstar']), np.array( l_dat['BCG_age'] )

	#... mag
	pdat = pds.read_csv( '/home/xkchen/figs/%s_BCG-color.csv' % cat_lis[mm] )
	p_g_mag, p_r_mag, p_i_mag = np.array( pdat['g_mag'] ), np.array( pdat['r_mag'] ), np.array( pdat['i_mag'] )

	gr_ = p_g_mag - p_r_mag
	ri_ = p_r_mag - p_i_mag

	#... Lumi
	pdat = pds.read_csv( '/home/xkchen/figs/%s_BCG_cmag.csv' % cat_lis[mm] )
	i_cMag = np.array( pdat['i_cMag'] )
	i_Lumi = 10**( -0.4 * ( i_cMag - Mag_sun[2] ) )

	r_cMag = np.array( pdat['r_cMag'] )
	r_Lumi = 10**( -0.4 * ( r_cMag - Mag_sun[0] ) )

	g_cMag = np.array( pdat['g_cMag'] )
	g_Lumi = 10**( -0.4 * ( g_cMag - Mag_sun[1] ) )

	L_i = i_Lumi + 0.
	L_g = g_Lumi + 0. 
	L_r = r_Lumi + 0.

	i_Mag = i_cMag + 0.
	g_Mag = g_cMag + 0.
	r_Mag = r_cMag + 0.

	gr_arr = gr_ + 0.
	ri_arr = ri_ + 0.
	lg_Mstar = l_lgM - 2 * np.log10( h )

	dd_gr.append( gr_ )
	dd_ri.append( ri_ )
	dd_Li.append( L_i )
	dd_lgM.append( lg_Mstar )

	fit_dat = pds.read_csv( out_path + '%s_least-square_M-to-%s-band-Lumi&color.csv' % (cat_lis[mm], based_str),)	
	a_fit, b_fit = np.array( fit_dat['a'])[0], np.array( fit_dat['b'])[0]
	c_fit, d_fit = np.array( fit_dat['c'])[0], np.array( fit_dat['d'])[0]

	dd_fit_p.append( [a_fit, b_fit, c_fit, d_fit] )


	lg_Lumi = np.log10( L_i )
	put_x = np.array( [ gr_arr, ri_arr, lg_Lumi ] )
	fit_M2L = lg_bi_linear_func( put_x, a_fit, b_fit, c_fit, d_fit)

	_lg_M = lg_Mstar + 0.
	Var = np.sum( (fit_M2L - _lg_M)**2 ) / len( _lg_M )
	sigma = np.sqrt( Var )
	sp_R = sts.spearmanr( fit_M2L, _lg_M)[0]

	dd_sigma.append( sigma )
	dd_sp_R.append( sp_R )

	# bin_x = np.linspace(11.1, 12.3, 7)
	# bin_x = np.r_[ 10.8, bin_x ]

	# bin_cen = 0.5 * ( bin_x[1:] + bin_x[:-1])
	# bin_medi, bin_std = np.array([]), np.array([])
	# bin_Ng = np.array([])

	# for oo in range( len(bin_x) - 1):
	# 	id_in = ( fit_M2L >= bin_x[oo] ) & ( fit_M2L <= bin_x[oo+1] )
	# 	sub_lgM = _lg_M[id_in]

	# 	bin_medi = np.r_[ bin_medi, np.median( sub_lgM ) ]
	# 	bin_std = np.r_[ bin_std, np.std( sub_lgM ) ]
	# 	bin_Ng = np.r_[ bin_Ng, np.sum(id_in) ]

	# plt.figure()
	# ax = plt.subplot(111)
	# ax.set_title('least square')

	# ax.scatter( fit_M2L, _lg_M, marker = 'o', s = 1.5, color = 'k', alpha = 0.15, zorder = 100)
	# ax.plot( fit_M2L, fit_M2L, 'b--',)

	# ax.errorbar( bin_cen[bin_Ng > 1], bin_medi[bin_Ng > 1], yerr = bin_std[bin_Ng > 1], color = 'g', marker = 'o', capsize = 2.5,)

	# ax.annotate( text = fig_name[mm], xy = (0.65, 0.15), xycoords = 'axes fraction',)
	# ax.annotate( text = '$\\sigma = %.3f\, ; \; R = %.3f$' % (sigma, sp_R), xy = (0.65, 0.05), xycoords = 'axes fraction',)
	# ax.annotate( text = 'a = %.3f' % a_fit + '\n' + 'b = %.3f' % b_fit + '\n' + 'c = %.3f' % c_fit + 
	# 	'\n' + 'd = %.3f' % d_fit, xy = (0.10, 0.75), xycoords = 'axes fraction',)
	# ax.set_xlabel(
	# 	'$ \\lg \, (M_{\\ast}) = a \\cdot (g-r) + b \\cdot (r-i) + c \\cdot \\lg(L_{%s}) + d $' % (based_str),)
	# ax.set_ylabel('$ \\lg \, (M_{\\ast}) $' )
	# ax.set_xlim(10.80, 12.25)
	# ax.set_ylim(10.50, 12.50)
	# plt.savefig('/home/xkchen/%s_M-to-L_estimate.png' % cat_lis[mm], dpi = 300)
	# plt.close()


dd_gr, dd_ri, dd_Li = np.array( dd_gr ), np.array( dd_ri ), np.array( dd_Li )
dd_lgM = np.array( dd_lgM )
dd_lg_Li = [ np.log10( dd_Li[0] ), np.log10( dd_Li[1] ) ]

#... all cluster sample
fit_dat = pds.read_csv( '/home/xkchen/figs/M2L_Lumi_selected/least-square_M-to-%s-band-Lumi&color.csv' % based_str,)
all_a_, all_b_ = np.array( fit_dat['a'])[0], np.array( fit_dat['b'])[0]
all_c_, all_d_ = np.array( fit_dat['c'])[0], np.array( fit_dat['d'])[0]

tot_gr = np.hstack( (dd_gr[0], dd_gr[1]) )
tot_ri = np.hstack( (dd_ri[0], dd_ri[1]) )
tot_lg_Li = np.hstack( (dd_lg_Li[0], dd_lg_Li[1]) )
tot_lgM = np.hstack( (dd_lgM[0], dd_lgM[1]) )


put_x = np.array( [tot_gr, tot_ri, tot_lg_Li] )
tot_fit_line = lg_bi_linear_func( put_x, all_a_, all_b_, all_c_, all_d_ )

# lo_put_x = np.array( [dd_gr[0], dd_ri[0], dd_lg_Li[0] ] )
# lo_fit_line = lg_bi_linear_func( lo_put_x, dd_fit_p[0][0], dd_fit_p[0][1], dd_fit_p[0][2], dd_fit_p[0][3])

# hi_put_x = np.array( [dd_gr[1], dd_ri[1], dd_lg_Li[1] ] )
# hi_fit_line = lg_bi_linear_func( hi_put_x, dd_fit_p[1][0], dd_fit_p[1][1], dd_fit_p[1][2], dd_fit_p[1][3])


#... all cluster sample
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

_cmap_lis = []
for ii in range( 9 ):
	sub_color = mpl.cm.Greys_r( ii / 8)
	_cmap_lis.append( sub_color )

# sp_R, Ps = sts.spearmanr(tot_lgM, tot_fit_line)

# Var = np.sum( ( tot_fit_line - tot_lgM )**2 ) / len( tot_lgM )
# sigma = np.sqrt( Var )

# bin_x = np.linspace(10.75, 12.25, 9)

# bin_cen = 0.5 * ( bin_x[1:] + bin_x[:-1])
# bin_medi, bin_std = np.array([]), np.array([])
# bin_Ng = np.array([])

# for oo in range( len(bin_x) - 1):
# 	id_in = ( tot_fit_line >= bin_x[oo] ) & ( tot_fit_line <= bin_x[oo+1] )
# 	sub_lgM = tot_lgM[id_in]

# 	bin_medi = np.r_[ bin_medi, np.median( sub_lgM ) ]
# 	bin_std = np.r_[ bin_std, np.std( sub_lgM ) ]
# 	bin_Ng = np.r_[ bin_Ng, np.sum(id_in) ]


# levels = (1 - np.exp(-0.5), 1-np.exp(-2),)
# H_arr, H2_arr, X2_arr, Y2_arr, V_arr = hist2d_pdf_func( tot_fit_line, tot_lgM, bins = [100, 100], levels = levels, 
# 	smooth = (1.5, 1.0), weights = None,)

# fig = plt.figure( figsize = (5.8, 5.2) )
# ax = fig.add_axes( [0.165, 0.115, 0.80, 0.80] )
# ax.scatter( tot_fit_line, tot_lgM, marker = 'o', s = 3.5, color = 'k', alpha = 0.35,)

# ax.plot( tot_fit_line, tot_fit_line, ls = '--', color = 'tomato',) # 'lightcoral',)
# ax.errorbar( bin_cen[bin_Ng > 1], bin_medi[bin_Ng > 1], yerr = bin_std[bin_Ng > 1], color = 'royalblue', marker = 'o', capsize = 2.5,)

# ax.contourf( X2_arr, Y2_arr, H2_arr.T, np.concatenate([ V_arr, [ H_arr.max() * (1 + 1e-4) ] ] ), 
# 	colors = [ _cmap_lis[5], _cmap_lis[7], _cmap_lis[8] ],)

# ax.set_ylim(10.48, 12.30)
# ax.set_xlim(10.75, 12.30)

# ax.annotate( text = '$\\mathcal{\\sigma} = %.2f$' % sigma + '\n' + '$\\mathcal{r}_{s} = %.2f$' % sp_R + '\n' + '$\\mathcal{p}_{s} = %.2f$' % Ps, 
# 	xy = (0.75, 0.10), fontsize = 15, xycoords = 'axes fraction', color = 'k',)
# ax.set_ylabel('$ {\\rm \\mathcal{lg} } \, M_{\\ast \; , \, SED} \; [M_{\\odot}] $', fontsize = 15,)
# ax.set_xlabel('$ {\\rm \\mathcal{lg} } \, M_{\\ast \; , \, CLMR} \; [M_{\\odot}] $', fontsize = 15,)
# ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

# plt.savefig('/home/xkchen/color-Lumi-Mass_relation.png', dpi = 300)
# plt.close()


# fig with M/L
obs_M2L = tot_lgM - tot_lg_Li
fit_M2L = tot_fit_line - tot_lg_Li

sp_R, Ps = sts.spearmanr(obs_M2L, fit_M2L)
Var = np.sum( (fit_M2L - obs_M2L )**2 ) / len( tot_lgM )
sigma = np.sqrt( Var )

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

levels = (1 - np.exp(-0.5), 1-np.exp(-2),)
H_arr, H2_arr, X2_arr, Y2_arr, V_arr = hist2d_pdf_func( fit_M2L, obs_M2L, bins = [100, 100], levels = levels, 
	smooth = (1.5, 1.0), weights = None,)


fig = plt.figure( figsize = (5.8, 5.2) )
ax = fig.add_axes( [0.155, 0.115, 0.80, 0.80] )

ax.scatter( fit_M2L, obs_M2L, marker = 'o', s = 3.5, color = 'k', alpha = 0.35,)

ax.plot( fit_M2L, fit_M2L, ls = '--', color = 'tomato',)

ax.errorbar( bin_cen[bin_Ng > 1], bin_medi[bin_Ng > 1], yerr = bin_std[bin_Ng > 1], color = 'royalblue', marker = 'o', capsize = 2.5,)

ax.contourf( X2_arr, Y2_arr, H2_arr.T, np.concatenate([ V_arr, [ H_arr.max() * (1 + 1e-4) ] ] ), 
	colors = [ _cmap_lis[5], _cmap_lis[7], _cmap_lis[8] ],)

ax.set_ylim(-0.18, 0.70)
ax.set_xlim(0.14, 0.47)

ax.annotate( text = '$\\mathcal{\\sigma} = %.2f$' % sigma + '\n' + '$\\mathcal{r}_{s} = %.2f$' % sp_R + '\n' + '$\\mathcal{p}_{s} = %.2f$' % Ps, 
	xy = (0.75, 0.10), fontsize = 15, xycoords = 'axes fraction', color = 'k',)

ax.set_ylabel('$ {\\rm \\mathcal{lg} } \, (M_{\\ast \; , \, SED} \, / \, \\mathrm{L}_{i}) $', fontsize = 15,)
ax.set_xlabel('$ {\\rm \\mathcal{lg} } \, (M_{\\ast \; , \, CLMR} \, / \, \\mathrm{L}_{i}) $', fontsize = 15,)

ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
plt.savefig('/home/xkchen/mass-to-Li.png', dpi = 300)
plt.show()

