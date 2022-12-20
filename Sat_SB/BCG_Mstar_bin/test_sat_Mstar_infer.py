"""
use to infer stellar mass with Luminosity~(i.e. r-band luminosity)
"""
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
from scipy import stats as sts
from scipy import signal
from scipy import interpolate as interp
from scipy import optimize
from scipy import integrate as integ


### === ### cosmology
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

##. constant in r, g, i bands
rad2asec = U.rad.to(U.arcsec)

band = ['r', 'g', 'i']
Mag_sun = [ 4.65, 5.11, 4.53 ]
L_wave = np.array([ 6166, 4686, 7480 ])

psf_FWHM = [ 1.56, 1.67, 1.50 ] # arcsec

z_ref = 0.25
pixel = 0.396


### === ### M/L --> Lx~(x: r, i)
##. use to calculate scatter
def lg_linear_func( x, a, b):
	lg_M2L = a + b * x
	return lg_M2L

def resi_func( po, x, y):
	a, b = po[:]
	lg_m2l = lg_linear_func( x, a, b)
	delta = lg_m2l - y
	return delta

##.
def lg_M_func( x, a, b):
	lg_M = a + b * x
	return lg_M

def resi_M2L_func( po, x, y):
	a, b = po[:]
	lg_m2l = lg_M_func( x, a, b)
	delta = lg_m2l - y
	return delta


### === ### data load
cat_path = '/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/SM_probe/'

#.
dat = pds.read_csv( cat_path + 'Extend_BCGM_bin_dered-mag_Pcen_cat.csv')
gr_arr = np.array( dat['dered_g-r'] )
ri_arr = np.array( dat['dered_r-i'] )
gi_arr = np.array( dat['dered_g-i'] )

lg_Mstar = np.array( dat['lg_Mstar'] )   # unit : M_sun / h^2
lg_Mstar = lg_Mstar - 2 * np.log10( h )


cc_dat = pds.read_csv( cat_path + 'Extend_BCGM_bin_cat_non-dered_color_cModemag.csv')

ra, dec, z = np.array( cc_dat['ra'] ), np.array( cc_dat['dec'] ), np.array( cc_dat['z'] )
p_IDs = np.array( cc_dat['objID'] )

cMod_mag_i = np.array( cc_dat['cMod_mag_i'] )
cMod_mag_r = np.array( cc_dat['cMod_mag_r'] )
cMod_mag_g = np.array( cc_dat['cMod_mag_g'] )

Dl_z = Test_model.luminosity_distance( z ).value
mod_L = 5 * np.log10( Dl_z * 1e6 ) - 5

i_cMag = cMod_mag_i - mod_L
r_cMag = cMod_mag_r - mod_L
g_cMag = cMod_mag_g - mod_L

i_Lumi = 10**( -0.4 * ( i_cMag - Mag_sun[2] ) )
r_Lumi = 10**( -0.4 * ( r_cMag - Mag_sun[0] ) )
g_Lumi = 10**( -0.4 * ( g_cMag - Mag_sun[1] ) )


##. points selection for formula estimate
def Lx2M_func_point_select():

	## L_r --> Mass
	lg_L_arr = np.log10( r_Lumi )
	lg_M_arr = lg_Mstar + 0.

	sigma = 5
	kk = 0

	id_R_lim = [ 1 ]   ### initial, assume there is one outlier
	sum_dex = np.sum( id_R_lim )
	D_sigma = 2    ### 2-sigma use for find outlier

	while sum_dex > 0:

		put_x = lg_L_arr

		p0 = [ 0.5, 0.02 ]

		res_lsq = optimize.least_squares( resi_M2L_func, x0 = np.array( p0 ), loss = 'cauchy', 
					f_scale = 0.1, args = ( put_x, lg_M_arr),)

		a_fit = res_lsq.x[0]
		b_fit = res_lsq.x[1]


		#... fitting lg_Mstar
		fit_M = lg_M_func( put_x, a_fit, b_fit )

		Var = np.sum( (fit_M - lg_M_arr)**2 ) / len( lg_M_arr )
		sigma = np.sqrt( Var )
		sp_R = sts.spearmanr( fit_M, lg_M_arr)[0]


		#... fit relation between fitting and obs. and exclude outliers
		po = [0.9, 10]
		popt, pcov = optimize.curve_fit( lg_linear_func, xdata = fit_M, ydata = lg_M_arr, p0 = po,)

		_a0, _b0 = popt
		com_line = lg_linear_func( fit_M, _a0, _b0 )

		dR_com_l = np.abs( _b0 * fit_M + _a0 - lg_M_arr) / np.sqrt(1 + _b0**2)
		id_R_lim = dR_com_l >= D_sigma * sigma

		sum_dex = np.sum( id_R_lim )


		plt.figure()
		ax = plt.subplot(111)
		ax.set_title('least square')
		ax.scatter( fit_M, lg_M_arr, marker = 'o', s = 1.5, color = 'k', alpha = 0.15, zorder = 100)
		ax.plot( fit_M, fit_M, 'b--',)

		ax.scatter( fit_M[id_R_lim], lg_M_arr[id_R_lim], marker = '*', s = 5.5, color = 'g',)

		ax.annotate( s = '$\\sigma = %.3f\, ; \; R = %.3f$' % (sigma, sp_R), xy = (0.65, 0.05), xycoords = 'axes fraction',)

		ax.set_xlabel('$ fit \; : \; \\lg \, (M_{\\ast} ) $')
		ax.set_ylabel('$ data \; : \; \\lg \, (M_{\\ast} ) $')
		plt.savefig('/home/xkchen/lgM_diag_selected_%d.png' % kk, dpi = 300)
		plt.close()

		##.
		lg_L_arr = lg_L_arr[ id_R_lim == False ]
		lg_M_arr = lg_M_arr[ id_R_lim == False ]

		kk += 1

		#.. save array use to product the fitting relation
		out_arr = np.array([ lg_M_arr, lg_L_arr ]).T
		np.savetxt( out_path + 'M_Lr_fit_selected_points.txt', out_arr, fmt = '%.8f, %.8f',)

	return

# Lx2M_func_point_select()


##. fitting formula for mass estimation
points = np.loadtxt( out_path + 'M_Lr_fit_selected_points.txt', delimiter = ',')

lg_M_arr = points[:,0]
lg_L_arr = points[:,1]

#.
p0 = [ 0.5, 0.02 ]

res_lsq = optimize.least_squares( resi_M2L_func, x0 = np.array( p0 ), loss = 'cauchy', 
						f_scale = 0.1, args = ( lg_L_arr, lg_M_arr), )

a_fit = res_lsq.x[0]
b_fit = res_lsq.x[1]

#.
keys = [ 'a', 'b' ]
values = [ a_fit, b_fit ]
fill = dict( zip( keys, values) )
out_data = pds.DataFrame( fill, index = ['k', 'v'])
out_data.to_csv( out_path + 'least-square_M-to-Lr_params.csv',)


##..
lgL_arr = np.log10( r_Lumi )
lgM_star = lg_Mstar + 0.

fit_M = lg_M_func( lgL_arr, a_fit, b_fit )

Var = np.sum( (fit_M - lgM_star)**2 ) / len( lgM_star )
sigma = np.sqrt( Var )
sp_R = sts.spearmanr( fit_M, lgM_star)[0]


##.
bin_x = np.linspace(11.1, 12.3, 7)
bin_x = np.r_[ 10.8, bin_x ]

bin_cen = 0.5 * ( bin_x[1:] + bin_x[:-1])
bin_medi, bin_std = np.array([]), np.array([])
bin_Ng = np.array([])

for oo in range( len(bin_x) - 1):
	id_in = ( fit_M >= bin_x[oo] ) & ( fit_M <= bin_x[oo+1] )
	sub_lgM = lgM_star[id_in]

	bin_medi = np.r_[ bin_medi, np.median( sub_lgM ) ]
	bin_std = np.r_[ bin_std, np.std( sub_lgM ) ]
	bin_Ng = np.r_[ bin_Ng, np.sum(id_in) ]

#.
plt.figure()
ax = plt.subplot(111)
ax.set_title('least square')

ax.scatter( fit_M, lgM_star, marker = 'o', s = 1.5, color = 'Gray', alpha = 0.15, zorder = 100)
ax.plot( fit_M, fit_M, 'k-', alpha = 0.75,)

ax.errorbar( bin_cen[bin_Ng > 1], bin_medi[bin_Ng > 1], yerr = bin_std[bin_Ng > 1], color = 'k', marker = 'o', capsize = 2.5, ls = '')

ax.annotate( s = '$\\sigma = %.3f\, ; \; R = %.3f$' % (sigma, sp_R), xy = (0.65, 0.05), xycoords = 'axes fraction',)
ax.annotate( s = 'a = %.3f' % a_fit + '\n' + 'b = %.3f' % b_fit, xy = (0.10, 0.75), xycoords = 'axes fraction',)

ax.set_xlabel('$ \\lg \, (M_{\\ast}) = a + b \\cdot \\lg \,(L_{r})$',)

ax.set_ylabel('$ \\lg \, (M_{\\ast}) $' )
ax.set_xlim(10.80, 12.25)
ax.set_ylim(10.50, 12.25)
plt.savefig('/home/xkchen/M-to-Lr_estimate.png', dpi = 300)
plt.close()

