import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
h = H0 / 100
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

### === data read and fitting the relation
cat_path = '/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/'

dat = pds.read_csv( cat_path + 'Extend_BCGM_bin_dered-mag_Pcen_cat.csv')

gr_arr = np.array( dat['dered_g-r'] )
ri_arr = np.array( dat['dered_r-i'] )

lg_Mstar = np.array( dat['lg_Mstar'] )   # unit : M_sun / h^2
lg_Mstar = lg_Mstar - 2 * np.log10( h )

cc_dat = pds.read_csv( cat_path + 'Extend_BCGM_bin_cat_non-dered_color_cModemag.csv')

ra, dec = np.array( cc_dat['ra'] ), np.array( cc_dat['dec'] )
p_IDs = np.array( cc_dat['objID'] )

z = np.array( cc_dat['z'] )
cMod_mag_i = np.array( cc_dat['cMod_mag_i'] )

Dl_z = Test_model.luminosity_distance( z ).value

i_cMag = cMod_mag_i - 5 * np.log10( Dl_z * 1e6 ) + 5
i_Lumi = 10**( -0.4 * ( i_cMag - Mag_sun[2] ) )

##... save for figure
# keys = ['ra', 'dec', 'z', 'objID', 'lg_Mstar_no-h', 'g-r', 'r-i', 'i_cmag', 'i_cMag', 'lg_Li']
# values = [ ra, dec,  z, p_IDs, lg_Mstar, gr_arr, ri_arr, cMod_mag_i, i_cMag, np.log10( i_Lumi ) ]
# fill = dict( zip( keys, values ) )
# data = pds.DataFrame( fill )
# data.to_csv( '/home/xkchen/figs/extend_bcgM_cat/Mass_Li_fit/' + 'least-square_fit_cat.csv',)


##... selecte clusters and fitting the relation between Mstar and color
def points_select():

	out_path = '/home/xkchen/figs/extend_bcgM_cat/Mass_Li_fit/'

	based_str = 'i'

	id_R_lim = [ 1 ]
	sum_dex = np.sum( id_R_lim )
	kk = 0

	cp_gr_arr = gr_arr + 0.
	cp_ri_arr = ri_arr + 0.
	cp_lgLi_arr = np.log10( i_Lumi )
	cp_lg_Mstar = lg_Mstar + 0.

	sigma = 5
	kk = 0

	while sum_dex > 0:

		put_x = np.array([ cp_gr_arr, cp_ri_arr, cp_lgLi_arr ])

		p0 = [ -0.05, 0.5, 0.02, 0.2 ]
		res_lsq = optimize.least_squares( resi_bi_line_func, x0 = np.array( p0 ), loss = 'cauchy', 
			f_scale = 0.1, args = ( put_x, cp_lg_Mstar),)

		a_fit = res_lsq.x[0]
		b_fit = res_lsq.x[1]
		c_fit = res_lsq.x[2]
		d_fit = res_lsq.x[3]

		#... fitting lg_Mstar
		fit_M2L = lg_bi_linear_func( put_x, a_fit, b_fit, c_fit, d_fit)

		Var = np.sum( (fit_M2L - cp_lg_Mstar)**2 ) / len( cp_lg_Mstar )
		sigma = np.sqrt( Var )
		sp_R = sts.spearmanr( fit_M2L, cp_lg_Mstar)[0]

		#... fit relation between fitting and obs.
		po = [0.9, 10]
		popt, pcov = optimize.curve_fit( lg_linea_func, xdata = fit_M2L, ydata = cp_lg_Mstar, p0 = po,)
		_a0, _b0 = popt
		com_line = lg_linea_func( fit_M2L, _a0, _b0 )

		dR_com_l = np.abs( _b0 * fit_M2L + _a0 - cp_lg_Mstar) / np.sqrt(1 + _b0**2)
		id_R_lim = dR_com_l >= 2 * sigma

		sum_dex = np.sum( id_R_lim )

		plt.figure()
		ax = plt.subplot(111)
		ax.set_title('least square')
		ax.scatter( fit_M2L, cp_lg_Mstar, marker = 'o', s = 1.5, color = 'k', alpha = 0.15, zorder = 100)
		ax.plot( fit_M2L, fit_M2L, 'b--',)
		ax.scatter( fit_M2L[id_R_lim], cp_lg_Mstar[id_R_lim], marker = '*', s = 5.5, color = 'g',)
		ax.annotate( text = '$\\sigma = %.3f\, ; \; R = %.3f$' % (sigma, sp_R), xy = (0.65, 0.05), 
			xycoords = 'axes fraction',)
		ax.set_xlabel('$ fit \; : \; \\lg \, (M_{\\ast} ) $')
		ax.set_ylabel('$ data \; : \; \\lg \, (M_{\\ast} ) $')
		plt.savefig('/home/xkchen/lgM_diag_selected_%d.png' % kk, dpi = 300)
		plt.close()

		cp_gr_arr = cp_gr_arr[ id_R_lim == False ]
		cp_ri_arr = cp_ri_arr[ id_R_lim == False ]
		cp_lgLi_arr = cp_lgLi_arr[ id_R_lim == False ]
		cp_lg_Mstar = cp_lg_Mstar[ id_R_lim == False ]

		kk += 1

		#.. save array use to product the fitting relation
		#.. including g-r, r-i, lg_Mstar (units of M_sun), and i-band Luminosity lg_Li
		out_arr = np.array([ cp_gr_arr, cp_ri_arr, cp_lg_Mstar, cp_lgLi_arr ]).T
		np.savetxt( out_path + 'M2L%s_selected_points.txt' % based_str, out_arr, fmt = '%.8f, %.8f, %.8f, %.8f',)

# points_select()

## ...
out_path = '/home/xkchen/figs/extend_bcgM_cat/Mass_Li_fit/'

based_str = 'i'
points = np.loadtxt( out_path + 'M2L%s_selected_points.txt' % based_str, delimiter = ',')
cp_gr_arr = points[:,0]
cp_ri_arr = points[:,1]
cp_lg_Mstar = points[:,2]
cp_lg_Lumi = points[:,3]

put_x = np.array([ cp_gr_arr, cp_ri_arr, cp_lg_Lumi ])
p0 = [ -0.05, 0.5, 0.02, 0.2 ]
res_lsq = optimize.least_squares( resi_bi_line_func, x0 = np.array( p0 ), loss = 'cauchy', 
	f_scale = 0.1, args = ( put_x, cp_lg_Mstar),)

a_fit = res_lsq.x[0]
b_fit = res_lsq.x[1]
c_fit = res_lsq.x[2]
d_fit = res_lsq.x[3]

keys = ['a', 'b', 'c', 'd']
values = [ a_fit, b_fit, c_fit, d_fit ]
fill = dict( zip( keys, values) )
out_data = pds.DataFrame( fill, index = ['k', 'v'])
out_data.to_csv( out_path + 'least-square_M-to-%s-band-Lumi&color.csv' % based_str )

_lg_Lumi = np.log10( i_Lumi )
put_x = np.array( [ gr_arr, ri_arr, _lg_Lumi ] )

#... fitting lg_Mstar
fit_M2L = lg_bi_linear_func( put_x, a_fit, b_fit, c_fit, d_fit)

_lg_M = lg_Mstar + 0.
Var = np.sum( (fit_M2L - _lg_M)**2 ) / len( _lg_M )
sigma = np.sqrt( Var )
sp_R = sts.spearmanr( fit_M2L, _lg_M)[0]


bin_x = np.linspace(11.1, 12.3, 7)
bin_x = np.r_[ 10.8, bin_x ]

bin_cen = 0.5 * ( bin_x[1:] + bin_x[:-1])
bin_medi, bin_std = np.array([]), np.array([])
bin_Ng = np.array([])

for oo in range( len(bin_x) - 1):
	id_in = ( fit_M2L >= bin_x[oo] ) & ( fit_M2L <= bin_x[oo+1] )
	sub_lgM = _lg_M[id_in]

	bin_medi = np.r_[ bin_medi, np.median( sub_lgM ) ]
	bin_std = np.r_[ bin_std, np.std( sub_lgM ) ]
	bin_Ng = np.r_[ bin_Ng, np.sum(id_in) ]

plt.figure()
ax = plt.subplot(111)
ax.set_title('least square')

ax.scatter( fit_M2L, _lg_M, marker = 'o', s = 1.5, color = 'Gray', alpha = 0.15, zorder = 100)
ax.plot( fit_M2L, fit_M2L, 'k-', alpha = 0.75,)

ax.errorbar( bin_cen[bin_Ng > 1], bin_medi[bin_Ng > 1], yerr = bin_std[bin_Ng > 1], color = 'k', marker = 'o', capsize = 2.5, ls = '')

ax.annotate( text = '$\\sigma = %.3f\, ; \; R = %.3f$' % (sigma, sp_R), xy = (0.65, 0.05), xycoords = 'axes fraction',)
ax.annotate( text = 'a = %.3f' % a_fit + '\n' + 'b = %.3f' % b_fit + '\n' + 'c = %.3f' % c_fit + 
	'\n' + 'd = %.3f' % d_fit, xy = (0.10, 0.75), xycoords = 'axes fraction',)
ax.set_xlabel(
	'$ \\lg \, (M_{\\ast}) = a \\cdot (g-r) + b \\cdot (r-i) + c \\cdot \\lg(L_{%s}) + d $' % (based_str),)
ax.set_ylabel('$ \\lg \, (M_{\\ast}) $' )
ax.set_xlim(10.80, 12.25)
ax.set_ylim(10.50, 12.25)
plt.savefig('/home/xkchen/M-to-L_estimate.png', dpi = 300)
plt.close()

