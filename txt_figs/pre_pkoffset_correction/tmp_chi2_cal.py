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
from astropy.coordinates import SkyCoord
from scipy import optimize

import corner
import emcee

from img_random_SB_fit import cc_rand_sb_func

# cosmology model
vc = C.c.to(U.km/U.s).value
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
DH = vc/H0

# constant
kpc2cm = U.kpc.to(U.cm)
Mpc2cm = U.Mpc.to(U.cm)

rad2arcsec = U.rad.to(U.arcsec)
Lsun = C.L_sun.value*10**7 # (erg/s/cm^2)
Jy = 10**(-23) # (erg/s)/cm^2/Hz
F0 = 3.631 * 10**(-6) * Jy
L_speed = C.c.value # m/s

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']
L_wave = np.array([ 6166, 4686, 7480 ])

## SB model for cluster component (in sersic formula, fixed sersic index)
def sersic_func(r, Ie, re):
	ndex = 2.1 # Zhang et a., 2019, for large scale, n~2.1
	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

def err_fit_func(p, x, y, params, yerr):

	a, b, x0, A, alpha, B, cov_mx = params 
	pf0 = cc_rand_sb_func(x, a, b, x0, A, alpha, B)

	d_off, I_e, R_e = p[:]
	pf1 = sersic_func(x, I_e, R_e)
	pf = pf0 + pf1 - d_off

	cov_inv = np.linalg.pinv( cov_mx )
	delta = pf - y

	return np.sum( delta**2 / yerr**2 )

def chi2_test():
	### === ### calculation compare
	A = np.array( [ [1, 2], [4, 5] ]) # cov

	A_inv = np.linalg.inv(A) # cov_inv
	print( A_inv )

	y0 = np.array( [2, 4] )

	y_obs = np.zeros( (6, 2) )

	for kk in range( 6 ):
		np.random.seed( kk )
		y_obs[kk, 0] = np.random.randn() * 0.1 + 2
		y_obs[kk, 1] = np.random.randn() * 0.1 + 4

	delta_1 = y_obs[0,:] - y0

	chi2_1 = delta_1.T.dot( A_inv ).dot( delta_1 ) # use now
	print( chi2_1 )

	chi2_2 = delta_1.reshape(1,2).dot( A_inv ).dot( delta_1 )
	print( chi2_2 / chi2_1 )

	chi2_3 = delta_1.reshape(2,1).T.dot( A_inv ).dot( delta_1 )
	print( chi2_3 / chi2_1 )

	### 
	pa0 = delta_1[0] * A_inv[0,0] + delta_1[1] * A_inv[1,0]
	pa1 = delta_1[0] * A_inv[0,1] + delta_1[1] * A_inv[1,1]
	pa = np.array([ pa0, pa1 ])
	chi2_def = pa[0] * delta_1[0] + pa[1] * delta_1[1]
	print( chi2_def / chi2_1 )

### === ###
color_s = ['r', 'g', 'b']

path = '/home/xkchen/mywork/ICL/code/rig_common_cat/mass_bin/'
BG_path = '/home/xkchen/mywork/ICL/code/rig_common_cat/mass_bin_BG/'

cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$']

# path = '/home/xkchen/mywork/ICL/code/rig_common_cat/age_bin/'
# BG_path = '/home/xkchen/mywork/ICL/code/rig_common_cat/age_bin_BG/'

# fig_name = [ 'younger', 'older' ]
# cat_lis = [ 'younger', 'older' ]

cov_path = '/home/xkchen/mywork/ICL/code/rig_common_cat/cov_arr/'
rand_path = '/home/xkchen/mywork/ICL/code/ref_BG_profile/'

out_path = '/home/xkchen/figs/'

## ...
for mass_dex in (0, 1):

	for kk in range( 3 ):

		R_low = 200 # betond 200 kpc
		R_up = 1.4e3 # around point at which SB start to increase again

		with h5py.File( path + 'photo-z_match_gri-common_%s_%s-band_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mass_dex], band[kk]), 'r') as f:
			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		idmx = tt_r >= 10
		com_r = tt_r[idmx]
		com_sb = tt_sb[idmx]
		com_err = tt_err[idmx]

		out_file = cov_path + 'photo-z_%s_%s-band_cov-cor_arr.h5' % ( cat_lis[mass_dex], band[kk] )
		with h5py.File( out_file, 'r') as f:
			cov_MX = np.array( f['cov_Mx'])
			cor_MX = np.array( f['cor_Mx'])
			R_mean = np.array( f['R_kpc'])

		cov_inv = np.linalg.pinv( cov_MX )

		## read params of random point SB profile
		p_dat = pds.read_csv( rand_path + '%s-band_random_SB_fit_params.csv' % band[kk] )
		( e_a, e_b, e_x0, e_A, e_alpha, e_B ) = ( np.array(p_dat['e_a'])[0], np.array(p_dat['e_b'])[0], np.array(p_dat['e_x0'])[0], 
												np.array(p_dat['e_A'])[0], np.array(p_dat['e_alpha'])[0], np.array(p_dat['e_B'])[0],)

		idx1 = com_r >= R_low
		fx = com_r[idx1]
		fy = com_sb[idx1]
		ferr = com_err[idx1]

		## for cov_arr
		id_lim = np.where( idx1 == True )[0][0]
		lim_cov = cov_MX[ id_lim:, id_lim: ]
		lim_cov_inv = cov_inv[ id_lim:, id_lim: ]

		tmp_chi2_diag, tmp_chi2_cov = [], []
		tmp_chi2_lim_diag, tmp_chi2_lim_cov = [], []

		nr = len( fx )

		for jj in range( nr ):

			cut_cov_0 = np.delete( lim_cov, jj, axis = 0)
			cut_cov = np.delete( cut_cov_0, jj, axis = 1)

			fit_r, fit_sb, fit_err = np.delete( fx, jj), np.delete( fy, jj), np.delete( ferr, jj)

			params = np.array( [ e_a, e_b, e_x0, e_A, e_alpha, e_B, cut_cov ] )

			po = [ 2e-4, 4.8e-4, 6.8e2 ]
			bonds = [ [0, 1e-2], [0, 1e1], [2e2, 3e3] ]
			E_return = optimize.minimize( err_fit_func, x0 = np.array(po), args = ( fit_r, fit_sb, params, fit_err), 
				method = 'L-BFGS-B', bounds = bonds,)
			popt = E_return.x
			offD, I_e, R_e = popt

			# print(E_return)
			# print(popt)

			_cov_inv_0 = np.delete( lim_cov_inv, jj, axis = 0)
			_cov_inv = np.delete( _cov_inv_0, jj, axis = 1)

			params = np.array([e_a, e_b, e_x0, e_A, e_alpha, e_B])
			sb_trunk = sersic_func( 2e3, I_e, R_e )

			idx2 = fit_r <= R_up

			fit_line = cc_rand_sb_func( fit_r, e_a, e_b, e_x0, e_A, e_alpha, e_B) + sersic_func( fit_r, I_e, R_e ) - offD
			mode_sign = sersic_func( fit_r, I_e, R_e ) - sb_trunk
			BG_line = cc_rand_sb_func( fit_r, e_a, e_b, e_x0, e_A, e_alpha, e_B) - offD + sb_trunk

			## chi2 based on diagonal err only
			delta = fit_line - fit_sb
			chi2_diag = np.sum( delta**2 / fit_err**2 ) / ( len(fit_sb) - len(popt) )
			chi2_lim_diag = np.sum( (fit_line[idx2] - fit_sb[idx2])**2 / fit_err[idx2]**2 ) / ( len(fit_sb[idx2]) - len(popt) )

			## chi2 with covariance matrix
			chi2_cov = delta.T.dot( _cov_inv ).dot( delta ) / ( len(fit_sb) - len(popt) )

			lis_index = np.where( idx2 )[0][-1]
			_lim_cov_inv = _cov_inv[ :lis_index + 1, :lis_index + 1 ]

			cut_delta = fit_line[idx2] - fit_sb[idx2]
			chi2_lim_cov = cut_delta.T.dot( _lim_cov_inv ).dot( cut_delta ) / ( np.sum(idx2) - len(popt) )

			tmp_chi2_diag.append( chi2_diag )
			tmp_chi2_lim_diag.append( chi2_lim_diag )

			tmp_chi2_cov.append( chi2_cov )
			tmp_chi2_lim_cov.append( chi2_lim_cov )

		idvx = R_mean >= R_low
		lim_dex = np.where( idvx )[0][0]
		cut_R = R_mean[ idvx ]
		cut_cor = cor_MX[ lim_dex:, lim_dex: ]

		## save chi2_arr
		keys = ['R', 'chi2_diag', 'chi2_cov']
		values = [ fx, np.array(tmp_chi2_diag), np.array(tmp_chi2_cov) ]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_path + 'photo-z_%s_%s-band_BG-estimate-fit_chi2_compare.csv' % (cat_lis[mass_dex], band[kk]),)

		# figs
		plt.figure()
		ax = plt.subplot(111)
		ax.set_title( fig_name[ mass_dex] + ',%s band' % band[kk] )

		ax.plot( fx, np.array( tmp_chi2_cov), ls = '-', color = 'r', alpha = 0.5, label = 'except point at R',)
		ax.plot( fx, np.array( tmp_chi2_lim_cov), ls = '--', color = 'r', alpha = 0.5, label = 'points within 1.4Mpc',)

		ax.plot( fx, np.array( tmp_chi2_diag), ls = '-', color = 'k', alpha = 0.5, label = '$\\chi^2_{diag}$',)
		ax.plot( fx, np.array( tmp_chi2_lim_diag), ls = '--', color = 'k', alpha = 0.5, )

		ax.set_xlim(1e2, 5e3)
		ax.set_xscale( 'log' )
		ax.set_xlabel('R [kpc]')
		ax.set_yscale( 'log' )
		ax.set_ylabel('$ \\chi^2_{cov} / \\nu $')
		ax.grid(which = 'both', axis = 'both', alpha = 0.25,)
		ax.legend( loc = 4,)

		plt.subplots_adjust( left = 0.15 )
		plt.savefig('/home/xkchen/figs/%s_%s-band_chi2_range.png' % (cat_lis[mass_dex], band[kk]), dpi = 300)
		plt.close()


		fig = plt.figure( figsize = (13.12, 4.8) )
		ax0 = fig.add_axes([0.09, 0.10, 0.31, 0.80])
		ax1 = fig.add_axes([0.50, 0.10, 0.40, 0.80])

		ax0.set_title( fig_name[ mass_dex] + ',%s band' % band[kk] )
		ax0.plot( fx, np.array( tmp_chi2_cov) / np.array( tmp_chi2_diag ), ls = '-', color = 'r', alpha = 0.5, label = 'points beyond R',)
		ax0.plot( fx, np.array( tmp_chi2_lim_cov) / np.array( tmp_chi2_lim_diag ), ls = '--', color = 'r', alpha = 0.5, label = 'points between R and 1.4Mpc',)
		
		ax0.legend( loc = 2)
		ax0.set_xlabel( 'inner radius(R)[kpc]' )
		ax0.set_ylabel( '$ \\frac {\\chi^{2}_{cov} } {\\nu} \; / \; \\frac { \\chi^2_{diag} } {\\nu} $' )

		ax0.set_xlim(1e2, 5e3)
		ax0.set_xscale( 'log' )
		ax0.set_xlabel('R [kpc]')
		ax0.set_yscale( 'log' )
		ax0.grid(which = 'both', axis = 'both', alpha = 0.25,)


		ax1.set_title(' correlation matrix')
		tf = ax1.imshow( cut_cor, origin = 'lower', cmap = 'seismic', vmin = -1, vmax = 1,)
		plt.colorbar( tf, ax = ax1, fraction = 0.035, pad = 0.01,)

		ax1.set_ylim(0, len(cut_R) - 1 )
		yticks = ax1.get_yticks( )
		tik_lis = ['%.1f' % ll for ll in cut_R[ yticks[:-1].astype( np.int ) ] ]
		ax1.set_yticks( yticks[:-1] )
		ax1.set_yticklabels( labels = tik_lis, )
		ax1.set_ylim(-0.5, len( cut_R ) - 0.5 )
		ax1.set_xticklabels( labels = [] )

		plt.savefig( '/home/xkchen/figs/%s_%s-band_chi2_compare.png' % (cat_lis[mass_dex], band[kk]), dpi = 300)
		plt.close()

