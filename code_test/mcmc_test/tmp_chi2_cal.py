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

### === ### calculation compare
# A = np.array( [ [1, 2], [4, 5] ]) # cov

# A_inv = np.linalg.inv(A) # cov_inv
# print( A_inv )

# y0 = np.array( [2, 4] )

# y_obs = np.zeros( (6, 2) )

# for kk in range( 6 ):
# 	np.random.seed( kk )
# 	y_obs[kk, 0] = np.random.randn() * 0.1 + 2
# 	y_obs[kk, 1] = np.random.randn() * 0.1 + 4

# delta_1 = y_obs[0,:] - y0

# chi2_1 = delta_1.T.dot( A_inv ).dot( delta_1 ) # use now
# print( chi2_1 )

# chi2_2 = delta_1.reshape(1,2).dot( A_inv ).dot( delta_1 )
# print( chi2_2 / chi2_1 )

# chi2_3 = delta_1.reshape(2,1).T.dot( A_inv ).dot( delta_1 )
# print( chi2_3 / chi2_1 )

# ### 
# pa0 = delta_1[0] * A_inv[0,0] + delta_1[1] * A_inv[1,0]
# pa1 = delta_1[0] * A_inv[0,1] + delta_1[1] * A_inv[1,1]
# pa = np.array([ pa0, pa1 ])
# chi2_def = pa[0] * delta_1[0] + pa[1] * delta_1[1]
# print( chi2_def / chi2_1 )

### === ###
path = '/home/xkchen/mywork/ICL/code/photo_z_match_SB/'
out_path = '/home/xkchen/mywork/ICL/code/photo_z_match_BG_pros/'

# path = '/home/xkchen/mywork/ICL/code/00_jk_number_test/SB_pros/'
# out_path = '/home/xkchen/mywork/ICL/code/00_jk_number_test/'

low_r, low_sb, low_err = [], [], [] 
for ii in range( 3 ):
	with h5py.File( path + 'photo-z_match_low_BCG_star-Mass_%s-band_Mean_jack_SB-pro_z-ref.h5' % band[ii], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	low_r.append( tt_r )
	low_sb.append( tt_sb )
	low_err.append( tt_err )

hi_r, hi_sb, hi_err = [], [], []
for ii in range( 3 ):
	with h5py.File( path + 'photo-z_match_high_BCG_star-Mass_%s-band_Mean_jack_SB-pro_z-ref.h5' % band[ii], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	hi_r.append( tt_r )
	hi_sb.append( tt_sb )
	hi_err.append( tt_err )

## ... labels
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass', 'total_star-Mass']
fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$', 'low $M_{\\ast}$ + high $M_{\\ast}$']

color_s = ['r', 'g', 'b']

R_tt = np.arange(200, 800, 50)
R_r = 1.4e3

mass_dex = 0 # 0, 1

for kk in range( 3 ):

	tmp_chi2_diag, tmp_chi2_cov = [], []
	tmp_chi2_lim_diag, tmp_chi2_lim_cov = [], []
	tmp_SNR = []

	for ii in range( len(R_tt) ):

		R_low = R_tt[ ii ]	

		if mass_dex == 0:
			idmx = low_r[kk] >= 10
			com_r = low_r[kk][idmx]
			com_sb = low_sb[kk][idmx]
			com_err = low_err[kk][idmx]

			with h5py.File(out_path + 'cov_arr/low_BCG-M-star_%s-band_10-kpc-out_cov-cor_arr_before_BG-sub.h5' % band[kk], 'r') as f:
			#with h5py.File(out_path + 'low_BCG-M-star_%s-band_10-kpc-out_cov-cor_arr_before_BG-sub.h5' % band[kk], 'r') as f:
				cov_MX = np.array( f['cov_Mx'] )
				cor_MX = np.array( f['cor_Mx'] )
				R_mean = np.array( f['R_kpc'] )

		elif mass_dex == 1:
			idmx = hi_r[kk] >= 10
			com_r = hi_r[kk][idmx]
			com_sb = hi_sb[kk][idmx]
			com_err = hi_err[kk][idmx]

			with h5py.File(out_path + 'cov_arr/high_BCG-M-star_%s-band_10-kpc-out_cov-cor_arr_before_BG-sub.h5' % band[kk], 'r') as f:
			#with h5py.File(out_path + 'high_BCG-M-star_%s-band_10-kpc-out_cov-cor_arr_before_BG-sub.h5' % band[kk], 'r') as f:
				cov_MX = np.array( f['cov_Mx'] )
				cor_MX = np.array( f['cor_Mx'] )
				R_mean = np.array( f['R_kpc'] )

		## read params of random point SB profile
		p_dat = pds.read_csv( out_path + 'BG_estimate/%s-band_random_SB_fit_params.csv' % band[kk],)
		#p_dat = pds.read_csv( '/home/xkchen/mywork/ICL/code/photo_z_match_BG_pros/' + '%s-band_random_SB_fit_params.csv' % band[kk],)
		( e_a, e_b, e_x0, e_A, e_alpha, e_B ) = ( np.array(p_dat['e_a'])[0], np.array(p_dat['e_b'])[0], np.array(p_dat['e_x0'])[0], 
												np.array(p_dat['e_A'])[0], np.array(p_dat['e_alpha'])[0], np.array(p_dat['e_B'])[0],)

		idx1 = (com_r >= R_low)
		fx = com_r[idx1]
		fy = com_sb[idx1]
		ferr = com_err[idx1]

		params = np.array( [ e_a, e_b, e_x0, e_A, e_alpha, e_B, cov_MX ] )

		po = [ 2e-4, 4.8e-4, 6.8e2 ]
		bonds = [ [0, 1e-2], [0, 1e1], [2e2, 3e3] ]
		E_return = optimize.minimize(err_fit_func, x0 = np.array(po), args = (fx, fy, params, ferr), method = 'L-BFGS-B', bounds = bonds,)
		popt = E_return.x
		offD, I_e, R_e = popt

		print(E_return)
		print(popt)

		cov_inv = np.linalg.pinv( cov_MX )
		params = np.array([e_a, e_b, e_x0, e_A, e_alpha, e_B])
		sb_trunk = sersic_func( 2e3, I_e, R_e )

		idx2 = (com_r >= R_low) & (com_r <= R_r)

		fit_line = cc_rand_sb_func( com_r, e_a, e_b, e_x0, e_A, e_alpha, e_B) + sersic_func( com_r, I_e, R_e ) - offD
		mode_sign = sersic_func( com_r, I_e, R_e ) - sb_trunk
		BG_line = cc_rand_sb_func( com_r, e_a, e_b, e_x0, e_A, e_alpha, e_B) - offD + sb_trunk

		## chi2 based on diagonal err only
		delta = fit_line[idx1] - fy
		chi2_diag = np.sum( delta**2 / ferr**2 ) / ( len(fy) - len(popt) )
		chi2_lim_diag = np.sum( (fit_line[idx2] - com_sb[idx2])**2 / com_err[idx2]**2 ) / ( np.sum(idx2) - len(popt) )

		## chi2 with full covariance matrix
		lis_index = np.where( idx1 )[0]
		left_index, right_index = lis_index[0], lis_index[-1]

		cut_cov_inv = cov_inv[ left_index: right_index + 1, left_index: right_index + 1 ]
		chi2_cov = delta.T.dot( cut_cov_inv ).dot( delta ) / ( len(fy) - len(popt) )

		lis_index = np.where( idx2 )[0]
		left_index, right_index = lis_index[0], lis_index[-1]
		cut_cov_inv = cov_inv[ left_index: right_index + 1, left_index: right_index + 1 ]

		cut_delta = fit_line[idx2] - com_sb[idx2]
		chi2_lim_cov = cut_delta.T.dot( cut_cov_inv ).dot( cut_delta ) / ( np.sum(idx2) - len(popt) )

		tmp_SNR.append( com_sb / com_err )

		tmp_chi2_diag.append( chi2_diag )
		tmp_chi2_lim_diag.append( chi2_lim_diag )

		tmp_chi2_cov.append( chi2_cov )
		tmp_chi2_lim_cov.append( chi2_lim_cov )

	idvx = R_mean >= R_tt[0]
	lim_dex = np.where( idvx )[0]
	cut_R = R_mean[ idvx ]
	cut_cor = cor_MX[ lim_dex[0]: lim_dex[-1]+1, lim_dex[0]: lim_dex[-1]+1 ]

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title( fig_name[ mass_dex] + ',%s band SNR(r)' % band[kk] )

	ax.plot( R_tt, np.array( tmp_chi2_cov), ls = '-', color = 'r', alpha = 0.5, label = 'points beyond R',)
	ax.plot( R_tt, np.array( tmp_chi2_lim_cov), ls = '--', color = 'r', alpha = 0.5, label = 'points between R and 1.4Mpc',)

	ax.set_yscale( 'log' )
	ax.set_xlabel('R [kpc]')
	ax.set_ylabel('$ \\chi^2_{cov} / \\nu $')
	ax.grid(which = 'both', axis = 'both', alpha = 0.25,)
	ax.legend( loc = 4,)

	plt.subplots_adjust( left = 0.15 )
	plt.savefig('/home/xkchen/figs/%s_%s-band_chi2_range.jpg' % (cat_lis[mass_dex], band[kk]), dpi = 300)
	plt.close()


	fig = plt.figure( figsize = (13.12, 4.8) )
	ax0 = fig.add_axes([0.09, 0.10, 0.31, 0.80])
	ax1 = fig.add_axes([0.50, 0.10, 0.40, 0.80])

	ax0.set_title( fig_name[ mass_dex] + ',%s band' % band[kk] )
	ax0.plot( R_tt, np.array( tmp_chi2_cov) / np.array( tmp_chi2_diag ), ls = '-', color = 'r', alpha = 0.5, label = 'points beyond R',)
	ax0.plot( R_tt, np.array( tmp_chi2_lim_cov) / np.array( tmp_chi2_lim_diag ), ls = '--', color = 'r', alpha = 0.5, label = 'points between R and 1.4Mpc',)
	ax0.legend( loc = 2)
	ax0.set_xlabel( 'inner radius(R)[kpc]' )
	ax0.set_ylabel( '$ \\frac {\\chi^{2}_{cov} } {\\nu} \; / \; \\frac { \\chi^2_{diag} } {\\nu} $' )

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

