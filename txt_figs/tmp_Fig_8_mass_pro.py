import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.special as special
import astropy.units as U
import astropy.constants as C

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
from scipy import signal
from scipy import interpolate as interp
from scipy import integrate as integ

from surface_mass_density import sigmam, sigmac, input_cosm_model, cosmos_param, rhom_set

### === ### cosmology
rad2asec = U.rad.to(U.arcsec)

Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)

H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

pixel = 0.396
band = ['r', 'g', 'i']
L_wave = np.array([ 6166, 4686, 7480 ])

### === ### initial surface_mass_density.py module
input_cosm_model( get_model = Test_model )
cosmos_param()

### === ### miscentering nfw profile (Zu et al. 2020, section 3.)
def mis_p_func( r_off, sigma_off):
	"""
	r_off : the offset between cluster center and BCGs
	sigma_off : characteristic offset
	"""

	pf0 = r_off / sigma_off**2
	pf1 = np.exp( - r_off / sigma_off )

	return pf0 * pf1

def misNFW_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m):

	theta = np.linspace( 0, 2 * np.pi, 100)
	d_theta = np.diff( theta )
	N_theta = len( theta )

	try:
		NR = len( rp )
	except:
		rp = np.array( [rp] )
		NR = len( rp )

	r_off = np.arange( 0, 15 * sigma_off, 0.02 * sigma_off )
	off_pdf = mis_p_func( r_off, sigma_off )
	dr_off = np.diff( r_off )

	NR_off = len( r_off )

	surf_dens_off = np.zeros( NR, dtype = np.float32 )

	for ii in range( NR ):

		surf_dens_arr = np.zeros( (NR_off, N_theta), dtype = np.float32 )

		for jj in range( NR_off ):

			r_cir = np.sqrt( rp[ii]**2 + 2 * rp[ii] * r_off[jj] * np.cos( theta ) + r_off[jj]**2 )
			surf_dens_arr[jj,:] = sigmam( r_cir, lgM, z, c_mass,)

		## integration on theta
		medi_surf_dens = ( surf_dens_arr[:,1:] + surf_dens_arr[:,:-1] ) / 2
		sum_theta_fdens = np.sum( medi_surf_dens * d_theta, axis = 1) / ( 2 * np.pi )

		## integration on r_off
		integ_f = sum_theta_fdens * off_pdf

		medi_integ_f = ( integ_f[1:] + integ_f[:-1] ) / 2

		surf_dens_ii = np.sum( medi_integ_f * dr_off )

		surf_dens_off[ ii ] = surf_dens_ii

	off_sigma = surf_dens_off

	if NR == 1:
		return off_sigma[0]
	return off_sigma

def obs_sigma_func( rp, f_off, sigma_off, z, c_mass, lgM, v_m):

	off_sigma = misNFW_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m)
	norm_sigma = sigmam( rp, lgM, z, c_mass)

	obs_sigma = f_off * off_sigma + ( 1 - f_off ) * norm_sigma

	return obs_sigma

def err_fit_func(p, x, y, params, yerr):

	bf = p[0]

	cov_mx, lg_sigma = params[:]

	_mass_out = 10**lg_sigma
	_sum_mass = np.log10( _mass_out * 10**bf )

	delta = _sum_mass - y
	cov_inv = np.linalg.pinv( cov_mx )
	# chi2 = delta.T.dot( cov_inv ).dot(delta)
	chi2 = np.sum( delta**2 / yerr**2 )

	if np.isfinite( chi2 ):
		return chi2
	return np.inf

### === load data
z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value
a_ref = 1 / (z_ref + 1)

## ... satellite number density
bin_R, siglow, errsiglow, sighig, errsighig, highoverlow, errhighoverlow = np.genfromtxt('/home/xkchen/tmp_run/data_files/figs/result_high_over_low.txt', unpack = True)
bin_R = bin_R * 1e3 * a_ref / h
siglow, errsiglow, sighig, errsighig = np.array( [ siglow * h**2 / 1e6, errsiglow * h**2 / 1e6, sighig * h**2 / 1e6, errsighig * h**2 / 1e6 ] ) / a_ref**2

id_nan = np.isnan( bin_R )
bin_R = bin_R[ id_nan == False]
siglow, errsiglow, sighig, errsighig = siglow[ id_nan == False], errsiglow[ id_nan == False], sighig[ id_nan == False], errsighig[ id_nan == False]

lo_Ng_int_F = interp.interp1d( bin_R, siglow, kind = 'linear', fill_value = 'extrapolate',)
hi_Ng_int_F = interp.interp1d( bin_R, sighig, kind = 'linear', fill_value = 'extrapolate',)


## ... DM mass profile
rho_c, rho_m = rhom_set( 0 ) # in unit of M_sun * h^2 / kpc^3

lo_xi_file = '/home/xkchen/tmp_run/data_files/figs/low_BCG_M_xi-rp.txt'
hi_xi_file = '/home/xkchen/tmp_run/data_files/figs/high_BCG_M_xi-rp.txt'

lo_dat = np.loadtxt( lo_xi_file )
lo_rp, lo_xi = lo_dat[:,0], lo_dat[:,1]
lo_rho_m = ( lo_xi * 1e3 * rho_m ) * h / a_ref**2
lo_rp = lo_rp * 1e3 * a_ref / h

hi_dat = np.loadtxt( hi_xi_file )
hi_rp, hi_xi = hi_dat[:,0], hi_dat[:,1]
hi_rho_m = ( hi_xi * 1e3 * rho_m ) * h / a_ref**2
hi_rp = hi_rp * 1e3 * a_ref / h

lo_interp_F = interp.interp1d( lo_rp, lo_rho_m, kind = 'cubic',)
hi_interp_F = interp.interp1d( hi_rp, hi_rho_m, kind = 'cubic',)

lo_xi2M_2Mpc = lo_interp_F( 2e3 )
hi_xi2M_2Mpc = hi_interp_F( 2e3 )


##... SM(r)
BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/BCG_M_bin/BGs/'
path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/BCG_M_bin/SBs/'

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']

color_s = ['r', 'g', 'b']
line_c = ['b', 'r']
mark_s = ['s', 'o']

## miscen params for high mass
v_m = 200 # rho_mean = 200 * rho_c * omega_m
c_mass = [5.87, 6.95]
Mh0 = [14.24, 14.24]
off_set = [230, 210] # in unit kpc / h
f_off = [0.37, 0.20]

fit_path = '/home/xkchen/tmp_run/data_files/figs/mass_pro_fit/'
###... subsamples
"""
base_lis = ['gi', 'ri', 'gri', 'gr', 'ir', 'gir', 'ig', 'rg', 'rig']
line_name = ['g-i + $L_{i}$', 'r-i + $L_{i}$', 'g-r + $L_{i}$', 
			 'g-r + $L_{r}$', 'r-i + $L_{r}$', 'g-i + $L_{r}$', 
			 'g-i + $L_{g}$', 'g-r + $L_{g}$', 'r-i + $L_{g}$' ]

## fitting outer region
# icl_mode = 'misNFW'
# icl_mode = 'xi2M'
icl_mode = 'SG_N'

out_path = '/home/xkchen/tmp_run/data_files/figs/M2L_fit_test_M/' ## test files

for mm in range( 2 ):

	lg_fb_arr = []

	for tt in range( 2,3 ):
		band_str = base_lis[ tt ]

		# dat = pds.read_csv( BG_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str) )
		# obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['correct_surf_M'] ), np.array( dat['surf_M_err'] )

		dat = pds.read_csv( out_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str),)
		obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['surf_mass'] ), np.array( dat['surf_mass_err'] )

		id_rx = obs_R >= 9
		obs_R, surf_M, surf_M_err = obs_R[id_rx], surf_M[id_rx], surf_M_err[id_rx]

		##.. cov_arr
		# with h5py.File( BG_path + '%s_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % (cat_lis[mm], band_str), 'r') as f:
		with h5py.File( out_path + '%s_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % (cat_lis[mm], band_str), 'r') as f:
			cov_arr = np.array( f['cov_MX'] )
			cor_arr = np.array( f['cor_MX'] )

		id_cov = np.where( id_rx )[0][0]
		cov_arr = cov_arr[id_cov:, id_cov:]

		lg_M, lg_M_err = np.log10( surf_M ), surf_M_err / ( np.log(10) * surf_M )

		if icl_mode == 'misNFW':
			## .. ICL part (mis_NFW)
			misNFW_sigma = obs_sigma_func( obs_R * h, f_off[mm], off_set[mm], z_ref, c_mass[mm], Mh0[mm], v_m )
			misNFW_sigma = misNFW_sigma * h # in unit of M_sun / kpc^2
			sigma_2Mpc = obs_sigma_func( 2e3 * h, f_off[mm], off_set[mm], z_ref, c_mass[mm], Mh0[mm], v_m ) * h
			lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

		if icl_mode == 'xi2M':
			## .. use xi_hm for sigma estimation
			if mm == 0:
				xi_to_Mf = lo_interp_F

			if mm == 1:
				xi_to_Mf = hi_interp_F

			misNFW_sigma = xi_to_Mf( obs_R )
			sigma_2Mpc = xi_to_Mf( 2e3 )
			lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

		if icl_mode == 'SG_N':
			#... satellite number density
			if mm == 0:
				sig_rho_f = lo_Ng_int_F

			if mm == 1:
				sig_rho_f = hi_Ng_int_F

			misNFW_sigma = sig_rho_f( obs_R )
			sigma_2Mpc = sig_rho_f( 2e3 )
			lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

		out_lim_R = 250 # 200, 250, 300 kpc
		idx_lim = obs_R >= out_lim_R
		id_dex = np.where( idx_lim == True )[0][0]

		fit_R = obs_R[idx_lim]
		fit_M = lg_M[idx_lim]
		fit_Merr = lg_M_err[idx_lim]
		cut_cov = cov_arr[id_dex:, id_dex:]

		po_param = [ cut_cov, lg_M_sigma[idx_lim] ]

		if icl_mode != 'SG_N':
			po = -2.5
			bounds = [ [-4, -2] ]
			E_return = optimize.minimize( err_fit_func, x0 = np.array( po ), args = ( fit_R, fit_M, po_param, fit_Merr), method = 'L-BFGS-B', bounds = bounds,)

		else:
			po = 10.7
			bounds = [ [9, 11] ]
			E_return = optimize.minimize( err_fit_func, x0 = np.array( po ), args = ( fit_R, fit_M, po_param, fit_Merr), method = 'L-BFGS-B', bounds = bounds,)

		print(E_return)
		popt = E_return.x
		bf_fit = popt

		lg_fb_arr.append( bf_fit )

		_out_M = 10**lg_M_sigma * 10** bf_fit
		_sum_fit = np.log10( _out_M )
		devi_lgM = np.log10( surf_M - _out_M )

		##.. model lines
		new_R = np.logspace( 0, np.log10(2.5e3), 100)

		if icl_mode == 'misNFW':
			fit_out_M = ( obs_sigma_func( new_R * h, f_off[mm], off_set[mm], z_ref, c_mass[mm], Mh0[mm], v_m ) * h - sigma_2Mpc ) * 10** bf_fit

		if icl_mode == 'xi2M':
			fit_out_M = ( xi_to_Mf( new_R ) - sigma_2Mpc ) * 10**bf_fit

		if icl_mode == 'SG_N':
			fit_out_M = ( sig_rho_f( new_R ) - sigma_2Mpc ) * 10** bf_fit

		plt.figure()
		ax = plt.subplot(111)
		ax.set_title( fig_name[mm] + ',%s-band based' % band_str)

		ax.errorbar( obs_R, lg_M, yerr = lg_M_err, xerr = None, color = 'r', marker = '.', ls = 'none', ecolor = 'r', 
			alpha = 0.75, mec = 'r', mfc = 'r', label = 'observed')

		ax.plot( new_R, np.log10( fit_out_M ), ls = '-', color = 'b', alpha = 0.75, 
			label = 'scaled $NFW_{projected}^{miscentering}$',)

		ax.text( 1e1, 3.5, s = '$\\lgf{=}%.5f \; [R>=%dkpc]$' % (bf_fit, out_lim_R), color = 'k',)

		ax.legend( loc = 1, )
		ax.set_ylim( 3, 8.5)
		ax.set_ylabel( '$ lg \\Sigma [M_{\\odot} / kpc^2]$' )

		ax.set_xlim( 1e1, 3e3)
		ax.set_xlabel( 'R [kpc]')
		ax.set_xscale( 'log' )
		plt.savefig('/home/xkchen/%s_%s-band-based_beyond-%dkpc_fit_test.png' % (cat_lis[mm], band_str, out_lim_R), dpi = 300)
		plt.close()

	# keys = ['lg_fb_gi', 'lg_fb_ri', 'lg_fb_gri', 'lg_fb_gr', 'lg_fb_ir', 'lg_fb_gir', 'lg_fb_ig', 'lg_fb_rg', 'lg_fb_rig']
	# values = [ ]
	# for oo in range( 9 ):
	# 	values.append( lg_fb_arr[oo] )
	# fill = dict( zip( keys, values) )
	# out_data = pds.DataFrame( fill, index = ['k', 'v'])
	# out_data.to_csv( fit_path + '%s_all-color-to-M_beyond-%dkpc_%s-fit.csv' % (cat_lis[mm], out_lim_R, icl_mode),)
"""


### ... total sample
base_lis = ['gi', 'gr', 'ri']

# icl_mode = 'xi2M'
icl_mode = 'SG_N'

BG_path = '/home/xkchen/tmp_run/data_files/jupyter/total_bcgM/BGs/'
out_path = '/home/xkchen/tmp_run/data_files/figs/M2L_fit_test_M/' ## test files

lg_fb_arr = []

for tt in range( 3 ):

	band_str = base_lis[ tt ]

	# dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_corrected_aveg-jack_mass-Lumi.csv' % band_str )
	# obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['correct_surf_M'] ), np.array( dat['surf_M_err'] )

	dat = pds.read_csv( out_path + 'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_mass-Lumi.csv')
	obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['surf_mass'] ), np.array( dat['surf_mass_err'] )

	id_rx = obs_R >= 9
	obs_R, surf_M, surf_M_err = obs_R[id_rx], surf_M[id_rx], surf_M_err[id_rx]

	##.. cov_arr
	# with h5py.File( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % band_str, 'r') as f:
	with h5py.File( out_path + 'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_log-surf-mass_cov_arr.h5', 'r') as f:
		cov_arr = np.array( f['cov_MX'] )
		cor_arr = np.array( f['cor_MX'] )

	id_cov = np.where( id_rx )[0][0]
	cov_arr = cov_arr[id_cov:, id_cov:]

	lg_M, lg_M_err = np.log10( surf_M ), surf_M_err / ( np.log(10) * surf_M )

	if icl_mode == 'xi2M':
		## .. use xi_hm for sigma estimation

		xi_rp = (lo_xi + hi_xi) / 2

		tot_rho_m = ( xi_rp * 1e3 * rho_m ) / a_ref**2 * h

		xi_to_Mf = interp.interp1d( lo_rp, tot_rho_m, kind = 'cubic',)

		misNFW_sigma = xi_to_Mf( obs_R )
		sigma_2Mpc = xi_to_Mf( 2e3 )
		lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

	if icl_mode == 'SG_N':
		#... satellite number density
		sig_aveg = (siglow + sighig) / 2
		sig_rho_f = interp.interp1d( bin_R, sig_aveg, kind = 'linear', fill_value = 'extrapolate',)

		misNFW_sigma = sig_rho_f( obs_R )
		sigma_2Mpc = sig_rho_f( 2e3 )
		lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

	out_lim_R = 250 # 200, 250, 300, 350, 400
	idx_lim = obs_R >= out_lim_R
	id_dex = np.where( idx_lim == True )[0][0]

	fit_R = obs_R[idx_lim]
	fit_M = lg_M[idx_lim]
	fit_Merr = lg_M_err[idx_lim]
	cut_cov = cov_arr[id_dex:, id_dex:]

	po_param = [ cut_cov, lg_M_sigma[idx_lim] ]

	if icl_mode != 'SG_N':
		po = -2.5
		bounds = [ [-4, -2] ]
		E_return = optimize.minimize( err_fit_func, x0 = np.array( po ), args = ( fit_R, fit_M, po_param, fit_Merr), method = 'L-BFGS-B', bounds = bounds,)

	else:
		po = 10.7
		bounds = [ [9, 11] ]
		E_return = optimize.minimize( err_fit_func, x0 = np.array( po ), args = ( fit_R, fit_M, po_param, fit_Merr), method = 'L-BFGS-B', bounds = bounds,)

	print(E_return)
	popt = E_return.x
	bf_fit = popt

	lg_fb_arr.append( bf_fit )

	_out_M = 10**lg_M_sigma * 10** bf_fit
	_sum_fit = np.log10( _out_M )
	devi_lgM = np.log10( surf_M - _out_M )

	##.. model lines
	new_R = np.logspace( 0, np.log10(2.5e3), 100)

	if icl_mode == 'xi2M':
		fit_out_M = ( xi_to_Mf( new_R ) - sigma_2Mpc ) * 10**bf_fit

	if icl_mode == 'SG_N':
		fit_out_M = ( sig_rho_f( new_R ) - sigma_2Mpc ) * 10** bf_fit

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('total, %s-band based' % band_str)

	ax.errorbar( obs_R, lg_M, yerr = lg_M_err, xerr = None, color = 'r', marker = '.', ls = 'none', ecolor = 'r', 
		alpha = 0.75, mec = 'r', mfc = 'r', label = 'observed')

	if icl_mode == 'xi2M':
		ax.plot( new_R, np.log10( fit_out_M ), ls = '-', color = 'b', alpha = 0.75, 
			label = '$\\lg \\Sigma_{dm} + lgf$',)
		ax.text( 1e1, 3.5, s = '$\\lgf{=}%.5f \; [R>=%dkpc]$' % (bf_fit, out_lim_R), color = 'k',)

	if icl_mode == 'SG_N':
		ax.plot( new_R, np.log10( fit_out_M ), ls = '-', color = 'b', alpha = 0.75, 
			label = '$N_{g} * 10^{%.3f} M_{\\odot}$' % bf_fit,)

	ax.legend( loc = 1, )
	ax.set_ylim( 3, 8.5)
	ax.set_ylabel( '$ lg \\Sigma [M_{\\odot} / kpc^2]$' )

	ax.axvline( x = out_lim_R, ls = '--', color = 'g',)

	ax.set_xlim( 1e1, 3e3)
	ax.set_xlabel( 'R [kpc]')
	ax.set_xscale( 'log' )
	plt.savefig('/home/xkchen/total_%s-band-based_beyond-%dkpc_%s-fit.png' % (band_str, out_lim_R, icl_mode), dpi = 300)
	plt.close()

keys = ['lg_fb_gi', 'lg_fb_gr', 'lg_fb_ri']
values = [ ]
for oo in range( 3 ):
	values.append( lg_fb_arr[oo] )
fill = dict( zip( keys, values) )
out_data = pds.DataFrame( fill, index = ['k', 'v'])
# out_data.to_csv( fit_path + 'total_all-color-to-M_beyond-%dkpc_%s-fit.csv' % (out_lim_R, icl_mode) )
out_data.to_csv( out_path + 'total_all-color-to-M_beyond-%dkpc_%s-fit.csv' % (out_lim_R, icl_mode) )

