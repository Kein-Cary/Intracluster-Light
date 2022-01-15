import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
import pandas as pds
import emcee
import corner

import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
from scipy import signal
from scipy import interpolate as interp
from scipy import optimize
from scipy import integrate as integ

from fig_out_module import arr_jack_func
from light_measure import cov_MX_func
from surface_mass_density import sigmam, sigmac, input_cosm_model, cosmos_param

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

# constant
M_sun = C.M_sun.value # in unit of kg
kpc2m = U.kpc.to(U.m)
Mpc2cm = U.Mpc.to(U.cm)
Msun2kg = U.M_sun.to(U.kg)

rad2arcsec = U.rad.to(U.arcsec)
Lsun = C.L_sun.value*10**7 # (erg/s/cm^2)
Jy = 10**(-23) # (erg/s)/cm^2/Hz
F0 = 3.631 * 10**(-6) * Jy
L_speed = C.c.value # m/s

pixel = 0.396
band = ['r', 'g', 'i']
L_wave = np.array([ 6166, 4686, 7480 ])
## solar Magnitude corresponding to SDSS filter
Mag_sun = [ 4.65, 5.11, 4.53 ]

### === ### initial surface_mass_density.py module
input_cosm_model( get_model = Test_model )
cosmos_param()

### === ### halo profile model
from colossus.cosmology import cosmology
from colossus.halo import profile_einasto
from colossus.halo import profile_nfw
from colossus.halo import profile_hernquist

cosmos_param = {'flat': True, 'H0': H0, 'Om0': Omega_m, 'Ob0': Omega_b, 'sigma8' : 0.811, 'ns': 0.965}
cosmology.addCosmology('myCosmo', cosmos_param )
cosmo = cosmology.setCosmology( 'myCosmo' )


### === ### miscentering nfw profile (Zu et al. 2020, section 3.)
def mis_p_func( r_off, sigma_off):
	"""
	r_off : the offset between cluster center and BCGs
	sigma_off : characteristic offset
	"""
	pf0 = r_off / sigma_off**2
	pf1 = np.exp( - r_off / sigma_off )

	return pf0 * pf1

def off_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m):

	theta = np.linspace( 0, 2 * np.pi, 100)
	d_theta = np.diff( theta )

	try:
		NR = len( rp )
	except:
		rp = np.array( [rp] )
		NR = len( rp )

	r_off = np.arange( 0, 15 * sigma_off, 0.02 * sigma_off )
	off_pdf = mis_p_func( r_off, sigma_off )

	NR_off = len( r_off )

	surf_dens_off = np.zeros( NR, dtype = np.float32 )

	for ii in range( NR ):

		surf_dens_arr = np.zeros( NR_off, dtype = np.float32 )

		for jj in range( NR_off ):

			r_cir = np.sqrt( rp[ii]**2 + 2 * rp[ii] * r_off[jj] * np.cos( theta ) + r_off[jj]**2 )

			surf_dens_of_theta = sigmam( r_cir, lgM, z, c_mass )

			## integration on theta
			surf_dens_arr[jj] = integ.simps( surf_dens_of_theta, theta) / ( 2 * np.pi )

		## integration on r_off
		integ_f = surf_dens_arr * off_pdf

		surf_dens_ii = integ.simps( integ_f, r_off )

		surf_dens_off[ ii ] = surf_dens_ii

	off_sigma = surf_dens_off

	return off_sigma

def cc_off_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m):

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

	return off_sigma

def aveg_sigma_func(rp, sigma_arr, N_grid = 100):

	NR = len( rp )
	aveg_sigma = np.zeros( NR, dtype = np.float32 )

	tR = rp
	intep_sigma_F = interp.interp1d( tR , sigma_arr, kind = 'cubic', fill_value = 'extrapolate',)

	for ii in range( NR ):

		new_rp = np.logspace(-3, np.log10( tR[ii] ), N_grid)
		new_sigma = intep_sigma_F( new_rp )

		cumu_sigma = integ.simps( new_rp * new_sigma, new_rp)

		aveg_sigma[ii] = 2 * cumu_sigma / tR[ii]**2

	return aveg_sigma

def obs_sigma_func( rp, f_off, sigma_off, z, c_mass, lgM, v_m):

	# off_sigma = off_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m)
	off_sigma = cc_off_sigma_func( rp, sigma_off, z, c_mass, lgM, v_m)

	norm_sigma = sigmam( rp, lgM, z, c_mass)

	obs_sigma = f_off * off_sigma + ( 1 - f_off ) * norm_sigma

	return obs_sigma

def delta_sigma_func(rp, f_off, sigma_off, z, c_mass, lgM, v_m, N_grid = 100):

	sigma_arr = obs_sigma_func( rp, f_off, sigma_off, z, c_mass, lgM, v_m )
	aveg_sigma = aveg_sigma_func( rp, sigma_arr, N_grid = N_grid)
	delta_sigma = aveg_sigma - sigma_off

	return delta_sigma

### === ### SB profile to delta SB profile
def SB_to_Lumi_func(sb_arr, obs_z, band_str):
	"""
	sb_arr : need in terms of absolute magnitude, in AB system
	"""
	if band_str == 'r':
		Mag_dot = Mag_sun[0]

	if band_str == 'g':
		Mag_dot = Mag_sun[1]

	if band_str == 'i':
		Mag_dot = Mag_sun[2]

	# luminosity, in unit of  L_sun / pc^2
	lumi = 10**( -0.4 * (sb_arr - Mag_dot + 21.572 - 10 * np.log10( obs_z + 1 ) ) )

	Lumi = lumi * 1e6 # in unit of  L_sun / kpc^2

	return Lumi

def aveg_lumi_func(rp, lumi_arr, N_grid = 100):

	NR = len( rp )

	aveg_lumi = np.zeros( NR, dtype = np.float32 )

	intep_lumi_F = interp.interp1d( rp, lumi_arr, kind = 'linear', fill_value = 'extrapolate',)

	for ii in range( NR ):

		new_rp = np.logspace(-3, np.log10( rp[ii] ), N_grid)
		new_lumi = intep_lumi_F( new_rp )

		cumu_lumi = integ.simps( new_rp * new_lumi, new_rp)

		aveg_lumi[ii] = 2 * cumu_lumi / rp[ii]**2

	return aveg_lumi

def delta_SB_func( rp, lumi_arr, N_grid = 100):

	aveg_lumi = aveg_lumi_func(rp, lumi_arr, N_grid = N_grid)
	delta_lumi = aveg_lumi - lumi_arr

	return delta_lumi

def sersic_func(r, Ie, re, ndex):

	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )

	return Ir

### === ### fitting function
def prior_p_func( p ):

	M0, c_mass, m2l, f_off, sigma_off, Ie, Re, n = p[:]

	identi_0 = ( 1e-2 < Ie < 1e1 ) & ( 5 < Re < 50) & ( 1 < n < 9 ) & (1 < c_mass < 50) & (2e2 < m2l < 5e3) 
	identi_1 = (13.5 <= M0 <= 15) & ( 0 < f_off < 1) & ( 10 < sigma_off < 400)

	if ( identi_0 & identi_1 ):
		return 0
	return -np.inf

def ln_p_func(p, x, y, params, yerr):

	pre_p = prior_p_func( p )

	if not np.isfinite( pre_p ):
		return -np.inf
	else:
		M0, c_mass, m2l, f_off, sigma_off, Ie, Re, n = p[:]
		z0, cov_mx, v_m = params[:]

		## sersic
		I_r = sersic_func( x, Ie, Re, n )
		aveg_I = aveg_sigma_func( x, I_r )
		delta_I = aveg_I - I_r

		## miscen-nfw
		off_sigma = obs_sigma_func( x * h, f_off, sigma_off, z0, c_mass, M0, v_m) # unit M_sun * h / kpc^2
		mean_off_sigma = aveg_sigma_func( x * h, off_sigma )

		off_delta_sigma = mean_off_sigma - off_sigma
		off_D_sigma = off_delta_sigma * h * 1e-6 # unit M_sun / pc^2

		## model SB
		mode_mu = delta_I + off_D_sigma / m2l

		cov_inv = np.linalg.pinv( cov_mx )
		delta = mode_mu - y
		chi2 = -0.5 * delta.T.dot( cov_inv ).dot(delta)

		return pre_p + chi2

def err_fit_func(p, x, y, params, yerr):

	# m2l Ie, Re, n = p[:]
	m2l = p
	cov_mx, input_sigma = params[:]

	## model SB
	mode_mu = input_sigma / m2l

	cov_inv = np.linalg.pinv( cov_mx )
	delta = mode_mu - y

	chi2 = delta.T.dot( cov_inv ).dot(delta)

	return chi2

## use high mass bin for test
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$'] ## or line name

color_s = ['r', 'g', 'b']
line_c = ['b', 'r']
line_s = ['--', '-']

z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value

### === ### take sub-sample profile to estimate error on delta_mu (the surface brightness )
def cov_arr():

	BG_path = '/home/xkchen/mywork/ICL/code/rig_common_cat/mass_bin_BG/'
	Lpro_path = '/home/xkchen/mywork/ICL/code/rig_common_cat/SB_in_Lsun/'

	N_samples = 30

	for mm in range( 2 ):

		for kk in range( 3 ):

			nt_r, nt_l, nt_ml, nt_Dl = [], [], [], []

			for nn in range( N_samples ):

				n_dat = pds.read_csv( BG_path + '%s_%s-band_jack-sub-%d_BG-sub_SB.csv' % (cat_lis[mm], band[kk], nn),)
				nn_r, nn_sb, nn_err = np.array( n_dat['R']), np.array( n_dat['BG_sub_SB']), np.array( n_dat['sb_err'])

				idvx = ( nn_r >= 10 ) & ( nn_r <= 1e3)

				nn_mag = 22.5 - 2.5 * np.log10( nn_sb[idvx] )
				nn_Mag = nn_mag - 5 * np.log10( Dl_ref * 10**6 / 10)
				
				nn_Lumi = SB_to_Lumi_func( nn_Mag, z_ref, band[kk] )

				aveg_Lumi = aveg_lumi_func( nn_r[idvx], nn_Lumi )

				delt_Lumi = aveg_Lumi - nn_Lumi

				keys = [ 'R', 'Lumi', 'mean_Lumi', 'delta_Lumi' ]
				values = [ nn_r[idvx], nn_Lumi, aveg_Lumi, delt_Lumi ]
				fill = dict(zip( keys, values) )
				out_data = pds.DataFrame( fill )
				out_data.to_csv( Lpro_path + '%s_%s-band_jack-sub-%d_Lumi-pros.csv' % (cat_lis[mm], band[kk], nn),)	

				nt_r.append( nn_r[idvx] )
				nt_l.append( nn_Lumi )
				nt_ml.append( aveg_Lumi )
				nt_Dl.append( delt_Lumi )

			### jack-mean pf mass and lumi profile
			aveg_R, aveg_L, aveg_L_err = arr_jack_func( nt_l, nt_r, N_samples)[:3]
			aveg_R, aveg_mL, aveg_mL_err = arr_jack_func( nt_ml, nt_r, N_samples)[:3]
			aveg_R, aveg_DL, aveg_DL_err = arr_jack_func( nt_Dl, nt_r, N_samples)[:3]

			keys = ['R', 'Lumi', 'Lumi_err', 'm_Lumi', 'm_Lumi_err', 'd_Lumi', 'd_Lumi_err']
			values = [ aveg_R, aveg_L, aveg_L_err, aveg_mL, aveg_mL_err, aveg_DL, aveg_DL_err ]
			fill = dict(zip( keys, values) )
			out_data = pds.DataFrame( fill )
			out_data.to_csv( Lpro_path + '%s_%s-band_aveg-jack_Lumi-pros.csv' % (cat_lis[mm], band[kk]),)

	## cov-matrix
	for mm in range( 2 ):

		for kk in range( 3 ):

			nt_r, nt_l, nt_ml, nt_Dl = [], [], [], []

			for nn in range( N_samples ):

				d_dat = pds.read_csv( Lpro_path + '%s_%s-band_jack-sub-%d_Lumi-pros.csv' % (cat_lis[mm], band[kk], nn) )
				dd_r, dd_L, dd_mL, dd_DL = np.array( d_dat['R'] ), np.array( d_dat['Lumi'] ), np.array( d_dat['mean_Lumi'] ), np.array( d_dat['delta_Lumi'] )

				## cov-matrix calculate with unit L_sun / pc**2
				nt_r.append( dd_r )
				nt_l.append( dd_L * 1e-6 )
				nt_ml.append( dd_mL * 1e-6 )
				nt_Dl.append( dd_DL * 1e-6 )

			R_m_0, cov_MX_0, cor_MX_0 = cov_MX_func( nt_r, nt_l, id_jack = True)
			R_m_1, cov_MX_1, cor_MX_1 = cov_MX_func( nt_r, nt_Dl, id_jack = True)

			with h5py.File( Lpro_path + '%s_%s-band_Lumi-pros_cov-cor.h5' % (cat_lis[mm], band[kk]), 'w') as f:
				f['cov_Mx'] = np.array( cov_MX_0 )
				f['cor_Mx'] = np.array( cor_MX_0 )
				f['R_kpc'] = np.array( R_m_0 )

			with h5py.File( Lpro_path + '%s_%s-band_Delta-Lumi-pros_cov-cor.h5' % (cat_lis[mm], band[kk]), 'w') as f:
				f['cov_Mx'] = np.array( cov_MX_1 )
				f['cor_Mx'] = np.array( cor_MX_1 )
				f['R_kpc'] = np.array( R_m_1 )

			fig = plt.figure( figsize = (13.12, 4.8) )
			ax0 = fig.add_axes([0.05, 0.10, 0.45, 0.80])
			ax1 = fig.add_axes([0.50, 0.10, 0.45, 0.80])

			ax0.set_title( fig_name[ mm ] + ',%s band, coV_arr' % band[kk] )
			tf = ax0.imshow(cov_MX_0, origin = 'lower', cmap = 'rainbow', norm = mpl.colors.LogNorm(),)
			ax0.set_ylim(-0.5, len(R_m_0) - 0.5 )

			ax1.set_title( fig_name[ mm ] + ', %s band, coR_arr' % band[kk] )
			tf = ax1.imshow(cor_MX_0, origin = 'lower', cmap = 'bwr', vmin = -1, vmax = 1,)
			ax1.set_ylim(-0.5, len(R_m_0) - 0.5 )

			plt.savefig('/home/xkchen/figs/%s_%s-band_Lumi-pros_coV-coR_arr.jpg' % (cat_lis[mm], band[kk]), dpi = 300)
			plt.close()

			fig = plt.figure( figsize = (13.12, 4.8) )
			ax0 = fig.add_axes([0.05, 0.10, 0.45, 0.80])
			ax1 = fig.add_axes([0.50, 0.10, 0.45, 0.80])

			ax0.set_title( fig_name[ mm ] + ',%s band, coV_arr' % band[kk] )
			tf = ax0.imshow(cov_MX_1, origin = 'lower', cmap = 'rainbow', norm = mpl.colors.LogNorm(),)
			ax0.set_ylim(-0.5, len(R_m_1) - 0.5 )

			ax1.set_title( fig_name[ mm ] + ', %s band, coR_arr' % band[kk] )
			tf = ax1.imshow(cor_MX_1, origin = 'lower', cmap = 'bwr', vmin = -1, vmax = 1,)
			ax1.set_ylim(-0.5, len(R_m_1) - 0.5 )

			plt.savefig('/home/xkchen/figs/%s_%s-band_Delta-Lumi-pros_coV-coR_arr.jpg' % (cat_lis[mm], band[kk]), dpi = 300)
			plt.close()

	fig = plt.figure( figsize = (19.84, 4.8) )
	ax0 = fig.add_axes([0.05, 0.10, 0.27, 0.83])
	ax1 = fig.add_axes([0.37, 0.10, 0.25, 0.83])
	ax2 = fig.add_axes([0.67, 0.10, 0.25, 0.83])

	ax0.set_title( '$\\mu(r)$' )
	ax1.set_title( '$\\bar{\\mu}(<r)$' )
	ax2.set_title( '$\\Delta \\mu$ = $\\bar{\\mu}(<r)$ - $\\mu(r)$')

	for mm in range( 2 ):

		for kk in range( 3 ):

			d_dat = pds.read_csv( Lpro_path + '%s_%s-band_aveg-jack_Lumi-pros.csv' % (cat_lis[mm], band[kk]) )
			dd_R, dd_L, dd_L_err = np.array( d_dat['R'] ), np.array( d_dat['Lumi'] ), np.array( d_dat['Lumi_err'] )
			dd_mL, dd_mL_err = np.array( d_dat['m_Lumi'] ), np.array( d_dat['m_Lumi_err'] )
			dd_DL, dd_DL_err = np.array( d_dat['d_Lumi'] ), np.array( d_dat['d_Lumi_err'] )

			if kk == 0:
				ax0.plot( dd_R, dd_L, ls = line_s[mm], color = color_s[kk], alpha = 0.5, label = fig_name[mm] + ',%s band' % band[kk],)
				ax0.fill_between( dd_R, y1 = dd_L - dd_L_err, y2 = dd_L + dd_L_err, color = color_s[kk], alpha = 0.12,)

				ax1.plot( dd_R, dd_mL, ls = line_s[mm], color = color_s[kk], alpha = 0.5, label = fig_name[mm] + ',%s band' % band[kk],)
				ax1.fill_between( dd_R, y1 = dd_mL - dd_mL_err, y2 = dd_mL + dd_mL_err, color = color_s[kk], alpha = 0.12,)

				ax2.plot( dd_R, dd_DL, ls = line_s[mm], color = color_s[kk], alpha = 0.5, label = fig_name[mm] + ',%s band' % band[kk],)
				ax2.fill_between( dd_R, y1 = dd_DL - dd_DL_err, y2 = dd_DL + dd_DL_err, color = color_s[kk], alpha = 0.12,)

			else:
				ax0.plot( dd_R, dd_L, ls = line_s[mm], color = color_s[kk], alpha = 0.5, label = '%s band' % band[kk],)
				ax0.fill_between( dd_R, y1 = dd_L - dd_L_err, y2 = dd_L + dd_L_err, color = color_s[kk], alpha = 0.12,)

				ax1.plot( dd_R, dd_mL, ls = line_s[mm], color = color_s[kk], alpha = 0.5, label = '%s band' % band[kk],)
				ax1.fill_between( dd_R, y1 = dd_mL - dd_mL_err, y2 = dd_mL + dd_mL_err, color = color_s[kk], alpha = 0.12,)

				ax2.plot( dd_R, dd_DL, ls = line_s[mm], color = color_s[kk], alpha = 0.5, label = '%s band' % band[kk],)
				ax2.fill_between( dd_R, y1 = dd_DL - dd_DL_err, y2 = dd_DL + dd_DL_err, color = color_s[kk], alpha = 0.12,)

	ax0.legend( loc = 1,)
	ax0.set_xscale('log')
	ax0.set_yscale('log')
	ax0.set_ylabel('$\\mu(r) [L_{\\odot}/ kpc^2]$')
	ax0.set_xlabel('R[kpc]')
	ax0.set_xlim( 1e1, 1e3)
	ax0.set_ylim( 5e2, 1e7)

	ax1.legend( loc = 1,)
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax1.set_ylabel('$\\bar{\\mu}(<r) [L_{\\odot}/ kpc^2]$')
	ax1.set_xlabel('R[kpc]')
	ax1.set_xlim( 1e1, 1e3)
	ax1.set_ylim( 5e2, 1e7)

	ax2.legend( loc = 1,)
	ax2.set_xscale('log')
	ax2.set_yscale('log')
	ax2.set_ylabel('$\\Delta\\mu(r) [L_{\\odot}/ kpc^2]$')
	ax2.set_xlabel('R[kpc]')
	ax2.set_xlim( 1e1, 1e3)
	ax2.set_ylim( 5e2, 1e7)

	plt.savefig('/home/xkchen/delta-mu_compare.png', dpi = 300)
	plt.close()

# cov_arr()

### === ### MCMC fitting
a_ref = 1 / (1 + z_ref)

v_m = 200 # rho_mean = 200 * rho_c * omega_m
c_mass = [5.87, 6.95]
Mh0 = [14.24, 14.24]
sigm_off = [230, 210]
f_off = [0.37, 0.20]

path = '/home/xkchen/mywork/ICL/code/rig_common_cat/SB_in_Lsun/'
out_path = '/home/xkchen/figs/'

for mm in range( 2 ):

	for kk in range( 3 ):

		d_dat = pds.read_csv( path + '%s_%s-band_aveg-jack_Lumi-pros.csv' % (cat_lis[mm], band[kk]) )
		dd_R, dd_L, dd_L_err = np.array( d_dat['R'] ), np.array( d_dat['Lumi'] ), np.array( d_dat['Lumi_err'] )
		dd_mL, dd_mL_err = np.array( d_dat['m_Lumi'] ), np.array( d_dat['m_Lumi_err'] )
		dd_DL, dd_DL_err = np.array( d_dat['d_Lumi'] ), np.array( d_dat['d_Lumi_err'] )

		## use delta_lumi for fitting, in unit L_sun / pc^2
		_dd_R = dd_R
		_DL = dd_DL * 1e-6
		_DL_err = dd_DL_err * 1e-6

		## cov_arr
		with h5py.File( path + '%s_%s-band_Delta-Lumi-pros_cov-cor.h5' % (cat_lis[mm], band[kk]), 'r') as f:
			cov_MX = np.array(f['cov_Mx'])

		## compare large scale signal only
		idx1 = dd_R >= 10
		cut_dex = np.where( idx1 == True )[0][0]

		com_r = dd_R[idx1]
		com_sb = dd_DL[idx1] * 1e-6
		com_err = dd_DL_err[idx1] * 1e-6

		p_cov_MX = cov_MX[ cut_dex:, cut_dex:]

		## miscen-nfw
		norm_sigma = obs_sigma_func( _dd_R * h, f_off[mm], sigm_off[mm], z_ref, c_mass[mm], Mh0[mm], v_m) # unit M_sun * h / kpc^2
		mean_norm_sigma = aveg_sigma_func( _dd_R * h, norm_sigma )
		delt_n_sigma = mean_norm_sigma - norm_sigma
		delt_n_sigma = delt_n_sigma * h * 1e-6


		m2l_fit = 1000

		## compare
		mode_mu = delt_n_sigma / m2l_fit

		## other halo profile
		rho_Hern = profile_hernquist.HernquistProfile( M = 10**(Mh0[mm]), c = c_mass[mm], z = z_ref, mdef = '200m')
		delta_Hern = rho_Hern.deltaSigma( _dd_R * h )

		delta_Hern = delta_Hern * h * 1e-6

		plt.figure()
		ax = plt.subplot(111)
		ax.set_title( fig_name[mm] + ',%s band' % band[kk])

		ax.plot( _dd_R, _DL, ls = '-', color = 'k', alpha = 0.45, label = 'signal',)
		ax.fill_between( _dd_R, y1 = _DL - _DL_err, y2 = _DL + _DL_err, color = 'k', alpha = 0.12,)

		ax.plot( _dd_R, mode_mu, 'g-', alpha = 0.45, label = '$NFW_{mis}$')
		ax.plot( _dd_R, _DL - mode_mu, ls = '--', color = 'g', alpha = 0.45, label = 'signal - $NFW_{mis}$')

		ax.plot( _dd_R, delta_Hern / m2l_fit, 'b-', alpha = 0.45, label = '$ Hernquist $')
		ax.plot( _dd_R, _DL - delta_Hern / m2l_fit, ls = '--', color = 'b', alpha = 0.45, label = 'signal - $ Hernquist $')		

		ax.set_xlim(1e1, 1e3)
		# ax.set_ylim(1e-2, 1e1)
		ax.set_yscale('log')

		ax.legend( loc = 1)
		ax.set_xscale('log')
		ax.set_xlabel('R[kpc]')
		ax.set_ylabel('$ \\Delta\\mu $ $ [L_{\\odot} / pc^{2}] $')
		ax.grid(which = 'both', axis = 'both', alpha = 0.25,)

		plt.savefig('/home/xkchen/figs/%s_%s-band_sersic+1h-miscen_Delta-L_mcmc_fit.png' % (cat_lis[mm], band[kk]), dpi = 300 )
		plt.close()

raise
