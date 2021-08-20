import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp

import astropy.units as U
import astropy.constants as C

import emcee
import corner
import time

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize

from scipy import signal
from scipy import interpolate as interp
from scipy import optimize
from scipy import integrate as integ

from surface_mass_density import sigmam, sigmac, input_cosm_model, cosmos_param, rhom_set
from color_2_mass import get_c2mass_func
from multiprocessing import Pool

### === ### cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

z_ref = 0.25
pixel = 0.396
band = ['r', 'g', 'i']
L_wave = np.array([ 6166, 4686, 7480 ])
psf_FWHM = [ 1.56, 1.67, 1.50 ] # arcsec

color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

Da_ref = Test_model.angular_diameter_distance( z_ref ).value
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec

# sample_lis = 'BCG_M'
sample_lis = 'rich'

if sample_lis == 'BCG_M':

	BG_path = '/home/xkchen/jupyter/fixed_BCG_M/age_bin/BGs/'
	path = '/home/xkchen/jupyter/fixed_BCG_M/age_bin/SBs/'
	cat_lis = [ 'low-age', 'hi-age' ]
	fig_name = [ 'Low $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 'High $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $' ]
	file_s = 'age-bin_fixed_BCG-M'

	# BG_path = '/home/xkchen/jupyter/fixed_BCG_M/rich_bin_SBs/BGs/'
	# path = '/home/xkchen/jupyter/fixed_BCG_M/rich_bin_SBs/SBs/'
	# cat_lis = [ 'low-rich', 'hi-rich' ]
	# fig_name = [ 'Low $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 'High $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $']
	# file_s = 'rich-bin_fixed_BCG-M'

if sample_lis == 'rich':

	# BG_path = '/home/xkchen/jupyter/fixed_rich/age_bin_SBs/BGs/'
	# path = '/home/xkchen/jupyter/fixed_rich/age_bin_SBs/SBs/'
	# cat_lis = [ 'younger', 'older' ]
	# fig_name = ['Low $ t_{\\mathrm{age}} $ $ \\mid \\lambda $', 'High $ t_{\\mathrm{age}} $ $ \\mid \\lambda $']
	# file_s = 'age-bin'

	BG_path = '/home/xkchen/jupyter/fixed_rich/BCG_M_bin/BGs/'
	path = '/home/xkchen/jupyter/fixed_rich/BCG_M_bin/SBs/'
	cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']
	fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']
	file_s = 'BCG_M-bin'

## ... satellite number density
a_ref = 1 / (1 + z_ref)
bin_R, siglow, errsiglow, sighig, errsighig, highoverlow, errhighoverlow = np.genfromtxt('/home/xkchen/figs/result_high_over_low.txt', unpack = True)
bin_R = bin_R * 1e3 / h * a_ref
siglow, errsiglow, sighig, errsighig = siglow * h**2 / 1e6, errsiglow * h**2 / 1e6, sighig * h**2 / 1e6, errsighig * h**2 / 1e6,

id_nan = np.isnan( bin_R )
bin_R = bin_R[ id_nan == False]
siglow, errsiglow, sighig, errsighig = siglow[ id_nan == False], errsiglow[ id_nan == False], sighig[ id_nan == False], errsighig[ id_nan == False]

lo_Ng_int_F = interp.interp1d( bin_R, siglow, kind = 'linear', fill_value = 'extrapolate')
hi_Ng_int_F = interp.interp1d( bin_R, sighig, kind = 'linear', fill_value = 'extrapolate')


## ... DM mass profile
input_cosm_model( get_model = Test_model )
cosmos_param()

rho_c, rho_m = rhom_set( 0 ) # in unit of M_sun * h^2 / kpc^3

lo_xi_file = '/home/xkchen/figs/low_BCG_M_xi-rp.txt'
hi_xi_file = '/home/xkchen/figs/high_BCG_M_xi-rp.txt'

lo_dat = np.loadtxt( lo_xi_file )
lo_rp, lo_xi = lo_dat[:,0], lo_dat[:,1]
lo_rho_m = ( lo_xi * 1e3 * rho_m ) / a_ref**2 * h
lo_rp = lo_rp * 1e3 / h * a_ref

hi_dat = np.loadtxt( hi_xi_file )
hi_rp, hi_xi = hi_dat[:,0], hi_dat[:,1]
hi_rho_m = ( hi_xi * 1e3 * rho_m ) / a_ref**2 * h
hi_rp = hi_rp * 1e3 / h * a_ref

lo_interp_F = interp.interp1d( lo_rp, lo_rho_m, kind = 'cubic',)
hi_interp_F = interp.interp1d( hi_rp, hi_rho_m, kind = 'cubic',)

lo_xi2M_2Mpc = lo_interp_F( 2e3 )
hi_xi2M_2Mpc = hi_interp_F( 2e3 )


base_lis = ['gi', 'ri', 'gri', 'gr', 'ir', 'gir', 'ig', 'rg', 'rig']
line_name = ['g-i + $L_{i}$', 'r-i + $L_{i}$', 'g-r + $L_{i}$', 
			 'g-r + $L_{r}$', 'r-i + $L_{r}$', 'g-i + $L_{r}$', 
			 'g-i + $L_{g}$', 'g-r + $L_{g}$', 'r-i + $L_{g}$' ]

tl_lis = ['b-', 'b--', 'b:', 'r-', 'r--', 'r:', 'g-', 'g--', 'g:']

fig = plt.figure( figsize = (12.8, 4.8) )
ax0 = fig.add_axes([0.07, 0.12, 0.40, 0.80])
ax1 = fig.add_axes([0.55, 0.12, 0.40, 0.80])

dpt_R, dpt_M = [], []
dpt_M_err = []

for mm in range( 2 ):

	if mm == 0:
		ax = ax0
	if mm == 1:
		ax = ax1

	#... satellite number density
	if mm == 0:
		bin_R, sgN, sgN_err = bin_R, siglow, errsiglow
		sgN_2Mpc = lo_Ng_int_F( 2e3 )
	if mm == 1:
		bin_R, sgN, sgN_err = bin_R, sighig, errsighig
		sgN_2Mpc = hi_Ng_int_F( 2e3 )

	#... mass profile
	if mm == 0:
		new_R = lo_rp
		misNFW_sigma = lo_interp_F( new_R )
		sigma_2Mpc = lo_xi2M_2Mpc + 0.
		lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

	if mm == 1:
		new_R = hi_rp
		misNFW_sigma = hi_interp_F( new_R )
		sigma_2Mpc = hi_xi2M_2Mpc + 0.
		lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

	_R_, _M_, _M_err_ = [], [], []

	for ll in range( 9 ):

		if ll == 4 or ll == 1:
			pass
		else:
			continue
		band_str = base_lis[ ll ]

		dat = pds.read_csv( BG_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str),)
		aveg_R = np.array(dat['R'])

		aveg_surf_m, aveg_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])
		aveg_lgM, aveg_lgM_err = np.array(dat['lg_M']), np.array(dat['lg_M_err'])

		_R_.append( aveg_R )
		_M_.append( aveg_surf_m )
		_M_err_.append( aveg_surf_m_err )


		ax.plot( aveg_R, aveg_surf_m, tl_lis[ll], alpha = 0.5, label = line_name[ll],)
		ax.plot( bin_R, (sgN - sgN_2Mpc) * 10**10.12, 'c-')
		ax.plot( new_R, 10**lg_M_sigma * 10**(-2.78), 'k--')

		ax.set_xlim( 6e0, 1.1e3)
		ax.set_xscale('log')
		ax.set_xlabel('R [kpc]', fontsize = 15)
		ax.set_ylim( 10**3.5, 10**9)
		ax.set_yscale('log')
		ax.legend( loc = 1, frameon = False, fontsize = 11,)
		ax.set_ylabel('$\\Sigma_{\\ast} \; [M_{\\odot} \, / \, kpc^2]$', fontsize = 15,)
		ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
		ax.annotate( text = fig_name[mm], xy = (0.10, 0.10), xycoords = 'axes fraction', fontsize = 15,)

	dpt_R.append( _R_ )
	dpt_M.append( _M_ )
	dpt_M_err.append( _M_err_ )

plt.savefig('/home/xkchen/%s_surface_mass_profile_compare.png' % file_s, dpi = 300)
plt.close()

raise

for ll in range( 1, 9 ):

	plt.figure()
	ax = plt.subplot(111)

	ax.plot( dpt_R[0][0], dpt_M[0][0], ls = '--', color = 'k', alpha = 0.75, label = fig_name[0] +'$\; \;$'+ line_name[0],)
	ax.fill_between( dpt_R[0][0], y1 = dpt_M[0][0] - dpt_M_err[0][0], y2 = dpt_M[0][0] + dpt_M_err[0][0], 
		color = 'k', alpha = 0.12, ls = '--',)

	ax.plot( dpt_R[1][0], dpt_M[1][0], ls = '-', color = 'k', alpha = 0.75, label = fig_name[1] +'$\; \;$'+ line_name[0],)
	ax.fill_between( dpt_R[1][0], y1 = dpt_M[1][0] - dpt_M_err[1][0], y2 = dpt_M[1][0] + dpt_M_err[1][0], 
		color = 'k', alpha = 0.12,)

	ax.plot( dpt_R[0][ll], dpt_M[0][ll], ls = '--', color = 'b', alpha = 0.75, label = fig_name[0] +'$\; \;$'+ line_name[ll],)
	ax.fill_between( dpt_R[0][ll], y1 = dpt_M[0][ll] - dpt_M_err[0][ll], y2 = dpt_M[0][ll] + dpt_M_err[0][ll], 
		color = 'b', alpha = 0.12, ls = '--',)

	ax.plot( dpt_R[1][ll], dpt_M[1][ll], ls = '-', color = 'r', alpha = 0.75, label = fig_name[1] +'$\; \;$'+ line_name[ll],)
	ax.fill_between( dpt_R[1][ll], y1 = dpt_M[1][ll] - dpt_M_err[1][ll], y2 = dpt_M[1][ll] + dpt_M_err[1][ll], 
		color = 'r', alpha = 0.12,)

	ax.set_xlim( 6e0, 1.1e3)
	ax.set_xscale('log')
	ax.set_xlabel('R [kpc]', fontsize = 15)
	ax.set_ylim( 10**3.5, 10**9)
	ax.set_yscale('log')
	ax.legend( loc = 1, frameon = False, fontsize = 11,)
	ax.set_ylabel('$\\Sigma_{\\ast} \; [M_{\\odot} \, / \, kpc^2]$', fontsize = 15,)
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	plt.savefig('/home/xkchen/%s_surface_mass_profile_compare_%d.png' % (file_s, ll), dpi = 300)
	plt.close()	
