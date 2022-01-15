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
import astropy.units as U
import astropy.constants as C
from scipy import interpolate as interp
from astropy import cosmology as apcy

from surface_mass_density import sigmam, sigmac, input_cosm_model, cosmos_param, rhom_set

### === ### cosmology
# Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
Test_model = apcy.Planck15.clone(H0 = 67.4, Om0 = 0.315)

H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

rad2asec = U.rad.to(U.arcsec)

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

## miscen params
v_m = 200 # rho_mean = 200 * rho_c * omega_m
c_mass = [5.87, 6.95]
Mh0 = [14.2389, 14.2408]
off_set = [230, 210] # in unit kpc / h
f_off = [0.37, 0.20]

fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']

# 14.238883, 5.861026, 2.906274, 0.234561, 0.369895  ## low
# 14.240789, 6.939821, 2.592336, 0.214603, 0.203696  ## high

z_ref = 0.25
a_ref = 1 / (z_ref + 1)

#...xi_rp load (python2 read)

# import pickle
# xi_file = '/home/xkchen/ffitdata_jointfit_shan_long2.p'

# fitdatalist = pickle.load( open( xi_file, 'r') )
# p, rp, delsig, delsig_pred, xirp, xirpmean, rp_data, delsig_data, covmat, r, xihm, rvir, rs, sigoff = fitdatalist[I]

# infile = open( xi_file, 'r')
# data_dict = pickle.load( infile, encoding = 'bytes' )
# p, rp, delsig, delsig_pred, xirp, xirpmean, rp_data, delsig_data, covmat, r, xihm, rvir, rs, sigoff = data_dict[0]


rho_c, rho_m = rhom_set( z_ref ) # in unit of M_sun * h^2 / kpc^3
rho_m = rho_m / (1 + z_ref)**3

lo_xi_file = '/home/xkchen/tmp_run/data_files/figs/low_BCG_M_xi-rp.txt'
hi_xi_file = '/home/xkchen/tmp_run/data_files/figs/high_BCG_M_xi-rp.txt'

lo_dat = np.loadtxt( lo_xi_file )
lo_rp, lo_xi = lo_dat[:,0], lo_dat[:,1]
lo_rho_m = lo_xi * rho_m * 1e9

hi_dat = np.loadtxt( hi_xi_file )
hi_rp, hi_xi = hi_dat[:,0], hi_dat[:,1]
hi_rho_m = hi_xi * rho_m * 1e9

lo_interp_F = interp.interp1d( lo_rp, lo_xi, kind = 'cubic')
hi_interp_F = interp.interp1d( hi_rp, hi_xi, kind = 'cubic')

lo_xi2M_2Mpc = lo_interp_F( 2 / a_ref )  * rho_m * 1e9
hi_xi2M_2Mpc = hi_interp_F( 2 / a_ref )  * rho_m * 1e9


lo_NFW_sigma = obs_sigma_func( lo_rp * 1e3 * a_ref, f_off[0], off_set[0], z_ref, c_mass[0], Mh0[0], v_m )
hi_NFW_sigma = obs_sigma_func( hi_rp * 1e3 * a_ref, f_off[1], off_set[1], z_ref, c_mass[1], Mh0[1], v_m )

lo_2Mpc_sigma = obs_sigma_func( 2e3, f_off[0], off_set[0], z_ref, c_mass[0], Mh0[0], v_m ) * 1e6 / (1 + z_ref)**2
hi_2Mpc_sigma = obs_sigma_func( 2e3, f_off[1], off_set[1], z_ref, c_mass[1], Mh0[1], v_m ) * 1e6 / (1 + z_ref)**2

# lo_NFW_sigma = sigmam( lo_rp * 1e3 * a_ref, Mh0[0], z_ref, c_mass[0] )
# hi_NFW_sigma = sigmam( hi_rp * 1e3 * a_ref, Mh0[1], z_ref, c_mass[1] )

# lo_2Mpc_sigma = sigmam( 2e3, Mh0[0], z_ref, c_mass[0] ) * 1e6 / (1 + z_ref)**2
# hi_2Mpc_sigma = sigmam( 2e3, Mh0[1], z_ref, c_mass[1] ) * 1e6 / (1 + z_ref)**2

lo_NFW_sigma = lo_NFW_sigma * 1e6 / (1 + z_ref)**2
hi_NFW_sigma = hi_NFW_sigma * 1e6 / (1 + z_ref)**2


plt.figure()
plt.title( 'z = %.3f' % z_ref )
plt.plot( hi_rp, hi_rho_m - hi_xi2M_2Mpc, 'r-', alpha = 0.5, label = fig_name[1] + ', $\\xi$ * $\\rho_{m}(z)$')
plt.plot( lo_rp, lo_rho_m - lo_xi2M_2Mpc, 'b-', alpha = 0.5, label = fig_name[0] + ', $\\xi$ * $\\rho_{m}(z)$')

plt.plot( hi_rp, hi_NFW_sigma - hi_2Mpc_sigma, 'm--', alpha = 0.5, label = fig_name[1] + ', Mine')
plt.plot( lo_rp, lo_NFW_sigma - lo_2Mpc_sigma, 'g--', alpha = 0.5, label = fig_name[0] + ', Mine')

plt.legend( loc = 3)
plt.xscale('log')
plt.xlim(1e-3, 1e1)
plt.xlabel('rp [Mpc/h]')
plt.yscale('log')
plt.ylim(1e12, 5e15)
plt.ylabel('$\\Sigma \; [M_{\\odot} \, h \, / \, Mpc^2]$')
plt.savefig('/home/xkchen/surface_density_compare_z%.3f.png' % z_ref, dpi = 300)
plt.show()

raise

#... physical check
BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/BCG_M_bin/BGs/'
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

a_ref = 1 / (z_ref + 1)
rho_c, rho_m = rhom_set( 0 ) # in unit of M_sun * h^2 / kpc^3, comoving coordinate

lo_xi_file = '/home/xkchen/tmp_run/data_files/figs/low_BCG_M_xi-rp.txt'
lo_dat = np.loadtxt( lo_xi_file )

lo_rp, lo_xi = lo_dat[:,0], lo_dat[:,1] ## xi in unit Mpc/h, comoving
lo_rho_m = ( lo_xi * 1e3 * rho_m ) / a_ref**2 * h
lo_rp = lo_rp * 1e3 / h * a_ref

band_str = 'gi'
mm = 0

dat = pds.read_csv( BG_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str) )
obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['surf_mass'] ), np.array( dat['surf_mass_err'] )


misNFW_sigma = obs_sigma_func( obs_R * h, f_off[mm], off_set[mm], z_ref, c_mass[mm], Mh0[mm], v_m ) * h

plt.figure()
plt.plot( obs_R, misNFW_sigma, 'r-',)
plt.plot( lo_rp, lo_rho_m, 'b--',)

plt.xscale('log')
plt.yscale('log')
plt.show()


# from colossus.cosmology import cosmology as cc_cosmo
# from colossus.halo import profile_nfw

# params = cc_cosmo.cosmologies['planck15']
# params['H0'] = H0
# params['Om0'] = Omega_m
# params['Ob0'] = Omega_b

# cosmos = cc_cosmo.setCosmology('planck15', params)

# nfw_pros_0 = profile_nfw.NFWProfile( M = 10**(Mh0[0]), mdef = '200m', z = z_ref, c = c_mass[0] )
# cc_Sigma_0 = nfw_pros_0.surfaceDensity( lo_rp * 1e3 * a_ref )
# lo_cc_Sigma = cc_Sigma_0 * 1e6

# nfw_pros_1 = profile_nfw.NFWProfile( M = 10**(Mh0[1]), mdef = '200m', z = z_ref, c = c_mass[1] )
# cc_Sigma_1 = nfw_pros_1.surfaceDensity( hi_rp * 1e3 * a_ref )
# hi_cc_Sigma = cc_Sigma_1 * 1e6

# plt.figure()
# plt.plot( hi_rp, hi_NFW_sigma, 'm--', alpha = 0.5, label = 'Mine')
# plt.plot( lo_rp, lo_NFW_sigma, 'g--', alpha = 0.5, )

# plt.plot( hi_rp, hi_cc_Sigma, 'r-', alpha = 0.5)
# plt.plot( lo_rp, lo_cc_Sigma, 'b-', alpha = 0.5, label = 'colossus')

# plt.legend( loc = 3)
# plt.xscale('log')
# plt.xlim(1e-3, 1e1)
# plt.xlabel('rp [Mpc/h]')
# plt.yscale('log')
# plt.ylim(1e12, 5e15)
# plt.ylabel('$\\Sigma \; [M_{\\odot} \, h \, / \, Mpc^2]$')
# plt.savefig('/home/xkchen/surface_density_compare_0.png', dpi = 300)
# plt.close()

