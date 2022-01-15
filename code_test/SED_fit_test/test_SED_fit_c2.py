import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle

import scipy.interpolate as interp
import scipy.stats as sts
from scipy.interpolate import splev, splrep

import astropy.units as U
import astropy.constants as C

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import integrate as integ
from scipy import optimize

import emcee
import ezgal
# from multiprocessing import Pool

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

## cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['g', 'r', 'i']
l_wave = np.array([4686, 6166, 7480])

Jky = 10**(-23) # erg * s^(-1) * cm^(-2) * Hz^(-1)
F0 = 3.631 * 10**(-6) * Jky

def z_at_lb_time( z_min, z_max, dt, N_grid = 100):
	"""
	dt : time interval, in unit of Gyr
	"""
	z_arr = np.linspace( z_min, z_max, N_grid)
	t_arr = Test_model.lookback_time( z_arr ).value ## unit Gyr

	lb_time_low = Test_model.lookback_time( z_min ).value
	lb_time_up = Test_model.lookback_time( z_max ).value

	intep_f = splrep( t_arr, z_arr )
	equa_dt = np.arange( lb_time_low, lb_time_up, dt )
	equa_dt_z = splev( equa_dt, intep_f )

	return equa_dt_z

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$'] ## or line name

color_s = ['g', 'r', 'b']

path = '/home/xkchen/project/tmp_sed_fit/'
out_path = '/home/xkchen/project/tmp_sed_fit/'

zs = 0.25
Dl_ref = Test_model.luminosity_distance( zs ).value

## SED models
model_0 = ezgal.model('bc03_ssp_z_0.008_chab.model')
model_0.set_cosmology( Om = Test_model.Om0, Ol = Test_model.Ode0, h = Test_model.h )
model_0.add_filter('sloan_g',)
model_0.add_filter('sloan_r',)
model_0.add_filter('sloan_i',)

model_1 = ezgal.model('bc03_ssp_z_0.02_chab.model')
model_1.set_cosmology( Om = Test_model.Om0, Ol = Test_model.Ode0, h = Test_model.h )
model_1.add_filter('sloan_g',)
model_1.add_filter('sloan_r',)
model_1.add_filter('sloan_i',)

#===
model_me = ezgal.weight( 97 ) * model_1
model_me = model_me + ezgal.weight( 3 ) * model_0
model_me.set_cosmology( Om = Test_model.Om0, Ol = Test_model.Ode0, h = Test_model.h )
model_me.add_filter('sloan_g',)
model_me.add_filter('sloan_r',)
model_me.add_filter('sloan_i',)

#===
# model_me = model_0

import time

dpt_r = []

path = '/home/xkchen/mywork/ICL/code/rig_common_cat/mass_bin_BG/'
out_path = '/home/xkchen/mywork/ICL/code/ezgal_files/'

# for mm in range( rank, rank + 1):
for mm in range( 2 ):

	tot_r, tot_sb, tot_err = [], [], []

	## observed flux
	for kk in range( 3 ):

		with h5py.File( path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[ mm ], band[kk]), 'r') as f:
			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		tot_r.append( tt_r )
		tot_sb.append( tt_sb )
		tot_err.append( tt_err )

	tot_r = np.array( tot_r )
	tot_sb = np.array( tot_sb )
	tot_err = np.array( tot_err )

	obs_mag = []
	obs_mag_err = []
	obs_R = []

	for kk in range( 3 ):

		idux = (tot_r[kk] >= 10) & (tot_r[kk] <= 1e3) ## use data points within 1Mpc only

		obs_R.append( tot_r[kk][idux] )

		tt_mag = 22.5 - 2.5 * np.log10( tot_sb[kk] )
		tt_mag_err = 2.5 * tot_err[kk] / ( np.log(10) * tot_sb[kk] )

		obs_mag.append( tt_mag[idux] )
		obs_mag_err.append( tt_mag_err[idux] )

	obs_mag = np.array( obs_mag )
	obs_mag_err = np.array( obs_mag_err )
	obs_R = np.array( obs_R )

	dpt_r.append( obs_R[0] )

	## r band absolute mag
	abs_Mag = obs_mag[1] - 5 * np.log10( Dl_ref * 10**6 / 10)

	NR = len( obs_R[0] )

	## fitting for z_form and Mstar
"""
	rec_zfs, rec_Mag, rec_mass = [], [], []
	rec_g_mag, rec_r_mag, rec_i_mag = [], [], []

	for jj in range( NR ):

		lop_g_mag = obs_mag[0][jj]
		lop_r_mag = obs_mag[1][jj]
		lop_i_mag = obs_mag[2][jj]

		lop_g_err = obs_mag_err[0][jj]
		lop_r_err = obs_mag_err[1][jj]
		lop_i_err = obs_mag_err[2][jj]

		lop_abs_Mag = abs_Mag[jj]

		## mcmc fitting
		def likelihood_func(p, ssp_model, zs, obs_g_mag, obs_r_mag, obs_i_mag, obs_g_err, obs_r_err, obs_i_err, norm_abs_mag):

			guss_mag = p[0]
			guss_zf = p[1]

			ssp_model.set_normalization('sloan_r', zs, guss_mag,)
			g_mag_mock, r_mag_mock, i_mag_mock = ssp_model.get_apparent_mags(zf = guss_zf, filters = ['sloan_g', 'sloan_r', 'sloan_i'], zs = zs, ab = True)[0]

			p_return = -0.5 * ( (g_mag_mock - obs_g_mag)**2 / obs_g_err**2 + (r_mag_mock - obs_r_mag)**2 / obs_r_err**2 
								+ (i_mag_mock - obs_i_mag)**2 / obs_i_err**2 )
			if np.isnan( p_return ):
				print( 'something wrong' )
			return p_return

		def prior_p_func(p, ssp_model, zs, obs_g_mag, obs_r_mag, obs_i_mag, obs_g_err, obs_r_err, obs_i_err, norm_abs_mag):

			delta_mag = np.abs( norm_abs_mag - p[0] )
			delta_z = p[1] - zs

			if (delta_mag >= 1.5) or (delta_z < 0.2):
				return -np.inf
			return 0.

		def post_p_func(p, ssp_model, zs, obs_g_mag, obs_r_mag, obs_i_mag, obs_g_err, obs_r_err, obs_i_err, norm_abs_mag):

			pre_p = prior_p_func(p, ssp_model, zs, obs_g_mag, obs_r_mag, obs_i_mag, obs_g_err, obs_r_err, obs_i_err, norm_abs_mag)

			if not np.isfinite( pre_p ):
				return -np.inf

			p_like = likelihood_func(p, ssp_model, zs, obs_g_mag, obs_r_mag, obs_i_mag, obs_g_err, obs_r_err, obs_i_err, norm_abs_mag)

			p_sum = pre_p + p_like

			return p_sum

		def negative_post_p(p, ssp_model, zs, obs_g_mag, obs_r_mag, obs_i_mag, obs_g_err, obs_r_err, obs_i_err, norm_abs_mag):
			negative_p = 0 - post_p_func(p, ssp_model, zs, obs_g_mag, obs_r_mag, obs_i_mag, obs_g_err, obs_r_err, obs_i_err, norm_abs_mag)
			return negative_p

		def negative_post_p_at_zfs(p, ssp_model, zs, obs_g_mag, obs_r_mag, obs_i_mag, obs_g_err, obs_r_err, obs_i_err, norm_abs_mag, fix_zf):
			p[1] = fix_zf
			negative_p = 0 - post_p_func(p, ssp_model, zs, obs_g_mag, obs_r_mag, obs_i_mag, obs_g_err, obs_r_err, obs_i_err, norm_abs_mag)
			return negative_p

		def negative_post_p_at_Mag(p, ssp_model, zs, obs_g_mag, obs_r_mag, obs_i_mag, obs_g_err, obs_r_err, obs_i_err, norm_abs_mag, fix_Mag):
			p[0] = fix_Mag
			negative_p = 0 - post_p_func(p, ssp_model, zs, obs_g_mag, obs_r_mag, obs_i_mag, obs_g_err, obs_r_err, obs_i_err, norm_abs_mag)
			return negative_p

		zfs = z_at_lb_time( 0.25, 8, dt = 0.1)
		pp0 = [ lop_abs_Mag, 3.0 ]


		if obs_R[0][jj] < 100:

			model_put = model_me
			F_return = optimize.fmin( negative_post_p, pp0, xtol = 1e-6, ftol = 1e-6, 
				args = ( model_put, zs, lop_g_mag, lop_r_mag, lop_i_mag, lop_g_err, lop_r_err, lop_i_err, lop_abs_Mag), full_output = True,)
			xopt, fopt = F_return[:2]

			print('%s, %d bin, case done !' % (cat_lis[mm], jj) )

		if obs_R[0][jj] >= 100:

			model_put = model_0
			F_return = optimize.fmin( negative_post_p, pp0, xtol = 1e-6, ftol = 1e-6, 
				args = ( model_put, zs, lop_g_mag, lop_r_mag, lop_i_mag, lop_g_err, lop_r_err, lop_i_err, lop_abs_Mag), full_output = True,)
			xopt, fopt = F_return[:2]

			print('%s, %d bin, case done !' % (cat_lis[mm], jj) )

		### use bc03-ssp_z0.008 only case
		# pre_Mag = np.linspace( lop_abs_Mag - 10 * lop_r_err, lop_abs_Mag + 10 * lop_r_err, len(zfs),)

		# Nz = len( zfs )
		# chi2_arr = np.zeros( (Nz, Nz),)

		# for ti in range( 101 ):

		# 	for tj in range( 101 ):
		# 		tp = [ pre_Mag[ti], zfs[tj] ]
		# 		chi2_arr[ti, tj] = negative_post_p( tp, zs, lop_g_mag, lop_r_mag, lop_i_mag, lop_g_err, lop_r_err, lop_i_err, lop_abs_Mag)

		# chi2arr = 2 * chi2_arr

		# id_inf = np.isnan( chi2arr )
		# idy, idx = np.where( id_inf == False )
		# value_chi2 = chi2arr[ id_inf == False ]

		# id_min = np.where( value_chi2 == value_chi2.min() )[0][0]
		# xopt = [ pre_Mag[idy[id_min]], zfs[idx[id_min]] ]


		## take the zfs and mass
		model_put.set_normalization('sloan_r', zs, xopt[0], apparent = False)
		mod_Mass = model_put.get_masses( xopt[1], zs ) * model_put.get_normalization( xopt[1], flux = True)
		lg_Mass = np.log10( mod_Mass[0][0] )
		mod_g_mag, mod_r_mag, mod_i_mag = model_put.get_apparent_mags( zf = xopt[1], filters = ['sloan_g', 'sloan_r', 'sloan_i'], zs = zs, ab = True)[0]

		rec_zfs.append( xopt[1] )
		rec_Mag.append( xopt[0] )
		rec_mass.append( lg_Mass )
		rec_g_mag.append( mod_g_mag )
		rec_r_mag.append( mod_r_mag )
		rec_i_mag.append( mod_i_mag )

		print( 'to here' )

	keys = ['fit_zf', 'fit_Mag', 'fit_Mass', 'mod_g_mag', 'mod_r_mag', 'mod_i_mag']
	values = [ np.array( rec_zfs), np.array(rec_Mag), np.array(rec_mass), np.array(rec_g_mag), np.array(rec_r_mag), np.array(rec_i_mag) ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )

	# out_data.to_csv( out_path + '%s_bc03-ssp_z0.02+0.008_sed_fit_test.csv' % (cat_lis[ mm ]), )
	# out_data.to_csv( out_path + '%s_bc03-ssp_z0.02_sed_fit_test.csv' % (cat_lis[ mm ]), )
	# out_data.to_csv( out_path + '%s_bc03-ssp_z0.008_sed_fit_test.csv' % (cat_lis[ mm ]), )
	out_data.to_csv( out_path + '%s_bc03-ssp_in100-z0.02+0.008_out-0.008_sed_fit_test.csv' % (cat_lis[ mm ]),)

	print( 'saved the data!' )

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title( fig_name[mm] )
	ax.plot(obs_R[0], rec_g_mag, 'g-', alpha = 0.5, label = 'g band, SED fit')
	ax.errorbar( obs_R[0], obs_mag[0], yerr = obs_mag_err[0], color = 'g', marker = '.', ecolor = 'g', ls = 'none', alpha = 0.5, label = 'observed',)

	ax.plot(obs_R[1], rec_r_mag, 'r-', alpha = 0.5, label = 'r band')
	ax.errorbar( obs_R[1], obs_mag[1], yerr = obs_mag_err[1], color = 'r', marker = '.', ecolor = 'r', ls = 'none', alpha = 0.5,)

	ax.plot(obs_R[2], rec_i_mag, 'b-', alpha = 0.5, label = 'i band')
	ax.errorbar( obs_R[2], obs_mag[2], yerr = obs_mag_err[2], color = 'b', marker = '.', ecolor = 'b', ls = 'none', alpha = 0.5,)

	ax.set_xlabel('R[kpc]')
	ax.set_xscale('log')
	ax.set_ylabel('SB [mag / $ arcsec^2 $]')
	ax.set_ylim( 22, 34)
	ax.set_xlim( 1e1, 1e3 )
	ax.legend( loc = 1)
	ax.invert_yaxis()

	plt.savefig('/home/xkchen/figs/%s_ezgal_mod-mag_check.png' % cat_lis[mm], dpi = 300)
	plt.close()
"""

def cumu_mass_func(rp, surf_mass, N_grid = 50):

	NR = len(rp)
	intep_sigma_F = interp.interp1d( rp, surf_mass, kind = 'linear', fill_value = 'extrapolate',)

	cumu_mass = np.zeros( NR, )

	for ii in range( NR ):

		new_rp = np.logspace(0, np.log10( rp[ii] ), N_grid)
		new_mass = intep_sigma_F( new_rp )

		cumu_mass[ ii ] = integ.simps( 2 * np.pi * new_rp * new_mass, new_rp)

	return cumu_mass

dpt_mass = []
cumu_M_70 = []

Da_ref = Test_model.angular_diameter_distance( zs ).value
phy_area = 1**2 * Da_ref**2 * 10**6 / rad2asec**2

for mm in range( 2 ):

	# rec_dat = pds.read_csv('/home/xkchen/figs/%s_bc03-ssp_z0.02+0.008_sed_fit_test.csv' % cat_lis[mm] )
	# rec_dat = pds.read_csv('/home/xkchen/figs/%s_bc03-ssp_z0.008_sed_fit_test.csv' % (cat_lis[ mm ]) )
	rec_dat = pds.read_csv('/home/xkchen/figs/%s_bc03-ssp_in100-z0.02+0.008_out-0.008_sed_fit_test.csv' % (cat_lis[ mm ]) )
	rec_zfs, rec_Mag, rec_mass = np.array( rec_dat['fit_zf'] ), np.array( rec_dat['fit_Mag'] ), np.array( rec_dat['fit_Mass'] )
	rec_g_mag, rec_r_mag, rec_i_mag = np.array( rec_dat['mod_g_mag'] ), np.array( rec_dat['mod_r_mag'] ), np.array( rec_dat['mod_i_mag'])

	dpt_mass.append( rec_mass )

	idvx = dpt_r[mm] <= 70 #kpc
	cen_mass = cumu_mass_func( dpt_r[mm][idvx], 10**(rec_mass[idvx]) / phy_area, )

	cumu_M_70.append( cen_mass )

# plt.plot(dpt_r[0][:22], cumu_M_70[0], )
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

plt.figure()
for pp in range(2):

	plt.plot( dpt_r[pp], 10**(dpt_mass[pp]) / phy_area, color = color_s[pp], alpha = 0.5, label = fig_name[pp])

plt.legend( loc = 1)
plt.xlabel('R[kpc]')
plt.xscale('log')
plt.ylabel('$M_{\\ast}[ M_{\\odot} / kpc^2 ] $')
plt.yscale('log')
plt.xlim( 1e1, 1e3)
plt.ylim( 5e3, 2e8)
plt.savefig('/home/xkchen/figs/ezgal_mod-mass_view.png', dpi = 300)
plt.close()

