import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp
import scipy.stats as sts

import astropy.units as U
import astropy.constants as C

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
from scipy.stats import binned_statistic as binned
from scipy import optimize

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle

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

Jky = 10**(-23) # erg * s^(-1) * cm^(-2) * Hz^(-1)
F0 = 3.631 * 10**(-6) * Jky

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

## observed profile
#path = '/home/xkchen/project/tmp_mcmc/'
path = '/home/xkchen/Downloads/spec-z_catalog_21_02_28_match/match_2_28_BG_pros/'


cat_lis = ['high_BCG_star-Mass', 'low_BCG_star-Mass', 'tot-M-star']
fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$'] ## or line name

color_s = ['g', 'r', 'b']

tot_r, tot_sb, tot_err = [], [], []

rank = 0

for kk in range( 3 ):

	with h5py.File( path + '%s_%s-band_BG-sub_SB.h5' % (cat_lis[ rank ], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tot_r.append( tt_r )
	tot_sb.append( tt_sb )
	tot_err.append( tt_err )

tot_r = np.array( tot_r )
tot_sb = np.array( tot_sb )
tot_err = np.array( tot_err )

tot_Fv = np.zeros( (3, len(tot_r[0]) ), dtype = np.float32)
tot_Fv_err = np.zeros( (3, len(tot_r[0]) ), dtype = np.float32)

print(tot_Fv.shape)

for kk in range( 3 ):
	tot_Fv[kk] = tot_sb[kk] * F0
	tot_Fv_err[kk] = tot_err[kk] * F0

obs_mag = []
obs_mag_err = []
obs_R = []
for kk in range( 3 ):

	idux = (tot_r[kk] >= 10) & (tot_r[kk] <= 1e3)

	obs_R.append( tot_r[kk][idux] )

	tt_mag = 22.5 - 2.5 * np.log10( tot_sb[kk] )
	tt_mag_err = 2.5 * tot_err[kk] / ( np.log(10) * tot_sb[kk] )

	obs_mag.append( tt_mag[idux] )
	obs_mag_err.append( tt_mag_err[idux] )

obs_mag = np.array( obs_mag )
obs_mag_err = np.array( obs_mag_err )
obs_R = np.array( obs_R )

Dl_ref = Test_model.luminosity_distance( 0.25 ).value
obs_Mag = obs_mag - 5 * np.log10( Dl_ref * 10**6 / 10)

## ezgal for SED fitting
import emcee
import ezgal

def pdf_func(data_arr, bins_arr,):

	N_pix, edg_f = binned(data_arr, data_arr, statistic = 'count', bins = bins_arr)[:2]
	pdf_pix = (N_pix / np.sum(N_pix) ) / (edg_f[1] - edg_f[0])
	pdf_err = ( np.sqrt(N_pix) / np.sum(N_pix) ) / (edg_f[1] - edg_f[0])
	f_cen = 0.5 * ( edg_f[1:] + edg_f[:-1])

	id_zero = N_pix < 1.
	pdf_arr = pdf_pix[ id_zero == False]
	err_arr = pdf_err[ id_zero == False]
	pdf_x = f_cen[ id_zero == False]

	return pdf_arr, err_arr, pdf_x

model_me = ezgal.model('bc03_ssp_z_0.008_chab.model')
model_me.add_filter('sloan_g')
model_me.add_filter('sloan_r')
model_me.add_filter('sloan_i')
model_me.set_cosmology(Om = Test_model.Om0, Ol = Test_model.Odm0, h = Test_model.h,)

# model_0 = ezgal.model('bc03_ssp_z_0.008_chab.model')
# model_1 = ezgal.model('bc03_ssp_z_0.02_chab.model')

# model_me = ezgal.weight( 97 ) * model_1
# model_me = model_me + ezgal.weight( 3 ) * model_0
# model_me.set_cosmology( Om = Test_model.Om0, Ol = Test_model.Ode0, h = Test_model.h, )
# model_me.add_filter('sloan_g')
# model_me.add_filter('sloan_r')
# model_me.add_filter('sloan_i')

zf = 5.5

predict_mag = np.zeros( len(obs_R[0]), dtype = np.float32 )
predict_err = np.zeros( len(obs_R[0]), dtype = np.float32 )
# loop for all of the obs point

import time

for tt in range( len(obs_R[0]) ):

	#dpt_mag = obs_mag[:, tt]
	dpt_mag = obs_Mag[:, tt] ## use absolute magnitude

	dpt_err = obs_mag_err[:, tt]

	def likelihood_func(p, y, zs, zf, yerr):

		mock_mag = p

		model_me.set_normalization('sloan_r', zs, mock_mag,)
		g_mag_mock, r_mag_mock, i_mag_mock = model_me.get_absolute_mags(zf, filters = ['sloan_g', 'sloan_r', 'sloan_i'], zs = zs, ab = True)[0]

		p_return = -1 * ( (g_mag_mock - y[0])**2 / yerr[0]**2 + (r_mag_mock - y[1])**2 / yerr[1]**2 + 
							(i_mag_mock - y[2])**2 / yerr[2]**2 )

		# print( p_return )
		# time.sleep( 5 )

		return p_return

	def prior_p_func(p, y, zs, zf, yerr):

		delta_mag = np.abs( y[1] - p )

		if delta_mag <= 2.5:
			return 0
		return -np.inf

	def post_p_func(p, y, zs, zf, yerr):

		pre_p = prior_p_func(p, y, zs, zf, yerr)

		if not np.isfinite( pre_p ):
			return -np.inf
		return prior_p_func(p, y, zs, zf, yerr) + likelihood_func(p, y, zs, zf, yerr)

	def gau_func(x, mu, sigma):
		return sts.norm.pdf(x, mu, sigma)

	n_dim = 1
	n_walk = 50
	L_chain = 2000

	zs = 0.25

	## save the process data
	# out_files = path + 'M_estimate/%s_%d_obs-R_mcmc_fit.h5' % (cat_lis[ rank ], tt)
	# backend = emcee.backends.HDFBackend( out_files )
	# backend.reset( n_walk, n_dim )

	ini_mags = dpt_mag[1] + 10 * dpt_err[1] * np.random.randn( n_walk,1 )

	# sampler = emcee.EnsembleSampler(n_walk, n_dim, post_p_func, args = (dpt_mag, zs, zf, dpt_err), backend = backend,)
	sampler = emcee.EnsembleSampler(n_walk, n_dim, post_p_func, args = (dpt_mag, zs, zf, dpt_err),)
	sampler.run_mcmc(ini_mags, L_chain, progress = True,)

	## fig the result
	try:
		tau = sampler.get_autocorr_time()
		samples = sampler.get_chain( discard = np.int( 2.5 * tau ), thin = np.int( 0.5 * tau ), flat = True)
	except:
		samples = sampler.get_chain( discard = 100, thin = 20, flat = True,)

	#bin_edg = np.linspace( dpt_mag[1] - 20 * dpt_err[1], dpt_mag[1] + 20 * dpt_err[1], 150)
	bin_edg = np.linspace( samples.min(), samples.max(), 1000)

	pdf_arr, pdf_err, pdf_x = pdf_func( samples.T[0], bin_edg )

	popt, pcov = optimize.curve_fit(gau_func, pdf_x, pdf_arr, p0 = [np.median(samples.T), np.sqrt( np.var(samples.T) )], )#sigma = pdf_err,)

	fit_G_line = gau_func(pdf_x, popt[0], popt[1])

	plt.figure()
	plt.hist(samples, bins = bin_edg, density = True, color = 'b', alpha = 0.5,)
	plt.plot(pdf_x, fit_G_line, color = 'r', alpha = 0.5, ls = '-',)

	plt.axvline(popt[0], color = 'g', ls = '-', label = 'model')
	plt.axvline(popt[0] + popt[1], color = 'g', ls = ':',)
	plt.axvline(popt[0] - popt[1], color = 'g', ls = ':',)
	plt.errorbar( dpt_mag[1], 0.5 * pdf_arr.max(), yerr = None, color = 'r', xerr = dpt_err[1], 
		marker = '.', ecolor = 'r', ls = 'none', label = '$observation}$',)

	plt.legend( loc = 2)
	plt.xlabel( '$mag_{r, 0.25}$' )
	plt.ylabel('pdf' )

	#plt.xlim(dpt_mag[1] - 1.5, dpt_mag[1] + 1.5)
	plt.xlim(dpt_mag[1] - 0.5, dpt_mag[1] + 0.5)
	plt.savefig('/home/xkchen/figs/%s_%d_R-bin_r-mag_predict.png' % (cat_lis[ rank ], tt), dpi = 300)
	plt.close()

	predict_mag[ tt ] = popt[0]
	predict_err[ tt ] = popt[1]

	raise

## save
keys = ['mag', 'err']
values = [ predict_mag, predict_err ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv(path + 'M_estimate/%s_R-bin_r-mag_fit.csv' % cat_lis[ rank ],)

print('%d rank finished !' % rank)

p_dat = pds.read_csv(path + 'M_estimate/%s_R-bin_r-mag_fit.csv' % cat_lis[ rank ],)
predict_mag, predict_err = np.array(p_dat['mag']), np.array(p_dat['err'])

## mass estimate
model_out = ezgal.model('bc03_burst_0.1_z_0.008_chab.model')
model_out.add_filter('sloan_g')
model_out.add_filter('sloan_r')
model_out.add_filter('sloan_i')
model_out.set_cosmology(Om = Test_model.Om0, Ol = Test_model.Odm0, h = Test_model.h,)

zf = 5.5
zs = 0.25

pre_dict_Mass = np.zeros( len(obs_R[0]), dtype = np.float32 )
pre_dict_r_mag = np.zeros( len(obs_R[0]), dtype = np.float32 )
pre_dict_g_mag = np.zeros( len(obs_R[0]), dtype = np.float32 )
pre_dict_i_mag = np.zeros( len(obs_R[0]), dtype = np.float32 )
pre_dict_chi2 = np.zeros( len(obs_R[0]), dtype = np.float32 )

for ll in range( len(obs_R[0]) ):

	#dpt_mag = obs_mag[:,ll]
	dpt_mag = obs_Mag[:, ll] ## use absolute magnitude

	dpt_err = obs_mag_err[:,ll]

	#model_out.set_normalization('sloan_r', zs, predict_mag[ll], apparent = True,)
	model_out.set_normalization('sloan_r', zs, predict_mag[ll],)

	dt_Mass = model_out.get_masses(zf, zs) * (model_out.get_normalization( zf, flux = True),)
	pre_dict_Mass[ll] = np.log10( dt_Mass[0][0] )

	dt_mag = model_out.get_apparent_mags( zf = zf, filters = ['sloan_g', 'sloan_r', 'sloan_i'], 
						normalize = True, ab = True, zs = zs,)[0]

	pre_dict_g_mag[ll] = dt_mag[0]
	pre_dict_r_mag[ll] = dt_mag[1]
	pre_dict_i_mag[ll] = dt_mag[2]

	pre_dict_chi2[ll] = ( (dpt_mag[0] - dt_mag[0])**2 / dpt_err[0]**2 + (dpt_mag[1] - dt_mag[1])**2 / dpt_err[0]**2 + 
							(dpt_mag[2] - dt_mag[2])**2 / dpt_err[2]**2 )

## save the data
keys = ['R_obs', 'r_mag', 'g_mag', 'i_mag', 'chi2', 'lg(Mass)']
values = [ obs_R[0], pre_dict_r_mag, pre_dict_g_mag, pre_dict_i_mag, pre_dict_chi2, pre_dict_Mass ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv(path + 'M_estimate/%s_R-bin_mag-mass_estimate.csv' % cat_lis[ rank ],)


## figure the result
# pdat = pds.read_csv('/home/xkchen/jupyter/low_BCG_star-Mass_R-bin_r-mag_fit.csv')
# lo_magr = np.array( pdat['mag'] )
# lo_magr_err = np.array( pdat['err'] )

# pdat = pds.read_csv('/home/xkchen/jupyter/high_BCG_star-Mass_R-bin_r-mag_fit.csv')
# hi_magr = np.array( pdat['mag'] )
# hi_magr_err = np.array( pdat['err'] )

# pdat = pds.read_csv('/home/xkchen/jupyter/tot-M-star_R-bin_r-mag_fit.csv')
# tot_magr = np.array( pdat['mag'] )
# tot_magr_err = np.array( pdat['err'] )

# cc_mag, cc_err = [], []

# for kk in range( 3 ):

# 	with h5py.File( path + '%s_r-band_BG-sub_SB.h5' % cat_lis[kk], 'r') as f:
# 		tt_r = np.array(f['r'])
# 		tt_sb = np.array(f['sb'])
# 		tt_err = np.array(f['sb_err'])

# 	idux = ( tt_r >= 10 ) & ( tt_r <= 1e3 )

# 	tt_mag = 22.5 - 2.5 * np.log10( tt_sb[ idux ] )
# 	tt_mag_err = 2.5 * tt_err[idux] / ( np.log(10) * tt_sb[idux] )

# 	tt_mag = tt_mag - 5 * np.log10( Dl_ref * 10**6 / 10)

# 	cc_mag.append( tt_mag )
# 	cc_err.append( tt_mag_err )

# plt.figure()
# plt.errorbar( cc_mag[0], cc_mag[0], yerr = cc_err[0], color = 'b', marker = '', ecolor = 'b', ls = '-', 
# 	alpha = 0.5, label = '1:1, low $M_{\\ast}$',)
# plt.errorbar( cc_mag[1], cc_mag[1] + 2, yerr = cc_err[1], color = 'r', marker = '', ecolor = 'r', ls = '-', 
# 	alpha = 0.5, label = 'high $M_{\\ast}$',)
# plt.errorbar( cc_mag[2], cc_mag[2] + 4, yerr = cc_err[2], color = 'k', marker = '', ecolor = 'k', ls = '-', 
# 	alpha = 0.5, label = 'total',)

# plt.errorbar( cc_mag[0], lo_magr, yerr = lo_magr_err, color = 'b', marker = '.', ecolor = 'b', ls = 'none', 
# 	alpha = 0.5, label = 'SED',)
# plt.errorbar( cc_mag[1], hi_magr + 2, yerr = hi_magr_err, color = 'r', marker = '.', ecolor = 'r', ls = 'none', alpha = 0.5,)
# plt.errorbar( cc_mag[2], tot_magr + 4, yerr = tot_magr_err, color = 'k', marker = '.', ecolor = 'k', ls = 'none', alpha = 0.5,)

# plt.legend( loc = 2)
# plt.xlabel('$ observed \; SB_{r} [mag/arcsec^2]$')
# plt.ylabel('$ predicted \; SB_{r} [mag/arcsec^2]$')
# plt.savefig('sed_mag_compare.png', dpi = 300)
# plt.close()


# ## mass estimate
# Da_ref = Test_model.angular_diameter_distance( z_ref ).value
# phy_S = pixel**2 * Da_ref**2 * 10**6 / rad2asec**2

# pdat = pds.read_csv('/home/xkchen/high_BCG_star-Mass_R-bin_mag-mass_estimate.csv')
# hi_obs_R = np.array(pdat['R_obs'])
# hi_mode_mass = np.array(pdat['lg(Mass)'])

# pdat = pds.read_csv('/home/xkchen/low_BCG_star-Mass_R-bin_mag-mass_estimate.csv')
# low_obs_R = np.array(pdat['R_obs'])
# low_mode_mass = np.array(pdat['lg(Mass)'])

# pdat = pds.read_csv('/home/xkchen/tot-M-star_R-bin_mag-mass_estimate.csv')
# tot_obs_R = np.array(pdat['R_obs'])
# tot_mode_mass = np.array(pdat['lg(Mass)'])

# plt.figure()
# ax = plt.subplot(111)

# ax.plot(hi_obs_R, 10**hi_mode_mass / phy_S, ls = '--', color = 'r', alpha = 0.5, label = 'high $M_{\\ast}$',)
# ax.plot(low_obs_R, 10**low_mode_mass / phy_S, ls = '--', color = 'b', alpha = 0.5, label = 'low $M_{\\ast}$',)
# ax.plot(tot_obs_R, 10**tot_mode_mass / phy_S, ls = '-', color = 'k', alpha = 0.5, label = 'total',)

# ax.set_xlim(1e1, 1.1e3)
# ax.set_xscale('log')
# ax.set_xlabel('R [kpc]')
# ax.set_ylabel('$M_{\\ast} [M_{\\odot} / kpc^2]$')
# ax.set_yscale('log')
# ax.set_ylim(1e5, 1e9)
# ax.legend( loc = 3 )
# ax.grid(which = 'both', axis = 'both', alpha = 0.25,)

# plt.subplots_adjust(left = 0.15,)
# plt.savefig('mass_estimate_compare.png', dpi = 300)
# plt.close()

