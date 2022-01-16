import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.patches import Circle, Ellipse, Rectangle

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
import scipy.signal as signal

from scipy import stats as sts
from scipy import integrate as integ
import seaborn as sns

from surface_mass_density import sigmam, sigmac
from surface_mass_density import input_cosm_model, cosmos_param, rhom_set
from surface_mass_density import misNFW_sigma_func, obs_sigma_func

# cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396

# psf_FWHM = [ 1.56, 1.67, 1.50 ] # arcsec
psf_FWHM = 1.32 # arcsec

band = ['r', 'g', 'i']
Mag_sun = [ 4.65, 5.11, 4.53 ]
l_wave = np.array( [6166, 4686, 7480] )

### === ### initial surface_mass_density.py module
input_cosm_model( get_model = Test_model )
cosmos_param()

def aveg_sigma_func(rp, sigma_arr, N_grid = 100):

	NR = len( rp )
	aveg_sigma = np.zeros( NR, dtype = np.float32 )

	tR = rp
	intep_sigma_F = interp.interp1d( tR , sigma_arr, kind = 'cubic', fill_value = 'extrapolate',)

	cumu_mass = np.zeros( NR, )
	lg_r_min = np.log10( np.min( rp ) / 10 )

	for ii in range( NR ):

		new_rp = np.logspace( lg_r_min, np.log10( tR[ii] ), N_grid)
		new_sigma = intep_sigma_F( new_rp )

		cumu_sigma = integ.simps( new_rp * new_sigma, new_rp)

		aveg_sigma[ii] = 2 * cumu_sigma / tR[ii]**2

	return aveg_sigma

def sersic_func(r, Ie, re, ndex):
	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir


### === ### data load, BG-sub SB and color
z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value
a_ref = 1 / (z_ref + 1)

Da_ref = Test_model.angular_diameter_distance( z_ref ).value
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec


# BG_path = '/home/xkchen/figs/re_measure_SBs/BGs/'
# SM_path = '/home/xkchen/figs/re_measure_SBs/SM_profile/'
# sdss_path = '/home/xkchen/tmp_run/data_files/figs/All_sample_BCG_pros/aveg_sdss_SB/'

BG_path = '/home/xkchen/figs/extend_bcgM_cat/BGs/'
SM_path = '/home/xkchen/figs/extend_bcgM_cat/SM_pros/'
sdss_path = '/home/xkchen/figs/extend_bcgM_cat/aveg_BCG_SB/'


nbg_tot_r = []
nbg_tot_mag, nbg_tot_mag_err = [], []

for kk in range( 3 ):
	with h5py.File( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-sub_SB.h5' % band[kk], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	nbg_tot_r.append( tt_r )
	nbg_tot_mag.append( tt_mag )
	nbg_tot_mag_err.append( tt_mag_err )

nbg_tot_r = np.array( nbg_tot_r )
nbg_tot_r = nbg_tot_r / 1e3


### === ### mass profile based on SDSS photometric SB profile
sdss_r, sdss_sb, sdss_err = [], [], []

for ii in range( 3 ):
	dat = pds.read_csv( sdss_path + 'total-sample_%s-band_Mean-jack_BCG_photo-SB_pros.csv' % band[ii] )
	ii_R = np.array( dat['R'] )
	ii_sb = np.array( dat['aveg_sb'] )
	ii_err = np.array( dat['aveg_sb_err'] )

	tt_mag = 22.5 - 2.5 * np.log10( ii_sb )
	tt_mag_err = 2.5 * ii_err / ( np.log(10) * ii_sb )

	id_Rx = ii_R < 100

	sdss_r.append( ii_R[ id_Rx ] / 1e3 )
	sdss_sb.append( tt_mag[ id_Rx ] )
	sdss_err.append( tt_mag_err[ id_Rx ] )


### === ### Z19, Z05, SB profile and color
Z05_r, Z05_sb, Z05_mag = [], [], []

for kk in range( 3 ):
	SB_tt = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/%s_band_BCG_ICL.csv' % band[kk],)
	R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
	R_obs = R_obs**4
	flux_obs = 10**( (22.5 - SB_obs + 2.5 * np.log10(pixel**2) ) / 2.5 ) / pixel**2

	Z05_r.append( R_obs )
	Z05_sb.append( flux_obs )
	Z05_mag.append( SB_obs )

Z05_r = np.array( Z05_r )
Z05_r = Z05_r / 1e3

##...Z05 with mask incompleteness correction
last_Z05_r, last_Z05_sb = [], []

for kk in range( 3 ):
	SB_tt = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/%s_band_sub_unmask.csv' % band[kk],)
	R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
	R_obs = R_obs**4

	last_Z05_r.append( R_obs )
	last_Z05_sb.append( SB_obs )

last_Z05_r = np.array( last_Z05_r )
last_Z05_r = last_Z05_r / 1e3

##...Z05, color
pdat_0 = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/Z05_g-r.csv')
r_05_0, g2r_05 = np.array(pdat_0['(R)^(1/4)']), np.array(pdat_0['g-r'])
r_05_0 = r_05_0**4
r_05_0 = r_05_0 / 1e3

pdat_2 = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/Z05_r-i.csv')
r_05_1, r2i_05 = np.array(pdat_2['(R)^(1/4)']), np.array(pdat_2['r-i'])
r_05_1 = r_05_1**4
r_05_1 = r_05_1 / 1e3

#...Z19 SB and color
z19_g_dat = np.loadtxt('/home/xkchen/mywork/ICL/data/Zhang_SB/pure_ICL_measurement_DES_g.txt')
cc_R = z19_g_dat[:,0]
cc_g_mag = z19_g_dat[:,1] 

z19_r_dat = np.loadtxt('/home/xkchen/mywork/ICL/data/Zhang_SB/pure_ICL_measurement_DES_r.txt')
cc_R = z19_r_dat[:,0]
cc_r_mag = z19_r_dat[:,1]

idx = cc_R <= 200
r_19, g2r_19 = cc_R[idx] / 1e3, cc_g_mag[idx] - cc_r_mag[idx]


## g-r color profile
c_dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_dered_color_profile.csv')
tot_c_R, tot_g2r, tot_g2r_err = np.array( c_dat['R_kpc'] ), np.array( c_dat['g-r'] ), np.array( c_dat['g-r_err'] )

sm_tot_g2r = signal.savgol_filter( tot_g2r, 5, 1)  # 7, 3
sm_tot_r = tot_c_R / 1e3
sm_tot_g2r_err = tot_g2r_err


## mass profile
dat = pds.read_csv( SM_path + 'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_mass-Lumi_with-dered.csv')
aveg_R, aveg_surf_m, aveg_surf_m_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])



### === figs
color_s = [ 'r', 'g', 'darkred' ]

fig = plt.figure( figsize = (10.6, 4.8) )
ax0 = fig.add_axes([0.07, 0.12, 0.42, 0.85])
ax1 = fig.add_axes([0.56, 0.12, 0.42, 0.85])

for kk in ( 2, 0, 1 ):

	ax0.plot( nbg_tot_r[kk], nbg_tot_mag[kk], ls = '-', color = color_s[kk], alpha = 0.75, label = '%s' % band[kk],)
	ax0.fill_between( nbg_tot_r[kk], y1 = nbg_tot_mag[kk] - nbg_tot_mag_err[kk], 
		y2 = nbg_tot_mag[kk] + nbg_tot_mag_err[kk], color = color_s[kk], alpha = 0.15,)


	if kk == 1:
		comp_r, comp_sb, comp_err = sdss_r[kk][:-1], sdss_sb[kk][:-1], sdss_err[kk][:-1]
	else:
		comp_r, comp_sb, comp_err = sdss_r[kk], sdss_sb[kk], sdss_err[kk]

	ax0.plot( comp_r, comp_sb, ls = ':', color = color_s[kk], alpha = 0.75,)
	ax0.fill_between( comp_r, y1 = comp_sb - comp_err, y2 = comp_sb + comp_err, color = color_s[kk], alpha = 0.15,)

	ax0.plot(Z05_r[kk], Z05_mag[kk], ls = '--', color = color_s[kk], alpha = 0.75,)


legend_1 = ax0.legend( [ 'This work (redMaPPer)', 'SDSS ($ \\mathrm{ \\tt{profMean} }$)', '$\\mathrm{Z05}$ (maxBCG)'], 
	loc = 1, frameon = False, fontsize = 14, markerfirst = False,)
legend_0 = ax0.legend( loc = 3, frameon = False, fontsize = 14,)

ax0.add_artist( legend_1 )

ax0.axvline( x = phyR_psf / 1e3, ls = '-.', linewidth = 3.5, color = 'k', alpha = 0.20,)
ax0.text( 3.1e-3, 27, s = 'PSF', fontsize = 22, rotation = 'vertical', color = 'k', alpha = 0.25, fontstyle = 'italic',)

ax0.set_ylim( 20, 33.5 )
ax0.invert_yaxis()

ax0.set_xlim( 3e-3, 1e0)
ax0.set_xscale('log')
ax0.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15)
ax0.set_ylabel('$ \\mu \; [mag \, / \, arcsec^2] $', fontsize = 15,)

x_tick_arr = [ 1e-2, 1e-1, 1e0]
tick_lis = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$']
ax0.set_xticks( x_tick_arr )
ax0.get_xaxis().set_major_formatter( ticker.FixedFormatter( tick_lis ) )
ax0.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)


ax1.plot( sm_tot_r, sm_tot_g2r, ls = '-', color = 'r', alpha = 0.75, label = 'This work (SDSS DR8)')
ax1.fill_between( sm_tot_r, y1 = sm_tot_g2r - sm_tot_g2r_err, y2 = sm_tot_g2r + sm_tot_g2r_err, color = 'r', alpha = 0.12,)

ax1.plot(r_19, g2r_19, ls = '--', color = 'c', alpha = 0.75, label = '$\\mathrm{Zhang}{+}2019$ (DES)')
ax1.plot(r_05_0, g2r_05, ls = '-.', color = 'k', alpha = 0.75, label = '$\\mathrm{Z05}$ (SDSS DR1)')

ax1.axvline( x = phyR_psf / 1e3, ls = '-.', linewidth = 3.5, color = 'k', alpha = 0.20,)
ax1.text( 3.1e-3, 1.25, s = 'PSF', fontsize = 22, rotation = 'vertical', color = 'k', alpha = 0.25, fontstyle = 'italic',)

ax1.legend( loc = 3, frameon = False, fontsize = 13.5,)

ax1.set_ylim( 0.90, 1.52 )
ax1.set_xlim( 3e-3, 1e0)
ax1.set_xscale('log')
ax1.set_ylabel('$ g - r $', fontsize = 17,)
ax1.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15)

x_tick_arr = [ 1e-2, 1e-1, 1e0]
tick_lis = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$']
ax1.set_xticks( x_tick_arr )
ax1.get_xaxis().set_major_formatter( ticker.FixedFormatter( tick_lis ) )
ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

# plt.savefig('/home/xkchen/total_sample_compare_to_Z05.png', dpi = 300)
plt.savefig('/home/xkchen/total_sample_compare_to_Z05.pdf', dpi = 300)
plt.close()

raise


##### === ##### lensing profile read and surface galaxy number density profile
id_dered = True
dered_str = 'with-dered_'

#. profiles fitting in large scale
fit_path = '/home/xkchen/figs/extend_bcgM_cat/SM_pros_fit/'

c_dat = pds.read_csv( fit_path + '%stotal_all-color-to-M_beyond-350kpc_xi2M-fit.csv' % dered_str,)
lg_fb_gi = np.array( c_dat['lg_fb_gi'] )[0]
lg_fb_gr = np.array( c_dat['lg_fb_gr'] )[0]
lg_fb_ri = np.array( c_dat['lg_fb_ri'] )[0]

c_dat = pds.read_csv( fit_path + '%stotal_all-color-to-M_beyond-300kpc_SG_N-fit.csv' % dered_str,)
lg_Ng_gi = np.array( c_dat['lg_fb_gi'] )[0]
lg_Ng_gr = np.array( c_dat['lg_fb_gr'] )[0]
lg_Ng_ri = np.array( c_dat['lg_fb_ri'] )[0]

const = 10**(-1 * lg_fb_gi)


##... satellite number density (M_i - 5 * log h < -19.43 is the galaxies Lumi. limit in number density profile measurement)
tot_Ng_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/data_Zhiwei/extend_bcgM/total_sigma-g_profile.csv')
tot_N_R, tot_Ng, tot_Ng_err = np.array(tot_Ng_dat['rbins']), np.array(tot_Ng_dat['sigma']), np.array(tot_Ng_dat['sigma_err'])

tot_Ng, tot_Ng_err = tot_Ng * h**2 / a_ref**2, tot_Ng_err * h**2 / a_ref**2
tot_N_R = tot_N_R * 1e3 / h / (1 + z_ref)

tot_Ng_int_F = interp.interp1d( tot_N_R, tot_Ng, kind = 'linear', fill_value = 'extrapolate',)

#...
Ng_sigma = tot_Ng_int_F( tot_N_R )
Ng_2Mpc = tot_Ng_int_F( 2e3 )
lg_Ng_sigma = np.log10( Ng_sigma - Ng_2Mpc )


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

#...
xi_rp = (lo_xi + hi_xi) / 2
tot_rho_m = ( xi_rp * 1e3 * rho_m ) / a_ref**2 * h
xi_to_Mf = interp.interp1d( lo_rp, tot_rho_m, kind = 'cubic',)

misNFW_sigma = xi_to_Mf( lo_rp )
sigma_2Mpc = xi_to_Mf( 2e3 )
lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )



### === weak lensing signal comparison
Mh_clus = 10**14.41  # M_sun
mrho_zref = rhom_set( z_ref )[1]  ## M_sun * h^2 / kpc^3
mrho_zref = mrho_zref * h**2      ## M_sun / kpc^3
R200m = ( 3 * Mh_clus / (4 * np.pi * mrho_zref * 200) )**(1 / 3)


N_grid = 250

##... 1-halo term only
v_m = 200 # rho_mean = 200 * rho_c * omega_m
c_mass = [5.87, 6.95]
Mh0 = [14.24, 14.24] # in unit M_sun / h
off_set = [230, 210] # in unit kpc / h
f_off = [0.37, 0.20]

mis_sigma = obs_sigma_func( lo_rp * h, np.mean(f_off), np.mean(off_set), z_ref, np.mean(c_mass), np.mean(Mh0), v_m)
mis_sigma = mis_sigma * h # M_sun / kpc^2
mean_mis_sigma = aveg_sigma_func( lo_rp, mis_sigma, N_grid = N_grid,)
mis_delta_sigma = ( mean_mis_sigma - mis_sigma ) * 1e-6 # M_sun / pc^2


##... lensing profile (measured on overall sample)
tot_sigma_dat = np.loadtxt('/home/xkchen/figs/Delta_sigma_all_sample/gglensing_decals_dr8_kNN_cluster_lowhigh.cat')
tt_Rc, tt_sigma, tt_sigma_err = tot_sigma_dat[:,0], tot_sigma_dat[:,1], tot_sigma_dat[:,2]
tt_calib_f = tot_sigma_dat[:,3]
tt_boost = tot_sigma_dat[:,4]

tt_Rp = tt_Rc / (1 + z_ref) / h
tt_sigma_p, tt_sigma_perr = tt_sigma * h * (1 + z_ref)**2, tt_sigma_err * h * (1 + z_ref)**2

inter_F_sigm = interp.interp1d( tt_Rp[1:], tt_sigma_p[1:], kind = 'linear', fill_value = 'extrapolate',)


##... subsample case (comoving coordinate)
hi_obs_dat = pds.read_csv('/home/xkchen/figs/Delta_sigma_all_sample/high_BCG_M_delta-sigm.txt')
hi_obs_R, hi_obs_Detsigm = np.array( hi_obs_dat['R'] ), np.array( hi_obs_dat['delta_sigma'] )

hi_obs_cov = np.loadtxt('/home/xkchen/figs/Delta_sigma_all_sample/high_BCG_M_delta-sigm_covmat.txt')
hi_obs_err = np.sqrt( np.diag( hi_obs_cov ) )

lo_obs_dat = pds.read_csv('/home/xkchen/figs/Delta_sigma_all_sample/low_BCG_M_delta-sigm.txt')
lo_obs_R, lo_obs_Detsigm = np.array( lo_obs_dat['R'] ), np.array( lo_obs_dat['delta_sigma'] )

lo_obs_cov = np.loadtxt('/home/xkchen/figs/Delta_sigma_all_sample/low_BCG_M_delta-sigm_covmat.txt')
lo_obs_err = np.sqrt( np.diag( lo_obs_cov ) )

aveg_Delta_sigm = ( hi_obs_Detsigm / hi_obs_err**2 + lo_obs_Detsigm / lo_obs_err**2 ) / ( 1 / hi_obs_err**2 + 1 / lo_obs_err**2 )

wi_lo = ( 1 / lo_obs_err**2 ) / ( 1 / hi_obs_err**2 + 1 / lo_obs_err**2 )
wi_hi = ( 1 / hi_obs_err**2 ) / ( 1 / hi_obs_err**2 + 1 / lo_obs_err**2 )

aveg_delta_err = np.sqrt( wi_lo**2 * lo_obs_err**2 + wi_hi**2 * hi_obs_err**2 )


##... central profile and delta_sigma_cen
p_dat = pds.read_csv( fit_path + '%stotal-sample_gri-band-based_mass-profile_cen-deV_fit.csv' % dered_str,)
c_Ie, c_Re, c_ne = np.array( p_dat['Ie'] )[0], np.array( p_dat['Re'] )[0], np.array( p_dat['ne'] )[0]

cen_SM = sersic_func( lo_rp, 10**c_Ie, c_Re, c_ne)
cen_aveg_sm = aveg_sigma_func( lo_rp, cen_SM, N_grid = N_grid )
cen_deta_sigm = ( cen_aveg_sm - cen_SM ) * 1e-6


#. delta sigma of large scale (1-halo term + 2-halo term)
R_aveg_sigma_1 = aveg_sigma_func( lo_rp, misNFW_sigma, N_grid = N_grid )
delt_sigm_1 = ( R_aveg_sigma_1 - misNFW_sigma ) * 1e-6 # no weight


#. estimation on the BCG+ICL
_cc_aveg_sigma = aveg_sigma_func( aveg_R, aveg_surf_m, N_grid = N_grid )
_cc_delta_sigma = ( _cc_aveg_sigma - aveg_surf_m ) * 1e-6



#### === fig
fig = plt.figure( figsize = ( 15.40, 4.8 ) )
ax0 = fig.add_axes( [0.05, 0.12, 0.275, 0.85] )
ax1 = fig.add_axes( [0.38, 0.12, 0.275, 0.85] )
ax2 = fig.add_axes( [0.71, 0.12, 0.275, 0.85] )

ax0.errorbar( aveg_R / 1e3, aveg_surf_m, yerr = aveg_surf_m_err, ls = 'none', marker = 'o', ms = 8, mec = 'k', mfc = 'none', alpha = 0.85, 
	capsize = 3, ecolor = 'k', label = '$\\mathrm{ {BCG} \, {+} \, {ICL} }$',)

ax0.plot( lo_rp / 1e3, 10**lg_M_sigma * 10**lg_fb_gi, ls = '-', color = 'k', alpha = 0.65, 
	label = '$ \\gamma \, \\Sigma_{m} $',)
ax0.plot( tot_N_R / 1e3, (Ng_sigma - Ng_2Mpc) * 10**lg_Ng_gi * 1e-6, ls = '--', color = 'k', alpha = 0.75, 
	label = '$ M_{\\ast}^{\\mathrm{loss} } \\times \\Sigma_{g} $',)

_handles, _labels = ax0.get_legend_handles_labels()
ax0.legend( handles = _handles[::-1], labels = _labels[::-1], loc = 1, frameon = False, fontsize = 14, markerfirst = False,)

# ax0.axvline( x = R200m / 1e3, ls = ':', color = 'k', ymin = 0.0, ymax = 0.225,)

ax0.set_xlim( 9e-3, 2e0)
ax0.set_xscale('log')
ax0.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15)
ax0.set_yscale('log')
ax0.set_ylim( 1e4, 3e8 )
ax0.set_ylabel( '$\\Sigma_{\\ast}^{ \\mathrm{ \\tt{B+I} } } \; [M_{\\odot} \, / \, \\mathrm{k}pc^2]$', fontsize = 15,)

ax0.set_xticks([ 1e-2, 1e-1, 1e0, 2e0])
ax0.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$', '$\\mathrm{2}$'] )
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)


# ax1.errorbar( tt_Rp[1:], tt_sigma_p[1:] * tt_boost[1:], yerr = tt_sigma_perr[1:], xerr = None, color = 'k', marker = 's', ms = 8, ls = 'none', 
# 	ecolor = 'k', alpha = 0.75, mec = 'k', mfc = 'none', capsize = 2, label = 'Observed')

ax1.errorbar( lo_obs_R[1:] / (1 + z_ref) / h, aveg_Delta_sigm[1:] * h / a_ref**2, yerr = aveg_delta_err[1:] * h / a_ref**2, xerr = None, ms = 8, color = 'k', 
	marker = 's', ls = 'none', ecolor = 'k', alpha = 0.75, mec = 'k', mfc = 'none', capsize = 2, label = '$\\mathrm{ {Weak} \; {Lensing} }$',)

ax1.plot( lo_rp / 1e3, cen_deta_sigm, ls = ':', color = 'k', label = '$\\Delta \\Sigma_{\\ast}^{\\mathrm{deV} }$',)
ax1.plot( lo_rp / 1e3, delt_sigm_1 + cen_deta_sigm, ls = '--', color = 'k', label = '$\\Delta \\Sigma_{m}{+} \\Delta \\Sigma_{\\ast}^{\\mathrm{deV} }$')
ax1.plot( lo_rp / 1e3, delt_sigm_1, ls = '-', color = 'k', label = '$M_{h}{=}10^{14.41} \, M_{\\odot}$',)

_handles, _labels = ax1.get_legend_handles_labels()
ax1.legend( handles = _handles[::-1], labels = _labels[::-1], loc = 1, frameon = False, fontsize = 14, markerfirst = False,)

ax1.axvline( x = R200m / 1e3, ls = '-', color = 'k', ymin = 0.0, ymax = 0.225,)
ax1.text( R200m / 1e3 + 0.001, 1.5e0, s = '$R_{ \\mathrm{200m} }$', fontsize = 16, rotation = -90, color = 'k', alpha = 0.4, fontstyle = 'italic',)

ax1.set_xlim( 9e-3, 2e1)
ax1.set_xscale('log')
ax1.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15)
ax1.set_yscale('log')

ax1.set_ylim( 0.9e0, 5e2)
ax1.set_ylabel('$\\Delta \\Sigma \; [M_{\\odot} \, / \, pc^2]$', fontsize = 15,)

ax1.set_xticks([ 1e-2, 1e-1, 1e0, 1e1, 2e1])
ax1.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$', '$\\mathrm{10}$', '$\\mathrm{20}$'])
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)


ax2.errorbar( tot_N_R / 1e3, Ng_sigma, yerr = tot_Ng_err, xerr = None, color = 'k', marker = 's', ms = 8, ls = 'none', 
	ecolor = 'k', alpha = 0.75, mec = 'k', mfc = 'none', capsize = 2, label = 'Galaxies ($M_{i}{<}{-}%.2f$)' % np.abs( (-19.43 + 5 * np.log10(h) ) ),)
ax2.axvline( x = R200m / 1e3, ls = '-', color = 'k', ymin = 0.0, ymax = 0.225,)
ax2.text( R200m / 1e3 + 0.001, 4e-1, s = '$R_{ \\mathrm{200m} }$', fontsize = 16, rotation = -90, color = 'k', alpha = 0.4, fontstyle = 'italic',)

ax2.set_xlim( 9e-3, 2e1)
ax2.set_xscale('log')
ax2.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15)
ax2.set_yscale('log')
ax2.set_ylim( 2.9e-1, 2e2 )
ax2.set_ylabel('$\\Sigma_{g} \; [ \\# \, / \, \\mathrm{M}pc^{2}]$', fontsize = 15,)
ax2.legend( loc = 1, frameon = False, fontsize = 14, markerfirst = False,)

ax2.set_xticks([ 1e-2, 1e-1, 1e0, 1e1, 2e1])
ax2.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$', '$\\mathrm{10}$', '$\\mathrm{20}$'])
ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

# plt.savefig('/home/xkchen/%stotal_sample_SB_SM.png' % dered_str, dpi = 300)
plt.savefig('/home/xkchen/total_sample_SB_SM.pdf', dpi = 300)
plt.close()

