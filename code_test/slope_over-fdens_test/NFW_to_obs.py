import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
import pandas as pds

import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
from scipy.optimize import curve_fit
from scipy import signal
from scipy import interpolate as interp

from surface_mass_density import sigma_m, sigma_m_c
from fig_out_module import arr_jack_func, arr_slope_func

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
rad2arcsec = U.rad.to(U.arcsec)
Lsun = C.L_sun.value*10**7
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3.631 * 10**(-6) * Jy

pixel = 0.396

### initial set
Mh_0 = 14.21 # unit M_sun / h
C0 = 3.84

Mh_1 = 14.23 
C1 = 5.49

Z0 = 0.25 # 0.242
z_ref = 0.25

a0 = 1 / (1 + Z0)
a_ref = 1 / (1 + z_ref)

Nbins = 150

low_R200, low_mass_rho, low_R = sigma_m(Mh_0, Z0, Nbins, C0)
low_R = low_R / a0
low_R200 = low_R200 / a0

high_R200, high_mass_rho, high_R = sigma_m(Mh_1, Z0, Nbins, C1)
high_R = high_R / a0
high_R200 = high_R200 / a0

def SB_fit(r, I_e, r_e,):
	"""
	SB profile : Mo. galaxy evolution and evolution, Chapter 2, eq. 2.23
	"""
	ndex = 4
	belta_n = 2 * ndex - 0.324
	f_n = - belta_n * ( r / r_e)**(1 / ndex) + belta_n
	I_r = I_e * np.exp( f_n )

	return I_r

def mode_flux_densi(rho_2d, m2l, z):

	m0 = Lsun / (np.pi * 4 * m2l * h)
	m1 = rho_2d / (z + 1)**4
	
	f_dens = m0 * m1 * kpc2cm**(-2) * rad2arcsec**(-2)

	fdens = f_dens / f0

	return fdens

def mode_SB(rho_2d, m2l, z):
	"""
	rho_2d : surface mass density (unit: M_sun / kpc^2)
	m2l : mass-to-light ratio
	z : given redshift
	"""
	lumi = (rho_2d * h / m2l) * 10**(-6)
	SB_pros = -2.5 * np.log10( lumi ) + 10 * np.log10(1 + z) - 21.572

	return SB_pros

Mag_sun = 5
shift_Mag = 44

M2L = 2e6
shift_dens = 10**(-12)

mock_low_SB = mode_SB(low_mass_rho, M2L, Z0)
mock_low_SB = mock_low_SB + Mag_sun + shift_Mag
mock_hi_SB = mode_SB(high_mass_rho, M2L, Z0)
mock_hi_SB = mock_hi_SB + Mag_sun + shift_Mag

mock_low_dens = mode_flux_densi(low_mass_rho, M2L, Z0)
mock_low_dens = shift_dens * mock_low_dens
mock_hi_dens = mode_flux_densi(high_mass_rho, M2L, Z0)
mock_hi_dens = shift_dens * mock_hi_dens

idr = np.abs(low_R - low_R200)
idrx = np.where(idr == idr.min())[0][0]
sub_low_dens = mock_low_dens - mock_low_dens[idrx]

idr = np.abs(high_R - high_R200)
idrx = np.where(idr == idr.min())[0][0]
sub_hi_dens = mock_hi_dens - mock_hi_dens[idrx]

interp_low_fdens = interp.interp1d(low_R, sub_low_dens, kind = 'cubic')
interp_hi_fdens = interp.interp1d(high_R, sub_hi_dens, kind = 'cubic')


######### obs. case

with h5py.File('tmp_test/low-BCG-star-Mass_Mean_jack_SB-pro_z-ref_with-selection_sub-BG_on-jk-sub.h5', 'r') as f:
	obs_low_r = np.array(f['r'])
	obs_low_sb = np.array(f['sb'])
	obs_low_err = np.array(f['sb_err'])
'''
with h5py.File('/home/xkchen/jupyter/stack/low-BCG-star-Mass_Mean_jack_SB-pro_z-ref_with-selection.h5', 'r') as f:
	alt_r_arr = np.array(f['r'])
	alt_sb_arr = np.array(f['sb'])
	alt_sb_err = np.array(f['sb_err'])
id_Nul = alt_sb_arr > 0
obs_low_r = alt_r_arr[id_Nul]
obs_low_sb = alt_sb_arr[id_Nul]
obs_low_err = alt_sb_err[id_Nul]
'''

obs_low_r = (obs_low_r / a_ref) * h
obs_low_fdens = obs_low_sb * ( (1 + z_ref) / (1 + Z0) )**4
obs_low_ferr = obs_low_err * ( (1 + z_ref) / (1 + Z0) )**4

idvx = (obs_low_r >= 12.7) & ( obs_low_r <= 17)
idR = obs_low_r[idvx]
idSB = obs_low_fdens[idvx]
idSB_err = obs_low_err[idvx]

Re = 13
mu_e = 5e-1

po = np.array([mu_e, Re])
popt, pcov = curve_fit(SB_fit, idR, idSB, p0 = po, bounds = ([1e-1, 11], [1e0, 15]), sigma = idSB_err, method = 'trf')

Ie, Re = popt
low_fit_line = SB_fit(obs_low_r, Ie, Re,)

id_lim = ( obs_low_r <= low_R.max() ) & ( obs_low_r >= low_R.min() )
lim_low_r = obs_low_r[ id_lim ]
lim_low_sb = obs_low_fdens[ id_lim ]
lim_low_sb_err = obs_low_ferr[ id_lim ]
eff_low_fdens = interp_low_fdens( lim_low_r )
low_resi_fdens = lim_low_sb - eff_low_fdens


with h5py.File('tmp_test/high-BCG-star-Mass_Mean_jack_SB-pro_z-ref_with-selection_sub-BG_on-jk-sub.h5', 'r') as f:
	obs_hi_r = np.array(f['r'])
	obs_hi_sb = np.array(f['sb'])
	obs_hi_err = np.array(f['sb_err'])
'''
with h5py.File('/home/xkchen/jupyter/stack/high-BCG-star-Mass_Mean_jack_SB-pro_z-ref_with-selection.h5', 'r') as f:
	alt_r_arr = np.array(f['r'])
	alt_sb_arr = np.array(f['sb'])
	alt_sb_err = np.array(f['sb_err'])
id_Nul = alt_sb_arr > 0
obs_hi_r = alt_r_arr[id_Nul]
obs_hi_sb = alt_sb_arr[id_Nul]
obs_hi_err = alt_sb_err[id_Nul]
'''

obs_hi_r = (obs_hi_r  / a_ref) * h
obs_hi_fdens = obs_hi_sb * ( (1 + z_ref) / (1 + Z0) )**4
obs_hi_ferr = obs_hi_err * ( (1 + z_ref) / (1 + Z0) )**4

idux = (obs_hi_r >= 12.7) & ( obs_hi_r <= 17)
idR = obs_hi_r[idvx]
idSB = obs_hi_fdens[idvx]
idSB_err = obs_hi_err[idvx]

Re = 13
mu_e = 5e-1

po = np.array([mu_e, Re])
popt, pcov = curve_fit(SB_fit, idR, idSB, p0 = po, bounds = ([1e-1, 11], [1e0, 15]), sigma = idSB_err, method = 'trf')

Ie, Re = popt
hi_fit_line = SB_fit(obs_hi_r, Ie, Re,)

id_lim = ( obs_hi_r <= high_R.max() ) & ( obs_hi_r >= high_R.min() )
lim_hi_r = obs_hi_r[ id_lim ]
lim_hi_sb = obs_hi_fdens[ id_lim ]
lim_hi_sb_err = obs_hi_ferr[ id_lim ]
eff_hi_fdens = interp_low_fdens( lim_hi_r )
hi_resi_fdens = lim_hi_sb - eff_hi_fdens


## Z05 result
SB_tt = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/r_band_BCG_ICL.csv')
R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
R_obs = R_obs**4
flux_obs = 10**( (22.5 - SB_obs + 2.5 * np.log10(pixel**2) ) / 2.5 ) / pixel**2
R_obs = R_obs * h / a_ref

## SDSS BCG pros (mean pros.)
low_sdss_bcg_pro = pds.read_csv('tmp_test/low-BCG-star-Mass_aveg-BCG-pros.csv')
low_bcg_r = np.array( low_sdss_bcg_pro['R_ref'] )
low_bcg_mu = np.array( low_sdss_bcg_pro['SB_fdens'] )
low_bcg_mu_err = np.array( low_sdss_bcg_pro['SB_fdens_err'] )
low_bcg_r = low_bcg_r * h / a_ref

hi_sdss_bcg_pro = pds.read_csv('tmp_test/high-BCG-star-Mass_aveg-BCG-pros.csv')
hi_bcg_r = np.array( hi_sdss_bcg_pro['R_ref'] )
hi_bcg_mu = np.array( hi_sdss_bcg_pro['SB_fdens'] )
hi_bcg_mu_err = np.array( hi_sdss_bcg_pro['SB_fdens_err'] )
hi_bcg_r = hi_bcg_r * h / a_ref

## slope profiles
low_kdat = pds.read_csv('tmp_test/low-BCG-star-Mass_mean_SB-pros_slope.csv')
low_sign_r, low_dsign_dlgr, low_k_err = np.array(low_kdat['R']), np.array(low_kdat['dlogsb_dlogr']), np.array(low_kdat['slop_err'])
low_sign_r = low_sign_r * h / a_ref

hi_kdat = pds.read_csv('tmp_test/high-BCG-star-Mass_mean_SB-pros_slope.csv')
hi_sign_r, hi_dsign_dlgr, hi_k_err = np.array(hi_kdat['R']), np.array(hi_kdat['dlogsb_dlogr']), np.array(hi_kdat['slop_err'])
hi_sign_r = hi_sign_r * h / a_ref


NBG_low_kdat = pds.read_csv('tmp_test/low-BCG-star-Mass_mean_BG-sub_SB-pros_slope.csv')
NBG_low_sign_r, NBG_low_dsign_dlgr, NBG_low_k_err = ( np.array(NBG_low_kdat['R']), 
													np.array(NBG_low_kdat['dlogsb_dlogr']), np.array(NBG_low_kdat['slop_err']) )
NBG_low_sign_r = NBG_low_sign_r * h / a_ref

NBG_hi_kdat = pds.read_csv('tmp_test/high-BCG-star-Mass_mean_BG-sub_SB-pros_slope.csv')
NBG_hi_sign_r, NBG_hi_dsign_dlgr, NBG_hi_k_err = ( np.array(NBG_hi_kdat['R']), 
												np.array(NBG_hi_kdat['dlogsb_dlogr']), np.array(NBG_hi_kdat['slop_err']) )
NBG_hi_sign_r = NBG_hi_sign_r * h / a_ref

### truly ICL slope (the BCG region not real, just sersic fitting)
low_icl_resi = obs_low_fdens - low_fit_line
hi_icl_resi = obs_hi_fdens - hi_fit_line

wind_len = 9
poly_order = 3

low_icl_res_R, low_icl_res_slope = arr_slope_func(low_icl_resi, obs_low_r, wind_len, poly_order, id_log = True,)
hi_icl_res_R, hi_icl_res_slope = arr_slope_func(hi_icl_resi, obs_hi_r, wind_len, poly_order, id_log = True,)

def power_fit(r, I_0, alpha):

	I_r = I_0 * r**( alpha )

	return I_r

id_Rx = ( obs_low_r >= 10) & ( obs_low_r <= 40)
low_lim_R = obs_low_r[ id_Rx ]
low_lim_sb = obs_low_fdens[ id_Rx ]
low_lim_err = obs_low_ferr[ id_Rx ]

id_Rx = ( obs_hi_r >= 10) & ( obs_hi_r <= 40)
hi_lim_R = obs_hi_r[ id_Rx ]
hi_lim_sb = obs_hi_fdens[ id_Rx ]
hi_lim_err = obs_hi_ferr[ id_Rx ]

I0 = 4e1
alpha = -1.85

po = np.array([I0, alpha])
popt, pcov = curve_fit( power_fit, low_lim_R, low_lim_sb, p0 = po, bounds = ( [4e1, -2.4], [6e1, -1.4] ), sigma = low_lim_err, method = 'trf')
low_Ie, low_alpha = popt
low_pow_line = power_fit( low_lim_R, low_Ie, low_alpha)

I0 = 5.2e1
alpha = -1.8
popt, pcov = curve_fit( power_fit, hi_lim_R, hi_lim_sb, p0 = po, bounds = ( [4e1, -2.3], [6e1, -1.3] ), sigma = hi_lim_err, method = 'trf')
hi_Ie, hi_alpha = popt
hi_pow_line = power_fit( hi_lim_R, hi_Ie, hi_alpha)

R_alpha = -1 * low_alpha

plt.figure()
ax = plt.subplot(111)
ax.set_title('$ SB \\times R^{\\alpha} [\\alpha = %.3f]$' % R_alpha,)

ax.plot(obs_low_r, obs_low_fdens * obs_low_r**(R_alpha), ls = '-', color = 'b', alpha = 0.5, 
	label = 'low BCG $M_{\\ast}$',)
ax.plot(obs_hi_r, obs_hi_fdens * obs_hi_r**(R_alpha), ls = '-', color = 'r', alpha = 0.5, 
	label = 'high BCG $M_{\\ast}$',)

ax.fill_between(obs_low_r, y1 = (obs_low_fdens - obs_low_ferr)* obs_low_r**(R_alpha), y2 = (obs_low_fdens + obs_low_ferr)* obs_low_r**(R_alpha),
	color = 'b', alpha = 0.15,)
ax.fill_between(obs_hi_r, y1 = (obs_hi_fdens - obs_hi_ferr)* obs_hi_r**(R_alpha), y2 = (obs_hi_fdens + obs_hi_ferr)* obs_hi_r**(R_alpha), 
	color = 'r', alpha = 0.15,)

'''
ax.errorbar(obs_low_r, obs_low_fdens, yerr = obs_low_ferr, xerr = None, color = 'b', marker = '.', ms = 3, mec = 'b', mfc = 'none', 
	ls = '', ecolor = 'b', elinewidth = 1, label = 'low BCG $M_{\\ast}$', alpha = 0.5)
ax.errorbar(obs_hi_r, obs_hi_fdens, yerr = obs_hi_ferr, xerr = None, color = 'r', marker = '.', ms = 3, mec = 'r', mfc = 'none', 
	ls = '', ecolor = 'r', elinewidth = 1, label = 'high BCG $M_{\\ast}$', alpha = 0.5)

ax.plot(low_lim_R, low_pow_line, ls = '-', color = 'b', alpha = 0.5, label = 'power-law fitting')
ax.plot(hi_lim_R, hi_pow_line, ls = '-', color = 'r', alpha = 0.5, )
ax.text( 100, 2e-1, s = '$\\alpha = %.3f$' % low_alpha, color = 'b', alpha = 0.5,)
ax.text( 100, 1e-1, s = '$\\alpha = %.3f$' % hi_alpha, color = 'r', alpha = 0.5,)
'''
ax.set_xlabel('$R_{c}[kpc / h]$')
ax.set_ylabel('$ SB \\times R^{\\alpha}$')
ax.set_xscale('log')
ax.set_yscale('log')

#ax.set_ylim(3e-3, 1e0)
ax.set_ylim(3e1, 8e1)
ax.set_xlim(1e1, 1e3)

ax.legend(loc = 3, )
ax.grid(which = 'both', axis = 'both', alpha = 0.20)
ax.tick_params(axis = 'both', which = 'both', direction = 'in')

plt.subplots_adjust(left = 0.15, right = 0.9, hspace = 0.02 )
#plt.savefig('power_fit.png', dpi = 300)
plt.savefig('R-weit_SB.png', dpi = 300)
plt.close()


raise

plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
ax = plt.subplot(gs[0])
bx = plt.subplot(gs[1])

ax.plot(obs_low_r, obs_low_fdens, ls = '-', color = 'b', alpha = 0.5, label = 'low BCG $M_{\\ast}$', )
ax.plot(obs_hi_r, obs_hi_fdens, ls = '-', color = 'r', alpha = 0.5, label = 'high BCG $M_{\\ast}$', )
'''
ax.plot(obs_low_r, obs_low_fdens - low_fit_line, ls = '-', color = 'b', alpha = 0.5, label = 'low BCG $M_{\\ast}$',)
ax.plot(obs_hi_r, obs_hi_fdens - hi_fit_line, ls = '-', color = 'r', alpha = 0.5, label = 'high BCG $M_{\\ast}$',)
'''

ax.set_xlabel('$R_{c}[kpc / h]$')
ax.set_ylabel('$SB [nanomaggies / arcsec^2]$')
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_ylim(3e-3, 3e-1)
#ax.set_ylim(2e-3, 2e-2)

ax.set_xlim(2e1, 2e2)
ax.legend(loc = 1, fontsize = 8,)
ax.grid(which = 'both', axis = 'both', alpha = 0.20)
ax.tick_params(axis = 'both', which = 'both', direction = 'in')

bx.plot( NBG_low_sign_r, NBG_low_dsign_dlgr, ls = '-', color = 'b', alpha = 0.5, )
bx.plot( NBG_hi_sign_r, NBG_hi_dsign_dlgr, ls = '-', color = 'r', alpha = 0.5, )
bx.fill_between( NBG_low_sign_r, y1 = NBG_low_dsign_dlgr - NBG_low_k_err, y2 = NBG_low_dsign_dlgr + NBG_low_k_err, color = 'b', alpha = 0.2,)
bx.fill_between( NBG_hi_sign_r, y1 = NBG_hi_dsign_dlgr - NBG_hi_k_err, y2 = NBG_hi_dsign_dlgr + NBG_hi_k_err, color = 'r', alpha = 0.2,)
'''
bx.plot(low_icl_res_R, low_icl_res_slope, ls = '-', color = 'b', alpha = 0.5,)
bx.plot(hi_icl_res_R, hi_icl_res_slope, ls = '-', color = 'r', alpha = 0.5,)
'''

bx.set_xlabel('$R_{c}[kpc / h]$')
bx.set_ylabel('d(lgSB) / d(lgR)')
bx.set_xscale('log')
bx.set_xlim( ax.get_xlim() )

bx.set_ylim(-2.5, -1)
#bx.set_ylim(-2.5, 2.5)

bx.grid(which = 'both', axis = 'both', alpha = 0.20)
bx.tick_params(axis = 'both', which = 'both', direction = 'in')
ax.set_xticklabels( labels = [], minor = True,)

plt.subplots_adjust(left = 0.15, right = 0.9, hspace = 0.02 )
plt.savefig('grident_compare.png', dpi = 300)
plt.close()

raise

## compare with SDSS (core region)
plt.figure()
ax = plt.subplot(111)

ax.plot(obs_low_r, obs_low_fdens, ls = '-', color = 'b', alpha = 0.5, label = 'low BCG $M_{\\ast}$')
ax.plot(obs_hi_r, obs_hi_fdens, ls = '-', color = 'r', alpha = 0.5, label = 'high BCG $M_{\\ast}$')

ax.plot(obs_hi_r, hi_fit_line, ls = '-.', color = 'r', alpha = 0.5,)
ax.plot(obs_low_r, low_fit_line, ls = '-.', color = 'b', alpha = 0.5, label = 'sersic fitting [n=4]')

ax.errorbar(low_bcg_r, low_bcg_mu, yerr = low_bcg_mu_err, xerr = None, color = 'b', marker = 's', ms = 3, mec = 'b', mfc = 'b',
	ls = '', ecolor = 'b', elinewidth = 1, label = '$ \\bar{\\mu}_{SDSS} $', alpha = 0.35,)
ax.errorbar(hi_bcg_r, hi_bcg_mu, yerr = hi_bcg_mu_err, xerr = None, color = 'r', marker = 's', ms = 3, mec = 'r', mfc = 'r',
	ls = '', ecolor = 'r', elinewidth = 1, alpha = 0.35, )

ax.set_xlabel('$R_{c}[kpc / h]$')
ax.set_ylabel('$SB [nanomaggies / arcsec^2]$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-5, 1e1)
ax.set_xlim(1e0, 2e3)
ax.legend(loc = 1, fontsize = 9,)
ax.grid(which = 'both', axis = 'both', alpha = 0.20)
ax.tick_params(axis = 'both', which = 'both', direction = 'in')

plt.savefig('BCG_pros_compare.png', dpi = 300)
plt.close()

raise

"""
## slope of SB profiles in logarithmic coordinate
plt.figure()
ax = plt.subplot(111)
'''
ax.plot(low_sign_r, low_dsign_dlgr, ls = '-', color = 'b', alpha = 0.5, label = 'low BCG $M_{\\ast}$',)
ax.plot(hi_sign_r, hi_dsign_dlgr, ls = '-', color = 'r', alpha = 0.5, label = 'high BCG $M_{\\ast}$',)

ax.fill_between(low_sign_r, y1 = low_dsign_dlgr - low_k_err, y2 = low_dsign_dlgr + low_k_err, color = 'b', alpha = 0.2,)
ax.fill_between(hi_sign_r, y1 = hi_dsign_dlgr - hi_k_err, y2 = hi_dsign_dlgr + hi_k_err, color = 'r', alpha = 0.2,)
'''
ax.plot( NBG_low_sign_r, NBG_low_dsign_dlgr, ls = '--', color = 'b', alpha = 0.5, label = 'Background subtracted',)
ax.plot( NBG_hi_sign_r, NBG_hi_dsign_dlgr, ls = '--', color = 'r', alpha = 0.5, )

ax.fill_between( NBG_low_sign_r, y1 = NBG_low_dsign_dlgr - NBG_low_k_err, y2 = NBG_low_dsign_dlgr + NBG_low_k_err, color = 'b', alpha = 0.2,)
ax.fill_between( NBG_hi_sign_r, y1 = NBG_hi_dsign_dlgr - NBG_hi_k_err, y2 = NBG_hi_dsign_dlgr + NBG_hi_k_err, color = 'r', alpha = 0.2,)

ax.set_xlabel('$R_{c}[kpc / h]$')
ax.set_ylabel('d(lgSB) / d(lgR)')
ax.set_xscale('log')
ax.set_xlim(1e1, 2e3)
ax.set_ylim(-2.5, 0)
ax.legend(loc = 2,)
ax.grid(which = 'both', axis = 'both', alpha = 0.20)
ax.tick_params(axis = 'both', which = 'both', direction = 'in')

plt.savefig('grident_compare.png', dpi = 300)
plt.close()
"""

## compare with 'NFW' mock light profile
plt.figure()
ax = plt.subplot(111)
'''
ax.plot(low_R, mock_low_dens, ls = '--', color = 'b', alpha = 0.5, )
ax.plot(high_R, mock_hi_dens, ls = '--', color = 'r', alpha = 0.5, )
'''
#ax.plot(R_obs, flux_obs, ls = '-', color = 'k', alpha = 0.5, label = 'Z05',)

ax.plot(low_R, sub_low_dens, ls = '-', color = 'b', alpha = 0.5, label = 'NFW')
ax.plot(high_R, sub_hi_dens, ls = '-', color = 'r', alpha = 0.5, )

ax.axvline(low_R200, ls = '--', color = 'b', alpha = 0.5, label = '$R_{200m}$')
ax.axvline(low_R200 / C0, ls = ':', color = 'b', alpha = 0.5, label = '$r_{c}$')
ax.axvline(high_R200, ls = '--', color = 'r', alpha = 0.5,)
ax.axvline(high_R200 / C1, ls = ':', color = 'r', alpha = 0.5,)

ax.errorbar(obs_low_r, obs_low_fdens, yerr = obs_low_ferr, xerr = None, color = 'b', marker = '.', ms = 3, mec = 'b', mfc = 'none', 
	ls = '', ecolor = 'b', elinewidth = 1, label = 'low BCG $M_{\\ast}$', alpha = 0.5)
ax.errorbar(obs_hi_r, obs_hi_fdens, yerr = obs_hi_ferr, xerr = None, color = 'r', marker = '.', ms = 3, mec = 'r', mfc = 'none', 
	ls = '', ecolor = 'r', elinewidth = 1, label = 'high BCG $M_{\\ast}$', alpha = 0.5)

ax.errorbar(lim_low_r, low_resi_fdens, yerr = lim_low_sb_err, xerr = None, color = 'b', marker = '^', ms = 3, mec = 'b', mfc = 'b', 
	ls = '', ecolor = 'b', elinewidth = 1, alpha = 0.5, label = 'NFW subtracted')
ax.errorbar(lim_hi_r, hi_resi_fdens, yerr = lim_hi_sb_err, xerr = None, color = 'r', marker = '^', ms = 3, mec = 'r', mfc = 'r', 
	ls = '', ecolor = 'r', elinewidth = 1, alpha = 0.5, )
'''
ax.errorbar(obs_low_r, obs_low_fdens - low_fit_line, yerr = obs_low_ferr, xerr = None, color = 'b', marker = '.', ms = 2, mec = 'b', mfc = 'b', 
	ls = '', ecolor = 'b', elinewidth = 1, alpha = 0.5,)
ax.errorbar(obs_hi_r, obs_hi_fdens - hi_fit_line, yerr = obs_hi_ferr, xerr = None, color = 'r', marker = '.', ms = 2, mec = 'r', mfc = 'r', 
	ls = '', ecolor = 'r', elinewidth = 1, alpha = 0.5,)
'''
ax.plot(obs_hi_r, hi_fit_line, ls = '-.', color = 'r', alpha = 0.5,)
ax.plot(obs_low_r, low_fit_line, ls = '-.', color = 'b', alpha = 0.5, label = 'sersic fitting [n=4]')

ax.plot(obs_low_r[idvx], obs_low_fdens[idvx], 'b*', alpha = 0.35,)
ax.plot(obs_hi_r[idux], obs_hi_fdens[idux], 'r*', alpha = 0.35, )

ax.bar(x = 5, height = 20, width = 10, align = 'center', color = 'c', alpha = 0.1,)

ax.set_xlabel('$R_{c}[kpc / h]$')
ax.set_ylabel('$SB [nanomaggies / arcsec^2]$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1e-5, 1e0)
ax.set_xlim(1e1, 2e3)
ax.legend(loc = 1, fontsize = 9,)
ax.grid(which = 'both', axis = 'both', alpha = 0.20)
ax.tick_params(axis = 'both', which = 'both', direction = 'in')

plt.savefig('model_obs_compare.png', dpi = 300)
plt.close()

