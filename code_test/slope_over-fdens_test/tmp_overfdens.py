import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from scipy import signal
from scipy import interpolate as interp

from fig_out_module import cc_grid_img, grid_img
from fig_out_module import arr_jack_func
from light_measure import cumula_flux_func

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
# constant
rad2asec = U.rad.to(U.arcsec)
pixel = 0.396

#*********************#
'''
def over_fdens_func( r, mu_func, obs_r, obs_fdens,):

	Ns = len(r)

	cumuli_f = np.zeros( Ns, dtype = r[0].dtype )

	delt_fdens = np.zeros( Ns, dtype = r[0].dtype)

	for n in range( Ns ):

		cumuli_f[n] = interp.splint(r[0], r[n], mu_func) * (np.pi * 2)

	mean_fdens = cumuli_f / (np.pi * r**2)

	tmp_func = interp.interp1d( r[1:], mean_fdens[1:], kind = 'cubic',)

	cc_mfdens = tmp_func( obs_r[1:-1] )

	delt_fdens = cc_mfdens - obs_fdens[1:-1]

	return cc_mfdens, delt_fdens
'''
def over_fdens_func( r, mu_func, obs_r, obs_fdens,):
	new_sb = interp.splev( r, mu_func, der = 0)
	cumuli_f = cumula_flux_func(r, new_sb,)

	mean_fdens = cumuli_f / ( np.pi * r**2)
	tmp_func = interp.interp1d( r, mean_fdens)

	cc_mfdens = tmp_func( obs_r[1:-1] )
	delt_fdens = cc_mfdens - obs_fdens[1:-1]

	return cc_mfdens, delt_fdens

pre_lis = ['low-BCG-star-Mass', 'high-BCG-star-Mass']
name_lis = ['low $M_{\\ast}$', 'high $M_{\\ast}$']
line_c = ['b', 'r']

z_ref = 0.25
Da_ref = Test_model.angular_diameter_distance(z_ref).value
N_bin = 30

plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2])
ax = plt.subplot(gs[0])
bx = plt.subplot(gs[1])

tmp_R, tmp_fdens = [], []
tmp_dR, tmp_dfdens = [], []
tmp_sbr, tmp_sb = [], []

for kk in range( 2 ):

	with h5py.File('/home/xkchen/jupyter/stack/' + pre_lis[kk] + '_Mean_jack_SB-pro_z-ref_with-selection.h5', 'r') as f:
		c_r_arr = np.array(f['r'])
		c_sb_arr = np.array(f['sb'])
		c_sb_err = np.array(f['sb_err'])

	id_Nul = c_sb_arr > 0
	c_r_arr = c_r_arr[id_Nul]
	c_sb_arr = c_sb_arr[id_Nul]
	c_sb_err = c_sb_err[id_Nul]

	angl_r = c_r_arr * 1e-3 * rad2asec / Da_ref

	interp_fdens = interp.splrep(angl_r, c_sb_arr, s = 0)
	interp_Rfdens = interp.splrep(angl_r, c_sb_arr * angl_r, s = 0)

	lim_r0 = np.log10( angl_r.min() )
	lim_r1 = np.log10( angl_r.max() )

	N_rebin = 500
	new_r = np.logspace(lim_r0, lim_r1, N_rebin)

	mfdens, delt_fdens = over_fdens_func( new_r, interp_fdens, angl_r, c_sb_arr,)

	tmp_R.append( c_r_arr[1:-1] )
	tmp_fdens.append( mfdens )

	tmp_dR.append( c_r_arr[1:-1] )
	tmp_dfdens.append( delt_fdens )

	tmp_sbr.append( c_r_arr )
	tmp_sb.append( c_sb_arr )

	ax.plot(c_r_arr[1:-1], mfdens, ls = '--', color = line_c[kk], alpha = 0.5, label = '$\\bar{\\mu}(<r)$')
	ax.plot(c_r_arr[1:-1], delt_fdens, ls = '-', color = line_c[kk], alpha = 0.5, label = '$\\Delta{\\mu}$=$\\bar{\\mu}(<r) - \\mu(r)$')
	ax.plot(c_r_arr, c_sb_arr, ls = ':', color = line_c[kk], alpha = 0.5, label = name_lis[kk] + '[$\\mu(r)$]')

tt_func = interp.interp1d( tmp_R[0], tmp_fdens[0], kind = 'cubic',)
idmx = ( tmp_R[1] >= tmp_R[0].min() ) & ( tmp_R[1] <= tmp_R[0].max() )
com_fdens = tt_func( tmp_R[1][idmx] )

bx.plot( tmp_R[1][idmx], tmp_fdens[1][idmx] - com_fdens, ls = '-', color = 'm', alpha = 0.5, 
	label = '$\\bar{\\mu}(<r)_{high \, M_{\\ast}}$ - $\\bar{\\mu}(<r)_{low \, M_{\\ast}}$')

tp_func = interp.interp1d( tmp_sbr[0], tmp_sb[0], kind = 'cubic',)
idnx = ( tmp_sbr[1] >= tmp_sbr[0].min() ) & ( tmp_sbr[1] <=  tmp_sbr[0].max() )
com_sb = tp_func( tmp_sbr[1][idnx] )

bx.plot( tmp_sbr[1][idnx], np.abs(tmp_sb[1][idnx] - com_sb), ls = '-', color = 'c', alpha = 0.5, 
	label = '$\\mu(r)_{high \, M_{\\ast}}$ - $\\mu(r)_{low \, M_{\\ast}}$')

td_func = interp.interp1d( tmp_dR[0], tmp_dfdens[0], kind = 'cubic',)
idvx = ( tmp_dR[1] >= tmp_dR[0].min() ) & ( tmp_dR[1] <=  tmp_dR[0].max() )
com_dfdens = td_func( tmp_dR[1][idvx] )

bx.plot( tmp_dR[1][idvx], np.abs(tmp_dfdens[1][idvx] - com_dfdens), ls = '-', color = 'k', alpha = 0.5, 
	label = '$\\Delta{\\mu}_{high \, M_{\\ast}}$ - $\\Delta{\\mu}_{low \, M_{\\ast}}$')

ax.set_ylim(1e-4, 1e1)
ax.set_xlim(1e0, 4e3)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xlabel('R [kpc]')
ax.legend(loc = 3, fontsize = 9,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in',)

bx.set_xlim( ax.get_xlim() )
bx.set_ylim( 1e-6, 1e0)
bx.set_xscale('log')
bx.set_yscale('log')
bx.set_xlabel('R [kpc]')
bx.legend(loc = 3, frameon = False,)
bx.grid(which = 'both', axis = 'both', alpha = 0.25)
bx.tick_params(axis = 'both', which = 'both', direction = 'in',)
ax.set_xticklabels( labels = [], fontsize = 0.005)

plt.subplots_adjust( hspace = 0.05,)
plt.savefig('over_fdens_test.png', dpi = 300)
plt.close()


