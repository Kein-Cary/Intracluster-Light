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

from astropy.table import Table
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
import scipy.interpolate as interp
import scipy.stats as sts
from scipy.interpolate import splev, splrep

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

band = [ 'r', 'g', 'i' ]
l_wave = np.array( [6166, 4686, 7480] )


Jy = 10**(-23) # erg * s^(-1) * cm^(-2) * Hz^(-1)
F0 = 3.631 * 10**(-6) * Jy
v_c = C.c.value # in unit m/s

mJy = 10**(-3) * Jy

### === ### obs data load
path = '/home/xkchen/mywork/ICL/code/rig_common_cat/mass_bin_BG/'

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$'] ## or line name

color_s = ['r', 'g', 'b']

"""
for rank in ( 0,1 ):

	tot_r, tot_sb, tot_err = [], [], []

	## observed flux
	for kk in range( 3 ):
		with h5py.File( path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[rank], band[kk]), 'r') as f:
			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		tot_r.append( tt_r )
		tot_sb.append( tt_sb )
		tot_err.append( tt_err )

	tot_r = np.array( tot_r )
	tot_sb = np.array( tot_sb )
	tot_err = np.array( tot_err )

	## get the mag, fv
	tot_Fv = []
	tot_Fv_err = []
	obs_R = []

	for kk in range( 3 ):

		idux = ( tot_r[kk] >= 10 ) & ( tot_r[kk] <= 1e3 )

		obs_R.append( tot_r[kk][idux] )

		tot_Fv.append( tot_sb[kk][idux] * F0 / mJy )
		tot_Fv_err.append( tot_err[kk][idux] * F0 / mJy )

	tot_Fv = np.array( tot_Fv )
	tot_Fv_err = np.array( tot_Fv_err )
	obs_R = np.array( obs_R )

	ID_lis = np.arange( 0, len(obs_R[0]) )
	z_lis = np.ones( len(obs_R[0]) ) * 0.25

	arr_to_save = np.array( [ID_lis, z_lis, tot_Fv[0,:], tot_Fv_err[0,:], 
							tot_Fv[1,:], tot_Fv_err[1,:], tot_Fv[2,:], tot_Fv_err[2,:] ] ).T

	keys = ['id', 'redshift', 'sdss.rp', 'sdss.rp_err', 'sdss.gp', 'sdss.gp_err', 'sdss.ip', 'sdss.ip_err']
	values = []
	for pp in range( len(arr_to_save[0]) ):
		values.append( arr_to_save[:,pp] )

	fill = dict( zip(keys, values) )
	data_Table = Table( fill )
	data_Table.write( '/home/xkchen/jupyter/%s_gri_mag.fits' % cat_lis[ rank ], overwrite = True)
"""

### === ### pcigale
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

import subprocess as subpro

obs_file = 'high_BCG_star-Mass_gri_mag.fits'
sed_module_str = 'sfh2exp, bc03, redshifting'
core_n = 1

"""
### creat the config_file
cmd = 'pcigale init'
ppa = subpro.Popen(cmd, shell = True)
ppa.wait()

config_file = 'pcigale.ini'

### fill information into config_file

f = open( config_file, 'r')
lines = f.readlines()
f.close()

f = open( config_file, 'w+')
N_lines = len(lines)

for qq in range( N_lines ):

	if 'data_file = ' in lines[qq]:
		dts = lines[qq].replace( 'data_file = ', 'data_file = %s' % obs_file )
		f.writelines( dts )

	elif 'sed_modules = ' in lines[qq]:
		dts = lines[qq].replace( 'sed_modules = ', 'sed_modules = %s' % sed_module_str )
		f.writelines( dts )

	elif 'analysis_method = ' in lines[qq]:
		dts = lines[qq].replace( 'analysis_method = ', 'analysis_method = pdf_analysis')
		f.writelines( dts )

	elif 'cores = ' in lines[qq]:
		dts = lines[qq].replace( 'cores = ', 'cores = %d' % core_n )
		f.writelines( dts )

	else:
		dts = lines[qq]
		f.writelines( dts )

f.writelines( 'create_table = True\n' )
f.close()

cmd = 'pcigale genconf'
ppa = subpro.Popen( cmd, shell = True )
ppa.wait()
"""

### fill information in to config_file
config_file = 'pcigale.ini'

tau_main = 50         # 10, 20, 30, 40, 50
imf = 1               # use Chabrier imf
meta = [0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05]  # metallicity

# age in unit of Myr, mass in unit of M_sun
phy_proper = [ 'sfh.age', 'sfh.integrated', 'sfh.sfr', 'sfh.sfr10Myrs', 'sfh.sfr100Myrs', 
				'stellar.imf', 'stellar.metallicity', 'stellar.age_m_star', 'stellar.lum', 
				'stellar.m_star_young', 'stellar.m_gas_young', 'stellar.m_star_old', 
				'stellar.m_gas_old', 'stellar.m_star', 'stellar.m_gas',]

z_ini = z_at_lb_time( 0.25, 8, dt = 0.1)
age_main = Test_model.lookback_time( z_ini ).value * 10**3
"""
f = open( config_file, 'r')
lines = f.readlines()
f.close()

f = open( config_file, 'w+', encoding = 'UTF-8')
N_lines = len(lines)

for qq in range( N_lines ):

	if 'tau_main = ' in lines[qq]:
		dts = 'tau_main = %.1f' % tau_main

		index_0 = lines[qq].index( 'tau' )
		index_1 = lines[qq].index( '\n' )

		pts = lines[qq].replace( lines[qq][index_0: index_1], dts )
		f.writelines( pts )

		continue

	if 'f_burst = ' in lines[qq]:
		dts = 'f_burst = 0'

		index_0 = lines[qq].index( 'f_burst' )
		index_1 = lines[qq].index( '\n' )

		pts = lines[qq].replace( lines[qq][index_0: index_1], dts )
		f.writelines( pts )

		continue

	if 'age = ' in lines[qq]:
		dts = 'age = '

		arr_lis = ['%.2f' % mm for mm in age_main ]

		add_str = ''
		N_age = len( age_main )
		for jj in range( N_age ):
			add_atr = dts + arr_lis[ jj ]
			dts = add_atr + ','
			add_atr = ''

		index_0 = lines[qq].index( 'age' )
		index_1 = lines[qq].index( '\n' )

		pts = lines[qq].replace( lines[qq][index_0: index_1], dts )
		f.writelines( pts )

		continue

	if 'imf = ' in lines[qq]:

		dts = 'imf = 1'

		index_0 = lines[qq].index( 'imf' )
		index_1 = lines[qq].index( '\n' )

		pts = lines[qq].replace( lines[qq][index_0: index_1], dts )
		f.writelines( pts )

		continue

	if 'metallicity = ' in lines[qq]:

		dts = 'metallicity = '
		arr_lis = ['%.4f' % mm for mm in meta]
		add_str = ''

		N_meta = len( meta )

		for jj in range( N_meta ):
			add_atr = dts + arr_lis[ jj ]
			dts = add_atr + ','
			add_atr = ''

		index_0 = lines[qq].index( 'metallicity' )
		index_1 = lines[qq].index( '\n' )

		pts = lines[qq].replace( lines[qq][index_0: index_1], dts )
		f.writelines( pts )

		continue

	if 'save_best_sed = ' in lines[qq]:
		dts = 'save_best_sed = True'

		index_0 = lines[qq].index( 'save_best_sed' )
		index_1 = lines[qq].index( '\n' )

		pts = lines[qq].replace( lines[qq][index_0: index_1], dts )
		f.writelines( pts )

		continue

	if 'variables = ' in lines[qq]:
		dts = 'variables = '
		N_info = len( phy_proper )

		for jj in range( N_info ):
			add_atr = dts + phy_proper[ jj ]
			p_dts = add_atr + ','
			dts = p_dts + ''

			add_atr = ''
			p_dts = ''

		index_0 = lines[qq].index( 'variables' )
		index_1 = lines[qq].index( '\n' )

		pts = lines[qq].replace( lines[qq][index_0: index_1], dts )
		f.writelines( pts )

		continue

	pts = lines[qq].replace( lines[qq], lines[qq])
	f.writelines( pts )

f.close()



f = open( config_file, 'r')
lines = f.readlines()
f.close()

f = open( config_file, 'w+', encoding = 'UTF-8')
N_lines = len(lines)

for qq in range( N_lines ):

	if 'separation_age = ' in lines[qq]:

		dts = 'separation_age = 10,'

		index_0 = lines[qq].index( 'separation_age = ' )
		index_1 = lines[qq].index( '\n' )

		pts = lines[qq].replace( lines[qq][index_0: index_1], dts )
		f.writelines( pts )

	elif 'burst_age = ' in lines[qq]:
		dts = 'burst_age = 20'

		index_0 = lines[qq].index( 'burst_age' )
		index_1 = lines[qq].index( '\n' )

		pts = lines[qq].replace( lines[qq][index_0: index_1], dts )
		f.writelines( pts )

	else:
		pts = lines[qq].replace( lines[qq], lines[qq])
		f.writelines( pts )

f.close()
"""

### check and run SED fitting
# cmd = 'pcigale check'
# ppa = subpro.Popen( cmd, shell = True )
# ppa.wait()

# cmd = 'pcigale run'
# ppa = subpro.Popen( cmd, shell = True )
# ppa.wait()


### === ### figure the result
def get_eff_fv( filter_nu, filter_curve, obs_nu, obs_Fv, ):

	delta_v = obs_nu[1:] - obs_nu[:-1]
	delta_v = np.r_[ delta_v, delta_v[-1] ]

	intep_F = interp.interp1d( filter_nu, filter_curve, kind = 'cubic',)
	id_lim = ( obs_nu >= filter_nu.min() ) & ( obs_nu <= filter_nu.max() )

	sum_f0 = np.sum( delta_v[ id_lim] * obs_Fv[ id_lim] * intep_F( obs_nu[ id_lim] ) / obs_nu[ id_lim] )
	sum_f1 = np.sum( delta_v[ id_lim] * intep_F( obs_nu[ id_lim] ) / obs_nu[ id_lim] )
	eff_Fv = sum_f0 / sum_f1

	return eff_Fv

# SDSS filter
sdss_respone = fits.open( '/home/xkchen/mywork/ICL/code/ezgal_files/filter_curves.fits')
g_wave_x = sdss_respone[2].data['wavelength']
g_pass_curve = sdss_respone[2].data['respt']

r_wave_x = sdss_respone[3].data['wavelength']
r_pass_curve = sdss_respone[3].data['respt']

i_wave_x = sdss_respone[4].data['wavelength']
i_pass_curve = sdss_respone[4].data['respt']

# best sed
R0_mode = fits.open( '/home/xkchen/jupyter/out/0.0_best_model.fits' )
mode_wl = R0_mode[1].data['wavelength']
mode_Fv = R0_mode[1].data['Fnu']
mode_nu = v_c * 10**10 / mode_wl
mode_Fl = (v_c * 10**10 / mode_wl**2) * mode_Fv


intep_F = interp.interp1d( g_wave_x, g_pass_curve, kind = 'cubic',)

idvx = ( mode_wl >= g_wave_x.min() ) & ( mode_wl <= g_wave_x.max() )
delta_wl = mode_wl[1:] - mode_wl[:-1]
delta_wl = np.r_[ delta_wl, delta_wl[-1] ]

aveg_fl = np.sum( delta_wl[idvx] * intep_F( mode_wl[idvx] ) * mode_Fl[idvx] ) / np.sum( delta_wl[idvx] * intep_F( mode_wl[idvx] ) )
wl_pivot = np.sqrt( np.sum( delta_wl[idvx] * intep_F( mode_wl[idvx] ) ) / np.sum( delta_wl[idvx] * intep_F( mode_wl[idvx] ) / mode_wl[idvx]**2 ) )
aveg_fv = wl_pivot**2 * aveg_fl / (v_c * 10**10) 

g_wave_nu = v_c * 10**10 / g_wave_x
r_wave_nu = v_c * 10**10 / r_wave_x
i_wave_nu = v_c * 10**10 / i_wave_x

g_eff_Fv = get_eff_fv( g_wave_nu, g_pass_curve, mode_nu, mode_Fv,)


rank = 1

obs_data = fits.open('/home/xkchen/jupyter/%s_gri_mag.fits' % cat_lis[ rank ])
obs_Rid = obs_data[1].data['id']
obs_gF = obs_data[1].data['sdss.gp']
obs_rF = obs_data[1].data['sdss.rp']
obs_iF = obs_data[1].data['sdss.ip']
obs_gF_err = obs_data[1].data['sdss.gp_err']
obs_rF_err = obs_data[1].data['sdss.rp_err']
obs_iF_err = obs_data[1].data['sdss.ip_err']

mode_file = '/home/xkchen/jupyter/out/results.fits'
mode_data = fits.open( mode_file )

mode_gF = mode_data[1].data['bayes.sdss.gp']
mode_rF = mode_data[1].data['bayes.sdss.rp']
mode_iF = mode_data[1].data['bayes.sdss.ip']
mode_gF_err = mode_data[1].data['bayes.sdss.gp_err']
mode_rF_err = mode_data[1].data['bayes.sdss.rp_err']
mode_iF_err = mode_data[1].data['bayes.sdss.ip_err']

mode_g_mag = 22.5 - 2.5 * np.log10( mode_gF * 10**(-3) * Jy / F0 )
mode_r_mag = 22.5 - 2.5 * np.log10( mode_gF * 10**(-3) * Jy / F0 )
mode_i_mag = 22.5 - 2.5 * np.log10( mode_gF * 10**(-3) * Jy / F0 )

mode_g_mag_err = 2.5 * (mode_gF_err * 10**(-3) * Jy / F0) / ( np.log(10) * mode_gF * 10**(-3) * Jy / F0 )
mode_r_mag_err = 2.5 * (mode_rF_err * 10**(-3) * Jy / F0) / ( np.log(10) * mode_rF * 10**(-3) * Jy / F0 )
mode_i_mag_err = 2.5 * (mode_iF_err * 10**(-3) * Jy / F0) / ( np.log(10) * mode_iF * 10**(-3) * Jy / F0 )

mode_Z = mode_data[1].data['bayes.stellar.metallicity']
mode_Z_err = mode_data[1].data['bayes.stellar.metallicity_err']

mode_age = mode_data[1].data['bayes.stellar.age_m_star'] / 1e3 # Gyr
mode_age_err = mode_data[1].data['bayes.stellar.age_m_star_err'] / 1e3

mode_Mstar = mode_data[1].data['bayes.stellar.m_star']
mode_Mstar_err = mode_data[1].data['bayes.stellar.m_star_err']

Da_ref = Test_model.angular_diameter_distance( z_ref ).value 
L_pix = Da_ref * 10**3 * pixel / rad2asec

mode_sigma_Mstar = mode_Mstar / (L_pix**2)
mode_Mstar_err = mode_Mstar_err / (L_pix**2)

## observed flux
tot_r, tot_sb, tot_err = [], [], []

for kk in range( 3 ):
	with h5py.File( path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[rank], band[kk]), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	idux = ( tt_r >= 10 ) & ( tt_r <= 1e3 )

	tot_r.append( tt_r[idux] )
	tot_sb.append( tt_sb[idux] )
	tot_err.append( tt_err[idux] )

tot_r = np.array( tot_r )
tot_sb = np.array( tot_sb )
tot_err = np.array( tot_err )

obs_mag = 22.5 - 2.5 * np.log10( tot_sb )
obs_mag_err = 2.5 * tot_err / ( np.log(10) * tot_sb )


## predict mag
for pp in range( 3 ):

	plt.figure()
	gs = gridspec.GridSpec(2, 1, height_ratios=[4,1])
	ax = plt.subplot(gs[0])
	bx = plt.subplot(gs[1])

	ax.set_title('%s band, SED fitting, mag compare' % band[pp] )
	ax.errorbar( obs_mag[pp], obs_mag[pp], yerr = obs_mag_err[pp], color = color_s[pp], marker = '.', ecolor = color_s[pp], ls = 'none', 
		alpha = 0.5, label = '%s band, obs' % band[pp],)
	if pp == 0:
		ax.errorbar( obs_mag[pp], mode_g_mag, yerr = mode_g_mag_err, color = 'k', ls = '-', marker = '.', ecolor = 'k', alpha = 0.5, label = 'SED',)
		bx.plot(obs_mag[pp], mode_g_mag - obs_mag[pp], color = color_s[pp], alpha = 0.5, ls = '-',)
		bx.errorbar( obs_mag[pp], mode_g_mag - mode_g_mag, yerr = mode_g_mag_err, color = 'k', ls = '-', marker = '.', ecolor = 'k', alpha = 0.5,)

	if pp == 1:
		ax.errorbar( obs_mag[pp], mode_r_mag, yerr = mode_r_mag_err, color = 'k', ls = '-', marker = '.', ecolor = 'k', alpha = 0.5, label = 'SED',)
		bx.plot(obs_mag[pp], mode_r_mag - obs_mag[pp], color = color_s[pp], alpha = 0.5, ls = '-',)
		bx.errorbar( obs_mag[pp], mode_r_mag - mode_r_mag, yerr = mode_r_mag_err, color = 'k', ls = '-', marker = '.', ecolor = 'k', alpha = 0.5,)

	if pp == 2:
		ax.errorbar( obs_mag[pp], mode_i_mag, yerr = mode_i_mag_err, color = 'k', ls = '-', marker = '.', ecolor = 'k', alpha = 0.5, label = 'SED',)
		bx.plot(obs_mag[pp], mode_i_mag - obs_mag[pp], color = color_s[pp], alpha = 0.5, ls = '-',)
		bx.errorbar( obs_mag[pp], mode_i_mag - mode_i_mag, yerr = mode_i_mag_err, color = 'k', ls = '-', marker = '.', ecolor = 'k', alpha = 0.5,)

	ax.invert_yaxis()
	ax.set_ylabel('SB [mag / $ arcsec^2 $ ]')
	ax.set_xlabel('SB [mag / $ arcsec^2 $ ]')
	ax.legend( loc = 1 )

	bx.set_xlim( ax.get_xlim() )
	bx.set_ylim( -0.5, 2.5 )
	bx.set_xlabel( 'SB [mag / $ arcsec^2 $ ]' )
	bx.set_ylabel( '$ SB - SB_{obs} $')
	ax.set_xticklabels( labels = [], )

	plt.subplots_adjust( hspace = 0.0 )
	plt.savefig('/home/xkchen/figs/%s-band_mag_compare.png' % band[pp], dpi = 300)
	plt.close()

## predict Mass
plt.figure()
ax = plt.subplot(111)
ax.set_title('SED fitting, $ \\Sigma M_{\\ast} [M_{\\odot} / kpc^2]$')
ax.errorbar( tot_r[0], mode_Mstar, yerr = mode_Mstar_err, color = 'r', marker = '.', ecolor = 'r', ls = 'none', alpha = 0.5,)

ax.set_yscale( 'log' )
ax.set_ylabel('$ \\Sigma M_{\\ast} [M_{\\odot} / kpc^2]$')
ax.set_xlabel('R [kpc]')

ax.set_xscale( 'log' )
ax.set_xlim(1e1, 2e3)

plt.savefig('/home/xkchen/figs/mode_surface_Mstar.png', dpi = 300)
plt.close()

## predict age
plt.figure()
ax = plt.subplot(111)
ax.set_title('SED fitting, $ M_{\\ast} $ weighted age')
ax.errorbar( tot_r[0], mode_age, yerr = mode_age_err, color = 'r', marker = '.', ecolor = 'r', ls = 'none', alpha = 0.5,)

#ax.set_yscale( 'log' )
ax.set_ylabel('Gyr')
ax.set_xlabel('R [kpc]')

ax.set_xscale( 'log' )
ax.set_xlim(1e1, 2e3)
ax.set_ylim(0, 8)

plt.savefig('/home/xkchen/figs/mode_Mstar-weit_age.png', dpi = 300)
plt.close()

