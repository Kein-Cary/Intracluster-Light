import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C

from astropy.table import Table, QTable
from astropy import cosmology as apcy
from scipy import interpolate as interp
from scipy import integrate as integ
from astropy.coordinates import SkyCoord
from pynverse import inversefunc
from scipy import optimize
import scipy.signal as signal

##.
from img_sat_Rt_estimate import Mh_c_func
from Mass_rich_radius import rich2R_Simet

##.
from Gauss_Legendre_factor import GaussLegendreQuadrature
from Gauss_Legendre_factor import GaussLegendreQuad_arr

from colossus.cosmology import cosmology as co_cosmos
from colossus.halo import profile_nfw


### === ### cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100

Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

#. setting of cosmology (for density profile calculation)
params = {'flat': True, 'H0': 67.74, 'Om0': 0.311, 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95}
co_cosmos.addCosmology('myCosmo', params = params )
my_cosmo = co_cosmos.setCosmology( 'myCosmo' )

#. constant
pixel = 0.396

band = ['r', 'g', 'i']
L_wave = np.array([ 6166, 4686, 7480 ])
Mag_sun = [ 4.65, 5.11, 4.53 ]


##. funcs
def R_func( R, a, b, c):
	return a + (R + b)**c



### === ### data load
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/cat/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/theory_Rt/'

##. halo mass of satellites~( Li et al. 2016)
ref_sub_Mh = [ 11.37, 11.92, 12.64 ]
ref_Mh_err_0 = [ 0.35, 0.19, 0.12 ]
ref_Mh_err_1 = [ 0.35, 0.18, 0.11 ]

ref_sat_Ms = [ 10.68, 10.72, 10.78 ]
ref_R_edg = [ 0.1, 0.3, 0.6, 0.9 ]


Li_dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/theory_Rt/Li_data/Li_Mh2Mstar_data_point.csv')
Li_xerr = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/theory_Rt/Li_data/Li_Mh2Mstar_data_Xerr.csv')
Li_yerr = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/theory_Rt/Li_data/Li_Mh2Mstar_data_Yerr.csv')

Li_R = np.array( Li_dat['R'] )


##. Mass infer for scaled R subsamples
def fig_mass_infer():

	##.
	R_str = 'scale'
	# R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m
	R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

	bin_rich = [ 20, 30, 50, 210 ]

	##.
	marker_s = ['o', 's', '^']
	color_s = ['b', 'g', 'r', 'c', 'm']

	fig_name = []

	##.
	for dd in range( len(R_bins) - 1 ):

		if dd == 0:
			fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

		elif dd == len(R_bins) - 2:
			fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

		else:
			fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)

	line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']


	##. Li's data fit params
	cat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/theory_Rt/Li_data/R_Mh_fit_params.csv')
	a_fit, b_fit, c_fit = np.array( cat['a'] )[0], np.array( cat['b'] )[0], np.array( cat['c'] )[0]

	cat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/theory_Rt/Li_data/R_Mstar_fit_params.csv')
	sa_fit, sb_fit, sc_fit = np.array( cat['a'] )[0], np.array( cat['b'] )[0], np.array( cat['c'] )[0]

	new_R = np.logspace( -2, 1, 50 )


	fig = plt.figure( figsize = ( 10, 5 ) )
	ax0 = fig.add_axes( [0.09, 0.10, 0.40, 0.85] )
	ax1 = fig.add_axes( [0.58, 0.10, 0.40, 0.85] )

	ax0.errorbar( Li_R, ref_sub_Mh, yerr = [ref_Mh_err_1, ref_Mh_err_0 ], marker = 'o', ls = '', color = 'k',
				ecolor = 'k', mfc = 'k', mec = 'k', capsize = 1.5, label = 'Li+2016',)
	ax0.plot( new_R, R_func( new_R, a_fit, b_fit, c_fit), 'k-', label = 'Fitting')

	ax0.set_xlabel('$R_{p} \; [Mpc / h]$', fontsize = 12,)
	ax0.set_ylabel('$\\lg M_{h} \; [M_{\\odot} / h]$', fontsize = 12)
	ax0.set_xscale('log')
	ax0.set_xlim( 3e-2, 1.1 )
	ax0.set_ylim( 10.8, 12.8 )

	ax0.fill_betweenx( y = np.linspace(8, 20, 100), x1 = 0.1 * np.ones( 100, ), x2 = 0.3 * np.ones( 100, ), color = 'b', alpha = 0.10,)
	ax0.fill_betweenx( y = np.linspace(8, 20, 100), x1 = 0.3 * np.ones( 100, ), x2 = 0.6 * np.ones( 100, ), color = 'g', alpha = 0.10,)
	ax0.fill_betweenx( y = np.linspace(8, 20, 100), x1 = 0.6 * np.ones( 100, ), x2 = 0.9 * np.ones( 100, ), color = 'r', alpha = 0.10,)


	ax1.plot( Li_R, ref_sat_Ms, 'ko', label = 'Li+2016',)
	ax1.plot( new_R, R_func( new_R, sa_fit, sb_fit, sc_fit), 'k-', label = 'Fitting' )

	ax1.set_xlabel('$R_{p} \; [Mpc / h]$', fontsize = 12,)
	ax1.set_ylabel('$\\lg M_{\\ast} \; [M_{\\odot} / h]$', fontsize = 12)
	ax1.set_xscale('log')
	ax1.set_xlim( 3e-2, 1.1 )
	ax1.set_ylim( 10.6, 10.9 )

	ax1.fill_betweenx( y = np.linspace(8, 20, 100), x1 = 0.1 * np.ones( 100, ), x2 = 0.3 * np.ones( 100, ), color = 'b', alpha = 0.10,)
	ax1.fill_betweenx( y = np.linspace(8, 20, 100), x1 = 0.3 * np.ones( 100, ), x2 = 0.6 * np.ones( 100, ), color = 'g', alpha = 0.10,)
	ax1.fill_betweenx( y = np.linspace(8, 20, 100), x1 = 0.6 * np.ones( 100, ), x2 = 0.9 * np.ones( 100, ), color = 'r', alpha = 0.10,)

	##.
	for tt in range( len(R_bins) - 1 ):

		tt_bcg_z = np.array( [ ] )
		tt_Rsat = np.array( [ ] )

		for ll in range( 3 ):

			dat = pds.read_csv( cat_path + 
				'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_cat.csv' 
				% (bin_rich[ll], bin_rich[ll+1], R_bins[tt], R_bins[tt+1]),)

			tt_bcg_z = np.r_[ tt_bcg_z, np.array( dat['bcg_z'] ) ]
			tt_Rsat = np.r_[ tt_Rsat, np.array( dat['R_sat'] ) ]

		aveg_R_sat = np.median( tt_Rsat )   ##. Mpc / h
		aveg_R_std = np.std( tt_Rsat )

		aveg_Mh = R_func( aveg_R_sat, a_fit, b_fit, c_fit )    ##. M_sun / h
		aveg_Ms = R_func( aveg_R_sat, sa_fit, sb_fit, sc_fit ) ##. M_sun / h


		ax0.errorbar( aveg_R_sat, aveg_Mh, xerr = aveg_R_std, marker = marker_s[1], color = color_s[ tt ], 
				ecolor = color_s[ tt ], mfc = 'none', mec = color_s[ tt ], capsize = 1.5, label = fig_name[ tt ],)

		ax1.errorbar( aveg_R_sat, aveg_Ms, xerr = aveg_R_std, marker = marker_s[1], color = color_s[ tt ], 
				ecolor = color_s[ tt ], mfc = 'none', mec = color_s[ tt ], capsize = 1.5, label = fig_name[ tt ],)

	ax0.legend( loc = 2, frameon = False, fontsize = 12)
	ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	ax1.legend( loc = 2, frameon = False, fontsize = 12)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig('/home/xkchen/sat_Mh_Ms_infer.png', dpi = 300)
	plt.close()

	return

# fig_mass_infer()
# raise


##. halo mass infer
bin_rich = [ 20, 30, 50, 210 ]
sub_name = ['low-rich', 'medi-rich', 'high-rich']

R_str = 'scale'
# R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m
R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

#.
C_arr = []
Mvir_arr = []
Rvir_arr = []
z_arr = []


##. ref. cluster catalog
dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/' + 
					'Extend-BCGM_rgi-common_cat.csv')

bcg_ra, bcg_dec, bcg_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

rich = np.array( dat['rich'] )
clust_ID = np.array( dat['clust_ID'] )
clust_ID = clust_ID.astype( int )

for tt in range( len( R_bins ) - 1 ):

	tt_rich, tt_z = np.array([]), np.array([])

	for ll in range( 3 ):

		# keys = bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, R_sat, R2Rv, clus_ID

		dat = pds.read_csv( cat_path + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_cat.csv' 
			% (bin_rich[ll], bin_rich[ll+1], R_bins[tt], R_bins[tt+1]),)

		sub_IDs = np.array( dat['clus_ID'] )
		
		set_IDs = np.array( list( set( sub_IDs ) ) )	
		set_IDs = set_IDs.astype( int )

		idx = np.where( clust_ID == set_IDs[:,None] )[-1]

		tt_rich = np.r_[ tt_rich, rich[ idx ] ]
		tt_z = np.r_[ tt_z, bcg_z[ idx ] ]

	##.
	tt_M200, tt_R200 = rich2R_Simet( tt_z, tt_rich )   ##. M_sun, kpc
	tt_M200 = tt_M200 * h        ##. M_sun / h
	tt_R200 = tt_R200 * h / 1e3  ##. Mpc / h

	aveg_z = np.mean( tt_z )
	aveg_Mh = np.mean( tt_M200 )
	aveg_c = Mh_c_func( aveg_z, aveg_Mh )

	C_arr.append( aveg_c )
	Mvir_arr.append( aveg_Mh )
	Rvir_arr.append( np.mean( tt_R200 ) )
	z_arr.append( aveg_z )


##. Li's data fit params
cat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/theory_Rt/Li_data/R_Mh_fit_params.csv')
a_fit, b_fit, c_fit = np.array( cat['a'] )[0], np.array( cat['b'] )[0], np.array( cat['c'] )[0]

cat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/theory_Rt/Li_data/R_Mstar_fit_params.csv')
sa_fit, sb_fit, sc_fit = np.array( cat['a'] )[0], np.array( cat['b'] )[0], np.array( cat['c'] )[0]


##.
marker_s = ['o', 's', '^']
color_s = ['b', 'g', 'r', 'c', 'm']

fig_name = []

for dd in range( len(R_bins) - 1 ):

	if dd == 0:
		fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

	elif dd == len(R_bins) - 2:
		fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

	else:
		fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)


##. Rt calculation
pp_Rt = []
pp_alpha_k = []

fig = plt.figure( figsize = ( 10, 5 ) )
ax0 = fig.add_axes( [0.09, 0.10, 0.40, 0.85] )
ax1 = fig.add_axes( [0.58, 0.10, 0.40, 0.85] )

for tt in range( len( R_bins ) - 1 ):

	tt_bcg_ra = np.array([])
	tt_bcg_dec = np.array([])
	tt_bcg_z = np.array([])

	tt_ra = np.array([])
	tt_dec = np.array([])

	tt_Rsat = np.array([])
	tt_R2Rv = np.array([])

	for ll in range( 3 ):

		# keys = bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, R_sat, R2Rv, clus_ID

		dat = pds.read_csv( cat_path + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_cat.csv' 
			% (bin_rich[ll], bin_rich[ll+1], R_bins[tt], R_bins[tt+1]),)

		tt_bcg_z = np.r_[ tt_bcg_z, np.array( dat['bcg_z'] ) ]

		tt_ra = np.r_[ tt_ra, np.array( dat['sat_ra'] ) ]
		tt_dec = np.r_[ tt_dec, np.array( dat['sat_dec'] ) ]

		tt_Rsat = np.r_[ tt_Rsat, np.array( dat['R_sat'] ) ]
		tt_R2Rv = np.r_[ tt_R2Rv, np.array( dat['R2Rv'] ) ]

	##.
	aveg_R_sat = np.median( tt_Rsat )   ##. Mpc / h
	aveg_R_std = np.std( tt_Rsat )

	aveg_Mh = R_func( aveg_R_sat, a_fit, b_fit, c_fit )    ##. M_sun / h
	aveg_Ms = R_func( aveg_R_sat, sa_fit, sb_fit, sc_fit ) ##. M_sun / h

	aveg_z = np.mean( tt_bcg_z )
	aveg_c = Mh_c_func( aveg_z, 10**aveg_Mh )


	##. density profile of satellite halo
	halo_nfw = profile_nfw.NFWProfile( M = Mvir_arr[ tt ], c = C_arr[ tt ], z = z_arr[ tt ], mdef = '200m')
	sub_nfw = profile_nfw.NFWProfile( M = 10**aveg_Mh, c = aveg_c, z = aveg_z, mdef = '200m')

	##. kpc / h
	nr = 200
	r_bins = np.logspace( -3, 3.2, nr )

	##. assume the r_bins is projected distance, calculate the 3D distance and background
	dR0 = 1e3 * aveg_R_sat    ##. kpc / h
	dR1 = np.sqrt( (1e3 * Rvir_arr[ tt ] )**2 - dR0**2 )

	x_mm = np.logspace( -3, np.log10( dR1 ), 200 )
	r_mm = np.sqrt( dR0**2 + x_mm**2 )

	pm_F_halo = halo_nfw.density( r_mm )
	pm_F_weit = halo_nfw.density( r_mm ) * r_mm

	##.
	rho_m_z = my_cosmo.rho_m( aveg_z )
	rho_delta = 200 * rho_m_z

	pm_rho_h = pm_F_halo / rho_delta
	pm_rho_weit = pm_F_weit / rho_delta

	order = 7

	[ans, err] = GaussLegendreQuad_arr( x_mm, pm_rho_h, order, 0, x_mm[-1] )
	I0 = ans * rho_delta

	[ans, err] = GaussLegendreQuad_arr( x_mm, pm_rho_weit, order, 0, x_mm[-1] )
	I1 = ans * rho_delta

	##. 3D centric distance of subhalo
	tag_R = I1 / I0


	##.
	enclos_Mh_sat = sub_nfw.enclosedMass( r_bins )
	aveg_rho_sat = 3 * enclos_Mh_sat / ( 4 * np.pi * r_bins**3 )

	tag_enclos_mass = halo_nfw.enclosedMass( tag_R )
	tag_BG = 3 * tag_enclos_mass / (4 * np.pi * tag_R**3 )

	##. slope of the host halo density profile
	r_enclos_M = halo_nfw.enclosedMass( r_bins )

	mean_rho = 3 * r_enclos_M / ( 4 * np.pi * r_bins**3 )

	ln_rho = np.log( mean_rho )
	ln_R = np.log( r_bins )

	diff_x = np.diff( ln_R )
	diff_y = np.diff( ln_rho )
	slop_Mh = diff_y / diff_x

	tmp_k_F = interp.interp1d( r_bins[1:], slop_Mh, kind = 'cubic', fill_value = 'extrapolate',)
	alpha_k = tmp_k_F( tag_R )

	##.
	tmp_F = interp.interp1d( r_bins, aveg_rho_sat, kind = 'cubic', fill_value = 'extrapolate')
	c_rt = inversefunc( tmp_F, np.abs( alpha_k ) * tag_BG )

	pp_Rt.append( c_rt.min() )
	pp_alpha_k.append( alpha_k.min() )


	##. figs
	ax0.plot( r_bins, aveg_rho_sat, ls = '-', color = color_s[tt], alpha = 0.75, label = fig_name[tt],)
	ax0.axhline( y = np.abs( alpha_k.min() ) * tag_BG, ls = ':', color = color_s[tt],)
	ax0.axvline( x = c_rt.min(), ls = '--', color = color_s[tt], ymin = 0.9, ymax = 1.)

	ax1.scatter( 0.5 * ( R_bins[tt] + R_bins[tt+1]), alpha_k, marker = 's', s = 75, color = color_s[tt], label = fig_name[tt],)


##. save the tidal radius~(kpc/h)
keys = [ '%s' % ll for ll in fig_name ]
values = pp_Rt
fill = dict( zip( keys, values ), index = ('k', 'v') )
data = pds.DataFrame( fill )
data.to_csv( out_path + 
			'Extend_BCGM_gri-common_Rs-bin_over-rich_sat_Rt.csv',)


##.
ax1.plot( 0.5 * ( R_bins[1:] + R_bins[:-1]), pp_alpha_k, 'k-', alpha = 0.65)
ax1.legend( loc = 1, frameon = False, fontsize = 12,)
ax1.set_xlabel('$R_{sat} / R_{200m}$', fontsize = 12,)
ax1.set_ylim( -2.2, -1.6 )
ax1.set_ylabel('$\\alpha \, = \, d \, \\ln \, \\bar{\\rho} \, / \, d \, \\ln \, r $', fontsize = 12,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
ax1.xaxis.set_minor_locator( ticker.AutoMinorLocator() )


ax0.legend( loc = 3, frameon = False, fontsize = 12,)
ax0.set_xlim( 1, 2e2 )
ax0.set_xscale('log')
ax0.set_xlabel('$r \; [kpc / h]$', fontsize = 12,)
ax0.set_ylim( 1e3, 4e8 )
ax0.set_yscale('log')
ax0.set_ylabel('$\\bar{\\rho}(r) [M_\odot h^{2} kpc^{-3}]$', fontsize = 12,)
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

plt.savefig( '/home/xkchen/halo_sat_ebclose_3D_aveg_rho.png', dpi = 300)
plt.close()

raise
