"""
3d centric distance infer for given halo density profiles
"""
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

import time
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


##. halo mass infer
bin_rich = [ 20, 30, 50, 210 ]
sub_name = ['low-rich', 'medi-rich', 'high-rich']

R_str = 'scale'
R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m


### === ### func.s
def R_func( R, a, b, c):
    return a + (R + b)**c



### === ### data load
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/cat/'
out_path = '/home/xkchen/figs_cp/theory_Rt/'


##. Li's data fit params
cat = pds.read_csv('/home/xkchen/figs_cp/theory_Rt/Li_data/R_Mh_fit_params.csv')
a_fit, b_fit, c_fit = np.array( cat['a'] )[0], np.array( cat['b'] )[0], np.array( cat['c'] )[0]


##. ref. cluster catalog
dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/' + 
					'Extend-BCGM_rgi-common_cat.csv')

bcg_ra, bcg_dec, bcg_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

rich = np.array( dat['rich'] )
clust_ID = np.array( dat['clust_ID'] )
clust_ID = clust_ID.astype( int )

#.
C_arr = []
Mvir_arr = []
Rvir_arr = []
z_arr = []

for tt in range( len( R_bins ) - 1 ):

	#.
	kk_C = []
	kk_M = []
	kk_R = []
	kk_z = []

	for ll in range( 3 ):

		#. keys = bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, R_sat, R2Rv, clus_ID
		dat = pds.read_csv( cat_path + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_cat.csv' 
			% (bin_rich[ll], bin_rich[ll+1], R_bins[tt], R_bins[tt+1]),)

		sub_IDs = np.array( dat['clus_ID'] )
		
		set_IDs = np.array( list( set( sub_IDs ) ) )	
		set_IDs = set_IDs.astype( int )

		idx = np.where( clust_ID == set_IDs[:,None] )[-1]

		#.
		tt_rich = rich[ idx ]
		tt_z = bcg_z[ idx ]

		tt_M200, tt_R200 = rich2R_Simet( tt_z, tt_rich )   ##. M_sun, kpc
		tt_M200 = tt_M200 * h        ##. M_sun / h
		tt_R200 = tt_R200 * h / 1e3  ##. Mpc / h

		aveg_z = np.mean( tt_z )
		aveg_Mh = np.mean( tt_M200 )
		aveg_c = Mh_c_func( aveg_z, aveg_Mh )

		#.
		kk_C.append( aveg_c )
		kk_M.append( aveg_Mh )
		kk_z.append( aveg_z )
		kk_R.append( np.mean( tt_R200 ) )

	#.
	C_arr.append( kk_C )
	Mvir_arr.append( kk_M )
	Rvir_arr.append( kk_R )
	z_arr.append( kk_z )


##. 3D distance infer
R_str = 'scale'
R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m

R_3d_arr = []
alpha_arr = []
Rt_arr = []

#.
for tt in range( len( R_bins ) - 1 ):

	#.
	tt_Rt = np.array( [] )
	tt_alpha = np.array( [] )
	tt_R3d = np.array( [] )

	for ll in range( 3 ):

		# keys = bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, R_sat, R2Rv, clus_ID

		dat = pds.read_csv( cat_path + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_cat.csv' 
			% (bin_rich[ll], bin_rich[ll+1], R_bins[tt], R_bins[tt+1]),)

		tt_bcg_z = np.array( dat['bcg_z'] )
		tt_Rsat = np.array( dat['R_sat'] )
		tt_R2Rv = np.array( dat['R2Rv'] )

		##. density profile of satellite halo
		halo_nfw = profile_nfw.NFWProfile( M = Mvir_arr[tt][ll], c = C_arr[tt][ll], z = z_arr[tt][ll], mdef = '200m')

		##. kpc / h
		nr = 200
		r_bins = np.logspace( -3, 3.2, nr )

		##. background density
		rho_m_z = my_cosmo.rho_m( z_arr[tt][ll] )
		rho_delta = 200 * rho_m_z

		##. assume the r_bins is projected distance, calculate the 3D distance and background
		NN_k = len( tt_Rsat )

		pp_Rt = np.zeros( NN_k,)
		pp_Rx = np.zeros( NN_k,)
		pp_alpha_k = np.zeros( NN_k,)

		for pp in range( NN_k ):

			dR0 = 1e3 * tt_Rsat[ pp ]    ##. kpc / h
			dR1 = np.sqrt( (1e3 * Rvir_arr[tt][ll] )**2 - dR0**2 )

			x_mm = np.logspace( -3, np.log10( dR1 ), 200 )
			r_mm = np.sqrt( dR0**2 + x_mm**2 )

			pm_F_halo = halo_nfw.density( r_mm )
			pm_F_weit = halo_nfw.density( r_mm ) * r_mm

			pm_rho_h = pm_F_halo / rho_delta
			pm_rho_weit = pm_F_weit / rho_delta

			##.
			# order = 5  ##. 3, 5, 7

			# [ans, err] = GaussLegendreQuad_arr( x_mm, pm_rho_h, order, 0, x_mm[-1] )
			# I0 = ans + 0.

			# [ans, err] = GaussLegendreQuad_arr( x_mm, pm_rho_weit, order, 0, x_mm[-1] )
			# I1 = ans + 0.

			I0 = integ.simps( pm_rho_h, x_mm )
			I1 = integ.simps( pm_rho_weit, x_mm )

			##. 3D centric distance of subhalo
			tag_R = I1 / I0


			##.
			tag_enclos_mass = halo_nfw.enclosedMass( tag_R )
			tag_BG = 3 * tag_enclos_mass / (4 * np.pi * tag_R**3 )

			##. slope of the host halo density profile
			r_enclos_M = halo_nfw.enclosedMass( r_bins )

			mean_rho = 3 * r_enclos_M / ( 4 * np.pi * r_bins**3 )

			ln_rho = np.log( mean_rho )
			ln_R = np.log( r_bins )

			diff_x = np.gradient( ln_R )
			diff_y = np.gradient( ln_rho )
			slop_Mh = diff_y / diff_x

			tmp_k_F = interp.interp1d( r_bins, slop_Mh, kind = 'cubic', fill_value = 'extrapolate',)
			alpha_k = tmp_k_F( tag_R )


			##.
			tp_Mh = R_func( tt_Rsat[ pp ], a_fit, b_fit, c_fit )    ##. M_sun / h
			tp_c = Mh_c_func( tt_bcg_z[ pp ], 10**tp_Mh )

			sub_nfw = profile_nfw.NFWProfile( M = 10**tp_Mh, c = tp_c, z = tt_bcg_z[ pp ], mdef = '200m')

			enclos_Mh_sat = sub_nfw.enclosedMass( r_bins )
			aveg_rho_sat = 3 * enclos_Mh_sat / ( 4 * np.pi * r_bins**3 )

			tmp_F = interp.interp1d( r_bins, aveg_rho_sat, kind = 'cubic', fill_value = 'extrapolate')
			c_rt = inversefunc( tmp_F, np.abs( alpha_k ) * tag_BG )

			##.
			pp_alpha_k[ pp ] = alpha_k.min()
			pp_Rx[ pp ] = tag_R
			pp_Rt[ pp ] = c_rt.min()

		##.
		tt_Rt = np.r_[ tt_Rt, pp_Rt ]
		tt_alpha = np.r_[ tt_alpha, pp_alpha_k ]
		tt_R3d = np.r_[ tt_R3d, pp_Rx ]

	##.
	R_3d_arr.append( tt_R3d )
	alpha_arr.append( tt_alpha )
	Rt_arr.append( tt_Rt )

raise

##. figs
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


#.
fig = plt.figure( figsize = (10.4, 4.8) )
ax0 = fig.add_axes([0.07, 0.11, 0.40, 0.80])
ax1 = fig.add_axes([0.57, 0.11, 0.40, 0.80])

for tt in range( len( R_bins ) - 1 ):

	ax0.hist( R_3d_arr[ tt ], bins = 35, density = True, histtype = 'step', color = color_s[ tt ], 
				label = fig_name[tt],)

	ax1.hist( alpha_arr[ tt ], bins = 35, density = True, histtype = 'step', color = color_s[ tt ], 
				label = fig_name[tt],)

ax0.set_xlabel('3D cnetric distance [kpc / h]')
ax0.set_xscale('log')
ax0.legend( loc = 2, frameon = False,)

ax1.set_xlabel('$\\alpha_{k}$')
ax1.legend( loc = 1, frameon = False,)

plt.savefig('/home/xkchen/sat_3D-pos.png', dpi = 300)
plt.close()

