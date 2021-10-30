import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C
import scipy.signal as signal

from astropy import cosmology as apcy
from scipy import interpolate as interp
from scipy import integrate as integ
from astropy.coordinates import SkyCoord

from scipy import optimize
from scipy import stats as sts

from BCG_SB_pro_stack import single_img_SB_func
from color_2_mass import SB_to_Lumi_func
from tmp_color_to_mass import gr_ri_band_c2m_func

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

### === ### cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

pixel = 0.396
band = ['r', 'g', 'i']
L_wave = np.array([ 6166, 4686, 7480 ])
Mag_sun = [ 4.65, 5.11, 4.53 ]
z_ref = 0.25

def M20_binned_func( x_bins, data_arr, out_file_hi, out_file_low, divid_f):

	hi_map_rich, hi_map_M20 = np.array( [] ), np.array( [] )
	hi_map_ra, hi_map_dec, hi_map_z = np.array( [] ), np.array( [] ), np.array( [] )
	hi_map_Mstar = np.array( [] )

	lo_map_rich, lo_map_M20 = np.array( [] ), np.array( [] )
	lo_map_ra, lo_map_dec, lo_map_z = np.array( [] ), np.array( [] ), np.array( [] )
	lo_map_Mstar = np.array( [] )

	N_bin = len( x_bins )
	ra, dec, z, rich, lg_Mstar, lg_M20 = data_arr[:]

	for ii in range( N_bin - 1):

		lg_rich = np.log10( rich )
		id_lim = ( lg_rich >= x_bins[ii] ) & ( lg_rich < x_bins[ii+1] )

		axis_x = ( x_bins[ii] + x_bins[ii+1] ) / 2

		thresh_lgM20 = divid_f( axis_x )

		dd_lim_rich = rich[ id_lim ]
		dd_lim_ra, dd_lim_dec, dd_lim_zc = ra[id_lim], dec[id_lim], z[id_lim]
		dd_lim_Mstar = lg_Mstar[ id_lim ]

		dd_lim_M20 = lg_M20[ id_lim ]


		idvx_hi = dd_lim_M20 >= thresh_lgM20
		idvx_lo = dd_lim_M20 < thresh_lgM20


		hi_map_rich = np.r_[ hi_map_rich, dd_lim_rich[ idvx_hi ]  ]
		hi_map_ra = np.r_[ hi_map_ra, dd_lim_ra[ idvx_hi ] ]
		hi_map_dec = np.r_[ hi_map_dec, dd_lim_dec[ idvx_hi ] ]
		hi_map_z = np.r_[ hi_map_z, dd_lim_zc[ idvx_hi ] ]
		hi_map_Mstar = np.r_[ hi_map_Mstar, dd_lim_Mstar[ idvx_hi ] ]
		hi_map_M20 = np.r_[ hi_map_M20, dd_lim_M20[ idvx_hi ] ]

		lo_map_rich = np.r_[ lo_map_rich, dd_lim_rich[ idvx_lo ] ]
		lo_map_ra = np.r_[ lo_map_ra, dd_lim_ra[ idvx_lo ] ]
		lo_map_dec = np.r_[ lo_map_dec, dd_lim_dec[ idvx_lo ] ]
		lo_map_z = np.r_[ lo_map_z, dd_lim_zc[ idvx_lo ] ]
		lo_map_Mstar = np.r_[ lo_map_Mstar, dd_lim_Mstar[ idvx_lo ] ]
		lo_map_M20 = np.r_[ lo_map_M20, dd_lim_M20[ idvx_lo ] ]

	## save the divided sample
	keys = ['ra', 'dec', 'z', 'rich', 'lg_Mstar', 'lg_M_R-lim']
	values = [ hi_map_ra, hi_map_dec, hi_map_z, hi_map_rich, hi_map_Mstar, hi_map_M20 ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_file_hi )

	keys = ['ra', 'dec', 'z', 'rich', 'lg_Mstar', 'lg_M_R-lim']
	values = [ lo_map_ra, lo_map_dec, lo_map_z, lo_map_rich, lo_map_Mstar, lo_map_M20 ]
	fill = dict( zip(keys, values) )
	data = pds.DataFrame(fill)
	data.to_csv( out_file_low )

	print( len( lo_map_ra ) )
	print( len( hi_map_ra ) )

	return

def lg_linea_func( x, a, b):
	y = a + b * x
	return y

def resi_func( po, x, y):

	a, b = po[:]
	y_mod = lg_linea_func( x, a, b)
	delta = y_mod - y

	return delta

### === ### subsample division
tot_lo_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/' + 'low_star-Mass_cat.csv')
tot_lo_lgM = np.array( tot_lo_dat['lg_Mass'] )
tot_lo_rich = np.array( tot_lo_dat['rich'] )
tot_lo_ra, tot_lo_dec, tot_lo_z = np.array( tot_lo_dat['ra'] ), np.array( tot_lo_dat['dec'] ), np.array( tot_lo_dat['z'] )

tot_hi_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/' + 'high_star-Mass_cat.csv')
tot_hi_lgM = np.array( tot_hi_dat['lg_Mass'] )
tot_hi_rich = np.array( tot_hi_dat['rich'] )
tot_hi_ra, tot_hi_dec, tot_hi_z = np.array( tot_hi_dat['ra'] ), np.array( tot_hi_dat['dec'] ), np.array( tot_hi_dat['z'] )

tot_ra = np.r_[ tot_lo_ra, tot_hi_ra ]
tot_dec = np.r_[ tot_lo_dec, tot_hi_dec ]
tot_z = np.r_[ tot_lo_z, tot_hi_z ]
tot_rich = np.r_[ tot_lo_rich, tot_hi_rich ]

#. change the mass unit from M_sun / h^2 to M_sun
tot_lgM = np.r_[ tot_lo_lgM, tot_hi_lgM ] - 2 * np.log10( h )

#... lgM20 or lgM10 binned subsamples
# y_label_s = '$ \\lg M^{20}_{\\ast} \; [M_{\\odot}]$'
# sub_divi_s = '/home/xkchen/%s-band_M_20_division.png'

y_label_s = '$ \\lg M^{10}_{\\ast} \; [M_{\\odot}]$'
sub_divi_s = '/home/xkchen/%s-band_M_10_division.png'

for kk in range( 3 ):

	# p_dat = pds.read_csv('/home/xkchen/figs/BCG_aper_M/%s-band_BCG-M_bin_aperture_M.csv' % band[kk] )
	# p_ra, p_dec, p_z = np.array( p_dat['ra'] ), np.array( p_dat['dec'] ), np.array( p_dat['z'] )

	# #. Aperture Mass
	# p_M20 = np.array( p_dat['M_20'] )
	# # p_M20 = np.array( p_dat['M_10'] )

	m_dat = pds.read_csv( '/home/xkchen/figs/BCG_aper_M/%s-band_BCG-mag_cat.csv' % band[kk] )
	p_ra, p_dec, p_z = np.array( m_dat['ra'] ), np.array( m_dat['dec'] ), np.array( m_dat['z'] )
	# p_M20 = np.array( m_dat['Mstar_20'] )
	p_M20 = np.array( m_dat['Mstar_10'] )

	id_nul = np.isnan( p_M20 )
	p_ra, p_dec, p_z = p_ra[ id_nul == False ], p_dec[ id_nul == False ], p_z[ id_nul == False ]
	p_M20 = p_M20[ id_nul == False ]


	p_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg,)
	tot_coord = SkyCoord( ra = tot_ra * U.deg, dec = tot_dec * U.deg,)

	idx_0, sep_0, d3d_0 = p_coord.match_to_catalog_sky( tot_coord )
	id_lim_0 = sep_0.value < 2.7e-4

	mp_ra, mp_dec, mp_z = tot_ra[ idx_0[ id_lim_0 ] ], tot_dec[ idx_0[ id_lim_0 ] ], tot_z[ idx_0[ id_lim_0 ] ]
	mp_rich, mp_lgM = tot_rich[ idx_0[ id_lim_0 ] ], tot_lgM[ idx_0[ id_lim_0 ] ]

	lim_ra, lim_dec, lim_z = p_ra[ id_lim_0 ], p_dec[ id_lim_0 ], p_z[ id_lim_0 ]
	lim_lgM20 = np.log10( p_M20[ id_lim_0 ] )


	#. save the matched properties
	# keys = ['ra', 'dec', 'z', 'rich', 'lg_Mbcg', 'lg_Mbcg_20']
	# values = [ lim_ra, lim_dec, lim_z, mp_rich, mp_lgM, lim_lgM20 ]
	# fill = dict( zip( keys, values ) )
	# data = pds.DataFrame( fill )
	# data.to_csv( '/home/xkchen/%s-band_BCG_aperture-M_cat_params.csv' % band[kk] )

	keys = ['ra', 'dec', 'z', 'rich', 'lg_Mbcg', 'lg_Mbcg_10']
	values = [ lim_ra, lim_dec, lim_z, mp_rich, mp_lgM, lim_lgM20 ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( '/home/xkchen/%s-band_BCG_aperture-M-10-kpc_cat_params.csv' % band[kk] )


	#. divide sample based on M20
	pre_N = 8
	
	# x_bins = np.linspace( 10.5, 12.3, pre_N )
	x_bins = np.linspace( np.log10( mp_rich.min() ), np.log10( mp_rich.max() ), pre_N )

	medi_lgM20 = []

	for jj in range( pre_N - 1 ):

		id_x = ( x_bins[jj] < np.log10( mp_rich ) ) & ( np.log10( mp_rich ) <= x_bins[jj+1] ) 

		_sub_lgM20 = lim_lgM20[ id_x ]
		medi_lgM20.append( np.median( _sub_lgM20 ) )

	cen_x = 0.5 * ( x_bins[1:] + x_bins[:-1] )

	div_Pf = interp.interp1d( cen_x, medi_lgM20, kind = 'linear', fill_value = 'extrapolate',)

	N_bins = 30
	bins_lg_rich = np.linspace( 1.30, 2.31, N_bins )

	plt.figure()
	plt.plot( np.log10( mp_rich ), lim_lgM20, 'ro', alpha = 0.5, markersize = 1,)
	
	plt.plot( cen_x, medi_lgM20, 'ks-', )
	plt.plot( bins_lg_rich, div_Pf( bins_lg_rich ), 'b--',)

	plt.ylabel( y_label_s )
	plt.xlabel('$ \\lg \, \\lambda $')
	plt.ylim( 10.5, 12 )
	plt.savefig( sub_divi_s % band[kk], dpi = 300)
	plt.close()

	#. subsample division
	data_arr = [ lim_ra, lim_dec, lim_z, mp_rich, mp_lgM, lim_lgM20 ]

	# out_file_hi = '/home/xkchen/photo-z_match_%s-band_hi-lgM20_cluster_cat.csv' % band[kk]
	# out_file_low = '/home/xkchen/photo-z_match_%s-band_low-lgM20_cluster_cat.csv' % band[kk]

	out_file_hi = '/home/xkchen/photo-z_match_%s-band_hi-lgM10_cluster_cat.csv' % band[kk]
	out_file_low = '/home/xkchen/photo-z_match_%s-band_low-lgM10_cluster_cat.csv' % band[kk]

	M20_binned_func( bins_lg_rich, data_arr, out_file_hi, out_file_low, div_Pf )


#.. overlap catalog of g, r, i band
# cat_lis = ['low-lgM20', 'hi-lgM20']
# fig_name = ['$Low \; M_{\\ast, \, 20}$', '$High \; M_{\\ast, \, 20}$']

cat_lis = ['low-lgM10', 'hi-lgM10']
fig_name = ['$Low \; M_{\\ast, \, 10}$', '$High \; M_{\\ast, \, 10}$']

dmp_rich, dmp_lgM, dmp_lgM20 = [], [], []

for ll in range( 2 ):

	r_dat = pds.read_csv('/home/xkchen/' + 'photo-z_match_r-band_%s_cluster_cat.csv' % cat_lis[ll] )
	r_ra, r_dec, r_z = np.array( r_dat['ra'] ), np.array( r_dat['dec'] ), np.array( r_dat['z'] )

	r_coord = SkyCoord( ra = r_ra * U.deg, dec = r_dec * U.deg,)


	i_dat = pds.read_csv('/home/xkchen/' + 'photo-z_match_i-band_%s_cluster_cat.csv' % cat_lis[ll])
	i_ra, i_dec, i_z = np.array( i_dat['ra'] ), np.array( i_dat['dec'] ), np.array( i_dat['z'] )

	i_coord = SkyCoord( ra = i_ra * U.deg, dec = i_dec * U.deg,)


	idx_1, sep_1, d3d_1 = i_coord.match_to_catalog_sky( r_coord )
	id_lim_1 = sep_1.value < 2.7e-4

	_medi_ra, _medi_dec, _medi_z = i_ra[ id_lim_1 ], i_dec[ id_lim_1 ], i_z[ id_lim_1 ]

	_medi_coord = SkyCoord( ra = _medi_ra * U.deg, dec = _medi_dec * U.deg,)


	g_dat = pds.read_csv('/home/xkchen/' + 'photo-z_match_g-band_%s_cluster_cat.csv' % cat_lis[ll])
	g_ra, g_dec, g_z = np.array( g_dat['ra'] ), np.array( g_dat['dec'] ), np.array( g_dat['z'] )

	_ii_rich, _ll_lgM, _ii_lgM20 = np.array( g_dat['rich'] ), np.array( g_dat['lg_Mstar'] ), np.array( g_dat['lg_M_R-lim'] )

	g_coord = SkyCoord( ra = g_ra * U.deg, dec = g_dec * U.deg,)

	idx_0, sep_0, d3d_0 = _medi_coord.match_to_catalog_sky( g_coord )
	id_lim_0 = sep_0.value < 2.7e-4

	ovep_ra, ovep_dec, ovep_z = _medi_ra[ id_lim_0 ], _medi_dec[ id_lim_0 ], _medi_z[ id_lim_0 ]


	print( 'N_g = ', len( ovep_ra) )

	keys = ['ra', 'dec', 'z']
	values = [ ovep_ra, ovep_dec, ovep_z ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( '/home/xkchen/photo-z_match_%s_gri-common_cluster_cat.csv' % cat_lis[ll] )

	#. sample properties
	dmp_rich.append( _ii_rich[ idx_0[ id_lim_0 ] ] )
	dmp_lgM.append( _ll_lgM[ idx_0[ id_lim_0 ] ] )
	dmp_lgM20.append( _ii_lgM20[ idx_0[ id_lim_0 ] ] )


fig = plt.figure()
ax = plt.subplot(111)
ax.hist( dmp_rich[0], bins = 50, density = True, histtype = 'step', color = 'b', alpha = 0.75, 
		ls = '--', label = fig_name[0],)
ax.axvline( x = np.median( dmp_rich[0] ), ls = '--', color = 'b', alpha = 0.75, ymin = 0, ymax = 0.35,)
ax.hist( dmp_rich[1], bins = 50, density = True, histtype = 'step', color = 'r', alpha = 0.75, label = fig_name[1],)
ax.axvline( x = np.median( dmp_rich[1] ), ls = '-', color = 'r', alpha = 0.75, ymin = 0, ymax = 0.35,)

ax.set_xlabel('$\\lambda$')
ax.set_ylabel('PDF')
ax.set_xscale('log')
ax.set_yscale('log')

ax.legend( loc = 1, frameon = False,)

plt.savefig('/home/xkchen/rich_compare.png', dpi = 300)
plt.close()


fig = plt.figure()
ax = plt.subplot(111)
ax.hist( dmp_lgM[0], bins = 50, density = True, histtype = 'step', color = 'b', alpha = 0.75, 
		ls = '--', label = fig_name[0],)
ax.axvline( x = np.median( dmp_lgM[0] ), ls = '--', color = 'b', alpha = 0.75, ymin = 0, ymax = 0.35,)
ax.hist( dmp_lgM[1], bins = 50, density = True, histtype = 'step', color = 'r', alpha = 0.75, label = fig_name[1],)
ax.axvline( x = np.median( dmp_lgM[1] ), ls = '-', color = 'r', alpha = 0.75, ymin = 0, ymax = 0.35,)

ax.set_xlabel('$\\lg \, M^{ \\mathrm{SED} }_{\\ast} \; [M_{\\odot}]$')
ax.set_ylabel('PDF')

ax.legend( loc = 1, frameon = False,)

plt.savefig('/home/xkchen/lgM_compare.png', dpi = 300)
plt.close()


fig = plt.figure()
ax = plt.subplot(111)
ax.hist( dmp_lgM20[0], bins = 50, density = True, histtype = 'step', color = 'b', alpha = 0.75, 
		ls = '--', label = fig_name[0],)
ax.axvline( x = np.median( dmp_lgM20[0] ), ls = '--', color = 'b', alpha = 0.75, ymin = 0, ymax = 0.35,)
ax.hist( dmp_lgM20[1], bins = 50, density = True, histtype = 'step', color = 'r', alpha = 0.75, label = fig_name[1],)
ax.axvline( x = np.median( dmp_lgM20[1] ), ls = '-', color = 'r', alpha = 0.75, ymin = 0, ymax = 0.35,)

ax.set_xlabel( y_label_s )

ax.set_ylabel('PDF')
ax.set_xlim( 10.8, 12.0)
ax.legend( loc = 1, frameon = False,)

plt.savefig('/home/xkchen/lgM-R-lim_compare.png', dpi = 300)
plt.close()

