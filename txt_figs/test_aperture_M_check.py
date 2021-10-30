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


### === ### catalog comparison
'''
#... catalog among different division
for kk in range( 3 ):

	hi_dat = pds.read_csv( '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_match/' + 
							'high_BCG_star-Mass_%s-band_photo-z-match_BCG-pos_cat.csv' % band[kk] )

	# hi_dat = pds.read_csv( '/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/rich_bin/' + 
	# 						'hi-rich_%s-band_photo-z-match_BCG-pos_cat.csv' % band[kk] )

	# hi_dat = pds.read_csv( '/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/age_bin/' + 
	# 						'hi-age_%s-band_photo-z-match_BCG-pos_cat.csv' % band[kk] )

	hi_ra, hi_dec, hi_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )
	hi_bcgx, hi_bcgy = np.array( hi_dat['bcg_x'] ), np.array( hi_dat['bcg_y'] )


	lo_dat = pds.read_csv( '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_match/' + 
							'low_BCG_star-Mass_%s-band_photo-z-match_BCG-pos_cat.csv' % band[kk] )

	# lo_dat = pds.read_csv( '/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/rich_bin/' + 
	# 						'low-rich_%s-band_photo-z-match_BCG-pos_cat.csv' % band[kk] )

	# lo_dat = pds.read_csv( '/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/age_bin/' + 
	# 						'low-age_%s-band_photo-z-match_BCG-pos_cat.csv' % band[kk] )

	lo_ra, lo_dec, lo_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )
	lo_bcgx, lo_bcgy = np.array( lo_dat['bcg_x'] ), np.array( lo_dat['bcg_y'] )

	cc_ra = np.r_[ hi_ra, lo_ra ]
	cc_dec = np.r_[ hi_dec, lo_dec ]
	cc_z = np.r_[ hi_z, lo_z ]

	cc_bcgx = np.r_[ hi_bcgx, lo_bcgx ]
	cc_bcgy = np.r_[ hi_bcgy, lo_bcgy ]

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
	values = [ cc_ra, cc_dec, cc_z, cc_bcgx, cc_bcgy ]

	fill = dict( zip(keys, values) )
	data = pds.DataFrame( fill )

	data.to_csv( '/home/xkchen/%s-band_BCG-M_cat.csv' % band[kk] )
	# data.to_csv( '/home/xkchen/%s-band_rich-bin_cat.csv' % band[kk] )
	# data.to_csv( '/home/xkchen/%s-band_age-bin_cat.csv' % band[kk] )

for kk in range( 3 ):

	dat_0 = pds.read_csv( '/home/xkchen/%s-band_BCG-M_cat.csv' % band[kk] )
	ra_0, dec_0 = np.array( dat_0['ra'] ), np.array( dat_0['dec'] )

	dat_1 = pds.read_csv( '/home/xkchen/%s-band_age-bin_cat.csv' % band[kk] )
	ra_1, dec_1 = np.array( dat_1['ra'] ), np.array( dat_1['dec'] )

	dat_2 = pds.read_csv( '/home/xkchen/%s-band_rich-bin_cat.csv' % band[kk] )
	ra_2, dec_2 = np.array( dat_2['ra'] ), np.array( dat_2['dec'] )

	## check overlap catalog
	m_coord = SkyCoord( ra = ra_0*U.deg, dec = dec_0*U.deg,)
	age_coord = SkyCoord( ra = ra_1*U.deg, dec = dec_1*U.deg,)
	rich_coord = SkyCoord( ra = ra_2*U.deg, dec = dec_2*U.deg,)

	idx_0, sep_0, d3d_0 = age_coord.match_to_catalog_sky( m_coord )
	idx_1, sep_1, d3d_1 = rich_coord.match_to_catalog_sky( m_coord )

	id_lim_0 = sep_0.value < 2.7e-4
	id_lim_1 = sep_1.value < 2.7e-4

	print( len( ra_0 ) )

	print('*' * 10)
	print( 'N_age = ', len(ra_1) )
	print( np.sum( id_lim_0) )

	print('*' * 10)
	print( 'N_rich = ', len(ra_1) )
	print( np.sum( id_lim_1) )
'''

### == ### magnitude comparison
"""
p_dat = pds.read_csv('/home/xkchen/figs/BCG_aper_M/r-band_BCG-mag_cat.csv')
p_imag_20, p_imag_10 = np.array( p_dat['imag_20'] ), np.array( p_dat['imag_10'] )

id_nul = np.isnan( p_imag_20 )
idx_0 = id_nul == False
p_imag_20 = p_imag_20[ idx_0 ]

id_nul = np.isnan( p_imag_10 )
idx_1 = id_nul == False
p_imag_10 = p_imag_10[ idx_1 ]

#.
sql_dat = pds.read_csv('/home/xkchen/figs/BCG_aper_M/r-band_BCG-properties.csv', skiprows = 1)
sql_imag = np.array( sql_dat['cModelMag_i'] )


p0 = [ 1.3, 10.5 ]
res_lsq = optimize.least_squares( resi_func, x0 = np.array( p0 ), loss = 'cauchy', f_scale = 0.1, 
									args = ( p_imag_20, sql_imag[idx_0] ),)

a_fit = res_lsq.x[0]
b_fit = res_lsq.x[1]

bins_x = np.linspace(15.5, 19.5, 100)
fit_line = lg_linea_func( bins_x, a_fit, b_fit )

fit_points = lg_linea_func(p_imag_20, a_fit, b_fit )
Var = np.sum( ( fit_points - sql_imag[idx_0] )**2 ) / len( sql_imag[idx_0] )
sigma = np.sqrt( Var )

plt.figure()
plt.scatter( p_imag_20, sql_imag[idx_0], marker = '.', color = 'k', s = 5, alpha = 0.45,)
plt.plot( bins_x, bins_x, 'r--',)

plt.plot( bins_x, fit_line, ls = '-', color = 'b', alpha = 0.75,)
plt.plot( bins_x, fit_line - sigma, ls = ':', color = 'b', alpha = 0.75,)
plt.plot( bins_x, fit_line + sigma, ls = ':', color = 'b', alpha = 0.75,)
plt.text( 17, 18.5, s = '$\\alpha$ = %.3f, $\\beta$ = %.3f' % (b_fit, a_fit) + '\n' + '$\\sigma = %.3f$' % sigma,)

plt.xlim( 19.5, 15.5 )
plt.ylim( 19.5, 15.5 )
plt.xlabel('i_mag $[R \\leq 20 \\mathrm{k}pc]$')
plt.ylabel('SDSS, i_cmag')
plt.savefig('/home/xkchen/i_mag20_compare.png', dpi = 300)
plt.close()


p0 = [ 1.3, 10.5 ]
res_lsq = optimize.least_squares( resi_func, x0 = np.array( p0 ), loss = 'cauchy', f_scale = 0.1, 
									args = ( p_imag_10, sql_imag[idx_1] ),)

a_fit = res_lsq.x[0]
b_fit = res_lsq.x[1]

bins_x = np.linspace(15, 19.5, 100)
fit_line = lg_linea_func( bins_x, a_fit, b_fit )

fit_points = lg_linea_func(p_imag_10, a_fit, b_fit )
Var = np.sum( ( fit_points - sql_imag[idx_1] )**2 ) / len( sql_imag[idx_1] )
sigma = np.sqrt( Var )

plt.figure()
plt.scatter( p_imag_10, sql_imag[idx_1], marker = '.', color = 'k', s = 5, alpha = 0.45,)
plt.plot( bins_x, bins_x, 'r--',)

plt.plot( bins_x, fit_line, ls = '-', color = 'b', alpha = 0.75,)
plt.plot( bins_x, fit_line - sigma, ls = ':', color = 'b', alpha = 0.75,)
plt.plot( bins_x, fit_line + sigma, ls = ':', color = 'b', alpha = 0.75,)
plt.text( 17.5, 18.5, s = '$\\alpha$ = %.3f, $\\beta$ = %.3f' % (b_fit, a_fit) + '\n' + '$\\sigma = %.3f$' % sigma,)

plt.xlim( 19.5, 16.5 )
plt.ylim( 19.5, 15 )
plt.xlabel('i_mag $[R \\leq 10 \\mathrm{k}pc]$')
plt.ylabel('SDSS, i_cmag')
plt.savefig('/home/xkchen/i_mag10_compare.png', dpi = 300)
plt.close()

raise
"""

### == ### mass comparison
"""
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
tot_lgM = np.r_[ tot_lo_lgM, tot_hi_lgM ] - 2 * np.log10( h )


#... estimate the M/L along projected radius, and convert to the cumulative mass within 20 or 10 kpc.
#... (introduce M/L(R) along the projected radius)
# p_dat = pds.read_csv( '/home/xkchen/figs/BCG_aper_M/r-band_BCG-M_bin_aperture_M.csv' )
# p_ra, p_dec, p_z = np.array( p_dat['ra'] ), np.array( p_dat['dec'] ), np.array( p_dat['z'] )
# p_M20 = np.array( p_dat['M_20'] )
# p_M10 = np.array( p_dat['M_10'] )


#... estimate the M/L cumulative mass within 20 or 10 kpc, and convert to the mass.
#... (assume a uniform M/L along the projected radius)
m_dat = pds.read_csv( '/home/xkchen/figs/BCG_aper_M/r-band_BCG-mag_cat.csv' )
p_ra, p_dec, p_z = np.array( m_dat['ra'] ), np.array( m_dat['dec'] ), np.array( m_dat['z'] )

p_M20 = np.array( m_dat['Mstar_20'] )
p_Li = np.array( m_dat['Li_20'] )

x_labels = '$ \\lg M_{\\ast} [R \\leq 20kpc] \; [M_{\\odot}]$'
fig_str = 'lgM20_to_Mtot.png'

# p_M20 = np.array( m_dat['Mstar_10'] )
# p_Li = np.array( m_dat['Li_10'] )

# x_labels = '$ \\lg M_{\\ast} [R \\leq 10kpc] \; [M_{\\odot}]$'
# fig_str = 'lgM10_to_Mtot.png'


id_nul = np.isnan( p_M20 )
p_ra, p_dec, p_z = p_ra[ id_nul == False ], p_dec[ id_nul == False ], p_z[ id_nul == False ]
p_M20 = p_M20[ id_nul == False ]
p_Li = p_Li[ id_nul == False ]


p_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg,)
tot_coord = SkyCoord( ra = tot_ra * U.deg, dec = tot_dec * U.deg,)

idx_0, sep_0, d3d_0 = p_coord.match_to_catalog_sky( tot_coord )
id_lim_0 = sep_0.value < 2.7e-4

mp_ra, mp_dec, mp_z = tot_ra[ idx_0[ id_lim_0 ] ], tot_dec[ idx_0[ id_lim_0 ] ], tot_z[ idx_0[ id_lim_0 ] ]
mp_rich, mp_lgM = tot_rich[ idx_0[ id_lim_0 ] ], tot_lgM[ idx_0[ id_lim_0 ] ]

lim_ra, lim_dec, lim_z = p_ra[ id_lim_0 ], p_dec[ id_lim_0 ], p_z[ id_lim_0 ]
lim_lgM20 = np.log10( p_M20[ id_lim_0 ] )
lim_lgLi = np.log10( p_Li[ id_lim_0 ] )

false_M2L = np.median( lim_lgM20 - lim_lgLi )
false_M = lim_lgLi + false_M2L


p0 = [ 1.3, 10.5 ]
res_lsq = optimize.least_squares( resi_func, x0 = np.array( p0 ), loss = 'cauchy', f_scale = 0.1, args = (lim_lgM20, mp_lgM ),)

a_fit = res_lsq.x[0]
b_fit = res_lsq.x[1]

bins_x = np.linspace(10.5, 12.5, 100)
fit_line = lg_linea_func( bins_x, a_fit, b_fit )

fit_points = lg_linea_func( lim_lgM20, a_fit, b_fit )
Var = np.sum( ( fit_points - mp_lgM )**2 ) / len( mp_lgM )
sigma = np.sqrt( Var )

#. clusters around cross point
# idx0 = mp_lgM < lim_lgM20
# idx1 = mp_lgM < fit_points - sigma
# idx = idx0 & idx1

# put_arr = np.array( [ mp_ra[ idx ], mp_dec[ idx ], mp_z[ idx ] ] ).T
# np.savetxt('/home/xkchen/surf_M_check_cat.txt', put_arr, )

plt.figure()

# plt.plot( lim_lgM20[ idx ], mp_lgM[ idx ], 'g*', markersize = 1, alpha = 0.75,)

plt.scatter( lim_lgM20, mp_lgM, marker = '.', color = 'k', s = 5, alpha = 0.45,)
plt.plot( bins_x, bins_x, 'r:',)

plt.plot( bins_x, fit_line, 'b-', alpha = 0.75,)
plt.plot( bins_x, fit_line + sigma, 'b--', alpha = 0.75,)
plt.plot( bins_x, fit_line - sigma, 'b--', alpha = 0.75,)

plt.text( 11.4, 10.5, s = '$\\alpha$ = %.3f, $\\beta$ = %.3f' % (b_fit, a_fit) + '\n' + '$\\sigma = %.3f$' % sigma,)

plt.xlabel( x_labels )
plt.ylabel('$ \\lg M_{\\ast}^{ \\mathrm{SED} } \; [M_{\\odot}]$')
plt.xlim(10.5, 12)
plt.ylim(10, 12.5)
plt.savefig( '/home/xkchen/%s' % fig_str, dpi = 300)
plt.close()


p0 = [ 1.3, 10.5 ]
res_lsq = optimize.least_squares( resi_func, x0 = np.array( p0 ), loss = 'cauchy', f_scale = 0.1, args = (false_M, mp_lgM ),)

a_fit = res_lsq.x[0]
b_fit = res_lsq.x[1]

bins_x = np.linspace(10.5, 12.5, 100)
fit_line = lg_linea_func( bins_x, a_fit, b_fit )

fit_points = lg_linea_func( false_M, a_fit, b_fit )
Var = np.sum( ( fit_points - mp_lgM )**2 ) / len( mp_lgM )
sigma = np.sqrt( Var )

plt.figure()
plt.scatter( false_M, mp_lgM, marker = '.', color = 'k', s = 5, alpha = 0.45,)
plt.plot( bins_x, bins_x, 'r:',)

plt.plot( bins_x, fit_line, 'b-', alpha = 0.75,)
plt.plot( bins_x, fit_line + sigma, 'b--', alpha = 0.75,)
plt.plot( bins_x, fit_line - sigma, 'b--', alpha = 0.75,)

plt.text( 11.4, 10.5, s = '$\\alpha$ = %.3f, $\\beta$ = %.3f' % (b_fit, a_fit) + '\n' + '$\\sigma = %.3f$' % sigma,)
plt.text( 10.6, 10.1, s = 'Using Median M/L of overall sample to convert $M_{\\ast, \, R \\leq R_{0} }$' )
plt.xlabel( x_labels )
plt.ylabel('$ \\lg M_{\\ast}^{ \\mathrm{SED} } \; [M_{\\odot}]$')
plt.xlim(10.5, 12)
plt.ylim(10, 12.5)
plt.savefig( '/home/xkchen/medi-M2L_%s' % fig_str, dpi = 300)
plt.close()

raise
"""

### === ### compare the binned cluster subsamples properties
fig_name = ['Low $\; M_{\\ast, \, 20}$', 'High $\; M_{\\ast, \, 20}$',
			'Low $\; M_{\\ast, \, 10}$', 'High $\; M_{\\ast, \, 10}$']

#. lgM20 binned
dat_20 = pds.read_csv('/home/xkchen/figs/BCG_aper_M/uniform_M2L_cat/r-band_BCG_aperture-M_cat_params.csv')
ra_20, dec_20, z_20 = np.array( dat_20['ra'] ), np.array( dat_20['dec'] ), np.array( dat_20['z'] )
rich_20, lg_Mbcg_20 = np.array( dat_20['rich'] ), np.array( dat_20['lg_Mbcg'] )

com_coord = SkyCoord( ra = ra_20 * U.deg, dec = dec_20 * U.deg,)

#. stacked cat.
dat_0 = pds.read_csv('/home/xkchen/figs/BCG_aper_M/uniform_M2L_cat/photo-z_match_low-lgM20_gri-common_cluster_cat.csv')
lo_ra_20 = np.array( dat_0['ra'] )
lo_dec_20 = np.array( dat_0['dec'] )
lo_z_20 = np.array( dat_0['z'] )

lo_coord = SkyCoord( ra = lo_ra_20 * U.deg, dec = lo_dec_20 * U.deg,)

idx_0, sep_0, d3d_0 = lo_coord.match_to_catalog_sky( com_coord )
id_lim_0 = sep_0.value < 2.7e-4

lo_rich_20 = rich_20[ idx_0[ id_lim_0 ] ]
lo_Mbcg_20 = lg_Mbcg_20[ idx_0[ id_lim_0 ] ]

#. save the sample properties
keys = ['ra', 'dec', 'z', 'lgM_bcg', 'rich']
values = [ lo_ra_20, lo_dec_20, lo_z_20, lo_Mbcg_20, lo_rich_20 ]
fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )
out_data.to_csv('/home/xkchen/photo-z_match_low-lgM20_gri-common-cat_params.csv')


dat_1 = pds.read_csv('/home/xkchen/figs/BCG_aper_M/uniform_M2L_cat/photo-z_match_hi-lgM20_gri-common_cluster_cat.csv')
hi_ra_20 = np.array( dat_1['ra'] )
hi_dec_20 = np.array( dat_1['dec'] )
hi_z_20 = np.array( dat_1['z'] )

hi_coord = SkyCoord( ra = hi_ra_20 * U.deg, dec = hi_dec_20 * U.deg,)

idx_0, sep_0, d3d_0 = hi_coord.match_to_catalog_sky( com_coord )
id_lim_0 = sep_0.value < 2.7e-4

hi_rich_20 = rich_20[ idx_0[ id_lim_0 ] ]
hi_Mbcg_20 = lg_Mbcg_20[ idx_0[ id_lim_0 ] ]

#. save the sample properties
keys = ['ra', 'dec', 'z', 'lgM_bcg', 'rich']
values = [ hi_ra_20, hi_dec_20, hi_z_20, hi_Mbcg_20, hi_rich_20 ]
fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )
out_data.to_csv('/home/xkchen/photo-z_match_hi-lgM20_gri-common-cat_params.csv')


#. lgM10 binned
dat_10 = pds.read_csv('/home/xkchen/figs/BCG_aper_M/uniform_M2L_cat/r-band_BCG_aperture-M-10-kpc_cat_params.csv')
ra_10, dec_10, z_10 = np.array( dat_10['ra'] ), np.array( dat_10['dec'] ), np.array( dat_10['z'] )
rich_10, lg_Mbcg_10 = np.array( dat_10['rich'] ), np.array( dat_10['lg_Mbcg'] )

com_coord = SkyCoord( ra = ra_10 * U.deg, dec = dec_10 * U.deg,)

#. stacked cat.
dat_0 = pds.read_csv('/home/xkchen/figs/BCG_aper_M/uniform_M2L_cat/photo-z_match_low-lgM10_gri-common_cluster_cat.csv')
lo_ra_10 = np.array( dat_0['ra'] )
lo_dec_10 = np.array( dat_0['dec'] )
lo_z_10 = np.array( dat_0['z'] )

lo_coord = SkyCoord( ra = lo_ra_10 * U.deg, dec = lo_dec_10 * U.deg,)

idx_0, sep_0, d3d_0 = lo_coord.match_to_catalog_sky( com_coord )
id_lim_0 = sep_0.value < 2.7e-4

lo_rich_10 = rich_10[ idx_0[ id_lim_0 ] ]
lo_Mbcg_10 = lg_Mbcg_10[ idx_0[ id_lim_0 ] ]

#. save the sample properties
keys = ['ra', 'dec', 'z', 'lgM_bcg', 'rich']
values = [ lo_ra_10, lo_dec_10, lo_z_10, lo_Mbcg_10, lo_rich_10 ]
fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )
out_data.to_csv('/home/xkchen/photo-z_match_low-lgM10_gri-common-cat_params.csv')


dat_1 = pds.read_csv('/home/xkchen/figs/BCG_aper_M/uniform_M2L_cat/photo-z_match_hi-lgM10_gri-common_cluster_cat.csv')
hi_ra_10 = np.array( dat_1['ra'] )
hi_dec_10 = np.array( dat_1['dec'] )
hi_z_10 = np.array( dat_1['z'] )

hi_coord = SkyCoord( ra = hi_ra_10 * U.deg, dec = hi_dec_10 * U.deg,)

idx_0, sep_0, d3d_0 = hi_coord.match_to_catalog_sky( com_coord )
id_lim_0 = sep_0.value < 2.7e-4

hi_rich_10 = rich_10[ idx_0[ id_lim_0 ] ]
hi_Mbcg_10 = lg_Mbcg_10[ idx_0[ id_lim_0 ] ]

#. save the sample properties
keys = ['ra', 'dec', 'z', 'lgM_bcg', 'rich']
values = [ hi_ra_10, hi_dec_10, hi_z_10, hi_Mbcg_10, hi_rich_10 ]
fill = dict( zip( keys, values ) )
out_data = pds.DataFrame( fill )
out_data.to_csv('/home/xkchen/photo-z_match_hi-lgM10_gri-common-cat_params.csv')


#. binned sample compare
lo_coord_20 = SkyCoord( ra = lo_ra_20 * U.deg, dec = lo_dec_20 * U.deg, )
lo_coord_10 = SkyCoord( ra = lo_ra_10 * U.deg, dec = lo_dec_10 * U.deg, )

idx, d2d, d3d = lo_coord_10.match_to_catalog_sky( lo_coord_20 )
id_lim = d2d.value < 2.7e-4

lo_diffi_Mbcg_20 = lo_Mbcg_20[ idx[ id_lim == False ] ]
lo_diffi_Mbcg_10 = lo_Mbcg_10[ id_lim == False ]

lo_diffi_rich_20 = lo_rich_20[ idx[ id_lim == False ] ]
lo_diffi_rich_10 = lo_rich_10[ id_lim == False ]


hi_coord_20 = SkyCoord( ra = hi_ra_20 * U.deg, dec = hi_dec_20 * U.deg, )
hi_coord_10 = SkyCoord( ra = hi_ra_10 * U.deg, dec = hi_dec_10 * U.deg, )

idx, d2d, d3d = hi_coord_10.match_to_catalog_sky( hi_coord_20 )
id_lim = d2d.value < 2.7e-4

hi_diffi_Mbcg_20 = hi_Mbcg_20[ idx[ id_lim == False ] ]
hi_diffi_Mbcg_10 = hi_Mbcg_10[ id_lim == False ]

hi_diffi_rich_20 = hi_rich_20[ idx[ id_lim == False ] ]
hi_diffi_rich_10 = hi_rich_10[ id_lim == False ]

raise

m_bins = np.linspace( 10, 12.5, 51 )

plt.figure()
plt.hist( lo_Mbcg_20, bins = m_bins, density = False, color = 'r', histtype = 'step', ls = '-', label = fig_name[0],)
plt.axvline( x = np.median( lo_Mbcg_20 ), ls = '-', color = 'r', label = 'median', ymin = 0.85, ymax = 1.0,)
plt.hist( lo_diffi_Mbcg_20, bins = m_bins, density = False, color = 'r', alpha = 0.75, label = fig_name[0] + ', difference',)
plt.axvline( x = np.median( lo_diffi_Mbcg_20 ), ls = ':', color = 'r', ymin = 0.0, ymax = 0.25,)

plt.hist( lo_Mbcg_10, bins = m_bins, density = False, color = 'b', histtype = 'step', ls = '--', label = fig_name[2],)
plt.axvline( x = np.median( lo_Mbcg_10 ), ls = '--', color = 'b', ymin = 0.85, ymax = 1.0,)
plt.hist( lo_diffi_Mbcg_10, bins = m_bins, density = False, color = 'b', alpha = 0.75, label = fig_name[2] + ', difference',)
plt.axvline( x = np.median( lo_diffi_Mbcg_10 ), ls = ':', color = 'b', ymin = 0.0, ymax = 0.25,)

plt.legend( loc = 2,)
plt.xlabel('$ \\lg \, M_{\\ast}^{ \\mathrm{BCG} } \; [M_{\\odot}]$')
plt.ylabel('# of Clusters')
plt.savefig('/home/xkchen/BCG-M_for_lo-aper-M_binned.png', dpi = 300)
plt.close()


plt.figure()
plt.hist( hi_Mbcg_20, bins = m_bins, density = False, color = 'r', histtype = 'step', ls = '-', label = fig_name[1],)
plt.axvline( x = np.median( hi_Mbcg_20 ), ls = '-', color = 'r', label = 'median', ymin = 0.85, ymax = 1.0,)
plt.hist( hi_diffi_Mbcg_20, bins = m_bins, density = False, color = 'r', alpha = 0.75, label = fig_name[1] + ', difference',)
plt.axvline( x = np.median( hi_diffi_Mbcg_20 ), ls = ':', color = 'r', ymin = 0.0, ymax = 0.25,)

plt.hist( hi_Mbcg_10, bins = m_bins, density = False, color = 'b', histtype = 'step', ls = '--', label = fig_name[3],)
plt.axvline( x = np.median( hi_Mbcg_10 ), ls = '--', color = 'b', ymin = 0.85, ymax = 1.0,)
plt.hist( hi_diffi_Mbcg_10, bins = m_bins, density = False, color = 'b', alpha = 0.75, label = fig_name[3] + ', difference',)
plt.axvline( x = np.median( hi_diffi_Mbcg_10 ), ls = ':', color = 'b', ymin = 0.0, ymax = 0.25,)

plt.legend( loc = 2,)
plt.xlabel('$ \\lg \, M_{\\ast}^{ \\mathrm{BCG} } \; [M_{\\odot}]$')
plt.ylabel('# of Clusters')
plt.savefig('/home/xkchen/BCG-M_for_hi-aper-M_binned.png', dpi = 300)
plt.close()


x_bins = np.logspace( 1.3, 2.3, 25)

plt.figure()
plt.hist( lo_rich_20, bins = x_bins, density = False, color = 'r', histtype = 'step', ls = '-', label = fig_name[0],)
plt.axvline( x = np.median( lo_rich_20 ), ls = '-', color = 'r', label = 'median', ymin = 0.85, ymax = 1.0,)
plt.hist( lo_diffi_rich_20, bins = x_bins, density = False, color = 'r', alpha = 0.75, label = fig_name[0] + ', difference',)
plt.axvline( x = np.median( lo_diffi_rich_20 ), ls = ':', color = 'r', ymin = 0.0, ymax = 0.25,)

plt.hist( lo_rich_10, bins = x_bins, density = False, color = 'b', histtype = 'step', ls = '--', label = fig_name[2],)
plt.axvline( x = np.median( lo_rich_10 ), ls = '--', color = 'b', ymin = 0.85, ymax = 1.0,)
plt.hist( lo_diffi_rich_10, bins = x_bins, density = False, color = 'b', alpha = 0.75, label = fig_name[2] + ', difference',)
plt.axvline( x = np.median( lo_diffi_rich_10 ), ls = ':', color = 'b', ymin = 0.0, ymax = 0.25,)

plt.legend( loc = 1,)
plt.xscale('log')
plt.xlabel('$\\lambda$')
plt.ylabel('# of Clusters')
plt.savefig('/home/xkchen/BCG-M_for_lo-aper-M_binned_rich.png', dpi = 300)
plt.close()


plt.figure()
plt.hist( hi_rich_20, bins = x_bins, density = False, color = 'r', histtype = 'step', ls = '-', label = fig_name[1],)
plt.axvline( x = np.median( hi_rich_20 ), ls = '-', color = 'r', label = 'median', ymin = 0.85, ymax = 1.0,)
plt.hist( hi_diffi_rich_20, bins = x_bins, density = False, color = 'r', alpha = 0.75, label = fig_name[1] + ', difference',)
plt.axvline( x = np.median( hi_diffi_rich_20 ), ls = ':', color = 'r', ymin = 0.0, ymax = 0.25,)

plt.hist( hi_rich_10, bins = x_bins, density = False, color = 'b', histtype = 'step', ls = '--', label = fig_name[3],)
plt.axvline( x = np.median( hi_rich_10 ), ls = '--', color = 'b', ymin = 0.85, ymax = 1.0,)
plt.hist( hi_diffi_rich_10, bins = x_bins, density = False, color = 'b', alpha = 0.75, label = fig_name[3] + ', difference',)
plt.axvline( x = np.median( hi_rich_10 ), ls = ':', color = 'b', ymin = 0.0, ymax = 0.25,)

plt.legend( loc = 1,)
plt.xscale('log')
plt.xlabel('$\\lambda$')
plt.ylabel('# of Clusters')
plt.savefig('/home/xkchen/BCG-M_for_hi-aper-M_binned_rich.png', dpi = 300)
plt.close()

