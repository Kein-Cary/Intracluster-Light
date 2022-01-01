import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp
import astropy.units as U
import astropy.constants as C

from scipy import integrate as integ
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize

#.
from BCG_SB_pro_stack import single_img_SB_func
from color_2_mass import SB_to_Lumi_func

from tmp_color_to_mass import gr_ri_band_c2m_func
from fig_out_module import absMag_to_Lumi_func


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


"""
in this section, we need to know the relation between color, mass, and luminosity
( in our case is 'g-r', 'r-i', Li and mass)
"""

### === ###  BCG aperture limited Mass estimate
def cumu_mass_func(rp, surf_mass, N_grid = 100):

	try:
		NR = len(rp)
	except:
		rp = np.array([ rp ])
		NR = len(rp)

	intep_sigma_F = interp.interp1d( rp, surf_mass, kind = 'linear', fill_value = 'extrapolate',)

	cumu_mass = np.zeros( NR, )
	lg_r_min = np.log10( np.min( rp ) / 10 )

	for ii in range( NR ):

		new_rp = np.logspace( lg_r_min, np.log10( rp[ii] ), N_grid)
		new_mass = intep_sigma_F( new_rp )

		cumu_mass[ ii ] = integ.simps( 2 * np.pi * new_rp * new_mass, new_rp)

	return cumu_mass

#. cumulative mass derived from mass profile (assuming M/L along radii) 
def BCG_M_L_color_pros_func( dat_file, lis_ra, lis_dec, lis_z, out_file, param_file, R_bins ):
	"""
	out_files : .csv files
	"""

	N_sam = len( lis_ra )

	for ii in range( N_sam ):

		ra_g, dec_g, z_g = lis_ra[ii], lis_dec[ii], lis_z[ii]

		#. compute surface brightness in order r, g, i band
		tmp_r, tmp_SB = [], []

		for kk in range( 3 ):

			band_str = band[kk]

			tt_rbins, tt_fdens = single_img_SB_func( band_str, z_g, ra_g, dec_g, dat_file, R_bins )

			id_inf = np.isinf( tt_fdens )
			id_nul = tt_fdens <= 0.

			tt_fdens[ id_inf ] = np.nan
			tt_fdens[ id_nul ] = np.nan

			tmp_r.append( tt_rbins )
			tmp_SB.append( tt_fdens )

		#... estimate color and surface mass

		#. i-band luminosity and g-r, r-i color
		r_mag = 22.5 - 2.5 * np.log10( tmp_SB[0] )
		g_mag = 22.5 - 2.5 * np.log10( tmp_SB[1] )
		i_mag = 22.5 - 2.5 * np.log10( tmp_SB[2] )

		i_Lumi = SB_to_Lumi_func( i_mag, z_g, 'i' ) # L_sun / pc^2
		i_Lumi = i_Lumi * 1e6  # L_sun / kpc^2 

		#. surface mass
		p_dat = pds.read_csv( param_file )

		a_fit, b_fit = np.array( p_dat['a'] )[0], np.array( p_dat['b'] )[0]
		c_fit, d_fit = np.array( p_dat['c'] )[0], np.array( p_dat['d'] )[0]

		id_nul = np.isnan( g_mag )
		idv = id_nul == False
		intep_gmag_F = interp.interp1d( tmp_r[1][ idv ], g_mag[ idv ], kind = 'linear', fill_value = 'extrapolate',)


		id_nul = np.isnan( i_mag )
		idv = id_nul == False
		intep_imag_F = interp.interp1d( tmp_r[2][ idv ], i_mag[ idv ], kind = 'linear', fill_value = 'extrapolate',)

		intep_lgLi_F = interp.interp1d( tmp_r[2][ idv ], np.log10( i_Lumi[ idv ] ), kind = 'linear', fill_value = 'extrapolate',)


		id_nul = np.isnan( r_mag )
		idv = id_nul == False

		lim_i_Lumi = 10**( intep_lgLi_F( tmp_r[0][ idv ] ) )


		g2r_arr = intep_gmag_F( tmp_r[0][ idv ] ) - r_mag[ idv ]
		r2i_arr = r_mag[ idv ] - intep_imag_F( tmp_r[0][ idv ] )

		fit_arr = [ a_fit, b_fit, c_fit, d_fit ]
		tt_M = gr_ri_band_c2m_func( g2r_arr, r2i_arr, lim_i_Lumi, fit_arr ) # M_sun / kpc^2

		#. save profiles
		keys = [ 'R_kpc', 'g-r', 'r-i', 'i_Lumi', 'surf_M']
		values = [ tmp_r[0][ idv ], g2r_arr, r2i_arr, lim_i_Lumi, tt_M ]
		fill = dict( zip( keys, values ) )
		data = pds.DataFrame( fill )
		data.to_csv( out_file % (ra_g, dec_g, z_g),)

	return

#. cumulative luminosity derived from magnitude within given radius
def BCG_mag_comu_func( dat_file, lis_ra, lis_dec, lis_z, out_file, R_bins, param_file ):
	"""
	out_files : .csv files
	"""
	N_sam = len( lis_ra )

	samp_rmag, samp_gmag, samp_imag = np.zeros( N_sam,), np.zeros( N_sam,), np.zeros( N_sam,)
	imag_20, imag_10 = np.zeros( N_sam,), np.zeros( N_sam,)
	g2r_20, g2r_10 = np.zeros( N_sam,), np.zeros( N_sam,)
	r2i_20, r2i_10 = np.zeros( N_sam,), np.zeros( N_sam,)
	Mstar_20, Mstar_10 = np.zeros( N_sam,), np.zeros( N_sam,)
	Li_20, Li_10 = np.zeros( N_sam,), np.zeros( N_sam,)

	for ii in range( N_sam ):

		ra_g, dec_g, z_g = lis_ra[ii], lis_dec[ii], lis_z[ii]

		Da_g = Test_model.angular_diameter_distance( z_g ).value # unit Mpc
		Dl_g = Test_model.luminosity_distance( z_g ).value # unit Mpc

		tmp_mag, tmp_mag_20, tmp_mag_10 = [], [], []

		for kk in range( 3 ):

			band_str = band[kk]

			tt_rbins, tt_fdens = single_img_SB_func( band_str, z_g, ra_g, dec_g, dat_file, R_bins )
			id_inf = np.isinf( tt_fdens )
			id_nul = tt_fdens <= 0.
			id_lim = id_inf & id_nul

			tt_rbins = tt_rbins[ id_lim == False ]
			tt_fdens = tt_fdens[ id_lim == False ]
			angl_r = ( tt_rbins / 1e3 ) * rad2asec / Da_g

			#. cumulative flux
			cumu_F = cumu_mass_func( angl_r, tt_fdens )

			#. integral luminosity
			tt_mag = 22.5 - 2.5 * np.log10( cumu_F[-1] )

			intep_cumu_F_f = interp.interp1d( tt_rbins, cumu_F, kind = 'linear', fill_value = 'extrapolate',)

			tt_mag_20 = 22.5 - 2.5 * np.log10( intep_cumu_F_f( 20 ) )
			tt_mag_10 = 22.5 - 2.5 * np.log10( intep_cumu_F_f( 10 ) )

			tmp_mag.append( tt_mag )
			tmp_mag_20.append( tt_mag_20 )
			tmp_mag_10.append( tt_mag_10 )

		#. surface mass
		p_dat = pds.read_csv( param_file )
		a_fit, b_fit = np.array( p_dat['a'] )[0], np.array( p_dat['b'] )[0]
		c_fit, d_fit = np.array( p_dat['c'] )[0], np.array( p_dat['d'] )[0]

		g2r_20[ii] = tmp_mag_20[1] - tmp_mag_20[0]
		r2i_20[ii] = tmp_mag_20[0] - tmp_mag_20[2]

		g2r_10[ii] = tmp_mag_10[1] - tmp_mag_10[0]
		r2i_10[ii] = tmp_mag_10[0] - tmp_mag_10[2]

		imag_20[ii] = tmp_mag_20[2]
		imag_10[ii] = tmp_mag_10[2]

		#. aperture mass estimate
		fit_arr = [ a_fit, b_fit, c_fit, d_fit ]

		_dd_gr = tmp_mag_20[1] - tmp_mag_20[0]
		_dd_ri = tmp_mag_20[0] - tmp_mag_20[2]

		abs_imag_20 = tmp_mag_20[2] - 5 * np.log10( Dl_g * 1e6 ) + 5

		_dd_Li = absMag_to_Lumi_func( abs_imag_20, 'i' ) # L_sun
		tt_M20 = gr_ri_band_c2m_func( _dd_gr, _dd_ri, _dd_Li, fit_arr ) # M_sun

		Li_20[ii] = _dd_Li

		_dd_gr = tmp_mag_10[1] - tmp_mag_10[0]
		_dd_ri = tmp_mag_10[0] - tmp_mag_10[2]

		abs_imag_10 = tmp_mag_10[2] - 5 * np.log10( Dl_g * 1e6 ) + 5

		_dd_Li = absMag_to_Lumi_func( abs_imag_10, 'i' ) # L_sun
		tt_M10 = gr_ri_band_c2m_func( _dd_gr, _dd_ri, _dd_Li, fit_arr ) # M_sun

		Li_10[ii] = _dd_Li

		Mstar_20[ii] = tt_M20
		Mstar_10[ii] = tt_M10

		samp_rmag[ii] = tmp_mag[0]
		samp_gmag[ii] = tmp_mag[1]
		samp_imag[ii] = tmp_mag[2]

	keys = ['ra', 'dec', 'z', 'rmag', 'gmag', 'imag', 'imag_20', 'imag_10',
			'g-r_20', 'g-r_10', 'r-i_20', 'r-i_10', 'Mstar_20', 'Mstar_10', 'Li_20', 'Li_10']
	values = [ lis_ra, lis_dec, lis_z, samp_rmag, samp_gmag, samp_imag, imag_20, imag_10, 
			g2r_20, g2r_10, r2i_20, r2i_10, Mstar_20, Mstar_10, Li_20, Li_10 ]

	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( out_file )

	return

