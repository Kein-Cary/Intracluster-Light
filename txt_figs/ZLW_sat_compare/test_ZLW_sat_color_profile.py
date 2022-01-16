import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal as signal

import time
import h5py
import numpy as np
import astropy.io.fits as fits

import pandas as pds
from io import StringIO

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

from fig_out_module import arr_jack_func
from fig_out_module import absMag_to_Lumi_func

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

### constant
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']

def P_mem_color( stacked_cat_file, N_sets, clust_mem_file, out_files, N_r_bins, band_str):

	d_cat = pds.read_csv( stacked_cat_file )
	ra, dec, z = np.array( d_cat['ra']), np.array( d_cat['dec']), np.array( d_cat['z'])
	N_radii = len( N_r_bins )

	## also divid sub-samples
	zN = len( ra )
	id_arr = np.arange(0, zN, 1)
	id_group = id_arr % N_sets

	lis_ra, lis_dec, lis_z = [], [], []

	## sub-sample
	for nn in range( N_sets ):

		id_xbin = np.where( id_group == nn )[0]

		lis_ra.append( ra[ id_xbin ] )
		lis_dec.append( dec[ id_xbin ] )
		lis_z.append( z[ id_xbin ] )

	## jackknife sub-sample
	for nn in range( N_sets ):

		id_arry = np.linspace( 0, N_sets - 1, N_sets )
		id_arry = id_arry.astype( int )
		jack_id = list( id_arry )
		jack_id.remove( jack_id[nn] )
		jack_id = np.array( jack_id )

		set_ra, set_dec, set_z = np.array([]), np.array([]), np.array([])

		for oo in ( jack_id ):
			set_ra = np.r_[ set_ra, lis_ra[oo] ]
			set_dec = np.r_[ set_dec, lis_dec[oo] ]
			set_z = np.r_[ set_z, lis_z[oo] ]

		ncs = len( set_z )

		bar_g2r = np.zeros( N_radii, dtype = np.float32)
		bar_g2r_err = np.zeros( N_radii, dtype = np.float32)

		bar_r2i = np.zeros( N_radii, dtype = np.float32)
		bar_r2i_err = np.zeros( N_radii, dtype = np.float32)

		N_galaxy = np.zeros( N_radii, dtype = np.float32)

		R_vals = np.zeros( N_radii, dtype = np.float32)

		for pp in range( N_radii - 1 ):

			tmp_m_g2r, tmp_m_r2i = np.array([]), np.array([])
			tmp_g2r_err, tmp_r2i_err = np.array([]), np.array([])
			tmp_Pmem = np.array([])
			tmp_radius = np.array([])

			for ii in range( ncs ):

				ra_g, dec_g, z_g = set_ra[ii], set_dec[ii], set_z[ii]

				sub_dat = pds.read_csv( clust_mem_file % ( band_str, ra_g, dec_g, z_g),)
				sub_cen_R = np.array(sub_dat['centric_R(Mpc/h)'])
				sub_Pmem = np.array(sub_dat['P_member'])

				sub_r_mag = np.array(sub_dat['r_mags'])
				sub_g_mag = np.array(sub_dat['g_mags'])
				sub_i_mag = np.array(sub_dat['i_mags'])

				sub_r_mag_err = np.array(sub_dat['r_mag_err'])
				sub_g_mag_err = np.array(sub_dat['g_mag_err'])
				sub_i_mag_err = np.array(sub_dat['i_mag_err'])

				# recalculate magnitude at z_ref
				Dl2 = Test_model.luminosity_distance( z_ref ).value
				Dl1 = Test_model.luminosity_distance( z_g ).value

				cc_sub_r_mag = sub_r_mag + 5 * np.log10( Dl2 / Dl1 )
				cc_sub_g_mag = sub_g_mag + 5 * np.log10( Dl2 / Dl1 )
				cc_sub_i_mag = sub_i_mag + 5 * np.log10( Dl2 / Dl1 )

				id_lim = ( sub_cen_R >= N_r_bins[pp] ) & ( sub_cen_R <= N_r_bins[pp + 1] )

				if np.sum(id_lim) > 0:

					dpt_g2r = cc_sub_g_mag[id_lim] - cc_sub_r_mag[id_lim]
					dpt_g2r_err = np.sqrt( sub_r_mag_err**2 + sub_g_mag_err**2 )

					dpt_r2i = cc_sub_r_mag[id_lim] - cc_sub_i_mag[id_lim]
					dpt_r2i_err = np.sqrt( sub_r_mag_err**2 + sub_i_mag_err**2 )

					dpt_P_mem = sub_Pmem[id_lim]

					dpt_radius = sub_cen_R[id_lim]

					tmp_m_g2r = np.r_[ tmp_m_g2r, dpt_g2r ]
					tmp_m_r2i = np.r_[ tmp_m_r2i, dpt_r2i ]
					tmp_g2r_err = np.r_[ tmp_g2r_err, dpt_g2r_err ]
					tmp_r2i_err = np.r_[ tmp_r2i_err, dpt_r2i_err ]
					tmp_Pmem = np.r_[ tmp_Pmem, dpt_P_mem ]
					tmp_radius = np.r_[ tmp_radius, dpt_radius ]

				else:
					dpt_g2r = 0.
					dpt_g2r_err = 0.
					dpt_r2i = 0.
					dpt_r2i_err = 0.
					dpt_P_mem = 0.
					dpt_radius = 0.

					tmp_m_g2r = np.r_[ tmp_m_g2r, dpt_g2r ]
					tmp_m_r2i = np.r_[ tmp_m_r2i, dpt_r2i ]
					tmp_g2r_err = np.r_[ tmp_g2r_err, dpt_g2r_err ]
					tmp_r2i_err = np.r_[ tmp_r2i_err, dpt_r2i_err ]
					tmp_Pmem = np.r_[ tmp_Pmem, dpt_P_mem ]
					tmp_radius = np.r_[ tmp_radius, dpt_radius ]

			mean_g2r = np.sum( tmp_m_g2r * tmp_Pmem ) / np.sum( tmp_Pmem )
			mean_r2i = np.sum( tmp_m_r2i * tmp_Pmem ) / np.sum( tmp_Pmem )

			## use the definition of std
			mean_g2r_err = np.sum( tmp_Pmem * (mean_g2r - tmp_m_g2r)**2 ) / np.sum( tmp_Pmem )
			mean_r2i_err = np.sum( tmp_Pmem * (mean_r2i - tmp_m_r2i)**2 ) / np.sum( tmp_Pmem )

			n_galax = len( tmp_m_g2r )

			mean_radius = np.sum( tmp_Pmem * tmp_radius ) / np.sum( tmp_Pmem )

			bar_g2r[ pp ] = mean_g2r
			bar_g2r_err[ pp ] = mean_g2r_err

			bar_r2i[ pp ] = mean_r2i
			bar_r2i_err[ pp ] = mean_r2i_err

			N_galaxy[ pp ] = n_galax

			R_vals[ pp ] = mean_radius

		## save
		keys = [ 'centric_R(Mpc/h)', 'bar_g2r', 'bar_r2i', 'bar_g2r_err', 'bar_r2i_err', 'n_galaxy']
		values = [ R_vals, bar_g2r, bar_r2i, bar_g2r_err, bar_r2i_err, N_galaxy ]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_files % nn )

	return

def rep_P_mem_color( sub_ra, sub_dec, sub_z, clust_mem_file, out_files, N_r_bins, band_str, z_cut = False,):
	"""
	z_cut : selection in line-of-sight
	"""
	N_radii = len( N_r_bins )
	ncs = len( sub_z )

	bar_g2r = np.zeros( N_radii, dtype = np.float32)
	bar_g2r_err = np.zeros( N_radii, dtype = np.float32)

	bar_r2i = np.zeros( N_radii, dtype = np.float32)
	bar_r2i_err = np.zeros( N_radii, dtype = np.float32)

	bar_g2i = np.zeros( N_radii, dtype = np.float32)
	bar_g2i_err = np.zeros( N_radii, dtype = np.float32)

	dered_bar_g2r = np.zeros( N_radii, dtype = np.float32)
	dered_bar_r2i = np.zeros( N_radii, dtype = np.float32)
	dered_bar_g2i = np.zeros( N_radii, dtype = np.float32)

	dered_bar_g2r_err = np.zeros( N_radii, dtype = np.float32)
	dered_bar_r2i_err = np.zeros( N_radii, dtype = np.float32)
	dered_bar_g2i_err = np.zeros( N_radii, dtype = np.float32)

	#. Luminosity weighted color
	Lwt_bar_g2r = np.zeros( N_radii, dtype = np.float32)
	Lwt_bar_g2i = np.zeros( N_radii, dtype = np.float32)
	Lwt_bar_r2i = np.zeros( N_radii, dtype = np.float32)

	Lwt_bar_g2r_err = np.zeros( N_radii, dtype = np.float32)
	Lwt_bar_g2i_err = np.zeros( N_radii, dtype = np.float32)
	Lwt_bar_r2i_err = np.zeros( N_radii, dtype = np.float32)

	Lwt_dered_bar_g2r = np.zeros( N_radii, dtype = np.float32)
	Lwt_dered_bar_g2i = np.zeros( N_radii, dtype = np.float32)
	Lwt_dered_bar_r2i = np.zeros( N_radii, dtype = np.float32)

	Lwt_dered_bar_g2r_err = np.zeros( N_radii, dtype = np.float32)
	Lwt_dered_bar_g2i_err = np.zeros( N_radii, dtype = np.float32)
	Lwt_dered_bar_r2i_err = np.zeros( N_radii, dtype = np.float32)

	N_galaxy = np.zeros( N_radii, dtype = np.float32)
	R_vals = np.zeros( N_radii, dtype = np.float32)

	for pp in range( N_radii - 1 ):

		tmp_m_g2r, tmp_m_r2i, tmp_m_g2i = np.array([]), np.array([]), np.array([])
		dered_tmp_m_g2r, dered_tmp_m_r2i, dered_tmp_m_g2i = np.array([]), np.array([]), np.array([])

		#. Luminosity record
		tmp_Lumi_r = np.array([])

		tmp_Pmem = np.array([])
		tmp_radius = np.array([])

		for ii in range( ncs ):

			ra_g, dec_g, z_g = sub_ra[ii], sub_dec[ii], sub_z[ii]

			sub_dat = pds.read_csv( clust_mem_file % ( band_str, ra_g, dec_g, z_g),)
			sub_cen_R = np.array( sub_dat['centric_R(Mpc/h)'])
			sub_Pmem = np.array( sub_dat['P_member'])

			sub_r_mag = np.array( sub_dat['r_mags'])
			sub_g_mag = np.array( sub_dat['g_mags'])
			sub_i_mag = np.array( sub_dat['i_mags'])

			dered_sub_r_mag = np.array( sub_dat['dered_r_mags'])
			dered_sub_g_mag = np.array( sub_dat['dered_g_mags'])
			dered_sub_i_mag = np.array( sub_dat['dered_i_mags'])

			sub_Lumi_r = np.array( sub_dat['L_r'] )

			#. selection with satellite redshift
			sub_sat_z = np.array( sub_dat['z'] ) 

			##. rule out obs. with nonreasonable values
			id_nul = sub_r_mag < 0
			sub_r_mag[ id_nul ] = np.nan
			sub_Pmem[ id_nul ] = np.nan

			id_nul = sub_g_mag < 0
			sub_g_mag[ id_nul ] = np.nan
			sub_Pmem[ id_nul ] = np.nan

			id_nul = sub_i_mag < 0
			sub_i_mag[ id_nul ] = np.nan
			sub_Pmem[ id_nul ] = np.nan

			id_nul = dered_sub_r_mag < 0
			dered_sub_r_mag[ id_nul ] = np.nan
			sub_Pmem[ id_nul ] = np.nan

			id_nul = dered_sub_g_mag < 0
			dered_sub_g_mag[ id_nul ] = np.nan
			sub_Pmem[ id_nul ] = np.nan

			id_nul = dered_sub_i_mag < 0
			dered_sub_i_mag[ id_nul ] = np.nan
			sub_Pmem[ id_nul ] = np.nan

			if z_cut == False:
				id_lim = ( sub_cen_R >= N_r_bins[pp] ) & ( sub_cen_R <= N_r_bins[pp + 1] )

			else:
				id_lim_0 = ( sub_cen_R >= N_r_bins[pp] ) & ( sub_cen_R <= N_r_bins[pp + 1] )

				abs_dev_z = np.abs(sub_sat_z - z_g)
				id_lim_1 = abs_dev_z <= 0.01 * ( 1 + z_g )

				id_lim = id_lim_0 & id_lim_1

			if np.sum(id_lim) > 0:

				dpt_g2r = sub_g_mag[id_lim] - sub_r_mag[id_lim]
				dpt_r2i = sub_r_mag[id_lim] - sub_i_mag[id_lim]
				dpt_g2i = sub_g_mag[id_lim] - sub_i_mag[id_lim]

				dpt_P_mem = sub_Pmem[id_lim]
				dpt_radius = sub_cen_R[id_lim]

				dered_dpt_g2r = dered_sub_g_mag[id_lim] - dered_sub_r_mag[id_lim]
				dered_dpt_r2i = dered_sub_r_mag[id_lim] - dered_sub_i_mag[id_lim]
				dered_dpt_g2i = dered_sub_g_mag[id_lim] - dered_sub_i_mag[id_lim]

				dpt_Lumi_r = sub_Lumi_r[id_lim]

			else:
				dpt_g2r = 0.
				dpt_r2i = 0.
				dpt_g2i = 0.

				dpt_P_mem = 0.
				dpt_radius = 0.
				dpt_Lumi_r = 0.

				dered_dpt_g2r = 0.
				dered_dpt_r2i = 0.
				dered_dpt_g2i = 0.

			tmp_m_g2r = np.r_[ tmp_m_g2r, dpt_g2r ]
			tmp_m_r2i = np.r_[ tmp_m_r2i, dpt_r2i ]
			tmp_m_g2i = np.r_[ tmp_m_g2i, dpt_g2i ]

			tmp_Pmem = np.r_[ tmp_Pmem, dpt_P_mem ]
			tmp_radius = np.r_[ tmp_radius, dpt_radius ]
			tmp_Lumi_r = np.r_[ tmp_Lumi_r, dpt_Lumi_r ]

			dered_tmp_m_g2r = np.r_[ dered_tmp_m_g2r, dered_dpt_g2r ]
			dered_tmp_m_r2i = np.r_[ dered_tmp_m_r2i, dered_dpt_r2i ]
			dered_tmp_m_g2i = np.r_[ dered_tmp_m_g2i, dered_dpt_g2i ]

		#. averaged color
		mean_g2r = np.nansum( tmp_m_g2r * tmp_Pmem ) / np.nansum( tmp_Pmem )
		mean_r2i = np.nansum( tmp_m_r2i * tmp_Pmem ) / np.nansum( tmp_Pmem )
		mean_g2i = np.nansum( tmp_m_g2i * tmp_Pmem ) / np.nansum( tmp_Pmem )

		dered_mean_g2r = np.nansum( dered_tmp_m_g2r * tmp_Pmem ) / np.nansum( tmp_Pmem )
		dered_mean_r2i = np.nansum( dered_tmp_m_r2i * tmp_Pmem ) / np.nansum( tmp_Pmem )
		dered_mean_g2i = np.nansum( dered_tmp_m_g2i * tmp_Pmem ) / np.nansum( tmp_Pmem )

		#. use the definition of std for error estimate
		mean_g2r_err = np.nansum( tmp_Pmem * (mean_g2r - tmp_m_g2r)**2 ) / np.nansum( tmp_Pmem )
		mean_r2i_err = np.nansum( tmp_Pmem * (mean_r2i - tmp_m_r2i)**2 ) / np.nansum( tmp_Pmem )
		mean_g2i_err = np.nansum( tmp_Pmem * (mean_g2i - tmp_m_g2i)**2 ) / np.nansum( tmp_Pmem )

		dered_mean_g2r_err = np.nansum( tmp_Pmem * (dered_mean_g2r - dered_tmp_m_g2r)**2 ) / np.nansum( tmp_Pmem )
		dered_mean_r2i_err = np.nansum( tmp_Pmem * (dered_mean_r2i - dered_tmp_m_r2i)**2 ) / np.nansum( tmp_Pmem )
		dered_mean_g2i_err = np.nansum( tmp_Pmem * (dered_mean_g2i - dered_tmp_m_g2i)**2 ) / np.nansum( tmp_Pmem )

		#. Luminosity weight color and error
		id_nul = tmp_Lumi_r < 1.
		id_vx = id_nul == False

		Lwt_aveg_g2r = np.nansum( tmp_m_g2r[ id_vx ] * tmp_Lumi_r[ id_vx ] ) / np.nansum( tmp_Lumi_r[ id_vx ] )
		Lwt_aveg_g2i = np.nansum( tmp_m_g2i[ id_vx ] * tmp_Lumi_r[ id_vx ] ) / np.nansum( tmp_Lumi_r[ id_vx ] )
		Lwt_aveg_r2i = np.nansum( tmp_m_r2i[ id_vx ] * tmp_Lumi_r[ id_vx ] ) / np.nansum( tmp_Lumi_r[ id_vx ] )

		Lwt_aveg_dered_g2r = np.nansum( dered_tmp_m_g2r[ id_vx ] * tmp_Lumi_r[ id_vx ] 
										) / np.nansum( tmp_Lumi_r[ id_vx ] )

		Lwt_aveg_dered_g2i = np.nansum( dered_tmp_m_g2i[ id_vx ] * tmp_Lumi_r[ id_vx ] 
										) / np.nansum( tmp_Lumi_r[ id_vx ] )

		Lwt_aveg_dered_r2i = np.nansum( dered_tmp_m_r2i[ id_vx ] * tmp_Lumi_r[ id_vx ] 
										) / np.nansum( tmp_Lumi_r[ id_vx ] )

		Lwt_aveg_g2r_err = np.nansum( (Lwt_aveg_g2r - tmp_m_g2r[ id_vx ] )**2 * tmp_Lumi_r[ id_vx ] 
										) / np.nansum( tmp_Lumi_r[ id_vx ] )

		Lwt_aveg_g2i_err = np.nansum( (Lwt_aveg_g2i - tmp_m_g2i[ id_vx ] )**2 * tmp_Lumi_r[ id_vx ] 
										) / np.nansum( tmp_Lumi_r[ id_vx ] )

		Lwt_aveg_r2i_err = np.nansum( (Lwt_aveg_r2i - tmp_m_r2i[ id_vx ] )**2 * tmp_Lumi_r[ id_vx ] 
										) / np.nansum( tmp_Lumi_r[ id_vx ] )

		Lwt_aveg_dered_g2r_err = np.nansum( (Lwt_aveg_dered_g2r - dered_tmp_m_g2r[ id_vx ] )**2 * tmp_Lumi_r[ id_vx ] 
											) / np.nansum( tmp_Lumi_r[ id_vx ] )

		Lwt_aveg_dered_g2i_err = np.nansum( (Lwt_aveg_dered_g2i - dered_tmp_m_g2i[ id_vx ] )**2 * tmp_Lumi_r[ id_vx ] 
											) / np.nansum( tmp_Lumi_r[ id_vx ] )

		Lwt_aveg_dered_r2i_err = np.nansum( (Lwt_aveg_dered_r2i - dered_tmp_m_r2i[ id_vx ] )**2 * tmp_Lumi_r[ id_vx ] 
											) / np.nansum( tmp_Lumi_r[ id_vx ] )

		# Lwt_aveg_g2r = np.nansum( tmp_m_g2r[ id_vx ] * np.log10( tmp_Lumi_r[ id_vx ] ) ) / np.nansum( np.log10( tmp_Lumi_r[ id_vx ] ) )
		# Lwt_aveg_g2i = np.nansum( tmp_m_g2i[ id_vx ] * np.log10( tmp_Lumi_r[ id_vx ] ) ) / np.nansum( np.log10( tmp_Lumi_r[ id_vx ] ) )
		# Lwt_aveg_r2i = np.nansum( tmp_m_r2i[ id_vx ] * np.log10( tmp_Lumi_r[ id_vx ] ) ) / np.nansum( np.log10( tmp_Lumi_r[ id_vx ] ) )

		# Lwt_aveg_dered_g2r = np.nansum( dered_tmp_m_g2r[ id_vx ] * np.log10( tmp_Lumi_r[ id_vx ] ) 
		# 								) / np.nansum( np.log10( tmp_Lumi_r[ id_vx ] ) )

		# Lwt_aveg_dered_g2i = np.nansum( dered_tmp_m_g2i[ id_vx ] * np.log10( tmp_Lumi_r[ id_vx ] ) 
		# 								) / np.nansum( np.log10( tmp_Lumi_r[ id_vx ] ) )

		# Lwt_aveg_dered_r2i = np.nansum( dered_tmp_m_r2i[ id_vx ] * np.log10( tmp_Lumi_r[ id_vx ] ) 
		# 								) / np.nansum( np.log10( tmp_Lumi_r[ id_vx ] ) )

		# Lwt_aveg_g2r_err = np.nansum( (Lwt_aveg_g2r - tmp_m_g2r[ id_vx ] )**2 * np.log10( tmp_Lumi_r[ id_vx ] ) 
		# 								) / np.nansum( np.log10( tmp_Lumi_r[ id_vx ] ) )

		# Lwt_aveg_g2i_err = np.nansum( (Lwt_aveg_g2i - tmp_m_g2i[ id_vx ] )**2 * np.log10( tmp_Lumi_r[ id_vx ] ) 
		# 								) / np.nansum( np.log10( tmp_Lumi_r[ id_vx ] ) )

		# Lwt_aveg_r2i_err = np.nansum( (Lwt_aveg_r2i - tmp_m_r2i[ id_vx ] )**2 * np.log10( tmp_Lumi_r[ id_vx ] ) 
		# 								) / np.nansum( np.log10( tmp_Lumi_r[ id_vx ] ) )

		# Lwt_aveg_dered_g2r_err = np.nansum( (Lwt_aveg_dered_g2r - dered_tmp_m_g2r[ id_vx ] )**2 * np.log10( tmp_Lumi_r[ id_vx ] ) 
		# 									) / np.nansum( np.log10( tmp_Lumi_r[ id_vx ] ) )

		# Lwt_aveg_dered_g2i_err = np.nansum( (Lwt_aveg_dered_g2i - dered_tmp_m_g2i[ id_vx ] )**2 * np.log10( tmp_Lumi_r[ id_vx ] ) 
		# 									) / np.nansum( np.log10( tmp_Lumi_r[ id_vx ] ) )

		# Lwt_aveg_dered_r2i_err = np.nansum( (Lwt_aveg_dered_r2i - dered_tmp_m_r2i[ id_vx ] )**2 * np.log10( tmp_Lumi_r[ id_vx ] ) 
		# 									) / np.nansum( np.log10( tmp_Lumi_r[ id_vx ] ) )

		n_galax = np.sum( id_vx )
		mean_radius = np.nansum( tmp_Pmem * tmp_radius ) / np.nansum( tmp_Pmem )

		bar_g2r[ pp ] = mean_g2r
		bar_r2i[ pp ] = mean_r2i
		bar_g2i[ pp ] = mean_g2i

		bar_g2r_err[ pp ] = mean_g2r_err
		bar_r2i_err[ pp ] = mean_r2i_err
		bar_g2i_err[ pp ] = mean_g2i_err

		dered_bar_g2r[ pp ] = dered_mean_g2r
		dered_bar_r2i[ pp ] = dered_mean_r2i
		dered_bar_g2i[ pp ] = dered_mean_g2i

		dered_bar_g2r_err[ pp ] = dered_mean_g2r_err
		dered_bar_r2i_err[ pp ] = dered_mean_r2i_err
		dered_bar_g2i_err[ pp ] = dered_mean_g2i_err


		Lwt_bar_g2r[ pp ] = Lwt_aveg_g2r
		Lwt_bar_g2i[ pp ] = Lwt_aveg_g2i
		Lwt_bar_r2i[ pp ] = Lwt_aveg_r2i

		Lwt_bar_g2r_err[ pp ] = Lwt_aveg_g2r_err
		Lwt_bar_g2i_err[ pp ] = Lwt_aveg_g2i_err
		Lwt_bar_r2i_err[ pp ] = Lwt_aveg_r2i_err

		Lwt_dered_bar_g2r[ pp ] = Lwt_aveg_dered_g2r
		Lwt_dered_bar_g2i[ pp ] = Lwt_aveg_dered_g2i
		Lwt_dered_bar_r2i[ pp ] = Lwt_aveg_dered_r2i

		Lwt_dered_bar_g2r_err[ pp ] = Lwt_aveg_dered_g2r_err
		Lwt_dered_bar_g2i_err[ pp ] = Lwt_aveg_dered_g2i_err
		Lwt_dered_bar_r2i_err[ pp ] = Lwt_aveg_dered_r2i_err

		N_galaxy[ pp ] = n_galax
		R_vals[ pp ] = mean_radius

	## save
	keys = [ 'centric_R(Mpc/h)', 'bar_g2r', 'bar_r2i', 'bar_g2i', 'bar_g2r_err', 'bar_r2i_err', 'bar_g2i_err', 
			'dered_bar_g2r', 'dered_bar_r2i', 'dered_bar_g2i', 'dered_bar_g2r_err', 'dered_bar_r2i_err', 'dered_bar_g2i_err', 
			'n_galaxy',
			'Lwt_g2r', 'Lwt_r2i', 'Lwt_g2i', 'Lwt_g2r_err', 'Lwt_r2i_err', 'Lwt_g2i_err', 
			'Lwt_dered_g2r', 'Lwt_dered_r2i', 'Lwt_dered_g2i', 'Lwt_dered_g2r_err', 'Lwt_dered_r2i_err', 'Lwt_dered_g2i_err']

	values = [ R_vals, bar_g2r, bar_r2i, bar_g2i, bar_g2r_err, bar_r2i_err, bar_g2i_err, 
				dered_bar_g2r, dered_bar_r2i, dered_bar_g2i, dered_bar_g2r_err, dered_bar_r2i_err, dered_bar_g2i_err, 
				N_galaxy, 
				Lwt_bar_g2r, Lwt_bar_r2i, Lwt_bar_g2i, Lwt_bar_g2r_err, Lwt_bar_r2i_err, Lwt_bar_g2i_err, 
				Lwt_dered_bar_g2r, Lwt_dered_bar_r2i, Lwt_dered_bar_g2i, 
				Lwt_dered_bar_g2r_err, Lwt_dered_bar_r2i_err, Lwt_dered_bar_g2i_err ]

	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_files )

	return

###...### ZLWen catalog color
home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'
out_path = '/home/xkchen/project/tmp/'

# cat_lis = [ 'low-age', 'hi-age' ]
# cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
cat_lis = [ 'low-rich', 'hi-rich' ]

band_info = band[ 0 ]

# samp_dex = np.int( rank + 1 ) # 1 -- low-t_age; 2 -- high-t_age

# #. cluster catalog
# clus_dat = pds.read_csv( home + 'ZLWen_cat/clust_sql_match_cat.csv' )
# orin_dex = np.array( clus_dat['clust_id'] )
# clus_ra, clus_dec, clus_z = np.array( clus_dat['ra'] ), np.array( clus_dat['dec'] ), np.array( clus_dat['clus_z'] )
# clus_R500, clus_rich = np.array( clus_dat['R500c'] ), np.array( clus_dat['rich'] )
# clus_Ng, clus_zf, clus_div = np.array( clus_dat['N500c'] ), np.array( clus_dat['z_flag'] ), np.array( clus_dat['samp_id'] )

# bcg_ra, bcg_dec = np.array( clus_dat['bcg_ra'] ), np.array( clus_dat['bcg_dec'] )
# bcg_r_mag, bcg_g_mag, bcg_i_mag = np.array( clus_dat['bcg_r_mag'] ), np.array( clus_dat['bcg_g_mag'] ), np.array( clus_dat['bcg_i_mag'] )
# bcg_r_cmag, bcg_g_cmag, bcg_i_cmag = np.array( clus_dat['bcg_r_cmag'] ), np.array( clus_dat['bcg_g_cmag'] ), np.array( clus_dat['bcg_i_cmag'] )

# #. tmp_table for measuring satellite color
# div_dex = clus_div == samp_dex
# tmp_ra, tmp_dec, tmp_z = clus_ra[ div_dex ], clus_dec[ div_dex ], clus_z[ div_dex ]

# keys = [ 'ra', 'dec', 'z' ]
# values = [ tmp_ra, tmp_dec, tmp_z ]
# fill = dict(zip( keys, values) )
# out_data = pds.DataFrame( fill )
# out_data.to_csv( home + 'ZLWen_cat/ZLWen_%s_match_cat.csv' % cat_lis[ rank ] )


#. average satellite color
N_samples = 30

# N_bins = 55
# R_bins = np.logspace(0, np.log10(2e3), N_bins) / 1e3 # Mpc

R_bins_0 = np.logspace( 1, 2, 5)
R_bins_1 = np.logspace( 2, 3.302, 20)
R_bins = np.r_[ R_bins_0[:-1], R_bins_1 ] / 1e3
N_bins = len( R_bins )

for ll in range( 2 ):

	stacked_cat_file = home + 'ZLWen_cat/ZLWen_%s_match_cat.csv' % cat_lis[ ll ]
	clust_mem_file = home + 'ZLWen_cat/mem_match/ZLW_%s-band_ra%.3f_dec%.3f_z%.3f_members_mag.csv'

	print( 'stack_cat = ', stacked_cat_file )

	d_cat = pds.read_csv( stacked_cat_file )
	ra, dec, z = np.array( d_cat['ra']), np.array( d_cat['dec']), np.array( d_cat['z'] )

	## also divid sub-samples
	zN = len( ra )
	id_arr = np.arange(0, zN, 1)
	id_group = id_arr % N_samples

	lis_ra, lis_dec, lis_z = [], [], []

	## sub-sample
	for nn in range( N_samples ):

		id_xbin = np.where( id_group == nn )[0]

		lis_ra.append( ra[ id_xbin ] )
		lis_dec.append( dec[ id_xbin ] )
		lis_z.append( z[ id_xbin ] )

	## jackknife sub-sample
	for nn in range( rank, rank + 1 ):

		id_arry = np.linspace( 0, N_samples - 1, N_samples )
		id_arry = id_arry.astype( int )
		jack_id = list( id_arry )
		jack_id.remove( jack_id[ nn ] )
		jack_id = np.array( jack_id )

		set_ra, set_dec, set_z = np.array([]), np.array([]), np.array([])

		for oo in ( jack_id ):
			set_ra = np.r_[ set_ra, lis_ra[oo] ]
			set_dec = np.r_[ set_dec, lis_dec[oo] ]
			set_z = np.r_[ set_z, lis_z[oo] ]

		out_files = out_path + 'ZLW_cat_%s_%s-band_%d-jack-sub_member_color.csv' % ( cat_lis[ ll ], band_info, nn )
		z_crit = True

		rep_P_mem_color( set_ra, set_dec, set_z, clust_mem_file, out_files, R_bins, band_info, z_cut = z_crit,)

commd.Barrier()
print('color bin finished!')


if rank == 0:
	## jackknife sample mean
	for ll in range( 2 ):

		jk_rs, jk_g2r = [], []
		jk_g2i, jk_r2i = [], []
		dered_jk_g2r, dered_jk_g2i, dered_jk_r2i = [], [], []

		jk_lwt_g2r, jk_lwt_g2i, jk_lwt_r2i = [], [], []
		jk_lwt_dered_g2r, jk_lwt_dered_g2i, jk_lwt_dered_r2i = [], [], []

		for nn in range( N_samples ):

			jk_sub_dat = pds.read_csv( out_path + 'ZLW_cat_%s_%s-band_%d-jack-sub_member_color.csv' % (cat_lis[ ll ], band_info, nn),)
			tt_R = np.array( jk_sub_dat['centric_R(Mpc/h)'] )

			tt_g2r = np.array( jk_sub_dat['bar_g2r'] )
			tt_g2i = np.array( jk_sub_dat['bar_g2i'] )
			tt_r2i = np.array( jk_sub_dat['bar_r2i'] )

			dered_tt_g2r = np.array( jk_sub_dat['dered_bar_g2r'] )
			dered_tt_g2i = np.array( jk_sub_dat['dered_bar_g2i'] )
			dered_tt_r2i = np.array( jk_sub_dat['dered_bar_r2i'] )

			tt_lwt_g2r = np.array( jk_sub_dat['Lwt_g2r'] )
			tt_lwt_g2i = np.array( jk_sub_dat['Lwt_g2i'] )
			tt_lwt_r2i = np.array( jk_sub_dat['Lwt_r2i'] )

			dered_lwt_g2r = np.array( jk_sub_dat['Lwt_dered_g2r'] )
			dered_lwt_g2i = np.array( jk_sub_dat['Lwt_dered_g2i'] )
			dered_lwt_r2i = np.array( jk_sub_dat['Lwt_dered_r2i'] )


			idx_lim = tt_R < 1e-5 ## rule out radius have no galaxy ocupation
			tt_R[ idx_lim ] = np.nan
			tt_g2r[ idx_lim ] = np.nan
			tt_g2i[ idx_lim ] = np.nan
			tt_r2i[ idx_lim ] = np.nan

			dered_tt_g2r[ idx_lim ] = np.nan
			dered_tt_g2i[ idx_lim ] = np.nan
			dered_tt_r2i[ idx_lim ] = np.nan

			tt_lwt_g2r[ idx_lim ] = np.nan
			tt_lwt_g2i[ idx_lim ] = np.nan
			tt_lwt_r2i[ idx_lim ] = np.nan

			dered_lwt_g2r[ idx_lim ] = np.nan
			dered_lwt_g2i[ idx_lim ] = np.nan
			dered_lwt_r2i[ idx_lim ] = np.nan

			jk_rs.append( tt_R )
			jk_g2r.append( tt_g2r )
			jk_g2i.append( tt_g2i )
			jk_r2i.append( tt_r2i )

			dered_jk_g2r.append( dered_tt_g2r )
			dered_jk_g2i.append( dered_tt_g2i )
			dered_jk_r2i.append( dered_tt_r2i )

			#. r_Lumi weighted case
			jk_lwt_g2r.append( tt_lwt_g2r )
			jk_lwt_g2i.append( tt_lwt_g2i )
			jk_lwt_r2i.append( tt_lwt_r2i )

			jk_lwt_dered_g2r.append( dered_lwt_g2r )
			jk_lwt_dered_g2i.append( dered_lwt_g2i )
			jk_lwt_dered_r2i.append( dered_lwt_r2i )

		m_jk_R, m_jk_g2r, m_jk_g2r_err, std_lim_R = arr_jack_func(jk_g2r, jk_rs, N_samples)
		m_jk_R, m_jk_g2i, m_jk_g2i_err, std_lim_R = arr_jack_func(jk_g2i, jk_rs, N_samples)
		m_jk_R, m_jk_r2i, m_jk_r2i_err, std_lim_R = arr_jack_func(jk_r2i, jk_rs, N_samples)

		m_dered_jk_R, m_dered_jk_g2r, m_dered_jk_g2r_err, std_lim_R = arr_jack_func(dered_jk_g2r, jk_rs, N_samples)
		m_dered_jk_R, m_dered_jk_g2i, m_dered_jk_g2i_err, std_lim_R = arr_jack_func(dered_jk_g2i, jk_rs, N_samples)
		m_dered_jk_R, m_dered_jk_r2i, m_dered_jk_r2i_err, std_lim_R = arr_jack_func(dered_jk_r2i, jk_rs, N_samples)

		#. r_Lumi weighted case
		m_jk_lwt_R, m_jk_lwt_g2r, m_jk_lwt_g2r_err, std_lim_R = arr_jack_func(jk_lwt_g2r, jk_rs, N_samples)
		m_jk_lwt_R, m_jk_lwt_g2i, m_jk_lwt_g2i_err, std_lim_R = arr_jack_func(jk_lwt_g2i, jk_rs, N_samples)
		m_jk_lwt_R, m_jk_lwt_r2i, m_jk_lwt_r2i_err, std_lim_R = arr_jack_func(jk_lwt_r2i, jk_rs, N_samples)

		m_dered_jk_lwt_R, m_dered_jk_lwt_g2r, m_dered_jk_lwt_g2r_err, std_lim_R = arr_jack_func(jk_lwt_dered_g2r, jk_rs, N_samples)
		m_dered_jk_lwt_R, m_dered_jk_lwt_g2i, m_dered_jk_lwt_g2i_err, std_lim_R = arr_jack_func(jk_lwt_dered_g2i, jk_rs, N_samples)
		m_dered_jk_lwt_R, m_dered_jk_lwt_r2i, m_dered_jk_lwt_r2i_err, std_lim_R = arr_jack_func(jk_lwt_dered_r2i, jk_rs, N_samples)


		## save
		keys = [ 'R(cMpc/h)', 'g2r', 'g2r_err', 'g2i', 'g2i_err', 'r2i', 'r2i_err', 
				'dered_g2r', 'dered_g2r_err', 'dered_g2i', 'dered_g2i_err', 'dered_r2i', 'dered_r2i_err', 
				'Lwt_g2r', 'Lwt_g2i', 'Lwt_r2i', 'Lwt_g2r_err', 'Lwt_g2i_err', 'Lwt_r2i_err',
				'Lwt_dered_g2r', 'Lwt_dered_g2i', 'Lwt_dered_r2i', 
				'Lwt_dered_g2r_err', 'Lwt_dered_g2i_err', 'Lwt_dered_r2i_err']

		values = [ m_jk_R, m_jk_g2r, m_jk_g2r_err, m_jk_g2i, m_jk_g2i_err, m_jk_r2i, m_jk_r2i_err, 
				m_dered_jk_g2r, m_dered_jk_g2r_err, m_dered_jk_g2i, m_dered_jk_g2i_err, m_dered_jk_r2i, m_dered_jk_r2i_err, 
				m_jk_lwt_g2r, m_jk_lwt_g2i, m_jk_lwt_r2i, m_jk_lwt_g2r_err, m_jk_lwt_g2i_err, m_jk_lwt_r2i_err, 
				m_dered_jk_lwt_g2r, m_dered_jk_lwt_g2i, m_dered_jk_lwt_r2i, 
				m_dered_jk_lwt_g2r_err, m_dered_jk_lwt_g2i_err, m_dered_jk_lwt_r2i_err ]

		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_path + 'ZLW_cat_%s_%s-band_Mean-jack_member_color.csv' % (cat_lis[ ll ], band_info),)

raise

