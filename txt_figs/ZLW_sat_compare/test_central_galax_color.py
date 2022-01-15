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

### === ###
def simple_match( ra_list, dec_list, z_lis, cat_ra, cat_dec, cat_z, sf_len = 5 ):
	"""
	cat_imgx, cat_imgy : BCG location in image frame
	cat_ra, cat_dec, cat_z : catalog information of image catalog
	ra_list, dec_list, z_lis : catalog information of which used to match to the image catalog
	"""
	lis_ra, lis_dec, lis_z = [], [], []

	com_s = '%.' + '%df' % sf_len

	origin_dex = []

	for kk in range( len(cat_ra) ):

		identi = ( com_s % cat_ra[kk] in ra_list) * (com_s % cat_dec[kk] in dec_list) #* (com_s % cat_z[kk] in z_lis)

		if identi == True:

			ndex_0 = ra_list.index( com_s % cat_ra[kk] )
			ndex_1 = dec_list.index( com_s % cat_dec[kk] )

			lis_ra.append( cat_ra[kk] )
			lis_dec.append( cat_dec[kk] )
			lis_z.append( cat_z[kk] )

			origin_dex.append( ndex_0 )

		else:
			continue

	match_ra = np.array( lis_ra )
	match_dec = np.array( lis_dec )
	match_z = np.array( lis_z )
	origin_dex = np.array( origin_dex )

	return match_ra, match_dec, match_z, origin_dex

def mem_match_func( clust_cat_file, clust_mem_cat_file, band_str, stacked_cat_file, out_files ):

	## cluster catalog
	cat_data = fits.open( clust_cat_file )
	goal_data = cat_data[1].data
	RA = np.array(goal_data.RA)
	DEC = np.array(goal_data.DEC)
	ID = np.array(goal_data.ID)

	z_phot = np.array(goal_data.Z_LAMBDA)
	idvx = (z_phot >= 0.2) & (z_phot <= 0.3)

	ref_Ra, ref_Dec, ref_Z = RA[idvx], DEC[idvx], z_phot[idvx]
	ref_ID = ID[idvx]

	## memeber catalog
	mem_data = fits.open( clust_mem_cat_file )
	sate_data = mem_data[1].data

	group_ID = np.array(sate_data.ID)
	centric_R = np.array(sate_data.R)
	P_member = np.array(sate_data.P)

	mem_r_mag = np.array(sate_data.MODEL_MAG_R)
	mem_r_mag_err = np.array(sate_data.MODEL_MAGERR_R)

	mem_g_mag = np.array(sate_data.MODEL_MAG_G)
	mem_g_mag_err = np.array(sate_data.MODEL_MAGERR_G)

	mem_i_mag = np.array(sate_data.MODEL_MAG_I)
	mem_i_mag_err = np.array(sate_data.MODEL_MAGERR_I)


	d_cat = pds.read_csv( stacked_cat_file )
	ra, dec, z = np.array( d_cat['ra']), np.array( d_cat['dec']), np.array( d_cat['z'])

	out_ra = ['%.5f' % ll for ll in ref_Ra ]
	out_dec = ['%.5f' % ll for ll in ref_Dec ]
	out_z = ['%.5f' % ll for ll in ref_Z ]

	sub_index = simple_match( out_ra, out_dec, out_z, ra, dec, z )[-1]
	match_ID = ref_ID[ sub_index ]

	Ns = len( ra )

	for qq in range( Ns ):

		ra_g, dec_g, z_g = ra[qq], dec[qq], z[qq]

		targ_ID = match_ID[qq]

		id_group = group_ID == targ_ID

		cen_R_arr = centric_R[ id_group ]
		sub_Pmem = P_member[ id_group ]

		sub_r_mag = mem_r_mag[ id_group ]
		sub_g_mag = mem_g_mag[ id_group ]
		sub_i_mag = mem_i_mag[ id_group ]

		sub_r_mag_err = mem_r_mag_err[ id_group ]
		sub_g_mag_err = mem_g_mag_err[ id_group ]
		sub_i_mag_err = mem_i_mag_err[ id_group ]

		keys = [ 'centric_R(Mpc/h)', 'r_mags', 'g_mags', 'i_mags', 'P_member', 'r_mag_err', 'g_mag_err', 'i_mag_err']
		values = [ cen_R_arr, sub_r_mag, sub_g_mag, sub_i_mag, sub_Pmem, sub_r_mag_err, sub_g_mag_err, sub_i_mag_err]
		fill = dict( zip( keys, values ) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_files  % ( band_str, ra_g, dec_g, z_g) )

	return

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

def rep_P_mem_color( sub_ra, sub_dec, sub_z, clust_mem_file, out_files, N_r_bins, band_str ):

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

	N_galaxy = np.zeros( N_radii, dtype = np.float32)

	R_vals = np.zeros( N_radii, dtype = np.float32)

	for pp in range( N_radii - 1 ):

		tmp_m_g2r, tmp_m_r2i, tmp_m_g2i = np.array([]), np.array([]), np.array([])
		tmp_g2r_err, tmp_r2i_err, tmp_g2i_err = np.array([]), np.array([]), np.array([])

		dered_tmp_m_g2r, dered_tmp_m_r2i, dered_tmp_m_g2i = np.array([]), np.array([]), np.array([])

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

			sub_r_mag_err = np.array( sub_dat['r_mag_err'])
			sub_g_mag_err = np.array( sub_dat['g_mag_err'])
			sub_i_mag_err = np.array( sub_dat['i_mag_err'])

			dered_sub_r_mag = np.array( sub_dat['dered_r_mags'])
			dered_sub_g_mag = np.array( sub_dat['dered_g_mags'])
			dered_sub_i_mag = np.array( sub_dat['dered_i_mags'])

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

			id_lim = ( sub_cen_R >= N_r_bins[pp] ) & ( sub_cen_R <= N_r_bins[pp + 1] )

			if np.sum(id_lim) > 0:

				dpt_g2r = sub_g_mag[id_lim] - sub_r_mag[id_lim]
				dpt_g2r_err = np.sqrt( sub_r_mag_err**2 + sub_g_mag_err**2 )

				dpt_r2i = sub_r_mag[id_lim] - sub_i_mag[id_lim]
				dpt_r2i_err = np.sqrt( sub_r_mag_err**2 + sub_i_mag_err**2 )

				dpt_g2i = sub_g_mag[id_lim] - sub_i_mag[id_lim]
				dpt_g2i_err = np.sqrt( sub_g_mag_err**2 + sub_i_mag_err**2 )

				dpt_P_mem = sub_Pmem[id_lim]
				dpt_radius = sub_cen_R[id_lim]

				dered_dpt_g2r = dered_sub_g_mag[id_lim] - dered_sub_r_mag[id_lim]
				dered_dpt_r2i = dered_sub_r_mag[id_lim] - dered_sub_i_mag[id_lim]
				dered_dpt_g2i = dered_sub_g_mag[id_lim] - dered_sub_i_mag[id_lim]

			else:
				dpt_g2r = 0.
				dpt_g2r_err = 0.
				dpt_r2i = 0.
				dpt_r2i_err = 0.
				dpt_g2i = 0.
				dpt_g2i_err = 0.

				dpt_P_mem = 0.
				dpt_radius = 0.

				dered_dpt_g2r = 0.
				dered_dpt_r2i = 0.
				dered_dpt_g2i = 0.

			tmp_m_g2r = np.r_[ tmp_m_g2r, dpt_g2r ]
			tmp_g2r_err = np.r_[ tmp_g2r_err, dpt_g2r_err ]

			tmp_m_r2i = np.r_[ tmp_m_r2i, dpt_r2i ]
			tmp_r2i_err = np.r_[ tmp_r2i_err, dpt_r2i_err ]

			tmp_m_g2i = np.r_[ tmp_m_g2i, dpt_g2i ]
			tmp_g2i_err = np.r_[ tmp_g2i_err, dpt_g2i_err ]

			tmp_Pmem = np.r_[ tmp_Pmem, dpt_P_mem ]
			tmp_radius = np.r_[ tmp_radius, dpt_radius ]

			dered_tmp_m_g2r = np.r_[ dered_tmp_m_g2r, dered_dpt_g2r ]
			dered_tmp_m_r2i = np.r_[ dered_tmp_m_r2i, dered_dpt_r2i ]
			dered_tmp_m_g2i = np.r_[ dered_tmp_m_g2i, dered_dpt_g2i ]

		mean_g2r = np.nansum( tmp_m_g2r * tmp_Pmem ) / np.nansum( tmp_Pmem )
		mean_r2i = np.nansum( tmp_m_r2i * tmp_Pmem ) / np.nansum( tmp_Pmem )
		mean_g2i = np.nansum( tmp_m_g2i * tmp_Pmem ) / np.nansum( tmp_Pmem )

		dered_mean_g2r = np.nansum( dered_tmp_m_g2r * tmp_Pmem ) / np.nansum( tmp_Pmem )
		dered_mean_r2i = np.nansum( dered_tmp_m_r2i * tmp_Pmem ) / np.nansum( tmp_Pmem )
		dered_mean_g2i = np.nansum( dered_tmp_m_g2i * tmp_Pmem ) / np.nansum( tmp_Pmem )

		## use the definition of std
		mean_g2r_err = np.nansum( tmp_Pmem * (mean_g2r - tmp_m_g2r)**2 ) / np.nansum( tmp_Pmem )
		mean_r2i_err = np.nansum( tmp_Pmem * (mean_r2i - tmp_m_r2i)**2 ) / np.nansum( tmp_Pmem )
		mean_g2i_err = np.nansum( tmp_Pmem * (mean_g2i - tmp_m_g2i)**2 ) / np.nansum( tmp_Pmem )

		dered_mean_g2r_err = np.nansum( tmp_Pmem * (dered_mean_g2r - dered_tmp_m_g2r)**2 ) / np.nansum( tmp_Pmem )
		dered_mean_r2i_err = np.nansum( tmp_Pmem * (dered_mean_r2i - dered_tmp_m_r2i)**2 ) / np.nansum( tmp_Pmem )
		dered_mean_g2i_err = np.nansum( tmp_Pmem * (dered_mean_g2i - dered_tmp_m_g2i)**2 ) / np.nansum( tmp_Pmem )

		n_galax = len( tmp_m_g2r )

		mean_radius = np.nansum( tmp_Pmem * tmp_radius ) / np.nansum( tmp_Pmem )

		bar_g2r[ pp ] = mean_g2r
		bar_g2r_err[ pp ] = mean_g2r_err

		bar_r2i[ pp ] = mean_r2i
		bar_r2i_err[ pp ] = mean_r2i_err

		bar_g2i[ pp ] = mean_g2i
		bar_g2i_err[ pp ] = mean_g2i_err


		dered_bar_g2r[ pp ] = dered_mean_g2r
		dered_bar_r2i[ pp ] = dered_mean_r2i
		dered_bar_g2i[ pp ] = dered_mean_g2i

		dered_bar_g2r_err[ pp ] = dered_mean_g2r_err
		dered_bar_r2i_err[ pp ] = dered_mean_r2i_err
		dered_bar_g2i_err[ pp ] = dered_mean_g2i_err


		N_galaxy[ pp ] = n_galax

		R_vals[ pp ] = mean_radius

	## save
	keys = [ 'centric_R(Mpc/h)', 'bar_g2r', 'bar_r2i', 'bar_g2i', 'bar_g2r_err', 'bar_r2i_err', 'bar_g2i_err', 
			'dered_bar_g2r', 'dered_bar_r2i', 'dered_bar_g2i', 'dered_bar_g2r_err', 'dered_bar_r2i_err', 'dered_bar_g2i_err', 
			'n_galaxy' ]
	values = [ R_vals, bar_g2r, bar_r2i, bar_g2i, bar_g2r_err, bar_r2i_err, bar_g2i_err, 
				dered_bar_g2r, dered_bar_r2i, dered_bar_g2i, dered_bar_g2r_err, dered_bar_r2i_err, dered_bar_g2i_err, 
				N_galaxy ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_files )

	return

home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'
out_path = '/home/xkchen/project/tmp/'

band_info = band[ 0 ]

#..fixed BCG stellar mass sub-sammples
# cat_lis = [ 'low-age', 'hi-age' ]
# cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
cat_lis = [ 'low-rich', 'hi-rich' ]

#... match member properties with clusters
# for ll in range( 2 ):
# 	clust_cat_file = home + 'redmapper/redmapper_dr8_public_v6.3_catalog.fits'
# 	clust_mem_cat_file = home + 'redmapper/redmapper_dr8_public_v6.3_members.fits'
# 	stacked_cat_file = load + 'pkoffset_cat/%s_%s-band_photo-z-match_pk-offset_BCG-pos_cat_z-ref.csv' % ( cat_lis[ll], band_info )
# 	out_files = home + 'member_match/photo-z_%s-band_ra%.3f_dec%.3f_z%.3f_members_mag.csv'
# 	mem_match_func( clust_cat_file, clust_mem_cat_file, band_info, stacked_cat_file, out_files )


#... table of member and cluster matched
cat_data = fits.open( home + 'redmapper/redmapper_dr8_public_v6.3_catalog.fits' )
goal_data = cat_data[1].data
RA = np.array(goal_data.RA)
DEC = np.array(goal_data.DEC)
ID = np.array(goal_data.ID)

z_phot = np.array(goal_data.Z_LAMBDA)
idvx = (z_phot >= 0.2) & (z_phot <= 0.3)

ref_Ra, ref_Dec, ref_Z = RA[idvx], DEC[idvx], z_phot[idvx]
ref_ID = ID[idvx]

mem_data = fits.open( home + 'redmapper/redmapper_dr8_public_v6.3_members.fits' )
sate_data = mem_data[1].data

group_ID = np.array(sate_data.ID)
centric_R = np.array(sate_data.R)
P_member = np.array(sate_data.P)

mem_r_mag = np.array(sate_data.MODEL_MAG_R)
mem_r_mag_err = np.array(sate_data.MODEL_MAGERR_R)

mem_g_mag = np.array(sate_data.MODEL_MAG_G)
mem_g_mag_err = np.array(sate_data.MODEL_MAGERR_G)

mem_i_mag = np.array(sate_data.MODEL_MAG_I)
mem_i_mag_err = np.array(sate_data.MODEL_MAGERR_I)

sat_ra, sat_dec = np.array(sate_data.RA), np.array(sate_data.DEC)
sat_Z = np.array( sate_data.Z_SPEC )
sat_objID = np.array( sate_data.OBJID )

for ll in range( 2 ):
	d_cat = pds.read_csv( load + 
			'pkoffset_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_cat_z-ref.csv' % ( cat_lis[ll], band_info ),)
	ra, dec, z = np.array( d_cat['ra']), np.array( d_cat['dec']), np.array( d_cat['z'])

	out_ra = ['%.5f' % ll for ll in ref_Ra ]
	out_dec = ['%.5f' % ll for ll in ref_Dec ]
	out_z = ['%.5f' % ll for ll in ref_Z ]

	sub_index = simple_match( out_ra, out_dec, out_z, ra, dec, z )[-1]
	match_ID = ref_ID[ sub_index ]
	Ns = len( ra )

	F_tree = h5py.File( '/home/xkchen/%s_clus-sat_record.h5' % cat_lis[ll], 'w')

	for qq in range( Ns ):

		ra_g, dec_g, z_g = ra[qq], dec[qq], z[qq]

		targ_ID = match_ID[qq]

		id_group = group_ID == targ_ID

		cen_R_arr = centric_R[ id_group ]
		sub_Pmem = P_member[ id_group ]

		sub_r_mag = mem_r_mag[ id_group ]
		sub_g_mag = mem_g_mag[ id_group ]
		sub_i_mag = mem_i_mag[ id_group ]

		sub_r_mag_err = mem_r_mag_err[ id_group ]
		sub_g_mag_err = mem_g_mag_err[ id_group ]
		sub_i_mag_err = mem_i_mag_err[ id_group ]

		sub_ra, sub_dec, sub_z = sat_ra[ id_group ], sat_dec[ id_group ], sat_Z[ id_group ]
		sub_obj_IDs = sat_objID[ id_group ]

		sub_g2r = sub_g_mag - sub_r_mag
		sub_g2i = sub_g_mag - sub_i_mag
		sub_r2i = sub_r_mag - sub_i_mag

		out_arr = np.array( [ sub_ra, sub_dec, sub_z, cen_R_arr, sub_Pmem, sub_g2r, sub_g2i, sub_r2i ] )
		gk = F_tree.create_group( "clust_%d/" % qq )
		dk0 = gk.create_dataset( "arr", data = out_arr )
		dk1 = gk.create_dataset( "IDs", data = sub_obj_IDs )

	F_tree.close()

raise


N_samples = 30

# N_bins = 55
# R_bins = np.logspace(0, np.log10(2e3), N_bins) / 1e3 # Mpc/h

R_bins_0 = np.logspace( 1, 2, 5)
R_bins_1 = np.logspace( 2, 3.302, 20)
R_bins = np.r_[ R_bins_0[:-1], R_bins_1 ] / 1e3
N_bins = len( R_bins )

for ll in range( 2 ):

	stacked_cat_file = load + 'pkoffset_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_cat_z-ref.csv' % ( cat_lis[ll], band_info )

	# clust_mem_file = home + 'member_match/photo-z_%s-band_ra%.3f_dec%.3f_z%.3f_members_mag.csv'
	clust_mem_file = home + 'ZLWen_cat/redMap_mem_match/photo-z_%s-band_ra%.3f_dec%.3f_z%.3f_members_mag.csv'

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

		out_files = out_path + '%s_%s-band_%d-jack-sub_member_color.csv' % ( cat_lis[ll], band_info, nn )
		rep_P_mem_color( set_ra, set_dec, set_z, clust_mem_file, out_files, R_bins, band_info )

commd.Barrier()
print('color bin finished!')

if rank == 0:
	## jackknife sample mean
	for ll in range( 2 ):

		jk_rs, jk_g2r = [], []
		jk_g2i, jk_r2i = [], []
		dered_jk_g2r, dered_jk_g2i, dered_jk_r2i = [], [], []
		for nn in range( N_samples ):

			jk_sub_dat = pds.read_csv( out_path + '%s_%s-band_%d-jack-sub_member_color.csv' % ( cat_lis[ll], band_info, nn ),)
			tt_R = np.array( jk_sub_dat['centric_R(Mpc/h)'] )

			tt_g2r = np.array( jk_sub_dat['bar_g2r'] )
			tt_g2i = np.array( jk_sub_dat['bar_g2i'] )
			tt_r2i = np.array( jk_sub_dat['bar_r2i'] )

			dered_tt_g2r = np.array( jk_sub_dat['dered_bar_g2r'] )
			dered_tt_g2i = np.array( jk_sub_dat['dered_bar_g2i'] )
			dered_tt_r2i = np.array( jk_sub_dat['dered_bar_r2i'] )

			idx_lim = tt_R < 1e-5 ## rule out radius have no galaxy ocupation
			tt_R[ idx_lim ] = np.nan
			tt_g2r[ idx_lim ] = np.nan
			tt_g2i[ idx_lim ] = np.nan
			tt_r2i[ idx_lim ] = np.nan

			dered_tt_g2r[ idx_lim ] = np.nan
			dered_tt_g2i[ idx_lim ] = np.nan
			dered_tt_r2i[ idx_lim ] = np.nan

			jk_rs.append( tt_R )
			jk_g2r.append( tt_g2r )
			jk_g2i.append( tt_g2i )
			jk_r2i.append( tt_r2i )

			dered_jk_g2r.append( dered_tt_g2r )
			dered_jk_g2i.append( dered_tt_g2i )
			dered_jk_r2i.append( dered_tt_r2i )

		m_jk_R, m_jk_g2r, m_jk_g2r_err, std_lim_R = arr_jack_func(jk_g2r, jk_rs, N_samples)
		m_jk_R, m_jk_g2i, m_jk_g2i_err, std_lim_R = arr_jack_func(jk_g2i, jk_rs, N_samples)
		m_jk_R, m_jk_r2i, m_jk_r2i_err, std_lim_R = arr_jack_func(jk_r2i, jk_rs, N_samples)

		m_dered_jk_R, m_dered_jk_g2r, m_dered_jk_g2r_err, std_lim_R = arr_jack_func(dered_jk_g2r, jk_rs, N_samples)
		m_dered_jk_R, m_dered_jk_g2i, m_dered_jk_g2i_err, std_lim_R = arr_jack_func(dered_jk_g2i, jk_rs, N_samples)
		m_dered_jk_R, m_dered_jk_r2i, m_dered_jk_r2i_err, std_lim_R = arr_jack_func(dered_jk_r2i, jk_rs, N_samples)

		## save
		keys = [ 'R(cMpc/h)', 'g2r', 'g2r_err', 'g2i', 'g2i_err', 'r2i', 'r2i_err', 
				'dered_g2r', 'dered_g2r_err', 'dered_g2i', 'dered_g2i_err', 'dered_r2i', 'dered_r2i_err']
		values = [ m_jk_R, m_jk_g2r, m_jk_g2r_err, m_jk_g2i, m_jk_g2i_err, m_jk_r2i, m_jk_r2i_err, 
				m_dered_jk_g2r, m_dered_jk_g2r_err, m_dered_jk_g2i, m_dered_jk_g2i_err, m_dered_jk_r2i, m_dered_jk_r2i_err ]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_path + '%s_%s-band_Mean-jack_member_color.csv' % (cat_lis[ll], band_info),)

print('color bin finished!')

