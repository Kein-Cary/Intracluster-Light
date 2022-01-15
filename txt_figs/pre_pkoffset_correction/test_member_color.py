import time
import h5py
import numpy as np
import astropy.io.fits as fits

import mechanize
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
def simple_match(ra_list, dec_list, z_lis, cat_ra, cat_dec, cat_z, sf_len = 5):
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

home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

## cluster catalog
cat_file = home + 'redmapper/redmapper_dr8_public_v6.3_catalog.fits'
cat_data = fits.open( cat_file )
goal_data = cat_data[1].data
RA = np.array(goal_data.RA)
DEC = np.array(goal_data.DEC)
ID = np.array(goal_data.ID)

z_phot = np.array(goal_data.Z_LAMBDA)
idvx = (z_phot >= 0.2) & (z_phot <= 0.3)

ref_Ra, ref_Dec, ref_Z = RA[idvx], DEC[idvx], z_phot[idvx]
ref_ID = ID[idvx]

## memeber catalog
member_file = home + 'redmapper/redmapper_dr8_public_v6.3_members.fits'
mem_data = fits.open( member_file )
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

band_str = band[ rank ]

"""
### === ### sample
for mass_id in (True, False):

	if mass_id == True:
		cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
		fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$']

	if mass_id == False:
		cat_lis = ['younger', 'older']
		fig_name = ['younger', 'older']

	for ll in range( 2 ):

		if mass_id == True:
			d_cat = pds.read_csv( load + 'photo_z_cat/%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat_z-ref.csv' % (cat_lis[ll], band_str), )

		if mass_id == False:
			d_cat = pds.read_csv( load + 'z_formed_cat/%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[ll], band_str), )

		ra, dec, z = np.array( d_cat['ra']), np.array( d_cat['dec']), np.array( d_cat['z'])
		cen_x, cen_y = np.array(d_cat['bcg_x']), np.array( d_cat['bcg_y'])

		out_ra = ['%.5f' % ll for ll in ref_Ra ]
		out_dec = ['%.5f' % ll for ll in ref_Dec ]
		out_z = ['%.5f' % ll for ll in ref_Z ]

		sub_index = simple_match( out_ra, out_dec, out_z, ra, dec, z )[-1]
		match_ID = ref_ID[ sub_index ]

		print( len(ra) )
		print( len(match_ID) )

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
			out_data.to_csv( home + 'member_match/photo-z_%s-band_ra%.3f_dec%.3f_z%.3f_members_mag.csv' % (band[ rank ], ra_g, dec_g, z_g),)

print('match finished !')
"""


## calculate mean color of galaxies in severals radius bins
N_bins = 55
N_samples = 30

R_bins = np.logspace(0, np.log10(2e3), N_bins) / 1e3 # Mpc/h

'''
for mass_id in (True, False):

	if mass_id == True:
		cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
		fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$']

	if mass_id == False:
		cat_lis = ['younger', 'older']
		fig_name = ['younger', 'older']

	for ll in range( 2 ):

		if mass_id == True:
			d_cat = pds.read_csv( load + 'photo_z_cat/%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat_z-ref.csv' % (cat_lis[ll], band_str), )

		if mass_id == False:
			d_cat = pds.read_csv( load + 'z_formed_cat/%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[ll], band_str), )

		ra, dec, z = np.array( d_cat['ra']), np.array( d_cat['dec']), np.array( d_cat['z'])

		## also divid sub-samples
		zN = len( ra )
		id_arr = np.arange(0, zN, 1)
		id_group = id_arr % N_samples

		lis_ra, lis_dec, lis_z = [], [], []
		lis_x, lis_y = [], []

		## sub-sample
		for nn in range( N_samples ):

			id_xbin = np.where( id_group == nn )[0]

			lis_ra.append( ra[ id_xbin ] )
			lis_dec.append( dec[ id_xbin ] )
			lis_z.append( z[ id_xbin ] )

		## jackknife sub-sample
		for nn in range( N_samples ):

			id_arry = np.linspace( 0, N_samples - 1, N_samples )
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

			bar_g2r = np.zeros( N_bins, dtype = np.float32)
			bar_g2r_err = np.zeros( N_bins, dtype = np.float32)

			bar_r2i = np.zeros( N_bins, dtype = np.float32)
			bar_r2i_err = np.zeros( N_bins, dtype = np.float32)

			N_galaxy = np.zeros( N_bins, dtype = np.float32)

			driv_r2i_err = np.zeros( N_bins, dtype = np.float32)
			driv_g2r_err = np.zeros( N_bins, dtype = np.float32)

			R_vals = np.zeros( N_bins, dtype = np.float32)

			for pp in range( len(R_bins) - 1 ):

				tmp_m_g2r, tmp_m_r2i = np.array([]), np.array([])
				tmp_g2r_err, tmp_r2i_err = np.array([]), np.array([])
				tmp_Pmem = np.array([])
				tmp_radius = np.array([])

				for ii in range( ncs ):

					ra_g, dec_g, z_g = set_ra[ii], set_dec[ii], set_z[ii]

					sub_dat = pds.read_csv( home + 'member_match/photo-z_%s-band_ra%.3f_dec%.3f_z%.3f_members_mag.csv' % (band[ rank ], ra_g, dec_g, z_g),)

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

					id_lim = ( sub_cen_R >= R_bins[pp] ) & ( sub_cen_R <= R_bins[pp + 1] )

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
			out_data.to_csv('/home/xkchen/figs/%s_%s-band_%d-jack-sub_member_color.csv' % (cat_lis[ll], band_str, nn),)
'''

for mass_id in (True, False):

	if mass_id == True:
		cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
		fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$']

	if mass_id == False:
		cat_lis = ['younger', 'older']
		fig_name = ['younger', 'older']

	for ll in range( 2 ):

		## jackknife sample mean
		jk_rs, jk_g2r = [], []
		for nn in range( N_samples ):

			jk_sub_dat = pds.read_csv( '/home/xkchen/figs/%s_%s-band_%d-jack-sub_member_color.csv' % (cat_lis[ll], band_str, nn),)
			tt_R = np.array( jk_sub_dat['centric_R(Mpc/h)'] )
			tt_g2r = np.array( jk_sub_dat['bar_g2r'] )

			idx_lim = tt_R < 1e-5 ## rule out radius have no galaxy ocupation
			tt_R[ idx_lim ] = np.nan
			tt_g2r[ idx_lim ] = np.nan

			jk_rs.append( tt_R )
			jk_g2r.append( tt_g2r )

		m_jk_R, m_jk_g2r, m_jk_g2r_err, std_lim_R = arr_jack_func(jk_g2r, jk_rs, N_samples)

		## save
		keys = [ 'R(cMpc/h)', 'g2r', 'g2r_err' ]
		values = [ m_jk_R, m_jk_g2r, m_jk_g2r_err ]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( '/home/xkchen/figs/%s_%s-band_Mean-jack_member_color.csv' % (cat_lis[ll], band_str),)

print('color bin finished!')

