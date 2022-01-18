import matplotlib as mpl
import matplotlib.pyplot as plt

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc
import scipy.stats as sts

import h5py
import numpy as np
import pandas as pds
from scipy import interpolate as interp
from scipy import integrate as integ
from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy.stats import binned_statistic as binned

#.
from img_sat_pros_stack import single_img_SB_func


from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1. - Omega_m
Omega_k = 1. - (Omega_lambda + Omega_m)

pixel = 0.396

rad2asec = U.rad.to(U.arcsec)

### === 
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

def sat_mag_comu_func( dat_file, bcg_ra, bcg_dec, lis_ra, lis_dec, lis_z, out_file, R_bins, z_ref = None, n_skip = None):
	"""
	out_files : .csv files
	"""

	band = ['r', 'g', 'i']

	N_sam = len( lis_ra )

	samp_rmag, samp_gmag, samp_imag = np.zeros( N_sam,), np.zeros( N_sam,), np.zeros( N_sam,)
	
	r_mag_10, i_mag_10, g_mag_10 = np.zeros( N_sam,), np.zeros( N_sam,), np.zeros( N_sam,)

	r_mag, i_mag, g_mag = np.zeros( N_sam,), np.zeros( N_sam,), np.zeros( N_sam,)


	for ii in range( N_sam ):

		ra_g, dec_g, z_g = lis_ra[ii], lis_dec[ii], lis_z[ii]

		Da_g = Test_model.angular_diameter_distance( z_g ).value # unit Mpc
		Dl_g = Test_model.luminosity_distance( z_g ).value # unit Mpc

		tmp_mag, tmp_mag_10 = [], []

		for kk in range( 3 ):

			band_str = band[kk]

			tt_rbins, tt_fdens = single_img_SB_func( band_str, z_g, ra_g, dec_g, dat_file, R_bins, z_ref = z_ref, n_skip = n_skip)

			id_inf = np.isinf( tt_fdens )
			id_nul = tt_fdens <= 0.
			id_lim = id_inf & id_nul

			tt_rbins = tt_rbins[ id_lim == False ]
			tt_fdens = tt_fdens[ id_lim == False ]
			angl_r = ( tt_rbins / 1e3 ) * rad2asec / Da_g

			#. cumulative flux
			cumu_F = cumu_mass_func( angl_r, tt_fdens )

			intep_cumu_F_f = interp.interp1d( tt_rbins, cumu_F, kind = 'linear', fill_value = 'extrapolate',)


			# tt_mag = 22.5 - 2.5 * np.log10( cumu_F[-1] )

			#. integral luminosity (limited with 30 kpc)
			tt_mag = 22.5 - 2.5 * np.log10( intep_cumu_F_f( 30 ) )

			tt_mag_10 = 22.5 - 2.5 * np.log10( intep_cumu_F_f( 10 ) )

			tmp_mag.append( tt_mag )
			tmp_mag_10.append( tt_mag_10 )

			# #. satellites have no profMean matched
			# tmp_mag.append( -100 )
			# tmp_mag_10.append( -100 )


		r_mag_10[ ii ], g_mag_10[ ii ], i_mag_10[ ii ] = tmp_mag_10
		r_mag[ ii ], g_mag[ ii ], i_mag[ ii ] = tmp_mag


	keys = [ 'bcg_ra', 'bcg_dec', 'ra', 'dec', 'clus_z', 'rmag', 'gmag', 'imag', 'rmag_10', 'gmag_10', 'imag_10']
	values = [ bcg_ra, bcg_dec, lis_ra, lis_dec, lis_z, r_mag, g_mag, i_mag, r_mag_10, g_mag_10, i_mag_10 ]

	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( out_file )

	return


### === 
home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

z_ref = 0.25

dat = pds.read_csv( home + 'member_files/sat_cat_z02_03/' + 
					'Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv')

prof_file = '/home/xkchen/data/SDSS/member_files/sat_profMean/sat_ra%.5f_dec%.5f_SDSS_prof.txt'

bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
sat_ra, sat_dec = np.array( dat['ra'] ), np.array( dat['dec'] )


Ns = len( sat_ra )
m, n = divmod( Ns, cpus)
N_sub0, N_sub1 = m * rank, (rank + 1) * m
if rank == cpus - 1:
	N_sub1 += n

set_ra, set_dec, set_z = bcg_ra[ N_sub0 : N_sub1], bcg_dec[ N_sub0 : N_sub1], bcg_z[ N_sub0 : N_sub1]
sub_ra, sub_dec = sat_ra[ N_sub0 : N_sub1], sat_dec[ N_sub0 : N_sub1]

n_skip = 0
r_bins = np.logspace( 0, 2.48, 27 )  ##. kpc

out_file = '/home/xkchen/project/tmp_obj_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat-aper-mag_%d.csv' %  rank

sat_mag_comu_func( prof_file, set_ra, set_dec, sub_ra, sub_dec, set_z, out_file, r_bins, n_skip = n_skip)

commd.Barrier()


if rank == 0:

	dat = pds.read_csv('/home/xkchen/project/tmp_obj_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat-aper-mag_0.csv')

	keys = dat.columns[1:]
	Nks = len( keys )

	tmp_arr = []

	for tt in range( Nks ):

		sub_arr = np.array( dat['%s' % keys[ tt ] ] )
		tmp_arr.append( sub_arr )

	for pp in range( 1, cpus ):

		dat = pds.read_csv('/home/xkchen/project/tmp_obj_cat/' + 
						'Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat-aper-mag_%d.csv' % pp )

		for tt in range( Nks ):

			sub_arr = np.array( dat['%s' % keys[ tt ] ] )
			tmp_arr[ tt ] = np.r_[ tmp_arr[ tt ], sub_arr ]

	#.
	fill = dict( zip( keys, tmp_arr ) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv('/home/xkchen/' + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat-aper-mag.csv' )

