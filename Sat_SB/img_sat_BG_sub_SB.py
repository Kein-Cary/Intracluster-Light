import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc

from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy.stats import binned_statistic as binned
import scipy.interpolate as interp

#.
from img_sat_fig_out_mode import arr_jack_func
from light_measure import jack_SB_func


### ==== ###

##... take the average value of the BCG+ICL+BG at the location of satellites
##... for each satellite, we use the physical separation to the central galaxy
def aveg_BG_sub_func( sat_sb_file, band_str, bg_sb_file, out_file, R_sat):
	"""
	sat_sb_file : SB(r) of satellites, .h5 files
	band_str : filter information
	bg_sb_file : background SB(r), .h5 file
	out_file : .csv file
	R_sat : centric distance of satellites, ('kpc')
	"""

	#. no-scale applied on BCG+ICL profile
	with h5py.File( sat_sb_file, 'r') as f:
		tmp_r = np.array(f['r'])
		tmp_sb = np.array(f['sb'])
		tmp_err = np.array(f['sb_err'])


	with h5py.File( bg_sb_file, 'r') as f:
		t_bg_r = np.array(f['r'])
		t_bg_sb = np.array(f['sb'])
		t_bg_err = np.array(f['sb_err'])

	interp_mu_F = interp.interp1d( t_bg_r, t_bg_sb, kind = 'linear', fill_value = 'extrapolate')

	_kk_BG = np.sum( interp_mu_F( R_sat ) ) / len( R_sat )
	_std_BG = np.std( interp_mu_F( R_sat ) ) / np.sqrt( len( R_sat ) - 1 )

	_out_sb = tmp_sb - _kk_BG
	_kk_bgs = np.ones( len( tmp_r ), ) * _kk_BG
	_kk_bg_err = np.ones( len( tmp_r ), ) * _std_BG

	#.
	keys = [ 'r', 'sb', 'sb_err', 'bg_sb', 'bg_err' ]
	values = [ tmp_r, _out_sb, tmp_err, _kk_bgs, _kk_bg_err ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( out_file )

	return


##... cut the mock of stacked 2D image of BCG+ICL+BG of corresponding sample
##... with the same size of pixels-array as satellites and stack those images as the background
def stack_BG_sub_func( sat_sb_file, bg_sb_file, band_str, N_sample, out_file, sub_out_file = None):
	"""
	sat_sb_file : SB(r) of satellites, .h5 files
	band_str : filter information
	bg_sb_file : background SB(r), .h5 file
	out_file : .csv file
	N_sample : number of subsample
	sub_out_file : output the BG-sub profile of subsamples or not
	"""

	tmp_SB, tmp_R = [], []

	with h5py.File( bg_sb_file, 'r') as f:
		t_bg_r = np.array(f['r'])
		t_bg_sb = np.array(f['sb'])
		t_bg_err = np.array(f['sb_err'])

	interp_mu_F = interp.interp1d( t_bg_r, t_bg_sb, kind = 'linear', fill_value = 'extrapolate')

	interp_mu_err = interp.interp1d( t_bg_r, t_bg_err, kind = 'linear', fill_value = 'extrapolate')


	for kk in range( N_sample ):

		with h5py.File( sat_sb_file % kk, 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])
			tt_npix = np.array(f['npix'])

		id_Nul = tt_npix < 1
		tt_r[ id_Nul ] = np.nan
		tt_sb[ id_Nul ] = np.nan
		tt_err[ id_Nul ] = np.nan

		_kk_sb = tt_sb - interp_mu_F( tt_r )

		if sub_out_file is not None:

			_kk_err = np.sqrt( tt_err**2 + interp_mu_err( tt_r )**2 )

			keys = [ 'r', 'sb', 'sb_err' ]
			values = [ tt_r, _kk_sb, _kk_err ]
			fill = dict( zip( keys, values ) )
			out_data = pds.DataFrame( fill )
			out_data.to_csv( sub_out_file % kk )

		tmp_R.append( tt_r )
		tmp_SB.append( _kk_sb )

	tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func( tmp_SB, tmp_R, band_str, N_sample)[4:]

	keys = [ 'r', 'sb', 'sb_err' ]
	values = [ tt_jk_R, tt_jk_SB, tt_jk_err ]
	fill = dict( zip( keys, values ) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_file )

	return

