"""
use to model SB profiles before background subtracted
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
from scipy import special

#.
from img_sat_fig_out_mode import arr_jack_func


##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']


### === ### func.s
def sersic_func( R, n, Ie, Re ):
	"""
	for n > 0.36
	"""
	bc0 = 4 / (405 * n)
	bc1 = 45 / (25515 * n**2)
	bc2 = 131 / (1148175 * n**3)
	bc3 = 2194697 / ( 30690717750 * n**4 )

	b_n = 2 * n - 1/3 + bc0 + bc1 + bc2 - bc3
	mf0 = -b_n * ( R / Re )**( 1 / n )
	Ir = Ie * np.exp( mf0 + b_n )
	return Ir

##. core-like funcs
def Moffat_func(R, A0, Rd, n):
	mf = A0 / ( 1 + (R / Rd)**2 )**n
	return mf

def Modi_Ferrer_func( R, A0, R_bk, belta, alpha):

	mf0 = 1 - ( R / R_bk )**( 2 - belta)
	mf = mf0**alpha

	return mf * A0

def Empi_King_func( R, A0, R_t, R_c, alpha):

	mf0 = 1 + (R_t / R_c)**2
	mf1 = mf0**( 1 / alpha )

	mf2 = 1 / ( 1 - 1 / mf1 )**alpha

	mf4 = 1 + ( R / R_c)**2
	mf5 = mf4**( 1 / alpha )

	mf6 = 1 / mf5 - 1 / mf1

	mf = mf2 * mf6**alpha

	return mf * A0

def Nuker_func( R, A0, R_bk, alpha, belta, gamma):

	mf0 = 2**( (belta - gamma) / alpha )
	mf1 = mf0 / ( R / R_bk)**gamma

	mf2 = 1 + ( R / R_bk  )**alpha
	mf3 = mf2**( (gamma - belta) / alpha )

	mf = mf1 * mf3

	return mf * A0

def power1_func( R, R0, alpha ):
	A0 = 1.
	mf = A0 * ( (R / R0)**2.5 + 1)**alpha * 2**(-alpha)
	return mf / mf[0] - 1



### === ### data load
def cumuL_resiL_estimate():

	##. residual Lumi and out fitting function integration
	BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGs/'
	out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/'
	path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_SBs/'

	bin_rich = [ 20, 30, 50, 210 ]

	R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m

	#.
	list_order = 13
	N_sample = 100

	band_str = 'r'

	##... BG-subtracted SBs
	nbg_R, nbg_SB, nbg_err = [], [], []

	for tt in range( len(R_bins) - 1 ):

		dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) 
										+ '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

		pt_r = np.array( dat['r'] )
		pt_sb = np.array( dat['sb'] )
		pt_err = np.array( dat['sb_err'] )

		#.
		nbg_R.append( pt_r )
		nbg_SB.append( pt_sb )
		nbg_err.append( pt_err )


	##... interp the outer radius bin
	cp_k_r = nbg_R[ -1 ]
	cp_k_sb = nbg_SB[ -1 ]
	cp_k_err = nbg_err[ -1 ]
	tmp_F0 = interp.splrep( cp_k_r, cp_k_sb, s = 0)


	##...
	for tt in range( len(R_bins) - 2 ):

		#.
		kk_r = nbg_R[ tt ]
		kk_sb = nbg_SB[ tt ]
		kk_err = nbg_err[ tt ]

		#.
		Da_ref = Test_model.angular_diameter_distance( z_ref ).value  ## Mpc.
		ang_r = rad2arcsec * kk_r / ( Da_ref * 1e3 )


		#.
		nbg_R.append( pt_r )
		nbg_SB.append( pt_sb )
		nbg_err.append( pt_err )

		#. fitting parameters
		dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/trunF_mod/' + 
					'sat_%.2f-%.2fR200m_%s-band_SB_trunF-model_fit.csv' % (R_bins[tt], R_bins[tt + 1], band_str),)

		A0_fit = np.array( dat['A_Moffat'] )[0]
		Rc0_fit = np.array( dat['Rd_Moffat'] )[0]
		alpha_fit = np.array( dat['n_Moffat'] )[0]

		Rc1_fit = np.array( dat['Rc'] )[0]
		beta_fit = np.array( dat['beta'] )[0]

		#.
		_SB0_arr = interp.splev( kk_r, tmp_F0, der = 0)

		mf0 = Moffat_func( kk_r, A0_fit, Rc0_fit, alpha_fit )
		mf1 = power1_func( kk_r, Rc1_fit, beta_fit )

		##...
		tmp_R, tmp_trunL = [], []

		for dd in range( N_sample ):

			dat = pds.read_csv( out_path + 
					'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_jack-sub-%d_BG-sub-SB-pro_z-ref.h5' 
					% (R_bins[-2], R_bins[-1], band_str, dd),)

			tt_r = np.array( dat['r'])
			tt_sb = np.array( dat['sb'])

			#.
			id_nan = np.isnan( tt_sb )
			dt_r, dt_sb = tt_r[ id_nan == False ], tt_sb[ id_nan == False ]

			tmp_F = interp.interp1d( dt_r, dt_sb, kind = 'linear', fill_value = 'extrapolate')

			cen_L = mf0 * tmp_F( kk_r )

			cumu_lx = integ.cumtrapz( ang_r * cen_L, x = ang_r, initial = np.min( ang_r ) / 10 )
			cumu_lx = cumu_lx * np.pi * 2

			#.
			tmp_R.append( ang_r )
			tmp_trunL.append( cumu_lx )

		#.
		aveg_R, aveg_L, aveg_err, lim_R = arr_jack_func( tmp_trunL, tmp_R, N_sample )
		keys = [ 'r', 'cumu_L', 'cumu_Lerr' ]
		values = [ kk_r, aveg_L, aveg_err ]
		fill = dict( zip( keys, values ) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/trunF_mod/' + 
					'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_BG-sub-SB_aveg_model-cen-cumu-L.csv'
					% (R_bins[tt], R_bins[tt + 1], band_str),)

		##...	
		tmp_R, tmp_cumL = [], []

		for dd in range( N_sample ):

			dat = pds.read_csv( out_path + 
					'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_jack-sub-%d_BG-sub-SB-pro_z-ref.h5' 
					% (R_bins[-2], R_bins[-1], band_str, dd),)

			tt_r = np.array( dat['r'])
			tt_sb = np.array( dat['sb'])

			#.
			id_nan = np.isnan( tt_sb )
			dt_r, dt_sb = tt_r[ id_nan == False ], tt_sb[ id_nan == False ]

			tmp_F = interp.interp1d( dt_r, dt_sb, kind = 'linear', fill_value = 'extrapolate')

			out_L = mf1 * tmp_F( kk_r )

			cumu_lx = integ.cumtrapz( ang_r * out_L, x = ang_r, initial = np.min( ang_r ) / 10 )
			cumu_lx = cumu_lx * np.pi * 2

			#.
			tmp_R.append( ang_r )
			tmp_cumL.append( cumu_lx )

		#.
		aveg_R, aveg_L, aveg_err, lim_R = arr_jack_func( tmp_cumL, tmp_R, N_sample )
		keys = [ 'r', 'cumu_L', 'cumu_Lerr' ]
		values = [ kk_r, aveg_L, aveg_err ]
		fill = dict( zip( keys, values ) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/trunF_mod/' + 
					'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_BG-sub-SB_aveg_model-out-cumu-L.csv'
					% (R_bins[tt], R_bins[tt + 1], band_str),)

		##... 
		tmp_R, tmp_resL = [], []

		for dd in range( N_sample ):

			dat = pds.read_csv( out_path + 
					'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_jack-sub-%d_BG-sub-SB-pro_z-ref.h5' 
					% (R_bins[-2], R_bins[-1], band_str, dd),)

			tt_r = np.array( dat['r'])
			tt_sb = np.array( dat['sb'])

			#.
			id_nan = np.isnan( tt_sb )
			dt_r, dt_sb = tt_r[ id_nan == False ], tt_sb[ id_nan == False ]

			tmp_F = interp.interp1d( dt_r, dt_sb, kind = 'linear', fill_value = 'extrapolate')

			#. resi_L
			devi_L = tmp_F( kk_r ) - mf0 * _SB0_arr

			cumu_lx = integ.cumtrapz( ang_r * devi_L, x = ang_r, initial = np.min( ang_r ) / 10 )
			cumu_lx = cumu_lx * np.pi * 2

			#.
			tmp_R.append( ang_r )
			tmp_resL.append( cumu_lx )

		#.
		aveg_R, aveg_L, aveg_err, lim_R = arr_jack_func( tmp_resL, tmp_R, N_sample )
		keys = [ 'r', 'cumu_L', 'cumu_Lerr' ]
		values = [ kk_r, aveg_L, aveg_err ]
		fill = dict( zip( keys, values ) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/trunF_mod/' + 
					'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_BG-sub-SB_aveg_model-resi-cumu-L.csv'
					% (R_bins[tt], R_bins[tt+1], band_str),)

	return

# cumuL_resiL_estimate()
# raise


### === data load and figs
##. sat_img without BCG
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/'
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_SBs/'

bin_rich = [ 20, 30, 50, 210 ]

R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m

##. background shuffle list order
list_order = 13

band_str = 'r'


##.. luminosity compare
tot_R, tot_L, tot_Lerr = [], [], []

for tt in range( len(R_bins) - 1 ):

	#.
	dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/' + 
					'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_BG-sub-SB_aveg-cumuL.csv' 
					% (R_bins[tt], R_bins[tt + 1], band_str),)

	tt_r, tt_L, tt_L_err = np.array( dat['r'] ), np.array( dat['cumu_L'] ), np.array( dat['cumu_Lerr'] )

	tot_R.append( tt_r )
	tot_L.append( tt_L )
	tot_Lerr.append( tt_L_err )

#.
res_R, res_L, res_Lerr = [], [], []

for tt in range( len(R_bins) - 2 ):

	#.
	dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/trunF_mod/' + 
					'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_BG-sub-SB_aveg_model-resi-cumu-L.csv'
					% (R_bins[tt], R_bins[tt+1], band_str),)

	tt_r, tt_L, tt_L_err = np.array( dat['r'] ), np.array( dat['cumu_L'] ), np.array( dat['cumu_Lerr'] )

	res_R.append( tt_r )
	res_L.append( tt_L )
	res_Lerr.append( tt_L_err )

#.
out_R, out_L, out_Lerr = [], [], []

for tt in range( len(R_bins) - 2 ):

	#.
	dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/trunF_mod/' + 
					'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_BG-sub-SB_aveg_model-out-cumu-L.csv'
					% (R_bins[tt], R_bins[tt + 1], band_str),)

	tt_r, tt_L, tt_L_err = np.array( dat['r'] ), np.array( dat['cumu_L'] ), np.array( dat['cumu_Lerr'] )

	out_R.append( tt_r )
	out_L.append( tt_L )
	out_Lerr.append( tt_L_err )

#.
cen_R, cen_L, cen_Lerr = [], [], []

for tt in range( len(R_bins) - 2 ):

	#.
	dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/trunF_mod/' + 
					'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band_BG-sub-SB_aveg_model-cen-cumu-L.csv'
					% (R_bins[tt], R_bins[tt + 1], band_str),)

	tt_r, tt_L, tt_L_err = np.array( dat['r'] ), np.array( dat['cumu_L'] ), np.array( dat['cumu_Lerr'] )

	cen_R.append( tt_r )
	cen_L.append( tt_L )
	cen_Lerr.append( tt_L_err )


### === for figs
color_s = ['b', 'g', 'r', 'm', 'k']

line_s = [':', '-.', '--', '-']

fig_name = []

for dd in range( len(R_bins) - 1 ):

	if dd == 0:
		fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

	elif dd == len(R_bins) - 2:
		fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

	else:
		fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)


### === ### figs of Luminosity
for tt in range( len(R_bins) - 2 ):

	##.
	f_act = 5

	fig = plt.figure( figsize = (10.8, 4.8) )
	ax0 = fig.add_axes([0.08, 0.10, 0.40, 0.80])
	ax1 = fig.add_axes([0.56, 0.10, 0.40, 0.80])

	ax0.plot( tot_R[-1], tot_L[-1], ls = '-', color = 'k', alpha = 0.75, label = 'Total (Observed)')
	ax0.fill_between( tot_R[-1], y1 = tot_L[-1] - tot_Lerr[-1], y2 = tot_L[-1] + tot_Lerr[-1], color = 'k', alpha = 0.25,)

	ax0.plot( cen_R[tt], cen_L[tt], ls = '--', color = 'r', alpha = 0.75, label = 'Truncated (model)')
	ax0.fill_between( cen_R[tt], y1 = cen_L[tt] - cen_Lerr[tt], y2 = cen_L[tt] + cen_Lerr[tt], color = 'r', alpha = 0.25,)

	ax0.plot( out_R[tt], f_act * out_L[tt], ls = '--', color = 'g', alpha = 0.75, label = '$5 \\times \, $Outer model')
	ax0.fill_between( out_R[tt], y1 = f_act * out_L[tt] - out_Lerr[tt], 
					y2 = f_act * out_L[tt] + out_Lerr[tt], color = 'g', alpha = 0.25,)

	ax0.plot( res_R[tt], res_L[tt], ls = '--', color = 'b', alpha = 0.75, label = 'Total - Truncated')
	ax0.fill_between( res_R[tt], y1 = res_L[tt] - res_Lerr[tt], 
					y2 = res_L[tt] + res_Lerr[tt], color = 'b', alpha = 0.25,)

	ax0.legend( loc = 4, frameon = False, fontsize = 11, markerfirst = False,)

	ax0.set_xscale('log')
	ax0.set_xlim( 3e0, 1e2 )
	ax0.set_xlabel('R [kpc]', fontsize = 12,)

	ax0.set_ylim( 1e0, 4e1 )
	ax0.set_ylabel('$L \; [ nanomaggy ] $', fontsize = 12,)
	ax0.set_yscale('log')

	ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	##.
	_tt_F = interp.splrep( tot_R[-1], tot_L[-1], s = 0)

	_tt_L_out = interp.splev( out_R[tt], _tt_F, der = 0)
	_tt_L_cen = interp.splev( cen_R[tt], _tt_F, der = 0)
	_tt_L_res = interp.splev( res_R[tt], _tt_F, der = 0)


	##.
	ax1.plot( out_R[tt], f_act * out_L[tt] / _tt_L_out, ls = '--', color = 'g', alpha = 0.75,
			label = '$5 \\times \, $Outer (Model)',)
	ax1.fill_between( out_R[tt], y1 = ( f_act * out_L[tt] - out_Lerr[tt] ) / _tt_L_out, 
			y2 = ( f_act * out_L[tt] + out_Lerr[tt] ) / _tt_L_out, color = 'g', alpha = 0.25,)

	ax1.plot( res_R[tt], res_L[tt] / _tt_L_res, ls = '--', color = 'b', alpha = 0.75,
			label = 'Total - Truncated',)
	ax1.fill_between( res_R[tt], y1 = ( res_L[tt] - res_Lerr[tt] ) / _tt_L_res, 
			y2 = ( res_L[tt] + res_Lerr[tt] ) / _tt_L_res, color = 'b', alpha = 0.25,)

	ax1.plot( cen_R[tt], cen_L[tt] / _tt_L_cen, ls = '--', color = 'r', alpha = 0.75,
			label = 'Truncated (model)',)
	ax1.fill_between( cen_R[tt], y1 = ( cen_L[tt] - cen_Lerr[tt] ) / _tt_L_cen, 
			y2 = ( cen_L[tt] + cen_Lerr[tt] ) / _tt_L_cen, color = 'r', alpha = 0.25,)

	ax1.legend( loc = 4, frameon = False, fontsize = 12, markerfirst = False, )
	ax1.annotate( s = fig_name[tt], xy = (0.55, 0.35), xycoords = 'axes fraction', fontsize = 12,)

	ax1.set_xscale('log')
	ax1.set_xlim( 3e0, 1e2 )
	ax1.set_xlabel('R [kpc]', fontsize = 12,)

	ax1.set_ylim( 1e-3, 1e0 )
	ax1.set_yscale('log')
	ax1.set_ylabel('$L \, / \, L_{Total}$', fontsize = 12,)

	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig('/home/xkchen/Sat_%.2f-%.2fR200m_%s-band_cumu_L_compare.png' 
				% (R_bins[tt], R_bins[tt + 1], band_str), dpi = 300)
	plt.close()

