"""
mass loss compare with ICL
"""
import sys
sys.path.append('/home/xkchen/tool/Conda/Tools/normitems')

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import h5py
import numpy as np
import pandas as pds

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc
import scipy.interpolate as interp

from astropy import modeling as Model
from pynverse import inversefunc
from astropy import cosmology as apcy
from scipy import optimize
from scipy.stats import binned_statistic as binned
from scipy import integrate as integ
from scipy import signal
from scipy import stats as sts
from astropy.coordinates import SkyCoord

#. dust map with the recalibration by Schlafly & Finkbeiner (2011)
import sfdmap
E_map = sfdmap.SFDMap('/home/xkchen/tool/Conda/dustmap/sfddata/')
from extinction_redden import A_wave

#.
from img_sat_fig_out_mode import arr_jack_func
from img_sat_BG_sub_SB import SB_to_Lumi_func
from img_sat_BG_sub_SB import color_func
from img_sat_BG_sub_SB import absMag_to_Lumi_func



##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)

#. parameter for Galactic
Rv = 3.1

##### constant
band = ['r', 'g', 'i']
L_wave = np.array( [ 6166, 4686, 7480 ] )
Mag_sun = [ 4.65, 5.11, 4.53 ]
pixel = 0.396

##.
z_ref = 0.25
Da_ref = Test_model.angular_diameter_distance( z_ref ).value
Dl_ref = Test_model.luminosity_distance( z_ref ).value
a_ref = 1 / (z_ref + 1)


### === func.s
##. Lr -- > Mass
def Lr_to_Mass_fun( Lr_arr ):

	lg_Lr = np.log10( Lr_arr )
	lg_M = 0.536 + 0.996 * lg_Lr

	return 10**lg_M

##. core-like funcs
def Moffat_func(R, A0, Rd, n):
	mf = A0 / ( 1 + (R / Rd)**2 )**n
	return mf

def Nuker_func( R, A0, R_bk, alpha, belta, gamma):

	mf0 = 2**( (belta - gamma) / alpha )
	mf1 = mf0 / ( R / R_bk)**gamma

	mf2 = 1 + ( R / R_bk  )**alpha
	mf3 = mf2**( (gamma - belta) / alpha )

	mf = mf1 * mf3

	return mf * A0

def KPA_func( R, A0, R_c0, R_c1 ):

	mod_F = Model.functional_models.KingProjectedAnalytic1D( amplitude = A0, r_core = R_c0, r_tide = R_c1)
	mf = mod_F( R )

	return mf


### === central and residuial Luminosity
def cumuL_resiL_estimate():

	#.
	out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/nobcg_BGsub_SBs/'

	bin_rich = [ 20, 30, 50, 210 ]

	R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

	cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

	#.
	N_sample = 100

	band_str = 'r'

	qq = 1  ##. 0, 1

	##... BG-subtracted SB profiles
	nbg_R, nbg_SB, nbg_err = [], [], []

	for ll in range( len(R_bins) - 1 ):

		#.
		dat = pds.read_csv( out_path + '%s_clust_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub_SB.csv' 
						% (cat_lis[qq], R_bins[ll], R_bins[ll+1], band_str),)

		tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

		nbg_R.append( tt_r )
		nbg_SB.append( tt_sb )
		nbg_err.append( tt_sb_err )


	##... interp the outer radius bin
	cp_k_r = nbg_R[ -1 ]
	cp_k_sb = nbg_SB[ -1 ]
	cp_k_err = nbg_err[ -1 ]
	tmp_F0 = interp.splrep( cp_k_r, cp_k_sb, s = 0)

	##...
	for tt in range( len(R_bins) - 2 ):
	# for tt in range( 3, 4 ):

		print( 'tt=', tt )
		print( '*' * 10 )

		#.
		kk_r = nbg_R[ tt ]
		kk_sb = nbg_SB[ tt ]
		kk_err = nbg_err[ tt ]

		#.
		ang_r = rad2arcsec * kk_r / ( Da_ref * 1e3 )

		#.
		dat = pds.read_csv(	'/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/SB_model/' + 
							'%s_sat_%.2f-%.2fR200m_%s-band_SB_inner-KPA_fit.csv' 
							% (cat_lis[qq], R_bins[tt], R_bins[tt + 1], band_str),)

		A0_fit = np.array( dat['A_power'] )[0]
		Rc0_fit = np.array( dat['R_core'] )[0]
		Rc1_fit = np.array( dat['R_tidal'] )[0]

		#.
		mf0 = KPA_func( kk_r, A0_fit, Rc0_fit, Rc1_fit )
		_SB0_arr = interp.splev( kk_r, tmp_F0, der = 0)

		##...
		tmp_R, tmp_trunL = [], []

		for dd in range( N_sample ):

			dat = pds.read_csv( out_path + 
					'%s_clust_%.2f-%.2fR200m_%s-band_jack-sub-%d_BG-sub-SB-pro_z-ref.h5' 
					% (cat_lis[qq], R_bins[-2], R_bins[-1], band_str, dd),)

			tt_r = np.array( dat['r'])
			tt_sb = np.array( dat['sb'])

			#.
			id_nan = np.isnan( tt_sb )
			dt_r, dt_sb = tt_r[ id_nan == False ], tt_sb[ id_nan == False ]

			tmp_F = interp.interp1d( dt_r, dt_sb, kind = 'linear', fill_value = 'extrapolate')

			cen_L = mf0 * tmp_F( kk_r )

			#.
			cumu_lx = integ.cumtrapz( ang_r * cen_L, x = ang_r, initial = np.min( ang_r ) / 10 )
			cumu_lx = cumu_lx * np.pi * 2

			#.
			mag_Lx = 22.5 - 2.5 * np.log10( cumu_lx )
			Mag_Lx = mag_Lx - 5 * np.log10( Dl_ref * 1e6 ) + 5

			cumu_Lx = absMag_to_Lumi_func( Mag_Lx, band_str )  ##. L_sun

			tmp_R.append( ang_r )
			tmp_trunL.append( cumu_Lx )

		#.
		aveg_R, aveg_L, aveg_err, lim_R = arr_jack_func( tmp_trunL, tmp_R, N_sample )

		aveg_phy_R = aveg_R * Da_ref * 1e3 / rad2arcsec

		keys = [ 'r', 'cumu_L', 'cumu_Lerr' ]
		values = [ aveg_phy_R, aveg_L, aveg_err ]
		fill = dict( zip( keys, values ) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/SB_model/' + 
					'%s_clust_%.2f-%.2fR200m_%s-band_BG-sub-SB_aveg_model-cen-cumu-L.csv'
					% (cat_lis[qq], R_bins[tt], R_bins[tt + 1], band_str),)

		##... 
		tmp_R, tmp_resL = [], []

		for dd in range( N_sample ):

			dat = pds.read_csv( out_path + 
					'%s_clust_%.2f-%.2fR200m_%s-band_jack-sub-%d_BG-sub-SB-pro_z-ref.h5' 
					% (cat_lis[qq], R_bins[-2], R_bins[-1], band_str, dd),)

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
			mag_Lx = 22.5 - 2.5 * np.log10( cumu_lx )
			Mag_Lx = mag_Lx - 5 * np.log10( Dl_ref * 1e6 ) + 5

			cumu_Lx = absMag_to_Lumi_func( Mag_Lx, band_str )  ##. L_sun

			#.
			tmp_R.append( ang_r )
			tmp_resL.append( cumu_Lx )

		#.
		aveg_R, aveg_L, aveg_err, lim_R = arr_jack_func( tmp_resL, tmp_R, N_sample )

		aveg_phy_R = aveg_R * Da_ref * 1e3 / rad2arcsec

		keys = [ 'r', 'cumu_L', 'cumu_Lerr' ]
		values = [ aveg_phy_R, aveg_L, aveg_err ]
		fill = dict( zip( keys, values ) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/SB_model/' + 
					'%s_clust_%.2f-%.2fR200m_%s-band_BG-sub-SB_aveg_model-resi-cumu-L.csv'
					% (cat_lis[qq], R_bins[tt], R_bins[tt + 1], band_str), )

	return

#.
def sat_color_profs():

	#.
	out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/nobcg_BGsub_SBs/'

	bin_rich = [ 20, 30, 50, 210 ]

	R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

	cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

	qq = 0  ##. 0, 1

	##... BG-subtracted SB profiles
	nbg_r_R, nbg_r_SB, nbg_r_err = [], [], []

	for ll in range( len(R_bins) - 1 ):

		#.
		dat = pds.read_csv( out_path + '%s_clust_%.2f-%.2fR200m_r-band_aveg-jack_BG-sub_SB.csv' 
						% (cat_lis[qq], R_bins[ll], R_bins[ll+1]),)

		tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

		nbg_r_R.append( tt_r )
		nbg_r_SB.append( tt_sb )
		nbg_r_err.append( tt_sb_err )

	#.
	nbg_g_R, nbg_g_SB, nbg_g_err = [], [], []

	for ll in range( len(R_bins) - 1 ):

		#.
		dat = pds.read_csv( out_path + '%s_clust_%.2f-%.2fR200m_g-band_aveg-jack_BG-sub_SB.csv' 
						% (cat_lis[qq], R_bins[ll], R_bins[ll+1]),)

		tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

		nbg_g_R.append( tt_r )
		nbg_g_SB.append( tt_sb )
		nbg_g_err.append( tt_sb_err )

	#.
	nbg_i_R, nbg_i_SB, nbg_i_err = [], [], []

	for ll in range( len(R_bins) - 1 ):

		#.
		dat = pds.read_csv( out_path + '%s_clust_%.2f-%.2fR200m_i-band_aveg-jack_BG-sub_SB.csv' 
						% (cat_lis[qq], R_bins[ll], R_bins[ll+1]),)

		tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

		nbg_i_R.append( tt_r )
		nbg_i_SB.append( tt_sb )
		nbg_i_err.append( tt_sb_err )

	##. g-r
	for ll in range( len(R_bins) - 1 ):

		#.
		d_gr, d_gr_err = color_func( nbg_g_SB[ll], nbg_g_err[ll], nbg_r_SB[ll], nbg_r_err[ll] )

		#.
		keys = ['r', 'g-r', 'g-r_err']
		values = [ nbg_r_R[ll], d_gr, d_gr_err ]
		fill = dict( zip( keys, values ) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_path + '%s_clust_%.2f-%.2fR200m_BG-sub-SB_aveg_g-r.csv'
					% (cat_lis[qq], R_bins[ll], R_bins[ll + 1]),)

	##. r-i
	for ll in range( len(R_bins) - 1 ):

		#.
		d_ri, d_ri_err = color_func( nbg_r_SB[ll], nbg_r_err[ll], nbg_i_SB[ll], nbg_i_err[ll] )

		#.
		keys = ['r', 'r-i', 'r-i_err']
		values = [ nbg_r_R[ll], d_ri, d_ri_err ]
		fill = dict( zip( keys, values ) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_path + '%s_clust_%.2f-%.2fR200m_BG-sub-SB_aveg_r-i.csv'
					% (cat_lis[qq], R_bins[ll], R_bins[ll + 1]),)

	return

#.
def extinction_factor():

	#.
	cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/cat/'

	#. fixed R for all richness subsample
	R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

	#.
	bin_rich = [ 20, 30, 50, 210 ]

	cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

	#.
	for pp in range( 2 ):

		for nn in range( len( R_bins ) - 1 ):

			dat = pds.read_csv( cat_path + '%s_clust_frame-lim_Pm-cut_rich_20-30_%.2f-%.2fR200m_mem_cat.csv' 
							% (cat_lis[pp], R_bins[nn], R_bins[nn + 1]),)

			bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
			sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
			R_sat, R2Rv = np.array( dat['R_sat'] ), np.array( dat['R2Rv'] )

			clust_ID = np.array( dat['clus_ID'] )

			#.
			sky_pos = SkyCoord( bcg_ra, bcg_dec, unit = 'deg')

			p_EBV = E_map.ebv( sky_pos )
			A_v = Rv * p_EBV

			A_r = A_wave( L_wave[0], Rv) * A_v
			A_g = A_wave( L_wave[1], Rv) * A_v
			A_i = A_wave( L_wave[2], Rv) * A_v

			#.
			keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'E_bv', 'A_r', 'A_g', 'A_i']
			values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, p_EBV, A_r, A_g, A_i ]
			fill = dict(zip( keys, values) )
			out_data = pds.DataFrame( fill )
			out_data.to_csv( cat_path + 
					'%s_clust_frame-lim_Pm-cut_rich_20-30_%.2f-%.2fR200m_mem_dust_value.csv' 
					% (cat_lis[pp], R_bins[nn], R_bins[nn + 1]),)

	return

#.
def surface_luminosity():

	#.
	out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/nobcg_BGsub_SBs/'

	bin_rich = [ 20, 30, 50, 210 ]

	R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

	cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

	#.
	N_sample = 100

	band_str = 'r'

	qq = 0  ##. 0, 1

	##...
	nbg_R, nbg_SB, nbg_err = [], [], []

	for ll in range( len(R_bins) - 1 ):

		#.
		dat = pds.read_csv( out_path + '%s_clust_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub_SB.csv' 
						% (cat_lis[qq], R_bins[ll], R_bins[ll+1], band_str),)

		tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

		nbg_R.append( tt_r )
		nbg_SB.append( tt_sb )
		nbg_err.append( tt_sb_err )

	#.
	for tt in range( len(R_bins) - 1 ):

		#.
		kk_r = nbg_R[ tt ]

		ang_r = rad2arcsec * kk_r / ( Da_ref * 1e3 )

		tmp_R, tmp_L = [], []

		for dd in range( N_sample ):

			#.
			dat = pds.read_csv( out_path + 
					'%s_clust_%.2f-%.2fR200m_%s-band_jack-sub-%d_BG-sub-SB-pro_z-ref.h5' 
					% (cat_lis[qq], R_bins[tt], R_bins[tt+1], band_str, dd), )

			tt_r = np.array( dat['r'])
			tt_sb = np.array( dat['sb'])

			#.
			id_nan = np.isnan( tt_sb )
			dt_r, dt_sb = tt_r[ id_nan == False ], tt_sb[ id_nan == False ]
			
			tmp_F = interp.interp1d( dt_r, dt_sb, kind = 'linear', fill_value = 'extrapolate')

			#.
			mag_Lx = 22.5 - 2.5 * np.log10( tmp_F( kk_r ) )
			Mag_Lx = mag_Lx - 5 * np.log10( Dl_ref * 1e6 ) + 5

			tt_Lx = SB_to_Lumi_func( Mag_Lx, z_ref, band_str )   ##. L_sun / pc^2
			tt_Lx = tt_Lx * 1e6

			#.
			tmp_R.append( ang_r )
			tmp_L.append( tt_Lx )

		#.
		aveg_R, aveg_L, aveg_err, lim_R = arr_jack_func( tmp_L, tmp_R, N_sample )

		aveg_phy_R = aveg_R * Da_ref * 1e3 / rad2arcsec

		#.
		keys = [ 'r', 'L_r', 'Lr_err' ]
		values = [ aveg_phy_R, aveg_L, aveg_err ]
		fill = dict( zip( keys, values ) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_path + 
					'%s_clust_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub-SB_Lumi.csv'
					% (cat_lis[qq], R_bins[tt], R_bins[tt+1], band_str),)

	return


##...
# cumuL_resiL_estimate()
# sat_color_profs()
# extinction_factor()
# surface_luminosity()

# raise



### === ### satellite data load
bin_rich = [ 20, 30, 50, 210 ]

R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

band_str = 'r'

list_order = 13

#.
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
samp_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']


##... radius catalog
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/cat/'

##. galaxy catalog information
R_sat = []
aveg_Rsat = []

for qq in range( 2 ):

	R_aveg = []
	R_sat_arr = []

	for tt in range( len(R_bins) - 1 ):

		cat = pds.read_csv( cat_path + '%s_clust_frame-lim_Pm-cut_%.2f-%.2fR200m_mem_cat.csv' 
						% (cat_lis[qq], R_bins[tt], R_bins[tt + 1]),)

		x_Rs = np.array( cat['R_sat'] )   ##. Mpc / h
		x_Rs = x_Rs * 1e3 / h

		#.
		R_aveg.append( np.median( x_Rs ) )
		R_sat_arr.append( x_Rs )

	#.
	R_sat.append( R_sat_arr )
	aveg_Rsat.append( R_aveg )

##. dust vaslue
Ar_arr = []
aveg_Ar = []

for pp in range( 2 ):

	dd_Ar = []
	dd_aveg_Ar = []

	for nn in range( len( R_bins ) - 1 ):

		dat = pds.read_csv( cat_path + 
				'%s_clust_frame-lim_Pm-cut_rich_20-30_%.2f-%.2fR200m_mem_dust_value.csv' 
				% (cat_lis[pp], R_bins[nn], R_bins[nn + 1]),)

		_Ar_ = np.array( dat['A_r'] )

		dd_Ar.append( _Ar_ )
		dd_aveg_Ar.append( np.median( _Ar_ ) )

	#.
	Ar_arr.append( dd_Ar )
	aveg_Ar.append( dd_aveg_Ar )


###... BCG+ICL surface mass load
dat = pds.read_csv( '/home/xkchen/figs/re_measure_SBs/SM_profile/' + '%s_gri-band-based_corrected_aveg-jack_mass-Lumi.csv' % cat_lis[0] )
lo_R, lo_surf_M, lo_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )

dat = pds.read_csv( '/home/xkchen/figs/re_measure_SBs/SM_profile/' + '%s_gri-band-based_corrected_aveg-jack_mass-Lumi.csv' % cat_lis[1] )
hi_R, hi_surf_M, hi_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )


###... ICL model profile
pat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/SM_pros_fit/' + 'BCGM_binned_gri-band-based_model-ICL_SM__with-dered.csv',)

mod_R = np.array( pat['r'] )
lo_ICL_M = np.array( pat['low_BCGM_BGsub_ICL'] )
hi_ICL_M = np.array( pat['high_BCGM_BGsub_ICL'] )

ICL_SM = [ lo_ICL_M, hi_ICL_M ] 


### === figs set
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/nobcg_BGsub_SBs/'
fit_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/SB_model/'

#.
color_s = ['b', 'g', 'r', 'm', 'k']

fig_name = []

for dd in range( len(R_bins) - 1 ):

	if dd == 0:
		fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

	elif dd == len(R_bins) - 2:
		fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

	else:
		fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)


##. surface brightness
for qq in range( 2 ):

	#.
	fig = plt.figure()
	ax = fig.add_axes( [ 0.12, 0.11, 0.85, 0.80 ] )

	for ll in range( len(R_bins) - 1 ):

		#.
		dat = pds.read_csv( out_path + '%s_clust_%.2f-%.2fR200m_r-band_aveg-jack_BG-sub_SB.csv' 
						% (cat_lis[qq], R_bins[ll], R_bins[ll+1]),)

		tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

		#.
		ax.plot( tt_r, tt_sb, ls = '-', color = color_s[ll], alpha = 0.75, label = fig_name[tt],)
		ax.fill_between( tt_r, y1 = tt_sb - tt_sb_err, y2 = tt_sb + tt_sb_err, color = color_s[ll], alpha = 0.15,)

	ax.legend( loc = 3, frameon = False, fontsize = 13,)
	ax.annotate( s = samp_name[qq], xy = (0.65, 0.85), xycoords = 'axes fraction', fontsize = 13,)

	ax.set_xlim( 1e0, 1e2 )
	ax.set_xscale('log')
	ax.set_xlabel('$R \; [kpc]$', fontsize = 13)

	ax.set_ylim( 1e-3, 5e0 )
	ax.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax.set_yscale('log')

	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)

	plt.savefig('/home/xkchen/%s_clust_r-band_SB.png' % cat_lis[qq], dpi = 300)
	plt.close()


##. surface mass
for pp in range( 2 ):

	#.
	fig = plt.figure()
	ax = fig.add_axes( [ 0.12, 0.11, 0.85, 0.80 ] )

	for tt in range( len( R_bins ) - 1 ):

		dat = pds.read_csv( out_path + 
				'%s_clust_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub-SB_Lumi.csv'
				% (cat_lis[pp], R_bins[tt], R_bins[tt+1], band_str),)

		tt_R = np.array( dat['r'] )
		tt_L = np.array( dat['L_r'] )
		tt_Lerr = np.array( dat['Lr_err'] )

		#.
		ax.plot( tt_R, tt_L, ls = '-', color = color_s[tt], alpha = 0.75, label = fig_name[tt],)
		ax.fill_between( tt_R, y1 = tt_L - tt_Lerr, y2 = tt_L + tt_Lerr, color = color_s[tt], alpha = 0.15,)

	ax.legend( loc = 3, frameon = False, fontsize = 13,)
	ax.annotate( s = samp_name[pp], xy = (0.65, 0.85), xycoords = 'axes fraction', fontsize = 13,)

	ax.set_xlim( 1e0, 1e2 )
	ax.set_xscale('log')
	ax.set_xlabel('$R \; [kpc]$', fontsize = 13)

	ax.set_ylim( 3e3, 4e7 )
	ax.set_ylabel('$\\Sigma_{\\ast} \; [M_{\\odot} \, / \, kpc^{2} ]$', fontsize = 13,)
	ax.set_yscale( 'log' )

	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)

	plt.savefig('/home/xkchen/%s_clust_%s-band-based_SM.png' % (cat_lis[pp], band_str), dpi = 300)
	plt.close()

raise


#.
for pp in range( 2 ):

	for nn in range( len( R_bins ) - 2 ):

		#.
		dat = pds.read_csv( fit_path + 
			'%s_clust_%.2f-%.2fR200m_%s-band_BG-sub-SB_aveg_model-resi-cumu-L.csv'
			% (cat_lis[pp], R_bins[nn], R_bins[nn + 1], band_str),)

		tt_R = np.array( dat['r'] )
		tt_L = np.array( dat['cumu_L'] )
		tt_L_err = np.array( dat['cumu_Lerr'] )

		#.
		tt_L2M = Lr_to_Mass_fun( tt_L )
		tt_Merr = Lr_to_Mass_fun( tt_L_err )

		#. corresponding R_sat
		tt_Rsat = R_sat[pp][ nn ]

		##. integral mass of ICL for different R_sat hist limit
		sum_ICL = []
		lim_Rx = []

		lo_limt = [ 0.05, 0.16, 0.25 ]
		hi_limt = [ 0.95, 0.84, 0.75 ]

		for dd in range( 3 ):

			dx_0, dx_1 = np.quantile( tt_Rsat, [ lo_limt[dd], hi_limt[dd] ] )

			#. icl mass
			id_vx = ICL_SM[pp] > 0.

			icl_F = interp.splrep( mod_R[id_vx], ICL_SM[pp][id_vx], s = 0 )

			new_x = np.logspace( np.log10(dx_0), np.log10(dx_1), 200 )

			icl_sm = interp.splev( new_x, icl_F, der = 0 )

			dd_sum = integ.simps( new_x**2 * np.log(10) * icl_sm * 2 * np.pi, np.log10( new_x ) )

			sum_ICL.append( dd_sum )
			lim_Rx.append( [ dx_0, dx_1 ] )

		#.
		fig = plt.figure()
		ax = fig.add_axes( [0.13, 0.74, 0.85, 0.21] )
		sub_ax = fig.add_axes( [0.13, 0.11, 0.85, 0.63] )

		x_bins = np.logspace( np.log10( tt_Rsat.min() ), np.log10( tt_Rsat.max() ), 65 )

		ax.hist( tt_Rsat, bins = x_bins, density = True, histtype = 'step', color = 'b',)
		sub_ax.plot( mod_R, ICL_SM[pp], ls = '-', color = 'r', alpha = 0.75,)

		ax.set_xlim( tt_Rsat.min() - 2, 1.5e3 )
		ax.set_xscale('log')
		ax.set_xlabel('$R \; [kpc]$', fontsize = 13)
		ax.set_ylabel('pdf', fontsize = 13)

		# ax.annotate( s = samp_name[pp] + '\n' + fig_name[nn], xy = (0.55, 0.10), xycoords = 'axes fraction', fontsize = 13,)
		sub_ax.annotate( s = samp_name[pp] + '\n' + fig_name[nn], xy = (0.60, 0.45), xycoords = 'axes fraction', fontsize = 13,)

		sub_ax.set_xlim( ax.get_xlim() )
		sub_ax.set_xscale('log')
		sub_ax.set_xlabel('$R \; [kpc]$', fontsize = 13)

		sub_ax.set_ylim( 1e4, 5e6 )
		sub_ax.set_yscale('log')
		sub_ax.set_ylabel('$\\Sigma_{\\ast}^{\\rm{ICL}} \; [M_{\\odot} \, / \, kpc^{2}]$', fontsize = 13,)

		#.
		for dd in range( 3 ):

			ax.axvline( lim_Rx[dd][0], ls = ':', color = color_s[dd], lw = 2, alpha = 0.65,)
			ax.axvline( lim_Rx[dd][1], ls = ':', color = color_s[dd], lw = 2, alpha = 0.65,)

			sub_ax.axvline( lim_Rx[dd][0], ls = ':', color = color_s[dd], lw = 2, alpha = 0.65, 
					label = '[%.2f, %.2f]' % (lo_limt[dd], hi_limt[dd]), )

			sub_ax.axvline( lim_Rx[dd][1], ls = ':', color = color_s[dd], lw = 2, alpha = 0.65,)

		sub_ax.legend( loc = 1, fontsize = 13,)

		ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)
		ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
		sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)

		ax.set_xticklabels( [] )

		plt.savefig('/home/xkchen/%s_clust_%.2f-%.2fR200m_ICL_interval.png'
				% (cat_lis[pp], R_bins[nn], R_bins[nn + 1] ), dpi = 300)
		plt.close()


		##.
		fig = plt.figure()
		ax = fig.add_axes( [ 0.12, 0.11, 0.85, 0.80 ] )

		ax.plot( tt_R, tt_L2M, ls = '-', color = 'k', alpha = 0.75, label = 'Mass loss of Satellite')
		ax.fill_between( tt_R, y1 = tt_L2M - tt_Merr, y2 = tt_L2M + tt_Merr, color = 'k', alpha = 0.15,)

		for dd in range( 3 ):

			ax.axhline( sum_ICL[dd], ls = '--', color = color_s[dd], alpha = 0.75, 
					label = 'ICL Mass [%.2f, %.2f]' % (lo_limt[dd], hi_limt[dd]),)

		ax.legend( loc = 4, frameon = False, fontsize = 13,)
		ax.annotate( s = samp_name[pp] + '\n' + fig_name[nn], xy = (0.03, 0.50), xycoords = 'axes fraction', fontsize = 13,)

		ax.set_xlim( 1e0, 1e2 )
		ax.set_xscale('log')
		ax.set_xlabel('$R \; [kpc]$', fontsize = 13)

		ax.set_ylim( 1e9, 3e11 )
		ax.set_ylabel('$M_{\\ast} \; [M_{\\odot}]$', fontsize = 13,)
		ax.set_yscale( 'log' )

		ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)

		plt.savefig('/home/xkchen/%s_clust_%.2f-%.2fR200m_%s-band-based_mass-losss.png'
					% (cat_lis[pp], R_bins[nn], R_bins[nn + 1], band_str), dpi = 300)
		plt.close()

