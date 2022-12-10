"""
mass loss compare with ICL
"""
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
band = ['r', 'g', 'i']

z_ref = 0.25
Da_ref = Test_model.angular_diameter_distance( z_ref ).value
Dl_ref = Test_model.luminosity_distance( z_ref ).value
a_ref = 1 / (z_ref + 1)


### === func.s
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

	qq = 0  ##. 0, 1

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

			tmp_R.append( ang_r )
			tmp_trunL.append( cumu_lx )

		#.
		aveg_R, aveg_L, aveg_err, lim_R = arr_jack_func( tmp_trunL, tmp_R, N_sample )
		keys = [ 'r', 'cumu_L', 'cumu_Lerr' ]
		values = [ kk_r, aveg_L, aveg_err ]
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
			tmp_R.append( ang_r )
			tmp_resL.append( cumu_lx )

		#.
		aveg_R, aveg_L, aveg_err, lim_R = arr_jack_func( tmp_resL, tmp_R, N_sample )
		keys = [ 'r', 'cumu_L', 'cumu_Lerr' ]
		values = [ kk_r, aveg_L, aveg_err ]
		fill = dict( zip( keys, values ) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/SB_model/' + 
					'%s_clust_%.2f-%.2fR200m_%s-band_BG-sub-SB_aveg_model-resi-cumu-L.csv'
					% (cat_lis[qq], R_bins[tt], R_bins[tt + 1], band_str),)

	return


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
		_tt_F

		d_gr = nbg_g_SB[ll] - nbg_r_SB[ll]
		d_gr_err = np.sqrt( nbg_g_err[ll]**2 + nbg_r_err[ll]**2 )

		#.
		keys = ['r', 'g-r', 'g-r_err']
		values = [ nbg_r_R, d_gr, d_gr_err ]
		fill = dict( zip( keys, values ) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_path + '%s_clust_%.2f-%.2fR200m_BG-sub-SB_aveg_g-r.csv'
					% (cat_lis[qq], R_bins[tt], R_bins[tt + 1]),)

	##. r-i
	for ll in range( len(R_bins) - 1 ):

		d_ri = nbg_r_SB[ll] - nbg_i_SB[ll]
		d_ri_err = np.sqrt( nbg_r_err[ll]**2 + nbg_i_err[ll]**2 )

		#.
		keys = ['r', 'r-i', 'r-i_err']
		values = [ kk_r, aveg_L, aveg_err ]
		fill = dict( zip( keys, values ) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_path + '%s_clust_%.2f-%.2fR200m_BG-sub-SB_aveg_r-i.csv'
					% (cat_lis[qq], R_bins[tt], R_bins[tt + 1]),)

	return


# cumuL_resiL_estimate()
sat_color_profs()
raise


### === ### satellite data load
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/nobcg_BGsub_SBs/'

bin_rich = [ 20, 30, 50, 210 ]

R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

band_str = 'r'

list_order = 13


##... figs set
color_s = ['b', 'g', 'r', 'm', 'k']

fig_name = []

for dd in range( len(R_bins) - 1 ):

	if dd == 0:
		fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

	elif dd == len(R_bins) - 2:
		fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

	else:
		fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)

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


##.. mass-to-light ratio formular
load_path = '/home/xkchen/figs/extend_bcgM_cat/Mass_Li_fit/'

#... all cluster sample
fit_dat = pds.read_csv( load_path + 'least-square_M-to-i-band-Lumi&color.csv' )

all_a_, all_b_ = np.array( fit_dat['a'])[0], np.array( fit_dat['b'])[0]
all_c_, all_d_ = np.array( fit_dat['c'])[0], np.array( fit_dat['d'])[0]


##... surface mass density profiles of satellite 
tmp_mR, tmp_SM = [], []


##. figs
fig = plt.figure( figsize = (10.8, 4.8) )
ax0 = fig.add_axes([0.07, 0.31, 0.42, 0.63])
sub_ax0 = fig.add_axes([0.07, 0.10, 0.42, 0.21])
ax1 = fig.add_axes([0.57, 0.10, 0.42, 0.84])	


plt.savefig('/home/xkchen/BCGM_sat_color.png', dpi = 300)
plt.close()



### === ### ICL data load

###... ICL surface mass load


###... transition component model


