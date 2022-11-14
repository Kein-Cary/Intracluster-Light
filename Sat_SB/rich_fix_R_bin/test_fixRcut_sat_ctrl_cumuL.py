"""
compare the cumulative Luminosity of satellite and control galaxy
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
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc

from scipy import interpolate as interp
from scipy import integrate as integ
from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord

#.
from light_measure import light_measure_weit
from img_sat_fig_out_mode import arr_jack_func
from img_sat_BG_sub_SB import aveg_BG_sub_func, stack_BG_sub_func


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

Da_ref = Test_model.angular_diameter_distance( z_ref ).value  ## Mpc.


### === ### data load
##... pre-cumu_L calculation
def cumu_L_fig():

	out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/noBG_SBs/'
	cp_out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/'

	R_str = 'scale'
	# R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m
	R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

	band_str = 'r'

	N_sample = 100

	#.
	id_ctrl = False
	# id_ctrl = True

	for tt in range( len(R_bins) - 1 ):

		##. average SB profiles
		if id_ctrl:
			dat = pds.read_csv( out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % 
								(R_bins[tt], R_bins[tt + 1], band_str) + '_aveg-jack_BG-sub_SB.csv',)

		else:
			dat = pds.read_csv( cp_out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) 
											+ '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

		pt_r, pt_sb = np.array( dat['r'] ), np.array( dat['sb'] )
		ang_r = rad2arcsec * pt_r / ( Da_ref * 1e3 )

		#.
		tmp_R, tmp_cumu_L = [], []

		for dd in range( N_sample ):

			if id_ctrl:
				sub_out_file = ( out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % 
							(R_bins[tt], R_bins[tt + 1], band_str) + '_jack-sub-%d_BG-sub-SB-pro_z-ref.h5' % dd,)[0]

			else:
				sub_out_file = ( cp_out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m_%s-band' % (R_bins[tt], R_bins[tt + 1], band_str) 
							+ '_jack-sub-%d_BG-sub-SB-pro_z-ref.h5' % dd,)[0]

			data = pds.read_csv( sub_out_file )

			tt_r = np.array( data['r'])
			tt_sb = np.array( data['sb'])
			tt_err = np.array( data['sb_err'])

			#.
			id_nan = np.isnan( tt_sb )

			dt_r, dt_sb = tt_r[ id_nan == False ], tt_sb[ id_nan == False ]

			tmp_F = interp.interp1d( dt_r, dt_sb, kind = 'linear', fill_value = 'extrapolate')

			cumu_lx = integ.cumtrapz( ang_r * tmp_F( pt_r ), x = ang_r, initial = np.min( ang_r ) / 10 )
			cumu_lx = cumu_lx * np.pi * 2

			##.
			tmp_R.append( ang_r )
			tmp_cumu_L.append( cumu_lx )

		##.
		aveg_R, aveg_L, aveg_err, lim_R = arr_jack_func( tmp_cumu_L, tmp_R, N_sample )

		#.
		keys = [ 'r', 'cumu_L', 'cumu_Lerr' ]
		values = [ pt_r, aveg_L, aveg_err ]
		fill = dict( zip( keys, values ) )
		out_data = pds.DataFrame( fill )

		if id_ctrl:
			out_data.to_csv( out_path + 
					'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band_BG-sub-SB_aveg-cumuL.csv' % 
					(R_bins[tt], R_bins[tt + 1], band_str),)

		else:
			out_data.to_csv( cp_out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) 
											+ '_%s-band_BG-sub-SB_aveg-cumuL.csv' % band_str)

	return


def cumu_L_with_BG():

	out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/SBs/'
	cp_out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_SBs/'

	R_str = 'scale'
	# R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m
	R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

	band_str = 'r'

	N_sample = 100

	#.
	# id_ctrl = True
	id_ctrl = False

	for tt in range( len(R_bins) - 1 ):

		##. average SB profiles
		if id_ctrl:

			with h5py.File( out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % 
				(R_bins[tt], R_bins[tt + 1], band_str) + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

				pt_r = np.array(f['r'])
				pt_sb = np.array(f['sb'])

		else:

			with h5py.File( cp_out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
									'_%s-band_Mean_jack_SB-pro_z-ref.h5' % band_str, 'r') as f:

				pt_r = np.array(f['r'])
				pt_sb = np.array(f['sb'])

		#.
		ang_r = rad2arcsec * pt_r / ( Da_ref * 1e3 )
		tmp_R, tmp_cumu_L = [], []

		for dd in range( N_sample ):

			if id_ctrl:
				with h5py.File( out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % 
					(R_bins[tt], R_bins[tt + 1], band_str) + '_jack-sub-%d_SB-pro_z-ref.h5' % dd, 'r') as f:

					tt_r = np.array(f['r'])
					tt_sb = np.array(f['sb'])
					tt_err = np.array(f['sb_err'])
					tt_npix = np.array(f['npix'])

				id_Nul = tt_npix < 1
				tt_r[ id_Nul ] = np.nan
				tt_sb[ id_Nul ] = np.nan
				tt_err[ id_Nul ] = np.nan

			else:
				with h5py.File( cp_out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
								'_%s-band_jack-sub-%d_SB-pro_z-ref.h5' % (band_str, dd), 'r') as f:

					tt_r = np.array(f['r'])
					tt_sb = np.array(f['sb'])
					tt_err = np.array(f['sb_err'])
					tt_npix = np.array(f['npix'])

				id_Nul = tt_npix < 1
				tt_r[ id_Nul ] = np.nan
				tt_sb[ id_Nul ] = np.nan
				tt_err[ id_Nul ] = np.nan

			#.
			id_nan = np.isnan( tt_sb )

			dt_r, dt_sb = tt_r[ id_nan == False ], tt_sb[ id_nan == False ]

			tmp_F = interp.interp1d( dt_r, dt_sb, kind = 'linear', fill_value = 'extrapolate')

			cumu_lx = integ.cumtrapz( ang_r * tmp_F( pt_r ), x = ang_r, initial = np.min( ang_r ) / 10 )
			cumu_lx = cumu_lx * np.pi * 2

			##.
			tmp_R.append( ang_r )
			tmp_cumu_L.append( cumu_lx )

		##.
		aveg_R, aveg_L, aveg_err, lim_R = arr_jack_func( tmp_cumu_L, tmp_R, N_sample )

		#.
		keys = [ 'r', 'cumu_L', 'cumu_Lerr' ]
		values = [ pt_r, aveg_L, aveg_err ]
		fill = dict( zip( keys, values ) )
		out_data = pds.DataFrame( fill )

		if id_ctrl:
			out_data.to_csv( out_path + 
					'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band_SB-with-BG_aveg-cumuL.csv' % 
					(R_bins[tt], R_bins[tt + 1], band_str),)

		else:
			out_data.to_csv( cp_out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) 
											+ '_%s-band_SB-with-BG_aveg-cumuL.csv' % band_str)

	return

# ##.
# cumu_L_fig()
# cumu_L_with_BG()
# raise


##...
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/BGs/'

cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/map_cat/'
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/SBs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/noBG_SBs/'

cp_cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/cat/'
cp_out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/'
cp_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_SBs/'


#.
bin_rich = [ 20, 30, 50, 210 ]

R_str = 'scale'
# R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m
R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

band_str = 'r'


##.
color_s = ['b', 'g', 'c', 'r', 'm']
line_s = [ ':', '--', '-' ]

fig_name = []
for dd in range( len(R_bins) - 1 ):

	if dd == 0:
		fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

	elif dd == len(R_bins) - 2:
		fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

	else:
		fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)


##... sample properties
tmp_r_cMag = []
tmp_r_mag = []

cp_r_cMag = []
cp_r_mag = []

for tt in range( len(R_bins) - 1 ):

	tt_cmag = np.array([])
	tt_mag = np.array([])

	dt_cmag = np.array([])
	dt_mag = np.array([])

	for ll in range( 3 ):

		#. member
		dat = fits.open( cp_cat_path + 
						'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_params.fits' % 
						(bin_rich[ll], bin_rich[ll + 1], R_bins[tt], R_bins[tt + 1]),)

		dat_arr = dat[1].data

		tt_cmag = np.r_[ tt_cmag, dat_arr['cModelMag_r'] ]
		tt_mag = np.r_[ tt_mag, dat_arr['modelMag_r'] ]

		#. control
		dat = fits.open( cat_path + 
						'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_rich_%.2f-%.2fR200m_r-band_cat.fits' 
						% (R_bins[tt], R_bins[tt + 1]),)

		dat_arr = dat[1].data

		dt_cmag = np.r_[ dt_cmag, dat_arr['cModelMag_r'] ]
		dt_mag = np.r_[ dt_mag, dat_arr['modelMag_r'] ]

	#.
	tmp_r_cMag.append( dt_cmag )
	tmp_r_mag.append( dt_mag )

	cp_r_cMag.append( tt_cmag )
	cp_r_mag.append( tt_mag )


##... before BG-subtraced
tmp_R, tmp_sb, tmp_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	with h5py.File( path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % 
		(R_bins[tt], R_bins[tt + 1], band_str) + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tmp_R.append( tt_r )
	tmp_sb.append( tt_sb )
	tmp_err.append( tt_err )

##.
cp_tmp_R, cp_tmp_sb, cp_tmp_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	with h5py.File( cp_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) + 
							'_%s-band_Mean_jack_SB-pro_z-ref.h5' % band_str, 'r') as f:

		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	cp_tmp_R.append( tt_r )
	cp_tmp_sb.append( tt_sb )
	cp_tmp_err.append( tt_err )


##... cumulative Luminosity
cumu_R, cumu_L, cumu_Lerr = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( path + 
			'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band_SB-with-BG_aveg-cumuL.csv' % 
			(R_bins[tt], R_bins[tt + 1], band_str),)

	tt_r, tt_L, tt_L_err = np.array( dat['r'] ), np.array( dat['cumu_L'] ), np.array( dat['cumu_Lerr'] )

	cumu_R.append( tt_r )
	cumu_L.append( tt_L )
	cumu_Lerr.append( tt_L_err )

##.
cp_cumu_R, cp_cumu_L, cp_cumu_Lerr = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( cp_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) 
											+ '_%s-band_SB-with-BG_aveg-cumuL.csv' % band_str,)

	tt_r, tt_L, tt_L_err = np.array( dat['r'] ), np.array( dat['cumu_L'] ), np.array( dat['cumu_Lerr'] )

	cp_cumu_R.append( tt_r )
	cp_cumu_L.append( tt_L )
	cp_cumu_Lerr.append( tt_L_err )


##. figs
for tt in range( len(R_bins) - 1 ):

	#.
	medi_cMag_f = np.mean( 10**( 0.4 * (22.5 - tmp_r_cMag[ tt ] ) ) )
	cc_medi_cMag_f = np.mean( 10**( 0.4 * (22.5 - cp_r_cMag[ tt ] ) ) )

	fig = plt.figure( figsize = (10.8, 4.8) )

	ax0 = fig.add_axes([0.08, 0.31, 0.40, 0.63])
	sub_ax0 = fig.add_axes([0.08, 0.10, 0.40, 0.21])

	ax1 = fig.add_axes([0.58, 0.31, 0.40, 0.63])
	sub_ax1 = fig.add_axes([0.58, 0.10, 0.40, 0.21])

	ax0.errorbar( tmp_R[tt], tmp_sb[tt], yerr = tmp_err[tt], marker = '.', ls = '--', color = 'b',
		ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, alpha = 0.75, label = 'Control',)

	ax0.errorbar( cp_tmp_R[tt], cp_tmp_sb[tt], yerr = cp_tmp_err[tt], marker = '.', ls = '-', color = 'r',
		ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, alpha = 0.75, label = 'Member',)

	##.
	_kk_tmp_F = interp.interp1d( tmp_R[tt], tmp_sb[tt], kind = 'cubic', fill_value = 'extrapolate',)

	sub_ax0.plot( cp_tmp_R[tt], cp_tmp_sb[tt] / _kk_tmp_F( cp_tmp_R[tt] ), ls = '--', color = 'r', alpha = 0.75,)

	sub_ax0.fill_between( cp_tmp_R[tt], y1 = (cp_tmp_sb[tt] - cp_tmp_err[tt]) / _kk_tmp_F( cp_tmp_R[tt]), 
			y2 = (cp_tmp_sb[tt] + cp_tmp_err[tt]) / _kk_tmp_F( cp_tmp_R[tt]), color = 'r', alpha = 0.12,)

	sub_ax0.axhline( y = 1, ls = ':', color = 'k',)

	ax0.legend( loc = 3, frameon = False, fontsize = 12,)

	ax0.set_xscale('log')
	ax0.set_xlim( 1e0, 1e2 )
	# ax0.set_xlabel('R [kpc]', fontsize = 12,)

	ax0.annotate( s = '%s, %s-band' % (fig_name[tt], band_str), xy = (0.40, 0.85), xycoords = 'axes fraction', fontsize = 12,)

	ax0.set_ylim( 1e-3, 5e0 )
	ax0.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax0.set_yscale('log')

	sub_ax0.set_xlim( ax0.get_xlim() )
	sub_ax0.set_xscale('log')
	sub_ax0.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	sub_ax0.set_ylabel('$\\mu_{Member} \; / \; \\mu_{Control}$', labelpad = 8, fontsize = 12,)
	# sub_ax0.set_ylim( 0.6, 1.35 )
	sub_ax0.set_ylim( 0.9, 1.1 )

	sub_ax0.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	sub_ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax0.set_xticklabels( labels = [] )


	ax1.errorbar( cumu_R[tt], cumu_L[tt], yerr = cumu_Lerr[tt], marker = '.', ls = '--', color = 'b',
		ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, alpha = 0.75, label = 'Control',)

	ax1.errorbar( cp_cumu_R[tt], cp_cumu_L[tt], yerr = cp_cumu_Lerr[tt], marker = '.', ls = '-', color = 'r',
		ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, alpha = 0.75, label = 'Member',)

	ax1.axhline( y = medi_cMag_f, ls = ':', color = 'b', lw = 2.5,)
	ax1.axhline( y = cc_medi_cMag_f, ls = ':', color = 'r',)

	##.
	_kk_tmp_F = interp.interp1d( cumu_R[tt], cumu_L[tt], kind = 'cubic', fill_value = 'extrapolate',)

	sub_ax1.plot( cp_cumu_R[tt], cp_cumu_L[tt] / _kk_tmp_F( cp_cumu_R[tt] ), ls = '--', color = 'r', alpha = 0.75,)

	sub_ax1.fill_between( cp_cumu_R[tt], y1 = (cp_cumu_L[tt] - cp_cumu_Lerr[tt]) / _kk_tmp_F( cp_cumu_R[tt]), 
			y2 = (cp_cumu_L[tt] + cp_cumu_Lerr[tt]) / _kk_tmp_F( cp_cumu_R[tt]), color = 'r', alpha = 0.12,)

	sub_ax1.axhline( y = 1, ls = ':', color = 'k',)

	ax1.legend( loc = 4, frameon = False, fontsize = 12,)

	ax1.set_xscale('log')
	ax1.set_xlim( 1e0, 3e2 )
	# ax1.set_xlabel('R [kpc]', fontsize = 12,)

	ax1.set_ylim( 8e-1, 5e1 )
	ax1.set_ylabel('F [ nanomaggy ]', fontsize = 12,)
	ax1.set_yscale('log')

	sub_ax1.set_xlim( ax1.get_xlim() )
	sub_ax1.set_xscale('log')
	sub_ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	sub_ax1.set_ylabel('$F_{Member} \; / \; F_{Control}$', labelpad = 8, fontsize = 12,)
	# sub_ax1.set_ylim( 0.5, 1.5 )
	sub_ax1.set_ylim( 0.9, 1.1 )

	sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax1.set_xticklabels( labels = [] )

	plt.savefig(
		'/home/xkchen/contrl-galx_%s-band_%.2f-%.2fR200m_SB-withBG_cumuL_compare.png' % 
		(band_str, R_bins[tt], R_bins[tt + 1]), dpi = 300,)
	plt.close()



##... BG-subtracted SB profiles
nbg_R, nbg_SB, nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % 
						(R_bins[tt], R_bins[tt + 1], band_str) + '_aveg-jack_BG-sub_SB.csv',)

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

	nbg_R.append( tt_r )
	nbg_SB.append( tt_sb )
	nbg_err.append( tt_sb_err )

##.
cp_nbg_R, cp_nbg_SB, cp_nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( cp_out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1])
									+ '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

	cp_nbg_R.append( tt_r )
	cp_nbg_SB.append( tt_sb )
	cp_nbg_err.append( tt_sb_err )


##... cumulative Luminosity
cumu_R, cumu_L, cumu_Lerr = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( out_path + 
		'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band_BG-sub-SB_aveg-cumuL.csv' % 
		(R_bins[tt], R_bins[tt + 1], band_str),)

	tt_r, tt_L, tt_L_err = np.array( dat['r'] ), np.array( dat['cumu_L'] ), np.array( dat['cumu_Lerr'] )

	cumu_R.append( tt_r )
	cumu_L.append( tt_L )
	cumu_Lerr.append( tt_L_err )


##.
cp_cumu_R, cp_cumu_L, cp_cumu_Lerr = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( cp_out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) 
						+ '_%s-band_BG-sub-SB_aveg-cumuL.csv' % band_str)

	tt_r, tt_L, tt_L_err = np.array( dat['r'] ), np.array( dat['cumu_L'] ), np.array( dat['cumu_Lerr'] )

	cp_cumu_R.append( tt_r )
	cp_cumu_L.append( tt_L )
	cp_cumu_Lerr.append( tt_L_err )


##. figs
for tt in range( len(R_bins) - 1 ):

	#.
	medi_cMag_f = np.mean( 10**( 0.4 * (22.5 - tmp_r_cMag[ tt ] ) ) )
	cc_medi_cMag_f = np.mean( 10**( 0.4 * (22.5 - cp_r_cMag[ tt ] ) ) )


	fig = plt.figure( figsize = (10.8, 4.8) )

	ax0 = fig.add_axes([0.08, 0.31, 0.40, 0.63])
	sub_ax0 = fig.add_axes([0.08, 0.10, 0.40, 0.21])

	ax1 = fig.add_axes([0.58, 0.31, 0.40, 0.63])
	sub_ax1 = fig.add_axes([0.58, 0.10, 0.40, 0.21])


	ax0.errorbar( nbg_R[tt], nbg_SB[tt], yerr = nbg_err[tt], marker = '.', ls = '--', color = 'b',
		ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, alpha = 0.75, label = 'Control',)

	ax0.errorbar( cp_nbg_R[tt], cp_nbg_SB[tt], yerr = cp_nbg_err[tt], marker = '.', ls = '-', color = 'r',
		ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, alpha = 0.75, label = 'Member',)

	##.
	_kk_tmp_F = interp.interp1d( nbg_R[tt], nbg_SB[tt], kind = 'cubic', fill_value = 'extrapolate',)

	sub_ax0.plot( cp_nbg_R[tt], cp_nbg_SB[tt] / _kk_tmp_F( cp_nbg_R[tt] ), ls = '--', color = 'r', alpha = 0.75,)

	sub_ax0.fill_between( cp_nbg_R[tt], y1 = (cp_nbg_SB[tt] - cp_nbg_err[tt]) / _kk_tmp_F( cp_nbg_R[tt]), 
			y2 = (cp_nbg_SB[tt] + cp_nbg_err[tt]) / _kk_tmp_F( cp_nbg_R[tt]), color = 'r', alpha = 0.12,)

	sub_ax0.axhline( y = 1, ls = ':', color = 'k',)

	ax0.legend( loc = 3, frameon = False, fontsize = 12,)

	ax0.set_xscale('log')
	ax0.set_xlim( 1e0, 1e2 )
	# ax0.set_xlabel('R [kpc]', fontsize = 12,)

	ax0.annotate( s = '%s, %s-band' % (fig_name[tt], band_str), xy = (0.40, 0.85), xycoords = 'axes fraction', fontsize = 12,)

	ax0.set_ylim( 1e-3, 5e0 )
	ax0.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax0.set_yscale('log')

	sub_ax0.set_xlim( ax0.get_xlim() )
	sub_ax0.set_xscale('log')
	sub_ax0.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	sub_ax0.set_ylabel('$\\mu_{Member} \; / \; \\mu_{Control}$', labelpad = 8, fontsize = 12,)
	# sub_ax0.set_ylim( 0.6, 1.35 )
	sub_ax0.set_ylim( 0.9, 1.1 )

	sub_ax0.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	sub_ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax0.set_xticklabels( labels = [] )


	ax1.errorbar( cumu_R[tt], cumu_L[tt], yerr = cumu_Lerr[tt], marker = '.', ls = '--', color = 'b',
		ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, alpha = 0.75, label = 'Control',)

	ax1.errorbar( cp_cumu_R[tt], cp_cumu_L[tt], yerr = cp_cumu_Lerr[tt], marker = '.', ls = '-', color = 'r',
		ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, alpha = 0.75, label = 'Member',)

	ax1.axhline( y = medi_cMag_f, ls = ':', color = 'b', lw = 2.5,)
	ax1.axhline( y = cc_medi_cMag_f, ls = ':', color = 'r',)

	##.
	_kk_tmp_F = interp.interp1d( cumu_R[tt], cumu_L[tt], kind = 'cubic', fill_value = 'extrapolate',)

	sub_ax1.plot( cp_cumu_R[tt], cp_cumu_L[tt] / _kk_tmp_F( cp_cumu_R[tt] ), ls = '--', color = 'r', alpha = 0.75,)

	sub_ax1.fill_between( cp_cumu_R[tt], y1 = (cp_cumu_L[tt] - cp_cumu_Lerr[tt]) / _kk_tmp_F( cp_cumu_R[tt]), 
			y2 = (cp_cumu_L[tt] + cp_cumu_Lerr[tt]) / _kk_tmp_F( cp_cumu_R[tt]), color = 'r', alpha = 0.12,)

	sub_ax1.axhline( y = 1, ls = ':', color = 'k',)

	ax1.legend( loc = 4, frameon = False, fontsize = 12,)

	ax1.set_xscale('log')
	ax1.set_xlim( 1e0, 3e2 )
	# ax1.set_xlabel('R [kpc]', fontsize = 12,)

	ax1.set_ylim( 8e-1, 5e1 )
	ax1.set_ylabel('F [ nanomaggy ]', fontsize = 12,)
	ax1.set_yscale('log')

	sub_ax1.set_xlim( ax1.get_xlim() )
	sub_ax1.set_xscale('log')
	sub_ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	sub_ax1.set_ylabel('$F_{Member} \; / \; F_{Control}$', labelpad = 8, fontsize = 12,)
	# sub_ax1.set_ylim( 0.5, 1.5 )
	sub_ax1.set_ylim( 0.9, 1.1 )

	sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
	ax1.set_xticklabels( labels = [] )

	plt.savefig(
		'/home/xkchen/contrl-galx_%s-band_%.2f-%.2fR200m_BG-sub_cumuL_compare.png' % 
		(band_str, R_bins[tt], R_bins[tt + 1]), dpi = 300,)
	plt.close()

