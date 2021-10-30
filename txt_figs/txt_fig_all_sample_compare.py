import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
import scipy.signal as signal

from scipy import stats as sts
from scipy import integrate as integ
import seaborn as sns

from surface_mass_density import sigmam, sigmac
from surface_mass_density import input_cosm_model, cosmos_param, rhom_set
from surface_mass_density import misNFW_sigma_func, obs_sigma_func

# cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396

# psf_FWHM = [ 1.56, 1.67, 1.50 ] # arcsec
psf_FWHM = 1.32 # arcsec

band = ['r', 'g', 'i']
Mag_sun = [ 4.65, 5.11, 4.53 ]
l_wave = np.array( [6166, 4686, 7480] )

### === ### initial surface_mass_density.py module
input_cosm_model( get_model = Test_model )
cosmos_param()

### === M/L - color
def lg_linea_func( x, a, b):
	lg_M2L = a + b * x
	return lg_M2L

def resi_func( po, x, y):

	a, b = po[:]
	lg_m2l = lg_linea_func( x, a, b)
	delta = lg_m2l - y
	return delta

def lg_bi_linear_func(x, A, B, C0, D0):
	"""
	x_arr : g-r, r-i, luminosity
	"""
	x_gr, x_ri, x_lgL = x[0], x[1], x[2]
	y = A * x_gr + B * x_ri + C0 * x_lgL + D0
	return y

def resi_bi_line_func( po, x, y):

	A, B, C0, D0 = po[:]
	pre_y = lg_bi_linear_func(x, A, B, C0, D0)
	delta = pre_y - y
	return delta

def hist2d_pdf_func(x, y, bins, levels, smooth = None, weights = None,):

	from scipy.ndimage import gaussian_filter

	H, X, Y = np.histogram2d( x.flatten(), y.flatten(), bins = bins, weights = weights)

	if smooth is not None:
		H = gaussian_filter(H, smooth)

	Hflat = H.flatten()
	inds = np.argsort(Hflat)[::-1]
	Hflat = Hflat[inds]
	sm = np.cumsum(Hflat)
	sm /= sm[-1]
	V = np.empty(len(levels))

	for i, v0 in enumerate(levels):
		try:
			V[i] = Hflat[sm <= v0][-1]
		except IndexError:
			V[i] = Hflat[0]
	V.sort()

	m = np.diff(V) == 0
	if np.any(m) and not quiet:
		logging.warning("Too few points to create valid contours")
	while np.any(m):
		V[np.where(m)[0][0]] *= 1.0 - 1e-4
		m = np.diff(V) == 0
	V.sort()

	# Compute the bin centers.
	X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

	# Extend the array for the sake of the contours at the plot edges.
	H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
	H2[2:-2, 2:-2] = H
	H2[2:-2, 1] = H[:, 0]
	H2[2:-2, -2] = H[:, -1]
	H2[1, 2:-2] = H[0]
	H2[-2, 2:-2] = H[-1]
	H2[1, 1] = H[0, 0]
	H2[1, -2] = H[0, -1]
	H2[-2, 1] = H[-1, 0]
	H2[-2, -2] = H[-1, -1]
	X2 = np.concatenate(
		[
			X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
			X1,
			X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
		]
	)
	Y2 = np.concatenate(
		[
			Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
			Y1,
			Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
		]
	)

	return H, H2, X2, Y2, V

def aveg_sigma_func(rp, sigma_arr, N_grid = 100):

	NR = len( rp )
	aveg_sigma = np.zeros( NR, dtype = np.float32 )

	tR = rp
	intep_sigma_F = interp.interp1d( tR , sigma_arr, kind = 'cubic', fill_value = 'extrapolate',)

	cumu_mass = np.zeros( NR, )
	lg_r_min = np.log10( np.min( rp ) / 10 )

	for ii in range( NR ):

		new_rp = np.logspace( lg_r_min, np.log10( tR[ii] ), N_grid)
		new_sigma = intep_sigma_F( new_rp )

		cumu_sigma = integ.simps( new_rp * new_sigma, new_rp)

		aveg_sigma[ii] = 2 * cumu_sigma / tR[ii]**2

	return aveg_sigma

def sersic_func(r, Ie, re, ndex):
	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

### === ### data load, BG-sub SB and color
z_ref = 0.25
Dl_ref = Test_model.luminosity_distance( z_ref ).value
a_ref = 1 / (z_ref + 1)

Da_ref = Test_model.angular_diameter_distance( z_ref ).value
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec

color_s = [ 'r', 'g', 'darkred' ]
BG_path = '/home/xkchen/figs/re_measure_SBs/BGs/'

nbg_tot_r, nbg_tot_sb, nbg_tot_err = [], [], []
nbg_tot_mag, nbg_tot_mag_err = [], []

for kk in range( 3 ):
	with h5py.File( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-sub_SB.h5' % band[kk], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	nbg_tot_r.append( tt_r )
	nbg_tot_sb.append( tt_sb )
	nbg_tot_err.append( tt_err )
	nbg_tot_mag.append( tt_mag )
	nbg_tot_mag_err.append( tt_mag_err )

nbg_tot_r = np.array( nbg_tot_r )
nbg_tot_r = nbg_tot_r / 1e3


### === ### Z19, Z05, SB profile and color
Z05_r, Z05_sb, Z05_mag = [], [], []

for kk in range( 3 ):
	SB_tt = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/%s_band_BCG_ICL.csv' % band[kk],)
	R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
	R_obs = R_obs**4
	flux_obs = 10**( (22.5 - SB_obs + 2.5 * np.log10(pixel**2) ) / 2.5 ) / pixel**2

	Z05_r.append( R_obs )
	Z05_sb.append( flux_obs )
	Z05_mag.append( SB_obs )

Z05_r = np.array( Z05_r )
Z05_r = Z05_r / 1e3

##...Z05 with mask incompleteness correction
last_Z05_r, last_Z05_sb = [], []

for kk in range( 3 ):
	SB_tt = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/%s_band_sub_unmask.csv' % band[kk],)
	R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
	R_obs = R_obs**4

	last_Z05_r.append( R_obs )
	last_Z05_sb.append( SB_obs )

last_Z05_r = np.array( last_Z05_r )
last_Z05_r = last_Z05_r / 1e3

##...Z05, color
pdat_0 = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/Z05_g-r.csv')
r_05_0, g2r_05 = np.array(pdat_0['(R)^(1/4)']), np.array(pdat_0['g-r'])
r_05_0 = r_05_0**4
r_05_0 = r_05_0 / 1e3

pdat_2 = pds.read_csv('/home/xkchen/mywork/ICL/data/Zibetti_SB/Z05_r-i.csv')
r_05_1, r2i_05 = np.array(pdat_2['(R)^(1/4)']), np.array(pdat_2['r-i'])
r_05_1 = r_05_1**4
r_05_1 = r_05_1 / 1e3

#...Z19 SB and color
z19_g_dat = np.loadtxt('/home/xkchen/mywork/ICL/data/Zhang_SB/pure_ICL_measurement_DES_g.txt')
cc_R = z19_g_dat[:,0]
cc_g_mag = z19_g_dat[:,1] 

z19_r_dat = np.loadtxt('/home/xkchen/mywork/ICL/data/Zhang_SB/pure_ICL_measurement_DES_r.txt')
cc_R = z19_r_dat[:,0]
cc_r_mag = z19_r_dat[:,1]

idx = cc_R <= 200
r_19, g2r_19 = cc_R[idx] / 1e3, cc_g_mag[idx] - cc_r_mag[idx]


### === ### color-Lumi-Mass relation
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
cat_path = '/home/xkchen/tmp_run/data_files/figs/'

dd_gr, dd_ri, dd_Li = [], [], []
dd_lgM = []
dd_fit_p = []
dd_sigma, dd_sp_R = [], []

for mm in range( 2 ):

	#... lg_Mstar
	l_dat = pds.read_csv( cat_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[mm] )
	l_obs_z, l_rich  = np.array( l_dat['z']), np.array( l_dat['rich'])
	l_lgM, l_age = np.array( l_dat['lg_Mstar']), np.array( l_dat['BCG_age'] )

	#... mag
	pdat = pds.read_csv( cat_path + '%s_BCG-color.csv' % cat_lis[mm] )
	p_g_mag, p_r_mag, p_i_mag = np.array( pdat['g_mag'] ), np.array( pdat['r_mag'] ), np.array( pdat['i_mag'] )

	gr_ = p_g_mag - p_r_mag
	ri_ = p_r_mag - p_i_mag

	#... Lumi
	pdat = pds.read_csv( cat_path + '%s_BCG_cmag.csv' % cat_lis[mm] )
	i_cMag = np.array( pdat['i_cMag'] )
	i_Lumi = 10**( -0.4 * ( i_cMag - Mag_sun[2] ) )

	r_cMag = np.array( pdat['r_cMag'] )
	r_Lumi = 10**( -0.4 * ( r_cMag - Mag_sun[0] ) )

	g_cMag = np.array( pdat['g_cMag'] )
	g_Lumi = 10**( -0.4 * ( g_cMag - Mag_sun[1] ) )

	L_i = i_Lumi + 0.
	L_g = g_Lumi + 0. 
	L_r = r_Lumi + 0.

	i_Mag = i_cMag + 0.
	g_Mag = g_cMag + 0.
	r_Mag = r_cMag + 0.

	gr_arr = gr_ + 0.
	ri_arr = ri_ + 0.
	lg_Mstar = l_lgM - 2 * np.log10( h )

	dd_gr.append( gr_ )
	dd_ri.append( ri_ )
	dd_Li.append( L_i )
	dd_lgM.append( lg_Mstar )

dd_gr, dd_ri, dd_Li = np.array( dd_gr ), np.array( dd_ri ), np.array( dd_Li )
dd_lgM = np.array( dd_lgM )
dd_lg_Li = [ np.log10( dd_Li[0] ), np.log10( dd_Li[1] ) ]


#... all cluster sample
fit_dat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/M2L_Lumi_selected/least-square_M-to-i-band-Lumi&color.csv' )
all_a_, all_b_ = np.array( fit_dat['a'])[0], np.array( fit_dat['b'])[0]
all_c_, all_d_ = np.array( fit_dat['c'])[0], np.array( fit_dat['d'])[0]

tot_gr = np.hstack( ( dd_gr[0], dd_gr[1] ) )
tot_ri = np.hstack( ( dd_ri[0], dd_ri[1] ) )
tot_lg_Li = np.hstack( ( dd_lg_Li[0], dd_lg_Li[1] ) )
tot_lgM = np.hstack( ( dd_lgM[0], dd_lgM[1] ) )
put_x = np.array( [tot_gr, tot_ri, tot_lg_Li] )
tot_fit_lgM = lg_bi_linear_func( put_x, all_a_, all_b_, all_c_, all_d_ )

#. sub-binning stellar mass 
bin_mx = np.linspace(11.1, 12.3, 7)
bin_mx = np.r_[ 10.8, bin_mx ]

bin_mc = 0.5 * ( bin_mx[1:] + bin_mx[:-1])
bin_medi_lgM, bin_lgM_std = np.array([]), np.array([])

for oo in range( len(bin_mx) - 1):
	id_in = ( tot_fit_lgM >= bin_mx[oo] ) & ( tot_fit_lgM <= bin_mx[oo+1] )
	sub_lgM = tot_lgM[id_in]

	bin_medi_lgM = np.r_[ bin_medi_lgM, np.median( sub_lgM ) ]
	bin_lgM_std = np.r_[ bin_lgM_std, np.std( sub_lgM ) ]

#. sub-binning mass-to-light ratio
obs_M2L = tot_lgM - tot_lg_Li
fit_M2L = tot_fit_lgM - tot_lg_Li

sp_R, Ps = sts.spearmanr(obs_M2L, fit_M2L)
Var = np.sum( (fit_M2L - obs_M2L )**2 ) / len( tot_lgM )
sigma = np.sqrt( Var )

bin_x = np.linspace(0.14, 0.48, 11)
bin_x = np.r_[ -0.2, bin_x ]

bin_cen = 0.5 * ( bin_x[1:] + bin_x[:-1])
bin_medi, bin_std = np.array([]), np.array([])
bin_Ng = np.array([])

for oo in range( len(bin_x) - 1):
	id_in = ( fit_M2L >= bin_x[oo] ) & ( fit_M2L <= bin_x[oo+1] )
	sub_M2L = obs_M2L[id_in]

	bin_medi = np.r_[ bin_medi, np.median( sub_M2L ) ]
	bin_std = np.r_[ bin_std, np.std( sub_M2L ) ]
	bin_Ng = np.r_[ bin_Ng, np.sum(id_in) ]

idx0 = (obs_M2L >= 0.05) & (obs_M2L <= 0.1)
idx1 = (fit_M2L >= 0.25) & (fit_M2L <= 0.32)

_Nlim = np.sum( idx0 & idx1 )
cp_obs = obs_M2L + 0.
cp_obs[ idx0 & idx1 ] = np.random.random( _Nlim ) * (-0.1)

levels = (1 - np.exp( -np.log(10) ), 1 - np.exp( np.log(5) - np.log(10) ),)
H_arr, H2_arr, X2_arr, Y2_arr, V_arr = hist2d_pdf_func( fit_M2L, cp_obs, bins = [100, 100], 
	levels = levels, smooth = (1.0, 1.0), weights = None,)


### === ### mass profile based on SDSS photometric SB profile
sdss_r, sdss_sb, sdss_err = [], [], []
for ii in range( 3 ):
	dat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/All_sample_BCG_pros/' + 
						'aveg_sdss_SB/total-sample_%s-band_Mean-jack_BCG_photo-SB_pros.csv' % band[ii] )
	ii_R = np.array( dat['R'] )
	ii_sb = np.array( dat['aveg_sb'] )
	ii_err = np.array( dat['aveg_sb_err'] )

	tt_mag = 22.5 - 2.5 * np.log10( ii_sb )
	tt_mag_err = 2.5 * ii_err / ( np.log(10) * ii_sb )
	id_nn = np.isnan( tt_mag )

	sdss_r.append( ii_R[id_nn == False] / 1e3 )
	sdss_sb.append( tt_mag[id_nn == False] )	
	sdss_err.append( tt_mag_err[id_nn == False] )


##### === ##### lensing profile read and surface galaxy number density profile
##... satellite number density
lo_Ng_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/data_Zhiwei/' + 'g2r_all_sample/data/g-r_deext/low-BCG_g-r_deext_allinfo_noRG.csv')
lo_n_rp, lo_Ng, lo_Ng_err = np.array(lo_Ng_dat['rbins']), np.array(lo_Ng_dat['sigma']), np.array(lo_Ng_dat['sigma_err'])

siglow, errsiglow = lo_Ng * h**2 / a_ref**2 / 1e6, lo_Ng_err * h**2 / a_ref**2 / 1e6 # unit, '/kpc^{-2}'
bin_R = lo_n_rp * 1e3 / h / (1 + z_ref)

hi_Ng_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/data_Zhiwei/' + 'g2r_all_sample/data/g-r_deext/hi-BCG_g-r_deext_allinfo_noRG.csv')
hi_n_rp, hi_Ng, hi_Ng_err = np.array(hi_Ng_dat['rbins']), np.array(hi_Ng_dat['sigma']), np.array(hi_Ng_dat['sigma_err'])

sighig, errsighig = hi_Ng * h**2 / a_ref**2 / 1e6, hi_Ng_err * h**2 / a_ref**2 / 1e6

lo_Ng_int_F = interp.interp1d( bin_R, siglow, kind = 'linear', fill_value = 'extrapolate')
hi_Ng_int_F = interp.interp1d( bin_R, sighig, kind = 'linear', fill_value = 'extrapolate')


## ... DM mass profile
rho_c, rho_m = rhom_set( 0 ) # in unit of M_sun * h^2 / kpc^3
lo_xi_file = '/home/xkchen/tmp_run/data_files/figs/low_BCG_M_xi-rp.txt'
hi_xi_file = '/home/xkchen/tmp_run/data_files/figs/high_BCG_M_xi-rp.txt'

lo_dat = np.loadtxt( lo_xi_file )
lo_rp, lo_xi = lo_dat[:,0], lo_dat[:,1]
lo_rho_m = ( lo_xi * 1e3 * rho_m ) * h / a_ref**2
lo_rp = lo_rp * 1e3 * a_ref / h

hi_dat = np.loadtxt( hi_xi_file )
hi_rp, hi_xi = hi_dat[:,0], hi_dat[:,1]
hi_rho_m = ( hi_xi * 1e3 * rho_m ) * h / a_ref**2
hi_rp = hi_rp * 1e3 * a_ref / h

lo_interp_F = interp.interp1d( lo_rp, lo_rho_m, kind = 'cubic',)
hi_interp_F = interp.interp1d( hi_rp, hi_rho_m, kind = 'cubic',)

lo_xi2M_2Mpc = lo_interp_F( 2e3 )
hi_xi2M_2Mpc = hi_interp_F( 2e3 )

#...
xi_rp = (lo_xi + hi_xi) / 2
tot_rho_m = ( xi_rp * 1e3 * rho_m ) / a_ref**2 * h
xi_to_Mf = interp.interp1d( lo_rp, tot_rho_m, kind = 'cubic',)

misNFW_sigma = xi_to_Mf( lo_rp )
sigma_2Mpc = xi_to_Mf( 2e3 )
lg_M_sigma = np.log10( misNFW_sigma - sigma_2Mpc )

#...
sig_aveg = (siglow + sighig) / 2
err_aveg = np.sqrt( errsiglow**2 / 4 + errsighig**2 / 4)

sig_rho_f = interp.interp1d( bin_R, sig_aveg, kind = 'linear', fill_value = 'extrapolate',)

Ng_sigma = sig_rho_f( bin_R )
Ng_2Mpc = sig_rho_f( 2e3 )
lg_Ng_sigma = np.log10( Ng_sigma - Ng_2Mpc )


## g-r color profile
id_dered = True
dered_str = 'with-dered_'

# id_dered = False
# dered_str = ''

if id_dered == False:
	c_dat = pds.read_csv( BG_path + 'tot-BCG-star-Mass_color_profile.csv' )
	tot_c_R, tot_g2r, tot_g2r_err = np.array( c_dat['R_kpc'] ), np.array( c_dat['g-r'] ), np.array( c_dat['g-r_err'] )

	sm_tot_g2r = signal.savgol_filter( tot_g2r, 7, 3)
	sm_tot_r = tot_c_R / 1e3
	sm_tot_g2r_err = tot_g2r_err

	## mass profile
	dat = pds.read_csv('/home/xkchen/figs/re_measure_SBs/SM_profile/' + 'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_mass-Lumi.csv')
	aveg_R, aveg_surf_m, aveg_surf_m_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])

if id_dered == True:
	c_dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_dered_color_profile.csv')
	tot_c_R, tot_g2r, tot_g2r_err = np.array( c_dat['R_kpc'] ), np.array( c_dat['g-r'] ), np.array( c_dat['g-r_err'] )

	sm_tot_g2r = signal.savgol_filter( tot_g2r, 7, 3)
	sm_tot_r = tot_c_R / 1e3
	sm_tot_g2r_err = tot_g2r_err

	## mass profile
	dat = pds.read_csv('/home/xkchen/figs/re_measure_SBs/SM_profile/' + 'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_mass-Lumi_with-dered.csv')
	aveg_R, aveg_surf_m, aveg_surf_m_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])


#. profiles fitting in large scale
out_lim_R = 350 # 400

fit_path = '/home/xkchen/figs/re_measure_SBs/SM_pro_fit/'

c_dat = pds.read_csv( fit_path + '%stotal_all-color-to-M_beyond-%dkpc_xi2M-fit.csv' % (dered_str, out_lim_R),)
lg_fb_gi = np.array( c_dat['lg_fb_gi'] )[0]
lg_fb_gr = np.array( c_dat['lg_fb_gr'] )[0]
lg_fb_ri = np.array( c_dat['lg_fb_ri'] )[0]

c_dat = pds.read_csv( fit_path + '%stotal_all-color-to-M_beyond-%dkpc_SG_N-fit.csv' % (dered_str, out_lim_R),)
lg_Ng_gi = np.array( c_dat['lg_fb_gi'] )[0]
lg_Ng_gr = np.array( c_dat['lg_fb_gr'] )[0]
lg_Ng_ri = np.array( c_dat['lg_fb_ri'] )[0]

const = 10**(-1 * lg_fb_gi)


### === ### weak lensing signal comparison
N_grid = 250

#. central BCG surface density profile
cen_dat = pds.read_csv( fit_path + '%stotal-sample_gri-band-based_mass-profile_cen-deV_fit.csv' % dered_str)
cen_Ie, cen_Re, cen_ne = np.array( cen_dat['Ie'] )[0], np.array( cen_dat['Re'] )[0], np.array( cen_dat['ne'] )[0]

cen_surf_M = sersic_func( lo_rp, 10**cen_Ie, cen_Re, cen_ne)
cen_aveg_sigm = aveg_sigma_func( lo_rp, cen_surf_M )
cen_deta_sigm = ( cen_aveg_sigm - cen_surf_M ) * 1e-6


#. 1-halo term only
v_m = 200 # rho_mean = 200 * rho_c * omega_m
c_mass = [5.87, 6.95]
Mh0 = [14.24, 14.24] # in unit M_sun / h
off_set = [230, 210] # in unit kpc / h
f_off = [0.37, 0.20]

mis_sigma = obs_sigma_func( lo_rp * h, np.mean(f_off), np.mean(off_set), z_ref, np.mean(c_mass), np.mean(Mh0), v_m)
mis_sigma = mis_sigma * h # M_sun / kpc^2
mean_mis_sigma = aveg_sigma_func( lo_rp, mis_sigma, N_grid = N_grid,)
mis_delta_sigma = ( mean_mis_sigma - mis_sigma ) * 1e-6 # M_sun / pc^2

#. estimation on the BCG+ICL
_cc_aveg_sigma = aveg_sigma_func( aveg_R, aveg_surf_m )
_cc_delta_sigma = ( _cc_aveg_sigma - aveg_surf_m ) * 1e-6


##... lensing profile (measured on overall sample)
tot_sigma_dat = np.loadtxt('/home/xkchen/figs/Delta_sigma_all_sample/gglensing_decals_dr8_kNN_cluster_lowhigh.cat')
tt_Rc, tt_sigma, tt_sigma_err = tot_sigma_dat[:,0], tot_sigma_dat[:,1], tot_sigma_dat[:,2]
tt_calib_f = tot_sigma_dat[:,3]
tt_boost = tot_sigma_dat[:,4]

tt_Rp = tt_Rc / (1 + z_ref) / h
tt_sigma_p, tt_sigma_perr = tt_sigma * h * (1 + z_ref)**2, tt_sigma_err * h * (1 + z_ref)**2

inter_F_sigm = interp.interp1d( tt_Rp[1:], tt_sigma_p[1:], kind = 'linear', fill_value = 'extrapolate',)

#. subsample case (comoving coordinate)
hi_obs_dat = pds.read_csv('/home/xkchen/figs/Delta_sigma_all_sample/high_BCG_M_delta-sigm.txt')
hi_obs_R, hi_obs_Detsigm = np.array( hi_obs_dat['R'] ), np.array( hi_obs_dat['delta_sigma'] )

hi_obs_cov = np.loadtxt('/home/xkchen/figs/Delta_sigma_all_sample/high_BCG_M_delta-sigm_covmat.txt')
hi_obs_err = np.sqrt( np.diag( hi_obs_cov ) )

lo_obs_dat = pds.read_csv('/home/xkchen/figs/Delta_sigma_all_sample/low_BCG_M_delta-sigm.txt')
lo_obs_R, lo_obs_Detsigm = np.array( lo_obs_dat['R'] ), np.array( lo_obs_dat['delta_sigma'] )

lo_obs_cov = np.loadtxt('/home/xkchen/figs/Delta_sigma_all_sample/low_BCG_M_delta-sigm_covmat.txt')
lo_obs_err = np.sqrt( np.diag( lo_obs_cov ) )

aveg_Delta_sigm = ( hi_obs_Detsigm / hi_obs_err**2 + lo_obs_Detsigm / lo_obs_err**2 ) / ( 1 / hi_obs_err**2 + 1 / lo_obs_err**2 )

wi_lo = ( 1 / lo_obs_err**2 ) / ( 1 / hi_obs_err**2 + 1 / lo_obs_err**2 )
wi_hi = ( 1 / hi_obs_err**2 ) / ( 1 / hi_obs_err**2 + 1 / lo_obs_err**2 )
# wi_lo, wi_hi = 1, 1
aveg_delta_err = np.sqrt( wi_lo**2 * lo_obs_err**2 + wi_hi**2 * hi_obs_err**2 )


#. delta sigma of large scale (1-halo term + 2-halo term)
R_aveg_sigma_1 = aveg_sigma_func( lo_rp, misNFW_sigma,)
delt_sigm_1 = ( R_aveg_sigma_1 - misNFW_sigma ) * 1e-6 # no weight



#### === ##### figs
phyR_psf = phyR_psf / 1e3 # unit : Mpc

_cmap_lis = []
for ii in range( 9 ):
	sub_color = mpl.cm.Greys_r( ii / 8 )
	_cmap_lis.append( sub_color )


# fig_tx = plt.figure( figsize = (5.4, 5.4) )
# ax0 = fig_tx.add_axes( [0.15, 0.11, 0.83, 0.83] )

# ax0.scatter( fit_M2L, obs_M2L, marker = '.', s = 5.0, color = 'k', alpha = 0.45,)
# ax0.plot( fit_M2L, fit_M2L, ls = '-', color = 'k', alpha = 0.75,)

# ax0.contourf( X2_arr, Y2_arr, H2_arr.T, np.concatenate([ V_arr, [ H_arr.max() * (1 + 1e-4) ] ] ), 
# 	colors = [ _cmap_lis[5], _cmap_lis[7], _cmap_lis[8] ],)
# ax0.errorbar( bin_cen[bin_Ng > 1], bin_medi[bin_Ng > 1], yerr = bin_std[bin_Ng > 1], color = 'k', marker = 'o', capsize = 2.5,
# 	linestyle = 'none', alpha = 0.75,)

# ax0.set_ylim( 0.0, 0.55)
# ax0.set_xlim( 0.14, 0.47)

# ax0.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
# ax0.xaxis.set_minor_locator( ticker.AutoMinorLocator() )

# ax0.set_ylabel('$ {\\rm \\mathcal{lg} } \, (M_{\\ast}^{ \\mathrm{SED} } \, / \, L_{i}) $', fontsize = 15,)
# ax0.set_xlabel('$ {\\rm \\mathcal{lg} } \, (M_{\\ast} \, / \, L_{i}) {=} a \\times (g-r)$' + 
# 	'${+} b \\times (r-i) {+} c \\times {\\rm \\mathcal{lg} } L_{i} {+} d$', fontsize = 15,)

# ax0.annotate( text = '$\\mathcal{a} = \; \; \; %.3f$' % all_a_ + '\n' + '$\\mathcal{b} = \; \; \; %.3f$' % all_b_ + '\n' + 
# 	'$\\mathcal{c} = \; \; \; %.3f$' % ( all_c_ - 1 ) + '\n' + '$\\mathcal{d}{=}{-}%.3f$' % np.abs(all_d_), xy = (0.70, 0.05), 
# 	fontsize = 15, xycoords = 'axes fraction', color = 'k',)

# ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

# plt.savefig('/home/xkchen/Mass-to-Li_estimation.png', dpi = 300)
# # plt.savefig('/home/xkchen/Mass-to-Li_estimation.pdf', dpi = 300)
# plt.close()


# fig_tx = plt.figure( figsize = (5.8, 5.8) )
# ax0 = fig_tx.add_axes( [0.15, 0.11, 0.83, 0.83] )

# ax0.scatter( tot_fit_lgM, tot_lgM, marker = '.', s = 5.0, color = 'grey', alpha = 0.5,)
# ax0.plot( tot_fit_lgM, tot_fit_lgM, ls = '-', color = 'k',)

# ax0.errorbar( bin_mc, bin_medi_lgM, yerr = bin_lgM_std, color = 'k', marker = 'o', capsize = 2.5, linestyle = 'none',)

# ax0.set_ylim( 10.5, 12.25 )
# ax0.set_xlim( 10.8, 12.25 )

# x_tick_arr = np.arange(10.8, 12.4, 0.2)
# tick_lis = [ '%.1f' % pp for pp in x_tick_arr ]
# ax0.set_xticks( x_tick_arr )
# ax0.get_xaxis().set_major_formatter( ticker.FixedFormatter( tick_lis ) )

# ax0.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
# ax0.xaxis.set_minor_locator( ticker.AutoMinorLocator() )

# ax0.set_ylabel('$ {\\rm \\mathcal{lg} } \, M_{\\ast}^{ \\mathrm{BCG} } $', fontsize = 15,)
# ax0.set_xlabel('$ \\langle {\\rm \\mathcal{lg} } \, M_{\\ast}^{\\mathrm{BCG} } \\rangle {=} a \\times (g-r)$' + 
# 	'${+} b \\times (r-i) {+} c \\times {\\rm \\mathcal{lg} } L_{i} {+} d$', fontsize = 15,)

# ax0.annotate( text = '$\\mathcal{a} = \; \; \; %.3f$' % all_a_ + '\n' + '$\\mathcal{b} = \; \; \; %.3f$' % all_b_ + '\n' + 
# 	'$\\mathcal{c} = \; \; \; %.3f$' % ( all_c_ ) + '\n' + '$\\mathcal{d}{=}{-}%.3f$' % np.abs(all_d_), xy = (0.70, 0.05), 
# 	fontsize = 15, xycoords = 'axes fraction', color = 'k',)

# ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

# # plt.savefig('/home/xkchen/Mass-to-Li_estimation.png', dpi = 300)
# plt.savefig('/home/xkchen/Mass-to-Li_estimation.pdf', dpi = 300)
# plt.close()



fig = plt.figure( figsize = ( 15.40, 4.8 ) )
ax0 = fig.add_axes( [0.05, 0.12, 0.275, 0.85] )
ax1 = fig.add_axes( [0.38, 0.12, 0.275, 0.85] )
ax2 = fig.add_axes( [0.71, 0.12, 0.275, 0.85] )

ax0.errorbar( aveg_R / 1e3, aveg_surf_m, yerr = aveg_surf_m_err, ls = 'none', marker = 'o', ms = 8, mec = 'k', mfc = 'none', alpha = 0.85, 
	capsize = 3, ecolor = 'k', label = '$\\mathrm{ {BCG} \, {+} \, {ICL} }$',)

# ax0.plot( lo_rp / 1e3, 10**lg_M_sigma * 10**lg_fb_gi, ls = '-', color = 'k', alpha = 0.65, 
# 	label = '$ \\Sigma_{m} \, / \, {%.0f} $' % const,)

ax0.plot( lo_rp / 1e3, 10**lg_M_sigma * 10**lg_fb_gi, ls = '-', color = 'k', alpha = 0.65, 
	label = '$ \\gamma \, \\Sigma_{m} $',)
ax0.plot( bin_R / 1e3, (Ng_sigma - Ng_2Mpc) * 10**lg_Ng_gi, ls = '--', color = 'k', alpha = 0.75, 
	label = '$ 10^{\\mathrm{%.1f}}\, M_{\\odot} \\times \\Sigma_{g} $' % lg_Ng_gi,)

ax0.set_xlim( 9e-3, 2e0)
ax0.set_xscale('log')
ax0.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15)
ax0.set_yscale('log')
ax0.set_ylim( 1e4, 3e8 )
ax0.set_ylabel( '$\\Sigma_{\\ast}^{ \\mathrm{ \\tt{B+I} } } \; [M_{\\odot} \, / \, \\mathrm{k}pc^2]$', fontsize = 15,)
ax0.legend( loc = 3, frameon = False, fontsize = 15,)

ax0.set_xticks([ 1e-2, 1e-1, 1e0, 2e0])
ax0.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$', '$\\mathrm{2}$'] )

ax0.fill_betweenx( y = np.logspace(3, 9.8, 200), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)


# ax1.errorbar( tt_Rp[1:], tt_sigma_p[1:] * tt_boost[1:], yerr = tt_sigma_perr[1:], xerr = None, color = 'k', marker = 's', ms = 8, ls = 'none', 
# 	ecolor = 'k', alpha = 0.75, mec = 'k', mfc = 'none', capsize = 2, label = 'Observed')

ax1.errorbar( lo_obs_R[1:] / (1 + z_ref) / h, aveg_Delta_sigm[1:] * h / a_ref**2, yerr = aveg_delta_err[1:] * h / a_ref**2, xerr = None, ms = 8, color = 'k', 
	marker = 's', ls = 'none', ecolor = 'k', alpha = 0.75, mec = 'k', mfc = 'none', capsize = 2, label = '$\\mathrm{ {Weak} \; {Lensing} }$',)
ax1.plot( lo_rp / 1e3, delt_sigm_1, ls = '-', color = 'k', label = '$\\mathrm{ \\mathcal{lg} } \, [ M_{h} \, / \, M_{\\odot} ] = 14.41$',)

ax1.set_xlim( 9e-3, 2e1)
ax1.set_xscale('log')
ax1.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15)
ax1.set_yscale('log')

ax1.set_ylim( 0.9e0, 5e2)
ax1.set_ylabel('$\\Delta \\Sigma \; [M_{\\odot} \, / \, pc^2]$', fontsize = 15,)
ax1.legend( loc = 3, frameon = False, fontsize = 15,)

ax1.set_xticks([ 1e-2, 1e-1, 1e0, 1e1, 2e1])
ax1.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$', '$\\mathrm{10}$', '$\\mathrm{20}$'])

ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
ax1.fill_betweenx( y = np.logspace(0, 3, 200), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)


# ax2.plot( bin_R / 1e3, Ng_sigma * 1e6, ls = '--', color = 'k', alpha = 0.75,)
# ax2.fill_between( bin_R / 1e3, y1 = ( Ng_sigma - err_aveg ) * 1e6, y2 = ( Ng_sigma + err_aveg ) * 1e6, color = 'k', alpha = 0.15,)
ax2.errorbar( bin_R / 1e3, Ng_sigma * 1e6, yerr = err_aveg * 1e6, xerr = None, color = 'k', marker = 's', ms = 8, ls = 'none', 
	ecolor = 'k', alpha = 0.75, mec = 'k', mfc = 'none', capsize = 2, label = 'Galaxies $(M_{i} \, < \, %.3f)$' % (-19.43 + 5 * np.log10(h) ),)

ax2.set_xlim( 9e-3, 2e1)
ax2.set_xscale('log')
ax2.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15)
ax2.set_yscale('log')
ax2.set_ylim( 2.9e-1, 2e2 )
ax2.set_ylabel('$\\Sigma_{g} \; [ \\# \, / \, \\mathrm{M}pc^{2}]$', fontsize = 15,)
ax2.legend( loc = 3, frameon = False, fontsize = 15,)

ax2.set_xticks([ 1e-2, 1e-1, 1e0, 1e1, 2e1])
ax2.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$', '$\\mathrm{10}$', '$\\mathrm{20}$'])

ax2.fill_betweenx( y = np.logspace(3, 9.8, 200), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)
ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

# plt.savefig('/home/xkchen/%stotal_sample_SB_SM.png' % dered_str, dpi = 300)
plt.savefig('/home/xkchen/total_sample_SB_SM.pdf', dpi = 300)
plt.close()



fig = plt.figure( figsize = (10.6, 4.8) )
ax0 = fig.add_axes([0.07, 0.12, 0.42, 0.85])
ax1 = fig.add_axes([0.56, 0.12, 0.42, 0.85])

for kk in ( 2, 0, 1 ):

	ax0.plot( nbg_tot_r[kk], nbg_tot_mag[kk], ls = '-', color = color_s[kk], alpha = 0.75, label = '%s' % band[kk],)
	ax0.fill_between( nbg_tot_r[kk], y1 = nbg_tot_mag[kk] - nbg_tot_mag_err[kk], 
		y2 = nbg_tot_mag[kk] + nbg_tot_mag_err[kk], color = color_s[kk], alpha = 0.15,)

	ax0.plot(Z05_r[kk], Z05_mag[kk], ls = '--', color = color_s[kk], alpha = 0.75,)

	if kk == 1:
		comp_r, comp_sb, comp_err = sdss_r[kk][:-1], sdss_sb[kk][:-1], sdss_err[kk][:-1]
	else:
		comp_r, comp_sb, comp_err = sdss_r[kk], sdss_sb[kk], sdss_err[kk]

	ax0.plot( comp_r, comp_sb, ls = ':', color = color_s[kk], alpha = 0.75,)
	ax0.fill_between( comp_r, y1 = comp_sb - comp_err, y2 = comp_sb + comp_err, color = color_s[kk], alpha = 0.15,)	

legend_1 = ax0.legend( [ 'This work (redMaPPer)', '$\\mathrm{Zibetti}{+}2005$ (maxBCG)', 'SDSS Photometry $(\\mathrm{V5}\_\\mathrm{6})$'], 
	loc = 1, frameon = False, fontsize = 14, markerfirst = False,)
legend_0 = ax0.legend( loc = 3, frameon = False, fontsize = 14,)

ax0.add_artist( legend_1 )

ax0.fill_betweenx( np.linspace(19, 36, 100), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)
ax0.text( 3e-3, 27, s = 'PSF', fontsize = 15,)

ax0.set_ylim( 20, 34 )
ax0.invert_yaxis()

ax0.set_xlim( 3e-3, 1e0)
ax0.set_xscale('log')
ax0.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15)
ax0.set_ylabel('$ \\mu \; [mag \, / \, arcsec^2] $', fontsize = 15,)

x_tick_arr = [ 1e-2, 1e-1, 1e0]
tick_lis = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$']
ax0.set_xticks( x_tick_arr )
ax0.get_xaxis().set_major_formatter( ticker.FixedFormatter( tick_lis ) )
ax0.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)


ax1.plot( sm_tot_r, sm_tot_g2r, ls = '-', color = 'r', alpha = 0.75, label = 'This work (SDSS DR8)')
ax1.fill_between( sm_tot_r, y1 = sm_tot_g2r - sm_tot_g2r_err, y2 = sm_tot_g2r + sm_tot_g2r_err, color = 'r', alpha = 0.12,)

ax1.plot(r_19, g2r_19, ls = '--', color = 'c', alpha = 0.75, label = '$\\mathrm{Zhang}{+}2019$ (DES)')
ax1.plot(r_05_0, g2r_05, ls = '-.', color = 'k', alpha = 0.75, label = '$\\mathrm{Zibetti}{+}2005$ (SDSS DR1)')

ax1.fill_betweenx( np.linspace(0, 2, 100), x1 = phyR_psf, x2 = 0, color = 'k', alpha = 0.12,)
ax1.legend( loc = 3, frameon = False, fontsize = 13.5,)
ax1.set_ylim( 0.95, 1.55 )

ax1.set_xlim( 3e-3, 1e0)
ax1.set_xscale('log')
ax1.set_ylabel('$ g - r $', fontsize = 17,)
ax1.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 15)

x_tick_arr = [ 1e-2, 1e-1, 1e0]
tick_lis = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$']
ax1.set_xticks( x_tick_arr )
ax1.get_xaxis().set_major_formatter( ticker.FixedFormatter( tick_lis ) )
ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

# plt.savefig('/home/xkchen/%stotal_sample_compare_to_Z05.png' % dered_str, dpi = 300)
plt.savefig('/home/xkchen/total_sample_compare_to_Z05.pdf', dpi = 300)
plt.close()

