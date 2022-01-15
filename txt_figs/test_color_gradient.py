import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp
import scipy.signal as signal

import astropy.units as U
import astropy.constants as C
import scipy.stats as sts
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
from scipy import ndimage

from fig_out_module import arr_jack_func
from img_BG_sub_SB_measure import sub_color_slope_func, M2L_slope_func


# cosmology model
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
psf_FWHM = 1.32

### === ### data load and figs
color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

## fixed richness samples
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']
file_s = 'BCG_Mstar_bin'
cat_path = '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_gri_common/'

# cat_lis = ['younger', 'older']
# fig_name = ['Low $ t_{\\mathrm{age}} $ $ \\mid \\lambda $', 'High $ t_{\\mathrm{age}} $ $ \\mid \\lambda $']
# file_s = 'BCG_age_bin'
# cat_path = '/home/xkchen/mywork/ICL/data/cat_z_form/age_bin_cat/gri_common_cat/'


## fixed BCG Mstar samples
# cat_lis = [ 'low-rich', 'hi-rich' ]
# fig_name = [ 'Low $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 'High $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $']
# file_s = 'rich_bin_fixed_BCG_M'
# cat_path = '/home/xkchen/tmp_run/data_files/figs/'

# cat_lis = [ 'low-age', 'hi-age' ]
# fig_name = [ 'Low $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 
# 			'High $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $' ]
# file_s = 'age_bin_fixed_BCG_M'
# cat_path = '/home/xkchen/tmp_run/data_files/figs/'


#. flux scaling correction
BG_path = '/home/xkchen/figs/re_measure_SBs/BGs/'


##### ===== ##### color or M/Li profile
c_dat = pds.read_csv( BG_path + '%s_color_profile.csv' % cat_lis[1] )
hi_c_r, hi_ri, hi_ri_err = np.array( c_dat['R_kpc'] ), np.array( c_dat['r-i'] ), np.array( c_dat['r-i_err'] )
hi_ri = signal.savgol_filter( hi_ri, 5, 1)
d_hi_ri = signal.savgol_filter( hi_ri, 5, 1, deriv = 1,)

c_dat = pds.read_csv( BG_path + '%s_color_profile.csv' % cat_lis[0] )
lo_c_r, lo_ri, lo_ri_err = np.array( c_dat['R_kpc'] ), np.array( c_dat['r-i'] ), np.array( c_dat['r-i_err'] )
lo_ri = signal.savgol_filter( lo_ri, 5, 1)
d_lo_ri = signal.savgol_filter( lo_ri, 5, 1, deriv = 1,)


fig = plt.figure( )
ax0 = fig.add_axes([0.15, 0.13, 0.80, 0.80])

ax0.plot( lo_c_r, lo_ri, ls = '--', color = 'b', alpha = 0.75, label = fig_name[0] )
ax0.fill_between( lo_c_r, y1 = lo_ri - lo_ri_err, y2 = lo_ri + lo_ri_err, color = 'b', alpha = 0.15,)

ax0.plot( hi_c_r, hi_ri, ls = '-', color = 'r', alpha = 0.75, label = fig_name[1] )
ax0.fill_between( hi_c_r, y1 = hi_ri - hi_ri_err, y2 = hi_ri + hi_ri_err, color = 'r', alpha = 0.15,)

ax0.legend( loc = 3, fontsize = 18, frameon = False,)
ax0.set_xscale('log')
ax0.set_xlabel('$R \; [\\mathrm{k}pc] $', fontsize = 18)
ax0.set_xlim( 3e0, 1.1e3)

ax0.set_ylabel('$ r \; - \; i $', fontsize = 20,)
ax0.set_ylim( 0.4, 0.8 )
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)

plt.savefig('/home/xkchen/%s_r-i_profile.png' % file_s, dpi = 300)
plt.close()


#. slope of color 
N_samples = 30

for mm in range( 2 ):

	sub_color_file = BG_path + '%s_jack-' % cat_lis[mm] + 'sub-%d_color_profile.csv'
	sub_slope_file = BG_path + '%s_jack-' % cat_lis[mm] + 'sub-%d_color_slope.csv'
	aveg_slope_file = BG_path + '%s_color_slope.csv' % cat_lis[mm]

	WL, p_order = 13, 1
	d_lgR = True
	sub_color_slope_func( N_samples, sub_color_file, sub_slope_file, aveg_slope_file, WL, p_order, d_lgR = d_lgR )

raise

tmp_dcR, tmp_dri, tmp_dri_err = [], [], []
for mm in range( 2 ):

	c_dat = pds.read_csv( BG_path + '%s_color_slope.csv' % cat_lis[mm],)
	tt_c_r = np.array( c_dat['R_kpc'] )

	# tt_dgr, tt_dgr_err = np.array( c_dat['d_gr'] ), np.array( c_dat['d_gr_err'] )
	# tt_dgi, tt_dgi_err = np.array( c_dat['d_gi'] ), np.array( c_dat['d_gi_err'] )
	tt_dri, tt_dri_err = np.array( c_dat['d_ri'] ), np.array( c_dat['d_ri_err'] )

	tmp_dcR.append( tt_c_r )
	tmp_dri.append( tt_dri )
	tmp_dri_err.append( tt_dri_err )


