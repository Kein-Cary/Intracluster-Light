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

from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy.stats import binned_statistic as binned
import scipy.interpolate as interp

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

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']


### === aveg of shuffles~(align with BCG)
path = '/home/xkchen/figs/tt_stack/'

N_shufl = 20

bin_rich = [ 20, 30, 50, 210 ]

##. fixed R for all richness subsample
R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

#.
band_str = 'r'


for tt in range( len(R_bins) - 1 ):

	fig = plt.figure()
	ax = fig.add_axes([0.12, 0.12, 0.80, 0.85])

	for dd in range( N_shufl ):

		with h5py.File( path + 
			'Sat-all_%.2f-%.2fR200m_%s-band_wBCG-PA_shufl-%d_BG_Mean_jack_SB-pro_z-ref.h5' 
			% (R_bins[tt], R_bins[tt + 1], band_str, dd), 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		#.
		ax.plot( tt_r, tt_sb, ls = '--', color = mpl.cm.rainbow(dd / N_shufl), alpha = 0.5,)		

	ax.set_xlim( 1e0, 5e2 )
	ax.set_xscale('log')
	ax.set_xlabel('$R \; [kpc]$', fontsize = 12,)

	ax.set_ylim( 5e-4, 2e-2 )
	ax.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax.set_yscale('log')

	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig(
		'/home/xkchen/sat_%s-band_%.2f-%.2fR200m_BG-profs.png' 
		% (band_str, R_bins[tt], R_bins[tt + 1]), dpi = 300)
	plt.close()


### === aveg of shuffles~(align with frame)

