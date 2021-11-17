import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C
import scipy.signal as signal

from astropy import cosmology as apcy
from scipy import interpolate as interp
from scipy import integrate as integ
from astropy.coordinates import SkyCoord

from scipy import optimize
from astropy.modeling import models, fitting

### === ### Peak func.
def mofft_F(x, Am, gamma, x_0, alpha):
	mf0 = ( x - x_0 )**2 / gamma**2
	mf1 = ( 1 + mf0 )**( -1 * alpha )

	return Am * mf1

def Lorentz_F( x, Am, gamma, x_0 ):
	mf0 = gamma**2 + ( x - x_0)**2
	mf1 = Am * gamma**2

	return mf1 / mf0

def Drude_F(x, Am, L_w, x_0 ):

	mf0 = (L_w / x_0)**2
	mf1 = (x / x_0 - x_0 / x)**2
	mf = mf0 / ( mf1 + mf0 )

	return Am * mf

def log_norm_func( r, lg_SM0, Rt, sigm_tt ):

	lg_A0 = np.log10( r ) + np.log10( sigm_tt ) + np.log10( 2 * np.pi ) / 2
	lg_A1 = np.log10( np.e) * (np.log( r ) - np.log( Rt ) )**2 / ( 2 * sigm_tt**2 )
	lg_M = lg_SM0 - lg_A0 - lg_A1

	return 10**lg_M


tt = 2

cat_lis = ['total', 'low_BCG_star-Mass', 'high_BCG_star-Mass'][ tt ]
fig_name = ['All clusters', 'Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 
			'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $'][ tt ]

# dat = np.loadtxt( '/home/xkchen/total_mid-SM_data.txt' )
# dat = np.loadtxt( '/home/xkchen/%s_mid-SM_data.txt' % cat_lis )
dat = np.loadtxt( '/home/xkchen/%s_mid-SM_data.txt' % cat_lis )

R = dat[:,0]
SM = dat[:,1]
SM_err = dat[:,2]

idx = SM >= 1e4
idr = ( 10 <= R ) & ( R <= 300)

R, SM, SM_err = R[ idx ], SM[ idx ], SM_err[ idx ]

#..
new_R = np.logspace(0, 3, 200)
po = [ 1e6, 5, 100 ]

popt, pcov = optimize.curve_fit( Lorentz_F, xdata = R, ydata = SM, p0 = po, sigma = SM_err,)

A_fit, gamm_fit, x0_fit = popt
fit_line = Lorentz_F( new_R, A_fit, gamm_fit, x0_fit )

#..
po = [ 1e6, 50, 100]

popt, pcov = optimize.curve_fit( Drude_F, xdata = R, ydata = SM, p0 = po, sigma = SM_err,)

Am_fit, Lw_fit, xc_fit = popt
fit_line_1 = Drude_F( new_R, Am_fit, Lw_fit, xc_fit )

#..
po = [ 6.5, 100, 1 ]
popt, pcov = optimize.curve_fit( log_norm_func, xdata = R, ydata = SM, p0 = po, sigma = SM_err,)
M0_fit, Rt_fit, sigma_fit = popt

fit_line_2 = log_norm_func( new_R, M0_fit, Rt_fit, sigma_fit )


plt.figure()

plt.title( fig_name )
plt.errorbar( R, SM, yerr = SM_err, fmt = 'k.',)

plt.plot( new_R, fit_line, 'r-', label = 'Lorentz')
plt.plot( new_R, fit_line_1, 'b--', label = 'Drude')
plt.plot( new_R, fit_line_2, 'g:', label = 'lognormal')

plt.legend( loc = 2)
plt.xscale('log')
plt.xlim( 1e0, 1e3 )
plt.yscale('log')
plt.ylim( 1e4, 3e6 )
plt.savefig('/home/xkchen/%s_mid_mass_fit.png' % cat_lis, dpi = 300)
plt.close()

