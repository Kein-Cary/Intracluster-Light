import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Ellipse

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

##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1. - Omega_m
Omega_k = 1. - (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']


### === sample properties (divid by centric radius only)
def centric_R_hist():

	##. all member
	# s_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_member-cat.csv' )

	##. frame limited catalog
	# s_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-limit_member-cat.csv')

	##. frame limited + P_mem cut catalog
	# s_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-limit_Pm-cut_member-cat.csv')

	##. frame limited + P_mem cut catalog + exclude BCGs
	s_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv')


	bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
	p_ra, p_dec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] )

	p_cID = np.array( s_dat['clus_ID'] )
	p_gr = np.array( s_dat['g-r'] )
	p_z = np.array( s_dat['z_spec'] )

	p_Rsat = np.array( s_dat['R_cen'] )
	p_R2Rv = np.array( s_dat['Rcen/Rv'] )


	R_cut = 0.191 # np.median( p_Rcen )
	id_vx = p_R2Rv <= R_cut


	##. satellite properties
	data = fits.open('/home/xkchen/figs/extend_Zphoto_cat/zphot_01_033_cat/redMaPPer_z-phot_0.1-0.33_member_params.fit')
	l_ra, l_dec = data[1].data['ra'], data[1].data['dec']
	l_z = data[1].data['z']
	Mag_r = data[1].data['absMagR']


	p_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )
	l_coord = SkyCoord( ra = l_ra * U.deg, dec = l_dec * U.deg )

	idx, sep, d3d = p_coord.match_to_catalog_sky( l_coord )
	id_lim = sep.value < 2.7e-4

	mp_zll = l_z[ idx[ id_lim ] ]
	mp_Magr = Mag_r[ idx[ id_lim ] ] 

	id_nul = mp_zll < 0.
	z_devi = mp_zll[ id_nul == False ] - bcg_z[ id_nul == False ]


	plt.figure()
	plt.text( 0.2, 1.0, s = '$P_{mem} \\geq 0.8 $')
	plt.plot( p_Rsat, p_R2Rv, 'k.', alpha = 0.5,)
	plt.plot( p_Rsat, p_Rsat, 'r-', alpha = 0.75,)

	plt.axvline( x = 0.213, ls = ':', color = 'b', alpha = 0.5,)
	plt.axhline( y = 0.191, ls = '--', color = 'b', alpha = 0.5,)

	plt.xlabel( '$R_{Sat} \; [Mpc / h]$' )
	plt.ylabel( '$R_{Sat} / R_{200m}$' )
	plt.savefig('/home/xkchen/R-phy_R-scaled.png', dpi = 300)
	plt.close()


	plt.figure()
	plt.hist( p_R2Rv, bins = 55, density = True, histtype = 'step',)
	plt.axvline( x = np.median(p_R2Rv), ls = '--', label = 'Median', ymin = 0, ymax = 0.35,)
	plt.axvline( x = np.mean( p_R2Rv), ls = '-', label = 'Mean', ymin = 0, ymax = 0.35,)

	plt.axvline( x = 0.191, ls = ':', label = 'Division', color = 'r',)

	plt.legend( loc = 1)
	plt.xlabel('$R_{Sat} \, / \, R_{200m}$')
	plt.savefig('/home/xkchen/redMap_sate_Rcen_hist.png', dpi = 300)
	plt.close()


	plt.figure()
	plt.hist( z_devi, bins = np.linspace(-0.2, 0.2, 55), density = True, histtype = 'step', color = 'k',)

	plt.axvline( x = np.median( z_devi ), ls = '--', label = 'Median', color = 'k',)
	plt.axvline( x = np.mean( z_devi ), ls = '-', label = 'Mean', color = 'k',)

	plt.axvline( x = np.median( z_devi ) - np.std( z_devi ), ls = ':', color = 'k',)
	plt.axvline( x = np.median( z_devi ) + np.std( z_devi ), ls = ':', color = 'k',)

	plt.legend( loc = 1)
	plt.xlabel('$z_{mem} - z_{BCG}$')
	plt.savefig('/home/xkchen/z_diffi.png', dpi = 300)
	plt.close()


	plt.figure()
	plt.hist( mp_Magr[ id_vx ], bins = np.linspace(-26, -16, 65), density = True, histtype = 'step', color = 'b', label = 'Inner')

	plt.axvline( x = np.median( mp_Magr[ id_vx ] ), ls = '--', label = 'Median', color = 'b',)
	plt.axvline( x = np.mean( mp_Magr[ id_vx ] ), ls = '-', label = 'Mean', color = 'b',)

	plt.hist( mp_Magr[ id_vx == False ], bins = np.linspace(-26, -16, 65), density = True, histtype = 'step', color = 'r', label = 'Outer')

	plt.axvline( x = np.median( mp_Magr[ id_vx == False ] ), ls = '--', color = 'r',)
	plt.axvline( x = np.mean( mp_Magr[ id_vx == False ] ), ls = '-', color = 'r',)

	plt.legend( loc = 1)
	plt.xlabel('$Mag_{r}$')
	plt.savefig('/home/xkchen/abs_r_mag_compare.png', dpi = 300)
	plt.close()

# centric_R_hist()


### === sample properties ( mag_10, radius )
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/cat/'

cat_lis = ['inner', 'middle', 'outer']
out_file = out_path + 'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_member.csv'

pp_Rs, pp_iMag = [], []

for pp in range( 3 ):

	dat = pds.read_csv( out_file % cat_lis[pp] )

	pp_Rs.append( np.array( dat['Rs/R200m'] ) )
	pp_iMag.append( np.array( dat['sat_iMag_10'] ) )

print( [len(ll) for ll in pp_Rs] )


plt.figure()
plt.hist( pp_iMag[0], bins = 55, density = True, histtype = 'step', color = 'b', label = 'Inner [%d]' % len(pp_iMag[0]) )
plt.axvline( x = np.median( pp_iMag[0]), ls = '--', color = 'b', label = 'Median', alpha = 0.5,)
plt.axvline( x = np.mean( pp_iMag[0]), ls = '-', color = 'b', label = 'Mean', alpha = 0.5,)

plt.hist( pp_iMag[1], bins = 55, density = True, histtype = 'step', color = 'g', label = 'Middle [%d]' % len(pp_iMag[1]) )
plt.axvline( x = np.median( pp_iMag[1]), ls = '--', color = 'g', alpha = 0.5,)
plt.axvline( x = np.mean( pp_iMag[1]), ls = '-', color = 'g', alpha = 0.5,)

plt.hist( pp_iMag[2], bins = 55, density =  True, histtype = 'step', color = 'r', label = 'Outer [%d]' % len(pp_iMag[2]) )
plt.axvline( x = np.median( pp_iMag[2]), ls = '--', color = 'r', alpha = 0.5,)
plt.axvline( x = np.mean( pp_iMag[2]), ls = '-', color = 'r', alpha = 0.5,)

plt.legend( loc = 1 )
plt.xlim( -23, -19 )
plt.xlabel('$Mag_{i,\;10}$')
plt.savefig('/home/xkchen/radii_bin_iMag.png', dpi = 300)
plt.close()


plt.figure()
plt.hist( pp_Rs[0], bins = 55, density = True, histtype = 'step', color = 'b', label = 'Inner [%d]' % len(pp_iMag[0]) )
plt.axvline( x = np.median( pp_Rs[0]), ls = '--', color = 'b', label = 'Median', alpha = 0.5,)
plt.axvline( x = np.mean( pp_Rs[0]), ls = '-', color = 'b', label = 'Mean', alpha = 0.5,)

plt.hist( pp_Rs[1], bins = 55, density = True, histtype = 'step', color = 'g', label = 'Middle [%d]' % len(pp_iMag[1]) )
plt.axvline( x = np.median( pp_Rs[1]), ls = '--', color = 'g', alpha = 0.5,)
plt.axvline( x = np.mean( pp_Rs[1]), ls = '-', color = 'g', alpha = 0.5,)

plt.hist( pp_Rs[2], bins = 55, density = True, histtype = 'step', color = 'r', label = 'Outer [%d]' % len(pp_iMag[2]) )
plt.axvline( x = np.median( pp_Rs[2]), ls = '--', color = 'r', alpha = 0.5,)
plt.axvline( x = np.mean( pp_Rs[2]), ls = '-', color = 'r', alpha = 0.5,)

plt.legend( loc = 1 )
plt.xlabel('$R_{sat} / R_{200m}$')
plt.savefig('/home/xkchen/radii_bin_Rs-ov-R200m.png', dpi = 300)
plt.close()

