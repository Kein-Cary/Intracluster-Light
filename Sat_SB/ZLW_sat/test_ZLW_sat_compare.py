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
import scipy.signal as signal
import scipy.stats as sts
from scipy import ndimage


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

rad2asec = U.rad.to(U.arcsec)


### === 
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


### === previous match catalog
"""
# p_dat = pds.read_csv('/home/xkchen/pre_BCGM_rgi-common_z0.2-0.3_cat.csv')
p_dat = pds.read_csv('/home/xkchen/figs/BCG_mags/clslowz_z0.17-0.30_img-cat_dered_color.csv')
p_ra, p_dec, p_z = np.array( p_dat['ra'] ), np.array( p_dat['dec'] ), np.array( p_dat['z'] )

id_vx = ( p_z >= 0.2 ) & ( p_z <= 0.3 )
lim_ra, lim_dec, lim_z = p_ra[ id_vx ], p_dec[ id_vx ], p_z[ id_vx ]

# p_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )
p_coord = SkyCoord( ra = lim_ra * U.deg, dec = lim_dec * U.deg )


# dat = pds.read_csv('/home/xkchen/Extend-BCGM_rgi-common_z-0.2-0.3_cat.csv')
dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/Extend_BCGM_bin_cat.csv')
ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

coord = SkyCoord( ra = ra * U.deg, dec = dec * U.deg )


idx, sep, d3d = coord.match_to_catalog_sky( p_coord )
id_lim = sep.value < 2.7e-4

np_ra, np_dec, np_z = ra[ id_lim == False ], dec[ id_lim == False ], z[ id_lim == False ]


keys = [ 'ra', 'dec', 'z' ]
values = [ np_ra, np_dec, np_z ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame( fill )
# out_data.to_csv( '/home/xkchen/BCGM_gri-common_extend_cat.csv')
out_data.to_csv( '/home/xkchen/z0.2-0.3_extend_clus_cat.csv')

"""

### === satellite number compare
bcg_ra, bcg_dec, bcg_z = np.array( [] ), np.array( [] ), np.array( [] )

cat_lis = ['low', 'high']

for mm in range( 2 ):

	zlw_dat = pds.read_csv('/home/xkchen/figs/sat_cat_ZLW/ZLWen_%s_BCG_star-Mass_match_cat.csv' % cat_lis[mm],)

	bcg_ra = np.r_[ bcg_ra, np.array( zlw_dat['ra'] ) ]
	bcg_dec = np.r_[ bcg_dec, np.array( zlw_dat['dec'] ) ]
	bcg_z = np.r_[ bcg_z, np.array( zlw_dat['z'] ) ]


tmp_N = np.array( [] )
tmp_r_mag = np.array( [] )
tmp_g_mag = np.array( [] )
tmp_r_Mag = np.array( [] )
tmp_R_sat = np.array( [] )
tmp_z_arr = np.array( [] )

Ns = len( bcg_ra )

for kk in range( Ns ):

	ra_g, dec_g, z_g = bcg_ra[ kk ], bcg_dec[ kk ], bcg_z[ kk ]
	dat = pds.read_csv('/home/xkchen/figs/sat_cat_ZLW/mem_match/' + 
						'ZLW_r-band_ra%.3f_dec%.3f_z%.3f_members_mag.csv' % (ra_g, dec_g, z_g),)

	r_mags = np.array( dat['r_mags'] )
	g_mags = np.array( dat['g_mags'] )
	kk_R_s = np.array( dat['centric_R(Mpc/h)'] )

	Dl_x = Test_model.luminosity_distance( z_g ).value
	D_modu = 5 * ( np.log10(Dl_x * 1e6) - 1 )

	r_Mag = r_mags - D_modu

	tmp_N = np.r_[ tmp_N, len( r_mags ) ]

	tmp_r_mag = np.r_[ tmp_r_mag, r_mags ]
	tmp_g_mag = np.r_[ tmp_g_mag, g_mags ]
	tmp_r_Mag = np.r_[ tmp_r_Mag, r_Mag ]
	tmp_R_sat = np.r_[ tmp_R_sat, kk_R_s ]
	tmp_z_arr = np.r_[ tmp_z_arr, np.ones( len( r_mags ),) * z_g ]


#. SDSS
sd_ra, sd_dec, sd_z = np.array( [] ), np.array( [] ), np.array( [] )

for mm in range( 2 ):

	sdss_dat = pds.read_csv('/home/xkchen/figs/ZLW_cat_15/redMapper_matched_%s_BCG_star-Mass.csv' % cat_lis[mm])

	sd_ra = np.r_[ sd_ra, np.array( sdss_dat['ra'] ) ]
	sd_dec = np.r_[ sd_dec, np.array( sdss_dat['dec'] ) ]
	sd_z = np.r_[ sd_z, np.array( sdss_dat['z'] ) ]

N_m = len( sd_ra )

cp_N = np.array( [] )
cp_r_mag = np.array( [] )
cp_g_mag = np.array( [] )
cp_r_Mag = np.array( [] )
cp_R_sat = np.array( [] )
cp_z_arr = np.array( [] )

for kk in range( N_m ):

	ra_g, dec_g, z_g = sd_ra[ kk ], sd_dec[ kk ], sd_z[ kk ]
	dat = pds.read_csv('/home/xkchen/figs/sat_cat_ZLW/redMap_mem_match/' + 
					'photo-z_r-band_ra%.3f_dec%.3f_z%.3f_members_mag.csv'% (ra_g, dec_g, z_g),)

	r_mags = np.array( dat['r_mags'] )
	g_mags = np.array( dat['g_mags'] )
	kk_R_s = np.array( dat['centric_R(Mpc/h)'] )

	Dl_x = Test_model.luminosity_distance( z_g ).value
	D_modu = 5 * ( np.log10(Dl_x * 1e6) - 1 )

	r_Mag = r_mags - D_modu

	cp_N = np.r_[ cp_N, len( r_mags ) ]

	cp_r_mag = np.r_[ cp_r_mag, r_mags ]
	cp_g_mag = np.r_[ cp_g_mag, g_mags ]
	cp_r_Mag = np.r_[ cp_r_Mag, r_Mag ]
	cp_R_sat = np.r_[ cp_R_sat, kk_R_s ]
	cp_z_arr = np.r_[ cp_z_arr, np.ones( len( r_mags ),) * z_g ]


## figs

diff_N = tmp_N - cp_N

tmp_g2r = tmp_g_mag - tmp_r_mag
cp_g2r = cp_g_mag - cp_r_mag


levels = (0.95, 0.9, 0.84, 0.64, 0.36, 0.16, 0.1, 0.05)

_cmap_lis = []

for ii in range( len(levels) ):

	sub_color = mpl.cm.rainbow( ii / len( levels ) )
	_cmap_lis.append( sub_color )


Mag_bins = np.linspace( -24, -20, 11)
z_bins = np.linspace( 0.2, 0.3, 6)

for tt in range( 5 ):

	if tt != 4:
		id_vx = ( tmp_z_arr >= z_bins[ tt ] ) & ( tmp_z_arr < z_bins[ tt+1 ] )
		id_ux = ( cp_z_arr >= z_bins[ tt ] ) & ( cp_z_arr < z_bins[ tt+1 ] )

	else:
		id_vx = ( tmp_z_arr >= z_bins[ tt ] ) & ( tmp_z_arr <= z_bins[ tt+1 ] )
		id_ux = ( cp_z_arr >= z_bins[ tt ] ) & ( cp_z_arr <= z_bins[ tt+1 ] )

	sub_r_Mag = tmp_r_Mag[ id_vx ]
	sub_g2r = tmp_g2r[ id_vx ]

	cc_sub_r_Mag = cp_r_Mag[ id_ux ]
	cc_sub_g2r = cp_g2r[ id_ux ]


	#. color division
	cc_sub_medi = []
	cc_sub_std = []

	for nn in range( 10 ):

		if nn != 9:
			id_ux = ( cc_sub_r_Mag >= Mag_bins[nn] ) & ( cc_sub_r_Mag < Mag_bins[nn+1] )

		else:
			id_ux = ( cc_sub_r_Mag >= Mag_bins[nn] ) & ( cc_sub_r_Mag <= Mag_bins[nn+1] )

		cc_lim_g2r = cc_sub_g2r[ id_ux ]
		cc_sub_medi.append( np.median( cc_lim_g2r ) )
		cc_sub_std.append( np.std( cc_lim_g2r ) )

	sub_cen_M = 0.5 * (Mag_bins[1:] + Mag_bins[:-1])

	#. 
	divid_F = np.polyfit( sub_cen_M, cc_sub_medi, 1)
	divid_lF = np.poly1d( divid_F )

	new_lx = np.linspace( -25, -19, 50)
	divid_line = divid_lF( new_lx )


	plt.figure()
	plt.title('$%.2f \\leq z < %.2f$' % (z_bins[tt], z_bins[tt+1]),)

	H_arr, H2_arr, X2_arr, Y2_arr, V_arr = hist2d_pdf_func( cc_sub_r_Mag, cc_sub_g2r, bins = [100, 100], 
		levels = levels, smooth = (1.5, 1.5), weights = None,)

	plt.scatter( cc_sub_r_Mag, cc_sub_g2r, marker = '.', color = 'k', s = 5, label = 'SDSS redMaPPer', alpha = 0.03)

	plt.errorbar( sub_cen_M, cc_sub_medi, yerr = cc_sub_std, marker = 'o', ls = 'none', color = 'k', ecolor = 'k', 
		mfc = 'none', mec = 'k', alpha = 0.75, capsize = 1.5,)

	plt.plot( new_lx, divid_line, 'k--', alpha = 0.75,)

	plt.contour( X2_arr, Y2_arr, H2_arr.T, np.concatenate([ V_arr, [ H_arr.max() * (1 + 1e-4) ] ] ), 
				colors = _cmap_lis,)

	plt.legend( loc = 3)
	plt.ylim( 0, 2)
	plt.ylabel('g - r')
	plt.xlim( -24.5, -19 )
	plt.xlabel('$M_{r}$')
	plt.savefig('/home/xkchen/redMaP_sat_z%.2f-%.2f_g-r_Mr_diag.png' % (z_bins[tt], z_bins[tt+1]), dpi = 300)
	plt.close()


	plt.figure()

	plt.title('$%.2f \\leq z < %.2f$' % (z_bins[tt], z_bins[tt+1]),)

	H_arr, H2_arr, X2_arr, Y2_arr, V_arr = hist2d_pdf_func( sub_r_Mag, sub_g2r, bins = [100, 100], 
		levels = levels, smooth = (1.5, 1.5), weights = None,)

	plt.scatter( sub_r_Mag, sub_g2r, marker = '.', color = 'k', s = 5, label = 'W15 catalog', alpha = 0.03)

	plt.plot( new_lx, divid_line, 'k--', alpha = 0.75, label = 'Red sequence')
	plt.plot( new_lx, divid_line - np.median(cc_sub_std) * 1, 'c:', alpha = 0.75, label = 'Red sequence - 1$\\sigma_{median}$')

	plt.contour( X2_arr, Y2_arr, H2_arr.T, np.concatenate([ V_arr, [ H_arr.max() * (1 + 1e-4) ] ] ), 
				colors = _cmap_lis,)

	plt.legend( loc = 3 )
	plt.ylim( 0, 2)
	plt.ylabel('g - r')
	plt.xlim( -24.5, -19 )
	plt.xlabel('$M_{r}$')
	plt.savefig('/home/xkchen/ZLW_sat_z%.2f-%.2f_g-r_Mr_diag.png' % (z_bins[tt], z_bins[tt+1]), dpi = 300)
	plt.close()


raise


plt.figure( figsize = (10, 5) )
ax0 = plt.subplot(121)
ax1 = plt.subplot(122)

ax0.hist( tmp_N, bins = 35, density = True, color = 'b', histtype = 'step', label = 'W15 catalog')
ax0.hist( cp_N, bins = 35, density = True, histtype = 'step', color = 'r', label = 'SDSS redMaPPer')

ax0.legend( loc = 1 )
ax0.set_xlabel('$N_{sat}$')

ax1.hist( diff_N, bins = 35, density = True, color = 'g', histtype = 'step',)
ax1.axvline( x = np.median( diff_N ), ls = '-', color = 'g', label = 'Median',)
ax1.axvline( x = np.mean( diff_N ), ls = '--', color = 'g', label = 'Mean',)

ax1.legend( loc = 1 )
ax1.set_xlabel('$N_{W15} \, - \, N_{redMaPPer}$')

plt.savefig('/home/xkchen/N_sat_compare.png', dpi = 300)
plt.close()


#.
sub_medi = []
sub_std = []

for nn in range( 10 ):

	if nn != 9:
		id_ux = ( cp_r_Mag >= Mag_bins[nn] ) & ( cp_r_Mag < Mag_bins[nn+1])
	else:
		id_ux = ( cp_r_Mag >= Mag_bins[nn] ) & ( cp_r_Mag <= Mag_bins[nn+1])

	lim_g2r = cp_g2r[ id_ux ]

	sub_medi.append( np.median( lim_g2r ) )
	sub_std.append( np.std( lim_g2r ) )


sub_cen_M = 0.5 * (Mag_bins[1:] + Mag_bins[:-1])

#. 
divid_F = np.polyfit( sub_cen_M, sub_medi, 1)
divid_lF = np.poly1d( divid_F )

pre_x = np.linspace( -25, -19, 55 )
divid_line = divid_lF( pre_x )


plt.figure()

H_arr, H2_arr, X2_arr, Y2_arr, V_arr = hist2d_pdf_func( tmp_r_Mag, tmp_g2r, bins = [100, 100], 
	levels = levels, smooth = (1.5, 1.5), weights = None,)

plt.scatter( tmp_r_Mag, tmp_g2r, marker = '.', color = 'k', s = 5, label = 'W15 catalog', alpha = 0.03)

plt.contour( X2_arr, Y2_arr, H2_arr.T, np.concatenate([ V_arr, [ H_arr.max() * (1 + 1e-4) ] ] ), 
			colors = _cmap_lis,)

plt.plot( pre_x, divid_line, ls = '--', color = 'darkred', label = 'Red sequence')
plt.plot( pre_x, divid_line - np.median( sub_std ) * 1, ls = ':', color = 'darkred', 
			label = 'Red sequence - 1$\\sigma_{median}$')

plt.legend( loc = 3)
plt.ylim( 0, 2.5)
plt.ylabel('g - r')
plt.xlim( -24.5, -19 )
plt.xlabel('$M_{r}$')
plt.savefig('/home/xkchen/ZLW_sat_g-r_Mr_diag.png', dpi = 300)
plt.close()


plt.figure()

H_arr, H2_arr, X2_arr, Y2_arr, V_arr = hist2d_pdf_func( cp_r_Mag, cp_g2r, bins = [100, 100], 
	levels = levels, smooth = (1.5, 1.5), weights = None,)

plt.scatter( cp_r_Mag, cp_g2r, marker = '.', color = 'k', s = 5, label = 'SDSS redMaPPer', alpha = 0.03)

plt.contour( X2_arr, Y2_arr, H2_arr.T, np.concatenate([ V_arr, [ H_arr.max() * (1 + 1e-4) ] ] ), 
			colors = _cmap_lis,)

plt.errorbar( sub_cen_M, sub_medi, yerr = sub_std, marker = 'o', ls = 'none', color = 'darkred', ecolor = 'darkred', 
		mfc = 'none', mec = 'darkred', alpha = 0.75, capsize = 1.5,)

plt.plot( pre_x, divid_line, ls = '--', color = 'darkred',)

plt.legend( loc = 3)
plt.ylim( 0, 2.5)
plt.ylabel('g - r')
plt.xlim( -24.5, -19 )
plt.xlabel('$M_{r}$')
plt.savefig('/home/xkchen/redMaP_sat_g-r_Mr_diag.png', dpi = 300)
plt.close()

