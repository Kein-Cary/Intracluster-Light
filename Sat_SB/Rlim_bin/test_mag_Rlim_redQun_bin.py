import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pds
import h5py

import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable
from scipy import optimize
from scipy import signal
from scipy import interpolate as interp
from scipy import integrate as integ
from scipy import stats as sts

#.
from img_sat_fig_out_mode import zref_sat_pos_func


###... cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)

band = ['r', 'g', 'i']


### === data load
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/'
pos_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/cat/'


##.. fixed radius bin + fixed iMag_10 + joint red sequence division

pat = pds.read_csv( cat_path + 'Extend-BCGM_rgi-common_member-cat.csv')
# pat = pds.read_csv( cat_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv')

pc_ra, pc_dec, pc_z = np.array( pat['bcg_ra']), np.array( pat['bcg_dec']), np.array( pat['bcg_z'])
p_ra, p_dec = np.array( pat['ra'] ), np.array( pat['dec'] )

p_R_sat = np.array( pat['Rcen/Rv'] )
p_g2r = np.array( pat['g-r'] )

p_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )


#. sdss table
dat = fits.open('/home/xkchen/figs/extend_Zphoto_cat/zphot_01_033_cat/redMaPPer_z-phot_0.1-0.33_member_params.fit')
data = dat[1].data

c_ra, c_dec = data['ra'], data['dec']
abs_Mag_r = data['absMagR']
frac_DeV_r = data['fracDeV_r']


c_coord = SkyCoord( ra = c_ra * U.deg, dec = c_dec * U.deg )

idx, sep, d3d = p_coord.match_to_catalog_sky( c_coord )
id_lim = sep.value < 2.7e-4

mp_ra, mp_dec = c_ra[ idx[ id_lim ] ], c_dec[ idx[ id_lim ] ]
mp_abs_Mag = abs_Mag_r[ idx[ id_lim ] ]
mp_frac_dev = frac_DeV_r[ idx[ id_lim ] ]


### === get redsequence
def lg_linear_func( mag_x, a, b):
	x_color = a * mag_x + b
	return x_color

def resi_func( po, x_mag, y_color):
	a, b = po[:]
	x_color = lg_linear_func( x_mag, a, b)
	delta = y_color - x_color
	return delta

#. red-sequence need to seperate redshift bins
N_z = 6

z_bins = np.linspace(0.2, 0.3, 6)

"""
for kk in range( N_z - 1 ):

	if kk != N_z - 2:
		id_zx = ( pc_z >= z_bins[ kk ] ) & ( pc_z < z_bins[ kk+1 ] )

	else:
		id_zx = ( pc_z >= z_bins[ kk ] ) & ( pc_z <= z_bins[ kk+1 ] )

	kk_g2r = p_g2r[ id_zx ]
	kk_abs_Mag = mp_abs_Mag[ id_zx ]

	def point_select( lim_z0, lim_z1 ):

		id_vx = ( kk_abs_Mag <= -18 ) & ( kk_abs_Mag >= -23 )

		cp_r_Mag = kk_abs_Mag[ id_vx ]
		cp_g2r = kk_g2r[ id_vx ]

		sigma = 3
		kk = 0

		id_R_lim = [ 1 ]
		sum_dex = np.sum( id_R_lim )

		while sum_dex > 0:

			put_x = cp_r_Mag

			p0 = [ -0.08, -0.32 ]
			res_lsq = optimize.least_squares( resi_func, x0 = np.array( p0 ), 
				jac = '3-point', loss = 'linear', 
				f_scale = 0.1, args = ( put_x, cp_g2r),)

			a_fit = res_lsq.x[0]
			b_fit = res_lsq.x[1]


			#... fitting g-r color
			fit_g2r = lg_linear_func( put_x, a_fit, b_fit )

			Var = np.sum( (fit_g2r - cp_g2r)**2 ) / len( cp_g2r )
			sigma = np.sqrt( Var )
			sp_R = sts.spearmanr( fit_g2r, cp_g2r)[0]


			#... fit relation between fitting and obs.
			po = [0.9, 10]
			popt, pcov = optimize.curve_fit( lg_linear_func, xdata = fit_g2r, ydata = cp_g2r, p0 = po,)
			_a0, _b0 = popt

			com_line = lg_linear_func( fit_g2r, _a0, _b0 )

			dR_com_l = np.abs( _a0 * fit_g2r + _b0 - cp_g2r ) / np.sqrt( 1 + _a0**2 )
			id_R_lim = dR_com_l >= 1.5 * sigma

			#. sum of residual
			sum_dex = np.sum( id_R_lim )


			plt.figure()
			ax = plt.subplot(111)
			ax.set_title('least square')

			ax.scatter( fit_g2r, cp_g2r, marker = 'o', s = 1.5, color = 'k', alpha = 0.05, zorder = 100)
			ax.plot( fit_g2r, fit_g2r, 'b--',)

			ax.scatter( fit_g2r[id_R_lim], cp_g2r[id_R_lim], marker = '*', s = 5.5, color = 'g',)
			ax.annotate( s = '$\\sigma = %.3f\, ; \; R = %.3f$' % (sigma, sp_R), xy = (0.65, 0.05), 
				xycoords = 'axes fraction',)

			ax.set_xlabel('$ fit \; : \; g-r $')
			ax.set_ylabel('$ data \; : \; g-r $')
			plt.savefig('/home/xkchen/Downloads/points_selected_%d.png' % kk, dpi = 300)
			plt.close()


			plt.figure()
			ax = plt.subplot(111)
			ax.set_title('least square')

			ax.scatter( cp_r_Mag, cp_g2r, marker = 'o', s = 1.5, color = 'k', alpha = 0.05, zorder = 100)
			ax.plot( cp_r_Mag, fit_g2r, 'b-',)

			ax.scatter( cp_r_Mag[id_R_lim], cp_g2r[id_R_lim], marker = '*', s = 5.5, color = 'g',)

			ax.set_xlabel('$ M_{r} $')
			ax.set_ylabel('$ g-r $')
			plt.savefig('/home/xkchen/Downloads/CMD_selected_%d.png' % kk, dpi = 300)
			plt.close()


			cp_r_Mag = cp_r_Mag[ id_R_lim == False ]
			cp_g2r = cp_g2r[ id_R_lim == False ]

			kk += 1


			#. save
			keys = ['g-r', 'r_Mag']
			values= [ cp_g2r, cp_r_Mag ]
			fill = dict( zip( keys, values ) )
			data = pds.DataFrame( fill )
			data.to_csv( '/home/xkchen/Downloads/red_sqequence/z%.2f_%.2f_red_sequence_points.txt' % (lim_z0, lim_z1),)

			keys = ['a_x', 'b', 'sigma']
			values = [ a_fit, b_fit, sigma ]
			fill = dict( zip( keys, values) )
			out_data = pds.DataFrame( fill, index = ['k', 'v'])
			out_data.to_csv( '/home/xkchen/Downloads/red_sqequence/z%.2f_%.2f_red_sequence_func.txt' % (lim_z0, lim_z1),)

	point_select( z_bins[ kk ], z_bins[ kk+1 ] )

"""


##. redsquence compare
Mag_xs = np.linspace( -25, -16, 100)

# for kk in range( N_z - 1 ):

# 	if kk != N_z - 2:
# 		id_zx = ( pc_z >= z_bins[ kk ] ) & ( pc_z < z_bins[ kk+1 ] )

# 	else:
# 		id_zx = ( pc_z >= z_bins[ kk ] ) & ( pc_z <= z_bins[ kk+1 ] )

# 	kk_g2r = p_g2r[ id_zx ]
# 	kk_abs_Mag = mp_abs_Mag[ id_zx ]

# 	lim_z0, lim_z1 = z_bins[ kk ], z_bins[ kk+1 ]

# 	fat = pds.read_csv('/home/xkchen/Downloads/red_sqequence/z%.2f_%.2f_red_sequence_func.txt' % (lim_z0, lim_z1),)
# 	a_fit, b_fit = np.array( fat['a_x'] )[0], np.array( fat['b'] )[0]
# 	sigma_fit = np.array( fat['sigma'] )[0]


# 	plt.figure()

# 	plt.title( 'z%.2f -- %.2f' % (lim_z0, lim_z1),)
# 	plt.plot( kk_abs_Mag, kk_g2r, 'k.', alpha = 0.01,)

# 	plt.plot( Mag_xs, lg_linear_func( Mag_xs, a_fit, b_fit), 'c-', )
# 	plt.plot( Mag_xs, lg_linear_func( Mag_xs, a_fit, b_fit) + sigma_fit, 'c:', )
# 	plt.plot( Mag_xs, lg_linear_func( Mag_xs, a_fit, b_fit) - sigma_fit, 'c:', )

# 	plt.ylim( 0, 3 )
# 	plt.ylabel('g - r')
# 	plt.xlim( -24, -16 )
# 	plt.xlabel('$M_{r}$')

# 	plt.savefig('/home/xkchen/z%.2f_%.2f_g2r_r_Mag.png' % (lim_z0, lim_z1), dpi = 300)
# 	plt.close()



##. fixed i_Mag10 subsamples
cat_lis = ['inner', 'middle', 'outer']

pre_file = out_path + 'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_member.csv'

out_low_file = out_path + 'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_below-red_member.csv'
out_up_file = out_path + 'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_above-red_member.csv'


N_binx = 15
cx_bins = np.linspace( -23, -18, N_binx )
medi_mx = 0.5 * ( cx_bins[1:] + cx_bins[:-1] )

medi_off = [ 0.35, 0.21, 0.20 ]

for kk in range( 3 ):

	s_dat = pds.read_csv( pre_file % cat_lis[kk],)

	bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
	s_ra, s_dec = np.array( s_dat['sat_ra'] ), np.array( s_dat['sat_dec'] )	
	s_clus_ID = np.array( s_dat['clus_ID'] )

	sub_coord = SkyCoord( ra = s_ra * U.deg, dec = s_dec * U.deg )

	#.
	idx, sep, d3d = sub_coord.match_to_catalog_sky( p_coord )
	id_lim = sep.value < 2.7e-4

	cp_ra, cp_dec = p_ra[ idx[ id_lim ] ], p_dec[ idx[ id_lim ] ]
	cp_abs_Mag = mp_abs_Mag[ idx[ id_lim ] ]	
	cp_g2r = p_g2r[ idx[ id_lim ] ]



	low_ra, low_dec = np.array( [] ), np.array( [] )
	low_bcg_ra, low_bcg_dec, low_bcg_z = np.array( [] ), np.array( [] ), np.array( [] )
	low_g2r, low_abs_Mag = np.array( [] ), np.array( [] )
	low_clus_ID = np.array( [] )


	up_ra, up_dec = np.array( [] ), np.array( [] )
	up_bcg_ra, up_bcg_dec, up_bcg_z = np.array( [] ), np.array( [] ), np.array( [] )
	up_g2r, up_abs_Mag = np.array( [] ), np.array( [] )
	up_clus_ID = np.array( [] )


	##.. red sequence in different z_bins
	for tt in range( N_z - 1 ):

		if tt != N_z - 2:
			id_zx = ( bcg_z >= z_bins[ tt ] ) & ( bcg_z < z_bins[ tt+1 ] )

		else:
			id_zx = ( bcg_z >= z_bins[ tt ] ) & ( bcg_z <= z_bins[ tt+1 ] )

		kk_g2r = cp_g2r[ id_zx ]
		kk_abs_Mag = cp_abs_Mag[ id_zx ]

		kk_bcg_ra = bcg_ra[ id_zx ]
		kk_bcg_dec = bcg_dec[ id_zx ]
		kk_bcg_z = bcg_z[ id_zx ]
		kk_clus_ID = s_clus_ID[ id_zx ]
		kk_sat_ra = s_ra[ id_zx ]
		kk_sat_dec = s_dec[ id_zx ]


		lim_z0, lim_z1 = z_bins[ tt ], z_bins[ tt+1 ]

		fat = pds.read_csv('/home/xkchen/Downloads/red_sqequence/z%.2f_%.2f_red_sequence_func.txt' % (lim_z0, lim_z1),)
		a_fit, b_fit = np.array( fat['a_x'] )[0], np.array( fat['b'] )[0]
		sigma_fit = np.array( fat['sigma'] )[0]


		dd_lo_g2r, dd_lo_Mag = np.array( [] ), np.array( [] )
		dd_hi_g2r, dd_hi_Mag = np.array( [] ), np.array( [] )

		for nn in range( N_binx - 1 ):

			id_magx = ( kk_abs_Mag >= cx_bins[ nn ] ) & ( kk_abs_Mag < cx_bins[ nn+1 ])

			tt_magx = kk_abs_Mag[ id_magx ]
			tt_g2r = kk_g2r[ id_magx ]


			tt_bcg_ra = kk_bcg_ra[ id_magx ]
			tt_bcg_dec = kk_bcg_dec[ id_magx ]
			tt_bcg_z = kk_bcg_z[ id_magx ]
			tt_clus_ID = kk_clus_ID[ id_magx ]
			tt_sat_ra = kk_sat_ra[ id_magx ]
			tt_sat_dec = kk_sat_dec[ id_magx ]


			tt_medi_magx = np.median( tt_magx )
			if np.isfinite( tt_medi_magx ):
				div_gr = lg_linear_func( tt_medi_magx, a_fit, b_fit)
			else:
				div_gr = lg_linear_func( medi_mx[ nn ], a_fit, b_fit)

			id_ux_0 = tt_g2r > div_gr + sigma_fit * medi_off[ kk ]
			id_ux_1 = tt_g2r <= div_gr + sigma_fit * medi_off[ kk ]

			dd_lo_g2r = np.r_[ dd_lo_g2r, tt_g2r[ id_ux_1 ] ]
			dd_lo_Mag = np.r_[ dd_lo_Mag, tt_magx[ id_ux_1 ] ] 

			dd_hi_g2r = np.r_[ dd_hi_g2r, tt_g2r[ id_ux_0 ] ]
			dd_hi_Mag = np.r_[ dd_hi_Mag, tt_magx[ id_ux_0 ] ]


			#. save array
			low_ra = np.r_[ low_ra, tt_sat_ra[ id_ux_1 ] ]
			low_dec = np.r_[ low_dec, tt_sat_dec[ id_ux_1 ] ]

			low_bcg_ra = np.r_[ low_bcg_ra, tt_bcg_ra[ id_ux_1 ] ]
			low_bcg_dec = np.r_[ low_bcg_dec, tt_bcg_dec[ id_ux_1 ] ]
			low_bcg_z = np.r_[ low_bcg_z, tt_bcg_z[ id_ux_1 ] ]

			low_clus_ID = np.r_[ low_clus_ID, tt_clus_ID[ id_ux_1 ] ]


			up_ra = np.r_[ up_ra, tt_sat_ra[ id_ux_0 ] ]
			up_dec = np.r_[ up_dec, tt_sat_dec[ id_ux_0 ] ]

			up_bcg_ra = np.r_[ up_bcg_ra, tt_bcg_ra[ id_ux_0 ] ]
			up_bcg_dec = np.r_[ up_bcg_dec, tt_bcg_dec[ id_ux_0 ] ]
			up_bcg_z = np.r_[ up_bcg_z, tt_bcg_z[ id_ux_0 ] ]

			up_clus_ID = np.r_[ up_clus_ID, tt_clus_ID[ id_ux_0 ] ]


		##.
		id_magx = ( kk_abs_Mag >= cx_bins[-1] ) | ( kk_abs_Mag < cx_bins[0] )

		tt_magx = kk_abs_Mag[ id_magx ]
		tt_g2r = kk_g2r[ id_magx ]

		tt_bcg_ra = kk_bcg_ra[ id_magx ]
		tt_bcg_dec = kk_bcg_dec[ id_magx ]
		tt_bcg_z = kk_bcg_z[ id_magx ]
		tt_clus_ID = kk_clus_ID[ id_magx ]
		tt_sat_ra = kk_sat_ra[ id_magx ]
		tt_sat_dec = kk_sat_dec[ id_magx ]


		tt_medi_magx = np.median( tt_magx )
		div_gr = lg_linear_func( tt_medi_magx, a_fit, b_fit)

		id_ux_0 = tt_g2r > div_gr
		id_ux_1 = tt_g2r <= div_gr

		dd_lo_g2r = np.r_[ dd_lo_g2r, tt_g2r[ id_ux_1 ] ]
		dd_lo_Mag = np.r_[ dd_lo_Mag, tt_magx[ id_ux_1 ] ]

		dd_hi_g2r = np.r_[ dd_hi_g2r, tt_g2r[ id_ux_0 ] ]
		dd_hi_Mag = np.r_[ dd_hi_Mag, tt_magx[ id_ux_0 ] ]


		low_g2r = np.r_[ low_g2r, dd_lo_g2r ]
		low_abs_Mag = np.r_[ low_abs_Mag, dd_lo_Mag ]
		
		up_g2r = np.r_[ up_g2r, dd_hi_g2r ]
		up_abs_Mag = np.r_[ up_abs_Mag, dd_hi_Mag ]


		#. save array
		low_ra = np.r_[ low_ra, tt_sat_ra[ id_ux_1 ] ]
		low_dec = np.r_[ low_dec, tt_sat_dec[ id_ux_1 ] ]

		low_bcg_ra = np.r_[ low_bcg_ra, tt_bcg_ra[ id_ux_1 ] ]
		low_bcg_dec = np.r_[ low_bcg_dec, tt_bcg_dec[ id_ux_1 ] ]
		low_bcg_z = np.r_[ low_bcg_z, tt_bcg_z[ id_ux_1 ] ]

		low_clus_ID = np.r_[ low_clus_ID, tt_clus_ID[ id_ux_1 ] ]


		up_ra = np.r_[ up_ra, tt_sat_ra[ id_ux_0 ] ]
		up_dec = np.r_[ up_dec, tt_sat_dec[ id_ux_0 ] ]

		up_bcg_ra = np.r_[ up_bcg_ra, tt_bcg_ra[ id_ux_0 ] ]
		up_bcg_dec = np.r_[ up_bcg_dec, tt_bcg_dec[ id_ux_0 ] ]
		up_bcg_z = np.r_[ up_bcg_z, tt_bcg_z[ id_ux_0 ] ]

		up_clus_ID = np.r_[ up_clus_ID, tt_clus_ID[ id_ux_0 ] ]


		# div_line = lg_linear_func( Mag_xs, a_fit, b_fit)

		# plt.figure()

		# plt.plot( Mag_xs, div_line, 'c--', alpha = 0.5,)

		# plt.plot( dd_lo_Mag, dd_lo_g2r, 'b.', alpha = 0.05,)
		# plt.plot( dd_hi_Mag, dd_hi_g2r, 'r.', alpha = 0.05,)

		# plt.ylim( 0, 3 )
		# plt.ylabel('g - r')
		# plt.xlim( -24, -16 )
		# plt.xlabel('$M_{r}$')

		# plt.savefig('/home/xkchen/%s_%d_subsample_g2r_r_Mag.png' % (cat_lis[kk], tt), dpi = 300)
		# plt.close()


	#. save
	keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'clus_ID']
	values = [ low_bcg_ra, low_bcg_dec, low_bcg_z, low_ra, low_dec, low_clus_ID ]
	fill = dict( zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_path + 'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_below-red_member.csv' % cat_lis[kk] )


	keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'clus_ID']
	values = [ up_bcg_ra, up_bcg_dec, up_bcg_z, up_ra, up_dec, up_clus_ID ]
	fill = dict( zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_path + 'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_above-red_member.csv' % cat_lis[kk] )

	print( len(low_g2r) )
	print( len(up_g2r) )

raise


### === stacking image information match
id_redQ = ['below', 'above']

for ll in range( 2 ):

	for pp in range( 3 ):

		s_dat = pds.read_csv( out_path + 'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_%s-red_member.csv' % ( cat_lis[pp], id_redQ[ll] ),)
		bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
		p_ra, p_dec = np.array( s_dat['sat_ra'] ), np.array( s_dat['sat_dec'] )	

		pre_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

		for kk in range( 3 ):

			#. z-ref position
			dat = pds.read_csv( pos_path + 'Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos_z-ref.csv' % band[ kk ])
			kk_ra, kk_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
			kk_imgx, kk_imgy = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

			kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

			idx, sep, d3d = pre_coord.match_to_catalog_sky( kk_coord )
			id_lim = sep.value < 2.7e-4

			mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]	
			mp_imgx, mp_imgy = kk_imgx[ idx[ id_lim ] ], kk_imgy[ idx[ id_lim ] ]

			keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_x', 'sat_y']
			values = [ bcg_ra, bcg_dec, bcg_z, p_ra, p_dec, mp_imgx, mp_imgy ]
			fill = dict( zip( keys, values ) )
			data = pds.DataFrame( fill )
			data.to_csv( out_path + 'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_%s-red_member_%s-band_pos-zref.csv' % (cat_lis[pp], id_redQ[ll], band[kk]) )

