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

### === 
def mag10_divid_func( cat_arr, sep_F0, sep_F1, px_divid, cx_bins, cx_params, out_file, cat_lis):
	"""
	sep_F0, sep_F1 : the fitting function for data separation
	cat_arr : including the division properties and other control parameters
	cx_bins : the bins division for control properties -- cx_params
	px_divid : at fixed cx, use px to separate sample
	"""

	keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_imag_10', 'R_sat', 'Rs/R200m', 'g-r', 'sat_iMag_10', 'clus_ID']

	Ns = len( cat_arr )

	N_bin = len( cx_bins )
	cen_px = 0.5 * ( cx_bins[1:] + cx_bins[:-1] )

	tmp_arr_0 = []
	tmp_arr_1 = []
	tmp_arr_2 = []

	for tt in range( N_bin - 1 ):

		id_ux = ( cx_params >= cx_bins[ tt ] ) & ( cx_params < cx_bins[ tt + 1 ] )

		sub_px = px_divid[ id_ux ]

		lim_r0 = sep_F0( cen_px[ tt ] )
		lim_r1 = sep_F1( cen_px[ tt ] )

		id_x0 = sub_px <= lim_r0
		id_x1 = (sub_px > lim_r0) & ( sub_px <= lim_r1)
		id_x2 = sub_px > lim_r1

		if tt == 0:
			for pp in range( Ns ):

				_pp_arr = cat_arr[ pp ]

				lim_pp_arr = _pp_arr[ id_ux ]

				tmp_arr_0.append( lim_pp_arr[ id_x0 ] )
				tmp_arr_1.append( lim_pp_arr[ id_x1 ] )
				tmp_arr_2.append( lim_pp_arr[ id_x2 ] )
		else:
			for pp in range( Ns ):

				_pp_arr = cat_arr[ pp ]

				lim_pp_arr = _pp_arr[ id_ux ]

				tmp_arr_0[ pp ] = np.r_[ tmp_arr_0[ pp ], lim_pp_arr[ id_x0 ] ]
				tmp_arr_1[ pp ] = np.r_[ tmp_arr_1[ pp ], lim_pp_arr[ id_x1 ] ]
				tmp_arr_2[ pp ] = np.r_[ tmp_arr_2[ pp ], lim_pp_arr[ id_x2 ] ]


	#. for satellite beyond the binned range
	id_px = ( cx_params >= cx_bins[-1] ) | ( cx_params < cx_bins[0] )

	lim_r0 = sep_F0( cx_params[ id_px ] )
	lim_r1 = sep_F1( cx_params[ id_px ] )

	res_sep_px = px_divid[ id_px ]

	id_x0 = res_sep_px <= lim_r0
	id_x1 = (res_sep_px > lim_r0) & ( res_sep_px <= lim_r1)
	id_x2 = res_sep_px > lim_r1

	for pp in range( Ns ):

		_pp_arr = cat_arr[ pp ]

		lim_pp_arr = _pp_arr[ id_px ]

		tmp_arr_0[ pp ] = np.r_[ tmp_arr_0[ pp ], lim_pp_arr[ id_x0 ] ]
		tmp_arr_1[ pp ] = np.r_[ tmp_arr_1[ pp ], lim_pp_arr[ id_x1 ] ]
		tmp_arr_2[ pp ] = np.r_[ tmp_arr_2[ pp ], lim_pp_arr[ id_x2 ] ]

	##. save array
	#. inner
	fill = dict( zip( keys, tmp_arr_0 ) )
	data = pds.DataFrame( fill )
	data.to_csv( out_file % cat_lis[0] )

	#. middle
	fill = dict( zip( keys, tmp_arr_1 ) )
	data = pds.DataFrame( fill )
	data.to_csv( out_file % cat_lis[1] )

	#. outer
	fill = dict( zip( keys, tmp_arr_2 ) )
	data = pds.DataFrame( fill )
	data.to_csv( out_file % cat_lis[2] )

	return


### === data load
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/'
pos_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/cat/'


### === sample division at fixed i_Mag10

dat = pds.read_csv( cat_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat-aper-mag.csv')
bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra']), np.array( dat['bcg_dec']), np.array( dat['clus_z'])

sat_ra, sat_dec = np.array( dat['ra']), np.array( dat['dec'])

i_mag_10 = np.array( dat['imag_10'])

Dl_z = Test_model.luminosity_distance( bcg_z ).value
D_modu = 5 * ( np.log10( Dl_z * 1e6 ) - 1 )

i_Mag_10 = i_mag_10 - D_modu


#. radius information
pat = pds.read_csv( cat_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv')

pc_ra, pc_dec, pc_z = np.array( pat['bcg_ra']), np.array( pat['bcg_dec']), np.array( pat['bcg_z'])
p_ra, p_dec = np.array( pat['ra']), np.array( pat['dec'])

p_clus_ID = np.array( pat['clus_ID'] )
p_R_sat = np.array( pat['Rcen/Rv'])

p_Rs_phy = np.array( pat['R_cen'] )
p_Rs_phy = p_Rs_phy * 1e3 / h   ## kpc

p_g2r = np.array( pat['g-r'])



N_bin = 9
Mag_bins = np.linspace( -23, -20, N_bin )

tmp_R0, tmp_R1 = [], []

tmp_mag_0, tmp_mag_1, tmp_mag_2 = np.array( [] ), np.array( [] ), np.array( [] )

for tt in range( N_bin - 1 ):

	if tt == N_bin - 1:
		idx = ( i_Mag_10 >= Mag_bins[tt] ) & ( i_Mag_10 <= Mag_bins[tt + 1] )

	else:
		idx = ( i_Mag_10 >= Mag_bins[tt] ) & ( i_Mag_10 < Mag_bins[tt + 1] )

	sub_Rs = p_R_sat[ idx ]
	sub_i_mag = i_Mag_10[ idx ]

	#. pre-division (make sure the number is soughly equal)
	lim_a0 = np.percentile( sub_Rs, 33.4 )
	lim_a1 = np.percentile( sub_Rs, 66.7 )

	tmp_R0.append( lim_a0 )
	tmp_R1.append( lim_a1 )

	#. sub-divide
	tmp_mag_0 = np.r_[ tmp_mag_0, sub_i_mag[ sub_Rs <= lim_a0 ] ]
	tmp_mag_1 = np.r_[ tmp_mag_1, sub_i_mag[ (sub_Rs > lim_a0) & ( sub_Rs <= lim_a1) ] ]
	tmp_mag_2 = np.r_[ tmp_mag_2, sub_i_mag[ sub_Rs > lim_a1 ] ]

print( tmp_mag_0.shape, tmp_mag_1.shape, tmp_mag_2.shape )


cen_point = 0.5 * ( Mag_bins[1:] + Mag_bins[:-1] )

Mag_xs = np.linspace( -25, -19, 50 )

fit_F_0 = np.polyfit( cen_point, tmp_R0, deg = 1)
Pf_0 = np.poly1d( fit_F_0 )
line_0 = Pf_0( Mag_xs )

fit_F_1 = np.polyfit( cen_point, tmp_R1, deg = 1)
Pf_1 = np.poly1d( fit_F_1 )
line_1 = Pf_1( Mag_xs )


#. save the parameters
keys = ['k_0', 'b_0', 'k_1', 'b_1']
values = [ Pf_0[1], Pf_0[0], Pf_1[1], Pf_1[0] ]
fill = dict( zip(keys, values) )
data = pds.DataFrame( fill, index = ['k', 'v'] )
data.to_csv( out_path + 'radii-bin_fixed-Mag10_separate_line.csv' )


def pre_view():

	plt.figure()

	plt.plot( i_Mag_10, p_R_sat, '.', color = 'k', alpha = 0.01)

	plt.plot( cen_point, tmp_R0, 'bo', alpha = 0.5,)
	plt.plot( Mag_xs, line_0, 'b-', alpha = 0.5,)

	plt.plot( cen_point, tmp_R1, 'rs', alpha = 0.5,)
	plt.plot( Mag_xs, line_1, 'r-', alpha = 0.5,)

	plt.xlim( -23, -19 )
	plt.xlabel('$Mag_{i, \; 10}$')
	plt.ylabel('$R \, / \, R_{200m}$')
	plt.savefig('/home/xkchen/pre_Mag_i_R-ov-R200m.png', dpi = 300)
	plt.close()


	plt.figure()
	plt.hist( tmp_mag_0, bins = 55, density = False, histtype = 'step', color = 'b', label = 'Inner')
	plt.axvline( x = np.median( tmp_mag_0), ls = '--', color = 'b', label = 'Median', alpha = 0.5,)
	plt.axvline( x = np.mean( tmp_mag_0), ls = '-', color = 'b', label = 'Mean', alpha = 0.5,)

	plt.hist( tmp_mag_1, bins = 55, density = False, histtype = 'step', color = 'g', label = 'Middle')
	plt.axvline( x = np.median( tmp_mag_1), ls = '--', color = 'g', alpha = 0.5,)
	plt.axvline( x = np.mean( tmp_mag_1), ls = '-', color = 'g', alpha = 0.5,)

	plt.hist( tmp_mag_2, bins = 55, density = False, histtype = 'step', color = 'r', label = 'Outer')
	plt.axvline( x = np.median( tmp_mag_2), ls = '--', color = 'r', alpha = 0.5,)
	plt.axvline( x = np.mean( tmp_mag_2), ls = '-', color = 'r', alpha = 0.5,)

	plt.legend( loc = 2 )
	plt.xlabel('$Mag_{i,\;10}$')
	plt.savefig('/home/xkchen/pre_radii_bin.png', dpi = 300)
	plt.close()

	return

# pre_view()


##. division at fixed i_Mag10
cat_arr = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, i_mag_10, p_Rs_phy, p_R_sat, p_g2r, i_Mag_10, p_clus_ID ]

px_divid = p_R_sat + 0.

cx_bins = np.linspace( -23, -20, 19 )
cx_params = i_Mag_10 + 0.

cat_lis = ['inner', 'middle', 'outer']
out_file = out_path + 'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_member.csv'

mag10_divid_func( cat_arr, Pf_0, Pf_1, px_divid, cx_bins, cx_params, out_file, cat_lis)

pp_Rs, pp_iMag = [], []

for pp in range( 3 ):

	dat = pds.read_csv( out_file % cat_lis[pp] )

	pp_Rs.append( np.array( dat['Rs/R200m'] ) )
	pp_iMag.append( np.array( dat['sat_iMag_10'] ) )

print( [len(ll) for ll in pp_Rs] )


### === stacking image information match
for pp in range( 3 ):

	s_dat = pds.read_csv( out_file % cat_lis[pp] )
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
		data.to_csv( out_path + 'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_member_%s-band_pos-zref.csv' % (cat_lis[pp], band[kk]) )

raise

### === Ng record for Background stacking
cat_lis = ['inner', 'middle', 'outer']
out_file = out_path + 'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_member.csv'

for kk in range( 3 ):

	dat = pds.read_csv( out_file % cat_lis[ kk ] )

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	kk_IDs = np.array( dat['clus_ID'] )

	d_IDs = list( set( kk_IDs ) )

	N_kk = len( d_IDs )

	tmp_ra, tmp_dec, tmp_z = np.zeros( N_kk, ), np.zeros( N_kk, ), np.zeros( N_kk, )
	tmp_ng = np.zeros( N_kk, )

	for pp in range( N_kk ):

		id_vx = kk_IDs == d_IDs[ pp ]
		n_pp = np.sum( id_vx )

		tmp_ra[ pp ] = np.mean( bcg_ra[ id_vx ] )
		tmp_dec[ pp ] = np.mean( bcg_dec[ id_vx ] )
		tmp_z[ pp ] = np.mean( bcg_z[ id_vx ] )
		tmp_ng[ pp ] = np.mean( n_pp )

	#.
	keys = ['ra', 'dec', 'z', 'Ng_80']
	values = [ tmp_ra, tmp_dec, tmp_z, tmp_ng ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( out_path + 'frame-lim_Pm-cut_exlu-BCG_iMag10-fix_%s_member_Ng.csv' % cat_lis[kk],)

