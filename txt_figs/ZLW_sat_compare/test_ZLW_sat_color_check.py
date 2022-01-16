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

### === ### cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

pixel = 0.396
band = ['r', 'g', 'i']
L_wave = np.array([ 6166, 4686, 7480 ])
Mag_sun = [ 4.65, 5.11, 4.53 ]
z_ref = 0.25

##### ===== ##### sample comparison
# cat_lis = [ 'low-age', 'hi-age' ]
# fig_name = ['Low $t_{ \\mathrm{age} } \\mid M_{\\ast}^{\\mathrm{BCG}}$', 
# 			'High $t_{ \\mathrm{age} } \\mid M_{\\ast}^{\\mathrm{BCG}}$']

# cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
# fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 
# 			'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']

cat_lis = [ 'low-rich', 'hi-rich' ]
fig_name = ['Low $ \\lambda \\mid M_{\\ast}^{\\mathrm{BCG}} $', 
			'High $ \\lambda \\mid M_{\\ast}^{\\mathrm{BCG}} $']


#. BCG color
# clus_dat = pds.read_csv( '/home/xkchen/figs/sat_cat_ZLW/ZLW_cat_clust_sql_match_cat.csv' )
clus_dat = pds.read_csv( '/home/xkchen/figs/sat_cat_ZLW/ZLW_cat_clust_BCG-Mstar_rich_sql_match_cat.csv' )

orin_dex = np.array( clus_dat['clust_id'] )
clus_ra, clus_dec, clus_z = np.array( clus_dat['ra'] ), np.array( clus_dat['dec'] ), np.array( clus_dat['clus_z'] )
clus_R500, clus_rich = np.array( clus_dat['R500c'] ), np.array( clus_dat['rich'] )
clus_Ng, clus_zf, clus_div = np.array( clus_dat['N500c'] ), np.array( clus_dat['z_flag'] ), np.array( clus_dat['samp_id'] )

bcg_ra, bcg_dec = np.array( clus_dat['bcg_ra'] ), np.array( clus_dat['bcg_dec'] )
bcg_r_mag, bcg_g_mag = np.array( clus_dat['bcg_r_mag'] ), np.array( clus_dat['bcg_g_mag'] )
bcg_i_mag = np.array( clus_dat['bcg_i_mag'] )
bcg_r_cmag, bcg_g_cmag = np.array( clus_dat['bcg_r_cmag'] ), np.array( clus_dat['bcg_g_cmag'] )
bcg_i_cmag = np.array( clus_dat['bcg_i_cmag'] )

#. color properties compasison
dex_lo = clus_div == 1
lo_bcg_gr = bcg_g_mag[ dex_lo ] - bcg_r_mag[ dex_lo ]
hi_bcg_gr = bcg_g_mag[ dex_lo == False ] - bcg_r_mag[ dex_lo == False ]


#. satellite color
# R_cri = np.array( [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0 ] )
R_cri = np.linspace( 0, 1.2, 9)
N_ri = len( R_cri )

"""
#. satellites radii bins color
for ll in range( 2 ):

	clust_mem_file = '/home/xkchen/figs/sat_cat_ZLW/mem_match/ZLW_%s-band_ra%.3f_dec%.3f_z%.3f_members_mag.csv'
	stacked_cat_file = '/home/xkchen/figs/sat_cat_ZLW/ZLWen_%s_match_cat.csv' % cat_lis[ ll ]

	d_cat = pds.read_csv( stacked_cat_file )
	ra, dec, z = np.array( d_cat['ra']), np.array( d_cat['dec']), np.array( d_cat['z'])	

	N_z = len( z )

	for jj in range( N_ri - 1 ):

		cri_Pm = []
		cri_gr, cri_ri, cri_gi = [], [], []
		dered_cri_gr, dered_cri_ri, dered_cri_gi = [], [], []

		for ii in range( N_z ):

			ra_g, dec_g, z_g = ra[ii], dec[ii], z[ii]

			sub_dat = pds.read_csv( clust_mem_file % ( 'r', ra_g, dec_g, z_g),)
			sub_cen_R = np.array( sub_dat['centric_R(Mpc/h)'] )
			sub_Pmem = np.array( sub_dat['P_member'] )

			sub_r_mag = np.array(sub_dat['r_mags'])
			sub_g_mag = np.array(sub_dat['g_mags'])
			sub_i_mag = np.array(sub_dat['i_mags'])

			dered_sub_r_mag = np.array(sub_dat['dered_r_mags'])
			dered_sub_g_mag = np.array(sub_dat['dered_g_mags'])
			dered_sub_i_mag = np.array(sub_dat['dered_i_mags'])

			id_lim = ( sub_cen_R > R_cri[ jj ] ) & ( sub_cen_R <= R_cri[ jj + 1] )

			#. except mags smaller than 0 (due to false obs.)
			id_nul = sub_r_mag < 0
			sub_r_mag[ id_nul ] = np.nan
			id_nul = sub_g_mag < 0
			sub_g_mag[ id_nul ] = np.nan
			id_nul = sub_i_mag < 0
			sub_i_mag[ id_nul ] = np.nan

			id_nul = dered_sub_r_mag < 0
			dered_sub_r_mag[ id_nul ] = np.nan
			id_nul = dered_sub_g_mag < 0
			dered_sub_g_mag[ id_nul ] = np.nan
			id_nul = dered_sub_i_mag < 0
			dered_sub_i_mag[ id_nul ] = np.nan

			limd_Pm = sub_Pmem[ id_lim ]
			limd_gr = sub_g_mag[ id_lim ] - sub_r_mag[ id_lim ]
			limd_ri = sub_r_mag[ id_lim ] - sub_i_mag[ id_lim ]
			limd_gi = sub_g_mag[ id_lim ] - sub_i_mag[ id_lim ]

			limd_dered_gr = dered_sub_g_mag[ id_lim ] - dered_sub_r_mag[ id_lim ]
			limd_dered_ri = dered_sub_r_mag[ id_lim ] - dered_sub_i_mag[ id_lim ]
			limd_dered_gi = dered_sub_g_mag[ id_lim ] - dered_sub_i_mag[ id_lim ]

			cri_Pm.append( limd_Pm )
			cri_gr.append( limd_gr )
			cri_ri.append( limd_ri )
			cri_gi.append( limd_gi )

			dered_cri_gr.append( limd_dered_gr )
			dered_cri_ri.append( limd_dered_ri )
			dered_cri_gi.append( limd_dered_gi )

		f_tree = h5py.File( '/home/xkchen/figs/sat_cat_ZLW/radii_bins/' + 
		'ZLW_cat_%s_within-%.1f-Mpc_mem_color.h5' % ( cat_lis[ll], R_cri[ jj + 1 ] ), 'w')
		for ii in range( N_z ):

			out_arr = np.array( [ cri_Pm[ii], cri_gr[ii], cri_ri[ii], cri_gi[ii], 
								dered_cri_gr[ii], dered_cri_ri[ii], dered_cri_gi[ii] ] )
			gk = f_tree.create_group( "clust_%d/" % ii )
			dk0 = gk.create_dataset( "arr", data = out_arr )

		f_tree.close()

"""

for jj in range( N_ri - 1 ):

	C_mem_low = []
	C_mem_hi = []

	dered_C_mem_low = []
	dered_C_mem_hi = []

	with h5py.File( '/home/xkchen/figs/sat_cat_ZLW/radii_bins/' + 
		'ZLW_cat_%s_within-%.1f-Mpc_mem_color.h5' % ( cat_lis[0], R_cri[ jj+1 ] ), 'r') as f:
		keys = list( f.keys() )
		_Ns_ = len( keys )

		for tt in range( _Ns_ ):
			C_mem_low.append( f[ "clust_%d/arr" % tt ][()][1] )
			dered_C_mem_low.append( f[ "clust_%d/arr" % tt ][()][4] )

	with h5py.File( '/home/xkchen/figs/sat_cat_ZLW/radii_bins/' + 
		'ZLW_cat_%s_within-%.1f-Mpc_mem_color.h5' % ( cat_lis[1], R_cri[ jj+1 ] ), 'r') as f:
		keys = list( f.keys() )
		_Ns_ = len( keys )

		for tt in range( _Ns_ ):
			C_mem_hi.append( f[ "clust_%d/arr" % tt ][()][1] )
			dered_C_mem_hi.append( f[ "clust_%d/arr" % tt ][()][4] )

	Len_x0 = np.array( [ len(ll) for ll in C_mem_low ] )
	id_lim0 = Len_x0 > 0
	lim_C_low = np.array( C_mem_low )[ id_lim0 ]
	lim_dered_C_low = np.array( dered_C_mem_low )[ id_lim0 ]

	Len_x1 = np.array( [ len(ll) for ll in C_mem_hi ] )
	id_lim1 = Len_x1 > 0
	lim_C_hi = np.array( C_mem_hi )[ id_lim1 ]
	lim_dered_C_hi = np.array( dered_C_mem_hi )[ id_lim1 ]


	c_min_0 = [ np.nanmin(ll) for ll in lim_C_low ]
	c_min_1 = [ np.nanmin(ll) for ll in lim_C_hi ]
	c_min = np.min( np.r_[ c_min_0, c_min_1 ] )

	c_max_0 = [ np.nanmax(ll) for ll in lim_C_low ]
	c_max_1 = [ np.nanmax(ll) for ll in lim_C_hi ]
	c_max = np.max( np.r_[ c_max_0, c_max_1 ] )

	c_bins = np.linspace( c_min, c_max, 55 )
	c_x = 0.5 * (c_bins[:-1] + c_bins[1:])

	Ns1 = len( lim_C_low )

	tmp_lo_c, tmp_hi_c = np.array([]), np.array([])
	tmp_lo_dered_c, tmp_hi_dered_c = np.array([]), np.array([])

	for tt in range( Ns1 ):

		id_nan = np.isnan( lim_C_low[tt] )
		tmp_lo_c = np.r_[ tmp_lo_c, lim_C_low[tt][ id_nan == False] ]

		id_nan = np.isnan( lim_dered_C_low[tt] )
		tmp_lo_dered_c = np.r_[ tmp_lo_dered_c, lim_dered_C_low[tt][ id_nan == False] ]

	Ns2 = len( lim_C_hi )

	for tt in range( Ns2 ):

		id_nan = np.isnan( lim_C_hi[tt] )
		tmp_hi_c = np.r_[ tmp_hi_c, lim_C_hi[tt][ id_nan == False] ]

		id_nan = np.isnan( lim_dered_C_hi[tt] )
		tmp_hi_dered_c = np.r_[ tmp_hi_dered_c, lim_dered_C_hi[tt][ id_nan == False] ]


	fig = plt.figure()
	ax = fig.add_axes( [0.12, 0.12, 0.83, 0.83] )

	ax.hist( lo_bcg_gr, bins = 25, density = True, color = 'b', ls = '-', alpha = 0.25,)
	ax.hist( hi_bcg_gr, bins = 25, density = True, color = 'r', alpha = 0.25, label = 'BCGs')

	l1 = ax.hist( tmp_lo_c, bins = c_bins, density = True, color = 'b', alpha = 0.75, ls = '-', 
		histtype = 'step', label = 'satellites')
	l2 = ax.hist( tmp_hi_c, bins = c_bins, density = True, color = 'r', alpha = 0.75, ls = '-', histtype = 'step')
	l3 = ax.hist( tmp_lo_dered_c, bins = c_bins, density = True, color = 'b', alpha = 0.65, ls = '--', histtype = 'step')
	ax.hist( tmp_hi_dered_c, bins = c_bins, density = True, color = 'r', alpha = 0.65, ls = '--', histtype = 'step')

	ax.axvline( np.median( tmp_lo_c), ls = '-', color = 'b', alpha = 0.75, ymin = 0.85, ymax = 1.0,)
	ax.axvline( np.median( tmp_hi_c), ls = '-', color = 'r', alpha = 0.75, ymin = 0.85, ymax = 1.0,)
	ax.axvline( np.median( tmp_lo_dered_c), ls = '--', color = 'b', alpha = 0.65, ymin = 0.85, ymax = 1.0,)
	ax.axvline( np.median( tmp_hi_dered_c), ls = '--', color = 'r', alpha = 0.65, ymin = 0.85, ymax = 1.0,)

	legend_0 = ax.legend( handles = [ l1[2][0], l2[2][0], l3[2][0] ], labels = [ fig_name[0], fig_name[1], 'Deredden mag'], 
		loc = 1, frameon = False, fontsize = 13,)
	ax.legend( loc = 2, frameon = False, fontsize = 13,)
	ax.add_artist( legend_0 )

	ax.set_xlim( 0.5, 2.5 )
	ax.set_xlabel('g-r', fontsize = 13,)
	ax.set_ylabel('PDF', fontsize = 13,)
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)

	ax.annotate( s = '$%.1f \\leq R \\leq %.1f Mpc$' % (R_cri[jj], R_cri[jj+1]), xy = (0.65, 0.15), 
		xycoords = 'axes fraction', fontsize = 13,)
	plt.savefig('/home/xkchen/BCG_color_compare_within-%.1f-Mpc.png' % R_cri[ jj+1 ], dpi = 300)
	plt.close()


