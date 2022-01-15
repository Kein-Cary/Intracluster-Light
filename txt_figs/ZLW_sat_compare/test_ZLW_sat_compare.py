import matplotlib as mpl
import matplotlib.pyplot as plt

import itertools
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

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 
			'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']

# cat_lis = [ 'low-rich', 'hi-rich' ]

# fig_name = ['Low $ \\lambda \\mid M_{\\ast}^{\\mathrm{BCG}} $', 
# 			'High $ \\lambda \\mid M_{\\ast}^{\\mathrm{BCG}} $']


# R_cri = np.array( [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0 ] )
R_cri = np.linspace( 0, 1.2, 9)
N_ri = len( R_cri )

#. BCG color
# clus_dat = pds.read_csv( '/home/xkchen/figs/sat_cat_ZLW/clust_sql_match_cat.csv' )
clus_dat = pds.read_csv( '/home/xkchen/figs/ZLW_cat_15/ZLW_sat/clust_bcg-m_rich_sql_match_cat.csv' )

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

# fig = plt.figure()
# ax = fig.add_axes( [0.12, 0.12, 0.83, 0.83] )
# ax.hist( lo_bcg_gr, bins = 35, density = True, histtype = 'step', color = 'b', alpha = 0.5, label = fig_name[0] )
# ax.hist( hi_bcg_gr, bins = 35, density = True, histtype = 'step', color = 'r', alpha = 0.5, label = fig_name[1] )
# ax.set_xlabel('g-r of BCGs', fontsize = 16,)
# ax.set_ylabel('PDF', fontsize = 16,)
# ax.legend( loc = 1, frameon = False,)
# ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 16,)
# plt.savefig('/home/xkchen/BCG_color_compare.png', dpi = 300)
# plt.close()


'''

#. satellite color
sat_dat = pds.read_csv( '/home/xkchen/figs/sat_cat_ZLW/sat_sql_match_cat.csv' )
clus_id = np.array( sat_dat['clust_id'] )
sat_ra, sat_dec, sat_z = np.array( sat_dat['ra'] ), np.array( sat_dat['dec'] ), np.array( sat_dat['z'] )
sat_rmag, sat_rMag = np.array( sat_dat['r_mag'] ), np.array( sat_dat['r_Mag'] )
sat_zf, centric_L = np.array( sat_dat['z_flag'] ), np.array( sat_dat['centric_R'] )
sat_r_mag, sat_g_mag, sat_i_mag = np.array( sat_dat['r_mag'] ), np.array( sat_dat['g_mag'] ), np.array( sat_dat['i_mag'] )
sat_r_cmag, sat_g_cmag, sat_i_cmag = np.array( sat_dat['r_cmag'] ), np.array( sat_dat['g_cmag'] ), np.array( sat_dat['i_cmag'] )
sat_dered_rmag, sat_dered_gmag = np.array( sat_dat['dered_r_mag'] ), np.array( sat_dat['dered_g_mag'] )
sat_dered_imag = np.array( sat_dat['dered_i_mag'] )

id_v0 = sat_g_mag > 0
id_v1 = sat_r_mag > 0
id_v2 = sat_dered_rmag > 0
id_v3 = sat_dered_gmag > 0
tot_sat_gr = sat_g_mag[ id_v0 & id_v1 ] - sat_r_mag[ id_v0 & id_v1 ]
tot_dered_sat_gr = sat_dered_gmag[ id_v2 & id_v3 ] - sat_dered_rmag[ id_v2 & id_v3 ]

# plt.figure()
# ax = plt.subplot( 111 )
# ax.hist( sat_r_mag[ id_v0 & id_v1 ], bins = 55, density = True, ls = '-', color = 'r', 
# 	alpha = 0.5, label = 'r-band' + ' no correction')
# ax.hist( sat_g_mag[ id_v0 & id_v1 ], bins = 55, density = True, ls = '-', color = 'g', 
# 	alpha = 0.5, label = 'g-band' + ' no correction')
# ax.hist( sat_dered_rmag[ id_v0 & id_v1 ], bins = 55, density = True, histtype = 'step', ls = '--', color = 'r', 
# 	alpha = 0.5, label = 'r-band' + ' deredden')
# ax.hist( sat_dered_gmag[ id_v0 & id_v1 ], bins = 55, density = True, histtype = 'step', ls = '--', color = 'g', 
# 	alpha = 0.5, label = 'r-band' + ' deredden')
# ax.legend( loc = 1, frameon = False)
# ax.set_ylabel('PDF')
# ax.set_xlabel('Apparent magnitude [mag]')
# plt.savefig('/home/xkchen/apparent_mag_compare.png', dpi = 300)
# plt.close()

# fig = plt.figure()
# ax = fig.add_axes( [0.12, 0.12, 0.83, 0.83] )
# ax.hist( tot_sat_gr, bins = 100, density = True, histtype = 'step', color = 'k',)

# for tt in range( 1, N_ri ):

# 	dL_dex = centric_L[ id_v0 & id_v1 ] < R_cri[ tt ]
# 	cen_sat_gr = tot_sat_gr[ dL_dex ]
# 	# cen_sat_gr = tot_dered_sat_gr[ dL_dex ]

# 	ax.hist( cen_sat_gr, bins = 100, density = True, histtype = 'step', 
# 		color = mpl.cm.rainbow(tt / N_ri), alpha = 0.5, label = 'within %.1f Mpc' % R_cri[tt],)
# 	ax.axvline( x = np.median( cen_sat_gr ), ls = '--', ymin = 0.0, ymax = 0.35, 
# 		color = mpl.cm.rainbow(tt / N_ri), alpha = 0.5,)

# ax.set_xlabel( 'g-r of satellites', fontsize = 16,)
# ax.set_xlim( 0, 2.5 )
# ax.set_ylabel( 'PDF', fontsize = 16,)
# ax.legend( loc = 2, frameon = False,)
# ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 16,)
# plt.savefig('/home/xkchen/Sat_color_compare.png', dpi = 300)
# plt.close()

'''

#. member galaxy color comparison
red_lo_mem_gr = []
red_lo_mem_dered_gr = []

red_lo_mem_cen_dl = []
red_lo_mem_ra = []
red_lo_mem_dec = []
red_lo_mem_z = []
with h5py.File('/home/xkchen/figs/sat_cat_ZLW/reload_%s_clus-sat_record.h5' % cat_lis[0], 'r') as f:
	keys = list( f.keys() )
	N_ks = len( keys )

	for jj in range( N_ks ):
		_sub_arr = f["/clust_%d/arr/" % jj][()]

		red_lo_mem_gr.append( _sub_arr[5] )
		red_lo_mem_dered_gr.append( _sub_arr[8] )

		red_lo_mem_cen_dl.append( _sub_arr[3] )
		red_lo_mem_ra.append( _sub_arr[0] )
		red_lo_mem_dec.append( _sub_arr[1] )
		red_lo_mem_z.append( _sub_arr[2] )


red_hi_mem_gr = []
red_hi_mem_dered_gr = []

red_hi_mem_cen_dl = []
red_hi_mem_ra = []
red_hi_mem_dec = []
red_hi_mem_z = []
with h5py.File('/home/xkchen/figs/sat_cat_ZLW/reload_%s_clus-sat_record.h5' % cat_lis[1], 'r') as f:
	keys = list( f.keys() )
	N_ks = len( keys )

	for jj in range( N_ks ):
		_sub_arr = f["/clust_%d/arr/" % jj][()]

		red_hi_mem_gr.append( _sub_arr[5] )
		red_hi_mem_dered_gr.append( _sub_arr[8] )

		red_hi_mem_cen_dl.append( _sub_arr[3] )
		red_hi_mem_ra.append( _sub_arr[0] )
		red_hi_mem_dec.append( _sub_arr[1] )
		red_hi_mem_z.append( _sub_arr[2] )

#. 
wen_lo_mem_gr = []
wen_lo_mem_dered_gr = []

wen_lo_mem_cen_dl = []
wen_lo_mem_ra = []
wen_lo_mem_dec = []
wen_lo_mem_z = []
with h5py.File( '/home/xkchen/figs/sat_cat_ZLW/ZLW_cat_%s_clus-sat_record.h5' % cat_lis[0], 'r') as f:
	keys = list( f.keys() )
	N_ks = len( keys )

	for jj in range( N_ks ):
		_sub_arr = f["/clust_%d/arr/" % jj ][()]

		wen_lo_mem_gr.append( _sub_arr[4] )
		wen_lo_mem_dered_gr.append( _sub_arr[7] )

		wen_lo_mem_cen_dl.append( _sub_arr[3] )
		wen_lo_mem_ra.append( _sub_arr[0] )
		wen_lo_mem_dec.append( _sub_arr[1] )
		wen_lo_mem_z.append( _sub_arr[2] )


wen_hi_mem_gr = []
wen_hi_mem_dered_gr = []

wen_hi_mem_cen_dl = []
wen_hi_mem_ra = []
wen_hi_mem_dec = []
wen_hi_mem_z = []
with h5py.File( '/home/xkchen/figs/sat_cat_ZLW/ZLW_cat_%s_clus-sat_record.h5' % cat_lis[1], 'r') as f:
	keys = list( f.keys() )
	N_ks = len( keys )

	for jj in range( N_ks ):
		_sub_arr = f["/clust_%d/arr/" % jj ][()]

		wen_hi_mem_gr.append( _sub_arr[4] )
		wen_hi_mem_dered_gr.append( _sub_arr[7] )

		wen_hi_mem_cen_dl.append( _sub_arr[3] )
		wen_hi_mem_ra.append( _sub_arr[0] )
		wen_hi_mem_dec.append( _sub_arr[1] )
		wen_hi_mem_z.append( _sub_arr[2] )

'''

#... overall comparison
cc_red_lo_mem_gr = np.array( list( itertools.chain(*red_lo_mem_gr) ) )
cc_red_hi_mem_gr = np.array( list( itertools.chain(*red_hi_mem_gr) ) )
cc_red_lo_mem_dL = np.array( list( itertools.chain(*red_lo_mem_cen_dl) ) )
cc_red_hi_mem_dL = np.array( list( itertools.chain(*red_hi_mem_cen_dl) ) )
cc_red_lo_mem_z = np.array( list( itertools.chain(*red_lo_mem_z) ) )
cc_red_hi_mem_z = np.array( list( itertools.chain(*red_hi_mem_z) ) )

cc_red_mem_gr = np.r_[ cc_red_lo_mem_gr, cc_red_hi_mem_gr ]
cc_red_mem_dL = np.r_[ cc_red_lo_mem_dL, cc_red_hi_mem_dL ]
cc_red_mem_z = np.r_[ cc_red_lo_mem_z, cc_red_hi_mem_z ]
cc_red_mem_dL = cc_red_mem_dL / h / ( 1 + cc_red_mem_z )

id_v0 = sat_g_mag > 0
id_v1 = sat_r_mag > 0
id_v2 = sat_dered_rmag > 0
id_v3 = sat_dered_gmag > 0
tot_sat_gr = sat_g_mag[ id_v0 & id_v1 ] - sat_r_mag[ id_v0 & id_v1 ]
tot_dered_sat_gr = sat_dered_gmag[ id_v2 & id_v3 ] - sat_dered_rmag[ id_v2 & id_v3 ]

plt.figure()
ax = plt.subplot( 111 )
ax.hist( tot_sat_gr, bins = 105, density = True, histtype = 'step', ls = '--', color = 'g', 
	alpha = 0.5, label = 'no correction')
ax.hist( tot_dered_sat_gr, bins = 105, density = True, histtype = 'step', ls = '-', color = 'r', 
	alpha = 0.5, label = 'deredden')
ax.legend( loc = 1, frameon = False)
ax.set_xlim(0, 2.5)
ax.set_ylabel('PDF')
ax.set_xlabel('g-r')
plt.savefig('/home/xkchen/g-r_changes.png', dpi = 300)
plt.close()

plt.figure()
ax = plt.subplot( 111 )
ax.hist( tot_sat_gr, bins = 105, density = True, histtype = 'step', ls = '--', color = 'g', 
	alpha = 0.5, label = 'Wen+2015')
ax.hist( cc_red_mem_gr, bins = 105, density = True, histtype = 'step', ls = '-', color = 'r', 
	alpha = 0.5, label = 'SDSS redMaPPer')
ax.legend( loc = 1, frameon = False)
ax.set_xlim(0, 2.5)
ax.set_ylabel('PDF')
ax.set_xlabel('g-r')
plt.savefig('/home/xkchen/g-r_compare.png', dpi = 300)
plt.close()

_bin_x = np.linspace( 0.25, 2.25, 75)

fig = plt.figure()
ax = fig.add_axes( [0.12, 0.12, 0.83, 0.83] )
ax.hist( tot_sat_gr, bins = _bin_x, density = True, histtype = 'step', color = 'k', label = 'Wen+2015 total')

for tt in range( 1, N_ri ):

	# if tt % 2 == 0:
	dL_dex = centric_L[ id_v0 & id_v1 ] < R_cri[ tt ]
	cen_sat_gr = tot_sat_gr[ dL_dex ]
	cc_dl_dex = cc_red_mem_dL < R_cri[ tt ]

	if tt == N_ri - 1:
		l1 = ax.axvline( x = np.median( cen_sat_gr ), ls = '-', ymin = 0.0, ymax = 0.15, 
			color = mpl.cm.autumn(tt / N_ri), alpha = 0.75,)
		l2 = ax.axvline( x = np.median( cc_red_mem_gr[ cc_dl_dex ] ), ls = '--', ymin = 0.85, ymax = 1.0, 
			color = mpl.cm.winter(tt / N_ri), alpha = 0.75,)

	else:
		ax.hist( cen_sat_gr, bins = _bin_x, density = True, histtype = 'step', 
			color = mpl.cm.autumn(tt / N_ri), ls = '-', alpha = 0.75, label = 'within %.1f Mpc' % R_cri[tt],)
		ax.axvline( x = np.median( cen_sat_gr ), ls = '-', ymin = 0.0, ymax = 0.15, 
			color = mpl.cm.autumn(tt / N_ri), alpha = 0.75,)

		ax.hist( cc_red_mem_gr[ cc_dl_dex ], bins = _bin_x, density = True, histtype = 'step', 
			color = mpl.cm.winter(tt / N_ri), ls = '--', alpha = 0.75,)
		ax.axvline( x = np.median( cc_red_mem_gr[ cc_dl_dex ] ), ls = '--', ymin = 0.85, ymax = 1.0, 
			color = mpl.cm.winter(tt / N_ri), alpha = 0.75,)

ax.set_xlabel( 'g-r of satellites', fontsize = 16,)
ax.set_xlim( 0, 2.5 )
ax.set_ylabel( 'PDF', fontsize = 16,)

legend_0 = ax.legend( handles = [l1, l2], labels = ['Wen+2015', 'redMaPPer'], loc = 1, frameon = False,)
ax.legend( loc = 2, frameon = False,)
ax.add_artist( legend_0 )

ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 16,)
plt.savefig('/home/xkchen/Sat_color_compare.png', dpi = 300)
plt.close()

'''

#... galaxy color comparison along radial direction
for ll in range( 2 ):

	with h5py.File('/home/xkchen/figs/sat_cat_ZLW/reload_%s_clus-sat_record.h5' % cat_lis[ll], 'r') as f:
		keys = list( f.keys() )
		N_z = len( keys )

		tt_mem_gr = []
		tt_mem_dered_gr = []

		tt_mem_gi = []
		tt_mem_dered_gi = []

		tt_mem_ri = []
		tt_mem_dered_ri = []

		tt_mem_cen_dl = []
		tt_mem_Pm = []

		for ii in range( N_z ):
			_sub_arr = f["/clust_%d/arr/" % ii ][()]

			tt_mem_gr.append( _sub_arr[5] )
			tt_mem_dered_gr.append( _sub_arr[8] )

			tt_mem_gi.append( _sub_arr[6] )
			tt_mem_dered_gi.append( _sub_arr[9] )

			tt_mem_ri.append( _sub_arr[7] )
			tt_mem_dered_ri.append( _sub_arr[10] )

			tt_mem_cen_dl.append( _sub_arr[3] )
			tt_mem_Pm.append( _sub_arr[4] )

	for jj in range( N_ri - 1 ):

		cri_Pm = []
		cri_gr, cri_ri, cri_gi = [], [], []
		dered_cri_gr, dered_cri_ri, dered_cri_gi = [], [], []

		for ii in range( N_z ):
			
			sub_cen_R = tt_mem_cen_dl[ ii ]
			sub_Pmem = tt_mem_Pm[ ii ]

			id_lim = ( sub_cen_R > R_cri[ jj ] ) & ( sub_cen_R <= R_cri[ jj + 1] )

			limd_Pm = sub_Pmem[id_lim]
			limd_gr = tt_mem_gr[ ii ][id_lim]
			limd_ri = tt_mem_ri[ ii ][id_lim]
			limd_gi = tt_mem_gi[ ii ][id_lim]

			limd_dered_gr = tt_mem_dered_gr[ ii ][id_lim]
			limd_dered_ri = tt_mem_dered_ri[ ii ][id_lim]
			limd_dered_gi = tt_mem_dered_gi[ ii ][id_lim]

			cri_Pm.append( limd_Pm )
			cri_gr.append( limd_gr )
			cri_ri.append( limd_ri )
			cri_gi.append( limd_gi )

			dered_cri_gr.append( limd_dered_gr )
			dered_cri_ri.append( limd_dered_ri )
			dered_cri_gi.append( limd_dered_gi )

		f_tree = h5py.File( '/home/xkchen/figs/sat_cat_ZLW/radii_bins/' + 
			'redMaP_cat_%s_within-%.1f-Mpc_mem_color.h5' % ( cat_lis[ll], R_cri[ jj + 1 ] ), 'w')
		for ii in range( N_z ):

			out_arr = np.array( [ cri_Pm[ii], cri_gr[ii], cri_ri[ii], cri_gi[ii], 
								dered_cri_gr[ii], dered_cri_ri[ii], dered_cri_gi[ii] ] )
			gk = f_tree.create_group( "clust_%d/" % ii )
			dk0 = gk.create_dataset( "arr", data = out_arr )

		f_tree.close()
print('done !')


for jj in range( N_ri - 1 ):

	#. ZLW catalog
	C_mem_low = []
	C_mem_hi = []

	with h5py.File( '/home/xkchen/figs/sat_cat_ZLW/radii_bins/' + 
		'ZLW_cat_%s_within-%.1f-Mpc_mem_color.h5' % ( cat_lis[0], R_cri[ jj+1 ] ), 'r') as f:
		keys = list( f.keys() )
		_Ns_ = len( keys )

		for tt in range( _Ns_ ):
			C_mem_low.append( f[ "clust_%d/arr" % tt ][()][1] )

	with h5py.File( '/home/xkchen/figs/sat_cat_ZLW/radii_bins/' + 
		'ZLW_cat_%s_within-%.1f-Mpc_mem_color.h5' % ( cat_lis[1], R_cri[ jj+1 ] ), 'r') as f:
		keys = list( f.keys() )
		_Ns_ = len( keys )

		for tt in range( _Ns_ ):
			C_mem_hi.append( f[ "clust_%d/arr" % tt ][()][1] )

	#. redMaPPer
	red_C_mem_low = []
	red_C_mem_hi = []

	with h5py.File( '/home/xkchen/figs/sat_cat_ZLW/radii_bins/' + 
		'redMaP_cat_%s_within-%.1f-Mpc_mem_color.h5' % ( cat_lis[0], R_cri[ jj + 1 ] ), 'r') as f:
		keys = list( f.keys() )
		_Ns_ = len( keys )

		for tt in range( _Ns_ ):
			red_C_mem_low.append( f[ "clust_%d/arr" % tt ][()][1] )

	with h5py.File( '/home/xkchen/figs/sat_cat_ZLW/radii_bins/' + 
		'redMaP_cat_%s_within-%.1f-Mpc_mem_color.h5' % ( cat_lis[1], R_cri[ jj + 1 ] ), 'r') as f:
		keys = list( f.keys() )
		_Ns_ = len( keys )

		for tt in range( _Ns_ ):
			red_C_mem_hi.append( f[ "clust_%d/arr" % tt ][()][1] )


	Len_x0 = np.array( [ len(ll) for ll in C_mem_low ] )
	id_lim0 = Len_x0 > 0
	lim_C_low = np.array( C_mem_low )[ id_lim0 ]

	Len_x1 = np.array( [ len(ll) for ll in C_mem_hi ] )
	id_lim1 = Len_x1 > 0
	lim_C_hi = np.array( C_mem_hi )[ id_lim1 ]


	Len_x0 = np.array( [ len(ll) for ll in red_C_mem_low ] )
	id_lim0 = Len_x0 > 0
	lim_red_C_low = np.array( red_C_mem_low )[ id_lim0 ]

	Len_x1 = np.array( [ len(ll) for ll in red_C_mem_hi ] )
	id_lim1 = Len_x1 > 0
	lim_red_C_hi = np.array( red_C_mem_hi )[ id_lim1 ]


	c_min_0 = [ np.nanmin(ll) for ll in lim_C_low ]
	c_min_1 = [ np.nanmin(ll) for ll in lim_C_hi ]
	c_min = np.min( np.r_[ c_min_0, c_min_1 ] )

	c_max_0 = [ np.nanmax(ll) for ll in lim_C_low ]
	c_max_1 = [ np.nanmax(ll) for ll in lim_C_hi ]
	c_max = np.max( np.r_[ c_max_0, c_max_1 ] )

	c_bins = np.linspace( c_min, c_max, 55 )
	c_x = 0.5 * (c_bins[:-1] + c_bins[1:])

	tmp_lo_c, tmp_hi_c = np.array([]), np.array([])
	tmp_red_lo_c, tmp_red_hi_c = np.array([]), np.array([])

	Ns1 = len( lim_C_low )
	for tt in range( Ns1 ):

		id_nan = np.isnan( lim_C_low[tt] )
		tmp_lo_c = np.r_[ tmp_lo_c, lim_C_low[tt][ id_nan == False] ]

	Ns2 = len( lim_C_hi )
	for tt in range( Ns2 ):

		id_nan = np.isnan( lim_C_hi[tt] )
		tmp_hi_c = np.r_[ tmp_hi_c, lim_C_hi[tt][ id_nan == False] ]

	Ns1 = len( lim_red_C_low )
	for tt in range( Ns1 ):

		id_nan = np.isnan( lim_red_C_low[tt] )
		tmp_red_lo_c = np.r_[ tmp_red_lo_c, lim_red_C_low[tt][ id_nan == False] ]

	Ns2 = len( lim_red_C_hi )
	for tt in range( Ns2 ):

		id_nan = np.isnan( lim_red_C_hi[tt] )
		tmp_red_hi_c = np.r_[ tmp_red_hi_c, lim_red_C_hi[tt][ id_nan == False] ]


	#.record the median of the central radius bin
	if jj == 0:
		medi_gr = np.median( tmp_hi_c )
		medi_red_gr = np.median( tmp_red_hi_c )

	fig = plt.figure()
	ax = fig.add_axes( [0.12, 0.12, 0.83, 0.83] )

	ax.hist( lo_bcg_gr, bins = 25, density = True, color = 'b', ls = '-', alpha = 0.25,)
	ax.hist( hi_bcg_gr, bins = 25, density = True, color = 'r', alpha = 0.25, label = 'BCGs')

	l1 = ax.hist( tmp_lo_c, bins = c_bins, density = True, color = 'b', alpha = 0.75, ls = '-', 
		histtype = 'step', )
	l2 = ax.hist( tmp_hi_c, bins = c_bins, density = True, color = 'r', alpha = 0.75, ls = '-', 
		histtype = 'step', label = 'Satellites',)
	l3 = ax.hist( tmp_red_lo_c, bins = c_bins, density = True, color = 'b', alpha = 0.65, ls = '--', histtype = 'step')
	l4 = ax.hist( tmp_red_hi_c, bins = c_bins, density = True, color = 'r', alpha = 0.65, ls = '--', histtype = 'step')

	ax.axvline( np.median( tmp_lo_c), ls = '-', color = 'b', alpha = 0.75, ymin = 0.85, ymax = 1.0,)
	ax.axvline( np.median( tmp_hi_c), ls = '-', color = 'r', alpha = 0.75, ymin = 0.85, ymax = 1.0,)
	ax.axvline( np.median( tmp_red_lo_c), ls = '--', color = 'b', alpha = 0.65, ymin = 0.85, ymax = 1.0,)
	ax.axvline( np.median( tmp_red_hi_c), ls = '--', color = 'r', alpha = 0.65, ymin = 0.85, ymax = 1.0,)

	if jj != 0:
		ax.axvline( medi_gr, ls = '-', color = 'k', alpha = 0.75, ymin = 0.90, ymax = 1.0,)
		ax.axvline( medi_red_gr, ls = '--', color = 'k', alpha = 0.75, ymin = 0.90, ymax = 1.0,)

	# legend_0 = ax.legend( handles = [ l1[2][0], l2[2][0], l3[2][0], l4[2][0] ], 
	# 	labels = [ 'Low $t_{ \\mathrm{age} }$, Wen+2015', 'High $t_{ \\mathrm{age} }$, Wen+2015', 
	# 				'Low $t_{ \\mathrm{age} }$, redMaPPer', 'High $t_{ \\mathrm{age} }$, redMaPPer'], 
	# 	loc = 2, frameon = False,)

	# legend_0 = ax.legend( handles = [ l1[2][0], l2[2][0], l3[2][0], l4[2][0] ], 
	# 	labels = [ 'Low $\\lambda$, Wen+2015', 'High $\\lambda$, Wen+2015', 
	# 				'Low $\\lambda$, redMaPPer', 'High $\\lambda$, redMaPPer'], 
	# 	loc = 2, frameon = False,)

	legend_0 = ax.legend( handles = [ l1[2][0], l2[2][0], l3[2][0], l4[2][0] ], 
		labels = [ 'Low $ M_{\\ast}^{\\mathrm{BCG}}$, Wen+2015', 'High $ M_{\\ast}^{\\mathrm{BCG}}$, Wen+2015', 
					'Low $ M_{\\ast}^{\\mathrm{BCG}}$, redMaPPer', 'High $ M_{\\ast}^{\\mathrm{BCG}}$, redMaPPer'], 
		loc = 2, frameon = False,)

	ax.legend( loc = 1, frameon = False,)
	ax.add_artist( legend_0 )

	ax.set_xlim( 0.5, 2.5 )
	ax.set_xlabel('g-r', fontsize = 13,)
	ax.set_ylabel('PDF', fontsize = 13,)
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)

	ax.annotate( text = '$%.1f \\leq R \\leq %.1f Mpc$' % (R_cri[jj], R_cri[jj+1]), xy = (0.65, 0.15), 
		xycoords = 'axes fraction', fontsize = 13,)
	plt.savefig('/home/xkchen/BCG_color_compare_within-%.1f-Mpc.png' % R_cri[ jj+1 ], dpi = 300)
	plt.close()

