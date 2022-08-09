import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pds
import h5py

import scipy.stats as sts
from io import StringIO
import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable

#.
from Mass_rich_radius import rich2R_Simet
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
z_ref = 0.25
pixel = 0.396
a_ref = 1 / (z_ref + 1)


### === subsample binned by richness
def lim_R_list():

	cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/cat/'
	out_path = '/home/xkchen/figs_cp/cc_rich_rebin/cat/'


	bin_rich = [ 20, 30, 50, 210 ]
	line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']
	line_c = ['b', 'g', 'r']


	#. rich ( 30, 50)
	R_lim_low = np.array( [ 0, 300, 400, 550, 5000] )

	cp_R_lim = []

	for tt in range( 1,3 ):

		pp_scal_R = R_lim_low + 0.
		cp_R_lim.append( pp_scal_R )

	##.
	out_arr = np.array( [ R_lim_low, cp_R_lim[0], cp_R_lim[1] ] )
	np.savetxt( out_path + 'subset_R-limit.txt', out_arr )

	return


##. radii binned subsamples
def sat_phyR_binned():

	cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/cat/'

	out_path = '/home/xkchen/figs_cp/cc_rich_rebin/cat/'

	bin_rich = [ 20, 30, 50, 210 ]

	##. radius binned satellite
	for kk in range( 3 ):

		if kk == 1:
			R_bins = np.array( [ 0, 300, 400, 550, 5000] )
			R_bins = [ R_bins ] * 3

		##. rich < 30
		if kk == 0:
			R_bins = np.array([0, 150, 300, 500, 2000])
			R_bins = [ R_bins ] * 3

		##. rich > 50
		if kk == 2:
			R_bins = np.array([0, 400, 600, 750, 2000])
			R_bins = [ R_bins ] * 3


		##.
		s_dat = pds.read_csv( cat_path + 
			'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv' % ( bin_rich[kk], bin_rich[kk + 1]),)

		bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
		p_ra, p_dec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] )

		p_Rsat = np.array( s_dat['R_cen'] )
		p_R2Rv = np.array( s_dat['Rcen/Rv'] )
		clus_IDs = np.array( s_dat['clus_ID'] )

		a_obs = 1 / ( bcg_z + 1 )

		# cp_Rsat = p_Rsat * 1e3 * a_ref / h  ##. physical radius
		cp_Rsat = p_Rsat * 1e3 * a_obs / h  ##. physical radius


		##. division
		for nn in range( len( R_bins[0] ) - 1 ):
		# 	sub_N = (cp_Rsat >= R_bins[kk][ nn ]) & (cp_Rsat < R_bins[kk][ nn + 1])

			if nn == len( R_bins[0] ) - 2:
				sub_N = cp_Rsat >= R_bins[kk][ nn ]
			else:
				sub_N = (cp_Rsat >= R_bins[kk][ nn ]) & (cp_Rsat < R_bins[kk][ nn + 1])

			##. save
			out_c_ra, out_c_dec, out_c_z = bcg_ra[ sub_N ], bcg_dec[ sub_N ], bcg_z[ sub_N ]
			out_s_ra, out_s_dec = p_ra[ sub_N ], p_dec[ sub_N ]
			out_Rsat = p_Rsat[ sub_N ]
			out_R2Rv = p_R2Rv[ sub_N ]
			out_clus_ID = clus_IDs[ sub_N ]

			keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'R_sat', 'R2Rv', 'clus_ID'] 
			values = [ out_c_ra, out_c_dec, out_c_z, out_s_ra, out_s_dec, out_Rsat, out_R2Rv, out_clus_ID]
			fill = dict( zip( keys, values ) )
			data = pds.DataFrame( fill )
			data.to_csv( out_path + 
				'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_mem_cat.csv' % 
				( bin_rich[kk], bin_rich[kk + 1], R_bins[kk][nn], R_bins[kk][nn + 1]),)


	##... match with stacked information
	pos_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/'

	for pp in range( 3 ):

		if pp == 1:
			R_bins = np.array( [ 0, 300, 400, 550, 5000] )
			R_bins = [ R_bins ] * 3

		##. rich < 30
		if pp == 0:
			R_bins = np.array([0, 150, 300, 500, 2000])
			R_bins = [ R_bins ] * 3

		##. rich > 50
		if pp == 2:
			R_bins = np.array([0, 400, 600, 750, 2000])
			R_bins = [ R_bins ] * 3


		for tt in range( len( R_bins[0] ) - 1 ):

			s_dat = pds.read_csv( out_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_mem_cat.csv' % 
								( bin_rich[pp], bin_rich[pp + 1], R_bins[pp][tt], R_bins[pp][tt + 1]), )

			bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
			p_ra, p_dec = np.array( s_dat['sat_ra'] ), np.array( s_dat['sat_dec'] )	

			pre_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

			##.
			for kk in range( 3 ):

				#. z-ref position
				dat = pds.read_csv( pos_path + 'Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos_z-ref.csv' % band[kk])
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
				data.to_csv( out_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_mem-%s-band_pos-zref.csv' % 
								( bin_rich[pp], bin_rich[pp + 1], R_bins[pp][tt], R_bins[pp][tt + 1], band[kk]),)

	return

# lim_R_list()
# sat_phyR_binned()



##. figs 
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/cat/'

out_path = '/home/xkchen/figs_cp/cc_rich_rebin/cat/'


bin_rich = [ 20, 30, 50, 210 ]
line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']
line_c = ['b', 'g', 'r']


fig = plt.figure()
ax = fig.add_axes([0.10, 0.10, 0.85, 0.85])

for kk in range( 3 ):

	##. rich < 30
	if kk == 0:
		R_bins = np.array([0, 150, 300, 500, 2000])

	##. 30 < rich < 50
	if kk == 1:
		R_bins = np.array( [ 0, 300, 400, 550, 5000] )

	##. rich > 50
	if kk == 2:
		R_bins = np.array([0, 400, 600, 750, 2000])


	s_dat = pds.read_csv( cat_path + 
		'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv' % ( bin_rich[kk], bin_rich[kk + 1]),)

	bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
	p_ra, p_dec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] )

	p_Rsat = np.array( s_dat['R_cen'] )

	a_obs = 1 / (bcg_z + 1)

	# cp_Rsat = p_Rsat * 1e3 * a_ref / h  ##. physical radius
	cp_Rsat = p_Rsat * 1e3 * a_obs / h    ##. physical radius

	#.
	for qq in range( len(R_bins) - 1 ):

		sub_N0 = ( cp_Rsat >= R_bins[ qq ] ) & ( cp_Rsat < R_bins[ qq + 1 ] )
		print( np.sum( sub_N0 ) )

	print( '*' * 10 )


	R_edgs = np.logspace( 0, 3.4, 55 )
	ax.hist( cp_Rsat, bins = R_edgs, histtype = 'step', density = False, color = line_c[kk], label = line_name[kk],)

	for qq in range( 1, len(R_bins) ):

		ax.axvline( R_bins[ qq ], ls = ':', color = line_c[kk], alpha = 0.55,)

ax.legend( loc = 2,)
ax.set_xlabel('$R_{sat} \; [kpc]$')
ax.set_xscale('log')
ax.set_xlim( 5, 2.5e3 )

ax.set_ylabel('# of galaxies')
ax.set_yscale('log')

plt.savefig('/home/xkchen/rich_R_rebin_hist.png', dpi = 300)
plt.close()

