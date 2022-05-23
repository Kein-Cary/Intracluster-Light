import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pds

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc

from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord

#.
import time
from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


##### cosmology model
Test_model = apcy.Planck15.clone( H0 = 67.74, Om0 = 0.311 )
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']


### === 
bin_rich = [ 20, 30, 50, 210 ]

##. radius binned satellite
sub_name = ['inner', 'middle', 'outer']


# out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/shufl_list/radius_bin_table/'
# out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/shufl_list/cp_tables/'

##. rebinned radii subsamples
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/shufl_list/'


fig_name = ['Inner', 'Middle', 'Outer']
line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']


band_str = 'i'
pp = 1

for tt in range( 3 ):

	tmp_pre_N_sat = []
	tmp_pos_N_sat = []
	tmp_diff_N_sat = []

	tmp_pre_PA_Rs_N = []
	tmp_pos_PA_Rs_N = []
	tmp_diff_PA_Rs_N = []

	tmp_pre_R_phy_N = []
	tmp_pos_R_phy_N = []
	tmp_diff_R_phy_N = []


	for dd in range( 20 ):
	# for dd in range( 40 ):
	# for dd in range( 100 ):

		##. member cat
		dat = pds.read_csv( out_path + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR-%s-mem_%s-band_' % 
			( bin_rich[pp], bin_rich[pp + 1], sub_name[tt], band_str) + 'sat-shufl-%d_origin-img_position.csv' % dd,)

		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

		bcg_x, bcg_y = np.array( dat['bcg_x'] ), np.array( dat['bcg_y'] )
		sat_x, sat_y = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

		sat_PA = np.array( dat['sat_PA2bcg'] )
		R_sat_pix = np.sqrt( (sat_x - bcg_x)**2 + (sat_y - bcg_y)**2 )


		##. shuffle list
		cat = pds.read_csv( out_path + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR-%s-mem_%s-band_' % 
			( bin_rich[pp], bin_rich[pp + 1], sub_name[tt], band_str) + 'sat-shufl-%d_cat.csv' % dd,)

		mp_sat_x, mp_sat_y = np.array( cat['cp_sx'] ), np.array( cat['cp_sy'] )
		mp_sat_PA = np.array( cat['cp_PA'] )
		mp_sat_R_pix = np.array( cat['cp_Rpix'] )

		pos_Rsat = np.array( cat['cp_Rsat_phy'] )
		pre_Rsat = np.array( cat['orin_Rsat_phy'] )

		ident_symP = np.array( cat['is_symP'] )

		##. PA in range of (0- pi/2)
		id_pax = mp_sat_PA >= 2 * np.pi
		cc_mp_sat_PA = mp_sat_PA + 0.

		cc_mp_sat_PA[ id_pax ] = 2 * np.pi - mp_sat_PA[ id_pax ]

		modi_sat_PA = np.mod( cc_mp_sat_PA, np.pi / 2 )
		pre_modi_PA = np.mod( sat_PA, np.pi / 2 )


		#. normal points
		id_Px0 = ident_symP < 1
		id_Px1 = ident_symP == 1
		id_Px2 = ident_symP == 2
		id_Px3 = ident_symP == 3

		modi_sat_PA[ id_Px2 ] = np.pi / 2 - np.mod( mp_sat_PA[ id_Px2 ], np.pi / 2)
		modi_sat_PA[ id_Px3 ] = np.pi / 2 - np.mod( mp_sat_PA[ id_Px3 ], np.pi / 2)


		##. binned (sat_x, sat_y)
		N_x, N_y = 25, 20
		edg_x = np.linspace( 0, 2050, N_x )
		edg_y = np.linspace( 0, 1490, N_y )

		pre_N_sat = np.zeros( (N_y, N_x), )
		pos_N_sat = np.zeros( (N_y, N_x), )

		for mm in range( N_y - 1 ):

			idy_0 = ( sat_y >= edg_y[ mm ] ) & ( sat_y < edg_y[ mm + 1] )
			idy_1 = ( mp_sat_y >= edg_y[ mm ] ) & ( mp_sat_y < edg_y[ mm + 1] )

			for nn in range( N_x - 1):

				idx_0 = ( sat_x >= edg_x[ nn ] ) & ( sat_x < edg_x[ nn + 1] )
				idx_1 = ( mp_sat_x >= edg_x[ nn ] ) & ( mp_sat_x < edg_x[ nn + 1] )

				pre_N_sat[ mm, nn ] = np.sum( ( idy_0 ) & ( idx_0 ) )
				pos_N_sat[ mm, nn ] = np.sum( ( idy_1 ) & ( idx_1 ) )

		diff_N_sat = pos_N_sat - pre_N_sat

		tmp_pre_N_sat.append( pre_N_sat )
		tmp_pos_N_sat.append( pos_N_sat )
		tmp_diff_N_sat.append( diff_N_sat )


		##. binned in coordinate~( relative to BCGs )
		pre_Cbcg_x = R_sat_pix * np.cos( sat_PA )
		pre_Cbcg_y = R_sat_pix * np.sin( sat_PA )

		pos_Cbcg_x = mp_sat_R_pix * np.cos( mp_sat_PA )
		pos_Cbcg_y = mp_sat_R_pix * np.sin( mp_sat_PA )

		edg_cen_x = np.linspace( -1 * np.max([ pre_Cbcg_x, pos_Cbcg_x ]), np.max([ pre_Cbcg_x, pos_Cbcg_x ]), N_x )
		edg_cen_y = np.linspace( -1 * np.max([ pre_Cbcg_y, pos_Cbcg_y ]), np.max([ pre_Cbcg_y, pos_Cbcg_y ]), N_y )

		cc_pre_N_sat = np.zeros( (N_y, N_x), )
		cc_pos_N_sat = np.zeros( (N_y, N_x), )

		for mm in range( N_y - 1 ):

			idy_0 = ( pre_Cbcg_y >= edg_cen_y[ mm ] ) & ( pre_Cbcg_y < edg_cen_y[ mm + 1] )
			idy_1 = ( pos_Cbcg_y >= edg_cen_y[ mm ] ) & ( pos_Cbcg_y < edg_cen_y[ mm + 1] )

			for nn in range( N_x - 1):

				idx_0 = ( pre_Cbcg_x >= edg_cen_x[ nn ] ) & ( pre_Cbcg_x < edg_cen_x[ nn + 1] )
				idx_1 = ( pos_Cbcg_x >= edg_cen_x[ nn ] ) & ( pos_Cbcg_x < edg_cen_x[ nn + 1] )

				cc_pre_N_sat[ mm, nn ] = np.sum( ( idy_0 ) & ( idx_0 ) )
				cc_pos_N_sat[ mm, nn ] = np.sum( ( idy_1 ) & ( idx_1 ) )

		cc_diff_N_sat = cc_pos_N_sat - cc_pre_N_sat

		tmp_pre_PA_Rs_N.append( cc_pre_N_sat )
		tmp_pos_PA_Rs_N.append( cc_pos_N_sat )
		tmp_diff_PA_Rs_N.append( cc_diff_N_sat )


		##. binned in coordinate~( relative to BCGs )
		pre_Rphy_x = pre_Rsat * np.cos( sat_PA )
		pre_Rphy_y = pre_Rsat * np.sin( sat_PA )

		pos_Rphy_x = pos_Rsat * np.cos( mp_sat_PA )
		pos_Rphy_y = pos_Rsat * np.sin( mp_sat_PA )

		edg_cen_Rx = np.linspace( -1 * np.max([ pre_Rphy_x, pos_Rphy_x ]), np.max([ pre_Rphy_x, pos_Rphy_x ]), N_x )
		edg_cen_Ry = np.linspace( -1 * np.max([ pre_Rphy_y, pos_Rphy_y ]), np.max([ pre_Rphy_y, pos_Rphy_y ]), N_y )

		cc_pre_N_sat = np.zeros( (N_y, N_x), )
		cc_pos_N_sat = np.zeros( (N_y, N_x), )

		for mm in range( N_y - 1 ):

			idy_0 = ( pre_Rphy_y >= edg_cen_Ry[ mm ] ) & ( pre_Rphy_y < edg_cen_Ry[ mm + 1] )
			idy_1 = ( pos_Rphy_y >= edg_cen_Ry[ mm ] ) & ( pos_Rphy_y < edg_cen_Ry[ mm + 1] )

			for nn in range( N_x - 1):

				idx_0 = ( pre_Rphy_x >= edg_cen_Rx[ nn ] ) & ( pre_Rphy_x < edg_cen_Rx[ nn + 1] )
				idx_1 = ( pos_Rphy_x >= edg_cen_Rx[ nn ] ) & ( pos_Rphy_x < edg_cen_Rx[ nn + 1] )

				cc_pre_N_sat[ mm, nn ] = np.sum( ( idy_0 ) & ( idx_0 ) )
				cc_pos_N_sat[ mm, nn ] = np.sum( ( idy_1 ) & ( idx_1 ) )

		cc_diff_N_sat = cc_pos_N_sat - cc_pre_N_sat

		tmp_pre_R_phy_N.append( cc_pre_N_sat )
		tmp_pos_R_phy_N.append( cc_pos_N_sat )
		tmp_diff_R_phy_N.append( cc_diff_N_sat )


	##. figs
	tmp_pre_R_phy_N = np.array( tmp_pre_R_phy_N )
	tmp_pos_R_phy_N = np.array( tmp_pos_R_phy_N )
	tmp_diff_R_phy_N = np.array( tmp_diff_R_phy_N )

	Mean_pre_R_phy_N = np.mean( tmp_pre_R_phy_N, axis = 0 )
	Mean_pos_R_phy_N = np.mean( tmp_pos_R_phy_N, axis = 0 )
	Mean_diff_R_phy_N = np.mean( tmp_diff_R_phy_N, axis = 0 )


	fig = plt.figure( figsize = (20.4, 4.8) )
	ax0 = fig.add_axes( [0.04, 0.09, 0.27, 0.84] )
	ax1 = fig.add_axes( [0.37, 0.09, 0.27, 0.84] )
	ax2 = fig.add_axes( [0.69, 0.09, 0.27, 0.84] )

	ax0.set_title('Before shuffling (%s, %s)' % (line_name[pp], fig_name[tt]),)
	tf = ax0.imshow( Mean_pre_R_phy_N, origin = 'lower', cmap = 'rainbow', vmin = 1, vmax = 500, norm = mpl.colors.LogNorm(),)
	plt.colorbar( tf, ax = ax0, fraction = 0.038, pad = 0.01, label = '$N_{sat}$')


	y_ticks = np.arange( -0.5, 19.5 )
	dex_y = np.arange( 0, 19, 2 )

	x_ticks = np.arange( -0.5, 24.5 )
	dex_x = np.arange( 0, 24, 3 )

	ax0.set_xticks( x_ticks[ dex_x ] )
	ax0.set_xticklabels( ['%.0f' % ll for ll in edg_cen_Rx[dex_x] ] )

	ax0.set_yticks( y_ticks[ dex_y ] )
	ax0.set_yticklabels( ['%.0f' % ll for ll in edg_cen_Ry[dex_y] ] )

	ax0.set_xlabel( 'X [kpc]' )
	# ax0.set_ylabel( 'Y [kpc]' )


	ax1.set_title('After shuffling')
	tf = ax1.imshow( Mean_pos_R_phy_N, origin = 'lower', cmap = 'rainbow', vmin = 1, vmax = 500, norm = mpl.colors.LogNorm(),)
	cb = plt.colorbar( tf, ax = ax1, fraction = 0.038, pad = 0.01, label = '$N_{sat}$')
	# cb.cmap.set_under('w')

	ax1.set_xticks( x_ticks[ dex_x ] )
	ax1.set_xticklabels( ['%.0f' % ll for ll in edg_cen_Rx[dex_x] ] )

	ax1.set_yticks( y_ticks[ dex_y ] )
	ax1.set_yticklabels( ['%.0f' % ll for ll in edg_cen_Ry[dex_y] ] )

	ax1.set_xlabel( 'X [kpc]' )
	# ax1.set_ylabel( 'Y [kpc]' )


	ax2.set_title('Diff_img = After shuffling - Before shuffling')
	tf = ax2.imshow( Mean_diff_R_phy_N / np.mean( Mean_pos_R_phy_N ), origin = 'lower', cmap = 'bwr', vmin = -1, vmax = 1,)
	plt.colorbar( tf, ax = ax2, fraction = 0.038, pad = 0.01, label = '$N_{diff} \, / \, \\sqrt{ \\bar{N}_{pos}}$')

	ax2.set_xticks( x_ticks[ dex_x ] )
	ax2.set_xticklabels( ['%.0f' % ll for ll in edg_cen_Rx[dex_x] ] )

	ax2.set_yticks( y_ticks[ dex_y ] )
	ax2.set_yticklabels( ['%.0f' % ll for ll in edg_cen_Ry[dex_y] ] )

	ax2.set_xlabel( 'X [kpc]' )
	# ax2.set_ylabel( 'Y [kpc]' )

	plt.savefig('/home/xkchen/clust_%d-%d-%s_%s-band_sat_R-phy_2D_hist.png' % 
				(bin_rich[pp], bin_rich[pp+1], sub_name[tt], band_str), dpi = 300)
	plt.close()


	tmp_pre_N_sat = np.array( tmp_pre_N_sat )
	tmp_pos_N_sat = np.array( tmp_pos_N_sat )
	tmp_diff_N_sat = np.array( tmp_diff_N_sat )

	Mean_pre_N_sat = np.mean( tmp_pre_N_sat, axis = 0 )
	Mean_pos_N_sat = np.mean( tmp_pos_N_sat, axis = 0 )
	Mean_diff_N_sat = np.mean( tmp_diff_N_sat, axis = 0 )


	fig = plt.figure( figsize = (20, 4.8) )
	ax0 = fig.add_axes([0.04, 0.09, 0.27, 0.84])
	ax1 = fig.add_axes([0.37, 0.09, 0.27, 0.84])
	ax2 = fig.add_axes([0.69, 0.09, 0.27, 0.84])

	ax0.set_title('Before shuffling (%s, %s)' % (line_name[pp], fig_name[tt]),)
	tf = ax0.imshow( Mean_pre_N_sat, origin = 'lower', cmap = 'rainbow', vmin = 1, vmax = 75,)
	plt.colorbar( tf, ax = ax0, fraction = 0.038, pad = 0.01, label = '$N_{sat}$')

	y_ticks = np.arange( -0.5, 19.5 )
	dex_y = np.arange( 0, 19, 3 )

	x_ticks = np.arange( -0.5, 24.5 )
	dex_x = np.arange( 0, 24, 4 )

	ax0.set_xticks( x_ticks[ dex_x ] )
	ax0.set_xticklabels( ['%.1f' % ll for ll in edg_x[dex_x] ] )

	ax0.set_yticks( y_ticks[ dex_y ] )
	ax0.set_yticklabels( ['%.1f' % ll for ll in edg_y[dex_y] ] )

	ax0.set_xlabel( 'X-coordinate of satellites' )
	ax0.set_ylabel( 'Y-coordinate of satellites' )


	ax1.set_title('After shuffling')
	tf = ax1.imshow( Mean_pos_N_sat, origin = 'lower', cmap = 'rainbow', vmin = 1, vmax = 75,)
	plt.colorbar( tf, ax = ax1, fraction = 0.038, pad = 0.01, label = '$N_{sat}$')

	ax1.set_xticks( x_ticks[ dex_x ] )
	ax1.set_xticklabels( ['%.1f' % ll for ll in edg_x[dex_x] ] )

	ax1.set_yticks( y_ticks[ dex_y ] )
	ax1.set_yticklabels( ['%.1f' % ll for ll in edg_y[dex_y] ] )
	ax1.set_xlabel( 'X-coordinate of satellites' )

	ax2.set_title('Diff_img = After shuffling - Before shuffling')
	tf = ax2.imshow( Mean_diff_N_sat / np.mean( Mean_pos_N_sat ), origin = 'lower', cmap = 'bwr', vmin = -1, vmax = 1,)
	plt.colorbar( tf, ax = ax2, fraction = 0.038, pad = 0.01, label = '$N_{diff} \, / \, \\sqrt{ \\bar{N}_{pos}}$')

	ax2.set_xticks( x_ticks[ dex_x ] )
	ax2.set_xticklabels( ['%.1f' % ll for ll in edg_x[dex_x] ] )

	ax2.set_yticks( y_ticks[ dex_y ] )
	ax2.set_yticklabels( ['%.1f' % ll for ll in edg_y[dex_y] ] )
	ax2.set_xlabel( 'X-coordinate of satellites' )

	plt.savefig('/home/xkchen/clust_%d-%d-%s_%s-band_sat_pos_comparison.png' % 
				(bin_rich[pp], bin_rich[pp+1], sub_name[tt], band_str), dpi = 300)
	plt.close()
