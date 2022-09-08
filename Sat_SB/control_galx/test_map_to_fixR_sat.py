import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import numpy as np
import pandas as pds
import h5py

import mechanize
import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C

from io import StringIO
from sklearn.neighbors import KDTree
from sklearn import metrics

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable


### === cosmology model
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



### === ### field galaxy catalog
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/cat/'
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/pre_map/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/map_cat/'

band_str = 'r'


##. field galaxy catalog
cat = pds.read_csv( cat_path + 'random_field-galx_map_%s-band_cat_params.csv' % band_str,)

all_ra, all_dec = np.array( cat['ra'] ), np.array( cat['dec'] )
all_z = np.array( cat['z'] )

all_objID = np.array( cat['objid'] )
all_objID = all_objID.astype( int )

all_mag_u = np.array( cat['modelMag_u'] )
all_mag_g = np.array( cat['modelMag_g'] )
all_mag_r = np.array( cat['modelMag_r'] )
all_mag_i = np.array( cat['modelMag_i'] )
all_mag_z = np.array( cat['modelMag_z'] )

all_dered_u = np.array( cat['dered_u'] )
all_dered_g = np.array( cat['dered_g'] )
all_dered_r = np.array( cat['dered_r'] )
all_dered_i = np.array( cat['dered_i'] )
all_dered_z = np.array( cat['dered_z'] )

all_cmag_u = np.array( cat['cModelMag_u'] )
all_cmag_g = np.array( cat['cModelMag_g'] )
all_cmag_r = np.array( cat['cModelMag_r'] )
all_cmag_i = np.array( cat['cModelMag_i'] )
all_cmag_z = np.array( cat['cModelMag_z'] )

all_Exint_u = np.array( cat['extinction_u'] )
all_Exint_g = np.array( cat['extinction_g'] )
all_Exint_r = np.array( cat['extinction_r'] )
all_Exint_i = np.array( cat['extinction_i'] )
all_Exint_z = np.array( cat['extinction_z'] )

all_coord = SkyCoord( ra = all_ra * U.deg, dec = all_dec * U.deg )

all_gr = all_dered_g - all_dered_r
all_ri = all_dered_r - all_dered_i
all_ug = all_dered_u - all_dered_g
all_gi = all_dered_g - all_dered_i
all_iz = all_dered_i - all_dered_z

# cp_arr = np.array( [ all_cmag_u, all_cmag_g, all_cmag_r, all_cmag_i, all_cmag_z, all_gr, all_ri, all_ug ] ).T
cp_arr = np.array( [ all_cmag_u, all_cmag_g, all_cmag_r, all_cmag_i, all_cmag_z, all_gr, all_ri, all_gi ] ).T

cp_Tree = KDTree( cp_arr )


##. field galaxy image cut information
cat_0 = pds.read_csv( cat_path + 'random_field-galx_map_frame-limit_%s-band_pos.csv' % band_str )

bcg_ra_0, bcg_dec_0, bcg_z_0 = np.array( cat_0['bcg_ra'] ), np.array( cat_0['bcg_dec'] ), np.array( cat_0['bcg_z'] )

sat_ra_0, sat_dec_0, sat_z_0 = np.array( cat_0['sat_ra'] ), np.array( cat_0['sat_dec'] ), np.array( cat_0['sat_z'] )

sat_x0, sat_y0 = np.array( cat_0['cut_cx'] ), np.array( cat_0['cut_cy'] )

sat_coord_0 = SkyCoord( ra = sat_ra_0 * U.deg, dec = sat_dec_0 * U.deg )


#.
cat_1 = pds.read_csv( cat_path + 'random_field-galx_map_frame-limit_%s-band_pos_z-ref.csv' % band_str )

bcg_ra_1, bcg_dec_1, bcg_z_1 = np.array( cat_1['bcg_ra'] ), np.array( cat_1['bcg_dec'] ), np.array( cat_1['bcg_z'] )

sat_ra_1, sat_dec_1, sat_z_1 = np.array( cat_1['sat_ra'] ), np.array( cat_1['sat_dec'] ), np.array( cat_1['sat_z'] )

sat_x1, sat_y1 = np.array( cat_1['sat_x'] ), np.array( cat_1['sat_y'] )

sat_coord_1 = SkyCoord( ra = sat_ra_1 * U.deg, dec = sat_dec_1 * U.deg )



### === ### rich-radius binned subsample match
def mag_color_map():

	##. subsample
	bin_rich = [ 20, 30, 50, 210 ]
	line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']


	##. R_limmits
	# R_str = 'phy'
	# R_bins = np.array( [ 0, 300, 400, 550, 5000] )     ### kpc

	R_str = 'scale'
	R_bins = np.array( [0, 1e-1, 2e-1, 3e-1, 4.5e-1, 1] )   ### times R200m


	if R_str == 'phy':

		fig_name = []
		for dd in range( len(R_bins) - 1 ):

			if dd == 0:
				fig_name.append( '$R \\leq %d \, kpc$' % R_bins[dd + 1] )

			elif dd == len(R_bins) - 2:
				fig_name.append( '$R \\geq %d \, kpc$' % R_bins[dd] )

			else:
				fig_name.append( '$%d \\leq R \\leq %d \, kpc$' % (R_bins[dd], R_bins[dd + 1]),)

	if R_str == 'scale':

		fig_name = []
		for dd in range( len(R_bins) - 1 ):

			if dd == 0:
				fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

			elif dd == len(R_bins) - 2:
				fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

			else:
				fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)

	##.
	for kk in range( 3 ):

		for nn in range( len( R_bins ) - 1 ):

			if R_str == 'phy':

				dat = fits.open( path + 
						'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_mem_params.fits' % 
						(bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1]),)

			if R_str == 'scale':

				dat = fits.open( path + 
						'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_params.fits' % 
						(bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1]),)

			sat_table = dat[1].data

			##.
			lim_z = sat_table['z']

			lim_cmag_u = sat_table['cModelMag_u']
			lim_cmag_g = sat_table['cModelMag_g']
			lim_cmag_r = sat_table['cModelMag_r']
			lim_cmag_i = sat_table['cModelMag_i']
			lim_cmag_z = sat_table['cModelMag_z']

			lim_mag_u = sat_table['modelMag_u']
			lim_mag_g = sat_table['modelMag_g']
			lim_mag_r = sat_table['modelMag_r']
			lim_mag_i = sat_table['modelMag_i']
			lim_mag_z = sat_table['modelMag_z']

			lim_ug = lim_mag_u - lim_mag_g
			lim_gr = lim_mag_g - lim_mag_r
			lim_ri = lim_mag_r - lim_mag_i
			lim_iz = lim_mag_i - lim_mag_z
			lim_gi = lim_mag_g - lim_mag_i

			# lim_arr = np.array( [ lim_cmag_u, lim_cmag_g, lim_cmag_r, lim_cmag_i, lim_cmag_z, lim_gr, lim_ri, lim_ug ] ).T
			lim_arr = np.array( [ lim_cmag_u, lim_cmag_g, lim_cmag_r, lim_cmag_i, lim_cmag_z, lim_gr, lim_ri, lim_gi ] ).T

			map_tree, map_idex = cp_Tree.query( lim_arr, k = 5 )

			#.
			map_cmag_u = all_cmag_u[ map_idex ].flatten()
			map_cmag_g = all_cmag_g[ map_idex ].flatten()
			map_cmag_r = all_cmag_r[ map_idex ].flatten()
			map_cmag_i = all_cmag_i[ map_idex ].flatten()
			map_cmag_z = all_cmag_z[ map_idex ].flatten()

			map_dered_u = all_dered_u[ map_idex ].flatten()
			map_dered_g = all_dered_g[ map_idex ].flatten()
			map_dered_r = all_dered_r[ map_idex ].flatten()
			map_dered_i = all_dered_i[ map_idex ].flatten()
			map_dered_z = all_dered_z[ map_idex ].flatten()

			map_mag_u = all_mag_u[ map_idex ].flatten()
			map_mag_g = all_mag_g[ map_idex ].flatten()
			map_mag_r = all_mag_r[ map_idex ].flatten()
			map_mag_i = all_mag_i[ map_idex ].flatten()
			map_mag_z = all_mag_z[ map_idex ].flatten()

			map_Exint_u = all_Exint_u[ map_idex ].flatten()
			map_Exint_g = all_Exint_g[ map_idex ].flatten()
			map_Exint_r = all_Exint_r[ map_idex ].flatten()
			map_Exint_i = all_Exint_i[ map_idex ].flatten()
			map_Exint_z = all_Exint_z[ map_idex ].flatten()

			map_ra, map_dec = all_ra[ map_idex ].flatten(), all_dec[ map_idex ].flatten()
			map_z = all_z[ map_idex ].flatten()
			map_objID = all_objID[ map_idex ].flatten()


			##. save selected catalog
			keys = ['ra', 'dec', 'z', 'objid', 
					'cModelMag_u', 'cModelMag_g', 'cModelMag_r', 'cModelMag_i', 'cModelMag_z', 
					'modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z', 
					'dered_u', 'dered_g', 'dered_r', 'dered_i', 'dered_z', 
					'extinction_u', 'extinction_g', 'extinction_r', 'extinction_i', 'extinction_z']

			values = [ map_ra, map_dec, map_z, map_objID, 
						map_cmag_u, map_cmag_g, map_cmag_r, map_cmag_i, map_cmag_z, 
						map_mag_u, map_mag_g, map_mag_r, map_mag_i, map_mag_z, 
						map_dered_u, map_dered_g, map_dered_r, map_dered_i, map_dered_z, 
						map_Exint_u, map_Exint_g, map_Exint_r, map_Exint_i, map_Exint_z ]

			##.
			if R_str == 'phy':

				tab_file = Table( values, names = keys )
				tab_file.write( out_path + 
								'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_%s-band_cat.fits' % 
								(bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1], band_str), overwrite = True )

			if R_str == 'scale':

				tab_file = Table( values, names = keys )
				tab_file.write( out_path + 
								'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_%s-band_cat.fits' % 
								(bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1], band_str), overwrite = True )

			map_ug = map_dered_u - map_dered_g
			map_gr = map_dered_g - map_dered_r
			map_gi = map_dered_g - map_dered_i
			map_ri = map_dered_r - map_dered_i
			map_iz = map_dered_i - map_dered_z

			print('Finished mapping!')

			##. figs
			bins_mag_u = np.linspace( np.median( lim_cmag_u ) - 5 * np.std( lim_cmag_u ), np.median( lim_cmag_u ) + 5 * np.std( lim_cmag_u ), 65)
			bins_mag_g = np.linspace( np.median( lim_cmag_g ) - 5 * np.std( lim_cmag_g ), np.median( lim_cmag_g ) + 5 * np.std( lim_cmag_g ), 65)
			bins_mag_r = np.linspace( np.median( lim_cmag_r ) - 5 * np.std( lim_cmag_r ), np.median( lim_cmag_r ) + 5 * np.std( lim_cmag_r ), 65)
			bins_mag_i = np.linspace( np.median( lim_cmag_i ) - 5 * np.std( lim_cmag_i ), np.median( lim_cmag_i ) + 5 * np.std( lim_cmag_i ), 65)
			bins_mag_z = np.linspace( np.median( lim_cmag_z ) - 5 * np.std( lim_cmag_z ), np.median( lim_cmag_z ) + 5 * np.std( lim_cmag_z ), 65)

			bins_ug = np.linspace( np.median( lim_ug ) - 5 * np.std( lim_ug ), np.median( lim_ug ) + 5 * np.std( lim_ug ), 65)
			bins_gr = np.linspace( np.median( lim_gr ) - 5 * np.std( lim_gr ), np.median( lim_gr ) + 5 * np.std( lim_gr ), 65)
			bins_gi = np.linspace( np.median( lim_gi ) - 5 * np.std( lim_gi ), np.median( lim_gi ) + 5 * np.std( lim_gi ), 65)
			bins_ri = np.linspace( np.median( lim_ri ) - 5 * np.std( lim_ri ), np.median( lim_ri ) + 5 * np.std( lim_ri ), 65)
			bins_iz = np.linspace( np.median( lim_iz ) - 5 * np.std( lim_iz ), np.median( lim_iz ) + 5 * np.std( lim_iz ), 65)

			bins_z = np.linspace( np.median( lim_z ) - 5 * np.std( lim_z ), np.median( lim_z ) + 5 * np.std( lim_z ), 65 )


			fig = plt.figure()
			ax = fig.add_axes( [0.11, 0.11, 0.80, 0.85] )
			ax.hist( lim_z, bins = bins_z, density = True, color = 'r', alpha = 0.75, label = 'RedMapper Satellites (%d)' % len( lim_gr ),)
			ax.hist( map_z, bins = bins_z, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')

			ax.annotate( s = fig_name[ nn ] + ', %s-band' % band_str, xy = (0.60, 0.85), xycoords = 'axes fraction',)
			ax.annotate( s = line_name[ kk ], xy = (0.60, 0.70), xycoords = 'axes fraction',)

			ax.legend( loc = 2, frameon = False)
			ax.set_xlabel('$z_{photo}$')
			plt.savefig('/home/xkchen/contrl_sat_z_compare_%d+%d.png' % ( kk, nn), dpi = 300)
			plt.close()


			fig = plt.figure( figsize = (20, 4) )
			axs = gridspec.GridSpec( 1, 5, figure = fig, left = 0.02, bottom = 0.12, right = 0.99, top = 0.96, wspace = 0.12, width_ratios = [1,1,1,1,1],)

			gax = fig.add_subplot( axs[0] )
			gax.hist( lim_cmag_u, bins = bins_mag_u, density = True, color = 'b', alpha = 0.75, label = 'RedMapper Satellites')
			gax.hist( map_cmag_u, bins = bins_mag_u, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
			gax.set_xlabel('cMag_u')
			gax.annotate( s = fig_name[ nn ] + '\n' + '%s-band (%d)' % (band_str, len(lim_gr) ), xy = (0.60, 0.85), xycoords = 'axes fraction',)
			gax.legend( loc = 2, frameon = False)

			gax = fig.add_subplot( axs[1] )
			gax.hist( lim_cmag_g, bins = bins_mag_g, density = True, color = 'g', alpha = 0.75, label = 'RedMapper Satellites')
			gax.hist( map_cmag_g, bins = bins_mag_g, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
			gax.set_xlabel('cMag_g')
			gax.legend( loc = 2, frameon = False)

			gax = fig.add_subplot( axs[2] )
			gax.hist( lim_cmag_r, bins = bins_mag_r, density = True, color = 'r', alpha = 0.75, label = 'RedMapper Satellites')
			gax.hist( map_cmag_r, bins = bins_mag_r, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
			gax.set_xlabel('cMag_r')
			gax.legend( loc = 2, frameon = False)

			gax = fig.add_subplot( axs[3] )
			gax.hist( lim_cmag_i, bins = bins_mag_i, density = True, color = 'm', alpha = 0.75, label = 'RedMapper Satellites')
			gax.hist( map_cmag_i, bins = bins_mag_i, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
			gax.set_xlabel('cMag_i')
			gax.legend( loc = 2, frameon = False)

			gax = fig.add_subplot( axs[4] )
			gax.hist( lim_cmag_z, bins = bins_mag_z, density = True, color = 'c', alpha = 0.75, label = 'RedMapper Satellites')
			gax.hist( map_cmag_z, bins = bins_mag_z, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
			gax.set_xlabel('cMag_z')
			gax.annotate( s = line_name[ kk ], xy = (0.60, 0.75), xycoords = 'axes fraction',)
			gax.legend( loc = 2, frameon = False)

			plt.savefig('/home/xkchen/SDSS_redMapper_mem_mag_%d+%d.png' % ( kk, nn), dpi = 300)
			plt.close()


			fig = plt.figure( figsize = (20, 4) )
			axs = gridspec.GridSpec( 1, 5, figure = fig, left = 0.02, bottom = 0.12, right = 0.99, top = 0.96, wspace = 0.12, width_ratios = [1,1,1,1,1],)

			gax = fig.add_subplot( axs[0] )
			gax.hist( lim_ug, bins = bins_ug, density = True, color = 'b', alpha = 0.75, label = 'RedMapper Satellites')
			gax.hist( map_ug, bins = bins_ug, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
			gax.set_xlabel('u - g')
			gax.annotate( s = fig_name[ nn ] + '\n' + '%s-band (%d)' % (band_str, len(lim_gr) ), xy = (0.60, 0.85), xycoords = 'axes fraction',)

			gax.annotate( s = 'std:%.3f' % np.std(lim_ug) + '\n' + 'aveg:%.3f' % np.mean(lim_ug) + '\n' + 'medi:%.3f' % np.median(lim_ug), 
						xy = (0.15, 0.45), xycoords = 'axes fraction', color = 'b',)

			gax.annotate( s = 'std:%.3f' % np.std(map_ug) + '\n' + 'aveg:%.3f' % np.mean(map_ug) + '\n' + 'medi:%.3f' % np.median(map_ug), 
						xy = (0.75, 0.45), xycoords = 'axes fraction', color = 'k',)

			gax.legend( loc = 2, frameon = False)


			gax = fig.add_subplot( axs[1] )
			gax.hist( lim_gr, bins = bins_gr, density = True, color = 'g', alpha = 0.75, label = 'RedMapper Satellites')
			gax.hist( map_gr, bins = bins_gr, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
			gax.set_xlabel('g - r')

			gax.annotate( s = 'std:%.3f' % np.std(lim_gr) + '\n' + 'aveg:%.3f' % np.mean(lim_gr) + '\n' + 'medi:%.3f' % np.median(lim_gr), 
						xy = (0.15, 0.45), xycoords = 'axes fraction', color = 'g',)

			gax.annotate( s = 'std:%.3f' % np.std(map_gr) + '\n' + 'aveg:%.3f' % np.mean(map_gr) + '\n' + 'medi:%.3f' % np.median(map_gr), 
						xy = (0.75, 0.45), xycoords = 'axes fraction', color = 'k',)

			gax.legend( loc = 2, frameon = False)


			gax = fig.add_subplot( axs[2] )
			gax.hist( lim_gi, bins = bins_gi, density = True, color = 'r', alpha = 0.75, label = 'RedMapper Satellites')
			gax.hist( map_gi, bins = bins_gi, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
			gax.set_xlabel('g - i')

			gax.annotate( s = 'std:%.3f' % np.std(lim_gi) + '\n' + 'aveg:%.3f' % np.mean(lim_gi) + '\n' + 'medi:%.3f' % np.median(lim_gi), 
						xy = (0.15, 0.45), xycoords = 'axes fraction', color = 'r',)

			gax.annotate( s = 'std:%.3f' % np.std(map_gi) + '\n' + 'aveg:%.3f' % np.mean(map_gi) + '\n' + 'medi:%.3f' % np.median(map_gi), 
						xy = (0.75, 0.45), xycoords = 'axes fraction', color = 'k',)

			gax.legend( loc = 2, frameon = False)


			gax = fig.add_subplot( axs[3] )
			gax.hist( lim_ri, bins = bins_ri, density = True, color = 'm', alpha = 0.75, label = 'RedMapper Satellites')
			gax.hist( map_ri, bins = bins_ri, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
			gax.set_xlabel('r - i')

			gax.annotate( s = 'std:%.3f' % np.std(lim_ri) + '\n' + 'aveg:%.3f' % np.mean(lim_ri) + '\n' + 'medi:%.3f' % np.median(lim_ri), 
						xy = (0.15, 0.45), xycoords = 'axes fraction', color = 'm',)

			gax.annotate( s = 'std:%.3f' % np.std(map_ri) + '\n' + 'aveg:%.3f' % np.mean(map_ri) + '\n' + 'medi:%.3f' % np.median(map_ri), 
						xy = (0.75, 0.45), xycoords = 'axes fraction', color = 'k',)

			gax.legend( loc = 2, frameon = False)


			gax = fig.add_subplot( axs[4] )
			gax.hist( lim_iz, bins = bins_iz, density = True, color = 'c', alpha = 0.75, label = 'RedMapper Satellites')
			gax.hist( map_iz, bins = bins_iz, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
			gax.set_xlabel('i - z')

			gax.annotate( s = 'std:%.3f' % np.std(lim_iz) + '\n' + 'aveg:%.3f' % np.mean(lim_iz) + '\n' + 'medi:%.3f' % np.median(lim_iz), 
						xy = (0.15, 0.45), xycoords = 'axes fraction', color = 'c',)

			gax.annotate( s = 'std:%.3f' % np.std(map_iz) + '\n' + 'aveg:%.3f' % np.mean(map_iz) + '\n' + 'medi:%.3f' % np.median(map_iz), 
						xy = (0.75, 0.45), xycoords = 'axes fraction', color = 'k',)

			gax.annotate( s = line_name[ kk ], xy = (0.60, 0.75), xycoords = 'axes fraction',)

			gax.legend( loc = 2, frameon = False)

			plt.savefig('/home/xkchen/SDSS_redMapper_mem_color_%d+%d.png' % ( kk, nn), dpi = 300)
			plt.close()

	return

def fig_properties():

	##. subsample
	bin_rich = [ 20, 30, 50, 210 ]
	line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']


	##. R_limmits
	# R_str = 'phy'
	# R_bins = np.array( [ 0, 300, 400, 550, 5000] )     ### kpc

	R_str = 'scale'
	R_bins = np.array( [0, 1e-1, 2e-1, 3e-1, 4.5e-1, 1] )   ### times R200m

	color_s = ['b', 'g', 'c', 'r', 'm']

	#.
	if R_str == 'phy':

		fig_name = []
		for dd in range( len(R_bins) - 1 ):

			if dd == 0:
				fig_name.append( '$R \\leq %d \, kpc$' % R_bins[dd + 1] )

			elif dd == len(R_bins) - 2:
				fig_name.append( '$R \\geq %d \, kpc$' % R_bins[dd] )

			else:
				fig_name.append( '$%d \\leq R \\leq %d \, kpc$' % (R_bins[dd], R_bins[dd + 1]),)

	#.
	if R_str == 'scale':

		fig_name = []
		for dd in range( len(R_bins) - 1 ):

			if dd == 0:
				fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

			elif dd == len(R_bins) - 2:
				fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

			else:
				fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)

	##.
	for kk in range( 3 ):

		fig = plt.figure( figsize = (16, 9) )
		axs = gridspec.GridSpec( 2, 3, figure = fig, 
								left = 0.03, bottom = 0.05, right = 0.98, top = 0.98, 
								hspace = 0.12, wspace = 0.12, width_ratios = [1,1,1], height_ratios = [1,1])

		for nn in range( len( R_bins ) - 1 ):

			##. member galaxies
			if R_str == 'phy':

				dat = fits.open( path + 
						'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_mem_params.fits' % 
						(bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1]),)

			if R_str == 'scale':

				dat = fits.open( path + 
						'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_params.fits' % 
						(bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1]),)

			sat_table = dat[1].data

			##.
			lim_z = sat_table['z']

			lim_cmag_g = sat_table['cModelMag_g']
			lim_cmag_r = sat_table['cModelMag_r']
			lim_cmag_i = sat_table['cModelMag_i']

			lim_mag_g = sat_table['modelMag_g']
			lim_mag_r = sat_table['modelMag_r']
			lim_mag_i = sat_table['modelMag_i']

			lim_gr = lim_mag_g - lim_mag_r
			lim_ri = lim_mag_r - lim_mag_i
			lim_gi = lim_mag_g - lim_mag_i

			##. filed galaxies
			if R_str == 'phy':

				cat = fits.open( out_path + 
							'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_%s-band_cat.fits' % 
							(bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1], band_str),)

			if R_str == 'scale':

				cat = fits.open( out_path + 
							'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_%s-band_cat.fits' % 
							(bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1], band_str),)

			cat_table = cat[1].data

			map_cmag_g = cat_table['cModelMag_g']
			map_cmag_r = cat_table['cModelMag_r']
			map_cmag_i = cat_table['cModelMag_i']

			map_mag_g = cat_table['dered_g']
			map_mag_r = cat_table['dered_r']
			map_mag_i = cat_table['dered_i']

			map_gr = map_mag_g - map_mag_r
			map_gi = map_mag_g - map_mag_i
			map_ri = map_mag_r - map_mag_i


			##. figs

			bins_mag_g = np.linspace( np.median( lim_cmag_g ) - 5 * np.std( lim_cmag_g ), np.median( lim_cmag_g ) + 5 * np.std( lim_cmag_g ), 65)
			bins_mag_r = np.linspace( np.median( lim_cmag_r ) - 5 * np.std( lim_cmag_r ), np.median( lim_cmag_r ) + 5 * np.std( lim_cmag_r ), 65)
			bins_mag_i = np.linspace( np.median( lim_cmag_i ) - 5 * np.std( lim_cmag_i ), np.median( lim_cmag_i ) + 5 * np.std( lim_cmag_i ), 65)

			bins_gr = np.linspace( np.median( lim_gr ) - 5 * np.std( lim_gr ), np.median( lim_gr ) + 5 * np.std( lim_gr ), 65)
			bins_gi = np.linspace( np.median( lim_gi ) - 5 * np.std( lim_gi ), np.median( lim_gi ) + 5 * np.std( lim_gi ), 65)
			bins_ri = np.linspace( np.median( lim_ri ) - 5 * np.std( lim_ri ), np.median( lim_ri ) + 5 * np.std( lim_ri ), 65)

			gax = fig.add_subplot( axs[0,0] )
			# gax.hist( lim_cmag_g, bins = bins_mag_g, density = True, color = color_s[ nn ], histtype = 'step', alpha = 0.75, label = fig_name[nn],)
			gax.hist( map_cmag_g, bins = bins_mag_g, density = True, color = color_s[ nn ], histtype = 'step', alpha = 0.75, label = fig_name[nn],)

			gax = fig.add_subplot( axs[0,1] )
			# gax.hist( lim_cmag_r, bins = bins_mag_r, density = True, color = color_s[ nn ], histtype = 'step', alpha = 0.75, label = fig_name[nn],)
			gax.hist( map_cmag_r, bins = bins_mag_r, density = True, color = color_s[ nn ], histtype = 'step', alpha = 0.75, label = fig_name[nn],)

			gax = fig.add_subplot( axs[0,2] )
			# gax.hist( lim_cmag_i, bins = bins_mag_i, density = True, color = color_s[ nn ], histtype = 'step', alpha = 0.75, label = fig_name[nn],)
			gax.hist( map_cmag_i, bins = bins_mag_i, density = True, color = color_s[ nn ], histtype = 'step', alpha = 0.75, label = fig_name[nn],)


			gax = fig.add_subplot( axs[1,0] )
			# gax.hist( lim_gr, bins = bins_gr, density = True, color = color_s[ nn ], histtype = 'step', alpha = 0.75, label = fig_name[nn],)
			gax.hist( map_gr, bins = bins_gr, density = True, color = color_s[ nn ], histtype = 'step', alpha = 0.75, label = fig_name[nn],)

			gax = fig.add_subplot( axs[1,1] )
			# gax.hist( lim_gi, bins = bins_gi, density = True, color = color_s[ nn ], histtype = 'step', alpha = 0.75, label = fig_name[nn],)
			gax.hist( map_gi, bins = bins_gi, density = True, color = color_s[ nn ], histtype = 'step', alpha = 0.75, label = fig_name[nn],)

			gax = fig.add_subplot( axs[1,2] )
			# gax.hist( lim_ri, bins = bins_ri, density = True, color = color_s[ nn ], histtype = 'step', alpha = 0.75, label = fig_name[nn],)
			gax.hist( map_ri, bins = bins_ri, density = True, color = color_s[ nn ], histtype = 'step', alpha = 0.75, label = fig_name[nn],)

		gax = fig.add_subplot( axs[0,0] )
		gax.set_xlabel('cMag_g')
		# gax.annotate( s = 'RedMapper Satellites' + '\n' + line_name[kk], xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'k',)
		gax.annotate( s = 'Control' + '\n' + line_name[kk], xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'k',)
		gax.legend( loc = 1, frameon = False)

		gax = fig.add_subplot( axs[0,1] )
		gax.set_xlabel('cMag_r')
		# gax.annotate( s = 'RedMapper Satellites' + '\n' + line_name[kk], xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'k',)
		gax.annotate( s = 'Control' + '\n' + line_name[kk], xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'k',)
		gax.legend( loc = 1, frameon = False)

		gax = fig.add_subplot( axs[0,2] )
		gax.set_xlabel('cMag_i')
		# gax.annotate( s = 'RedMapper Satellites' + '\n' + line_name[kk], xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'k',)
		gax.annotate( s = 'Control' + '\n' + line_name[kk], xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'k',)
		gax.legend( loc = 1, frameon = False)

		gax = fig.add_subplot( axs[1,0] )
		gax.set_xlabel('g - r')
		# gax.annotate( s = 'RedMapper Satellites' + '\n' + line_name[kk], xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'k',)
		gax.annotate( s = 'Control' + '\n' + line_name[kk], xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'k',)
		gax.legend( loc = 1, frameon = False)

		gax = fig.add_subplot( axs[1,1] )
		gax.set_xlabel('g - i')
		# gax.annotate( s = 'RedMapper Satellites' + '\n' + line_name[kk], xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'k',)
		gax.annotate( s = 'Control' + '\n' + line_name[kk], xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'k',)
		gax.legend( loc = 1, frameon = False)

		gax = fig.add_subplot( axs[1,2] )
		gax.set_xlabel('r - i')
		# gax.annotate( s = 'RedMapper Satellites' + '\n' + line_name[kk], xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'k',)
		gax.annotate( s = 'Control' + '\n' + line_name[kk], xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'k',)
		gax.legend( loc = 1, frameon = False)

		# plt.savefig('/home/xkchen/redMap_rich_%d-%d_%s-binned_sat_params_compare.png' % (bin_rich[kk], bin_rich[kk + 1], R_str), dpi = 300)
		plt.savefig('/home/xkchen/redMap_rich_%d-%d_%s-binned_Contrl-galx_params_compare.png' % (bin_rich[kk], bin_rich[kk + 1], R_str), dpi = 300)
		plt.close()

	return

# mag_color_map()
# fig_properties()
# raise


### ... mapping stacking information
def radiu_rich_bin_match():

	##. subsample
	bin_rich = [ 20, 30, 50, 210 ]

	##. R_limmits
	# R_str = 'phy'
	# R_bins = np.array( [ 0, 300, 400, 550, 5000] )     ### kpc

	R_str = 'scale'
	R_bins = np.array( [0, 1e-1, 2e-1, 3e-1, 4.5e-1, 1] )   ### times R200m


	for kk in range( 3 ):

		for nn in range( len( R_bins ) - 1 ):

			if R_str == 'phy':

				dat = fits.open( out_path + 
							'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_%s-band_cat.fits' % 
							(bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1], band_str),)

			if R_str == 'scale':

				dat = fits.open( out_path + 
							'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_%s-band_cat.fits' % 
							(bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1], band_str),)

			dat_table = dat[1].data

			sub_ra, sub_dec = np.array( dat_table['ra'] ), np.array( dat_table['dec'] )

			sub_coord = SkyCoord( ra = sub_ra * U.deg, dec = sub_dec * U.deg )

			##.
			idx, d2d, d3d = sub_coord.match_to_catalog_sky( sat_coord_0 )
			id_lim = d2d.value < 2.7e-4

			mp_ra = bcg_ra_0[ idx[ id_lim ] ]
			mp_dec = bcg_dec_0[ idx[ id_lim ] ]
			mp_z = bcg_z_0[ idx[ id_lim ] ]

			mp_sat_ra = sat_ra_0[ idx[ id_lim ] ]
			mp_sat_dec = sat_dec_0[ idx[ id_lim ] ]
			mp_sat_z = sat_z_0[ idx[ id_lim ] ]

			mp_sx = sat_x0[ idx[ id_lim ] ]
			mp_sy = sat_y0[ idx[ id_lim ] ]

			#.
			keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_z', 'sat_x', 'sat_y']
			values = [ mp_ra, mp_dec, mp_z, mp_sat_ra, mp_sat_dec, mp_sat_z, mp_sx, mp_sy ]
			fill = dict( zip( keys, values ) )
			data = pds.DataFrame( fill )

			if R_str == 'phy':
				data.to_csv( out_path + 
							'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_%s-band_cat_pos.csv' % 
							(bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1], band_str),)

			if R_str == 'scale':
				data.to_csv( out_path + 
							'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_%s-band_cat_pos.csv' % 
							(bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1], band_str),)


			##.
			idx, d2d, d3d = sub_coord.match_to_catalog_sky( sat_coord_1 )
			id_lim = d2d.value < 2.7e-4

			mp_ra = bcg_ra_1[ idx[ id_lim ] ]
			mp_dec = bcg_dec_1[ idx[ id_lim ] ]
			mp_z = bcg_z_1[ idx[ id_lim ] ]

			mp_sat_ra = sat_ra_1[ idx[ id_lim ] ]
			mp_sat_dec = sat_dec_1[ idx[ id_lim ] ]
			mp_sat_z = sat_z_1[ idx[ id_lim ] ]

			mp_sx = sat_x1[ idx[ id_lim ] ]
			mp_sy = sat_y1[ idx[ id_lim ] ]

			#.
			keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_z', 'sat_x', 'sat_y']
			values = [ mp_ra, mp_dec, mp_z, mp_sat_ra, mp_sat_dec, mp_sat_z, mp_sx, mp_sy ]
			fill = dict( zip( keys, values ) )
			data = pds.DataFrame( fill )

			if R_str == 'phy':
				data.to_csv( out_path + 
							'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_%s-band_cat_zref-pos.csv' % 
							(bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1], band_str),)

			if R_str == 'scale':
				data.to_csv( out_path + 
							'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_%s-band_cat_zref-pos.csv' % 
							(bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1], band_str),)

	return

# radiu_rich_bin_match()



### === ### Radius bin over_all richness
def radius_bin_over_rich():

	##... subsample combine
	bin_rich = [ 20, 30, 50, 210 ]

	##. R_limmits
	R_str = 'scale'
	R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m

	fig_name = []
	for dd in range( len(R_bins) - 1 ):

		if dd == 0:
			fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

		elif dd == len(R_bins) - 2:
			fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

		else:
			fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)


	for nn in range( len( R_bins ) - 1 ):

		##.
		keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'R_sat', 'R2Rv', 'clus_ID', 'objID', 'z', 'zErr', 
				'cModelMag_u', 'cModelMag_g', 'cModelMag_r', 'cModelMag_i', 'cModelMag_z', 
				'modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z' ]

		N_ks = len( keys )

		tmp_arr = []

		for kk in range( 3 ):

			dat = fits.open( path + 
					'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_params.fits' % 
					(bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1]),)

			sat_table = dat[1].data

			if kk == 0:

				for dd in range( N_ks ):

					tmp_arr.append( np.array( sat_table[ keys[ dd ] ] ) )

			else:

				for dd in range( N_ks ):

					tmp_arr[ dd ] = np.r_[ tmp_arr[ dd ], np.array( sat_table[ keys[ dd ] ] ) ]			

		##.
		tab_file = Table( tmp_arr, names = keys )
		tab_file.write( path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_%.2f-%.2fR200m_mem_params.fits' % 
								(R_bins[nn], R_bins[nn + 1]), overwrite = True )


	##... properties match
	for nn in range( len( R_bins ) - 1 ):

		dat = fits.open( path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_%.2f-%.2fR200m_mem_params.fits' % (R_bins[nn], R_bins[nn + 1]),)

		sat_table = dat[1].data

		##.
		lim_z = sat_table['z']

		lim_cmag_u = sat_table['cModelMag_u']
		lim_cmag_g = sat_table['cModelMag_g']
		lim_cmag_r = sat_table['cModelMag_r']
		lim_cmag_i = sat_table['cModelMag_i']
		lim_cmag_z = sat_table['cModelMag_z']

		lim_mag_u = sat_table['modelMag_u']
		lim_mag_g = sat_table['modelMag_g']
		lim_mag_r = sat_table['modelMag_r']
		lim_mag_i = sat_table['modelMag_i']
		lim_mag_z = sat_table['modelMag_z']

		lim_ug = lim_mag_u - lim_mag_g
		lim_gr = lim_mag_g - lim_mag_r
		lim_ri = lim_mag_r - lim_mag_i
		lim_iz = lim_mag_i - lim_mag_z
		lim_gi = lim_mag_g - lim_mag_i

		# lim_arr = np.array( [ lim_cmag_u, lim_cmag_g, lim_cmag_r, lim_cmag_i, lim_cmag_z, lim_gr, lim_ri, lim_ug ] ).T
		lim_arr = np.array( [ lim_cmag_u, lim_cmag_g, lim_cmag_r, lim_cmag_i, lim_cmag_z, lim_gr, lim_ri, lim_gi ] ).T

		if nn == len( R_bins ) - 2:
			map_tree, map_idex = cp_Tree.query( lim_arr, k = 10 )

		else:
			map_tree, map_idex = cp_Tree.query( lim_arr, k = 5 )

		#.
		map_cmag_u = all_cmag_u[ map_idex ].flatten()
		map_cmag_g = all_cmag_g[ map_idex ].flatten()
		map_cmag_r = all_cmag_r[ map_idex ].flatten()
		map_cmag_i = all_cmag_i[ map_idex ].flatten()
		map_cmag_z = all_cmag_z[ map_idex ].flatten()

		map_dered_u = all_dered_u[ map_idex ].flatten()
		map_dered_g = all_dered_g[ map_idex ].flatten()
		map_dered_r = all_dered_r[ map_idex ].flatten()
		map_dered_i = all_dered_i[ map_idex ].flatten()
		map_dered_z = all_dered_z[ map_idex ].flatten()

		map_mag_u = all_mag_u[ map_idex ].flatten()
		map_mag_g = all_mag_g[ map_idex ].flatten()
		map_mag_r = all_mag_r[ map_idex ].flatten()
		map_mag_i = all_mag_i[ map_idex ].flatten()
		map_mag_z = all_mag_z[ map_idex ].flatten()

		map_Exint_u = all_Exint_u[ map_idex ].flatten()
		map_Exint_g = all_Exint_g[ map_idex ].flatten()
		map_Exint_r = all_Exint_r[ map_idex ].flatten()
		map_Exint_i = all_Exint_i[ map_idex ].flatten()
		map_Exint_z = all_Exint_z[ map_idex ].flatten()

		map_ra, map_dec = all_ra[ map_idex ].flatten(), all_dec[ map_idex ].flatten()
		map_z = all_z[ map_idex ].flatten()
		map_objID = all_objID[ map_idex ].flatten()


		##. save selected catalog
		keys = ['ra', 'dec', 'z', 'objid', 
				'cModelMag_u', 'cModelMag_g', 'cModelMag_r', 'cModelMag_i', 'cModelMag_z', 
				'modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z', 
				'dered_u', 'dered_g', 'dered_r', 'dered_i', 'dered_z', 
				'extinction_u', 'extinction_g', 'extinction_r', 'extinction_i', 'extinction_z']

		values = [ map_ra, map_dec, map_z, map_objID, 
					map_cmag_u, map_cmag_g, map_cmag_r, map_cmag_i, map_cmag_z, 
					map_mag_u, map_mag_g, map_mag_r, map_mag_i, map_mag_z, 
					map_dered_u, map_dered_g, map_dered_r, map_dered_i, map_dered_z, 
					map_Exint_u, map_Exint_g, map_Exint_r, map_Exint_i, map_Exint_z ]

		tab_file = Table( values, names = keys )
		tab_file.write( out_path + 'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_rich_%.2f-%.2fR200m_%s-band_cat.fits' % 
						(R_bins[nn], R_bins[nn + 1], band_str), overwrite = True )

		##. figs
		map_ug = map_dered_u - map_dered_g
		map_gr = map_dered_g - map_dered_r
		map_gi = map_dered_g - map_dered_i
		map_ri = map_dered_r - map_dered_i
		map_iz = map_dered_i - map_dered_z

		print('Finished mapping!')

		##. figs
		bins_mag_u = np.linspace( np.median( lim_cmag_u ) - 5 * np.std( lim_cmag_u ), np.median( lim_cmag_u ) + 5 * np.std( lim_cmag_u ), 65)
		bins_mag_g = np.linspace( np.median( lim_cmag_g ) - 5 * np.std( lim_cmag_g ), np.median( lim_cmag_g ) + 5 * np.std( lim_cmag_g ), 65)
		bins_mag_r = np.linspace( np.median( lim_cmag_r ) - 5 * np.std( lim_cmag_r ), np.median( lim_cmag_r ) + 5 * np.std( lim_cmag_r ), 65)
		bins_mag_i = np.linspace( np.median( lim_cmag_i ) - 5 * np.std( lim_cmag_i ), np.median( lim_cmag_i ) + 5 * np.std( lim_cmag_i ), 65)
		bins_mag_z = np.linspace( np.median( lim_cmag_z ) - 5 * np.std( lim_cmag_z ), np.median( lim_cmag_z ) + 5 * np.std( lim_cmag_z ), 65)

		bins_ug = np.linspace( np.median( lim_ug ) - 5 * np.std( lim_ug ), np.median( lim_ug ) + 5 * np.std( lim_ug ), 65)
		bins_gr = np.linspace( np.median( lim_gr ) - 5 * np.std( lim_gr ), np.median( lim_gr ) + 5 * np.std( lim_gr ), 65)
		bins_gi = np.linspace( np.median( lim_gi ) - 5 * np.std( lim_gi ), np.median( lim_gi ) + 5 * np.std( lim_gi ), 65)
		bins_ri = np.linspace( np.median( lim_ri ) - 5 * np.std( lim_ri ), np.median( lim_ri ) + 5 * np.std( lim_ri ), 65)
		bins_iz = np.linspace( np.median( lim_iz ) - 5 * np.std( lim_iz ), np.median( lim_iz ) + 5 * np.std( lim_iz ), 65)

		bins_z = np.linspace( np.median( lim_z ) - 5 * np.std( lim_z ), np.median( lim_z ) + 5 * np.std( lim_z ), 65 )


		fig = plt.figure()
		ax = fig.add_axes( [0.11, 0.11, 0.80, 0.85] )
		ax.hist( lim_z, bins = bins_z, density = True, color = 'r', alpha = 0.75, label = 'RedMapper Satellites (%d)' % len( lim_gr ),)
		ax.hist( map_z, bins = bins_z, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')

		ax.annotate( s = fig_name[ nn ] + ', %s-band' % band_str, xy = (0.60, 0.85), xycoords = 'axes fraction',)

		ax.legend( loc = 2, frameon = False)
		ax.set_xlabel('$z_{photo}$')
		plt.savefig('/home/xkchen/contrl_sat_z_compare_%d+%d.png' % ( kk, nn), dpi = 300)
		plt.close()


		fig = plt.figure( figsize = (20, 4) )
		axs = gridspec.GridSpec( 1, 5, figure = fig, left = 0.02, bottom = 0.12, right = 0.99, top = 0.96, wspace = 0.12, width_ratios = [1,1,1,1,1],)

		gax = fig.add_subplot( axs[0] )
		gax.hist( lim_cmag_u, bins = bins_mag_u, density = True, color = 'b', alpha = 0.75, label = 'RedMapper Satellites')
		gax.hist( map_cmag_u, bins = bins_mag_u, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
		gax.set_xlabel('cMag_u')
		gax.annotate( s = fig_name[ nn ] + '\n' + '%s-band (%d)' % (band_str, len(lim_gr) ), xy = (0.60, 0.85), xycoords = 'axes fraction',)
		gax.legend( loc = 2, frameon = False)

		gax = fig.add_subplot( axs[1] )
		gax.hist( lim_cmag_g, bins = bins_mag_g, density = True, color = 'g', alpha = 0.75, label = 'RedMapper Satellites')
		gax.hist( map_cmag_g, bins = bins_mag_g, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
		gax.set_xlabel('cMag_g')
		gax.legend( loc = 2, frameon = False)

		gax = fig.add_subplot( axs[2] )
		gax.hist( lim_cmag_r, bins = bins_mag_r, density = True, color = 'r', alpha = 0.75, label = 'RedMapper Satellites')
		gax.hist( map_cmag_r, bins = bins_mag_r, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
		gax.set_xlabel('cMag_r')
		gax.legend( loc = 2, frameon = False)

		gax = fig.add_subplot( axs[3] )
		gax.hist( lim_cmag_i, bins = bins_mag_i, density = True, color = 'm', alpha = 0.75, label = 'RedMapper Satellites')
		gax.hist( map_cmag_i, bins = bins_mag_i, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
		gax.set_xlabel('cMag_i')
		gax.legend( loc = 2, frameon = False)

		gax = fig.add_subplot( axs[4] )
		gax.hist( lim_cmag_z, bins = bins_mag_z, density = True, color = 'c', alpha = 0.75, label = 'RedMapper Satellites')
		gax.hist( map_cmag_z, bins = bins_mag_z, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
		gax.set_xlabel('cMag_z')
		gax.legend( loc = 2, frameon = False)

		plt.savefig('/home/xkchen/SDSS_redMapper_mem_mag_%d+%d.png' % ( kk, nn), dpi = 300)
		plt.close()


		fig = plt.figure( figsize = (20, 4) )
		axs = gridspec.GridSpec( 1, 5, figure = fig, left = 0.02, bottom = 0.12, right = 0.99, top = 0.96, wspace = 0.12, width_ratios = [1,1,1,1,1],)

		gax = fig.add_subplot( axs[0] )
		gax.hist( lim_ug, bins = bins_ug, density = True, color = 'b', alpha = 0.75, label = 'RedMapper Satellites')
		gax.hist( map_ug, bins = bins_ug, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
		gax.set_xlabel('u - g')
		gax.annotate( s = fig_name[ nn ] + '\n' + '%s-band (%d)' % (band_str, len(lim_gr) ), xy = (0.60, 0.85), xycoords = 'axes fraction',)

		gax.annotate( s = 'std:%.3f' % np.std(lim_ug) + '\n' + 'aveg:%.3f' % np.mean(lim_ug) + '\n' + 'medi:%.3f' % np.median(lim_ug), 
					xy = (0.15, 0.45), xycoords = 'axes fraction', color = 'b',)

		gax.annotate( s = 'std:%.3f' % np.std(map_ug) + '\n' + 'aveg:%.3f' % np.mean(map_ug) + '\n' + 'medi:%.3f' % np.median(map_ug), 
					xy = (0.75, 0.45), xycoords = 'axes fraction', color = 'k',)

		gax.legend( loc = 2, frameon = False)


		gax = fig.add_subplot( axs[1] )
		gax.hist( lim_gr, bins = bins_gr, density = True, color = 'g', alpha = 0.75, label = 'RedMapper Satellites')
		gax.hist( map_gr, bins = bins_gr, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
		gax.set_xlabel('g - r')

		gax.annotate( s = 'std:%.3f' % np.std(lim_gr) + '\n' + 'aveg:%.3f' % np.mean(lim_gr) + '\n' + 'medi:%.3f' % np.median(lim_gr), 
					xy = (0.15, 0.45), xycoords = 'axes fraction', color = 'g',)

		gax.annotate( s = 'std:%.3f' % np.std(map_gr) + '\n' + 'aveg:%.3f' % np.mean(map_gr) + '\n' + 'medi:%.3f' % np.median(map_gr), 
					xy = (0.75, 0.45), xycoords = 'axes fraction', color = 'k',)

		gax.legend( loc = 2, frameon = False)


		gax = fig.add_subplot( axs[2] )
		gax.hist( lim_gi, bins = bins_gi, density = True, color = 'r', alpha = 0.75, label = 'RedMapper Satellites')
		gax.hist( map_gi, bins = bins_gi, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
		gax.set_xlabel('g - i')

		gax.annotate( s = 'std:%.3f' % np.std(lim_gi) + '\n' + 'aveg:%.3f' % np.mean(lim_gi) + '\n' + 'medi:%.3f' % np.median(lim_gi), 
					xy = (0.15, 0.45), xycoords = 'axes fraction', color = 'r',)

		gax.annotate( s = 'std:%.3f' % np.std(map_gi) + '\n' + 'aveg:%.3f' % np.mean(map_gi) + '\n' + 'medi:%.3f' % np.median(map_gi), 
					xy = (0.75, 0.45), xycoords = 'axes fraction', color = 'k',)

		gax.legend( loc = 2, frameon = False)


		gax = fig.add_subplot( axs[3] )
		gax.hist( lim_ri, bins = bins_ri, density = True, color = 'm', alpha = 0.75, label = 'RedMapper Satellites')
		gax.hist( map_ri, bins = bins_ri, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
		gax.set_xlabel('r - i')

		gax.annotate( s = 'std:%.3f' % np.std(lim_ri) + '\n' + 'aveg:%.3f' % np.mean(lim_ri) + '\n' + 'medi:%.3f' % np.median(lim_ri), 
					xy = (0.15, 0.45), xycoords = 'axes fraction', color = 'm',)

		gax.annotate( s = 'std:%.3f' % np.std(map_ri) + '\n' + 'aveg:%.3f' % np.mean(map_ri) + '\n' + 'medi:%.3f' % np.median(map_ri), 
					xy = (0.75, 0.45), xycoords = 'axes fraction', color = 'k',)

		gax.legend( loc = 2, frameon = False)


		gax = fig.add_subplot( axs[4] )
		gax.hist( lim_iz, bins = bins_iz, density = True, color = 'c', alpha = 0.75, label = 'RedMapper Satellites')
		gax.hist( map_iz, bins = bins_iz, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
		gax.set_xlabel('i - z')

		gax.annotate( s = 'std:%.3f' % np.std(lim_iz) + '\n' + 'aveg:%.3f' % np.mean(lim_iz) + '\n' + 'medi:%.3f' % np.median(lim_iz), 
					xy = (0.15, 0.45), xycoords = 'axes fraction', color = 'c',)

		gax.annotate( s = 'std:%.3f' % np.std(map_iz) + '\n' + 'aveg:%.3f' % np.mean(map_iz) + '\n' + 'medi:%.3f' % np.median(map_iz), 
					xy = (0.75, 0.45), xycoords = 'axes fraction', color = 'k',)

		gax.legend( loc = 2, frameon = False)

		plt.savefig('/home/xkchen/SDSS_redMapper_mem_color_%d+%d.png' % ( kk, nn), dpi = 300)
		plt.close()

	return

# radius_bin_over_rich()
# raise

###... stacking information
def radiu_bin_over_rich__match():

	##. mapped field galaxy
	R_str = 'scale'
	R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m

	for nn in range( len( R_bins ) - 1 ):

		dat = fits.open( out_path + 
							'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_rich_%.2f-%.2fR200m_%s-band_cat.fits' % 
							(R_bins[nn], R_bins[nn + 1], band_str), )

		dat_table = dat[1].data

		sub_ra, sub_dec = np.array( dat_table['ra'] ), np.array( dat_table['dec'] )

		sub_coord = SkyCoord( ra = sub_ra * U.deg, dec = sub_dec * U.deg )

		##.
		idx, d2d, d3d = sub_coord.match_to_catalog_sky( sat_coord_0 )
		id_lim = d2d.value < 2.7e-4

		mp_ra = bcg_ra_0[ idx[ id_lim ] ]
		mp_dec = bcg_dec_0[ idx[ id_lim ] ]
		mp_z = bcg_z_0[ idx[ id_lim ] ]

		mp_sat_ra = sat_ra_0[ idx[ id_lim ] ]
		mp_sat_dec = sat_dec_0[ idx[ id_lim ] ]
		mp_sat_z = sat_z_0[ idx[ id_lim ] ]

		mp_sx = sat_x0[ idx[ id_lim ] ]
		mp_sy = sat_y0[ idx[ id_lim ] ]

		#.
		keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_z', 'sat_x', 'sat_y']
		values = [ mp_ra, mp_dec, mp_z, mp_sat_ra, mp_sat_dec, mp_sat_z, mp_sx, mp_sy ]
		fill = dict( zip( keys, values ) )
		data = pds.DataFrame( fill )
		data.to_csv( out_path + 'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_%.2f-%.2fR200m_%s-band_cat_pos.csv' % 
					(R_bins[nn], R_bins[nn + 1], band_str),)


		##.
		idx, d2d, d3d = sub_coord.match_to_catalog_sky( sat_coord_1 )
		id_lim = d2d.value < 2.7e-4

		mp_ra = bcg_ra_1[ idx[ id_lim ] ]
		mp_dec = bcg_dec_1[ idx[ id_lim ] ]
		mp_z = bcg_z_1[ idx[ id_lim ] ]

		mp_sat_ra = sat_ra_1[ idx[ id_lim ] ]
		mp_sat_dec = sat_dec_1[ idx[ id_lim ] ]
		mp_sat_z = sat_z_1[ idx[ id_lim ] ]

		mp_sx = sat_x1[ idx[ id_lim ] ]
		mp_sy = sat_y1[ idx[ id_lim ] ]

		#.
		keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_z', 'sat_x', 'sat_y']
		values = [ mp_ra, mp_dec, mp_z, mp_sat_ra, mp_sat_dec, mp_sat_z, mp_sx, mp_sy ]
		fill = dict( zip( keys, values ) )
		data = pds.DataFrame( fill )
		data.to_csv( out_path + 'contrl-galx_Extend-BCGM_frame-lim_Pm-cut_%.2f-%.2fR200m_%s-band_cat_zref-pos.csv' % 
					(R_bins[nn], R_bins[nn + 1], band_str),)

	print('To here!')
	return

# radiu_bin_over_rich__match()
# raise



### === ### mapping to alll member galaxy with Pm >= 0.8

###... redMap member
dat = fits.open( path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_mem_params.fits',)

sat_table = dat[1].data

##.
lim_z = sat_table['z']

lim_cmag_u = sat_table['cModelMag_u']
lim_cmag_g = sat_table['cModelMag_g']
lim_cmag_r = sat_table['cModelMag_r']
lim_cmag_i = sat_table['cModelMag_i']
lim_cmag_z = sat_table['cModelMag_z']

lim_mag_u = sat_table['modelMag_u']
lim_mag_g = sat_table['modelMag_g']
lim_mag_r = sat_table['modelMag_r']
lim_mag_i = sat_table['modelMag_i']
lim_mag_z = sat_table['modelMag_z']

lim_ug = lim_mag_u - lim_mag_g
lim_gr = lim_mag_g - lim_mag_r
lim_ri = lim_mag_r - lim_mag_i
lim_iz = lim_mag_i - lim_mag_z
lim_gi = lim_mag_g - lim_mag_i

# lim_arr = np.array( [ lim_cmag_u, lim_cmag_g, lim_cmag_r, lim_cmag_i, lim_cmag_z, lim_gr, lim_ri, lim_ug ] ).T
lim_arr = np.array( [ lim_cmag_u, lim_cmag_g, lim_cmag_r, lim_cmag_i, lim_cmag_z, lim_gr, lim_ri, lim_gi ] ).T

map_tree, map_idex = cp_Tree.query( lim_arr, k = 5 )

#.
map_cmag_u = all_cmag_u[ map_idex ].flatten()
map_cmag_g = all_cmag_g[ map_idex ].flatten()
map_cmag_r = all_cmag_r[ map_idex ].flatten()
map_cmag_i = all_cmag_i[ map_idex ].flatten()
map_cmag_z = all_cmag_z[ map_idex ].flatten()

map_dered_u = all_dered_u[ map_idex ].flatten()
map_dered_g = all_dered_g[ map_idex ].flatten()
map_dered_r = all_dered_r[ map_idex ].flatten()
map_dered_i = all_dered_i[ map_idex ].flatten()
map_dered_z = all_dered_z[ map_idex ].flatten()

map_mag_u = all_mag_u[ map_idex ].flatten()
map_mag_g = all_mag_g[ map_idex ].flatten()
map_mag_r = all_mag_r[ map_idex ].flatten()
map_mag_i = all_mag_i[ map_idex ].flatten()
map_mag_z = all_mag_z[ map_idex ].flatten()

map_Exint_u = all_Exint_u[ map_idex ].flatten()
map_Exint_g = all_Exint_g[ map_idex ].flatten()
map_Exint_r = all_Exint_r[ map_idex ].flatten()
map_Exint_i = all_Exint_i[ map_idex ].flatten()
map_Exint_z = all_Exint_z[ map_idex ].flatten()

map_ra, map_dec = all_ra[ map_idex ].flatten(), all_dec[ map_idex ].flatten()
map_z = all_z[ map_idex ].flatten()
map_objID = all_objID[ map_idex ].flatten()


##. save selected catalog
keys = ['ra', 'dec', 'z', 'objid', 
		'cModelMag_u', 'cModelMag_g', 'cModelMag_r', 'cModelMag_i', 'cModelMag_z', 
		'modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z', 
		'dered_u', 'dered_g', 'dered_r', 'dered_i', 'dered_z', 
		'extinction_u', 'extinction_g', 'extinction_r', 'extinction_i', 'extinction_z']

values = [ map_ra, map_dec, map_z, map_objID, 
			map_cmag_u, map_cmag_g, map_cmag_r, map_cmag_i, map_cmag_z, 
			map_mag_u, map_mag_g, map_mag_r, map_mag_i, map_mag_z, 
			map_dered_u, map_dered_g, map_dered_r, map_dered_i, map_dered_z, 
			map_Exint_u, map_Exint_g, map_Exint_r, map_Exint_i, map_Exint_z ]

tab_file = Table( values, names = keys )
tab_file.write( out_path + 'contrl-galx_Extend-BCGM_rgi-common_frame-lim_Pm-cut_%s-band_cat.fits' % band_str, overwrite = True )


##. figs
map_ug = map_dered_u - map_dered_g
map_gr = map_dered_g - map_dered_r
map_gi = map_dered_g - map_dered_i
map_ri = map_dered_r - map_dered_i
map_iz = map_dered_i - map_dered_z

print('Finished mapping!')

##. stacking information
pat = fits.open( out_path + 'contrl-galx_Extend-BCGM_rgi-common_frame-lim_Pm-cut_%s-band_cat.fits' % band_str,)

pat_table = pat[1].data
sub_ra, sub_dec = np.array( pat_table['ra'] ), np.array( pat_table['dec'] )

sub_coord = SkyCoord( ra = sub_ra * U.deg, dec = sub_dec * U.deg )

##.
idx, d2d, d3d = sub_coord.match_to_catalog_sky( sat_coord_0 )
id_lim = d2d.value < 2.7e-4

mp_ra = bcg_ra_0[ idx[ id_lim ] ]
mp_dec = bcg_dec_0[ idx[ id_lim ] ]
mp_z = bcg_z_0[ idx[ id_lim ] ]

mp_sat_ra = sat_ra_0[ idx[ id_lim ] ]
mp_sat_dec = sat_dec_0[ idx[ id_lim ] ]
mp_sat_z = sat_z_0[ idx[ id_lim ] ]

mp_sx = sat_x0[ idx[ id_lim ] ]
mp_sy = sat_y0[ idx[ id_lim ] ]

#.
keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_z', 'sat_x', 'sat_y']
values = [ mp_ra, mp_dec, mp_z, mp_sat_ra, mp_sat_dec, mp_sat_z, mp_sx, mp_sy ]
fill = dict( zip( keys, values ) )
data = pds.DataFrame( fill )
data.to_csv( out_path + 'contrl-galx_Extend-BCGM_rgi-common_frame-lim_Pm-cut_%s-band_cat_pos.csv' % band_str,)


##.
idx, d2d, d3d = sub_coord.match_to_catalog_sky( sat_coord_1 )
id_lim = d2d.value < 2.7e-4

mp_ra = bcg_ra_1[ idx[ id_lim ] ]
mp_dec = bcg_dec_1[ idx[ id_lim ] ]
mp_z = bcg_z_1[ idx[ id_lim ] ]

mp_sat_ra = sat_ra_1[ idx[ id_lim ] ]
mp_sat_dec = sat_dec_1[ idx[ id_lim ] ]
mp_sat_z = sat_z_1[ idx[ id_lim ] ]

mp_sx = sat_x1[ idx[ id_lim ] ]
mp_sy = sat_y1[ idx[ id_lim ] ]

#.
keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_z', 'sat_x', 'sat_y']
values = [ mp_ra, mp_dec, mp_z, mp_sat_ra, mp_sat_dec, mp_sat_z, mp_sx, mp_sy ]
fill = dict( zip( keys, values ) )
data = pds.DataFrame( fill )
data.to_csv( out_path + 'contrl-galx_Extend-BCGM_rgi-common_frame-lim_Pm-cut_%s-band_cat_zref-pos.csv' % band_str,)


raise

##. figs
bins_mag_u = np.linspace( np.median( lim_cmag_u ) - 5 * np.std( lim_cmag_u ), np.median( lim_cmag_u ) + 5 * np.std( lim_cmag_u ), 65)
bins_mag_g = np.linspace( np.median( lim_cmag_g ) - 5 * np.std( lim_cmag_g ), np.median( lim_cmag_g ) + 5 * np.std( lim_cmag_g ), 65)
bins_mag_r = np.linspace( np.median( lim_cmag_r ) - 5 * np.std( lim_cmag_r ), np.median( lim_cmag_r ) + 5 * np.std( lim_cmag_r ), 65)
bins_mag_i = np.linspace( np.median( lim_cmag_i ) - 5 * np.std( lim_cmag_i ), np.median( lim_cmag_i ) + 5 * np.std( lim_cmag_i ), 65)
bins_mag_z = np.linspace( np.median( lim_cmag_z ) - 5 * np.std( lim_cmag_z ), np.median( lim_cmag_z ) + 5 * np.std( lim_cmag_z ), 65)

bins_ug = np.linspace( np.median( lim_ug ) - 5 * np.std( lim_ug ), np.median( lim_ug ) + 5 * np.std( lim_ug ), 65)
bins_gr = np.linspace( np.median( lim_gr ) - 5 * np.std( lim_gr ), np.median( lim_gr ) + 5 * np.std( lim_gr ), 65)
bins_gi = np.linspace( np.median( lim_gi ) - 5 * np.std( lim_gi ), np.median( lim_gi ) + 5 * np.std( lim_gi ), 65)
bins_ri = np.linspace( np.median( lim_ri ) - 5 * np.std( lim_ri ), np.median( lim_ri ) + 5 * np.std( lim_ri ), 65)
bins_iz = np.linspace( np.median( lim_iz ) - 5 * np.std( lim_iz ), np.median( lim_iz ) + 5 * np.std( lim_iz ), 65)

bins_z = np.linspace( np.median( lim_z ) - 5 * np.std( lim_z ), np.median( lim_z ) + 5 * np.std( lim_z ), 65 )


fig = plt.figure()
ax = fig.add_axes( [0.11, 0.11, 0.80, 0.85] )
ax.hist( lim_z, bins = bins_z, density = True, color = 'r', alpha = 0.75, label = 'RedMapper Satellites',)
ax.hist( map_z, bins = bins_z, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')

ax.annotate( s = '%s-band' % band_str, xy = (0.60, 0.85), xycoords = 'axes fraction',)

ax.legend( loc = 2, frameon = False)
ax.set_xlabel('$z_{photo}$')
plt.savefig('/home/xkchen/contrl-G_Pm-cut-sat_z_compare.png', dpi = 300)
plt.close()


fig = plt.figure( figsize = (20, 4) )
axs = gridspec.GridSpec( 1, 5, figure = fig, left = 0.02, bottom = 0.12, right = 0.99, top = 0.96, wspace = 0.12, width_ratios = [1,1,1,1,1],)

gax = fig.add_subplot( axs[0] )
gax.hist( lim_cmag_u, bins = bins_mag_u, density = True, color = 'b', alpha = 0.75, label = 'RedMapper Satellites')
gax.hist( map_cmag_u, bins = bins_mag_u, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
gax.set_xlabel('cMag_u')
gax.annotate( s = '%s-band' % band_str, xy = (0.60, 0.85), xycoords = 'axes fraction',)
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[1] )
gax.hist( lim_cmag_g, bins = bins_mag_g, density = True, color = 'g', alpha = 0.75, label = 'RedMapper Satellites')
gax.hist( map_cmag_g, bins = bins_mag_g, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
gax.set_xlabel('cMag_g')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[2] )
gax.hist( lim_cmag_r, bins = bins_mag_r, density = True, color = 'r', alpha = 0.75, label = 'RedMapper Satellites')
gax.hist( map_cmag_r, bins = bins_mag_r, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
gax.set_xlabel('cMag_r')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[3] )
gax.hist( lim_cmag_i, bins = bins_mag_i, density = True, color = 'm', alpha = 0.75, label = 'RedMapper Satellites')
gax.hist( map_cmag_i, bins = bins_mag_i, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
gax.set_xlabel('cMag_i')
gax.legend( loc = 2, frameon = False)

gax = fig.add_subplot( axs[4] )
gax.hist( lim_cmag_z, bins = bins_mag_z, density = True, color = 'c', alpha = 0.75, label = 'RedMapper Satellites')
gax.hist( map_cmag_z, bins = bins_mag_z, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
gax.set_xlabel('cMag_z')
gax.legend( loc = 2, frameon = False)

plt.savefig('/home/xkchen/SDSS_redMapper_mem_mag.png', dpi = 300)
plt.close()


fig = plt.figure( figsize = (20, 4) )
axs = gridspec.GridSpec( 1, 5, figure = fig, left = 0.02, bottom = 0.12, right = 0.99, top = 0.96, wspace = 0.12, width_ratios = [1,1,1,1,1],)

gax = fig.add_subplot( axs[0] )
gax.hist( lim_ug, bins = bins_ug, density = True, color = 'b', alpha = 0.75, label = 'RedMapper Satellites')
gax.hist( map_ug, bins = bins_ug, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
gax.set_xlabel('u - g')
gax.annotate( s = '%s-band' % band_str, xy = (0.60, 0.85), xycoords = 'axes fraction',)

gax.annotate( s = 'std:%.3f' % np.std(lim_ug) + '\n' + 'aveg:%.3f' % np.mean(lim_ug) + '\n' + 'medi:%.3f' % np.median(lim_ug), 
			xy = (0.15, 0.45), xycoords = 'axes fraction', color = 'b',)

gax.annotate( s = 'std:%.3f' % np.std(map_ug) + '\n' + 'aveg:%.3f' % np.mean(map_ug) + '\n' + 'medi:%.3f' % np.median(map_ug), 
			xy = (0.75, 0.45), xycoords = 'axes fraction', color = 'k',)

gax.legend( loc = 2, frameon = False)


gax = fig.add_subplot( axs[1] )
gax.hist( lim_gr, bins = bins_gr, density = True, color = 'g', alpha = 0.75, label = 'RedMapper Satellites')
gax.hist( map_gr, bins = bins_gr, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
gax.set_xlabel('g - r')

gax.annotate( s = 'std:%.3f' % np.std(lim_gr) + '\n' + 'aveg:%.3f' % np.mean(lim_gr) + '\n' + 'medi:%.3f' % np.median(lim_gr), 
			xy = (0.15, 0.45), xycoords = 'axes fraction', color = 'g',)

gax.annotate( s = 'std:%.3f' % np.std(map_gr) + '\n' + 'aveg:%.3f' % np.mean(map_gr) + '\n' + 'medi:%.3f' % np.median(map_gr), 
			xy = (0.75, 0.45), xycoords = 'axes fraction', color = 'k',)

gax.legend( loc = 2, frameon = False)


gax = fig.add_subplot( axs[2] )
gax.hist( lim_gi, bins = bins_gi, density = True, color = 'r', alpha = 0.75, label = 'RedMapper Satellites')
gax.hist( map_gi, bins = bins_gi, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
gax.set_xlabel('g - i')

gax.annotate( s = 'std:%.3f' % np.std(lim_gi) + '\n' + 'aveg:%.3f' % np.mean(lim_gi) + '\n' + 'medi:%.3f' % np.median(lim_gi), 
			xy = (0.15, 0.45), xycoords = 'axes fraction', color = 'r',)

gax.annotate( s = 'std:%.3f' % np.std(map_gi) + '\n' + 'aveg:%.3f' % np.mean(map_gi) + '\n' + 'medi:%.3f' % np.median(map_gi), 
			xy = (0.75, 0.45), xycoords = 'axes fraction', color = 'k',)

gax.legend( loc = 2, frameon = False)


gax = fig.add_subplot( axs[3] )
gax.hist( lim_ri, bins = bins_ri, density = True, color = 'm', alpha = 0.75, label = 'RedMapper Satellites')
gax.hist( map_ri, bins = bins_ri, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
gax.set_xlabel('r - i')

gax.annotate( s = 'std:%.3f' % np.std(lim_ri) + '\n' + 'aveg:%.3f' % np.mean(lim_ri) + '\n' + 'medi:%.3f' % np.median(lim_ri), 
			xy = (0.15, 0.45), xycoords = 'axes fraction', color = 'm',)

gax.annotate( s = 'std:%.3f' % np.std(map_ri) + '\n' + 'aveg:%.3f' % np.mean(map_ri) + '\n' + 'medi:%.3f' % np.median(map_ri), 
			xy = (0.75, 0.45), xycoords = 'axes fraction', color = 'k',)

gax.legend( loc = 2, frameon = False)


gax = fig.add_subplot( axs[4] )
gax.hist( lim_iz, bins = bins_iz, density = True, color = 'c', alpha = 0.75, label = 'RedMapper Satellites')
gax.hist( map_iz, bins = bins_iz, density = True, color = 'k', histtype = 'step', ls = '--', alpha = 0.75, label = 'Control')
gax.set_xlabel('i - z')

gax.annotate( s = 'std:%.3f' % np.std(lim_iz) + '\n' + 'aveg:%.3f' % np.mean(lim_iz) + '\n' + 'medi:%.3f' % np.median(lim_iz), 
			xy = (0.15, 0.45), xycoords = 'axes fraction', color = 'c',)

gax.annotate( s = 'std:%.3f' % np.std(map_iz) + '\n' + 'aveg:%.3f' % np.mean(map_iz) + '\n' + 'medi:%.3f' % np.median(map_iz), 
			xy = (0.75, 0.45), xycoords = 'axes fraction', color = 'k',)

gax.legend( loc = 2, frameon = False)

plt.savefig('/home/xkchen/SDSS_redMapper_mem_color.png', dpi = 300)
plt.close()

