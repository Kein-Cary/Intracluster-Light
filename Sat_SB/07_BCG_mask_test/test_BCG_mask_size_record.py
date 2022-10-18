import matplotlib as mpl
import matplotlib.pyplot as plt

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

##.
from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2asec = U.rad.to( U.arcsec )

pixel = 0.396
z_ref = 0.25

band = ['r', 'g', 'i']


### === data load
home = '/home/xkchen/data/SDSS/'

d_file = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
gal_file = home + 'photo_files/detect_source_cat/photo-z_img_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'


bin_rich = [ 20, 30, 50, 210 ]

sub_name = ['low-rich', 'medi-rich', 'high-rich']
line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']


"""
for ll in range( 3 ):

	##. cluster catalog
	dat = pds.read_csv( '/home/xkchen/fig_tmp/Extend_Mbcg_richbin_sat_cat/' + 
						'clust_rich_%d-%d_cat.csv' % (bin_rich[ll], bin_rich[ll + 1]),)

	bcg_ra, bcg_dec, bcg_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

	N_clus = len( bcg_ra )

	for kk in range( rank, rank + 1 ):

		band_str = band[ kk ]

		bcg_mas_ax, bcg_mas_bx = np.zeros( N_clus, ), np.zeros( N_clus, )

		for dd in range( N_clus ):

			ra_g, dec_g, z_g = bcg_ra[ dd ], bcg_dec[ dd ], bcg_z[ dd ]  

			data = fits.open( d_file % (band_str, ra_g, dec_g, z_g),)
			
			Head = data[0].header
			wcs_lis = awc.WCS( Head )

			cen_x, cen_y = wcs_lis.all_world2pix( ra_g, dec_g, 1 )

			##. detected source
			source = asc.read( gal_file % (band_str, ra_g, dec_g, z_g), )
			Numb = np.array(source['NUMBER'][-1])
			A = np.array(source['A_IMAGE'])
			B = np.array(source['B_IMAGE'])
			theta = np.array(source['THETA_IMAGE'])
			cx = np.array(source['X_IMAGE'])
			cy = np.array(source['Y_IMAGE'])

			##. taget the BCG location
			dR = np.sqrt( (cx - cen_x)**2 + (cy - cen_y)**2 )

			id_tag = dR == np.min( dR )

			A_x, B_x = A[ id_tag ][0], B[ id_tag ][0]

			bcg_mas_ax[ dd ] = A_x
			bcg_mas_bx[ dd ] = B_x

		##. save
		keys = [ 'ra', 'dec', 'z', 'major_pix', 'minor_pix' ]
		values = [ bcg_ra, bcg_dec, bcg_z, bcg_mas_ax, bcg_mas_bx ]
		fill = dict( zip( keys, values ) )
		data = pds.DataFrame( fill )
		data.to_csv('/home/xkchen/' + 
			'clust_rich_%d-%d_%s-band_bcg-detected-pixels.csv' % (bin_rich[ll], bin_rich[ll + 1], band_str),)

"""

### === 
for ll in range( 3 ):

	#.
	cat_0 = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/bcg_mask_size/' + 
						'clust_rich_%d-%d_r-band_bcg-detected-pixels.csv' % (bin_rich[ll], bin_rich[ll + 1]),)
	ra, dec, z = np.array( cat_0['ra'] ), np.array( cat_0['dec'] ), np.array( cat_0['z'] )
	Ax_0, Bx_0 = np.array( cat_0['major_pix'] ), np.array( cat_0['minor_pix'] )

	#.
	cat_1 = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/bcg_mask_size/' + 
						'clust_rich_%d-%d_g-band_bcg-detected-pixels.csv' % (bin_rich[ll], bin_rich[ll + 1]),)
	Ax_1, Bx_1 = np.array( cat_1['major_pix'] ), np.array( cat_1['minor_pix'] )

	#.
	cat_2 = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/bcg_mask_size/' + 
						'clust_rich_%d-%d_i-band_bcg-detected-pixels.csv' % (bin_rich[ll], bin_rich[ll + 1]),)
	Ax_2, Bx_2 = np.array( cat_2['major_pix'] ), np.array( cat_2['minor_pix'] )


	Da_g = Test_model.angular_diameter_distance( z ).value

	simi_ax_0 = Ax_0 * 8 * pixel * Da_g * 1e3 / rad2asec
	simi_bx_0 = Bx_0 * 8 * pixel * Da_g * 1e3 / rad2asec

	simi_ax_1 = Ax_1 * 8 * pixel * Da_g * 1e3 / rad2asec
	simi_bx_1 = Bx_1 * 8 * pixel * Da_g * 1e3 / rad2asec

	simi_ax_2 = Ax_2 * 8 * pixel * Da_g * 1e3 / rad2asec
	simi_bx_2 = Bx_2 * 8 * pixel * Da_g * 1e3 / rad2asec


	bins_R = np.logspace(0.6, 2.48, 55)

	fig = plt.figure()
	ax = fig.add_axes( [0.12, 0.10, 0.80, 0.85] )

	ax.hist( simi_ax_0, bins = bins_R, ls = '-', color = 'r', alpha = 0.5, histtype = 'step', label = 'a, r-band',)
	ax.hist( simi_bx_0, bins = bins_R, ls = '--', color = 'r', alpha = 0.75, histtype = 'step', label = 'b, r-band',)

	ax.hist( simi_ax_1, bins = bins_R, ls = '-', color = 'g', alpha = 0.5, histtype = 'step', label = 'g-band',)
	ax.hist( simi_bx_1, bins = bins_R, ls = '--', color = 'g', alpha = 0.75, histtype = 'step',)

	ax.hist( simi_ax_2, bins = bins_R, ls = '-', color = 'b', alpha = 0.5, histtype = 'step', label = 'i-band')
	ax.hist( simi_bx_2, bins = bins_R, ls = '--', color = 'b', alpha = 0.75, histtype = 'step',)

	ax.annotate( s = line_name[ll], xy = (0.05, 0.35), xycoords = 'axes fraction',)
	ax.legend( loc = 1, frameon = False, fontsize = 12,)

	ax.set_xlim( 5, 300 )
	ax.set_xscale('log')
	ax.set_xlabel('$ R \; [kpc] $',)

	plt.savefig('/home/xkchen/clus_rich-%d-%d_BCG_mask_size_hist.png' % (bin_rich[ll], bin_rich[ll + 1]), dpi = 300)
	plt.close()

