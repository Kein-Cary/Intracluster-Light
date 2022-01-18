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


from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


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


### === match the mophology properties and compare to SDSS
def member_size_match_func( set_ra, set_dec, set_z, sat_ra, sat_dec, band_str, img_file, obj_file, out_file ):

	#. image information
	img_data = fits.open( img_file % ( band_str, set_ra, set_dec, set_z ),)
	Header = img_data[0].header
	wcs_lis = awc.WCS( Header )

	#. obj_cat
	source = asc.read( obj_file % ( band_str, set_ra, set_dec, set_z),)
	Numb = np.array(source['NUMBER'][-1])
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])

	theta = np.array(source['THETA_IMAGE'])

	cx = np.array(source['X_IMAGE'])
	cy = np.array(source['Y_IMAGE'])
	p_type = np.array(source['CLASS_STAR'])

	peak_x = np.array( source['XPEAK_IMAGE'])
	peak_y = np.array( source['YPEAK_IMAGE'])

	Kron = 16 # 8-R_kron

	a = Kron * A
	b = Kron * B

	#. satellite locations
	N_z = len( sat_ra )

	s_cx, s_cy = wcs_lis.all_world2pix( sat_ra, sat_dec, 0)

	tmp_R_a, tmp_R_b, tmp_PA = [], [], []

	for tt in range( N_z ):

		pp_cx, pp_cy = s_cx[ tt ], s_cy[ tt ]

		#. find target galaxy in source catalog
		d_cen_R = np.sqrt( (cx - pp_cx)**2 + (cy - pp_cy)**2 )

		id_xcen = d_cen_R == d_cen_R.min()
		id_order = np.where( id_xcen )[0][0]

		kk_px, kk_py = cx[ id_order ], cy[ id_order ]

		cen_ar = A[ id_order ] * 8
		cen_br = B[ id_order ] * 8

		cen_chi = theta[ id_order ]

		tmp_R_a.append( cen_ar )
		tmp_R_b.append( cen_br )
		tmp_PA.append( cen_chi )

	#.
	keys = ['ra', 'dec', 'a', 'b', 'phi']
	values = [ sat_ra, sat_dec, np.array(tmp_R_a), np.array(tmp_R_b), np.array(tmp_PA) ]
	fill = dict( zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_file % (band_str, set_ra, set_dec, set_z),)

	return

def member_size_record():
	home = '/home/xkchen/data/SDSS/'

	c_dat = pds.read_csv( home + 'member_files/sat_cat_z02_03/Extend-BCGM_rgi-common_cat.csv')
	bcg_ra, bcg_dec, bcg_z = np.array( c_dat['ra'] ), np.array( c_dat['dec'] ), np.array( c_dat['z'] )
	clus_IDs = np.array( c_dat['clust_ID'] )

	s_dat = pds.read_csv( home + 'member_files/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv')
	sat_ra, sat_dec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] )
	p_host_ID = np.array( s_dat['clus_ID'] )


	N_ss = len( bcg_ra )

	#. match

	m, n = divmod( N_ss, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	sub_ra, sub_dec, sub_z = bcg_ra[N_sub0 : N_sub1], bcg_dec[N_sub0 : N_sub1], bcg_z[N_sub0 : N_sub1]

	sub_clusID = clus_IDs[N_sub0 : N_sub1]

	N_sub = len( sub_ra )

	for pp in range( 3 ):

		band_str = band[ pp ]

		for ll in range( N_sub ):

			ra_g, dec_g, z_g = sub_ra[ ll ], sub_dec[ ll ], sub_z[ ll ]
			ID_tag = sub_clusID[ ll ]

			id_vx = p_host_ID == ID_tag
			lim_ra, lim_dec = sat_ra[ id_vx ], sat_dec[ id_vx ]


			img_file = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
			gal_file = home + 'photo_files/detect_source_cat/photo-z_img_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
			out_file = '/home/xkchen/project/tmp_obj_cat/clus_%s-band_ra%.3f_dec%.3f_z%.3f_mem-size_cat.csv'

			member_size_match_func( ra_g, dec_g, z_g, lim_ra, lim_dec, band_str, img_file, gal_file, out_file )

	print('done!')
	commd.Barrier()


	#. combine all satellites
	if rank == 0:

		for tt in range( 3 ):

			band_str = band[ tt ]

			tmp_s_ra, tmp_s_dec = np.array( [] ), np.array( [] )
			tmp_s_a, tmp_s_b, tmp_s_phi = np.array( [] ), np.array( [] ), np.array( [] )
			cp_bcg_ra, cp_bcg_dec, cp_bcg_z = np.array( [] ), np.array( [] ), np.array( [] )

			for pp in range( N_ss ):

				ra_g, dec_g, z_g = bcg_ra[ pp ], bcg_dec[ pp ], bcg_z[ pp ]

				dat = pds.read_csv('/home/xkchen/project/tmp_obj_cat/' + 
								'clus_%s-band_ra%.3f_dec%.3f_z%.3f_mem-size_cat.csv' % (band_str, ra_g, dec_g, z_g),)

				kk_ra, kk_dec = np.array( dat['ra'] ), np.array( dat['dec'] )
				kk_a, kk_b, kk_phi = np.array( dat['a'] ), np.array( dat['b'] ), np.array( dat['phi'] )

				tmp_s_ra = np.r_[ tmp_s_ra, kk_ra ]
				tmp_s_dec = np.r_[ tmp_s_dec, kk_dec ]

				tmp_s_a = np.r_[ tmp_s_a, kk_a ]
				tmp_s_b = np.r_[ tmp_s_b, kk_b ]
				tmp_s_phi = np.r_[ tmp_s_phi, kk_phi ]

				cp_bcg_ra = np.r_[ cp_bcg_ra, np.ones( len(kk_ra), ) * ra_g ]
				cp_bcg_dec = np.r_[ cp_bcg_dec, np.ones( len(kk_ra), ) * dec_g ]
				cp_bcg_z = np.r_[ cp_bcg_z, np.ones( len(kk_ra), ) * z_g ]

			keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'a', 'b', 'phi']
			values = [ cp_bcg_ra, cp_bcg_dec, cp_bcg_z, tmp_s_ra, tmp_s_dec, tmp_s_a, tmp_s_b, tmp_s_phi ]
			fill = dict( zip( keys, values ) )
			data = pds.DataFrame( fill )
			data.to_csv('/home/xkchen/Extend-BCGM_rgi-common_cat_%s-band_sat-size.csv' % band_str,)

	return

# member_size_record()

def member_size_compare():

	dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_size/Extend-BCGM_rgi-common_cat_r-band_sat-size.csv')

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
	sat_a, sat_b = np.array( dat['a'] ), np.array( dat['b'] )

	sat_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )

	#. SDSS cat. table
	cat = fits.open('/home/xkchen/figs/extend_Zphoto_cat/zphot_01_033_cat/redMaPPer_z-phot_0.1-0.33_member_params.fit')
	cat_arr = cat[1].data

	ref_ra, ref_dec = cat_arr['ra'], cat_arr['dec']

	deV_R_r = cat_arr['deVRad_r'] 
	exp_R_r = cat_arr['expRad_r']

	petr_R_r = cat_arr['petroRad_r']
	petr_R90_r = cat_arr['petroR90_r']
	petr_R50_r = cat_arr['petroR50_r']

	ref_coord = SkyCoord( ra = ref_ra * U.deg, dec = ref_dec * U.deg )

	idx, sep, d3d = sat_coord.match_to_catalog_sky( ref_coord )
	id_lim = sep.value < 2.7e-4


	mp_deV_R_r = deV_R_r[ idx[ id_lim ] ]
	mp_exp_R_r = exp_R_r[ idx[ id_lim ] ]

	mp_petr_R_r = petr_R_r[ idx[ id_lim ] ]
	mp_petr_R90_r = petr_R90_r[ idx[ id_lim ] ]
	mp_petr_R50_r = petr_R50_r[ idx[ id_lim ] ]

	lim_a, lim_b = sat_a[ id_lim ] * pixel, sat_b[ id_lim ] * pixel

	R_arr = [ mp_deV_R_r, mp_exp_R_r, mp_petr_R_r, mp_petr_R90_r, mp_petr_R50_r ]
	line_s = ['$R_{deV}$', '$R_{exp}$', '$R_{petro}$', '$R_{petro\,90}$', '$R_{petro\,50}$' ]


	bins_x = np.linspace( 0, 20, 100)
	bins_dx = np.linspace( -20, 10, 100)

	fig = plt.figure( figsize = (12.8, 4.8) )
	ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
	ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

	ax0.hist( lim_a, bins = bins_x, density = True, histtype = 'step', color = 'k', alpha = 0.5, label = '$a_{mask}$')
	ax0.axvline( x = np.median( lim_a ), ls = '--', color = 'k', alpha = 0.5, ymin = 0.75, ymax = 1, label = 'Median')
	ax0.axvline( x = np.mean( lim_a ), ls = '-', color = 'k', alpha = 0.5, ymin = 0.75, ymax = 1, label = 'Mean')

	for pp in range( len(line_s) ):

		ax0.hist( R_arr[ pp ], bins = bins_x, density = True, histtype = 'step', color = mpl.cm.rainbow( pp / len(line_s) ), 
				alpha = 0.5, label = line_s[ pp ],)

		ax0.axvline( x = np.median( R_arr[pp] ), ls = '--', color = mpl.cm.rainbow( pp / len(line_s) ), alpha = 0.5, ymin = 0, ymax = 0.25,)
		ax0.axvline( x = np.mean( R_arr[pp]), ls = '-', color = mpl.cm.rainbow( pp / len(line_s) ), alpha = 0.5, ymin = 0, ymax = 0.25,)


		ax1.hist( R_arr[ pp ] - lim_a, bins = bins_dx, density = True, histtype = 'step', color = mpl.cm.rainbow( pp / len(line_s) ), 
				alpha = 0.5, label = line_s[ pp ] + ' ${-}$ ' + '$a_{mask}$', )
		ax1.axvline( x = np.median( R_arr[pp] - lim_a), ls = '--', color = mpl.cm.rainbow( pp / len(line_s) ), alpha = 0.5, ymin = 0, ymax = 0.25,)
		ax1.axvline( x = np.mean( R_arr[pp] - lim_a), ls = '-', color = mpl.cm.rainbow( pp / len(line_s) ), alpha = 0.5, ymin = 0, ymax = 0.25,)

	ax0.set_xlabel('$R \; [arcsec]$')
	ax1.set_xlabel('$\\Delta R \; [arcsec]$')
	ax0.legend( loc = 1)
	ax1.legend( loc = 1)

	plt.savefig('/home/xkchen/mask_a-size_compare.png', dpi = 300)
	plt.close()


	fig = plt.figure( figsize = (13.12, 4.8) )
	ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
	ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

	ax0.hist( lim_b, bins = bins_x, density = True, histtype = 'step', color = 'k', alpha = 0.5, label = '$b_{mask}$')
	ax0.axvline( x = np.median( lim_b ), ls = '--', color = 'k', alpha = 0.5, ymin = 0.75, ymax = 1, label = 'Median')
	ax0.axvline( x = np.mean( lim_b ), ls = '-', color = 'k', alpha = 0.5, ymin = 0.75, ymax = 1, label = 'Mean')

	for pp in range( len(line_s) ):

		ax0.hist( R_arr[ pp ], bins = bins_x, density = True, histtype = 'step', color = mpl.cm.rainbow( pp / len(line_s) ), 
				alpha = 0.5, label = line_s[ pp ],)

		ax0.axvline( x = np.median( R_arr[pp] ), ls = '--', color = mpl.cm.rainbow( pp / len(line_s) ), alpha = 0.5, ymin = 0, ymax = 0.25,)
		ax0.axvline( x = np.mean( R_arr[pp]), ls = '-', color = mpl.cm.rainbow( pp / len(line_s) ), alpha = 0.5, ymin = 0, ymax = 0.25,)


		ax1.hist( R_arr[ pp ] - lim_b, bins = bins_dx, density = True, histtype = 'step', color = mpl.cm.rainbow( pp / len(line_s) ), 
				alpha = 0.5, label = line_s[ pp ] + ' ${-}$ ' + '$b_{mask}$', )
		ax1.axvline( x = np.median( R_arr[pp] - lim_b), ls = '--', color = mpl.cm.rainbow( pp / len(line_s) ), alpha = 0.5, ymin = 0, ymax = 0.25,)
		ax1.axvline( x = np.mean( R_arr[pp] - lim_b), ls = '-', color = mpl.cm.rainbow( pp / len(line_s) ), alpha = 0.5, ymin = 0, ymax = 0.25,)

	ax0.set_xlabel('$R \; [arcsec]$')
	ax1.set_xlabel('$\\Delta R \; [arcsec]$')
	ax0.legend( loc = 1)
	ax1.legend( loc = 1)

	plt.savefig('/home/xkchen/mask_b-size_compare.png', dpi = 300)
	plt.close()

	return

member_size_compare()

