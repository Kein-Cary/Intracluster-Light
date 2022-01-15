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
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']


### === figs
def sat_cut_img():

	from img_sat_extract import sate_Extract_func
	from img_sat_extract import sate_surround_mask_func

	home = '/home/xkchen/data/SDSS/'
	load = '/home/xkchen/fig_tmp/'

	dat = pds.read_csv( home + 'member_files/sat_cat_z02_03/Extend-BCGM_rgi-common_cat.csv')
	ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
	clus_ID = np.array( dat['clust_ID'])

	Ns = len( ra )

	dat = pds.read_csv( home + 'member_files/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-limit_member-cat.csv')
	s_ra, s_dec, s_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z_spec'] )
	s_host_ID = np.array( dat['clus_ID'] )
	s_host_ID = s_host_ID.astype( int )


	for tt in range( 1 ):

		band_str = band[ tt ]

		for kk in range( 1 ):

			ra_g, dec_g, z_g = sub_ra[ kk ], sub_dec[ kk ], sub_z[ kk]

			kk_ID = sub_clusID[ kk ]

			id_vx = s_host_ID == kk_ID
			lim_ra, lim_dec, lim_z = s_ra[ id_vx ], s_dec[ id_vx ], s_z[ id_vx ]

			# R_cut = 2.5  ## scaled case
			R_cut = 320

			d_file = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
			gal_file = home + 'photo_files/detect_source_cat/photo-z_img_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
			offset_file = home + 'photo_files/pos_offset_correct_imgs/offset/%s-band_ra%.3f_dec%.3f_z%.3f_star-pos-offset.csv'

			##... pre cutout
			out_file = '/home/xkchen/figs/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_img.fits'
			sate_Extract_func( d_file, ra_g, dec_g, z_g, lim_ra, lim_dec, lim_z, band_str, gal_file, out_file, R_cut, offset_file = offset_file)

			##... image mask
			cat_file = home + 'photo_files/star_cats/source_SQL_Z%.3f_ra%.3f_dec%.3f.csv'
			out_mask_file = '/home/xkchen/figs/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img.fits'

			if band_str == 'r':
				extra_cat = [ home + 'photo_files/detect_source_cat/photo-z_img_g-band_mask_ra%.3f_dec%.3f_z%.3f.cat', 
							  home + 'photo_files/detect_source_cat/photo-z_img_i-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

				extra_img = [ home + 'photo_data/frame-g-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
							  home + 'photo_data/frame-i-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

			if band_str == 'g':
				extra_cat = [ home + 'photo_files/detect_source_cat/photo-z_img_r-band_mask_ra%.3f_dec%.3f_z%.3f.cat', 
							  home + 'photo_files/detect_source_cat/photo-z_img_i-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

				extra_img = [ home + 'photo_data/frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
							  home + 'photo_data/frame-i-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

			if band_str == 'i':
				extra_cat = [ home + 'photo_files/detect_source_cat/photo-z_img_r-band_mask_ra%.3f_dec%.3f_z%.3f.cat',
							  home + 'photo_files/detect_source_cat/photo-z_img_g-band_mask_ra%.3f_dec%.3f_z%.3f.cat']

				extra_img = [ home + 'photo_data/frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2',
							  home + 'photo_data/frame-g-ra%.3f-dec%.3f-redshift%.3f.fits.bz2']

			tt2 = time.time()

			sate_surround_mask_func(d_file, cat_file, ra_g, dec_g, z_g, lim_ra, lim_dec, lim_z, band_str, gal_file, out_mask_file, R_cut, 
									offset_file = offset_file, extra_cat = extra_cat, extra_img = extra_img, stack_info = stack_cat )

			print( time.time() - tt2 )

	return

def sat_cut_test():

	home = '/home/xkchen/data/SDSS/'
	load = '/home/xkchen/figs/'

	#. cluster cat
	dat = pds.read_csv( home + 'member_files/sat_cat_z02_03/Extend-BCGM_rgi-common_cat.csv')
	ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
	clus_ID = np.array( dat['clust_ID'])

	N_clus = len( ra )


	dat = pds.read_csv( home + 'member_files/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-limit_member-cat.csv')
	s_ra, s_dec, s_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z_spec'] )
	s_host_ID = np.array( dat['clus_ID'] )
	s_host_ID = s_host_ID.astype( int )


	### ... image cut out
	band_str = band[0]

	d_file = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
	out_file = '/home/xkchen/figs/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_img.fits'
	out_mask_file = '/home/xkchen/figs/Sat-tract_%s-band_clus_ra%.3f_dec%.3f_z%.3f_sat_ra%.4f_dec%.4f_mask-img.fits'

	for kk in range( 1 ):

		ra_g, dec_g, z_g = ra[ kk ], dec[ kk ], z[ kk]

		kk_ID = clus_ID[ kk ]
		id_vx = s_host_ID == kk_ID
		lim_ra, lim_dec, lim_z = s_ra[ id_vx ], s_dec[ id_vx ], s_z[ id_vx ]

		img_data = fits.open( d_file % (band_str, ra_g, dec_g, z_g), )
		img_arr = img_data[0].data
		Header = img_data[0].header

		wcs_lis = awc.WCS( Header )

		pos_x, pos_y = wcs_lis.all_world2pix( lim_ra, lim_dec, 0 )
		cen_x, cen_y = wcs_lis.all_world2pix( ra_g, dec_g, 0 )

		N_sat = len( lim_ra )


		#. view on source detection
		source = asc.read( home + 
					'photo_files/detect_source_cat/photo-z_img_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat' % (band_str, ra_g, dec_g, z_g),)
		Numb = np.array(source['NUMBER'][-1])
		A = np.array(source['A_IMAGE'])
		B = np.array(source['B_IMAGE'])
		theta = np.array(source['THETA_IMAGE'])
		
		cx = np.array(source['X_IMAGE'])
		cy = np.array(source['Y_IMAGE'])

		pk_cx = np.array(source['XPEAK_IMAGE'])
		pk_cy = np.array(source['YPEAK_IMAGE'])

		a = 10 * A
		b = 10 * B


		id_xm = A == np.max(A)
		t_cx, t_cy = cx[ id_xm ], cy[ id_xm ]
		t_a, t_b = a[ id_xm ], b[ id_xm ]
		t_chi = theta[ id_xm ]


		plt.figure()
		ax = plt.subplot(111)

		ax.imshow( img_arr, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm(),)
		ax.scatter( pos_x, pos_y, s = 20, edgecolors = 'r', facecolors = 'none',)
		ax.scatter( cen_x, cen_y, s = 20, marker = 's', edgecolors = 'b', facecolors = 'none',)

		# for ll in range( Numb ):
		# 	ellips = Ellipse( xy = (cx[ll], cy[ll]), width = a[ll], height = b[ll], angle = theta[ll], fill = False, 
		# 		ec = 'm', ls = '--', linewidth = 0.75, )
		# 	ax.add_patch( ellips )

		ellips = Ellipse( xy = (t_cx, t_cy), width = t_a, height = t_b, angle = t_chi, fill = False, 
			ec = 'm', ls = '-', linewidth = 0.75, )
		ax.add_patch( ellips )

		ax.set_xlim( 0, 2048 )
		ax.set_ylim( 0, 1489 )

		plt.savefig('/home/xkchen/figs/cluster_%s-band_ra%.3f_dec%.3f_z%.3f.png' % (band_str, ra_g, dec_g, z_g), dpi = 300)
		plt.close()


		for pp in range( N_sat ):

			kk_ra, kk_dec = lim_ra[ pp ], lim_dec[ pp ] 

			cut_img = fits.open( out_file % (band_str, ra_g, dec_g, z_g, kk_ra, kk_dec),)
			cut_img_arr = cut_img[0].data
			kk_px, kk_py = cut_img[0].header['CENTER_X'], cut_img[0].header['CENTER_Y']
			_pkx, _pky = cut_img[0].header['PEAK_X'], cut_img[0].header['PEAK_Y']

			cut_mask = fits.open( out_mask_file % (band_str, ra_g, dec_g, z_g, kk_ra, kk_dec),)
			cut_mask_arr = cut_mask[0].data
			cp_px, cp_py = cut_mask[0].header['CENTER_X'], cut_mask[0].header['CENTER_Y']

			fig = plt.figure( figsize = (13.12, 4.8) )
			ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
			ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

			ax0.imshow( cut_img_arr, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm(),)
			ax0.scatter( kk_px, kk_py, s = 20, marker = 's', edgecolors = 'b', facecolors = 'none',)
			ax0.scatter( _pkx, _pky, s = 20, marker = 'o', edgecolors = 'r', facecolors = 'none',)

			ax0.set_xlim( kk_px - 75, kk_px + 75 )
			ax0.set_ylim( kk_py - 75, kk_py + 75 )

			ax1.imshow( cut_mask_arr, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm(),)
			ax1.scatter( kk_px, kk_py, s = 20, marker = 's', edgecolors = 'b', facecolors = 'none',)
			ax1.scatter( cp_px, cp_py, s = 20, marker = 'o', edgecolors = 'r', facecolors = 'none',)

			ax1.set_xlim( kk_px - 75, kk_px + 75 )
			ax1.set_ylim( kk_py - 75, kk_py + 75 )

			plt.savefig('/home/xkchen/figs/' + 
					'cluster_%s-band_ra%.3f_dec%.3f_z%.3f_sat_ra%.3f_dec%.3f.png' % (band_str, ra_g, dec_g, z_g, kk_ra, kk_dec), dpi = 300)
			plt.close()

# sat_cut_img()
# sat_cut_test()


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

"""
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

"""

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


raise


"""
### === sample properties

##. all member
# s_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_member-cat.csv' )

##. frame limited catalog
# s_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-limit_member-cat.csv')

##. frame limited + P_mem cut catalog
# s_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-limit_Pm-cut_member-cat.csv')

##. frame limited + P_mem cut catalog + exclude BCGs
s_dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv')


bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
p_ra, p_dec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] )

p_cID = np.array( s_dat['clus_ID'] )
p_gr = np.array( s_dat['g-r'] )
p_z = np.array( s_dat['z_spec'] )

p_Rsat = np.array( s_dat['R_cen'] )
p_R2Rv = np.array( s_dat['Rcen/Rv'] )


R_cut = 0.191 # np.median( p_Rcen )
id_vx = p_R2Rv <= R_cut


##. satellite properties
data = fits.open('/home/xkchen/figs/extend_Zphoto_cat/zphot_01_033_cat/redMaPPer_z-phot_0.1-0.33_member_params.fit')
l_ra, l_dec = data[1].data['ra'], data[1].data['dec']
l_z = data[1].data['z']
Mag_r = data[1].data['absMagR']


p_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )
l_coord = SkyCoord( ra = l_ra * U.deg, dec = l_dec * U.deg )

idx, sep, d3d = p_coord.match_to_catalog_sky( l_coord )
id_lim = sep.value < 2.7e-4

mp_zll = l_z[ idx[ id_lim ] ]
mp_Magr = Mag_r[ idx[ id_lim ] ] 

id_nul = mp_zll < 0.
z_devi = mp_zll[ id_nul == False ] - bcg_z[ id_nul == False ]


plt.figure()
plt.text( 0.2, 1.0, s = '$P_{mem} \\geq 0.8 $')
plt.plot( p_Rsat, p_R2Rv, 'k.', alpha = 0.5,)
plt.plot( p_Rsat, p_Rsat, 'r-', alpha = 0.75,)

plt.axvline( x = 0.213, ls = ':', color = 'b', alpha = 0.5,)
plt.axhline( y = 0.191, ls = '--', color = 'b', alpha = 0.5,)

plt.xlabel( '$R_{Sat} \; [Mpc / h]$' )
plt.ylabel( '$R_{Sat} / R_{200m}$' )
plt.savefig('/home/xkchen/R-phy_R-scaled.png', dpi = 300)
plt.close()


plt.figure()
plt.hist( p_R2Rv, bins = 55, density = True, histtype = 'step',)
plt.axvline( x = np.median(p_R2Rv), ls = '--', label = 'Median', ymin = 0, ymax = 0.35,)
plt.axvline( x = np.mean( p_R2Rv), ls = '-', label = 'Mean', ymin = 0, ymax = 0.35,)

plt.axvline( x = 0.191, ls = ':', label = 'Division', color = 'r',)

plt.legend( loc = 1)
plt.xlabel('$R_{Sat} \, / \, R_{200m}$')
plt.savefig('/home/xkchen/redMap_sate_Rcen_hist.png', dpi = 300)
plt.close()


plt.figure()
plt.hist( z_devi, bins = np.linspace(-0.2, 0.2, 55), density = True, histtype = 'step', color = 'k',)

plt.axvline( x = np.median( z_devi ), ls = '--', label = 'Median', color = 'k',)
plt.axvline( x = np.mean( z_devi ), ls = '-', label = 'Mean', color = 'k',)

plt.axvline( x = np.median( z_devi ) - np.std( z_devi ), ls = ':', color = 'k',)
plt.axvline( x = np.median( z_devi ) + np.std( z_devi ), ls = ':', color = 'k',)

plt.legend( loc = 1)
plt.xlabel('$z_{mem} - z_{BCG}$')
plt.savefig('/home/xkchen/z_diffi.png', dpi = 300)
plt.close()


plt.figure()
plt.hist( mp_Magr[ id_vx ], bins = np.linspace(-26, -16, 65), density = True, histtype = 'step', color = 'b', label = 'Inner')

plt.axvline( x = np.median( mp_Magr[ id_vx ] ), ls = '--', label = 'Median', color = 'b',)
plt.axvline( x = np.mean( mp_Magr[ id_vx ] ), ls = '-', label = 'Mean', color = 'b',)

plt.hist( mp_Magr[ id_vx == False ], bins = np.linspace(-26, -16, 65), density = True, histtype = 'step', color = 'r', label = 'Outer')

plt.axvline( x = np.median( mp_Magr[ id_vx == False ] ), ls = '--', color = 'r',)
plt.axvline( x = np.mean( mp_Magr[ id_vx == False ] ), ls = '-', color = 'r',)

plt.legend( loc = 1)
plt.xlabel('$Mag_{r}$')
plt.savefig('/home/xkchen/abs_r_mag_compare.png', dpi = 300)
plt.close()

"""
