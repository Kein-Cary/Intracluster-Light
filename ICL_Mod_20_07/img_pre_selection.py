import glob
import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

import scipy.stats as sts
import astropy.units as U
import subprocess as subpro
import astropy.constants as C

from scipy.optimize import curve_fit
from scipy.stats import binned_statistic as binned
from astropy import cosmology as apcy

from fig_out_module import cc_grid_img

## cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

## constant
rad2asec = U.rad.to(U.arcsec)

#**************************#
def WCS_to_pixel_func(ra, dec, header_inf):
	"""
	according to SDSS Early Data Release paper (section 4.2.2 wcs)
	"""
	Ra0 = header_inf['CRVAL1']
	Dec0 = header_inf['CRVAL2']

	row_0 = header_inf['CRPIX2']
	col_0 = header_inf['CRPIX1']

	af = header_inf['CD1_1']
	bf = header_inf['CD1_2']

	cf = header_inf['CD2_1']
	df = header_inf['CD2_2']

	y1 = (ra - Ra0) * np.cos( Dec0 * np.pi / 180 )
	y2 = dec - Dec0

	delt_col = (bf * y2 - df * y1) / ( bf * cf - af * df )
	delt_row = (af * y2 - cf * y1) / ( af * df - bf * cf )

	id_col = col_0 + delt_col
	id_row = row_0 + delt_row

	return id_col, id_row

def pixel_to_WCS_func(x, y, header_inf):

	Ra0 = header_inf['CRVAL1']
	Dec0 = header_inf['CRVAL2']

	row_0 = header_inf['CRPIX2']
	col_0 = header_inf['CRPIX1']

	af = header_inf['CD1_1']
	bf = header_inf['CD1_2']

	cf = header_inf['CD2_1']
	df = header_inf['CD2_2']

	_delta = bf * cf - af * df

	delta_x = x - col_0
	delta_y = y - row_0

	delta_ra = _delta * ( delta_x * af + delta_y * bf ) / _delta
	delta_dec = _delta * ( delta_x * cf + delta_y * df ) / _delta

	dec = Dec0 + delta_dec
	ra = Ra0 + delta_ra / np.cos( Dec0 * np.pi / 180 )

	return ra, dec

def zref_BCG_pos_func(cat_file, z_ref, out_file, pix_size,):

	dat = pds.read_csv( cat_file )
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	Da_z = Test_model.angular_diameter_distance(z).value
	Da_ref = Test_model.angular_diameter_distance(z_ref).value

	L_ref = Da_ref * pix_size / rad2asec
	L_z = Da_z * pix_size / rad2asec
	eta = L_ref / L_z

	ref_bcgx = clus_x / eta
	ref_bcgy = clus_y / eta

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
	values = [ra, dec, z, ref_bcgx, ref_bcgy]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_file )

	return

def get_mu_sigma(cat_file, ref_cat, out_put, ):

	dat = pds.read_csv( cat_file )
	ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
	tt_ra = ['%.5f' % ll for ll in ra]
	tt_dec = ['%.5f' % ll for ll in dec]

	samp_dat = pds.read_csv( ref_cat )
	tmp_ra, tmp_dec, tmp_z = np.array( samp_dat['ra'] ), np.array( samp_dat['dec'] ), np.array( samp_dat['z'] )
	tmp_mu, tmp_sigm = np.array(samp_dat['img_mu']), np.array(samp_dat['img_sigma'])
	tmp_cen_mu, tmp_cen_sigm = np.array(samp_dat['cen_mu']), np.array(samp_dat['cen_sigma'])
	tmp_imgx, tmp_imgy = np.array(samp_dat['bcg_x']), np.array(samp_dat['bcg_y'])
	N_samp = len( tmp_z )

	cen_mu, cen_sigm = [], []
	img_mu, img_sigm = [], []
	dd_ra, dd_dec, dd_z = [], [], []
	dd_imgx, dd_imgy = [], []

	for kk in range( N_samp ):

		if ('%.5f' % tmp_ra[kk] in tt_ra) & ('%.5f' % tmp_dec[kk] in tt_dec):

			dd_ra.append( tmp_ra[kk])
			dd_dec.append( tmp_dec[kk])
			dd_z.append( tmp_z[kk])
			dd_imgx.append( tmp_imgx[kk])
			dd_imgy.append( tmp_imgy[kk])

			cen_mu.append( tmp_cen_mu[kk])
			cen_sigm.append( tmp_cen_sigm[kk])
			img_mu.append( tmp_mu[kk])
			img_sigm.append( tmp_sigm[kk])
		else:
			continue

	cen_sigm = np.array(cen_sigm)
	cen_mu = np.array(cen_mu)
	img_mu = np.array(img_mu)
	img_sigm = np.array(img_sigm)

	dd_ra = np.array( dd_ra )
	dd_dec = np.array( dd_dec )
	dd_z = np.array( dd_z )
	dd_imgx = np.array( dd_imgx )
	dd_imgy = np.array( dd_imgy )

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'cen_mu', 'cen_sigma', 'img_mu', 'img_sigma',]
	values = [dd_ra, dd_dec, dd_z, dd_imgx, dd_imgy, cen_mu, cen_sigm, img_mu, img_sigm]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_put )

	return

### match extra-catalog with the image catalog
def extra_match_func(ra_list, dec_list, z_lis, cat_ra, cat_dec, cat_z, cat_imgx, cat_imgy, sf_len = 5):
	"""
	cat_imgx, cat_imgy : BCG location in image frame
	cat_ra, cat_dec, cat_z : catalog information of image catalog
	ra_list, dec_list, z_lis : catalog information of which used to match to the image catalog
	"""
	lis_ra, lis_dec, lis_z = [], [], []
	lis_x, lis_y = [], []

	com_s = '%.' + '%df' % sf_len

	origin_dex = []

	for kk in range( len(cat_ra) ):

		identi = ( com_s % cat_ra[kk] in ra_list) * (com_s % cat_dec[kk] in dec_list) #* (com_s % cat_z[kk] in z_lis)

		if identi == True:

			## use the location of the source in catalog to make sure they are the same objects in different catalog
			ndex_0 = ra_list.index( com_s % cat_ra[kk] )
			ndex_1 = dec_list.index( com_s % cat_dec[kk] )

			# if ndex_0 == ndex_1:
			lis_ra.append( cat_ra[kk] )
			lis_dec.append( cat_dec[kk] )
			lis_z.append( cat_z[kk] )
			lis_x.append( cat_imgx[kk] )
			lis_y.append( cat_imgy[kk] )

			## origin_dex record the location of objs in the origin catalog (not the image catalog),
			origin_dex.append( ndex_0 )
			# else:
			# 	continue
		else:
			continue

	match_ra = np.array( lis_ra )
	match_dec = np.array( lis_dec )
	match_z = np.array( lis_z )
	match_x = np.array( lis_x )
	match_y = np.array( lis_y )
	origin_dex = np.array( origin_dex )

	return match_ra, match_dec, match_z, match_x, match_y, origin_dex

def gri_common_cat_func(r_band_file, g_band_file, i_band_file, medi_r_file, medi_g_file, out_r_file, out_g_file, out_i_file,):
	"""
	origin_ID : the catalog location of the matched sources
	"""
	r_dat = pds.read_csv( r_band_file )
	r_ra, r_dec, r_z = np.array( r_dat['ra'] ), np.array( r_dat['dec'] ), np.array( r_dat['z'] )
	r_imgx, r_imgy, r_origin_ID = np.array( r_dat['bcg_x'] ), np.array( r_dat['bcg_y'] ), np.array( r_dat['origin_ID'] )

	g_dat = pds.read_csv( g_band_file )
	g_ra, g_dec, g_z = np.array( g_dat['ra'] ), np.array( g_dat['dec'] ), np.array( g_dat['z'] )
	g_imgx, g_imgy, g_origin_ID = np.array( g_dat['bcg_x'] ), np.array( g_dat['bcg_y'] ), np.array( g_dat['origin_ID'] )

	i_dat = pds.read_csv( i_band_file )
	i_ra, i_dec, i_z = np.array( i_dat['ra'] ), np.array( i_dat['dec'] ), np.array( i_dat['z'] )
	i_imgx, i_imgy, i_origin_ID = np.array( i_dat['bcg_x'] ), np.array( i_dat['bcg_y'] ), np.array( i_dat['origin_ID'] )

	N_r, N_g, N_i = len(r_origin_ID), len(g_origin_ID), len(i_origin_ID)

	### common of r and g band
	com_id = []
	sub_lis_r = []
	sub_lis_g = []

	for ii in range( N_r ):

		id_dex = np.abs(r_origin_ID[ ii ] - g_origin_ID)
		id_order = id_dex == 0

		if np.sum( id_order ) == 1:
			get_id = np.where( id_dex == 0)[0][0]

			com_id.append( g_origin_ID[ get_id ] )
			sub_lis_g.append( get_id )
			sub_lis_r.append( ii )

	### save medi catalog
	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID']
	values = [ r_ra[sub_lis_r], r_dec[sub_lis_r], r_z[sub_lis_r], r_imgx[sub_lis_r], r_imgy[sub_lis_r], r_origin_ID[sub_lis_r] ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( medi_r_file )

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID']
	values = [ g_ra[sub_lis_g], g_dec[sub_lis_g], g_z[sub_lis_g], g_imgx[sub_lis_g], g_imgy[sub_lis_g], g_origin_ID[sub_lis_g] ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( medi_g_file )


	### match with i band
	medi_r_dat = pds.read_csv( medi_r_file )
	medi_r_ra, medi_r_dec, medi_r_z = np.array( medi_r_dat['ra'] ), np.array( medi_r_dat['dec'] ), np.array( medi_r_dat['z'] )
	medi_r_imgx, medi_r_imgy, medi_r_origin_ID = np.array( medi_r_dat['bcg_x'] ), np.array( medi_r_dat['bcg_y'] ), np.array( medi_r_dat['origin_ID'] )

	medi_g_dat = pds.read_csv( medi_g_file )
	medi_g_ra, medi_g_dec, medi_g_z = np.array( medi_g_dat['ra'] ), np.array( medi_g_dat['dec'] ), np.array( medi_g_dat['z'] )
	medi_g_imgx, medi_g_imgy, medi_g_origin_ID = np.array( medi_g_dat['bcg_x'] ), np.array( medi_g_dat['bcg_y'] ), np.array( medi_g_dat['origin_ID'] )


	N_mid = len( com_id )

	com_id_1 = []
	sub_lis_r_1 = []

	sub_lis_i = []

	for ii in range( N_mid ):

		id_dex = np.abs( medi_r_origin_ID[ ii ] - i_origin_ID )
		id_order = id_dex == 0

		if np.sum( id_order ) == 1:

			get_id = np.where( id_dex == 0)[0][0]
			com_id_1.append( i_origin_ID[ get_id ] )
			sub_lis_i.append( get_id )
			sub_lis_r_1.append( ii )

	### save the final common catalog
	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID']
	values = [ i_ra[sub_lis_i], i_dec[sub_lis_i], i_z[sub_lis_i], i_imgx[sub_lis_i], i_imgy[sub_lis_i], i_origin_ID[sub_lis_i] ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_i_file )

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID']
	values = [ medi_r_ra[sub_lis_r_1], medi_r_dec[sub_lis_r_1], medi_r_z[sub_lis_r_1], 
				medi_r_imgx[sub_lis_r_1], medi_r_imgy[sub_lis_r_1], medi_r_origin_ID[sub_lis_r_1] ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_r_file )

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID']
	values = [ medi_g_ra[sub_lis_r_1], medi_g_dec[sub_lis_r_1], medi_g_z[sub_lis_r_1], 
				medi_g_imgx[sub_lis_r_1], medi_g_imgy[sub_lis_r_1], medi_g_origin_ID[sub_lis_r_1] ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_g_file )

	return

### match between image catalogs or use for image selection
def cat_match_func(ra_list, dec_list, z_lis, cat_ra, cat_dec, cat_z, cat_imgx, cat_imgy, sf_len, id_choice = True,):
	"""
	id_choice : if it's True, then those imgs in given list will be used,
				if it's False, then those imgs in given list will be rule out
	cat_imgx, cat_imgy : BCG location in image frame
	"""
	lis_ra, lis_dec, lis_z = [], [], []
	lis_x, lis_y = [], []

	com_s = '%.' + '%df' % sf_len

	if id_choice == True:

		origin_dex = []

		for kk in range( len(cat_ra) ):

			identi = ( com_s % cat_ra[kk] in ra_list) * (com_s % cat_dec[kk] in dec_list) * (com_s % cat_z[kk] in z_lis)

			if identi == True:

				ndex_0 = ra_list.index( com_s % cat_ra[kk] )

				lis_ra.append( cat_ra[kk] )
				lis_dec.append( cat_dec[kk] )
				lis_z.append( cat_z[kk] )
				lis_x.append( cat_imgx[kk] )
				lis_y.append( cat_imgy[kk] )

				origin_dex.append( ndex_0 )

			else:
				continue

		match_ra = np.array( lis_ra )
		match_dec = np.array( lis_dec )
		match_z = np.array( lis_z )
		match_x = np.array( lis_x )
		match_y = np.array( lis_y )
		origin_dex = np.array( origin_dex )

		return match_ra, match_dec, match_z, match_x, match_y, origin_dex

	else:
		for kk in range( len(cat_ra) ):

			identi = ( com_s % cat_ra[kk] in ra_list) * (com_s % cat_dec[kk] in dec_list) * (com_s % cat_z[kk] in z_lis)

			if identi == True:
				continue
			else:
				lis_ra.append(cat_ra[kk])
				lis_dec.append(cat_dec[kk])
				lis_z.append(cat_z[kk])
				lis_x.append(cat_imgx[kk])
				lis_y.append(cat_imgy[kk])

		match_ra = np.array(lis_ra)
		match_dec = np.array(lis_dec)
		match_z = np.array(lis_z)
		match_x = np.array(lis_x)
		match_y = np.array(lis_y)

		return match_ra, match_dec, match_z, match_x, match_y

def map_mu_sigma_func(cat_ra, cat_dec, cat_z, cat_imgx, cat_imgy, img_file, band, L_cen, N_step, out_file,):
	"""
	cat_ra, cat_dec, cat_z, cat_imgx, cat_imgy : catalog information, including ra, dec, z and 
		BCG location (cat_imgx, cat_imgy) in image coordinate.
	img_file : imgs will be analysis, have applied masking ('XX/XX/xx.fits')
	L_cen : half length of centeral region box
	N_step : grid size.
	out_file : out-put file.(.csv files)
	band : filter imformation (eg. r, g, i, u, z), str type
	"""
	N_samp = len( cat_ra )

	cen_sigm, cen_mu = [], []
	img_mu, img_sigm = [], []

	for kk in range( N_samp ):

		ra_g, dec_g, z_g = cat_ra[kk], cat_dec[kk], cat_z[kk]
		xn, yn = cat_imgx[kk], cat_imgy[kk]

		# mask imgs
		res_file = img_file % (band, ra_g, dec_g, z_g)
		res_data = fits.open(res_file)
		remain_img = res_data[0].data

		# mask matrix
		idnn = np.isnan(remain_img)
		mask_arr = np.zeros((remain_img.shape[0], remain_img.shape[1]), dtype = np.float32)
		mask_arr[idnn == False] = 1

		ca0, ca1 = np.int( remain_img.shape[0] / 2), np.int( remain_img.shape[1] / 2)
		cen_D = L_cen
		flux_cen = remain_img[ca0 - cen_D: ca0 + cen_D, ca1 - cen_D: ca1 + cen_D]

		cen_lx = np.arange(0, 1100, N_step)
		cen_ly = np.arange(0, 1100, N_step)
		nl0, nl1 = len(cen_ly), len(cen_lx)

		sub_pock_pix = np.zeros((nl0 - 1, nl1 - 1), dtype = np.float)
		sub_pock_flux = np.zeros((nl0 - 1, nl1 - 1), dtype = np.float)
		for nn in range(nl0 - 1):
			for tt in range(nl1 - 1):
				sub_flux = flux_cen[ cen_ly[nn]: cen_ly[nn+1], cen_lx[tt]: cen_lx[tt+1] ]
				id_nn = np.isnan(sub_flux)
				sub_pock_flux[nn,tt] = np.nanmean(sub_flux)
				sub_pock_pix[nn,tt] = len(sub_flux[id_nn == False])

		## mu, sigma of center region
		id_Nzero = sub_pock_pix > 100
		mu = np.nanmean( sub_pock_flux[id_Nzero] )
		sigm = np.nanstd( sub_pock_flux[id_Nzero] )

		cen_sigm.append( sigm )
		cen_mu.append( mu )

		## grid img (for selecting flare, saturated region...)
		block_m, block_pix, block_Var, block_S0, x_edgs, y_edgs = cc_grid_img(remain_img, N_step, N_step)

		idzo = block_pix < 1.
		pix_eta = block_pix / block_S0
		idnn = np.isnan(pix_eta)
		pix_eta[idnn] = 0.
		idnul = pix_eta < 5e-2
		block_m[idnul] = 0.

		img_mu.append( np.nanmean( block_m[idnul == False] ) )
		img_sigm.append( np.nanstd( block_m[idnul == False] ) )

	cen_sigm = np.array(cen_sigm)
	cen_mu = np.array(cen_mu)
	img_mu = np.array(img_mu)
	img_sigm = np.array(img_sigm)

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'cen_mu', 'cen_sigma', 'img_mu', 'img_sigma',]
	values = [ cat_ra, cat_dec, cat_z, cat_imgx, cat_imgy, cen_mu, cen_sigm, img_mu, img_sigm ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_file )

	return

def img_cat_lis_func(img_file, ref_cat, out_put, sf_len, id_choice = True,):
	"""
	img_file : imgs need to capture information, only including data formats(.png, .pdf, .jpg, ...)
				and the path (in which imgs are saved.)

				format : /path/XXX_raX-decX-zX.xxx

	ref_cat : catalog in which match the imgs information, .csv files

	out_put : informations of match those imgs, including [ra, dec, z, bcg_x, bcg_y]
				(bcg_x, bcg_y) is the location of BCG in image frame. .csv files	
	"""
	lis = glob.glob( img_file )
	name_lis = [ ll.split('/')[-1] for ll in lis ]

	tt_lis = name_lis[0].split('_')

	dete0 = ['ra' in ll for ll in tt_lis]
	index0 = dete0.index( True )

	sub_str = tt_lis[ index0 ].split('-')
	dete0_1 = [ 'ra' in ll for ll in sub_str ]
	index0_1 = dete0_1.index( True )

	dete1 = ['dec' in ll for ll in tt_lis]
	index1 = dete1.index( True )

	sub_str = tt_lis[ index1 ].split('-')
	dete1_1 = [ 'dec' in ll for ll in sub_str ]
	index1_1 = dete1_1.index( True )	

	dete2 = ['-z0.' in ll for ll in tt_lis]
	index2 = dete2.index( True )

	sub_str = tt_lis[ index2 ].split('-')
	dete2_1 = [ 'z0.' in ll for ll in sub_str ]
	index2_1 = dete2_1.index( True )

	out_ra = [ ll.split('_')[ index0 ].split('-')[ index0_1 ][2:] for ll in name_lis ]
	out_dec = [ ll.split('_')[ index1 ].split('-')[ index1_1 ][3:] for ll in name_lis ]
	out_z = [ ll.split('_')[ index2 ].split('-')[ index2_1 ][1:] for ll in name_lis ]

	ref_dat = pds.read_csv( ref_cat )
	cat_ra, cat_dec, cat_z = np.array(ref_dat.ra), np.array(ref_dat.dec), np.array(ref_dat.z)
	clus_x, clus_y = np.array(ref_dat.bcg_x), np.array(ref_dat.bcg_y)

	lis_ra, lis_dec, lis_z, lis_x, lis_y, cat_order = cat_match_func( out_ra, out_dec, out_z, cat_ra, cat_dec, cat_z, clus_x, clus_y, sf_len, id_choice,)

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
	values = [lis_ra, lis_dec, lis_z, lis_x, lis_y]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_put )

	return

## === ## use for image over view
def gau_func(x, mu, sigma):
	return sts.norm.pdf(x, mu, sigma)

def get_cat(star_cat, gal_cat, pixel, wcs_lis, norm_star_r = 30, brit_star_r = 75, galax_r = 16,):
	"""
	for given catalogs of sources, pointing the image position, and show the mask size.
	pixel : pixel scale (unit -- arcsec)
	star_cat, gal_cat : stars and galaxies of imgs,
	wcs_lis : the World Coordinate System (saved in .fits file header)
	norm_star_r : masking size applied on normal stars (norm_star_r * (FWHM / 2) )
	brit_star_r : masking size applied on bright stars (brit_star_r * (FWHM / 2) )
	galax_r : masking size applied on galaxies (here is all of the sources detected by Source Extractor),
				( (galax_r / 2) * R_kron )
	(default value have been set for SDSS imgs)
	"""
	## read source catalog
	cat = pds.read_csv(star_cat, skiprows = 1)
	set_ra = np.array(cat['ra'])
	set_dec = np.array(cat['dec'])
	set_mag = np.array(cat['r'])
	OBJ = np.array(cat['type'])
	xt = cat['Column1']
	flags = [str(qq) for qq in xt]

	x, y = wcs_lis.all_world2pix(set_ra * U.deg, set_dec * U.deg, 1)

	set_A = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i']]) / pixel
	set_B = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i']]) / pixel
	set_chi = np.zeros(set_A.shape[1], dtype = np.float32)

	lln = np.array([len(set_A[:,ll][set_A[:,ll] > 0 ]) for ll in range(set_A.shape[1]) ])
	lr_iso = np.array([np.max(set_A[:,ll]) for ll in range(set_A.shape[1]) ])
	sr_iso = np.array([np.max(set_B[:,ll]) for ll in range(set_B.shape[1]) ])

	# normal stars
	iq = lln >= 2 ## at lest observed in 2 band
	ig = OBJ == 6
	ie = (set_mag <= 20)

	ic = (ie & ig & iq)
	sub_x0 = x[ic]
	sub_y0 = y[ic]

	sub_A0 = lr_iso[ic] * norm_star_r
	sub_B0 = sr_iso[ic] * norm_star_r
	sub_chi0 = set_chi[ic]

	# saturated source(may not stars)
	xa = ['SATURATED' in qq for qq in flags]
	xv = np.array(xa)
	idx = xv == True
	ipx = (idx)

	sub_x2 = x[ipx]
	sub_y2 = y[ipx]

	sub_A2 = lr_iso[ipx] * brit_star_r
	sub_B2 = sr_iso[ipx] * brit_star_r
	sub_chi2 = set_chi[ipx]

	comx = np.r_[sub_x0[sub_A0 > 0], sub_x2[sub_A2 > 0]]
	comy = np.r_[sub_y0[sub_A0 > 0], sub_y2[sub_A2 > 0]]
	Lr = np.r_[sub_A0[sub_A0 > 0], sub_A2[sub_A2 > 0]]
	Sr = np.r_[sub_B0[sub_A0 > 0], sub_B2[sub_A2 > 0]]
	phi = np.r_[sub_chi0[sub_A0 > 0], sub_chi2[sub_A2 > 0]]

	source = asc.read(gal_cat)
	Numb = np.array(source['NUMBER'][-1])
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])
	theta = np.array(source['THETA_IMAGE'])
	cx = np.array(source['X_IMAGE']) - 1
	cy = np.array(source['Y_IMAGE']) - 1

	Kron = galax_r
	a = Kron * A
	b = Kron * B

	tot_cx = np.r_[cx, comx]
	tot_cy = np.r_[cy, comy]
	tot_a = np.r_[a, Lr]
	tot_b = np.r_[b, Sr]
	tot_theta = np.r_[theta, phi]
	tot_Numb = Numb + len(comx)

	return tot_Numb, tot_cx, tot_cy, tot_a, tot_b, tot_theta

def hist_analysis_func(cat_file, img_file, mask_img_file, band, star_cat, gal_cat, out_imgs, pix_scale, N_step,):
	"""
	cat_file : cluster catalog (.csv file, including path information)
	img_file : imgs have no mask
	mask_img_file : imgs have been masked
	band : band (ie. 'g', 'r', 'i',), str type
	star_cat : star catalog. (.txt or .csv file, including path information)
	gal_cat : source catalog in image region (builded by Source Extractor, .cat file, including path information)
	out_imgs : ouput of figs,(.png or .pdf files, including path information)
	pix_scale : pixel size of CCD, in unit of 'arcsec'
	N_step : grid size for img division.
	"""
	### cat and select
	dat = pds.read_csv( cat_file )
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	Da = Test_model.angular_diameter_distance( z ).value

	N_samp = len( z )
	pixel = pix_scale * 1.

	Angu_l = ( 1 / Da ) * rad2asec
	R_pix = Angu_l / pixel

	cen_sigm, cen_mu = [], []
	img_mu, img_sigm = [], []
	### overview on img structure
	for kk in range( N_samp ):

		ra_g, dec_g, z_g = ra[kk], dec[kk], z[kk]
		# original img
		data = fits.open( img_file % (band, ra_g, dec_g, z_g),)
		img = data[0].data
		head = data[0].header
		wcs_lis = awc.WCS(head)
		xn, yn = clus_x[kk], clus_y[kk]

		# mask imgs
		res_data = fits.open( mask_img_file % (band, ra_g, dec_g, z_g),)
		remain_img = res_data[0].data

		# grid img (for selecting flare, saturated region...)
		block_m, block_pix, block_Var, block_S0, x_edgs, y_edgs = cc_grid_img(remain_img, N_step, N_step)

		idzo = block_pix < 1.
		pix_eta = block_pix / block_S0
		idnn = np.isnan(pix_eta)
		pix_eta[idnn] = 0.
		idnul = pix_eta < 5e-2
		block_m[idnul] = 0.

		img_mu.append( np.nanmean( block_m[idnul == False] ) )
		img_sigm.append( np.nanstd( block_m[idnul == False] ) )

		idnn = np.isnan(remain_img)
		bin_flux = remain_img[idnn == False]
		bin_di = np.linspace(bin_flux.min(), bin_flux.max(), 51) / pixel**2

		pix_n, edgs = binned(bin_flux / pixel**2, bin_flux / pixel**2, statistic = 'count', bins = bin_di)[:2]
		pdf_pix = (pix_n / np.sum(pix_n) ) / (edgs[1] - edgs[0])
		pdf_err = (np.sqrt(pix_n) / np.sum(pix_n) ) / (edgs[1] - edgs[0])
		x_cen = 0.5 * ( edgs[1:] + edgs[:-1])

		idu = pix_n != 0.
		use_obs = pix_n[idu]
		use_err = np.sqrt(use_obs)
		use_x = x_cen[idu]
		popt, pcov = curve_fit(gau_func, use_x, pdf_pix[idu], 
			p0 = [np.mean(bin_flux / pixel**2), np.std(bin_flux / pixel**2)], sigma = pdf_err[idu],)
		e_mu, e_chi = popt[0], popt[1]
		fit_line = gau_func(x_cen, e_mu, e_chi)

		### applied the mask region
		star_file = star_cat % (z_g, ra_g, dec_g)
		galax_file = gal_cat % (ra_g, dec_g, z_g)

		tot_Numb, tot_cx, tot_cy, tot_a, tot_b, tot_theta = get_cat(star_file, galax_file, pixel, wcs_lis,)
		sc_x, sc_y = tot_cx / (img.shape[1] / block_m.shape[1]), tot_cy / (img.shape[0] / block_m.shape[0])
		sc_a, sc_b = tot_a * (block_m.shape[1] / img.shape[1]), tot_b * (block_m.shape[0] / img.shape[0])
		sc_x, sc_y = sc_x - 0.5, sc_y - 0.5

		Rpp = R_pix[ kk ]

		fig = plt.figure( figsize = (13.12, 9.84) )
		ax0 = fig.add_axes([0.05, 0.55, 0.40, 0.45])
		ax1 = fig.add_axes([0.55, 0.55, 0.40, 0.45])
		ax2 = fig.add_axes([0.05, 0.05, 0.40, 0.45])
		ax3 = fig.add_axes([0.55, 0.05, 0.40, 0.40])

		ax0.set_title('img ra%.3f dec%.3f z%.3f' % (ra_g, dec_g, z_g),)
		tf = ax0.imshow(img / pixel**2, cmap = 'Greys', origin = 'lower', vmin = 1e-4, vmax = 1e0, norm = mpl.colors.LogNorm())
		clust = Circle(xy = (xn, yn), radius = Rpp, fill = False, ec = 'b', ls = '-', linewidth = 1, alpha = 0.50,)
		ax0.add_patch(clust)
		cb = plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)

		ax1.set_title('after masking')
		tg = ax1.imshow( remain_img / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-1, vmax = 4e-1,)
		cb = plt.colorbar(tg, ax = ax1, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
		cb.formatter.set_powerlimits((0,0))

		### image patch_mean case
		ax2.set_title('2D hist of sub-patch mean value')
		th = ax2.imshow( block_m / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
		cb = plt.colorbar(th, ax = ax2, fraction = 0.034, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
		cb.formatter.set_powerlimits((0,0))

		for mm in range( tot_Numb ):
			ellips = Ellipse(xy = (sc_x[mm], sc_y[mm]), width = sc_a[mm], height = sc_b[mm], angle = tot_theta[mm], fill = True, fc = 'w', 
				ec = 'w', ls = '-', linewidth = 0.75,)
			ax2.add_patch(ellips)

		for mm in range(block_m.shape[1]):
			for nn in range(block_m.shape[0]):
				ax2.text(mm, nn, s = '%.3f' % pix_eta[nn, mm], ha = 'center', va = 'center', color = 'g', fontsize = 8, alpha = 0.5)
		ax2.set_xlabel('effective pixel ratio shown in green text')

		ax3.set_title('pixel SB PDF [after masking]')
		ax3.hist(bin_flux / pixel**2, bins = bin_di, density = True, color = 'b', alpha = 0.5,)
		ax3.plot(x_cen, fit_line, color = 'r', alpha = 0.5, label = 'Gaussian \n $\\mu=%.4f$ \n $\\sigma=%.4f$' % (e_mu, e_chi),)
		ax3.axvline(x = 0, ls = '-', color = 'k', alpha = 0.5,)
		ax3.axvline(x = e_mu - e_chi, ls = '--', color = 'k', alpha = 0.5, label = '1 $\\sigma$')
		ax3.axvline(x = e_mu + e_chi, ls = '--', color = 'k', alpha = 0.5, )
		ax3.legend(loc = 1, frameon = False,)
		ax3.set_xlabel('pixel SB [nanomaggies / $arcsec^2$]')
		ax3.set_ylabel('PDF')

		plt.savefig( out_imgs % (band, ra_g, dec_g, z_g), dpi = 300)
		plt.close()

		raise

	return

