import h5py
import numpy as np
import pandas as pds
import astropy.constants as C
import astropy.units as U
import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.io.ascii as asc

from astropy import cosmology as apcy
from scipy import signal
from scipy import interpolate as interp
from scipy import ndimage
from scipy import integrate as integ
from io import StringIO

#constant
rad2arcsec = U.rad.to(U.arcsec)

# cosmology model
vc = C.c.to(U.km/U.s).value
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
DH = vc / H0

# band information of SDSS
band = ['r', 'g', 'i']
l_wave = np.array([6166, 4686, 7480])
mag_add = np.array([0, 0, 0 ])

Mag_sun = [ 4.65, 5.11, 4.53 ]  ## Abs_magnitude of the Sun (for SDSS filters)

#**************************#
def sersic_func(r, Ie, re, ndex):
	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

### === ### absMag to luminosity in unit of L_sun
def absMag_to_Lumi_func( absM_arr, band_str ):

	if band_str == 'r':
		Mag_dot = Mag_sun[0]

	if band_str == 'g':
		Mag_dot = Mag_sun[1]

	if band_str == 'i':
		Mag_dot = Mag_sun[2]

	L_obs = 10**( 0.4 * ( Mag_dot - absM_arr ) )

	return L_obs


### === ### coordinate or position translation
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

def zref_BCG_pos_func( cat_file, z_ref, out_file, pix_size,):
	"""
	this part use for calculate BCG position after pixel resampling. 
	"""
	dat = pds.read_csv( cat_file )
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	Da_z = Test_model.angular_diameter_distance(z).value
	Da_ref = Test_model.angular_diameter_distance(z_ref).value

	L_ref = Da_ref * pix_size / rad2arcsec
	L_z = Da_z * pix_size / rad2arcsec
	eta = L_ref / L_z

	ref_bcgx = clus_x / eta
	ref_bcgy = clus_y / eta

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
	values = [ra, dec, z, ref_bcgx, ref_bcgy]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_file )

	return


### === ### member galaxies matched to host cluster
#. for SDSS redMaPPer catalog 
def mem_clus_match_func( cat_file, mem_file, stack_cat_file, out_file):
	"""
	out_file : hdf5 files
	"""
	cat_data = fits.open( cat_file )
	goal_data = cat_data[1].data
	RA = np.array(goal_data.RA)
	DEC = np.array(goal_data.DEC)
	ID = np.array(goal_data.ID)

	z_phot = np.array(goal_data.Z_LAMBDA)
	idvx = (z_phot >= 0.2) & (z_phot <= 0.3)

	ref_Ra, ref_Dec, ref_Z = RA[idvx], DEC[idvx], z_phot[idvx]
	ref_ID = ID[idvx]

	mem_data = fits.open( mem_file )
	sate_data = mem_data[1].data

	group_ID = np.array(sate_data.ID)
	centric_R = np.array(sate_data.R)
	P_member = np.array(sate_data.P)

	mem_r_mag = np.array(sate_data.MODEL_MAG_R)
	mem_r_mag_err = np.array(sate_data.MODEL_MAGERR_R)

	mem_g_mag = np.array(sate_data.MODEL_MAG_G)
	mem_g_mag_err = np.array(sate_data.MODEL_MAGERR_G)

	mem_i_mag = np.array(sate_data.MODEL_MAG_I)
	mem_i_mag_err = np.array(sate_data.MODEL_MAGERR_I)

	sat_ra, sat_dec = np.array(sate_data.RA), np.array(sate_data.DEC)
	sat_Z = np.array( sate_data.Z_SPEC )
	sat_objID = np.array( sate_data.OBJID )

	d_cat = pds.read_csv( stack_cat_file )
	ra, dec, z = np.array( d_cat['ra']), np.array( d_cat['dec']), np.array( d_cat['z'])

	out_ra = ['%.5f' % ll for ll in ref_Ra ]
	out_dec = ['%.5f' % ll for ll in ref_Dec ]
	out_z = ['%.5f' % ll for ll in ref_Z ]

	sub_index = simple_match( out_ra, out_dec, out_z, ra, dec, z )[-1]
	match_ID = ref_ID[ sub_index ]
	Ns = len( ra )

	F_tree = h5py.File( out_file, 'w')

	for qq in range( Ns ):

		ra_g, dec_g, z_g = ra[qq], dec[qq], z[qq]

		targ_ID = match_ID[qq]

		id_group = group_ID == targ_ID

		cen_R_arr = centric_R[ id_group ]
		sub_Pmem = P_member[ id_group ]

		sub_r_mag = mem_r_mag[ id_group ]
		sub_g_mag = mem_g_mag[ id_group ]
		sub_i_mag = mem_i_mag[ id_group ]

		sub_r_mag_err = mem_r_mag_err[ id_group ]
		sub_g_mag_err = mem_g_mag_err[ id_group ]
		sub_i_mag_err = mem_i_mag_err[ id_group ]

		sub_ra, sub_dec, sub_z = sat_ra[ id_group ], sat_dec[ id_group ], sat_Z[ id_group ]
		sub_obj_IDs = sat_objID[ id_group ]

		sub_g2r = sub_g_mag - sub_r_mag
		sub_g2i = sub_g_mag - sub_i_mag
		sub_r2i = sub_r_mag - sub_i_mag

		out_arr = np.array( [ sub_ra, sub_dec, sub_z, cen_R_arr, sub_Pmem, sub_g2r, sub_g2i, sub_r2i ] )
		gk = F_tree.create_group( "clust_%d/" % qq )
		dk0 = gk.create_dataset( "arr", data = out_arr )
		dk1 = gk.create_dataset( "IDs", data = sub_obj_IDs )

	F_tree.close()

	return


### === ### img grid
def cc_grid_img( img_data, N_stepx, N_stepy):

	binx = img_data.shape[1] // N_stepx ## bin number along 2 axis
	biny = img_data.shape[0] // N_stepy

	beyon_x = img_data.shape[1] - binx * N_stepx ## for edge pixels divid
	beyon_y = img_data.shape[0] - biny * N_stepy

	odd_x = np.ceil(beyon_x / binx)
	odd_y = np.ceil(beyon_y / biny)

	n_odd_x = beyon_x // odd_x
	n_odd_y = beyon_y // odd_y

	d_odd_x = beyon_x - odd_x * n_odd_x
	d_odd_y = beyon_y - odd_y * n_odd_y

	# get the bin width
	wid_x = np.zeros(binx, dtype = np.float32)
	wid_y = np.zeros(biny, dtype = np.float32)
	for kk in range(binx):
		if kk == n_odd_x :
			wid_x[kk] = N_stepx + d_odd_x
		elif kk < n_odd_x :
			wid_x[kk] = N_stepx + odd_x
		else:
			wid_x[kk] = N_stepx

	for kk in range(biny):
		if kk == n_odd_y :
			wid_y[kk] = N_stepy + d_odd_y
		elif kk < n_odd_y :
			wid_y[kk] = N_stepy + odd_y
		else:
			wid_y[kk] = N_stepy

	# get the bin edge
	lx = np.zeros(binx + 1, dtype = np.int32)
	ly = np.zeros(biny + 1, dtype = np.int32)
	for kk in range(binx):
		lx[kk + 1] = lx[kk] + wid_x[kk]
	for kk in range(biny):
		ly[kk + 1] = ly[kk] + wid_y[kk]

	patch_mean = np.zeros( (biny, binx), dtype = np.float )
	patch_pix = np.zeros( (biny, binx), dtype = np.float )
	patch_S0 = np.zeros( (biny, binx), dtype = np.float )
	patch_Var = np.zeros( (biny, binx), dtype = np.float )
	for nn in range( biny ):
		for tt in range( binx ):

			sub_flux = img_data[ ly[nn]: ly[nn + 1], lx[tt]: lx[tt + 1] ]
			id_nn = np.isnan(sub_flux)

			patch_mean[nn,tt] = np.mean( sub_flux[id_nn == False] )
			patch_pix[nn,tt] = len( sub_flux[id_nn == False] )
			patch_Var[nn,tt] = np.std( sub_flux[id_nn == False] )
			patch_S0[nn,tt] = (ly[nn + 1] - ly[nn]) * (lx[tt + 1] - lx[tt])

	return patch_mean, patch_pix, patch_Var, patch_S0, lx, ly

def grid_img( img_data, N_stepx, N_stepy):

	ly = np.arange(0, img_data.shape[0], N_stepy)
	ly = np.r_[ly, img_data.shape[0] - N_stepy, img_data.shape[0] ]
	lx = np.arange(0, img_data.shape[1], N_stepx)
	lx = np.r_[lx, img_data.shape[1] - N_stepx, img_data.shape[1] ]

	lx = np.delete(lx, -1)
	lx = np.delete(lx, -2)
	ly = np.delete(ly, -1)
	ly = np.delete(ly, -2)

	patch_mean = np.zeros( (len(ly), len(lx) ), dtype = np.float )
	patch_pix = np.zeros( (len(ly), len(lx) ), dtype = np.float )
	patch_Var = np.zeros( (len(ly), len(lx) ), dtype = np.float )
	for nn in range( len(ly) ):
		for tt in range( len(lx) ):

			sub_flux = img_data[ly[nn]: ly[nn] + N_stepy, lx[tt]: lx[tt] + N_stepx]
			id_nn = np.isnan(sub_flux)
			patch_mean[nn,tt] = np.mean( sub_flux[id_nn == False] )
			patch_pix[nn,tt] = len( sub_flux[id_nn == False] )
			patch_Var[nn,tt] = np.std( sub_flux[id_nn == False] )

	return patch_mean, patch_pix, patch_Var, lx, ly


### === source location offset correction in each image frame
def cc_star_pos_func( star_cat, Head_info, pix_size):

	wcs_lis = awc.WCS( Head_info )

	## stars catalog
	p_cat = pds.read_csv( star_cat, skiprows = 1)
	set_ra = np.array( p_cat['ra'])
	set_dec = np.array( p_cat['dec'])
	set_mag = np.array( p_cat['r'])
	OBJ = np.array( p_cat['type'])
	xt = p_cat['Column1']
	flags = [str(qq) for qq in xt]

	x, y = wcs_lis.all_world2pix( set_ra * U.deg, set_dec * U.deg, 0, ra_dec_order = True,)

	set_A = np.array( [ p_cat['psffwhm_r'] , p_cat['psffwhm_g'], p_cat['psffwhm_i']]) / pix_size
	set_B = np.array( [ p_cat['psffwhm_r'] , p_cat['psffwhm_g'], p_cat['psffwhm_i']]) / pix_size
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

	sub_A0 = lr_iso[ic] * 15
	sub_B0 = sr_iso[ic] * 15
	sub_chi0 = set_chi[ic]

	sub_ra0, sub_dec0 = set_ra[ic], set_dec[ic]

	# saturated source(may not stars)
	xa = ['SATURATED' in qq for qq in flags]
	xv = np.array(xa)
	idx = xv == True
	ipx = (idx)

	sub_x2 = x[ipx]
	sub_y2 = y[ipx]

	sub_A2 = lr_iso[ipx] * 5
	sub_B2 = sr_iso[ipx] * 5
	sub_chi2 = set_chi[ipx]

	## for stars
	ddx = np.around( sub_x0 )
	d_x0 = np.array( list( set( ddx ) ) )

	m_A0, m_B0, m_chi0 = [], [], []
	m_x0, m_y0 = [], []

	for jj in range( len( d_x0 ) ):
		dex_0 = list( ddx ).index( d_x0[jj] )

		m_x0.append( sub_x0[dex_0] )
		m_y0.append( sub_y0[dex_0] )
		m_A0.append( sub_A0[dex_0] )
		m_B0.append( sub_B0[dex_0] )
		m_chi0.append( sub_chi0[dex_0] )

	m_A0, m_B0, m_chi0 = np.array( m_A0 ), np.array( m_B0 ), np.array( m_chi0 )
	m_x0, m_y0 = np.array( m_x0 ), np.array( m_y0 )

	cm_x0, cm_y0 = sub_x0 + 0, sub_y0 + 0
	cm_A0, cm_B0, cm_chi0 = sub_A0 + 0, sub_B0 + 0, sub_chi0 + 0

	return cm_x0, cm_y0, cm_A0, cm_B0, cm_chi0, sub_ra0, sub_dec0

def star_pos_func( star_cat, Head_info, pix_size ):

	## stars catalog
	p_cat = pds.read_csv( star_cat, skiprows = 1)
	set_ra = np.array( p_cat['ra'])
	set_dec = np.array( p_cat['dec'])
	set_mag = np.array( p_cat['r'])
	OBJ = np.array( p_cat['type'])
	xt = p_cat['Column1']
	flags = [str(qq) for qq in xt]

	x, y = WCS_to_pixel_func( set_ra, set_dec, Head_info ) ## SDSS EDR paper relation

	set_A = np.array( [ p_cat['psffwhm_r'] , p_cat['psffwhm_g'], p_cat['psffwhm_i']]) / pix_size
	set_B = np.array( [ p_cat['psffwhm_r'] , p_cat['psffwhm_g'], p_cat['psffwhm_i']]) / pix_size
	set_chi = np.zeros(set_A.shape[1], dtype = np.float32)

	lln = np.array([len(set_A[:,ll][ set_A[:,ll] > 0 ]) for ll in range(set_A.shape[1]) ])
	lr_iso = np.array([np.max(set_A[:,ll]) for ll in range(set_A.shape[1]) ])
	sr_iso = np.array([np.max(set_B[:,ll]) for ll in range(set_B.shape[1]) ])

	# normal stars
	iq = lln >= 2 ## at lest observed in 2 band
	ig = OBJ == 6
	ie = (set_mag <= 20)

	ic = (ie & ig & iq)
	sub_x0 = x[ic]
	sub_y0 = y[ic]

	sub_A0 = lr_iso[ic] * 15
	sub_B0 = sr_iso[ic] * 15
	sub_chi0 = set_chi[ic]

	sub_ra0, sub_dec0 = set_ra[ic], set_dec[ic]

	# saturated source(may not stars)
	xa = ['SATURATED' in qq for qq in flags]
	xv = np.array(xa)
	idx = xv == True
	ipx = (idx)

	sub_x2 = x[ipx]
	sub_y2 = y[ipx]

	sub_A2 = lr_iso[ipx] * 5
	sub_B2 = sr_iso[ipx] * 5
	sub_chi2 = set_chi[ipx]

	## for stars
	ddx = np.around( sub_x0 )
	d_x0 = np.array( list( set( ddx ) ) )

	m_A0, m_B0, m_chi0 = [], [], []
	m_x0, m_y0 = [], []

	for jj in range( len( d_x0 ) ):
		dex_0 = list( ddx ).index( d_x0[jj] )

		m_x0.append( sub_x0[dex_0] )
		m_y0.append( sub_y0[dex_0] )
		m_A0.append( sub_A0[dex_0] )
		m_B0.append( sub_B0[dex_0] )
		m_chi0.append( sub_chi0[dex_0] )

	m_A0, m_B0, m_chi0 = np.array( m_A0 ), np.array( m_B0 ), np.array( m_chi0 )
	m_x0, m_y0 = np.array( m_x0 ), np.array( m_y0 )

	cm_x0, cm_y0 = sub_x0 + 0, sub_y0 + 0
	cm_A0, cm_B0, cm_chi0 = sub_A0 + 0, sub_B0 + 0, sub_chi0 + 0

	return cm_x0, cm_y0, cm_A0, cm_B0, cm_chi0, sub_ra0, sub_dec0

def tractor_peak_pos( img_file, gal_cat ):

	data = fits.open( img_file )
	img = data[0].data

	source = asc.read(gal_cat)
	Numb = np.array(source['NUMBER'][-1])
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])
	theta = np.array(source['THETA_IMAGE'])
	cx = np.array(source['X_IMAGE'])
	cy = np.array(source['Y_IMAGE'])
	p_type = np.array(source['CLASS_STAR'])

	Kron = 7
	a = Kron * A
	b = Kron * B

	major = a / 2
	minor = b / 2
	senior = np.sqrt(major**2 - minor**2)

	x_peak, y_peak = [], []

	for k in range( Numb ):

		xc = cx[k]
		yc = cy[k]

		lr = major[k]
		sr = minor[k]
		cr = senior[k]
		chi = theta[k] * np.pi / 180

		set_r = np.int(np.ceil(1.2 * lr))
		la0 = np.max( [np.int(xc - set_r), 0])
		la1 = np.min( [np.int(xc + set_r + 1), img.shape[1] ] )
		lb0 = np.max( [np.int(yc - set_r), 0] ) 
		lb1 = np.min( [np.int(yc + set_r + 1), img.shape[0] ] )

		cut_img = img[ lb0 : lb1, la0 : la1 ]
		x_p, y_p = np.where( cut_img == np.nanmax( cut_img ) )

		x_peak.append( x_p[0] + la0 )
		y_peak.append( y_p[0] + lb0 )

	x_peak, y_peak = np.array( x_peak ), np.array( y_peak )

	return cx, cy, x_peak, y_peak


### === ### 1D profile fitting
def color_func( flux_arr_0, flux_err_0, flux_arr_1, flux_err_1):

	mag_arr_0 = 22.5 - 2.5 * np.log10( flux_arr_0 )
	mag_arr_1 = 22.5 - 2.5 * np.log10( flux_arr_1 )

	color_pros = mag_arr_0 - mag_arr_1

	sigma_0 = 2.5 * flux_err_0 / (np.log(10) * flux_arr_0 )
	sigma_1 = 2.5 * flux_err_1 / (np.log(10) * flux_arr_1 )

	color_err = np.sqrt( sigma_0**2 + sigma_1**2 )

	return color_pros, color_err

def SB_to_Lumi_func(sb_arr, obs_z, band_str):
	"""
	sb_arr : need in terms of absolute magnitude, in AB system
	"""
	if band_str == 'r':
		Mag_dot = Mag_sun[0]

	if band_str == 'g':
		Mag_dot = Mag_sun[1]

	if band_str == 'i':
		Mag_dot = Mag_sun[2]

	# luminosity, in unit of  L_sun / pc^2
	Lumi = 10**( -0.4 * (sb_arr - Mag_dot + 21.572 - 10 * np.log10( obs_z + 1 ) ) )

	return Lumi

def arr_jack_func(SB_array, R_array, N_sample):
	"""
	SB_array : y-data for jackknife resampling
	R_array : x-data for jackknife resampling
	( SB_array, R_array : list type )

	N_sample : number of sub-samples
	"""
	dx_r = np.array(R_array)
	dy_sb = np.array(SB_array)

	n_r = dx_r.shape[1]
	Len = np.zeros( n_r, dtype = np.float32)
	for nn in range( n_r ):
		tmp_I = dy_sb[:,nn]
		idnn = np.isnan(tmp_I)
		Len[nn] = N_sample - np.sum(idnn)

	Stack_R = np.nanmean(dx_r, axis = 0)
	Stack_SB = np.nanmean(dy_sb, axis = 0)
	std_Stack_SB = np.nanstd(dy_sb, axis = 0)

	### only calculate r bins in which sub-sample number larger than one
	id_one = Len > 1
	Stack_R = Stack_R[ id_one ]
	Stack_SB = Stack_SB[ id_one ]
	std_Stack_SB = std_Stack_SB[ id_one ]
	N_img = Len[ id_one ]
	jk_Stack_err = np.sqrt(N_img - 1) * std_Stack_SB

	### limit the radius bin contribution at least 1/10 * N_sample
	id_min = N_img >= np.int(N_sample / 10)
	lim_r = Stack_R[id_min]
	lim_R = np.nanmax(lim_r)

	return Stack_R, Stack_SB, jk_Stack_err, lim_R


### === ### 2D array, centeriod mean
def centric_2D_aveg(data, weit_data, pix_size, cx, cy, z0, R_bins):

	Da0 = Test_model.angular_diameter_distance(z0).value ## in unit 'Mpc'
	Nx = data.shape[1]
	Ny = data.shape[0]
	x0 = np.linspace(0, Nx-1, Nx)
	y0 = np.linspace(0, Ny-1, Ny)
	pix_id = np.array(np.meshgrid(x0,y0))

	theta = np.arctan2((pix_id[1,:]-cy), (pix_id[0,:]-cx))
	chi = theta * 180 / np.pi

	rbin = R_bins # have been divided bins, in unit of pixels
	set_r = rbin * pix_size * Da0 * 1e3 / rad2arcsec # in unit of kpc

	aveg_arr = np.ones( (Ny, Nx), dtype = np.float32 )

	dev_05_x = cx - np.int( cx )
	dev_05_y = cy - np.int( cy )

	if dev_05_x > 0.5:
		xn = np.int( cx ) + 1
	else:
		xn = np.int( cx )

	if dev_05_y > 0.5:
		yn = np.int( cy ) + 1
	else:
		yn = np.int( cy )

	dr = np.sqrt( ( (2 * pix_id[0] + 1) / 2 - (2 * xn + 1) / 2 )**2 + ( (2 * pix_id[1] + 1) / 2 - (2 * yn + 1) / 2)**2 )

	for k in range( 1,len(rbin) - 1):

		cdr = rbin[k + 1] - rbin[k]
		d_phi = (cdr / ( 0.5 * (rbin[k] + rbin[k + 1]) ) ) * 180 / np.pi
		N_phi = np.int(360 / d_phi) + 1
		phi = np.linspace(0, 360, N_phi)
		phi = phi - 180

		ir = (dr >= rbin[k]) & (dr < rbin[k + 1])
		bool_sum = np.sum(ir)

		if bool_sum == 0:
			continue
		else:
			weit_arr = weit_data[ir]

			samp_flux = data[ir]
			samp_chi = chi[ir]
			tot_flux = np.nansum(samp_flux * weit_arr) / np.nansum(weit_arr)

			aveg_arr[ir] = tot_flux + 0.

	## center pixel
	aveg_arr[yn, xn] = data[yn, xn]

	id_Nan = np.isnan( data )
	aveg_arr[id_Nan] = np.nan

	bins_edgs = np.r_[ np.sqrt( pix_size**2 / np.pi ), rbin[1:] * pix_size ]

	return bins_edgs, aveg_arr
