import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp

import astropy.units as U
import astropy.constants as C

from astropy import cosmology as apcy
from scipy import optimize
from scipy import stats as sts
from scipy import signal
from scipy import interpolate as interp
from scipy import optimize
from scipy import integrate as integ

### === ### cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

band = ['r', 'g', 'i']
Mag_sun = [ 4.65, 5.11, 4.53 ]
L_wave = np.array([ 6166, 4686, 7480 ])
psf_FWHM = [ 1.56, 1.67, 1.50 ] # arcsec

z_ref = 0.25
pixel = 0.396

Dl_ref = Test_model.luminosity_distance( z_ref ).value
Da_ref = Test_model.angular_diameter_distance( z_ref ).value
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec

def sql_color( cat_z, cat_ra, cat_dec, cat_ID, out_file ):

	import mechanize
	from io import StringIO

	url = 'http://skyserver.sdss.org/dr12/en/tools/search/sql.aspx'

	Nz = len(cat_z)

	for kk in range(Nz):

		z_g = cat_z[kk]
		ra_g = cat_ra[kk]
		dec_g = cat_dec[kk]

		data_set = """
		SELECT
			p.objID, p.cModelMag_g, p.cModelMag_r, p.cModelMag_i

		FROM PhotoObjAll as p

		WHERE
			p.objID = %d
		""" % cat_ID[kk]

		br = mechanize.Browser()
		resp = br.open(url)
		resp.info()
		#print(data_set)
		
		br.select_form(name = "sql")
		br['cmd'] = data_set
		br['format'] = ['csv']
		response = br.submit()
		s = str(response.get_data(), encoding = 'utf-8')
		doc = open( out_file % (z_g, ra_g, dec_g), 'w')
		print(s, file = doc)
		doc.close()

	print( 'down!' )

	return

def simple_match(ra_list, dec_list, z_lis, cat_ra, cat_dec, cat_z, sf_len = 5):
	"""
	cat_imgx, cat_imgy : BCG location in image frame
	cat_ra, cat_dec, cat_z : catalog information of image catalog
	ra_list, dec_list, z_lis : catalog information of which used to match to the image catalog
	"""
	lis_ra, lis_dec, lis_z = [], [], []

	com_s = '%.' + '%df' % sf_len

	origin_dex = []

	for kk in range( len(cat_ra) ):

		identi = ( com_s % cat_ra[kk] in ra_list) * (com_s % cat_dec[kk] in dec_list) #* (com_s % cat_z[kk] in z_lis)

		if identi == True:

			ndex_0 = ra_list.index( com_s % cat_ra[kk] )
			ndex_1 = dec_list.index( com_s % cat_dec[kk] )

			lis_ra.append( cat_ra[kk] )
			lis_dec.append( cat_dec[kk] )
			lis_z.append( cat_z[kk] )

			origin_dex.append( ndex_0 )

		else:
			continue

	match_ra = np.array( lis_ra )
	match_dec = np.array( lis_dec )
	match_z = np.array( lis_z )
	origin_dex = np.array( origin_dex )

	return match_ra, match_dec, match_z, origin_dex

#...cat information
ref_file = '/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits'

goal_data = fits.getdata( ref_file )
RA = np.array( goal_data.RA )
DEC = np.array( goal_data.DEC )
ID = np.array( goal_data.OBJID )

Z_phot = np.array( goal_data.Z_LAMBDA )

r_Mag_bcgs = np.array(goal_data.MODEL_MAG_R)
g_Mag_bcgs = np.array(goal_data.MODEL_MAG_G)
i_Mag_bcgs = np.array(goal_data.MODEL_MAG_I)

model_g2r = g_Mag_bcgs - r_Mag_bcgs
model_g2i = g_Mag_bcgs - i_Mag_bcgs

idz_lim = ( Z_phot >= 0.2 ) & ( Z_phot <= 0.3 )

lim_ra, lim_dec, lim_z = RA[ idz_lim ], DEC[ idz_lim ], Z_phot[ idz_lim ]
lim_ID = ID[ idz_lim ]

lim_g_mag = g_Mag_bcgs[ idz_lim ]
lim_r_mag = r_Mag_bcgs[ idz_lim ]
lim_i_mag = i_Mag_bcgs[ idz_lim ]
lim_g2r, lim_g2i = model_g2r[ idz_lim ], model_g2i[ idz_lim ]

cmag_path = '/home/xkchen/tmp_run/data_files/figs/BCG_M_bin_cModel_mag/'

### ... information match
# cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']
# fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']
# file_s = 'BCG_Mstar_bin'
# cat_path = '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_gri_common/'

cat_lis = [ 'low-age', 'hi-age' ]
fig_name = [ 'Low $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 'High $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $' ]
file_s = 'age_bin_fixed_BCG_M'
cat_path = '/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/age_bin/'

"""
for mm in range( 2 ):

	dat = pds.read_csv( cat_path + '%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[mm],)
	ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

	out_ra = [ '%.5f' % ll for ll in lim_ra ]
	out_dec = [ '%.5f' % ll for ll in lim_dec ]
	out_z = [ '%.5f' % ll for ll in lim_z ]

	sub_index = simple_match( out_ra, out_dec, out_z, ra, dec, z)[-1]

	match_ra, match_dec, match_z = lim_ra[ sub_index ], lim_dec[ sub_index ], lim_z[ sub_index ]
	match_g2r, match_g2i = lim_g2r[ sub_index ], lim_g2i[ sub_index ]

	match_ID = lim_ID[ sub_index ]

	match_g_mag = lim_g_mag[ sub_index ]
	match_r_mag = lim_r_mag[ sub_index ]
	match_i_mag = lim_i_mag[ sub_index ]

	g_mag_zref = match_g_mag + 10 * np.log10( (1 + z_ref) / (1 + z) )
	r_mag_zref = match_r_mag + 10 * np.log10( (1 + z_ref) / (1 + z) )
	i_mag_zref = match_i_mag + 10 * np.log10( (1 + z_ref) / (1 + z) )

	g2r_zref = g_mag_zref - r_mag_zref
	g2i_zref = g_mag_zref - i_mag_zref

	## ... save the catalog information
	keys = [ 'ra', 'dec', 'z', 'objID', 'g_mag', 'r_mag', 'i_mag', 'c_g2r', 'c_g2i', 'c_g2r_zref', 'c_g2i_zref', ]
	values = [ match_ra, match_dec, match_z, match_ID, match_g_mag, match_r_mag, match_i_mag, 
				match_g2r, match_g2i, g2r_zref, g2i_zref ]

	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/tmp_run/data_files/figs/%s_BCG-color.csv' % cat_lis[mm] )

	# plt.figure()
	# plt.plot( ra, dec, 'ro', alpha = 0.5,)
	# plt.plot( match_ra, match_dec, 'g*', alpha = 0.5,)
	# plt.show()
"""


# for mm in range( 2 ):

# 	## query
# 	pdat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/%s_BCG-color.csv' % cat_lis[mm] )
# 	p_ra, p_dec, p_z, p_ID = np.array( pdat['ra'] ), np.array( pdat['dec'] ), np.array( pdat['z'] ), np.array( pdat['objID'] )

# 	out_file = cmag_path + 'BCG_color_Z%.3f_ra%.3f_dec%.3f.txt'
# 	sql_color( p_z, p_ra, p_dec, p_ID, out_file ) ## SDSS data query


### ... Luminosity and color
for mm in range( 2 ):

	pdat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/%s_BCG-color.csv' % cat_lis[mm] )
	p_ra, p_dec, p_z, p_ID = np.array( pdat['ra'] ), np.array( pdat['dec'] ), np.array( pdat['z'] ), np.array( pdat['objID'] )
	p_g_mag, p_r_mag, p_i_mag = np.array( pdat['g_mag'] ), np.array( pdat['r_mag'] ), np.array( pdat['i_mag'] )

	Nz = len( p_ra )
	out_file = cmag_path + 'BCG_color_Z%.3f_ra%.3f_dec%.3f.txt'

	_r_mag_, _g_mag_, _i_mag_ = np.array([]), np.array([]), np.array([])
	_r_Mag_, _g_Mag_, _i_Mag_ = np.array([]), np.array([]), np.array([])

	for tt in range( Nz ):
		ra_g, dec_g, z_g = p_ra[tt], p_dec[tt], p_z[tt]

		cat = pds.read_csv( out_file % ( z_g, ra_g, dec_g ), skiprows = 1)
		sub_i_mag = cat['cModelMag_i'][0]
		sub_r_mag = cat['cModelMag_r'][0]
		sub_g_mag = cat['cModelMag_g'][0]

		Dl_g = Test_model.luminosity_distance( z_g ).value

		sub_i_Mag = sub_i_mag - 5 * np.log10( Dl_g * 1e6 ) + 5
		sub_r_Mag = sub_r_mag - 5 * np.log10( Dl_g * 1e6 ) + 5
		sub_g_Mag = sub_g_mag - 5 * np.log10( Dl_g * 1e6 ) + 5

		_r_mag_ = np.r_[ _r_mag_, sub_r_mag ]
		_r_Mag_ = np.r_[ _r_Mag_, sub_r_Mag ]

		_g_mag_ = np.r_[ _g_mag_, sub_g_mag ]
		_g_Mag_ = np.r_[ _g_Mag_, sub_g_Mag ]

		_i_mag_ = np.r_[ _i_mag_, sub_i_mag ]
		_i_Mag_ = np.r_[ _i_Mag_, sub_i_Mag ]

	keys = [ 'ra', 'dec', 'z', 'r_cmag', 'r_cMag', 'g_cmag', 'g_cMag', 'i_cmag', 'i_cMag']
	values = [ p_ra, p_dec, p_z, _r_mag_, _r_Mag_, _g_mag_, _g_Mag_, _i_mag_, _i_Mag_ ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/tmp_run/data_files/figs/%s_BCG_cmag.csv' % cat_lis[mm] )

	print(p_ra.shape)
