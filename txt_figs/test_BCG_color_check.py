"""
record the BCG magnitude information (deredden or not)
and dust map on location of BCGs
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import time
import h5py
import numpy as np
import astropy.io.fits as fits

import mechanize
import pandas as pds
from io import StringIO
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord

from BCG_SB_pro_stack import BCG_SB_pros_func
from fig_out_module import arr_jack_func
from fig_out_module import color_func

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

### constant
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']

### === ### query SDSS color
def sql_color( cat_z, cat_ra, cat_dec, cat_ID, out_file ):

	import mechanize
	from io import StringIO

	url = 'http://skyserver.sdss.org/dr12/en/tools/search/sql.aspx'

	Nz = len( cat_z )

	for kk in range( Nz ):

		z_g = cat_z[kk]
		ra_g = cat_ra[kk]
		dec_g = cat_dec[kk]

		data_set = """
		SELECT
			p.objID, p.modelMag_g, p.modelMag_r, p.modelMag_i,
			p.modelMagErr_g, p.modelMagErr_r, p.modelMagErr_i,
			p.cModelMag_g, p.cModelMag_r, p.cModelMag_i,

			p.deVMag_g, p.deVMag_r, p.deVMag_i,
			p.expMag_g, p.expMag_r, p.expMag_i,
			p.deVRad_g, p.deVRad_r, p.deVRad_i,
			p.expRad_g, p.expRad_r, p.expRad_i,
			p.dered_g, p.dered_r, p.dered_i

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


### === ### local
ref_file = '/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits'
goal_data = fits.getdata( ref_file )

RA = np.array( goal_data.RA )
DEC = np.array( goal_data.DEC )
ID = np.array( goal_data.OBJID )

rich = np.array( goal_data.LAMBDA )
Z_photo = np.array( goal_data.Z_LAMBDA )

#. deredden magnitude
r_mag_bcgs = np.array(goal_data.MODEL_MAG_R)
g_mag_bcgs = np.array(goal_data.MODEL_MAG_G)
i_mag_bcgs = np.array(goal_data.MODEL_MAG_I)

model_g2r = g_mag_bcgs - r_mag_bcgs
model_g2i = g_mag_bcgs - i_mag_bcgs
model_r2i = r_mag_bcgs - i_mag_bcgs

idx_lim = ( Z_photo >= 0.2 ) & ( Z_photo <= 0.3 )

lim_ra, lim_dec, lim_z = RA[ idx_lim ], DEC[ idx_lim ], Z_photo[ idx_lim ]
lim_ID = ID[ idx_lim ]

lim_g_mag, lim_r_mag, lim_i_mag = g_mag_bcgs[ idx_lim ], r_mag_bcgs[ idx_lim ], i_mag_bcgs[ idx_lim ]
lim_g2r, lim_g2i = model_g2r[ idx_lim ], model_g2i[ idx_lim ]

#... sample read and match
# cat_lis = [ 'low-age', 'hi-age' ]
cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']
out_path = '/home/xkchen/figs/BCG_mags/'

"""
for mm in range( 2 ):

	for kk in range( 3 ):

		# dat = pds.read_csv( '/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/age_bin/' + 
		# 					'%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[mm], band[kk]),)

		dat = pds.read_csv( '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_match/' + 
							'%s_%s-band_photo-z-match_BCG-pos_cat.csv' % (cat_lis[mm], band[kk]),)

		ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

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
		keys = [ 'ra', 'dec', 'z', 'objID', 'g_mag', 'r_mag', 'i_mag', 
				 'c_g2r', 'c_g2i', 'c_g2r_zref', 'c_g2i_zref', ]
		values = [ match_ra, match_dec, match_z, match_ID, match_g_mag, match_r_mag, match_i_mag, 
					match_g2r, match_g2i, g2r_zref, g2i_zref ]

		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_path + '%s_%s-band_BCG-color.csv' % (cat_lis[mm], band[kk]),)

	# plt.figure()
	# plt.plot( ra, dec, 'ro', alpha = 0.5,)
	# plt.plot( match_ra, match_dec, 'g*', alpha = 0.5,)
	# plt.show()

"""

### === ### SDSS image match
"""
cat = pds.read_csv( '/home/xkchen/mywork/ICL/data/cat_z_form/clslowz_z0.17-0.30_bc03_cat.csv' )
ext_ra, ext_dec, ext_z = np.array(cat['ra']), np.array(cat['dec']), np.array(cat['z'])

#. match sample with image catalog
out_ra = [ '%.5f' % ll for ll in RA ]
out_dec = [ '%.5f' % ll for ll in DEC ]
out_z = [ '%.5f' % ll for ll in Z_photo ]

sub_index = simple_match( out_ra, out_dec, out_z, ext_ra, ext_dec, ext_z)[-1]

match_ra, match_dec, match_z = RA[ sub_index ], DEC[ sub_index ], Z_photo[ sub_index ]
match_ID, match_rich = ID[ sub_index ], rich[ sub_index ]

match_gmag, match_rmag, match_imag = g_mag_bcgs[ sub_index ], r_mag_bcgs[ sub_index ], i_mag_bcgs[ sub_index ]
match_g2r, match_g2i, match_r2i = model_g2r[ sub_index ], model_g2i[ sub_index ], model_r2i[ sub_index ]


keys = [ 'ra', 'dec', 'z', 'objID', 'rich', 'mod_r', 'mod_g', 'mod_i', 'g-r', 'g-i', 'r-i' ]
values = [ match_ra, match_dec, match_z, match_ID, match_rich, match_rmag, match_gmag, match_imag, match_g2r, match_g2i, match_r2i ]

fill = dict(zip( keys, values) )
out_data = pds.DataFrame( fill )
# out_data.to_csv( out_path + 'clslowz_z0.17-0.30_img-cat_match.csv',)
out_data.to_csv( out_path + 'clslowz_z0.17-0.30_img-cat_dered_color.csv',)
"""

"""
### === ### SDSS catalog query
cat = pds.read_csv('/home/xkchen/figs/BCG_mags/clslowz_z0.17-0.30_img-cat_match.csv' )
ext_ra, ext_dec, ext_z = np.array(cat['ra']), np.array(cat['dec']), np.array(cat['z'])
ext_ID = np.array( cat['objID'] )
out_file = '/home/xkchen/figs/BCG_mags/BCG_mag_cat/BCG_mags_Z%.3f_ra%.3f_dec%.3f.txt'
# sql_color( ext_z, ext_ra, ext_dec, ext_ID, out_file )


tmp_gr, tmp_gi, tmp_ri = [], [], []
tmp_g_mag, tmp_r_mag, tmp_i_mag = [], [], []
cMod_mag_i = []

err_dex = []

Ns = len( ext_ra )

for kk in range( Ns ):
	
	ra_g, dec_g, z_g = ext_ra[ kk ], ext_dec[ kk ], ext_z[ kk ] 

	try:
		cat = pds.read_csv('/home/xkchen/figs/BCG_mags/BCG_mag_cat/' + 
							'BCG_mags_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g), skiprows = 1)
		_kk_g_mag = np.array( cat['modelMag_g'] )[0]
		_kk_r_mag = np.array( cat['modelMag_r'] )[0]
		_kk_i_mag = np.array( cat['modelMag_i'] )[0]

		cMod_mag_i.append( np.array( cat['cModelMag_i'] )[0] )

		tmp_gr.append( _kk_g_mag - _kk_r_mag )
		tmp_gi.append( _kk_g_mag - _kk_i_mag )
		tmp_ri.append( _kk_r_mag - _kk_i_mag )

		tmp_g_mag.append( _kk_g_mag )
		tmp_r_mag.append( _kk_r_mag )
		tmp_i_mag.append( _kk_i_mag )

	except:

		err_dex.append( kk )

tmp_gr, tmp_gi, tmp_ri = np.array( tmp_gr ), np.array( tmp_gi ), np.array( tmp_ri )
tmp_g_mag, tmp_r_mag, tmp_i_mag = np.array( tmp_g_mag ), np.array( tmp_r_mag ), np.array( tmp_i_mag )
cMod_mag_i = np.array( cMod_mag_i )


keys = [ 'ra', 'dec', 'z', 'objID', 'mod_r', 'mod_g', 'mod_i', 'g-r', 'g-i', 'r-i', 'cMod_mag_i']
values = [ ext_ra, ext_dec, ext_z, ext_ID, tmp_r_mag, tmp_g_mag, tmp_i_mag, tmp_gr, tmp_gi, tmp_ri, cMod_mag_i ]

fill = dict(zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( out_path + 'clslowz_z0.17-0.30_img-cat_non-dered_color.csv',)

"""

### === ### (dust map query) extinction on BCGs match
re_dat = pds.read_csv('/home/xkchen/figs/BCG_mags/clslowz_z0.17-0.30_img-cat_dust_value.csv')
re_ra, re_dec, re_z = np.array( re_dat['ra'] ), np.array( re_dat['dec'] ), np.array( re_dat['z'] )
re_Al_r, re_Al_g, re_Al_i = np.array( re_dat['Al_r'] ), np.array( re_dat['Al_g'] ), np.array( re_dat['Al_i'] )

re_coord = SkyCoord( ra = re_ra * U.deg, dec = re_dec * U.deg, )

"""
### === the three contral subsamples
# cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
# p_path = '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_gri_common/'

# cat_lis = [ 'low-rich', 'hi-rich' ]
# p_path = '/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/rich_bin/'

cat_lis = [ 'low-age', 'hi-age' ]
p_path = '/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/age_bin/'


lo_dat = pds.read_csv( p_path + '%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[0] )
lo_ra, lo_dec, lo_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )

lo_coord = SkyCoord( ra = lo_ra * U.deg, dec = lo_dec * U.deg, )

idx, sep, d3d = lo_coord.match_to_catalog_sky( re_coord )
id_lim = sep.value < 2.7e-4

mp_ra, mp_dec, mp_z = re_ra[ idx[ id_lim ] ], re_dec[ idx[ id_lim ] ], re_z[ idx[ id_lim ] ]
mp_Al_r, mp_Al_g, mp_Al_i = re_Al_r[ idx[ id_lim ] ], re_Al_g[ idx[ id_lim ] ], re_Al_i[ idx[ id_lim ] ]
lim_ra, lim_dec, lim_z = lo_ra[ id_lim ], lo_dec[ id_lim ], lo_z[ id_lim ]

keys = [ 'ra', 'dec', 'z', 'Al_r', 'Al_g', 'Al_i' ]
values = [ lim_ra, lim_dec, lim_z, mp_Al_r, mp_Al_g, mp_Al_i ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/%s_photo-z-match_rgi-common_cat_dust-value.csv' % cat_lis[0] )


hi_dat = pds.read_csv( p_path + '%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[1] )
hi_ra, hi_dec, hi_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )

hi_coord = SkyCoord( ra = hi_ra * U.deg, dec = hi_dec * U.deg, )

idx, sep, d3d = hi_coord.match_to_catalog_sky( re_coord )
id_lim = sep.value < 2.7e-4

mp_ra, mp_dec, mp_z = re_ra[ idx[ id_lim ] ], re_dec[ idx[ id_lim ] ], re_z[ idx[ id_lim ] ]
mp_Al_r, mp_Al_g, mp_Al_i = re_Al_r[ idx[ id_lim ] ], re_Al_g[ idx[ id_lim ] ], re_Al_i[ idx[ id_lim ] ]
lim_ra, lim_dec, lim_z = hi_ra[ id_lim ], hi_dec[ id_lim ], hi_z[ id_lim ]

keys = [ 'ra', 'dec', 'z', 'Al_r', 'Al_g', 'Al_i' ]
values = [ lim_ra, lim_dec, lim_z, mp_Al_r, mp_Al_g, mp_Al_i ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/%s_photo-z-match_rgi-common_cat_dust-value.csv' % cat_lis[1] )
"""

"""
### ===  all cluster sample
r_dat = pds.read_csv('/home/xkchen/figs/sat_color/aveg_clust_EBV/BCG-Mstar_bin_r-band_dust_value.csv')
r_ra, r_dec, r_z = np.array( r_dat['ra'] ), np.array( r_dat['dec'] ), np.array( r_dat['z'] )
r_coord = SkyCoord( ra = r_ra * U.deg, dec = r_dec * U.deg )

g_dat = pds.read_csv('/home/xkchen/figs/sat_color/aveg_clust_EBV/BCG-Mstar_bin_g-band_dust_value.csv')
g_ra, g_dec, g_z = np.array( g_dat['ra'] ), np.array( g_dat['dec'] ), np.array( g_dat['z'] )
g_coord = SkyCoord( ra = g_ra * U.deg, dec = g_dec * U.deg )

i_dat = pds.read_csv('/home/xkchen/figs/sat_color/aveg_clust_EBV/BCG-Mstar_bin_i-band_dust_value.csv')
i_ra, i_dec, i_z = np.array( i_dat['ra'] ), np.array( i_dat['dec'] ), np.array( i_dat['z'] )
i_coord = SkyCoord( ra = i_ra * U.deg, dec = i_dec * U.deg )

#. gri cross match
idx, sep, d3d = r_coord.match_to_catalog_sky( g_coord )
id_lim = sep.value < 2.7e-4
mp_ra, mp_dec, mp_z = g_ra[ idx[ id_lim ] ], g_dec[ idx[ id_lim ] ], g_z[ idx[ id_lim ] ]

medi_coord = SkyCoord( mp_ra * U.deg, mp_dec * U.deg, )

idx, sep, d3d = medi_coord.match_to_catalog_sky( i_coord )
id_lim = sep.value < 2.7e-4
mm_ra, mm_dec, mm_z = i_ra[ idx[ id_lim ] ], i_dec[ idx[ id_lim ] ], i_z[ idx[ id_lim ] ]

#. match to Al table
mm_coord = SkyCoord( mm_ra * U.deg, mm_dec * U.deg )

idx, sep, d3d = mm_coord.match_to_catalog_sky( re_coord )
id_lim = sep.value < 2.7e-4

tt_ra, tt_dec, tt_z = re_ra[ idx[ id_lim ] ], re_dec[ idx[ id_lim ] ], re_z[ idx[ id_lim ] ]
tt_Al_r, tt_Al_g, tt_Al_i = re_Al_r[ idx[ id_lim ] ], re_Al_g[ idx[ id_lim ] ], re_Al_i[ idx[ id_lim ] ]
lim_ra, lim_dec, lim_z = mm_ra[ id_lim ], mm_dec[ id_lim ], mm_z[ id_lim ]

keys = [ 'ra', 'dec', 'z', 'Al_r', 'Al_g', 'Al_i' ]
values = [ lim_ra, lim_dec, lim_z, tt_Al_r, tt_Al_g, tt_Al_i ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/tot-BCG-star-Mass_dust-value.csv' )
"""

### === aperture_M catalog
# cat_lis = ['low-lgM20', 'hi-lgM20']
# p_path = '/home/xkchen/figs/BCG_aper_M/uniform_M2L_cat/'

# lo_dat = pds.read_csv( p_path + 'photo-z_match_%s_gri-common_cluster_cat.csv' % cat_lis[0] )
# lo_ra, lo_dec, lo_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )
# lo_coord = SkyCoord( ra = lo_ra * U.deg, dec = lo_dec * U.deg, )

# hi_dat = pds.read_csv( p_path + 'photo-z_match_%s_gri-common_cluster_cat.csv' % cat_lis[1] )
# hi_ra, hi_dec, hi_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )
# hi_coord = SkyCoord( ra = hi_ra * U.deg, dec = hi_dec * U.deg, )


# cat_lis = ['low-lgM10', 'hi-lgM10']
# p_path = '/home/xkchen/figs/BCG_aper_M/uniform_M2L_cat/'

# lo_dat = pds.read_csv( p_path + 'photo-z_match_%s_gri-common_cluster_cat.csv' % cat_lis[0] )
# lo_ra, lo_dec, lo_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )
# lo_coord = SkyCoord( ra = lo_ra * U.deg, dec = lo_dec * U.deg, )

# hi_dat = pds.read_csv( p_path + 'photo-z_match_%s_gri-common_cluster_cat.csv' % cat_lis[1] )
# hi_ra, hi_dec, hi_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )
# hi_coord = SkyCoord( ra = hi_ra * U.deg, dec = hi_dec * U.deg, )


# idx, sep, d3d = lo_coord.match_to_catalog_sky( re_coord )
# id_lim = sep.value < 2.7e-4

# mp_ra, mp_dec, mp_z = re_ra[ idx[ id_lim ] ], re_dec[ idx[ id_lim ] ], re_z[ idx[ id_lim ] ]
# mp_Al_r, mp_Al_g, mp_Al_i = re_Al_r[ idx[ id_lim ] ], re_Al_g[ idx[ id_lim ] ], re_Al_i[ idx[ id_lim ] ]
# lim_ra, lim_dec, lim_z = lo_ra[ id_lim ], lo_dec[ id_lim ], lo_z[ id_lim ]

# keys = [ 'ra', 'dec', 'z', 'Al_r', 'Al_g', 'Al_i' ]
# values = [ lim_ra, lim_dec, lim_z, mp_Al_r, mp_Al_g, mp_Al_i ]
# fill = dict(zip( keys, values) )
# out_data = pds.DataFrame( fill )
# out_data.to_csv( '/home/xkchen/%s_photo-z-match_rgi-common_cat_dust-value.csv' % cat_lis[0] )


# idx, sep, d3d = hi_coord.match_to_catalog_sky( re_coord )
# id_lim = sep.value < 2.7e-4

# mp_ra, mp_dec, mp_z = re_ra[ idx[ id_lim ] ], re_dec[ idx[ id_lim ] ], re_z[ idx[ id_lim ] ]
# mp_Al_r, mp_Al_g, mp_Al_i = re_Al_r[ idx[ id_lim ] ], re_Al_g[ idx[ id_lim ] ], re_Al_i[ idx[ id_lim ] ]
# lim_ra, lim_dec, lim_z = hi_ra[ id_lim ], hi_dec[ id_lim ], hi_z[ id_lim ]

# keys = [ 'ra', 'dec', 'z', 'Al_r', 'Al_g', 'Al_i' ]
# values = [ lim_ra, lim_dec, lim_z, mp_Al_r, mp_Al_g, mp_Al_i ]
# fill = dict(zip( keys, values) )
# out_data = pds.DataFrame( fill )
# out_data.to_csv( '/home/xkchen/%s_photo-z-match_rgi-common_cat_dust-value.csv' % cat_lis[1] )


### === P_cen cut catalog
# cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
cat_lis = ['low-lgM20', 'hi-lgM20']

p_path = '/home/xkchen/figs/Pcen_cut/cat/'

lo_dat = pds.read_csv( p_path + '%s_gri-common_P-cen_lim_cat.csv' % cat_lis[0] )
lo_ra, lo_dec, lo_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )
lo_coord = SkyCoord( ra = lo_ra * U.deg, dec = lo_dec * U.deg, )

hi_dat = pds.read_csv( p_path + '%s_gri-common_P-cen_lim_cat.csv' % cat_lis[1] )
hi_ra, hi_dec, hi_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )
hi_coord = SkyCoord( ra = hi_ra * U.deg, dec = hi_dec * U.deg, )


idx, sep, d3d = lo_coord.match_to_catalog_sky( re_coord )
id_lim = sep.value < 2.7e-4

mp_ra, mp_dec, mp_z = re_ra[ idx[ id_lim ] ], re_dec[ idx[ id_lim ] ], re_z[ idx[ id_lim ] ]
mp_Al_r, mp_Al_g, mp_Al_i = re_Al_r[ idx[ id_lim ] ], re_Al_g[ idx[ id_lim ] ], re_Al_i[ idx[ id_lim ] ]
lim_ra, lim_dec, lim_z = lo_ra[ id_lim ], lo_dec[ id_lim ], lo_z[ id_lim ]

keys = [ 'ra', 'dec', 'z', 'Al_r', 'Al_g', 'Al_i' ]
values = [ lim_ra, lim_dec, lim_z, mp_Al_r, mp_Al_g, mp_Al_i ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/%s_photo-z-match_Pcen-lim_cat_dust-value.csv' % cat_lis[0] )


idx, sep, d3d = hi_coord.match_to_catalog_sky( re_coord )
id_lim = sep.value < 2.7e-4

mp_ra, mp_dec, mp_z = re_ra[ idx[ id_lim ] ], re_dec[ idx[ id_lim ] ], re_z[ idx[ id_lim ] ]
mp_Al_r, mp_Al_g, mp_Al_i = re_Al_r[ idx[ id_lim ] ], re_Al_g[ idx[ id_lim ] ], re_Al_i[ idx[ id_lim ] ]
lim_ra, lim_dec, lim_z = hi_ra[ id_lim ], hi_dec[ id_lim ], hi_z[ id_lim ]

keys = [ 'ra', 'dec', 'z', 'Al_r', 'Al_g', 'Al_i' ]
values = [ lim_ra, lim_dec, lim_z, mp_Al_r, mp_Al_g, mp_Al_i ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/%s_photo-z-match_Pcen-lim_cat_dust-value.csv' % cat_lis[1] )

