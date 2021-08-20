import matplotlib as mpl
# mpl.use('Agg')
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

### === ###
### query SDSS color
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
			p.objID, p.modelMag_g, p.modelMag_r, p.modelMag_i,
			p.modelMagErr_g, p.modelMagErr_r, p.modelMagErr_i,
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

### === ### color based on SDSS photo-SB profile
def sdss_photo_pros():

	home = '/home/xkchen/data/SDSS/'
	load = '/home/xkchen/fig_tmp/'

	## fixed BCG-M sub-samples
	cat_lis = [ 'low-age', 'hi-age' ]
	fig_name = [ 'younger', 'older' ]

	color_s = [ 'r', 'g', 'b' ]
	line_c = [ 'b', 'r' ]

	band_str = band[ rank ]

	if band_str == 'r':
		out_ra = [ '164.740', '141.265', ]
		out_dec = [ '11.637', '11.376', ]
		out_z = [ '0.298', '0.288', ]

	if band_str == 'g':
		out_ra = [ '206.511', '141.265', '236.438', ]
		out_dec = [ '38.731', '11.376', '1.767', ]
		out_z = [ '0.295', '0.288', '0.272', ]

	N_samples = 30
	r_bins = np.logspace(0, 2.48, 25) # unit : kpc

	for ll in range( 2 ):

		## fixed BCG-M sub-samples
		dat = pds.read_csv( load + 'bcg_M_simi_cat/%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[ll], band_str),)
		ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

		# .. directly mean of sample BCG profiles
		pros_file = home + 'photo_files/BCG_profile/BCG_prof_Z%.3f_ra%.3f_dec%.3f.txt'
		out_file = '/home/xkchen/%s_%s-band_aveg_BCG_photo-SB_pros.csv' % (cat_lis[ll], band_str)
		BCG_SB_pros_func( band_str, z, ra, dec, pros_file, z_ref, out_file, r_bins)

		print('N_sample = ', len(ra),)
		print('band = %s' % band_str,)

		## also divid sub-samples
		zN = len( ra )
		id_arr = np.arange(0, zN, 1)
		id_group = id_arr % N_samples

		lis_ra, lis_dec, lis_z = [], [], []
		lis_x, lis_y = [], []

		## sub-sample
		for nn in range( N_samples ):

			id_xbin = np.where( id_group == nn )[0]

			lis_ra.append( ra[ id_xbin ] )
			lis_dec.append( dec[ id_xbin ] )
			lis_z.append( z[ id_xbin ] )

		## jackknife sub-sample
		for nn in range( N_samples ):

			id_arry = np.linspace( 0, N_samples - 1, N_samples )
			id_arry = id_arry.astype( int )
			jack_id = list( id_arry )
			jack_id.remove( jack_id[nn] )
			jack_id = np.array( jack_id )

			set_ra, set_dec, set_z = np.array([]), np.array([]), np.array([])

			for oo in ( jack_id ):
				set_ra = np.r_[ set_ra, lis_ra[oo] ]
				set_dec = np.r_[ set_dec, lis_dec[oo] ]
				set_z = np.r_[ set_z, lis_z[oo] ]

			pros_file = home + 'photo_files/BCG_profile/BCG_prof_Z%.3f_ra%.3f_dec%.3f.txt'
			out_file = '/home/xkchen/figs/%s_%s-band_jack-sub-%d_BCG_photo-SB_pros.csv' % (cat_lis[ll], band_str, nn)

			BCG_SB_pros_func( band_str, set_z, set_ra, set_dec, pros_file, z_ref, out_file, r_bins)

		## mean of jackknife sample
		tmp_r, tmp_sb = [], []
		for nn in range( N_samples ):

			pro_dat = pds.read_csv( '/home/xkchen/figs/%s_%s-band_jack-sub-%d_BCG_photo-SB_pros.csv' % (cat_lis[ll], band_str, nn),)

			tt_r, tt_sb = np.array( pro_dat['R_ref'] ), np.array( pro_dat['SB_fdens'] )

			tmp_r.append( tt_r )
			tmp_sb.append( tt_sb )

		mean_R, mean_sb, mean_sb_err, lim_R = arr_jack_func( tmp_sb, tmp_r, N_samples)

		keys = [ 'R', 'aveg_sb', 'aveg_sb_err' ]
		values = [ mean_R, mean_sb, mean_sb_err ]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( '/home/xkchen/figs/%s_%s-band_Mean-jack_BCG_photo-SB_pros.csv' % (cat_lis[ll], band_str),)

		print( '%s, %s band' % (cat_lis[ll], band_str), )

	return

# sdss_photo_pros()

### === ### local
ref_file = '/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits'

goal_data = fits.getdata( ref_file )

RA = np.array( goal_data.RA )
DEC = np.array( goal_data.DEC )
ID = np.array( goal_data.OBJID )

Z_ref = np.array( goal_data.Z_LAMBDA )

r_Mag_bcgs = np.array(goal_data.MODEL_MAG_R)
g_Mag_bcgs = np.array(goal_data.MODEL_MAG_G)
i_Mag_bcgs = np.array(goal_data.MODEL_MAG_I)

model_g2r = g_Mag_bcgs - r_Mag_bcgs
model_g2i = g_Mag_bcgs - i_Mag_bcgs

idx_lim = ( Z_ref >= 0.2 ) & ( Z_ref <= 0.3 )

lim_ra, lim_dec, lim_z = RA[ idx_lim ], DEC[ idx_lim ], Z_ref[ idx_lim ]
lim_ID = ID[ idx_lim ]

lim_g2r, lim_g2i = model_g2r[ idx_lim ], model_g2i[ idx_lim ]

## .. cluster catalog
cat_lis = [ 'low-age', 'hi-age' ]
fig_name = [ 'younger', 'older' ]

color_s = [ 'r', 'g', 'b' ]
line_c = [ 'b', 'r' ]
line_s = [ '--', '-' ]

ref_path = '/home/xkchen/mywork/ICL/code/BCG_M_based_cat/age_bin_color_test/'
cat_path = '/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/age_bin/'

'''
### color based on redMapper catalog
for mm in range( 2 ):

	dat = pds.read_csv( cat_path + '%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[mm]),)
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
	out_data.to_csv( ref_path + '%s_BCG-color.csv' % cat_lis[mm] )

	# plt.figure()
	# plt.plot( ra, dec, 'ro', alpha = 0.5,)
	# plt.plot( match_ra, match_dec, 'g*', alpha = 0.5,)
	# plt.show()
'''

tmp_g2r = []
tmp_g2i = []

sql_g2r = []
sql_g2i = []

for mm in range( 2 ):

	## query
	pdat = pds.read_csv( ref_path + '%s_BCG-color.csv' % cat_lis[mm] )
	p_ra, p_dec, p_z, p_ID = np.array( pdat['ra'] ), np.array( pdat['dec'] ), np.array( pdat['z'] ), np.array( pdat['objID'] )
	p_g2r, p_g2i = np.array( pdat['c_g2r'] ), np.array( pdat['c_g2i'] )

	out_file = ref_path + 'BCG_mags/BCG_color_Z%.3f_ra%.3f_dec%.3f.txt'
	sql_color( p_z, p_ra, p_dec, p_ID, out_file ) ## SDSS data query

print('done !')
raise
'''
	## readout
	Ns = len( p_ra )

	dt_g2r, dt_g2i = np.array([]), np.array([])

	for jj in range( Ns ):

		ra_g, dec_g, z_g = p_ra[jj], p_dec[jj], p_z[jj]

		cat = pds.read_csv( out_file % ( z_g, ra_g, dec_g ), skiprows = 1)
		sub_g_mag = cat['modelMag_g'][0]
		sub_r_mag = cat['modelMag_r'][0]
		sub_i_mag = cat['modelMag_i'][0]

		dt_g2r = np.r_[ dt_g2r, sub_g_mag - sub_r_mag ]
		dt_g2i = np.r_[ dt_g2i, sub_g_mag - sub_i_mag ]

	tmp_g2r.append( p_g2r )
	tmp_g2i.append( p_g2i )

	sql_g2r.append( dt_g2r )
	sql_g2i.append( dt_g2i )

bins_0 = np.linspace(0.5, 2.5, 50)
bins_1 = np.linspace(0.5, 2.5, 100)

fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

ax.hist( tmp_g2r[0], bins = bins_0, density = True, histtype = 'step', color = line_c[0], alpha = 0.5, label = fig_name[0] + ', deredden',)
ax.axvline( x = np.mean( tmp_g2r[0] ), ls = '--', color = line_c[0], alpha = 0.5, label = 'mean',)
ax.axvline( x = np.median( tmp_g2r[0] ), ls = ':', color = line_c[0], alpha = 0.5, label = 'median',)

ax.hist( tmp_g2r[1], bins = bins_1, density = True, histtype = 'step', color = line_c[1], alpha = 0.5, label = fig_name[1] + ', deredden',)
ax.axvline( x = np.mean( tmp_g2r[1] ), ls = '--', color = line_c[1], alpha = 0.5,)
ax.axvline( x = np.median( tmp_g2r[1] ), ls = ':', color = line_c[1], alpha = 0.5,)


ax.hist( sql_g2r[0], bins = bins_0, density = True, histtype = 'step', color = 'g', alpha = 0.5, label = fig_name[0],)
ax.axvline( x = np.mean( sql_g2r[0] ), ls = '--', color = 'g', alpha = 0.5,)
ax.axvline( x = np.median( sql_g2r[0] ), ls = ':', color = 'g', alpha = 0.5,)

ax.hist( sql_g2r[1], bins = bins_1, density = True, histtype = 'step', color = 'm', alpha = 0.5, label = fig_name[1],)
ax.axvline( x = np.mean( sql_g2r[1] ), ls = '--', color = 'm', alpha = 0.5,)
ax.axvline( x = np.median( sql_g2r[1] ), ls = ':', color = 'm', alpha = 0.5,)


ax.set_xlabel( 'g-r of BCGs', fontsize = 15,)
ax.set_ylabel( 'pdf', fontsize = 15,)

ax.set_xlim( 1.0, 2.5 )

ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
ax.legend( loc = 1, frameon = False, fontsize = 13,)

plt.savefig('/home/xkchen/age-bin_BCG_g2r_compare.png', dpi = 300)
plt.close()


bins_0 = np.linspace(1.0, 3.0, 50)
bins_1 = np.linspace(1.0, 3.0, 100)

fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

ax.hist( tmp_g2i[0], bins = 50, density = True, histtype = 'step', color = line_c[0], alpha = 0.5, label = fig_name[0] + ', deredden',)
ax.axvline( x = np.mean( tmp_g2i[0] ), ls = '--', color = line_c[0], alpha = 0.5,)
ax.axvline( x = np.median( tmp_g2i[0] ), ls = ':', color = line_c[0], alpha = 0.5,)

ax.hist( tmp_g2i[1], bins = 100, density = True, histtype = 'step', color = line_c[1], alpha = 0.5, label = fig_name[1] + ', deredden',)
ax.axvline( x = np.mean( tmp_g2i[1] ), ls = '--', color = line_c[1], alpha = 0.5,)
ax.axvline( x = np.median( tmp_g2i[1] ), ls = ':', color = line_c[1], alpha = 0.5,)


ax.hist( sql_g2i[0], bins = 50, density = True, histtype = 'step', color = 'g', alpha = 0.5, label = fig_name[0],)
ax.axvline( x = np.mean( sql_g2i[0] ), ls = '--', color = 'g', alpha = 0.5, label = 'mean',)
ax.axvline( x = np.median( sql_g2i[0] ), ls = ':', color = 'g', alpha = 0.5, label = 'median',)

ax.hist( sql_g2i[1], bins = 100, density = True, histtype = 'step', color = 'm', alpha = 0.5, label = fig_name[1],)
ax.axvline( x = np.mean( sql_g2i[1] ), ls = '--', color = 'm', alpha = 0.5,)
ax.axvline( x = np.median( sql_g2i[1] ), ls = ':', color = 'm', alpha = 0.5,)

ax.set_xlabel( 'g-i of BCGs', fontsize = 15,)
ax.set_ylabel( 'pdf', fontsize = 15,)

ax.set_xlim( 0.7, 3.0 )

ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
ax.legend( loc = 2, frameon = False, fontsize = 13,)

plt.savefig('/home/xkchen/age-bin_BCG_g2i_compare.png', dpi = 300)
plt.close()
'''
