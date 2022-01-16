import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import time
import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C

from scipy import optimize
from scipy import interpolate as interp
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord

#.
from img_pre_selection import gri_common_cat_func

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


### === ### catalog compare
"""
dat_0 = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/low_star-Mass_cat.csv')
ra_0, dec_0, z_0 = np.array( dat_0['ra'] ), np.array( dat_0['dec'] ), np.array( dat_0['z'] )
lgM_bcg_0 = np.array( dat_0['lg_Mass'] )

dat_1 = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/high_star-Mass_cat.csv')
ra_1, dec_1, z_1 = np.array( dat_1['ra'] ), np.array( dat_1['dec'] ), np.array( dat_1['z'] )
lgM_bcg_1 = np.array( dat_1['lg_Mass'] )

cp_ra = np.r_[ ra_0, ra_1 ]
cp_dec = np.r_[ dec_0, dec_1 ]
cp_z = np.r_[ z_0, z_1 ]
cp_lgM = np.r_[ lgM_bcg_0, lgM_bcg_1 ]

id_vx = ( cp_z >= 0.2) & (cp_z <= 0.3)


##. extend BCG Mstar sample
extend_path = '/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/'
txt_file = [ 'lowmstarfixedlam_xkchen.dat', 'highmstarfixedlam_xkchen.dat']
put_keys = [ 'ra', 'dec', 'z', 'Lambda', 'LgMstar', 'Z_form', 'redMap_ID']
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

kk_ra, kk_dec, kk_z = [], [], []
kk_lgM = []

for tt in range( 2 ):

	dat = np.loadtxt( extend_path + txt_file[ tt ] )

	tmp_values = []

	for ii in range( len(put_keys) ):
		tmp_values.append( dat[:, ii] )

	kk_ra.append( dat[:,0] )
	kk_dec.append( dat[:,1] )
	kk_z.append( dat[:,2] )
	kk_lgM.append( dat[:,4] )

	keys = put_keys
	values = tmp_values

	print( cat_lis[tt] )
	print( txt_file[tt] )

	fill = dict( zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( extend_path + '%s_fixed_rich_cluster.csv' % cat_lis[ tt ],)

et_ra = np.r_[ kk_ra[0], kk_ra[1] ]
et_dec = np.r_[ kk_dec[0], kk_dec[1] ]
et_z = np.r_[ kk_z[0], kk_z[1] ]
et_lgM = np.r_[ kk_lgM[0], kk_lgM[1] ]

cp_coord = SkyCoord( ra = cp_ra * U.deg, dec = cp_dec * U.deg, )

et_coord = SkyCoord( ra = et_ra * U.deg, dec = et_dec * U.deg, )

idx, sep, d3d = cp_coord.match_to_catalog_sky( et_coord )
id_lim = sep.value < 2.7e-4
mp_ra, mp_dec, mp_z = et_ra[ idx[ id_lim ] ], et_dec[ idx[ id_lim ] ], et_z[ idx[ id_lim ] ]
mp_lgM = et_lgM[ idx[ id_lim ] ]


plt.figure()
plt.plot( cp_ra[id_vx], cp_dec[id_vx], 'ro', alpha = 0.5,)
plt.plot( mp_ra, mp_dec, 'g*', alpha = 0.5,)
plt.plot( et_ra, et_dec, 'bs', alpha = 0.5,)
plt.show()

raise
"""

### === match to image catalog
load = '/home/xkchen/mywork/ICL/data/photo_cat/'
extend_path = '/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/'
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

"""
for ll in range( 2 ):

	dat = pds.read_csv( extend_path + '%s_fixed_rich_cluster.csv' % cat_lis[ ll ] )
	ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
	rich, lg_Mbcg = np.array( dat['Lambda'] ), np.array( dat['LgMstar'] )
	z_form = np.array( dat['Z_form'] )

	rec_coord = SkyCoord( ra * U.deg, dec * U.deg,)

	for kk in range( 3 ):

		pdat = pds.read_csv( load + 'photo-z_%s-band_tot_remain_cat_set_200-grid_6.0-sigma.csv' % band[ kk ],)
		p_ra, p_dec, p_z = np.array(pdat['ra']), np.array(pdat['dec']), np.array(pdat['z'])
		bcg_x, bcg_y = np.array(pdat['bcg_x']), np.array(pdat['bcg_y'])

		sub_coord = SkyCoord( p_ra * U.deg, p_dec * U.deg )

		idx, sep, d3d = sub_coord.match_to_catalog_sky( rec_coord )
		id_lim = sep.value < 2.7e-4

		mp_ra, mp_dec, mp_z = ra[ idx[ id_lim ] ], dec[ idx[ id_lim ] ], z[ idx[ id_lim ] ]
		mp_rich, mp_lgMstar = rich[ idx[ id_lim ] ], lg_Mbcg[ idx[ id_lim ] ]
		mp_zform = z_form[ idx[ id_lim ] ]

		lim_ra, lim_dec, lim_z = p_ra[ id_lim ], p_dec[ id_lim ], p_z[ id_lim ]
		lim_bcgx, lim_bcgy = bcg_x[ id_lim ], bcg_y[ id_lim ]


		#. save catalog with properties
		keys = ['ra', 'dec', 'z', 'rich', 'zform', 'lg_Mstar']
		values = [ mp_ra, mp_dec, mp_z, mp_rich, mp_zform, mp_lgMstar ]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv( extend_path + 'BCG_M_bin/%s_%s-band_photo-z-match_cat_params.csv' % (cat_lis[ ll ], band[ kk ]),)

		#. save catalog of images
		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID' ]
		values = [ lim_ra, lim_dec, lim_z, lim_bcgx, lim_bcgy, idx[ id_lim ] ]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv( extend_path + 'BCG_M_bin/%s_%s-band_photo-z-match_BCG_cat.csv' % (cat_lis[ ll ], band[ kk ]),)

##.. gri band common catalog
for ll in range( 2 ):

		r_band_file = extend_path + 'BCG_M_bin/%s_r-band_photo-z-match_BCG_cat.csv' % cat_lis[ ll ]
		g_band_file = extend_path + 'BCG_M_bin/%s_g-band_photo-z-match_BCG_cat.csv' % cat_lis[ ll ]
		i_band_file = extend_path + 'BCG_M_bin/%s_i-band_photo-z-match_BCG_cat.csv' % cat_lis[ ll ]

		medi_r_file = extend_path + 'BCG_M_bin/%s_r-band_photo-z-match_rg-common_BCG_cat.csv' % cat_lis[ ll ]
		medi_g_file = extend_path + 'BCG_M_bin/%s_g-band_photo-z-match_rg-common_BCG_cat.csv' % cat_lis[ ll ]

		out_r_file = extend_path + 'BCG_M_bin/%s_r-band_photo-z-match_rgi-common_BCG_cat.csv' % cat_lis[ ll ]
		out_g_file = extend_path + 'BCG_M_bin/%s_g-band_photo-z-match_rgi-common_BCG_cat.csv' % cat_lis[ ll ]
		out_i_file = extend_path + 'BCG_M_bin/%s_i-band_photo-z-match_rgi-common_BCG_cat.csv' % cat_lis[ ll ]

		gri_common_cat_func(r_band_file, g_band_file, i_band_file, medi_r_file, medi_g_file, out_r_file, out_g_file, out_i_file,)

raise
"""

"""
##.. compare the image catalog to previous match
pre_path = '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_match/sample_view/'

for ll in range( 2 ):

	for kk in range( 3 ):

		dat = pds.read_csv(extend_path + 'BCG_M_bin/%s_%s-band_photo-z-match_BCG_cat.csv' % (cat_lis[ ll ], band[ kk ]),)
		
		ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
		bcg_x, bcg_y = np.array( dat['bcg_x'] ), np.array( dat['bcg_y'] )

		sub_coord = SkyCoord( ra * U.deg, dec * U.deg )

		#. previous match catalog
		p_dat = pds.read_csv( pre_path + '%s_%s-band_photo-z-match_BCG-pos_cat.csv' % (cat_lis[ ll ], band[ kk ]),)
		pre_ra, pre_dec, pre_z = np.array( p_dat['ra'] ), np.array( p_dat['dec'] ), np.array( p_dat['z'] )

		pre_coord = SkyCoord( pre_ra * U.deg, pre_dec * U.deg )

		idx, sep, d3d = sub_coord.match_to_catalog_sky( pre_coord )
		id_lim = sep.value < 2.7e-4

		diff_ra, diff_dec, diff_z = ra[ id_lim == False ], dec[ id_lim == False ], z[ id_lim == False ]
		diff_bcgx, diff_bcgy = bcg_x[ id_lim == False ], bcg_y[ id_lim == False ]

		#. save
		keys = [ 'ra', 'dec', 'z', 'bcg_x', 'bcg_y' ]
		values = [ diff_ra, diff_dec, diff_z, diff_bcgx, diff_bcgy ]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv( extend_path + 'compare_to_pre/%s_%s-band_pre-diffi_BCG_cat.csv' % (cat_lis[ ll ], band[ kk ]),)

"""


### === ### richness or age bin subsamples at fixed BCG stellar mass
def lg_rich_func(x, a, b, c):
	y = a * x**2 + b*x + c
	return y

def rich_binned_func(M_bins, data_arr, out_file_hi, out_file_low, divid_f):

	high_z_form = np.array( [] )
	hi_map_rich, hi_map_age = np.array( [] ), np.array( [] )
	hi_map_ra, hi_map_dec, hi_map_z = np.array( [] ), np.array( [] ), np.array( [] )
	hi_map_Mstar = np.array( [] )
	hi_map_ID = np.array( [] )

	low_z_form = np.array( [] )
	lo_map_rich, lo_map_age = np.array( [] ), np.array( [] )
	lo_map_ra, lo_map_dec, lo_map_z = np.array( [] ), np.array( [] ), np.array( [] )
	lo_map_Mstar = np.array( [] )
	lo_map_ID = np.array( [] )

	bins_lgM = M_bins
	N_bin = len( M_bins )
	ra, dec, z, z_form, rich, age_time, lg_Mstar, origin_ID = data_arr[:]

	for ii in range( N_bin - 1):

		id_lim = (lg_Mstar >= bins_lgM[ii] ) & (lg_Mstar < bins_lgM[ii+1] )

		axis_lgM = np.log10( (10**bins_lgM[ii] + 10**bins_lgM[ii+1]) / 2 )
		thresh_rich = 10**( divid_f(axis_lgM) )

		lim_rich = rich[ id_lim ]
		lim_ra, lim_dec, lim_zc = ra[id_lim], dec[id_lim], z[id_lim]

		lim_z = z_form[ id_lim ]
		lim_age = age_time[ id_lim ]
		lim_Mstar = lg_Mstar[ id_lim ]
		ID_lim = origin_ID[ id_lim ]


		idvx_hi = lim_rich >= thresh_rich
		idvx_lo = lim_rich < thresh_rich

		high_z_form = np.r_[ high_z_form, lim_z[ idvx_hi ] ]

		hi_map_rich = np.r_[ hi_map_rich, lim_rich[ idvx_hi ]  ]

		hi_map_ra = np.r_[ hi_map_ra, lim_ra[ idvx_hi ] ]

		hi_map_dec = np.r_[ hi_map_dec, lim_dec[ idvx_hi ] ]

		hi_map_z = np.r_[ hi_map_z, lim_zc[ idvx_hi ] ]

		hi_map_age = np.r_[ hi_map_age, lim_age[ idvx_hi ] ]

		hi_map_Mstar = np.r_[ hi_map_Mstar, lim_Mstar[ idvx_hi ] ]

		hi_map_ID = np.r_[ hi_map_ID, ID_lim[ idvx_hi] ]


		low_z_form = np.r_[ low_z_form, lim_z[ idvx_lo ] ]

		lo_map_rich = np.r_[ lo_map_rich, lim_rich[ idvx_lo ] ]

		lo_map_ra = np.r_[ lo_map_ra, lim_ra[ idvx_lo ] ]

		lo_map_dec = np.r_[ lo_map_dec, lim_dec[ idvx_lo ] ]

		lo_map_z = np.r_[ lo_map_z, lim_zc[ idvx_lo ] ]

		lo_map_age = np.r_[ lo_map_age, lim_age[ idvx_lo ] ]

		lo_map_Mstar = np.r_[ lo_map_Mstar, lim_Mstar[ idvx_lo ] ]

		lo_map_ID = np.r_[ lo_map_ID, ID_lim[ idvx_lo ] ]

	## save the divided sample
	keys = ['ra', 'dec', 'z', 'z_form', 'rich', 'age', 'lg_Mstar', 'origin_ID']
	values = [ hi_map_ra, hi_map_dec, hi_map_z, high_z_form, hi_map_rich, hi_map_age, hi_map_Mstar, hi_map_ID ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_file_hi )

	keys = ['ra', 'dec', 'z', 'z_form', 'rich', 'age', 'lg_Mstar', 'origin_ID']
	values = [ lo_map_ra, lo_map_dec, lo_map_z, low_z_form, lo_map_rich, lo_map_age, lo_map_Mstar, lo_map_ID ]
	fill = dict( zip(keys, values) )
	data = pds.DataFrame(fill)
	data.to_csv( out_file_low )

	return


# rich_path = '/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/rich_bin_fixed_bcgM/'
rich_path = '/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/re_bin_rich_bin/'

dat = pds.read_csv( extend_path + 'Extend_BCGM_bin_cat.csv' )
ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
rich, z_form, lg_Mstar = np.array( dat['rich'] ), np.array( dat['z_form'] ), np.array( dat['lg_Mbcg'] )
age_time = np.array( dat['age'] )

origin_ID = np.arange( 0, len(z), )


##... rich binned subsample at fixed BCG Mstar
pre_N = 7 # 5
bins_edg = np.linspace( 10.25, 11.75, pre_N)
bins_edg = np.r_[10., bins_edg, 12 ]

medi_rich = []
medi_perr = []

cen_lgM = np.log10( 0.5 * (10**bins_edg[1:] + 10**bins_edg[:-1]) )

for ii in range( len(bins_edg) - 1 ):

	id_lim = ( lg_Mstar >= bins_edg[ii] ) & ( lg_Mstar < bins_edg[ii+1] )
	lim_rich = rich[ id_lim ]

	medi_rich.append( np.median( lim_rich ) )
	medi_perr.append( np.std( lim_rich) / np.sqrt( len(lim_rich) ) )

medi_rich = np.array( medi_rich )
medi_perr = np.array( medi_perr )

lg_medi_rich = np.log10( medi_rich )
lg_medi_perr = medi_perr / ( medi_rich * np.log(10) )


## use a simple power law ( relation between lg_rich and lg_Mstar )
# fit_F = np.polyfit( cen_lgM, np.log10( medi_rich ), deg = 2)
# Pf = np.poly1d( fit_F )

po = [0.2, -4, 18]
popt, pcov = optimize.curve_fit( lg_rich_func, cen_lgM, lg_medi_rich, p0 = np.array( po ), sigma = lg_medi_perr,)

print( popt )
a_fit, b_fit, c_fit = popt


#. mass in units of M_sun / h^2
lgM_x = np.linspace( 10, 12, 35)

# fit_line = Pf( lgM_x )
fit_line = lg_rich_func( lgM_x, a_fit, b_fit, c_fit )


#...save this line
keys = [ 'a_x^2', 'b_x', 'c' ]

# values = [ Pf[2], Pf[1], Pf[0] ]
values = [ a_fit, b_fit, c_fit ]

fill = dict( zip(keys, values) )
data = pds.DataFrame( fill, index = ['k', 'v'] )
data.to_csv( rich_path + 'rich-bin_fixed-BCG-M_divid_line_params.csv' )


plt.figure( )
ax = plt.subplot(111)

ax.scatter( lg_Mstar, np.log10( rich ), s = 15, c = 'r', alpha = 0.5,)
ax.set_ylim(1.30, 2.30)
ax.set_xlim(10, 12)

for jj in range( pre_N ):
	ax.axvline( x = bins_edg[jj], ls = '--', color = 'k', alpha = 0.5,)

# ax.plot( cen_lgM, np.log10( medi_rich ), 'ks', alpha = 0.5)

ax.errorbar( cen_lgM, lg_medi_rich, yerr = lg_medi_perr, color = 'k', alpha = 0.5,)
ax.plot( lgM_x, fit_line, ls = '-', color = 'k',)

ax.set_ylabel('$ lg \\lambda $')
ax.set_xlabel('$ lg M_{\\ast} $')

plt.savefig('/home/xkchen/M-rich_view.png', dpi = 300)
plt.close()


#. rich bin division
def Pf( x ):
	y = a_fit * x**2 + b_fit * x + c_fit
	return y


N_bin = 31
bins_lgM = np.linspace( 10, 12, N_bin)

data_arr = [ ra, dec, z, z_form, rich, age_time, lg_Mstar, origin_ID ]

out_file_hi = rich_path + 'hi-rich_fixed_Mbcg_cat.csv'
out_file_low = rich_path + 'low-rich_fixed_Mbcg_cat.csv'

rich_binned_func( bins_lgM, data_arr, out_file_hi, out_file_low, Pf )


lo_dat = pds.read_csv( rich_path + 'low-rich_fixed_Mbcg_cat.csv' )
lo_ra, lo_dec, lo_z = np.array( lo_dat.ra), np.array( lo_dat.dec), np.array( lo_dat.z)
lo_rich, lo_z_form = np.array( lo_dat.rich), np.array( lo_dat.z_form)
lo_age = np.array( lo_dat.age )
lo_lg_Mstar = np.array( lo_dat['lg_Mstar'] )

lo_origin_ID = np.array( lo_dat.origin_ID )
lo_origin_ID = lo_origin_ID.astype( int )

lo_coord = SkyCoord( lo_ra * U.deg, lo_dec * U.deg )

print( 'all, low', lo_ra.shape )


hi_dat = pds.read_csv( rich_path + 'hi-rich_fixed_Mbcg_cat.csv' )
hi_ra, hi_dec, hi_z = np.array( hi_dat.ra), np.array( hi_dat.dec), np.array( hi_dat.z)
hi_rich, hi_z_form = np.array( hi_dat.rich), np.array( hi_dat.z_form)
hi_age = np.array( hi_dat.age )
hi_lg_Mstar = np.array( hi_dat['lg_Mstar'] )

hi_origin_ID = np.array( hi_dat.origin_ID )
hi_origin_ID = hi_origin_ID.astype( int )

hi_coord = SkyCoord( hi_ra * U.deg, hi_dec * U.deg )

print( 'all, high', hi_ra.shape )


plt.figure()
plt.hist( lo_lg_Mstar, bins = 45, density = 'True', histtype = 'step', color = 'b', alpha = 0.5, label = 'low $\\lambda$')
plt.hist( hi_lg_Mstar, bins = 45, density = 'True', histtype = 'step', color = 'r', alpha = 0.5, label = 'high $\\lambda$')	

plt.axvline( np.log10( np.median( 10**lo_lg_Mstar) ), ls = '--', color = 'b', alpha = 0.5, label = 'median',)
plt.axvline( np.log10( np.mean( 10**lo_lg_Mstar) ), ls = '-', color = 'b', alpha = 0.5, label = 'mean',)

plt.axvline( np.log10( np.median(10**hi_lg_Mstar) ), ls = '--', color = 'r', alpha = 0.5,)
plt.axvline( np.log10( np.mean(10**hi_lg_Mstar) ), ls = '-', color = 'r', alpha = 0.5,)

plt.legend( loc = 2 )
plt.xlabel('$\\lg M_{\\ast} \; [M_{\\odot} / h^{2}]$')
plt.savefig('/home/xkchen/subset_lgMstar_compare.png', dpi = 300)
plt.close()


##.. match to image catalog
for kk in range( 3 ):

	pdat = pds.read_csv( load + 'photo-z_%s-band_tot_remain_cat_set_200-grid_6.0-sigma.csv' % band[ kk ],)
	p_ra, p_dec, p_z = np.array(pdat['ra']), np.array(pdat['dec']), np.array(pdat['z'])
	bcg_x, bcg_y = np.array(pdat['bcg_x']), np.array(pdat['bcg_y'])

	sub_coord = SkyCoord( p_ra * U.deg, p_dec * U.deg )

	##. map to lo_rich subsamples
	idx, sep, d3d = sub_coord.match_to_catalog_sky( lo_coord )
	id_lim = sep.value < 2.7e-4

	mp_ra, mp_dec, mp_z = lo_ra[ idx[ id_lim ] ], lo_dec[ idx[ id_lim ] ], lo_z[ idx[ id_lim ] ]
	mp_rich, mp_lgMstar = lo_rich[ idx[ id_lim ] ], lo_lg_Mstar[ idx[ id_lim ] ]
	
	mp_zform = lo_z_form[ idx[ id_lim ] ]
	mp_age = lo_age[ idx[ id_lim ] ]
	mp_ord_dex = lo_origin_ID[ idx[ id_lim ] ]

	lim_ra, lim_dec, lim_z = p_ra[ id_lim ], p_dec[ id_lim ], p_z[ id_lim ]
	lim_bcgx, lim_bcgy = bcg_x[ id_lim ], bcg_y[ id_lim ]

	print('low,', mp_ra.shape )

	#. sample properties
	keys = ['ra', 'dec', 'z', 'z_form', 'rich', 'age', 'lg_Mstar', 'origin_ID' ]
	values = [ mp_ra, mp_dec, mp_z, mp_zform, mp_rich, mp_age, mp_lgMstar, mp_ord_dex ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( rich_path + 'low-rich_%s-band_photo-z-match_cat_params.csv' % band[ kk ] )

	#. image catalog
	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID']
	values = [ lim_ra, lim_dec, lim_z, lim_bcgx, lim_bcgy, mp_ord_dex ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( rich_path + 'low-rich_%s-band_photo-z-match_BCG-pos_cat.csv' % band[ kk ] )


	##. map to high rich subsamples
	idx, sep, d3d = sub_coord.match_to_catalog_sky( hi_coord )
	id_lim = sep.value < 2.7e-4

	mp_ra, mp_dec, mp_z = hi_ra[ idx[ id_lim ] ], hi_dec[ idx[ id_lim ] ], hi_z[ idx[ id_lim ] ]
	mp_rich, mp_lgMstar = hi_rich[ idx[ id_lim ] ], hi_lg_Mstar[ idx[ id_lim ] ]

	mp_zform = hi_z_form[ idx[ id_lim ] ]
	mp_age = hi_age[ idx[ id_lim ] ]
	mp_ord_dex = hi_origin_ID[ idx[ id_lim ] ]

	lim_ra, lim_dec, lim_z = p_ra[ id_lim ], p_dec[ id_lim ], p_z[ id_lim ]
	lim_bcgx, lim_bcgy = bcg_x[ id_lim ], bcg_y[ id_lim ]

	print('high,', mp_ra.shape )

	#. sample properties
	keys = ['ra', 'dec', 'z', 'z_form', 'rich', 'age', 'lg_Mstar', 'origin_ID' ]
	values = [ mp_ra, mp_dec, mp_z, mp_zform, mp_rich, mp_age, mp_lgMstar, mp_ord_dex ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( rich_path + 'hi-rich_%s-band_photo-z-match_cat_params.csv' % band[ kk ] )

	#. image catalog
	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID']
	values = [ lim_ra, lim_dec, lim_z, lim_bcgx, lim_bcgy, mp_ord_dex ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( rich_path + 'hi-rich_%s-band_photo-z-match_BCG-pos_cat.csv' % band[ kk ] )


##... gri-common catalog
sub_lis = ['low-rich', 'hi-rich']

for ll in range( 2 ):

	r_band_file = rich_path + '%s_r-band_photo-z-match_BCG-pos_cat.csv' % sub_lis[ ll ]
	g_band_file = rich_path + '%s_g-band_photo-z-match_BCG-pos_cat.csv' % sub_lis[ ll ]
	i_band_file = rich_path + '%s_i-band_photo-z-match_BCG-pos_cat.csv' % sub_lis[ ll ]

	medi_r_file = rich_path + '%s_r-band_photo-z-match_rg-common_BCG-pos_cat.csv' % sub_lis[ ll ]
	medi_g_file = rich_path + '%s_g-band_photo-z-match_rg-common_BCG-pos_cat.csv' % sub_lis[ ll ]

	out_r_file = rich_path + '%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % sub_lis[ ll ]
	out_g_file = rich_path + '%s_g-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % sub_lis[ ll ]
	out_i_file = rich_path + '%s_i-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % sub_lis[ ll ]

	gri_common_cat_func(r_band_file, g_band_file, i_band_file, medi_r_file, medi_g_file, out_r_file, out_g_file, out_i_file,)


for kk in range( 3 ):

	sub_lo_dat = pds.read_csv( rich_path + '%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % sub_lis[0],)
	sub_lo_ra, sub_lo_dec, sub_lo_z = np.array( sub_lo_dat['ra']), np.array( sub_lo_dat['dec']), np.array( sub_lo_dat['z'])
	sub_lo_order = np.array( sub_lo_dat['origin_ID'] )

	sub_hi_dat = pds.read_csv( rich_path + '%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % sub_lis[1],)
	sub_hi_ra, sub_hi_dec, sub_hi_z = np.array( sub_hi_dat['ra']), np.array( sub_hi_dat['dec']), np.array( sub_hi_dat['z'])
	sub_hi_order = np.array( sub_hi_dat['origin_ID'] )

	print('low,', len(sub_lo_ra) )
	print('high,', len(sub_hi_ra) )

	sub_lo_lgM = lg_Mstar[ sub_lo_order ]
	medi_lo_lgM = np.log10( np.median( 10**sub_lo_lgM ) ) 
	mean_lo_lgM = np.log10( np.mean( 10**sub_lo_lgM ) )

	sub_hi_lgM = lg_Mstar[ sub_hi_order ]
	medi_hi_lgM = np.log10( np.median( 10**sub_hi_lgM ) ) 
	mean_hi_lgM = np.log10( np.mean( 10**sub_hi_lgM ) )

	print( mean_lo_lgM - mean_hi_lgM )
	print( medi_lo_lgM - medi_hi_lgM )


	plt.figure( )
	plt.hist( sub_lo_lgM, bins = 45, density = True, histtype = 'step', color = 'b', alpha = 0.5, label = 'low $\\lambda$')
	plt.axvline( x = medi_lo_lgM, ls = '--', color = 'b', alpha = 0.5, label = 'median')
	plt.axvline( x = mean_lo_lgM, ls = '-', color = 'b', alpha = 0.5, label = 'mean')

	plt.hist( sub_hi_lgM, bins = 45, density = True, histtype = 'step', color = 'r', alpha = 0.5, label = 'high $\\lambda$')
	plt.axvline( x = medi_hi_lgM, ls = '--', color = 'r', alpha = 0.5,)
	plt.axvline( x = mean_hi_lgM, ls = '-', color = 'r', alpha = 0.5,)

	plt.legend( loc = 2 )
	plt.xlabel('$\\lg M_{\\ast} \; [M_{\\odot} / h^{2}]$')
	plt.savefig('/home/xkchen/matched_lgMstar_compare.png', dpi = 300)
	plt.close()

