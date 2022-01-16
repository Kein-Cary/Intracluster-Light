import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C
import scipy.signal as signal

from astropy import cosmology as apcy
from scipy import interpolate as interp
from scipy import integrate as integ
from astropy.coordinates import SkyCoord

from scipy import optimize
from scipy import stats as sts

from BCG_SB_pro_stack import single_img_SB_func
from color_2_mass import SB_to_Lumi_func

from tmp_color_to_mass import gr_ri_band_c2m_func
from img_pre_selection import gri_common_cat_func

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


### === ### cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

pixel = 0.396
band = ['r', 'g', 'i']
L_wave = np.array([ 6166, 4686, 7480 ])
Mag_sun = [ 4.65, 5.11, 4.53 ]
z_ref = 0.25

def M20_binned_func( x_bins, data_arr, out_file_hi, out_file_low, divid_f):

	hi_map_rich, hi_map_M20 = np.array( [] ), np.array( [] )
	hi_map_ra, hi_map_dec, hi_map_z = np.array( [] ), np.array( [] ), np.array( [] )
	hi_map_Mstar = np.array( [] )

	lo_map_rich, lo_map_M20 = np.array( [] ), np.array( [] )
	lo_map_ra, lo_map_dec, lo_map_z = np.array( [] ), np.array( [] ), np.array( [] )
	lo_map_Mstar = np.array( [] )

	N_bin = len( x_bins )
	ra, dec, z, rich, lg_Mstar, lg_M20 = data_arr[:]

	for ii in range( N_bin - 1):

		lg_rich = np.log10( rich )
		id_lim = ( lg_rich >= x_bins[ii] ) & ( lg_rich < x_bins[ii+1] )

		axis_x = ( x_bins[ii] + x_bins[ii+1] ) / 2

		thresh_lgM20 = divid_f( axis_x )

		dd_lim_rich = rich[ id_lim ]
		dd_lim_ra, dd_lim_dec, dd_lim_zc = ra[id_lim], dec[id_lim], z[id_lim]
		dd_lim_Mstar = lg_Mstar[ id_lim ]

		dd_lim_M20 = lg_M20[ id_lim ]


		idvx_hi = dd_lim_M20 >= thresh_lgM20
		idvx_lo = dd_lim_M20 < thresh_lgM20


		hi_map_rich = np.r_[ hi_map_rich, dd_lim_rich[ idvx_hi ]  ]
		hi_map_ra = np.r_[ hi_map_ra, dd_lim_ra[ idvx_hi ] ]
		hi_map_dec = np.r_[ hi_map_dec, dd_lim_dec[ idvx_hi ] ]
		hi_map_z = np.r_[ hi_map_z, dd_lim_zc[ idvx_hi ] ]
		hi_map_Mstar = np.r_[ hi_map_Mstar, dd_lim_Mstar[ idvx_hi ] ]
		hi_map_M20 = np.r_[ hi_map_M20, dd_lim_M20[ idvx_hi ] ]

		lo_map_rich = np.r_[ lo_map_rich, dd_lim_rich[ idvx_lo ] ]
		lo_map_ra = np.r_[ lo_map_ra, dd_lim_ra[ idvx_lo ] ]
		lo_map_dec = np.r_[ lo_map_dec, dd_lim_dec[ idvx_lo ] ]
		lo_map_z = np.r_[ lo_map_z, dd_lim_zc[ idvx_lo ] ]
		lo_map_Mstar = np.r_[ lo_map_Mstar, dd_lim_Mstar[ idvx_lo ] ]
		lo_map_M20 = np.r_[ lo_map_M20, dd_lim_M20[ idvx_lo ] ]

	## save the divided sample
	keys = ['ra', 'dec', 'z', 'rich', 'lg_Mstar', 'lg_M_R-lim']
	values = [ hi_map_ra, hi_map_dec, hi_map_z, hi_map_rich, hi_map_Mstar, hi_map_M20 ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_file_hi )

	keys = ['ra', 'dec', 'z', 'rich', 'lg_Mstar', 'lg_M_R-lim']
	values = [ lo_map_ra, lo_map_dec, lo_map_z, lo_map_rich, lo_map_Mstar, lo_map_M20 ]
	fill = dict( zip(keys, values) )
	data = pds.DataFrame(fill)
	data.to_csv( out_file_low )

	return

def lg_linea_func( x, a, b):
	y = a + b * x
	return y

def resi_func( po, x, y):

	a, b = po[:]
	y_mod = lg_linea_func( x, a, b)
	delta = y_mod - y

	return delta

### === ### subsample division
path = '/home/xkchen/figs/extend_bcgM_cat/aperM_bin/'

#. ref_cat
ref_cat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/Extend_BCGM_bin_cat.csv')
ref_ra, ref_dec, ref_z = np.array( ref_cat['ra'] ), np.array( ref_cat['dec'] ), np.array( ref_cat['z'] )

rich, age = np.array( ref_cat['rich'] ), np.array( ref_cat['age'] )
z_form = np.array( ref_cat['z_form'] )

lg_Mstar = np.array( ref_cat['lg_Mbcg'] )  ## in units of M_sun / h^2

ref_coord = SkyCoord( ra = ref_ra * U.deg, dec = ref_dec * U.deg )


dat = pds.read_csv(path + 'Extend_BCGM_BCG-mag_cat.csv')
cc_ra, cc_dec, cc_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

M20 = np.array( dat['Mstar_20'] )  ## in units of M_sun
Li_20 = np.array( dat['Li_20'] )

lg_M20 = np.log10( M20 )
lg_Li20 = np.log10( Li_20 )

cc_coord = SkyCoord( ra = cc_ra * U.deg, dec = cc_dec * U.deg )


idx, d2d, d3d = cc_coord.match_to_catalog_sky( ref_coord )
id_lim = d2d.value < 2.7e-4

mp_ra, mp_dec, mp_z = ref_ra[ idx[id_lim] ], ref_dec[ idx[id_lim] ], ref_z[ idx[id_lim] ]
mp_rich, mp_lgMstar, mp_age = rich[ idx[id_lim] ], lg_Mstar[ idx[id_lim] ], age[ idx[id_lim] ]

mp_lgMstar = mp_lgMstar - 2 * np.log10( h )   ## in units of M_sun

lim_lgM20 = lg_M20[ id_lim ]
lim_lgLi = lg_Li20[ id_lim ]


### ... divide sample based on M20 (at fixed richness)
pre_N = 8
x_bins = np.logspace( np.log10( mp_rich.min() ), np.log10( mp_rich.max() ), pre_N )

medi_lgM20 = []

for jj in range( pre_N - 1 ):

	id_x = ( x_bins[jj] < mp_rich ) & ( mp_rich <= x_bins[jj+1] )

	_sub_lgM20 = lim_lgM20[ id_x ]
	_medi_lgM20 = np.log10( np.nanmedian( 10**_sub_lgM20 ) )

	medi_lgM20.append( _medi_lgM20 )

cen_x = 0.5 * ( x_bins[1:] + x_bins[:-1] )

medi_lgM20 = np.array( medi_lgM20 )

fit_F = np.polyfit( np.log10(cen_x), medi_lgM20, deg = 1)
Pf = np.poly1d( fit_F )


N_bins = 35
bins_lgrich = np.linspace( 1.30, 2.31, N_bins )
fit_line = Pf( bins_lgrich )


plt.figure()

plt.plot( mp_rich, lim_lgM20, 'ko', markersize = 3, alpha = 0.15,)
plt.plot( cen_x, medi_lgM20, 'ks-', )
plt.plot( 10**bins_lgrich, fit_line, 'b--',)

plt.xlim( 20, 200 )
plt.ylim( 10.5, 12 )

plt.xscale('log')
plt.xlabel('$\\lambda$')
plt.ylabel('$\\lg \, M_{\\ast} \; [M_{\\odot}]$')

plt.savefig('/home/xkchen/lgM20_rich_divid.png', dpi = 300)
plt.close()


data_arr = [ mp_ra, mp_dec, mp_z, mp_rich, mp_lgMstar, lim_lgM20 ]

out_file_hi = path + 'photo-z_match_hi-lgM20_cluster_cat.csv'
out_file_low = path + 'photo-z_match_low-lgM20_cluster_cat.csv'

M20_binned_func( bins_lgrich, data_arr, out_file_hi, out_file_low, Pf )


lo_dat = pds.read_csv( path + 'photo-z_match_low-lgM20_cluster_cat.csv' )
lo_ra, lo_dec, lo_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )
lo_rich, lo_lgMstar, lo_lgM20 = np.array( lo_dat['rich'] ), np.array( lo_dat['lg_Mstar'] ), np.array( lo_dat['lg_M_R-lim'] )


hi_dat = pds.read_csv( path + 'photo-z_match_hi-lgM20_cluster_cat.csv' )
hi_ra, hi_dec, hi_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )
hi_rich, hi_lgMstar, hi_lgM20 = np.array( hi_dat['rich'] ), np.array( hi_dat['lg_Mstar'] ), np.array( hi_dat['lg_M_R-lim'] )


lo_coord = SkyCoord( ra = lo_ra * U.deg, dec = lo_dec * U.deg )
hi_coord = SkyCoord( ra = hi_ra * U.deg, dec = hi_dec * U.deg )

print( 'low,', len(lo_ra) )
print( 'high,', len(hi_ra) )


plt.figure()
plt.hist( lo_rich, bins = 45, density = True, color = 'b', alpha = 0.5, histtype = 'step', label = 'low lgM20',)
plt.axvline( x = np.median( lo_rich ), ls = '--', color = 'b', alpha = 0.5, label = 'median',)
plt.axvline( x = np.mean( lo_rich ), ls = '-', color = 'b', alpha = 0.5, label = 'mean',)

plt.hist( hi_rich, bins = 45, density = True, color = 'r', alpha = 0.5, histtype = 'step', label = 'high lgM20',)
plt.axvline( x = np.median( hi_rich ), ls = '--', color = 'r', alpha = 0.5,)
plt.axvline( x = np.mean( hi_rich ), ls = '-', color = 'r', alpha = 0.5,)

plt.legend( loc = 1 )
plt.yscale('log')
plt.xscale('log')

plt.xlabel('$\\lambda$')
plt.savefig('/home/xkchen/lgM20_divid_rich_compare.png', dpi = 300)
plt.close()


### === match to image catalog
load = '/home/xkchen/mywork/ICL/data/photo_cat/'

for kk in range( 3 ):

	pdat = pds.read_csv( load + 'photo-z_%s-band_tot_remain_cat_set_200-grid_6.0-sigma.csv' % band[ kk ],)
	p_ra, p_dec, p_z = np.array(pdat['ra']), np.array(pdat['dec']), np.array(pdat['z'])
	bcg_x, bcg_y = np.array(pdat['bcg_x']), np.array(pdat['bcg_y'])

	sub_coord = SkyCoord( p_ra * U.deg, p_dec * U.deg )

	##. map to lo_lgM20 subsamples
	idx, sep, d3d = sub_coord.match_to_catalog_sky( lo_coord )
	id_lim = sep.value < 2.7e-4

	mp_ra, mp_dec, mp_z = lo_ra[ idx[ id_lim ] ], lo_dec[ idx[ id_lim ] ], lo_z[ idx[ id_lim ] ]
	mp_rich, mp_lgMstar = lo_rich[ idx[ id_lim ] ], lo_lgMstar[ idx[ id_lim ] ]
	mp_lgM20 = lo_lgM20[ idx[ id_lim ] ]

	mp_ord_dex = idx[ id_lim ]

	lim_ra, lim_dec, lim_z = p_ra[ id_lim ], p_dec[ id_lim ], p_z[ id_lim ]
	lim_bcgx, lim_bcgy = bcg_x[ id_lim ], bcg_y[ id_lim ]

	print('low,', mp_ra.shape )

	#. sample properties
	keys = ['ra', 'dec', 'z', 'rich', 'lg_Mstar', 'lg_M20']
	values = [ mp_ra, mp_dec, mp_z, mp_rich, mp_lgMstar, mp_lgM20 ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( path + 'low-lgM20_%s-band_photo-z-match_cat_params.csv' % band[ kk ] )

	#. image catalog
	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID']
	values = [ lim_ra, lim_dec, lim_z, lim_bcgx, lim_bcgy, mp_ord_dex ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( path + 'low-lgM20_%s-band_photo-z-match_BCG-pos_cat.csv' % band[ kk ] )


	##. map to high_lgM20 subsamples
	idx, sep, d3d = sub_coord.match_to_catalog_sky( hi_coord )
	id_lim = sep.value < 2.7e-4

	mp_ra, mp_dec, mp_z = hi_ra[ idx[ id_lim ] ], hi_dec[ idx[ id_lim ] ], hi_z[ idx[ id_lim ] ]
	mp_rich, mp_lgMstar = hi_rich[ idx[ id_lim ] ], hi_lgMstar[ idx[ id_lim ] ]
	mp_lgM20 = hi_lgM20[ idx[ id_lim ] ]

	mp_ord_dex = idx[ id_lim ]

	lim_ra, lim_dec, lim_z = p_ra[ id_lim ], p_dec[ id_lim ], p_z[ id_lim ]
	lim_bcgx, lim_bcgy = bcg_x[ id_lim ], bcg_y[ id_lim ]

	print('high,', mp_ra.shape )


	#. sample properties
	keys = [ 'ra', 'dec', 'z', 'rich', 'lg_Mstar', 'lg_M20' ]
	values = [ mp_ra, mp_dec, mp_z, mp_rich, mp_lgMstar, mp_lgM20 ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( path + 'hi-lgM20_%s-band_photo-z-match_cat_params.csv' % band[ kk ] )


	#. image catalog
	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID']
	values = [ lim_ra, lim_dec, lim_z, lim_bcgx, lim_bcgy, mp_ord_dex ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( path + 'hi-lgM20_%s-band_photo-z-match_BCG-pos_cat.csv' % band[ kk ] )	


##.. gri band common catalog
cat_lis = ['low-lgM20', 'hi-lgM20']

for ll in range( 2 ):

	r_band_file = path + '%s_r-band_photo-z-match_BCG-pos_cat.csv' % cat_lis[ ll ]
	g_band_file = path + '%s_g-band_photo-z-match_BCG-pos_cat.csv' % cat_lis[ ll ]
	i_band_file = path + '%s_i-band_photo-z-match_BCG-pos_cat.csv' % cat_lis[ ll ]

	medi_r_file = path + '%s_r-band_photo-z-match_rg-common_BCG-pos_cat.csv' % cat_lis[ ll ]
	medi_g_file = path + '%s_g-band_photo-z-match_rg-common_BCG-pos_cat.csv' % cat_lis[ ll ]

	out_r_file = path + '%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[ ll ]
	out_g_file = path + '%s_g-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[ ll ]
	out_i_file = path + '%s_i-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[ ll ]

	gri_common_cat_func(r_band_file, g_band_file, i_band_file, medi_r_file, medi_g_file, out_r_file, out_g_file, out_i_file,)



##.. properties comparison
cat_lis = ['low-lgM20', 'hi-lgM20']
fig_name = ['$Low \; M_{\\ast, \, 20}$', '$High \; M_{\\ast, \, 20}$']

tmp_rich = []
tmp_lgMstar = []
tmp_lgM20 = []

for ll in range( 2 ):

	dat = pds.read_csv( path + '%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[ ll ] )
	ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
	map_dex = np.array( dat['origin_ID'] )

	print('*' * 10)
	print( len(ra) )

	sub_coord = SkyCoord( ra = ra * U.deg, dec = dec * U.deg )

	idx, sep, d3d = sub_coord.match_to_catalog_sky( ref_coord )
	id_lim = sep.value < 2.7e-4

	tt_rich, tt_lgMstar = rich[ idx[id_lim] ], lg_Mstar[ idx[id_lim] ]

	if ll == 0:
		tt_lgM20 = lo_lgM20[ map_dex ]

	if ll == 1:
		tt_lgM20 = hi_lgM20[ map_dex ]

	tmp_rich.append( tt_rich )
	tmp_lgMstar.append( tt_lgMstar )
	tmp_lgM20.append( tt_lgM20 )


plt.figure()
plt.hist( tmp_rich[0], bins = 45, density = True, color = 'b', alpha = 0.5, histtype = 'step', label = fig_name[0],)
plt.axvline( x = np.median( tmp_rich[0] ), ls = '--', color = 'b', alpha = 0.5, label = 'median',)
plt.axvline( x = np.mean( tmp_rich[0] ), ls = '-', color = 'b', alpha = 0.5, label = 'mean',)

plt.hist( tmp_rich[1], bins = 45, density = True, color = 'r', alpha = 0.5, histtype = 'step', label = fig_name[1],)
plt.axvline( x = np.median( tmp_rich[1] ), ls = '--', color = 'r', alpha = 0.5,)
plt.axvline( x = np.mean( tmp_rich[1] ), ls = '-', color = 'r', alpha = 0.5,)

plt.legend( loc = 1 )
plt.yscale('log')
plt.xscale('log')

plt.xlabel('$\\lambda$')
plt.savefig('/home/xkchen/lgM20_bin_rich_compare.png', dpi = 300)
plt.close()


plt.figure()
plt.hist( tmp_lgMstar[0], bins = 45, density = True, color = 'b', alpha = 0.5, histtype = 'step', label = fig_name[0],)
plt.axvline( x = np.log10( np.median( 10**tmp_lgMstar[0] ) ), ls = '--', color = 'b', alpha = 0.5, label = 'median',)
plt.axvline( x = np.log10( np.mean( 10**tmp_lgMstar[0] ) ), ls = '-', color = 'b', alpha = 0.5, label = 'mean',)

plt.hist( tmp_lgMstar[1], bins = 45, density = True, color = 'r', alpha = 0.5, histtype = 'step', label = fig_name[1],)
plt.axvline( x = np.log10( np.median( 10**tmp_lgMstar[1] ) ), ls = '--', color = 'r', alpha = 0.5,)
plt.axvline( x = np.log10( np.mean( 10**tmp_lgMstar[1] ) ), ls = '-', color = 'r', alpha = 0.5,)

plt.legend( loc = 1 )
plt.xlim( 10, 12 )
plt.xlabel('$\\lg \, M_{\\ast} \; [M_{\\odot}]$')
plt.savefig('/home/xkchen/lgM20_bin_lgMstar_compare.png', dpi = 300)
plt.close()


plt.figure()
plt.hist( tmp_lgM20[0], bins = 45, density = True, color = 'b', alpha = 0.5, histtype = 'step', label = fig_name[0],)
plt.axvline( x = np.log10( np.median( 10**tmp_lgM20[0] ) ), ls = '--', color = 'b', alpha = 0.5, label = 'median',)
plt.axvline( x = np.log10( np.mean( 10**tmp_lgM20[0] ) ), ls = '-', color = 'b', alpha = 0.5, label = 'mean',)

plt.hist( tmp_lgM20[1], bins = 45, density = True, color = 'r', alpha = 0.5, histtype = 'step', label = fig_name[1],)
plt.axvline( x = np.log10( np.median( 10**tmp_lgM20[1] ) ), ls = '--', color = 'r', alpha = 0.5,)
plt.axvline( x = np.log10( np.mean( 10**tmp_lgM20[1] ) ), ls = '-', color = 'r', alpha = 0.5,)

plt.legend( loc = 1 )
plt.xlim( 10.5, 12)
plt.xlabel('$\\lg \, M_{\\ast, 20} \; [M_{\\odot}]$')
plt.savefig('/home/xkchen/lgM20_bin_lgM20_compare.png', dpi = 300)
plt.close()

