import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator


import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C
import scipy.signal as signal

from astropy.table import Table
from astropy import cosmology as apcy
from scipy import interpolate as interp
from scipy import integrate as integ
from astropy.coordinates import SkyCoord

from scipy import optimize
from scipy import stats as sts

### cosmology
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

rad2asec = U.rad.to(U.arcsec)
pixel = 0.396

band = ['r', 'g', 'i']
Mag_sun = [ 4.65, 5.11, 4.53 ]
L_wave = np.array([ 6166, 4686, 7480 ])


##... image catalog within 0.2~z~0.3
ref_file = '/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits'
goal_data = fits.getdata( ref_file )

RA = np.array( goal_data.RA )
DEC = np.array( goal_data.DEC )
ID = np.array( goal_data.OBJID )

Z_photo = np.array( goal_data.Z_LAMBDA )

P_cen = np.array( goal_data.P_CEN )
RA_cen = np.array( goal_data.RA_CEN )
DEC_cen = np.array( goal_data.DEC_CEN )
ID_cen = np.array( goal_data.ID_CEN )


idx_lim = ( Z_photo >= 0.2 ) & ( Z_photo <= 0.3 )
lim_ra, lim_dec, lim_z = RA[ idx_lim ], DEC[ idx_lim ], Z_photo[ idx_lim ]
lim_ID = ID[ idx_lim ]

lim_Pcen = P_cen[ idx_lim ]
lim_RA_cen = RA_cen[ idx_lim ]
lim_DEC_cen = DEC_cen[ idx_lim ]
lim_ID_cen = ID_cen[ idx_lim ]

ref_coord = SkyCoord( ra = lim_ra * U.deg, dec = lim_dec * U.deg )


###... entire sample compare
def all_sample_view():

	dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/Extend_BCGM_bin_cat.csv')
	cc_ra, cc_dec, cc_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

	cc_coord = SkyCoord( ra = cc_ra * U.deg, dec = cc_dec * U.deg )

	idx, sep, d3d = cc_coord.match_to_catalog_sky( ref_coord )
	id_lim = sep.value < 2.7e-4

	mp_ra, mp_dec, mp_z = lim_ra[ idx[ id_lim ] ], lim_dec[ idx[ id_lim ] ], lim_z[ idx[ id_lim ] ]
	mp_IDs = lim_ID[ idx[ id_lim ] ]

	mp_Pcen = lim_Pcen[ idx[ id_lim ] ]
	mp_ra_cen = lim_RA_cen[ idx[ id_lim ] ]
	mp_dec_cen = lim_DEC_cen[ idx[ id_lim ] ]
	mp_ID_cen = lim_ID_cen[ idx[ id_lim ] ]


	plt.figure()
	plt.hist( mp_Pcen[:,0], bins = 45, density = True, histtype = 'step', color = 'r', alpha = 0.5, 
				label = '1st-$P_{cen}$')
	plt.hist( mp_Pcen[:,1], bins = 45, density = True, histtype = 'step', color = 'b', alpha = 0.5, 
				label = '2nd-$P_{cen}$')
	plt.legend( loc = 'upper center',)
	plt.yscale('log')
	plt.xlabel('$P_{cen}$')
	plt.xlim(0,1)
	plt.savefig('/home/xkchen/all_sample_Pcen_compare.png', dpi = 300)
	plt.close()


	keys = ['ra', 'dec', 'z', 'objID', 'Pcen', 'ra_cen', 'dec_cen', 'ID_cen']
	values = [ mp_ra, mp_dec, mp_z, mp_IDs, 
				mp_Pcen[:,0], mp_ra_cen[:,0], mp_dec_cen[:,0], mp_ID_cen[:,0] ]

	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( '/home/xkchen/Extend_BCGM_bin_1st-Pcen_cat.csv')


	keys = ['ra', 'dec', 'z', 'objID', 'Pcen', 'ra_cen', 'dec_cen', 'ID_cen']
	values = [ mp_ra, mp_dec, mp_z, mp_IDs, 
				mp_Pcen[:,1], mp_ra_cen[:,1], mp_dec_cen[:,1], mp_ID_cen[:,1] ]

	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( '/home/xkchen/Extend_BCGM_bin_2nd-Pcen_cat.csv')


	order_dex = np.arange(0, len(mp_ra) )

	out_arr = np.array( [ order_dex, mp_ra_cen[:,0], mp_dec_cen[:,0] ] ).T
	np.savetxt('/home/xkchen/Extend_BCGM_bin_1st-Pcen_cat.dat', out_arr, fmt = ('%.0f', '%.8f', '%.8f'),)

	out_arr = np.array( [ order_dex, mp_ra_cen[:,1], mp_dec_cen[:,1] ] ).T
	np.savetxt('/home/xkchen/Extend_BCGM_bin_2nd-Pcen_cat.dat', out_arr, fmt = ('%.0f', '%.8f', '%.8f'),)

	return


#. subsamples
cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['low $M_{\\ast} ^{BCG}$', 'high $M_{\\ast} ^{BCG}$']

tmp_Pcen_0 = []
tmp_Pcen_1 = []

for kk in range( 2 ):

	dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/BCG_M_bin/' + 
						'%s_r-band_photo-z-match_rgi-common_BCG_cat.csv' % cat_lis[kk],)

	cc_ra, cc_dec, cc_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
	cc_coord = SkyCoord( ra = cc_ra * U.deg, dec = cc_dec * U.deg )

	#... match
	idx, sep, d3d = cc_coord.match_to_catalog_sky( ref_coord )
	id_lim = sep.value < 2.7e-4

	mp_ra, mp_dec, mp_z = lim_ra[ idx[ id_lim ] ], lim_dec[ idx[ id_lim ] ], lim_z[ idx[ id_lim ] ]
	mp_IDs = lim_ID[ idx[ id_lim ] ]

	mp_Pcen = lim_Pcen[ idx[ id_lim ] ]
	mp_ra_cen = lim_RA_cen[ idx[ id_lim ] ]
	mp_dec_cen = lim_DEC_cen[ idx[ id_lim ] ]
	mp_ID_cen = lim_ID_cen[ idx[ id_lim ] ]

	tmp_Pcen_0.append( mp_Pcen[:,0] )
	tmp_Pcen_1.append( mp_Pcen[:,1] )


	# keys = ['ra', 'dec', 'z', 'objID', 'Pcen', 'ra_cen', 'dec_cen', 'ID_cen']
	# values = [ mp_ra, mp_dec, mp_z, mp_IDs, 
	# 			mp_Pcen[:,0], mp_ra_cen[:,0], mp_dec_cen[:,0], mp_ID_cen[:,0] ]

	# fill = dict( zip( keys, values ) )
	# data = pds.DataFrame( fill )
	# data.to_csv( '/home/xkchen/%s_bin_1st-Pcen_cat.csv' % cat_lis[kk],)


	# keys = ['ra', 'dec', 'z', 'objID', 'Pcen', 'ra_cen', 'dec_cen', 'ID_cen']
	# values = [ mp_ra, mp_dec, mp_z, mp_IDs, 
	# 			mp_Pcen[:,1], mp_ra_cen[:,1], mp_dec_cen[:,1], mp_ID_cen[:,1] ]

	# fill = dict( zip( keys, values ) )
	# data = pds.DataFrame( fill )
	# data.to_csv( '/home/xkchen/%s_bin_2nd-Pcen_cat.csv' % cat_lis[kk],)


	##... for SDSS data query
	# order_dex = np.arange(0, len(mp_ra) )

	# out_arr = np.array( [ order_dex, mp_ra_cen[:,0], mp_dec_cen[:,0] ] ).T
	# np.savetxt('/home/xkchen/%s_bin_1st-Pcen_cat.dat' % cat_lis[kk], out_arr, fmt = ('%.0f', '%.8f', '%.8f'),)

	# out_arr = np.array( [ order_dex, mp_ra_cen[:,1], mp_dec_cen[:,1] ] ).T
	# np.savetxt('/home/xkchen/%s_bin_2nd-Pcen_cat.dat' % cat_lis[kk], out_arr, fmt = ('%.0f', '%.8f', '%.8f'),)


### === ### i-band magnitude compare
def put_M_func( iMag_x ):

	#. assuming average color and estimate mass range
	fit_file = '/home/xkchen/figs/extend_bcgM_cat/Mass_Li_fit/least-square_M-to-i-band-Lumi&color.csv'
	pfit_dat = pds.read_csv( fit_file )
	a_fit = np.array( pfit_dat['a'] )[0]
	b_fit = np.array( pfit_dat['b'] )[0]
	c_fit = np.array( pfit_dat['c'] )[0]
	d_fit = np.array( pfit_dat['d'] )[0]

	#. averaged g-r and r-i, Rykoff et al. 2014 (90percen member color)
	mean_gr = 1.4
	mean_ri = 0.5

	# put_iMag = np.linspace(-25, -21, 25)
	# put_Li = 10**( -0.4 * ( put_iMag - Mag_sun[2] ) )

	put_Li = 10**( -0.4 * ( iMag_x - Mag_sun[2] ) )
	put_lgL = np.log10( put_Li )

	put_lgM = a_fit * mean_gr + b_fit * mean_ri + c_fit * put_lgL + d_fit
	# put_Mx = 10**put_lgM

	return put_lgM


tmp_diff_iMag = []
cp_diff_iMag = []
tmp_iMag_2nd_Pcen = []

for kk in range( 2 ):

	dat_0 = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/Pcen_compare_set/' + 
							'%s_1st-Pcen_mag_cat.csv' % cat_lis[kk], skiprows = 1)
	iMag_0 = np.array( dat_0['absMagI'] )
	z_0 = np.array( dat_0['z'] )
	cmod_magi_0 = np.array( dat_0['cModelMag_i'] )

	Dl_0 = Test_model.luminosity_distance( z_0 ).value
	cp_iMag_0 = cmod_magi_0 - 5 * np.log10( Dl_0 * 1e6 ) + 5


	dat_1 = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/Pcen_compare_set/' + 
						'%s_2nd-Pcen_mag_cat.csv' % cat_lis[kk], skiprows = 1)
	iMag_1 = np.array( dat_1['absMagI'] )
	z_1 = np.array( dat_1['z'] )
	cmod_magi_1 = np.array( dat_1['cModelMag_i'] )	

	Dl_1 = Test_model.luminosity_distance( z_1 ).value
	cp_iMag_1 = cmod_magi_1 - 5 * np.log10( Dl_1 * 1e6 ) + 5


	id_nul_0 = np.isnan( Dl_0 )
	id_nul_1 = np.isnan( Dl_1 )

	id_nul = id_nul_0 | id_nul_1
	id_lim = id_nul == False

	cp_diff_iMag.append( cp_iMag_0[id_lim] - cp_iMag_1[id_lim] )

	tmp_diff_iMag.append( iMag_0 - iMag_1 )
	tmp_iMag_2nd_Pcen.append( iMag_1 )


	plt.figure()
	ax = plt.subplot(111)

	# ax.set_title( fig_name[kk] )
	ax.hist( iMag_0, bins = 45, density = True, color = 'r', ls = '-', alpha = 0.5, histtype = 'step', 
				label = '1st-$P_{cen}$',)
	ax.axvline( np.median(iMag_0), ls = '-', color = 'r', alpha = 0.5, label = 'median')
	ax.axvline( np.mean(iMag_0), ls = '--', color = 'r', alpha = 0.5, label = 'mean')

	ax.hist( iMag_1, bins = 45, density = True, color = 'b', ls = '-', alpha = 0.5, histtype = 'step', 
				label = '2nd-$P_{cen}$',)

	ax.axvline( np.median(iMag_1), ls = '-', color = 'b', alpha = 0.5,)
	ax.axvline( np.mean(iMag_1), ls = '--', color = 'b', alpha = 0.5,)

	ax.text( -22.0, 0.4, s = fig_name[kk], fontsize = 14,)
	ax.legend( loc = 1, frameon = False, fontsize = 13,)
	ax.set_xlabel('Mag_i', fontsize = 13,)
	ax.set_xlim( -25, -21 )

	sub_ax = ax.twiny()
	sub_ax.set_xlim( ax.get_xlim() )

	x_ticks = ax.get_xticks()
	lgM_x = put_M_func( x_ticks )
	label_lis = ['%.1f' % ll for ll in lgM_x ]

	sub_ax.set_xticks( x_ticks )
	sub_ax.set_xticklabels( label_lis )
	sub_ax.set_xlabel('$\\lg \, M_{\\ast} \; [M_{\\odot}]$', fontsize = 13,)

	sub_ax.xaxis.set_minor_locator( ticker.AutoMinorLocator() )
	ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	ax.xaxis.set_minor_locator( ticker.AutoMinorLocator() )

	sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)

	plt.savefig('/home/xkchen/%s_iMag_compare.png' % cat_lis[kk], dpi = 300)
	# plt.savefig('/home/xkchen/%s_iMag_compare.pdf' % cat_lis[kk], dpi = 300)
	plt.close()


plt.figure()
ax = plt.subplot( 111 )

ax.hist( tmp_diff_iMag[0], bins = 45, density = True, color = 'b', histtype = 'step', alpha = 0.5, label = fig_name[0],)
ax.axvline( x = np.median( tmp_diff_iMag[0] ), ls = '-', color = 'b', alpha = 0.5, label = 'median')
ax.axvline( x = np.mean( tmp_diff_iMag[0] ), ls = '--', color = 'b', alpha = 0.5, label = 'mean')

ax.hist( tmp_diff_iMag[1], bins = 45, density = True, color = 'r', histtype = 'step', alpha = 0.5, label = fig_name[1],)
ax.axvline( x = np.median( tmp_diff_iMag[1] ), ls = '-', color = 'r', alpha = 0.5,)
ax.axvline( x = np.mean( tmp_diff_iMag[1] ), ls = '--', color = 'r', alpha = 0.5,)

ax.set_xlabel('$\\Delta$Mag_i = $\\mathrm{ Mag\_i_{1st,Pcen} } {-} \\mathrm{ Mag\_i_{2nd,Pcen} }$', fontsize = 13,)
ax.set_xlim( -3, 1.5 )
ax.legend( loc = 2, frameon = False, fontsize = 13,)

sub_ax = ax.twiny()
sub_ax.set_xlim( ax.get_xlim() )

x_ticks = ax.get_xticks()
delta_lgMx = x_ticks / 2.5

label_lis = ['%.1f' % ll for ll in delta_lgMx ]
sub_ax.set_xticks( x_ticks )
sub_ax.set_xticklabels( label_lis )
sub_ax.set_xlabel('$\\Delta \\lg \, M_{\\ast} \; [M_{\\odot}]$', fontsize = 13,)

sub_ax.xaxis.set_minor_locator( ticker.AutoMinorLocator() )
ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
ax.xaxis.set_minor_locator( ticker.AutoMinorLocator() )

sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)

plt.savefig('/home/xkchen/diff_iMag_compare.png', dpi = 300)
# plt.savefig('/home/xkchen/diff_iMag_compare.pdf', dpi = 300)
plt.close()


raise


##... view on samples with diff_iMag >= 0
bins_P = np.linspace(0, 1, 45)

for kk in range( 2 ):

	id_vx = tmp_diff_iMag[kk] >= 0
	sup_Pcen_0 = tmp_Pcen_0[kk][ id_vx ]
	sup_Pcen_1 = tmp_Pcen_1[kk][ id_vx ]

	plt.figure()
	plt.title( fig_name[kk] )
	plt.hist( tmp_Pcen_0[kk], bins = bins_P, density = False, histtype = 'step', color = 'r', alpha = 0.5, 
				label = '1st-$P_{cen}$ (N=%d)' % len(tmp_Pcen_0[kk]),)
	plt.hist( sup_Pcen_0, bins = bins_P, density = False, color = 'r', alpha = 0.5, 
				label = '1st-$P_{cen}$ diff_i-Mag$ \\geq 0$ (N=%d)' % np.sum(id_vx),)

	plt.hist( tmp_Pcen_1[kk], bins = bins_P, density = False, histtype = 'step', color = 'b', alpha = 0.5, 
				label = '2nd-$P_{cen}$')

	plt.hist( sup_Pcen_1, bins = bins_P, density = False, color = 'b', alpha = 0.5, 
				label = '2nd-$P_{cen}$ diff_i-Mag$ \\geq 0$')

	plt.legend( loc = 'upper center',)
	plt.yscale('log')
	plt.xlabel('$P_{cen}$')
	plt.xlim(0,1)
	plt.savefig('/home/xkchen/%s_sample_Pcen_compare.png' % cat_lis[kk], dpi = 300)
	plt.close()

