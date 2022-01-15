import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds

import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

from scipy.optimize import curve_fit
from scipy import signal
from scipy import interpolate as interp
from scipy.stats import binned_statistic as binned

from img_pre_selection import cat_match_func
from fig_out_module import cc_grid_img, grid_img
from light_measure import jack_SB_func
from fig_out_module import zref_BCG_pos_func
from BCG_SB_pro_stack import BCG_SB_pros_func

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
# constant
rad2asec = U.rad.to(U.arcsec)
pixel = 0.396
band = ['r', 'g', 'i',]
mag_add = np.array([0, 0, 0])

z_ref = 0.25

### local
#****************************#
load = '/home/xkchen/mywork/ICL/data/'

lo_dat = pds.read_csv(load + 'BCG_stellar_mass_cat/low_star-Mass_cat.csv')
lo_ra, lo_dec, lo_z = np.array(lo_dat.ra), np.array(lo_dat.dec), np.array(lo_dat.z)
lo_rich, lo_M_star = np.array(lo_dat.rich), np.array(lo_dat.lg_Mass)

hi_dat = pds.read_csv(load + 'BCG_stellar_mass_cat/high_star-Mass_cat.csv')
hi_ra, hi_dec, hi_z = np.array(hi_dat.ra), np.array(hi_dat.dec), np.array(hi_dat.z)
hi_rich, hi_M_star = np.array(hi_dat.rich), np.array(hi_dat.lg_Mass)

idmx = (lo_z >= 0.2) & (lo_z <= 0.3)
lo_sub_z, lo_sub_rich = lo_z[idmx], lo_rich[idmx]

idnx = (hi_z >= 0.2) & (hi_z <= 0.3)
hi_sub_z, hi_sub_rich = hi_z[idnx], hi_rich[idnx]


'''
plt.figure()
plt.hist( lo_z, bins = 50, density = True, color = 'b', alpha = 0.5, label = 'low $M_{\\ast}$ [%d]' % len(lo_z),)
plt.hist( hi_z, bins = 50, density = True, color = 'r', alpha = 0.5, label = 'high $M_{\\ast}$ [%d]' % len(hi_z),)
plt.axvline(x = 0.2, ls = '--', color = 'k', alpha = 0.5,)
plt.axvline(x = 0.3, ls = '--', color = 'k', alpha = 0.5,)
plt.xlabel('z')
plt.ylabel('pdf')
plt.legend(loc = 2,)
plt.savefig('redshift_compare.png', dpi = 300)
plt.close()


plt.figure()
plt.title('clusters in $ 0.2 \\sim z \\sim 0.3 $')
plt.hist( lo_rich[idmx], bins = 50, density = False, color = 'b', alpha = 0.5, label = 'low $M_{\\ast}$ [%d]' % len(lo_z[idmx]),)
plt.hist( hi_rich[idnx], bins = 50, density = False, color = 'r', alpha = 0.5, label = 'high $M_{\\ast}$ [%d]' % len(hi_z[idnx]),)

plt.axvline( x = np.mean(lo_rich[ idmx ]), ls = '-', color = 'b', alpha = 0.5, label = 'mean',)
plt.axvline( x = np.median(lo_rich[ idmx ]), ls = '--', color = 'b', alpha = 0.5, label = 'median',)
plt.axvline( x = np.mean(hi_rich[ idnx ]), ls = '-', color = 'r', alpha = 0.5,)
plt.axvline( x = np.median(hi_rich[ idnx ]), ls = '--', color = 'r', alpha = 0.5,)

plt.xlabel('$ \\lambda $')
plt.ylabel('# of cluster')
plt.yscale('log')
plt.legend( loc = 1)
plt.savefig('z_in_0.2_0.3_rich_compare.png', dpi = 300)
plt.close()


plt.figure()
plt.title('clusters in $ 0.2 \\sim z \\sim 0.3 $')
plt.hist( lo_z[idmx], bins = 50, density = False, color = 'b', alpha = 0.5, label = 'low $M_{\\ast}$ [%d]' % len(lo_z[idmx]),)
plt.hist( hi_z[idnx], bins = 50, density = False, color = 'r', alpha = 0.5, label = 'high $M_{\\ast}$ [%d]' % len(hi_z[idnx]),)

plt.axvline( x = np.mean(lo_z[ idmx ]), ls = '-', color = 'b', alpha = 0.5, label = 'mean',)
plt.axvline( x = np.median(lo_z[ idmx ]), ls = '--', color = 'b', alpha = 0.5, label = 'median',)
plt.axvline( x = np.mean(hi_z[ idnx ]), ls = '-', color = 'r', alpha = 0.5,)
plt.axvline( x = np.median(hi_z[ idnx ]), ls = '--', color = 'r', alpha = 0.5,)

plt.xlabel('$ z $')
plt.ylabel('# of cluster')
plt.legend( loc = 2)
plt.savefig('z_in_0.2_0.3_z_compare.png', dpi = 300)
plt.close()
'''

### match to normal images (spec_sample)
for ll in range( 3 ):

	#dat = pds.read_csv( load + 'cat_select/selection_by_flux_pdf/cluster_tot-%s-band_norm-img_cat.csv' % band[ll],) ##previous
	dat = pds.read_csv( load + 'cat_select/selection_by_flux_pdf/%s-band_tot_remain_cat.csv' % band[ll],)
	ra, dec, z = np.array(dat['ra']), np.array(dat['dec']), np.array(dat['z'])
	clus_x, clus_y = np.array(dat['bcg_x']), np.array(dat['bcg_y'])

	sf_len = 5
	f2str = '%.' + '%df' % sf_len

	## low mass sample
	out_ra = [ f2str % ll for ll in lo_ra]
	out_dec = [ f2str % ll for ll in lo_dec]

	lis_ra, lis_dec, lis_z, lis_x, lis_y = cat_match_func(
			out_ra, out_dec, ra, dec, z, clus_x, clus_y, sf_len, id_choice = True,)

	print( 'band, %s' % band[ll] )
	print( len(lis_z) )
	'''
	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y',]
	values = [lis_ra, lis_dec, lis_z, lis_x, lis_y]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( 'low_BCG_star-Mass_%s-band_BCG-pos_cat.csv' % band[ll],)
	'''
	## high mass sample
	out_ra_1 = [ f2str % ll for ll in hi_ra]
	out_dec_1 = [ f2str % ll for ll in hi_dec]

	lis_ra_1, lis_dec_1, lis_z_1, lis_x_1, lis_y_1 = cat_match_func(
			out_ra_1, out_dec_1, ra, dec, z, clus_x, clus_y, sf_len, id_choice = True,)

	print( len(lis_z_1) )
	'''
	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y',]
	values = [ lis_ra_1, lis_dec_1, lis_z_1, lis_x_1, lis_y_1 ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( 'high_BCG_star-Mass_%s-band_BCG-pos_cat.csv' % band[ll],)
	'''

### match to photo_z sample
'''
for kk in range( 3 ):

	#ref_file = '/home/xkchen/mywork/ICL/data/' + 'cat_select/selection_by_hand/%s_band_sky_catalog.csv' % band[kk]
	ref_file = '/home/xkchen/mywork/ICL/data/cat_select/SDSS_spec-sample_sql_cat.csv'

	out_ra = ['%.3f' % ll for ll in lo_ra]
	out_dec = ['%.3f' % ll for ll in lo_dec]
	out_z = ['%.3f' % ll for ll in lo_z]
	ra_0, dec_0, z_0, order_0 = simple_match(out_ra, out_dec, out_z, ref_file,)

	print( 'low,', len(ra_0) )


	out_ra = ['%.3f' % ll for ll in hi_ra]
	out_dec = ['%.3f' % ll for ll in hi_dec]
	out_z = ['%.3f' % ll for ll in hi_z]
	ra_1, dec_1, z_1, order_1 = simple_match(out_ra, out_dec, out_z, ref_file,)

	print( 'high', len(ra_1) )

'''

cat_file = '/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits'
sf_len = 5

data = fits.open( cat_file )
goal_data = data[1].data

RA = np.array(goal_data.RA)
DEC = np.array(goal_data.DEC)
ID = np.array(goal_data.OBJID)

z_spec = np.array(goal_data.Z_SPEC)
z_phot = np.array(goal_data.Z_LAMBDA)

lamda = np.array(goal_data.LAMBDA)
r_Mag_bcgs = np.array(goal_data.MODEL_MAG_R)


idx0 = (z_spec <= 0.3) & (z_spec >= 0.2)
print( np.sum(idx0) )

idx1 = (z_phot <= 0.3) & (z_phot >= 0.2)
print( np.sum(idx1) )

## rebuild the z_spec and z_photo list
keys = [ 'ra', 'dec', 'z', 'rich', 'bcg_r-Mag', 'objID',]
values = [ RA[idx0], DEC[idx0], z_spec[idx0], lamda[idx0], r_Mag_bcgs[idx0], ID[idx0] ]

fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
data.to_csv('/home/xkchen/mywork/ICL/data/photo_cat/' + 'redMapper_z-spec_cat.csv')


keys = [ 'ra', 'dec', 'z', 'rich', 'bcg_r-Mag', 'objID',]
values = [ RA[idx1], DEC[idx1], z_phot[idx1], lamda[idx1], r_Mag_bcgs[idx1], ID[idx1] ]

fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
data.to_csv('/home/xkchen/mywork/ICL/data/photo_cat/' + 'redMapper_z-photo_cat.csv')


'''
#out_file = '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/match_compare/' + 'low_star-Mass_z-spec_macth_cat.csv'
#match_func( lo_ra, lo_dec, lo_z, cat_file, out_file, sf_len,)
out_file = '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/match_compare/' + 'low_star-Mass_z-phot_macth_cat.csv'
match_func( lo_ra, lo_dec, lo_z, cat_file, out_file, sf_len, id_spec = False,)

#out_file = '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/match_compare/' + 'high_star-Mass_z-spec_macth_cat.csv'
#match_func( hi_ra, hi_dec, hi_z, cat_file, out_file, sf_len,)
out_file = '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/match_compare/' + 'high_star-Mass_z-phot_macth_cat.csv'
match_func( hi_ra, hi_dec, hi_z, cat_file, out_file, sf_len, id_spec = False,)
'''

'''
dat_0 = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/match_compare/' + 
					'low_star-Mass_z-spec_macth_cat.csv')
#dat_0 = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/match_compare/' + 
#					'high_star-Mass_z-spec_macth_cat.csv')
ra_0 = np.array(dat_0['ra'])
print( 'spec match', len(ra_0) )


dat_1 = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/match_compare/' + 
					'low_star-Mass_z-phot_macth_cat.csv')
#dat_1 = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/match_compare/' + 
#					'high_star-Mass_z-phot_macth_cat.csv')
ra_1 = np.array(dat_1['ra'])
print( 'phot match', len(ra_1) )
'''

