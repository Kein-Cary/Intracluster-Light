"""
img selection, sample overview...
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from scipy import signal
from scipy import interpolate as interp

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

def hist_scatter_fig(x_arr, y_arr, line_name, x_label, y_label, panel_name, out_file):

	fig = plt.figure( figsize = (7, 7) )
	fig.suptitle( panel_name )
	gs = gridspec.GridSpec(2, 2, width_ratios = (5, 1), height_ratios = (1,5),)

	ax = fig.add_subplot(gs[1, 0])
	ax_histx = fig.add_subplot(gs[0, 0])
	ax_histy = fig.add_subplot(gs[1, 1])

	ax.scatter(x_arr[0], y_arr[0], s = 20, marker = 'o', color = 'b', alpha = 0.5, label = line_name[0],)
	ax.scatter(x_arr[1], y_arr[1], s = 20, marker = 's', color = 'r', alpha = 0.5, label = line_name[1],)
	ax.set_ylim(9, 12)
	ax.set_xlabel( x_label )
	ax.set_ylabel( y_label )
	ax.legend(loc = 4,)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in',)

	ax_histy.hist(y_arr[0], bins = 50, density = True, color = 'b', alpha = 0.5, orientation = 'horizontal')
	ax_histy.hist(y_arr[1], bins = 50, density = True, color = 'r', alpha = 0.5, orientation = 'horizontal')
	ax_histy.tick_params(axis = 'both', which = 'both', direction = 'in',)
	ax_histy.set_ylim( ax.get_ylim() )
	ax_histy.set_yticks([])

	ax_histx.hist(x_arr[0], bins = 50, density = True, color = 'b', alpha = 0.5,)
	ax_histx.hist(x_arr[1], bins = 50, density = True, color = 'r', alpha = 0.5,)
	ax_histx.tick_params(axis = 'both', which = 'both', direction = 'in',)
	ax_histx.set_xlim( ax.get_xlim() )
	ax_histx.set_xticks([])

	plt.subplots_adjust(hspace = 0, wspace = 0)
	plt.savefig( out_file, dpi = 300)
	plt.close()

	return

def simple_match(ra_lis, dec_lis, ref_file,):

	ref_dat = pds.read_csv( ref_file )
	tt_ra, tt_dec, tt_z = np.array(ref_dat.ra), np.array(ref_dat.dec), np.array(ref_dat.z)
	tt_rich, tt_M_star = np.array(ref_dat.rich), np.array(ref_dat.lg_Mass)

	dd_ra, dd_dec, dd_z = [], [], []
	dd_Mass, dd_rich = [], []

	for kk in range( len(tt_z) ):
		if ('%.5f' % tt_ra[kk] in ra_lis) & ('%.5f' % tt_dec[kk] in dec_lis):
			dd_ra.append( tt_ra[kk])
			dd_dec.append( tt_dec[kk])
			dd_z.append( tt_z[kk])

			dd_Mass.append( tt_M_star[kk] )
			dd_rich.append( tt_rich[kk] )
		else:
			continue

	return dd_ra, dd_dec, dd_z, dd_Mass, dd_rich

z_ref = 0.25

#****************************#
load = '/home/xkchen/mywork/ICL/data/'

lo_dat = pds.read_csv(load + 'BCG_stellar_mass_cat/low_star-Mass_cat.csv')
lo_ra, lo_dec, lo_z = np.array(lo_dat.ra), np.array(lo_dat.dec), np.array(lo_dat.z)
lo_rich, lo_M_star = np.array(lo_dat.rich), np.array(lo_dat.lg_Mass)

hi_dat = pds.read_csv(load + 'BCG_stellar_mass_cat/high_star-Mass_cat.csv')
hi_ra, hi_dec, hi_z = np.array(hi_dat.ra), np.array(hi_dat.dec), np.array(hi_dat.z)
hi_rich, hi_M_star = np.array(hi_dat.rich), np.array(hi_dat.lg_Mass)

### match to normal images
for ll in range( 3 ):

	dat = pds.read_csv( load + 'cat_select/cluster_tot-%s-band_norm-img_cat.csv' % band[ll],)
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

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y',]
	values = [lis_ra, lis_dec, lis_z, lis_x, lis_y]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( load + 'BCG_stellar_mass_cat/band_match/' + 'low_BCG_star-Mass_%s-band_BCG-pos_cat.csv' % band[ll],)

	## high mass sample
	out_ra_1 = [ f2str % ll for ll in hi_ra]
	out_dec_1 = [ f2str % ll for ll in hi_dec]

	lis_ra_1, lis_dec_1, lis_z_1, lis_x_1, lis_y_1 = cat_match_func(
			out_ra_1, out_dec_1, ra, dec, z, clus_x, clus_y, sf_len, id_choice = True,)

	print( len(lis_z_1) )

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y',]
	values = [ lis_ra_1, lis_dec_1, lis_z_1, lis_x_1, lis_y_1 ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)

	data.to_csv( load + 'BCG_stellar_mass_cat/band_match/' + 'high_BCG_star-Mass_%s-band_BCG-pos_cat.csv' % band[ll],)

raise

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
'''
### BCG pos at z_ref

for kk in range( 3 ):

	for ll in range( 2 ):
		cat_file = load + 'BCG_stellar_mass_cat/band_match/%s_%s-band_BCG-pos_cat.csv' % (cat_lis[ll], band[kk])
		out_file = '/home/xkchen/mywork/ICL/%s_%s-band_resamp-BCG-pos.csv' % (cat_lis[ll], band[kk])

		#cat_file = load + 'BCG_stellar_mass_cat/band_match/%s_%s-band_remain_cat.csv' % (cat_lis[ll], band[kk])
		#out_file = '/home/xkchen/mywork/ICL/%s_%s-band_remain_cat_resamp_BCG-pos.csv' % (cat_lis[ll], band[kk])

		zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)
'''

'''
### (img_mu, img_sigma) for image selection

#mu_file = load + 'cat_select/g_band_3133-img_mu-sigma.csv'
mu_file = load + 'cat_select/i_band_2871-img_mu-sigma.csv'

dat = pds.read_csv( mu_file )
ra, dec, z = np.array(dat['ra']), np.array(dat['dec']), np.array(dat['z'])
img_mu, img_sigm = np.array(dat['img_mu']), np.array(dat['img_sigma'])
cen_mu, cen_sigm = np.array(dat['cen_mu']), np.array(dat['cen_sigma'])


#cc_dat = pds.read_csv( load + 'cat_select/cluster_tot-g-band_norm-img_cat.csv')
cc_dat = pds.read_csv( load + 'cat_select/cluster_tot-i-band_norm-img_cat.csv')
cc_ra, cc_dec, cc_z = np.array(cc_dat['ra']), np.array(cc_dat['dec']), np.array(cc_dat['z'])
clus_x, clus_y = np.array(cc_dat['bcg_x']), np.array(cc_dat['bcg_y'])

keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'cen_mu', 'cen_sigma', 'img_mu', 'img_sigma',]
values = [ra, dec, z, clus_x, clus_y, cen_mu, cen_sigm, img_mu, img_sigm]
fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
#data.to_csv( 'g_band_3133-img_mu-sigma.csv' )
data.to_csv( 'i_band_2871-img_mu-sigma.csv' )

mu_sigma_lis = [load + 'cat_select/r_band_3100-img_mu-sigma.csv', 
				load + 'cat_select/g_band_3133-img_mu-sigma.csv', 
				load + 'cat_select/i_band_2871-img_mu-sigma.csv']

for ll in range( 3 ):

	#cat_file = load + 'BCG_stellar_mass_cat/low_BCG_star-Mass_%s-band_BCG-pos_cat.csv' % band[ll]
	cat_file = load + 'BCG_stellar_mass_cat/high_BCG_star-Mass_%s-band_BCG-pos_cat.csv' % band[ll]

	ref_cat = mu_sigma_lis[ ll ]

	#out_file = 'low_BCG_star-Mass_%s-band_img-mean-sigm.csv' % band[ll]
	out_file = 'high_BCG_star-Mass_%s-band_img-mean-sigm.csv' % band[ll]

	get_mu_sigma(cat_file, ref_cat, out_file,)
'''

'''
## sample overview
for ll in range( 3 ):

	z_ref = 0.25
	pixel = 0.396

	ref_file = load + 'redmapper/redmapper_dr8_public_v6.3_catalog.fits'	

	cat_0 = pds.read_csv( load + 'BCG_stellar_mass_cat/band_match/low_BCG_star-Mass_%s-band_remain_cat.csv' % band[ll] )
	ra_0, dec_0, z_0 = np.array(cat_0.ra), np.array(cat_0.dec), np.array(cat_0.z)

	#out_file0 = 'low_BCG_star-Mass_%s-band_remain_cat_param-match.csv' % band[ll]
	#match_func(ra_0, dec_0, z_0, ref_file, out_file0)

	out_ra = ['%.5f' % ll for ll in ra_0 ]
	out_dec = ['%.5f' % ll for ll in dec_0 ]

	copy_file = load + 'BCG_stellar_mass_cat/low_star-Mass_cat.csv'

	Mass_0, rich_0 = simple_match(out_ra, out_dec, copy_file,)[3:]

	keys = ['ra', 'dec', 'z', 'M_star', 'rich',]
	values = [ra_0, dec_0, z_0, Mass_0, rich_0 ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv('low_BCG_star-Mass_%s-band_simple-match.csv' % band[ll] )


	cat_1 = pds.read_csv( load + 'BCG_stellar_mass_cat/band_match/high_BCG_star-Mass_%s-band_remain_cat.csv' % band[ll] )
	ra_1, dec_1, z_1 = np.array(cat_1.ra), np.array(cat_1.dec), np.array(cat_1.z)

	#out_file1 = 'high_BCG_star-Mass_%s-band_remain_cat_param-match.csv' % band[ll]
	#match_func(ra_1, dec_1, z_1, ref_file, out_file1)

	out_ra = ['%.5f' % ll for ll in ra_1 ]
	out_dec = ['%.5f' % ll for ll in dec_1 ]

	copy_file = load + 'BCG_stellar_mass_cat/high_star-Mass_cat.csv'

	Mass_1, rich_1 = simple_match(out_ra, out_dec, copy_file,)[3:]

	keys = ['ra', 'dec', 'z', 'M_star', 'rich',]
	values = [ra_1, dec_1, z_1, Mass_1, rich_1 ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv('high_BCG_star-Mass_%s-band_simple-match.csv' % band[ll] )

	x_arr = [np.log10(rich_0), np.log10(rich_1) ]
	y_arr = [Mass_0, Mass_1 ]
	line_name = ['low BCG $M_{\\ast}$', 'high BCG $M_{\\ast}$']
	x_label = 'lg$\\lambda$'
	y_label = 'lg$M_{\\ast}[M_{\\odot}]$'
	fig_name = '%s-band_cat_overview.png' % band[ll]
	plot_name = '%s band match sample' % band[ll]

	hist_scatter_fig(x_arr, y_arr, line_name, x_label, y_label, plot_name, fig_name,)
'''

### img selection based on (img_mu, img_sigma)

#**********# gravity
from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'

from cc_block_select import diffuse_identi_func

band = ['r', 'g', 'i']

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

m, n = divmod(3, cpus)
N_sub0, N_sub1 = m * rank, (rank + 1) * m
if rank == cpus - 1:
	N_sub1 += n
'''
for ll in range( 2 ):

	band_str = '%s' % np.array( band[N_sub0 : N_sub1] )[0]

	dat = pds.read_csv( load + 'img_cat/' + cat_lis[ll] + '_%s-band_BCG-pos_cat.csv' % band_str ) 
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

	d_file = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'
	mu_sigm_file = load + 'img_cat/' + cat_lis[ll] + '_%s-band_img-mean-sigm.csv' % band_str

	thres_S0, thres_S1 = 3, 5
	sigma = 6

	rule_file = '/home/xkchen/' + cat_lis[ll] + '_%s-band_rule-out_cat.csv' % band_str
	remain_file = '/home/xkchen/' + cat_lis[ll] + '_%s-band_remain_cat.csv' % band_str

	diffuse_identi_func( band_str, ra, dec, z, d_file, rule_file, remain_file, thres_S0, thres_S1, sigma, 
						mu_sigm_file, id_single = False, id_mode = True,)

print('Done !')
'''


'''
## total sample select
mu_sigm_lis = ['g_band_3133-img_mu-sigma.csv',
				'i_band_2871-img_mu-sigma.csv']

for kk in range( 2 ):

	dat = pds.read_csv(load + 'cluster_tot-%s-band_norm-img_cat.csv' % band[kk],)
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

	d_file = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'
	mu_sigm_file = load + mu_sigm_lis[kk]

	thres_S0, thres_S1 = 3, 5
	sigma = 6

	rule_file = '/home/xkchen/%s-band_tot_rule-out_cat.csv' % band[kk]
	remain_file = '/home/xkchen/%s-band_tot_remain_cat.csv' % band[kk]

	diffuse_identi_func( band[kk], ra, dec, z, d_file, rule_file, remain_file, thres_S0, thres_S1, sigma, 
						mu_sigm_file, id_single = False, id_mode = True,)

print('finished !')
'''

## selection on BCG-star-Mass sample
band = ['r', 'g', 'i']

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

m, n = divmod(3, cpus)
N_sub0, N_sub1 = m * rank, (rank + 1) * m
if rank == cpus - 1:
	N_sub1 += n

for ll in range( 2 ):

	band_str = '%s' % np.array( band[N_sub0 : N_sub1] )[0]

	dat = pds.read_csv( load + 'img_cat/' + cat_lis[ll] + '_%s-band_BCG-pos_cat.csv' % band_str ) 
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

	d_file = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'
	mu_sigm_file = load + 'img_cat/' + cat_lis[ll] + '_%s-band_img-mean-sigm.csv' % band_str

	thres_S0, thres_S1 = 3, 5
	sigma = 6

	rule_file = '/home/xkchen/' + cat_lis[ll] + '_%s-band_rule-out_cat.csv' % band_str
	remain_file = '/home/xkchen/' + cat_lis[ll] + '_%s-band_remain_cat.csv' % band_str

	cen_L = 500
	N_step = 200

	diffuse_identi_func( band_str, ra, dec, z, d_file, rule_file, remain_file, thres_S0, thres_S1, sigma, 
						mu_sigm_file, cen_L, N_step, id_single = False, id_mode = True,)

print('Done !')


