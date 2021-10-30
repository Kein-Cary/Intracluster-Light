"""
measure surface mass profile with surface brightness profile of SDSS database
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

from BCG_SB_pro_stack import BCG_SB_pros_func
from fig_out_module import arr_jack_func

from tmp_color_to_mass import get_c2mass_func
from tmp_color_to_mass import jk_sub_SB_func, jk_sub_Mass_func
from tmp_color_to_mass import aveg_mass_pro_func
from tmp_color_to_mass import cumu_mass_func

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

def simple_match(ra_lis, dec_lis, z_lis, ref_file, id_choose = False,):

	ref_dat = pds.read_csv( ref_file )
	tt_ra, tt_dec, tt_z = np.array(ref_dat.ra), np.array(ref_dat.dec), np.array(ref_dat.z)

	dd_ra, dd_dec, dd_z = [], [], []
	order_lis = []

	for kk in range( len(tt_z) ):
		identi = ('%.3f' % tt_ra[kk] in ra_lis) * ('%.3f' % tt_dec[kk] in dec_lis) # * ('%.3f' % tt_z[kk] in z_lis)

		if id_choose == True:
			if identi == True:
				dd_ra.append( tt_ra[kk])
				dd_dec.append( tt_dec[kk])
				dd_z.append( tt_z[kk])
				order_lis.append( kk )

			else:
				continue
		else:
			if identi == True:
				continue
			else:
				dd_ra.append( tt_ra[kk])
				dd_dec.append( tt_dec[kk])
				dd_z.append( tt_z[kk])
				order_lis.append( kk )

	dd_ra = np.array( dd_ra)
	dd_dec = np.array( dd_dec)
	dd_z = np.array( dd_z)
	order_lis = np.array( order_lis )

	return order_lis

### === data load
def sdss_photo_pros():

	load = '/home/xkchen/fig_tmp/'
	home = '/home/xkchen/data/SDSS/'

	cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']

	N_samples = 30
	r_bins = np.logspace(0, 2.48, 25) # unit : kpc

	band_str = band[ rank ]

	if band_str == 'r':
		out_ra = [ '164.740', '141.265', ]
		out_dec = [ '11.637', '11.376', ]
		out_z = [ '0.298', '0.288', ]

	if band_str == 'g':
		out_ra = [ '206.511', '141.265', '236.438', ]
		out_dec = [ '38.731', '11.376', '1.767', ]
		out_z = [ '0.295', '0.288', '0.272', ]


	lo_dat = pds.read_csv( load + 'pkoffset_cat/%s_%s-band_photo-z-match_pk-offset_BCG-pos_cat_z-ref.csv' % (cat_lis[0], band_str),)
	lo_ra, lo_dec, lo_z = np.array(lo_dat.ra), np.array(lo_dat.dec), np.array(lo_dat.z)
	lo_imgx, lo_imgy = np.array(lo_dat.bcg_x), np.array(lo_dat.bcg_y)

	if band_str != 'i':
		ref_file = load + 'pkoffset_cat/%s_%s-band_photo-z-match_pk-offset_BCG-pos_cat_z-ref.csv' % (cat_lis[0], band_str)
		order_lis = simple_match( out_ra, out_dec, out_z, ref_file,)

		lo_ra, lo_dec, lo_z = lo_ra[order_lis], lo_dec[order_lis], lo_z[order_lis]
		lo_imgx, lo_imgy = lo_imgx[order_lis], lo_imgy[order_lis]

	hi_dat = pds.read_csv( load + 'pkoffset_cat/%s_%s-band_photo-z-match_pk-offset_BCG-pos_cat_z-ref.csv' % (cat_lis[1], band_str),)
	hi_ra, hi_dec, hi_z = np.array(hi_dat.ra), np.array(hi_dat.dec), np.array(hi_dat.z)
	hi_imgx, hi_imgy = np.array(hi_dat.bcg_x), np.array(hi_dat.bcg_y)

	if band_str != 'i':
		ref_file = load + 'pkoffset_cat/%s_%s-band_photo-z-match_pk-offset_BCG-pos_cat_z-ref.csv' % (cat_lis[1], band_str)
		order_lis = simple_match( out_ra, out_dec, out_z, ref_file,)

		hi_ra, hi_dec, hi_z = hi_ra[order_lis], hi_dec[order_lis], hi_z[order_lis]
		hi_imgx, hi_imgy = hi_imgx[order_lis], hi_imgy[order_lis]

	#... all cluster sample
	ra = np.r_[ lo_ra, hi_ra ]
	dec = np.r_[ lo_dec, hi_dec ]
	z = np.r_[ lo_z, hi_z ]

	imgx = np.r_[ lo_imgx, hi_imgx ]
	imgy = np.r_[ lo_imgy, hi_imgy ]

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
		out_file = '/home/xkchen/tmp_run/data_files/figs/total-sample_%s-band_jack-sub-%d_BCG_photo-SB_pros.csv' % (band_str, nn)
		BCG_SB_pros_func( band_str, set_z, set_ra, set_dec, pros_file, z_ref, out_file, r_bins)

	## mean of jackknife sample
	tmp_r, tmp_sb = [], []
	for nn in range( N_samples ):

		pro_dat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/total-sample_%s-band_jack-sub-%d_BCG_photo-SB_pros.csv' % (band_str, nn),)

		tt_r, tt_sb = np.array( pro_dat['R_ref'] ), np.array( pro_dat['SB_fdens'] )

		tmp_r.append( tt_r )
		tmp_sb.append( tt_sb )

	mean_R, mean_sb, mean_sb_err, lim_R = arr_jack_func( tmp_sb, tmp_r, N_samples)

	keys = [ 'R', 'aveg_sb', 'aveg_sb_err' ]
	values = [ mean_R, mean_sb, mean_sb_err ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/tmp_run/data_files/figs/total-sample_%s-band_Mean-jack_BCG_photo-SB_pros.csv' % band_str,)

	print( '%s band, done!' % band_str,)

	return

# sdss_photo_pros()

### === mass estimate
Dl_ref = Test_model.luminosity_distance( z_ref ).value

load = '/home/xkchen/tmp_run/data_files/figs/All_sample_BCG_pros/'

N_samples = 30

band_str = 'gri'
c_inv = False

fit_file = '/home/xkchen/tmp_run/data_files/figs/M2L_Lumi_selected/least-square_M-to-i-band-Lumi&color.csv'
'''
sub_file_item = ['R_ref', 'SB_fdens', 'SB_fdens_err']

sub_SB_file = load + 'aveg_sdss_SB/total-sample_%s-band_jack-sub-%d_BCG_photo-SB_pros.csv'
low_R_lim, up_R_lim = 1e0, 3e2
out_file = load + 'mass_estimate/total-sample_%s-band-based_' % band_str + 'jack-sub-%d_BCG_mass-Lumi.csv'

jk_sub_Mass_func( N_samples, band_str, sub_SB_file, low_R_lim, up_R_lim, out_file, Dl_ref, z_ref, 
	fit_file = fit_file, sub_SB_file_item = sub_file_item )

jk_sub_m_file = load + 'mass_estimate/total-sample_%s-band-based_' % band_str + 'jack-sub-%d_BCG_mass-Lumi.csv'
jk_aveg_m_file = load + 'mass_estimate/total-sample_%s-band-based_aveg-jack_BCG_mass-Lumi.csv' % band_str
lgM_cov_file = load + 'mass_estimate/total-sample_%s-band-based_aveg-jack_BCG_log-surf-mass_cov_arr.h5' % band_str
M_cov_file = load + 'mass_estimate/total-sample_%s-band-based_aveg-jack_BCG_surf-mass_cov_arr.h5' % band_str
aveg_mass_pro_func(N_samples, band_str, jk_sub_m_file, jk_aveg_m_file, lgM_cov_file, M_cov_file = M_cov_file)
'''
dat = pds.read_csv( load + 'mass_estimate/total-sample_%s-band-based_aveg-jack_BCG_mass-Lumi.csv' % band_str,)
aveg_R = np.array(dat['R'])
aveg_surf_m, aveg_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])
aveg_cumu_m, aveg_cumu_m_err = np.array(dat['cumu_mass']), np.array(dat['cumu_mass_err'])
aveg_lumi, aveg_lumi_err = np.array(dat['lumi']), np.array(dat['lumi_err'])

#... adjust as Kravtsov 2018 
aveg_R = aveg_R[:-1] + 0.
aveg_surf_m = aveg_surf_m[:-1] / 10**0.1
aveg_surf_m_err = aveg_surf_m_err[:-1]
aveg_lumi, aveg_lumi_err = aveg_lumi[:-1], aveg_lumi_err[:-1]


plt.figure()
plt.title('%s-based surface mass density profile' % band_str )
plt.plot( aveg_R, aveg_surf_m, ls = '-', color = 'r', alpha = 0.5,)
plt.fill_between( aveg_R, y1 = aveg_surf_m - aveg_surf_m_err, y2 = aveg_surf_m + aveg_surf_m_err, 
	color = 'r', alpha = 0.12,)

plt.xlim(1e0, 1e2)
plt.xscale('log')
plt.xlabel('R[kpc]', fontsize = 15)
plt.yscale('log')
plt.ylim(5e6, 2e9)
plt.legend( loc = 1, frameon = False, fontsize = 15,)
plt.ylabel('$\\Sigma_{\\ast} \; [M_{\\odot} / kpc^2]$', fontsize = 15,)
plt.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
plt.savefig('/home/xkchen/tot_%s-band_based_BCG_surface_mass.png' % band_str, dpi = 300)
plt.close()


plt.figure()
plt.plot( aveg_R, aveg_lumi, ls = '-', color = 'r', alpha = 0.5,)
plt.fill_between( aveg_R, y1 = aveg_lumi - aveg_lumi_err, y2 = aveg_lumi + aveg_lumi_err, 
	color = 'r', alpha = 0.12,)

plt.xlim(1e0, 1e2)
plt.xscale('log')
plt.xlabel('R[kpc]', fontsize = 15)
plt.yscale('log')
plt.ylim( 2e6, 1e9)
plt.legend( loc = 1, frameon = False, fontsize = 15,)
plt.ylabel('$L_{%s} [L_{\\odot} / kpc^2]$' % band_str[-1], fontsize = 15,)
plt.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
plt.savefig('/home/xkchen/tot_%s-band_based_BCG_surface_Lumi.png' % band_str, dpi = 300)
plt.close()
