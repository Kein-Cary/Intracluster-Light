import matplotlib as mpl
mpl.use('Agg')
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
from scipy.ndimage import gaussian_filter
from scipy import interpolate as interp
# pipe-code
from img_pre_selection import cat_match_func, get_mu_sigma
from fig_out_module import cc_grid_img, grid_img
from fig_out_module import zref_BCG_pos_func
from img_mask import source_detect_func, mask_func
from img_mask_adjust import adjust_mask_func

from img_jack_stack import jack_main_func
from img_jack_stack import SB_pros_func
from light_measure import jack_SB_func
from img_jack_stack import lim_SB_pros_func, zref_lim_SB_adjust_func

from img_condition_select import img_condition_select_func
from cc_block_select import diffuse_identi_func
from img_pre_selection import map_mu_sigma_func
from img_resample import resamp_func
from img_jack_stack import aveg_stack_img

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

# constant
rad2asec = U.rad.to(U.arcsec)
band = ['r', 'g', 'i',]
mag_add = np.array([0, 0, 0])
pixel = 0.396

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'
z_ref = 0.25

### BCG-stacking case
kk = 0

S2N = 5 # set for pixel-count ratio limit
edg_bins = 4

id_cen = 0
n_rbins = 100
N_bin = 30

'''
############ angle coordinate
dat = pds.read_csv(load + 'random_cat/12_21/random_r-band_tot_remain_cat.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

d_file = home + 'tmp_stack/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'
# XXX
sub_img = load + 'stack/random_%s-band' % band[kk] + '_sub-%d_img.h5'
sub_pix_cont = load + 'stack/random_%s-band' % band[kk] + '_sub-%d_pix-cont.h5'
sub_sb = load + 'stack/random_%s-band' % band[kk] + '_sub-%d_SB-pro.h5'
# XXX

J_sub_img = load + 'stack/random_%s-band_BCG-stack' % band[kk] + '_jack-sub-%d_img.h5'
J_sub_pix_cont = load + 'stack/random_%s-band_BCG-stack' % band[kk] + '_jack-sub-%d_pix-cont.h5'
J_sub_sb = load + 'stack/random_%s-band_BCG-stack' % band[kk] + '_jack-sub-%d_SB-pro.h5'

jack_img = load + 'stack/random_%s-band_BCG-stack' % band[kk] + '_Mean_jack_img.h5'
jack_cont_arr = load + 'stack/random_%s-band_BCG-stack' % band[kk] + '_Mean_jack_pix-cont.h5'
jack_SB_file = load + 'stack/random_%s-band_BCG-stack' % band[kk] + '_Mean_jack_SB-pro.h5'

jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[kk], sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
	id_cut = False, N_edg = None, id_Z0 = True, z_ref = None, id_S2N = False, S2N = None, id_sub = True, edg_bins = None,)
'''

'''
Nedg = 500
# XXX
sub_img = load + 'stack/random_%s-band' % band[kk] + '_sub-%d_img' + '_cut-%d.h5' % Nedg
sub_pix_cont = load + 'stack/random_%s-band' % band[kk] + '_sub-%d_pix-cont' + '_cut-%d.h5' % Nedg
sub_sb = load + 'stack/random_%s-band' % band[kk] + '_sub-%d_SB-pro' + '_cut-%d.h5' % Nedg
# XXX

J_sub_img = load + 'stack/random_%s-band_BCG-stack' % band[kk] + '_jack-sub-%d_img' + '_cut-%d.h5' % Nedg
J_sub_pix_cont = load + 'stack/random_%s-band_BCG-stack' % band[kk] + '_jack-sub-%d_pix-cont' + '_cut-%d.h5' % Nedg
J_sub_sb = load + 'stack/random_%s-band_BCG-stack' % band[kk] + '_jack-sub-%d_SB-pro' + '_cut-%d.h5' % Nedg

jack_img = load + 'stack/random_%s-band_BCG-stack' % band[kk] + '_Mean_jack_img' + '_cut-%d.h5' % Nedg
jack_cont_arr = load + 'stack/random_%s-band_BCG-stack' % band[kk] + '_Mean_jack_pix-cont' + '_cut-%d.h5' % Nedg
jack_SB_file = load + 'stack/random_%s-band_BCG-stack' % band[kk] + '_Mean_jack_SB-pro' + '_cut-%d.h5' % Nedg

jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[kk], sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
	id_cut = True, N_edg = 500, id_Z0 = True, z_ref = None, id_S2N = False, S2N = None, id_sub = True, edg_bins = None,)

print('finished stacking 1')
'''

################ physical coordinate
dat = pds.read_csv( load + 'random_cat/12_21/random_r-band_tot_remain_zref_BCG-pos.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

d_file = home + 'tmp_stack/pix_resample/random_resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'
# XXX
sub_img = load + 'stack/random_%s-band' % band[kk] + '_sub-%d_img.h5'
sub_pix_cont = load + 'stack/random_%s-band' % band[kk] + '_sub-%d_pix-cont.h5'
sub_sb = load + 'stack/random_%s-band' % band[kk] + '_sub-%d_SB-pro.h5'
# XXX

J_sub_img = load + 'stack/random_%s-band_BCG-stack' % band[kk] + '_jack-sub-%d_img_z-ref.h5'
J_sub_pix_cont = load + 'stack/random_%s-band_BCG-stack' % band[kk] + '_jack-sub-%d_pix-cont_z-ref.h5'
J_sub_sb = load + 'stack/random_%s-band_BCG-stack' % band[kk] + '_jack-sub-%d_SB-pro_z-ref.h5'

jack_img = load + 'stack/random_%s-band_BCG-stack' % band[kk] + '_Mean_jack_img_z-ref.h5'
jack_cont_arr = load + 'stack/random_%s-band_BCG-stack' % band[kk] + '_Mean_jack_pix-cont_z-ref.h5'
jack_SB_file = load + 'stack/random_%s-band_BCG-stack' % band[kk] + '_Mean_jack_SB-pro_z-ref.h5'

jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[kk], sub_img, sub_pix_cont, sub_sb,
	J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr, id_cut = False, N_edg = None, id_Z0 = False,
	z_ref = z_ref, id_S2N = False, S2N = None, id_sub = True, edg_bins = None,)

print('physical stack finished !')

