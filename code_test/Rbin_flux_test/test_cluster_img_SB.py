import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord

from list_shuffle import find_unique_shuffle_lists as find_list
from light_measure import light_measure_Z0_weit
from light_measure import light_measure_rn_Z0_weit
from light_measure import jack_SB_func

from img_jack_stack import jack_main_func, zref_lim_SB_adjust_func
from img_jack_stack import SB_pros_func

from img_sky_jack_stack import sky_jack_main_func
from img_jack_stack import aveg_stack_img
from img_stack import stack_func
from img_edg_cut_stack import cut_stack_func
from fig_out_module import zref_BCG_pos_func

# cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
band = ['r', 'g', 'i']

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'


##### r-band img stacking, (combine low and high BCG star Mass sample), in angle coordinate
kk = 0

band_str = band[0]

lo_dat = pds.read_csv(load + 'img_cat/low_BCG_star-Mass_%s-band_remain_cat.csv' % band_str,)
lo_ra, lo_dec, lo_z = np.array(lo_dat.ra), np.array(lo_dat.dec), np.array(lo_dat.z)
lo_imgx, lo_imgy = np.array(lo_dat.bcg_x), np.array(lo_dat.bcg_y)

hi_dat = pds.read_csv(load + 'img_cat/high_BCG_star-Mass_%s-band_remain_cat.csv' % band_str,)
hi_ra, hi_dec, hi_z = np.array(hi_dat.ra), np.array(hi_dat.dec), np.array(hi_dat.z)
hi_imgx, hi_imgy = np.array(hi_dat.bcg_x), np.array(hi_dat.bcg_y)

ra = np.r_[ lo_ra, hi_ra ]
dec = np.r_[ lo_dec, hi_dec ]
z = np.r_[ lo_z, hi_z ]

clus_x = np.r_[ lo_imgx, hi_imgx ]
clus_y = np.r_[ lo_imgy, hi_imgy ]

print('N_sample = ', len(ra),)
print('band = %s' % band_str,)

id_cen = 0
n_rbins = 100
N_bin = 30

d_file = home + 'mask_imgs/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_30-FWHM-ov2.fits'

# XXX
sub_img = load + 'stack/cluster_%s-band' % band[kk] + '_sub-%d_img.h5'
sub_pix_cont = load + 'stack/cluster_%s-band' % band[kk] + '_sub-%d_pix-cont.h5'
sub_sb = load + 'stack/cluster_%s-band' % band[kk] + '_sub-%d_SB-pro.h5'
# XXX

J_sub_img = load + 'stack/cluster_%s-band_BCG-stack' % band[kk] + '_jack-sub-%d_img.h5'
J_sub_pix_cont = load + 'stack/cluster_%s-band_BCG-stack' % band[kk] + '_jack-sub-%d_pix-cont.h5'
J_sub_sb = load + 'stack/cluster_%s-band_BCG-stack' % band[kk] + '_jack-sub-%d_SB-pro.h5'

jack_img = load + 'stack/cluster_%s-band_BCG-stack' % band[kk] + '_Mean_jack_img.h5'
jack_cont_arr = load + 'stack/cluster_%s-band_BCG-stack' % band[kk] + '_Mean_jack_pix-cont.h5'
jack_SB_file = load + 'stack/cluster_%s-band_BCG-stack' % band[kk] + '_Mean_jack_SB-pro.h5'

jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[kk], sub_img, sub_pix_cont, sub_sb,
	J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr, id_cut = False, N_edg = None, id_Z0 = True,
	z_ref = None, id_S2N = False, S2N = None, id_sub = True, edg_bins = None,)

print('angle stacking finished !')


