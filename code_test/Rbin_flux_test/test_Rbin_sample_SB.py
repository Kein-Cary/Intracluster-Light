import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy.stats import binned_statistic as binned

import glob

def R_bins_mean_medi(Nr, r_bins, aveg_f_file, samp_f_file, out_file_0, out_file_1, pixel,):

	tmp_r, tmp_mean, tmp_medi, tmp_std = [], [], [], []
	tmp_samp_mean, tmp_samp_medi, tmp_samp_std = [], [], []

	for tt in range( Nr ):

		tmp_r.append( phy_r[tt] )

		with h5py.File( aveg_f_file % r_bins[tt], 'r') as f:
			dt_flux = np.array( f['flux'])
			dt_cont = np.array( f['pix_count'])

		id_nan = np.isnan( dt_flux )
		dt_flux = dt_flux[id_nan == False]
		dt_cont = dt_cont[id_nan == False]
		dt_cont = dt_cont.astype( np.int32 )

		lis_f = []
		for dd in range( len(dt_flux) ):
			rep_arr = np.ones( dt_cont[dd], dtype = np.float32) * dt_flux[dd]
			lis_f.append( rep_arr)

		F_arr = np.hstack(lis_f) / pixel**2

		tmp_mean.append( np.mean( F_arr ) )
		tmp_medi.append( np.median( F_arr ) )
		tmp_std.append( np.std(F_arr) )

		with h5py.File( samp_f_file % r_bins[tt], 'r') as f:
			dt_flux_s = np.array( f['pix_flux'])

		idnn = np.isnan( dt_flux_s )
		dt_flux_s = dt_flux_s[ idnn == False] / pixel**2

		tmp_samp_mean.append( np.mean(dt_flux_s) )
		tmp_samp_medi.append( np.median(dt_flux_s) )
		tmp_samp_std.append( np.std(dt_flux_s) )

	tmp_r = np.array( tmp_r )
	tmp_mean = np.array( tmp_mean )
	tmp_medi = np.array( tmp_medi )
	tmp_std = np.array( tmp_std )

	tmp_samp_mean = np.array( tmp_samp_mean )
	tmp_samp_medi = np.array( tmp_samp_medi )
	tmp_samp_std = np.array( tmp_samp_std )

	keys = ['R_kpc', 'mean_sb', 'medi_sb', 'std_sb']
	values = [ tmp_r, tmp_mean, tmp_medi, tmp_std ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame(fill)
	out_data.to_csv( out_file_0 )

	keys = ['R_kpc', 'mean_sb', 'medi_sb', 'std_sb']
	values = [ tmp_r, tmp_samp_mean, tmp_samp_medi, tmp_samp_std ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame(fill)
	out_data.to_csv( out_file_1 )

	return

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

load = '/home/xkchen/project/stack/'
home = '/home/xkchen/'

### cluster
cdat = pds.read_csv('/home/xkchen/project/stack/com-BCG-star-Mass_Mean_f_pros_z-ref.csv')
phy_r = np.array( cdat['R_kpc'] )
flux_r = np.array( cdat['f_mean'] )
npix = np.array( cdat['N_pix'])

idNul = npix > 0
phy_r = phy_r[idNul]
Nr = len( phy_r )

tmp_r, tmp_mean, tmp_medi, tmp_std = [], [], [], []
tmp_samp_mean, tmp_samp_medi, tmp_samp_std = [], [], []

aveg_f_file = '/home/xkchen/project/stack/phyR_stack/com-BCG-star-Mass_%.3f-kpc_flux-arr_z-ref.h5'
samp_f_file = '/home/xkchen/project/stack/phyR_stack/com-BCG-star-Mass_%.3f-kpc_sample-img_flux_z-ref.h5'

aveg_out_file = '/home/xkchen/project/ICL/com-BCG-star-Mass_BCG-stack_SB-pros_z-ref.csv'
samp_out_file = '/home/xkchen/project/ICL/com-BCG-star-Mass_BCG-stack_sample-img_SB-pros_z-ref.csv'

R_bins_mean_medi(Nr, phy_r, aveg_f_file, samp_f_file, aveg_out_file, samp_out_file, pixel,)


### random
cdat = pds.read_csv('/home/xkchen/project/stack/random_BCG-stack_Mean_f_pros_z-ref.csv')
phy_r = np.array( cdat['R_kpc'] )
flux_r = np.array( cdat['f_mean'] )
npix = np.array( cdat['N_pix'])

idNul = npix > 0
phy_r = phy_r[idNul]
Nr = len( phy_r )

aveg_f_file = '/home/xkchen/project/stack/phyR_stack/random_BCG-stack_%.3f-kpc_flux-arr_z-ref.h5'
samp_f_file = '/home/xkchen/project/stack/phyR_stack/random_BCG-stack_%.3f-kpc_sample-img_flux_z-ref.h5'

aveg_out_file = '/home/xkchen/project/ICL/random_BCG-stack_SB-pros_z-ref.csv'
samp_out_file = '/home/xkchen/project/ICL/random_BCG-stack_sample-img_SB-pros_z-ref.csv'

R_bins_mean_medi(Nr, phy_r, aveg_f_file, samp_f_file, aveg_out_file, samp_out_file, pixel,)


for ll in range( 10 ):

	cdat = pds.read_csv('/home/xkchen/project/stack/random_rand-stack_Mean_f_pros_z-ref_%d-rank.csv' % ll)
	phy_r = np.array( cdat['R_kpc'] )
	flux_r = np.array( cdat['f_mean'] )
	npix = np.array( cdat['N_pix'])

	idNul = npix > 0
	phy_r = phy_r[idNul]
	Nr = len( phy_r )

	aveg_f_file = '/home/xkchen/project/stack/phyR_stack/random_rand-stack_%.3f-kpc_flux-arr_z-ref' + '_%d-rank.h5' % ll
	samp_f_file = '/home/xkchen/project/stack/phyR_stack/random_rand-stack_%.3f-kpc_sample-img_flux_z-ref' + '_%d-rank.h5' % ll

	aveg_out_file = '/home/xkchen/project/ICL/random_rand-stack_SB-pros_z-ref_%d-rank.csv' % ll
	samp_out_file = '/home/xkchen/project/ICL/random_rand-stack_sample-img_SB-pros_z-ref_%d-rank.csv' % ll

	R_bins_mean_medi(Nr, phy_r, aveg_f_file, samp_f_file, aveg_out_file, samp_out_file, pixel,)


##### random rando-stack, total mean, median

