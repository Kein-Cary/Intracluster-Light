import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import handy.scatter as hsc

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.ascii as asc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C

from dustmaps.sfd import SFDQuery
from extinction_redden import A_wave
from astropy.coordinates import SkyCoord
from astropy import cosmology as apcy

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()
import time

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Lsun = C.L_sun.value*10**7
# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.250 
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # zero point

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])
mag_add = np.array([0, 0, 0, -0.04, 0.02])
zopt = np.array([22.5, 22.5, 22.5, 22.46, 22.52])
sb_lim = np.array([24.5, 25, 24, 24.35, 22.9])
Rv = 3.1
sfd = SFDQuery()
'''
### if use dr7 star catalogue as B mask parameter, next cat need to add, and both the
### the follow sample clusters need use the photometric data in dr8 image data
csv_UN = pds.read_csv(load + 'No_star_query_match.csv')
except_ra_Nu = ['%.3f' % ll for ll in csv_UN['ra'] ]
except_dec_Nu = ['%.3f' % ll for ll in csv_UN['dec'] ]
except_z_Nu = ['%.3f' % ll for ll in csv_UN['z'] ]

csv_BAD = pds.read_csv(load + 'Bad_match_dr7_cat.csv')
Bad_ra = ['%.3f' % ll for ll in csv_BAD['ra'] ]
Bad_dec = ['%.3f' % ll for ll in csv_BAD['dec'] ]
Bad_z = ['%.3f' % ll for ll in csv_BAD['z'] ]
'''
def spec_cat():
	## this mask is mainly for g, r, i band
	cat_ra = ['328.403', '196.004', '140.502', '180.973', '168.198', '140.296', '180.610', 
			'182.343', '241.928', '10.814',  '247.330', '186.372', '14.652', '125.865', 
			'162.469', '199.052', '211.730', '118.804', '200.095', '5.781',   '202.887', 
			'117.585', '176.039', '180.996', '231.786', '189.718', '162.793', '12.742', 
			'190.946', '154.684', '329.477', '167.124', '325.814', '340.792', '236.778', 
			'238.323', '37.662',  '222.321', '145.632', '237.132', '242.433', '244.912', 
			'189.884', '146.241', '170.810', '191.012', '127.750', '231.949', '200.200', 
			'117.736', '151.358', ]

	cat_dec = ['17.695', '67.507', '51.922', '1.783', '2.503', '34.860', '12.262', 
			'65.523', '7.482',  '2.531',  '26.183', '0.726', '4.352',  '54.168', 
			'16.156', '11.363', '55.067', '30.775', '8.433',  '-9.291', '62.642', 
			'26.293', '51.969', '31.968', '50.305', '34.660', '9.066',  '-9.490', 
			'51.831', '60.833', '14.251', '29.294', '8.540',  '-9.211', '19.257', 
			'10.319', '-4.991', '35.293', '30.025', '11.364', '7.709',  '9.767',  
			'42.660', '7.254',  '18.236', '67.781', '13.509', '9.711',  '4.054',  
			'15.913', '50.978', ]

	cat_z = ['0.230', '0.221', '0.204', '0.237', '0.268', '0.238', '0.226', 
			'0.204', '0.224', '0.262', '0.223', '0.238', '0.282', '0.243', 
			'0.209', '0.267', '0.251', '0.287', '0.227', '0.296', '0.219', 
			'0.205', '0.287', '0.204', '0.278', '0.226', '0.221', '0.200', 
			'0.268', '0.200', '0.272', '0.215', '0.255', '0.266', '0.270', 
			'0.226', '0.292', '0.287', '0.271', '0.226', '0.218', '0.237', 
			'0.287', '0.299', '0.270', '0.275', '0.258', '0.245', '0.211', 
			'0.286', '0.228', ]

	keys = ['ra', 'dec', 'z']
	values = [cat_ra, cat_dec, cat_z]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(load + 'special_mask_cat.csv')

def spec_mask():

def spec_resamp():


def main():
	spec_cat()
	cat_sample = pds.read_csv(load + 'special_mask_cat.csv')
	ra, dec, z = cat_sample['ra'], cat_sample['dec'], cat_sample['z']

if __name__ == "__main__":
	main()