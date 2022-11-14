"""
Use to combine satellite background image extraction stacking, including
------------------------------------------------------------------------

1). image extract for each shuffle ~ (PS: shuffle list is build on entire sample)
2). image stacking according to given stacking catalog
3). remove median process data, save the SB profile and jack-knife mean image for each shuffle

------------------------------------------------------------------------
pre-process for above
1). build shuffle list based on over all sample
2). build the stacking catalog of given sample bins

PS: for Background : shuffle align with frame
"""
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
from astropy.table import Table, QTable

from scipy import interpolate as interp
from scipy import integrate as integ

#.
import time
import subprocess as subpro

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

##.
from img_sat_BG_extract_tmp import zref_img_cut_func

##.
from img_sat_fast_stack import sat_img_fast_stack_func
from img_sat_fast_stack import sat_BG_fast_stack_func


### === ### constant
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)

band = ['r', 'g', 'i']
z_ref = 0.25
pixel = 0.396
a_ref = 1 / (z_ref + 1)



### === ### shell variables input~(kdx is loop for 20-shuffle random location)
import sys
kdx = sys.argv

list_order = np.int( kdx[1] )

if rank == 0:
	print( 'shufl-ID = ', list_order )


### === ### 


