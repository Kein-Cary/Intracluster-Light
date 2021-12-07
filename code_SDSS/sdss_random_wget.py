import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import h5py
import random
import numpy as np
import pandas as pds
import astropy.units as U
import astropy.constants as C
import astroquery.sdss as asds
import astropy.io.fits as fits

from io import StringIO
import changds
import wget as wt

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
##### section 1: read the redmapper data
rand_data = fits.getdata(load + 'redmapper/redmapper_dr8_public_v6.3_randoms.fits')
RA = rand_data.RA
DEC = rand_data.DEC
Z = rand_data.Z
LAMBDA = rand_data.LAMBDA

idx = (Z >= 0.2) & (Z <= 0.3)
# select the nearly universe
z_eff = Z[idx]
ra_eff = RA[idx]
dec_eff = DEC[idx]
lamda_eff = LAMBDA[idx]

zN = 5000
np.random.seed(1)
tt0 = np.random.choice(len(z_eff), size = zN, replace = False)
z = z_eff[tt0]
ra = ra_eff[tt0]
dec = dec_eff[tt0]
rich = lamda_eff[tt0]
# save the random catalogue
tmp_array = np.array([ra, dec, z, rich])
with h5py.File(load + 'mpi_h5/redMapper_rand_cat.h5', 'w') as f:
    f['a'] = np.array(tmp_array)
with h5py.File(load + 'mpi_h5/redMapper_rand_cat.h5') as f:
    for  kk in range(len(tmp_array)):
        f['a'][kk,:] = tmp_array[kk,:]

keys = ['ra', 'dec', 'z', 'rich']
values = [ra, dec, z, rich]
fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
data.to_csv(load + 'selection/redMapper_rand_cat.csv')

# calculate the angular size 
size_cluster = 2. # assumptiom: cluster size is 2.Mpc/h
from ICL_angular_diameter_reshift import mark_by_self
from ICL_angular_diameter_reshift import mark_by_plank
A_size, A_d= mark_by_self(z, size_cluster)
view_d = A_size * U.rad

#### section 2: cite the data and save fits figure
from astroquery.sdss import SDSS
from astropy import coordinates as coords
#from astroML.plotting import setup_text_plots
from astropy.table import Table
R_A = 0.5 * view_d.to(U.arcsec) # angular radius in angular second unit
band = ['u','g','r','i','z']

for k in range(4323, zN):
    pos = coords.SkyCoord('%fd %fd'%(ra[k],dec[k]), frame='icrs')
    try:
    # set for solve the ReadtimeOut error
        xid = SDSS.query_region(pos, spectro = False, radius = R_A[k], timeout = None)
        # for galaxy, don't take spectra into account
        name = xid.colnames
        aa = np.array([xid['ra'],xid['dec']])
        da = np.sqrt((aa[0,:]-ra[k])**2+(aa[1,:]-dec[k])**2)
        # select the goal region
        dl = da.tolist()
        pl = dl.index(np.min(da))
        s_bgn = changds.chidas_int(xid[pl][3],6)
        s_end = changds.chidas_int(xid[pl][6],4)
        # change the list as the astropy table type
        for q in range(len(band)):
            url_road = 'http://data.sdss.org/sas/dr12/boss/photoObj/frames/%.0f/%.0f/%.0f/frame-%s-%s-%.0f-%s.fits.bz2'%\
            (xid[pl][4],xid[pl][3],xid[pl][5],band[q],s_bgn,xid[pl][5],s_end)
            '''
            out_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            (band[q],ra[k],dec[k],z[k])
            '''
            out_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            (band[q], ra[k], dec[k], z[k])

            wt.download(url_road, out_file)
        print('**********-----')
        print('finish--',k/len(z))

    except KeyError:
        # save the "exception" case
        doc = open('/mnt/ddnfs/data_users/cxkttwl/PC/no_match_random.txt', 'w')
        s = '\n k = %d, ra%.3f, dec%.3f, z%.3f \n' % (k, ra[k], dec[k], z[k])
        print(s, file = doc)
        doc.close()
        continue

