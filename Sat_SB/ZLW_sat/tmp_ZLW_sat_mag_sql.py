import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pds

import mechanize
from io import StringIO

import scipy.stats as sts
import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
import scipy.interpolate as interp

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable


###... cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)

band = ['r', 'g', 'i']
pixel = 0.396
z_ref = 0.25


### === combine all Wen+2015 catalog (cluster)
dat = pds.read_csv('/home/xkchen/mywork/Sat_SB/data/remap_W15_cat/ZLW_cat_15/redMaPPer_BCG-Mstar_map_clust-cat.csv')
sdss_ra, sdss_dec, sdss_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
W15_ra, W15_dec, W15_z = np.array( dat['ra_W15'] ), np.array( dat['dec_W15']), np.array( dat['z_W15'])

d_coord_s = SkyCoord( ra = sdss_ra * U.deg, dec = sdss_dec * U.deg )
d_coord_w = SkyCoord( ra = W15_ra * U.deg, dec = W15_dec * U.deg )


#. BCG-Mstar bin match
cat_0 = pds.read_csv('/home/xkchen/mywork/Sat_SB/data/remap_W15_cat/age_bin_map/redMaPPer_map_clust-cat.csv')
sd_ra_0, sd_dec_0, sd_z_0 = np.array( cat_0['ra'] ), np.array( cat_0['dec'] ), np.array( cat_0['z'] )
w_ra_0, w_dec_0, w_z_0 = np.array( cat_0['ra_W15'] ), np.array( cat_0['dec_W15']), np.array( cat_0['z_W15'])

c_coord_s_0 = SkyCoord( ra = sd_ra_0 * U.deg, dec = sd_dec_0 * U.deg )
c_coord_w_0 = SkyCoord( ra = w_ra_0 * U.deg, dec = w_dec_0 * U.deg ) 


cat_1 = pds.read_csv('/home/xkchen/mywork/Sat_SB/data/remap_W15_cat/ZLW_cat_15/redMaPPer_rich-bin_map_clust-cat.csv')
sd_ra_1, sd_dec_1, sd_z_1 = np.array( cat_1['ra'] ), np.array( cat_1['dec'] ), np.array( cat_1['z'] )
w_ra_1, w_dec_1, w_z_1 = np.array( cat_1['ra_W15'] ), np.array( cat_1['dec_W15']), np.array( cat_1['z_W15'])

c_coord_s_1 = SkyCoord( ra = sd_ra_1 * U.deg, dec = sd_dec_1 * U.deg )
c_coord_w_1 = SkyCoord( ra = w_ra_1 * U.deg, dec = w_dec_1 * U.deg )


idx, sep, d3d = c_coord_s_0.match_to_catalog_sky( d_coord_s )
id_lim = sep.value < 2.7e-4

dif_sd_ra_0 = sd_ra_0[ id_lim == False ]
dif_sd_dec_0 = sd_dec_0[ id_lim == False ]
dif_sd_z_0 = sd_z_0[ id_lim == False ]


idx, sep, d3d = c_coord_w_0.match_to_catalog_sky( d_coord_w )
id_lim = sep.value < 2.7e-4

dif_w_ra_0 = w_ra_0[ id_lim == False ]
dif_w_dec_0 = w_dec_0[ id_lim == False ]
dif_w_z_0 = w_z_0[ id_lim == False ]


##.figs
color_arr_0 = np.arange( len( sdss_ra ) ) / len( sdss_ra )
color_arr_1 = np.arange( len( sd_ra_0 ) ) / len( sd_ra_0 )

plt.figure()

# plt.scatter( sdss_ra, sdss_dec, marker = 'o', c = color_arr_0, cmap = 'winter', alpha = 0.75,)
# plt.scatter( W15_ra, W15_dec, marker = '*', c = color_arr_0, cmap = 'winter', alpha = 0.75,)

# plt.scatter( sd_ra_0, sd_dec_0, marker = 's', c = color_arr_1, cmap = 'autumn', alpha = 0.75,)
# plt.scatter( w_ra_0, w_dec_0, marker = '^', c = color_arr_1, cmap = 'autumn', alpha = 0.75,)

plt.plot( W15_ra, W15_dec, 'g*', )
plt.plot( w_ra_0, w_dec_0, 'ro', )

plt.savefig('/home/xkchen/position_compare.png')
plt.show()



### === combine satellite catalog based on cluster catalog



