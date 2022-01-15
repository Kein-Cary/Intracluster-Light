import numpy as np
from astropy.io import fits
from astropy import coordinates as coord
from astropy import units as u

# load data
data_path = '/home/lizz/data/'
c = fits.getdata(data_path + '/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
w = fits.getdata(data_path + '/spectro/wisconsin_pca_m11-DR12-all.fits')

# CMASS mask
# import pymangle
# m = pymangle.Mangle(data_path + 'geometry/mask_DR12v5_CMASS_North.ply')
# ix_CMASS_North_mask = m.weight(c.RA, c.DEC) > 0

# use primary spec only
ix_prim = w.SPECPRIMARY > 0
w = w[ix_prim]

# matching by coordinates
c_coord = coord.SkyCoord(ra=c['ra'] * u.deg, dec=c['dec'] * u.deg)
w_coord = coord.SkyCoord(ra=w['ra'] * u.deg, dec=w['dec'] * u.deg)

ix1, ix2, sep = w_coord.search_around_sky(c_coord, 1 * u.arcsec)[:3]
w_ix = -np.ones(len(c), dtype=int)
w_ix[ix1] = ix2
ix_spec = w_ix != -1
ix_miss = w_ix == -1

# get quantities
lgMstar = w.MSTELLAR_MEDIAN[w_ix]
lgMstar[ix_miss] = -1

z_spec = w.Z[w_ix]
z_spec[ix_miss] = -1

z_spec_err = w.Z_ERR[w_ix]
z_spec_err[ix_miss] = -1

z_best = z_spec.copy()
z_best[ix_miss] = c.Z_LAMBDA[ix_miss]


RA, DEC = c.RA, c.DEC
P_CEN = c.P_CEN.T[0]
LAMBDA = c.LAMBDA
LAMBDA_ERR = c.LAMBDA_ERR
