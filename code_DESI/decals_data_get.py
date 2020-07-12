import numpy as np
import pandas as pds
import astropy.io.fits as fits
import threading
import wget
import subprocess as subpro

band = ['g', 'r', 'z']

##### read catalog
dat = pds.read_csv('/mnt/ddnfs/data_users/cxkttwl/ICL/BASS/catalog/sdss_z-lambda_cat.csv')
ra = np.array(dat.ra)
dec = np.array(dat.dec)
z = np.array(dat.z)
"""
doc = open('/mnt/ddnfs/data_users/cxkttwl/ICL/err_stop_pix05.txt', 'w')
for kk in range(len(z)):
    ra_g = ra[kk]
    dec_g = dec[kk]
    z_g = z[kk]
    try:
        dat = fits.open('/mnt/ddnfs/data_users/cxkttwl/ICL/BASS/desi_05/desi_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g))
    except FileNotFoundError:
        s = '%d, %.3f, %.3f, %.3f' % (kk, ra_g, dec_g, z_g)
        print(s, file = doc, )

doc.close()
"""

def fits_url2load(ra, dec, z):

    url_name = []
    out_name = []
 
    for ll in range(len(z)):
        ra_g = ra[ll]
        dec_g = dec[ll]
        z_g = z[ll]

        http = 'http://legacysurvey.org/viewer/fits-cutout?ra=%f&dec=%f&layer=dr8&pixscale=0.5&bands=grz&size=3000' % (ra_g, dec_g)
        out = '/mnt/ddnfs/data_users/cxkttwl/ICL/BASS/desi_05/desi_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g)

        url_name.append(http)
        out_name.append(out)

    return url_name, out_name

downs = fits_url2load(ra, dec, z)
url_lis, out_lis = downs[0], downs[1]


doc = open('/mnt/ddnfs/data_users/cxkttwl/ICL/BASS/err_stop_pix05.txt', 'w')
for kk in range(len(z)):
    ra_g = ra[kk]
    dec_g = dec[kk]
    z_g = z[kk]
    try:
        wget.download(url_lis[kk], out_lis[kk])
        print('\n finish--', kk / len(z))
    except:
        s = '%d, %.3f, %.3f, %.3f' % (kk, ra_g, dec_g, z_g)
        print(s, file = doc, )

doc.close()
