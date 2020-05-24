import numpy as np
import pandas as pds
import astropy.io.fits as fits
import wget as wt

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

##### read catalog
dat = pds.read_csv('/mnt/ddnfs/data_users/cxkttwl/ICL/BASS/catalog/sdss_z-lambda_cat.csv')
ra = np.array(dat.ra)
dec = np.array(dat.dec)
z = np.array(dat.z)
angle_radius = np.array(dat.angle_size)
band = ['g', 'r', 'z']

doc = open('/mnt/ddnfs/data_users/cxkttwl/ICL/BASS/err_stop_mzls.txt', 'w')

for kk in range(3):
    if kk == 0:
        lo = 180
    else:
        lo = 0
    for jj in range(lo, len(z)):
        ra_g = ra[jj]
        dec_g = dec[jj]
        z_g = z[jj]
        try:
            url_load = 'http://legacysurvey.org/viewer/fits-cutout?ra=%f&dec=%f&layer=dr8&pixscale=0.27&bands=%s&size=3000' % (ra_g, dec_g, band[kk] )
            out_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/BASS/mzls_img/mzls_img_ra%.3f_dec%.3f_z%.3f_%s_band.fits' % (ra_g, dec_g, z_g, band[kk] )
            wt.download(url_load, out_file)

            print('**********-----')
            print('finish--', jj / len(z))
        except:
            s = '%s, %d, %.3f, %.3f, %.3f' % (band[kk], jj, ra_g, dec_g, z_g)
            print(s, file = doc, )

doc.close()

print('Done!')
