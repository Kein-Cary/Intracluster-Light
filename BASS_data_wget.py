import numpy as np
import pandas as pds
import astropy.io.fits as fits
import wget as wt

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

def data_load(set_ra, set_dec, set_z, band_id):
    zn = len(ra)
    for kk in range(zn):
        ra_g = set_ra[kk]
        dec_g = set_dec[kk]
        z_g = set_z[kk]

        url_load = 'http://legacysurvey.org/viewer/fits-cutout?ra=%f&dec=%f&layer=dr8&pixscale=0.45&bands=%s&size=3000' % (
            ra_g, dec_g, band[band_id])
        out_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/BASS/DES_img/des_img_ra%.3f_dec%.3f_z%.3f_%s_band.fits' % (
            ra_g, dec_g, z_g, band[band_id])
        wt.download(url_load, out_file)

    return

##### read catalog
dat = pds.read_csv('/mnt/ddnfs/data_users/cxkttwl/ICL/BASS/catalog/sdss_z-lambda_cat.csv')
ra = np.array(dat.ra)
dec = np.array(dat.dec)
z = np.array(dat.z)
angle_radius = np.array(dat.angle_size)
band = ['g', 'r', 'z']

for kk in range(len(band)):

    m, n = divmod(len(z), cpus)
    N_sub0, N_sub1 = m * rank, (rank + 1) * m
    if rank == cpus - 1:
        N_sub1 += n
    data_load(ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1], z[N_sub0 :N_sub1], kk)

print('Done!')
