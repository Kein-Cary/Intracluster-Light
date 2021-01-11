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

def sdss_img_load_func(cat_file, R_query, band, out_file, err_log,):
    """
    cat_file : catalog which will use to query imgs (must include ra, dec, z)
    R_query : query region size / radius, in physical unit, Mpc
    band : band of imgs, list type. (eg. ['r', 'g', 'u'])
    out_file : output file information, including save path. 
                (eg. /XXX/XXX_band_ra_dec_redshift_XXX.fits)
    err_log : *.txt file, use to record fails imgs information, save in the path
            where the query function will be run. 
    """
    dat = pds.read_csv(cat_file)
    ra, dec, z = np.array(dat['ra']), np.array(dat['dec']), np.array(dat['z'])
    Ns = len(ra)

    # calculate the angular size 
    size_cluster = 2. # assumptiom: cluster size is 2.Mpc/h
    from angular_diameter_reshift import mark_by_self
    from angular_diameter_reshift import mark_by_plank
    A_size, A_d= mark_by_self(z, size_cluster)
    view_d = A_size * U.rad

    from astroquery.sdss import SDSS
    from astropy import coordinates as coords
    from astropy.table import Table
    R_A = 0.5 * view_d.to(U.arcsec) # angular radius in unit of arcsec

    doc = open(err_log, 'w')

    for k in range( Ns ):

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
            for q in range( len(band) ):
                url_road = 'http://data.sdss.org/sas/dr12/boss/photoObj/frames/%.0f/%.0f/%.0f/frame-%s-%s-%.0f-%s.fits.bz2'%\
                (xid[pl][4], xid[pl][3], xid[pl][5], band[q], s_bgn, xid[pl][5], s_end)

                out_lis = out_file % (band[q], ra[k], dec[k], z[k])

                wt.download(url_road, out_lis)

            print('**********-----')
            print('finish--', k/len(z))
        except KeyError:
            # save the "exception" case
            s = '\n k = %d, ra%.3f, dec%.3f, z%.3f \n' % (k, ra[k], dec[k], z[k])
            print(s, file = doc)

            continue

    doc.close()


