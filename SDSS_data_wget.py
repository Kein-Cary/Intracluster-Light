import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as C
import astropy.units as U
import astroquery.sdss as asds
import astropy.io.fits as aft
import changds
import wget as wt
##### section 1: read the redmapper data

goal_data = aft.getdata(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
sub_data = aft.getdata(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')

RA = np.array(goal_data.RA)
DEC = np.array(goal_data.DEC)
#redshift = np.array(goal_data.Z_SPEC)
redshift = np.array(goal_data.Z_LAMBDA)

idx = DEC >= 32.0
idy = (redshift <= 0.33) & (redshift >= 0.3)

# select the nearly universe
z = redshift[idx & idy]
ra = RA[idx & idy]
dec = DEC[idx & idy]

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

# set for solve the ReadtimeOut error
doc = open('/mnt/ddnfs/data_users/cxkttwl/ICL/err_stop_record.txt', 'w')

for k in range(len(z)):
    pos = coords.SkyCoord('%fd %fd'%(ra[k],dec[k]), frame='icrs')
    try:
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
            ## spec_z image sample
            out_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            (band[q],ra[k],dec[k],z[k])
            '''
            ## photo_z image sample
            out_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            (band[q],ra[k],dec[k],z[k])

            wt.download(url_road, out_file)
        print('**********-----')
        print('finish--',k/len(z))
    except:
        s = '%d, %.3f, %.3f, %.3f' % (k, ra_g, dec_g, z_g)
        print(s, file = doc, )
        continue

doc.close()
