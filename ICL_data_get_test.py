#this file use to cite data from SDSS,and save the image data
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as C
import astropy.units as U
import astroquery.sdss as asds
import astropy.io.fits as aft
from astropy.utils import data  
### 
##### section 1: read the redmapper data
goal_data = aft.getdata(
        '/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
sub_data = aft.getdata(
        '/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')
'''
goal_data = aft.getdata(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
sub_data = aft.getdata(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')
'''
RA = np.array(goal_data.RA)
DEC = np.array(goal_data.DEC)
redshift = np.array(goal_data.Z_SPEC)
# except the part with no spectra redshift
z_eff = redshift[redshift != -1]
ra_eff = RA[redshift != -1]
dec_eff = DEC[redshift != -1]
# select the nearly universe
z = z_eff[z_eff <= 0.3]
ra = ra_eff[z_eff <= 0.3]
dec = dec_eff[z_eff <= 0.3]
# calculate the angular size 
size_cluster = 2. # assumptiom: cluster size is 2.Mpc/h
from ICL_angular_diameter_reshift import mark_by_self
from ICL_angular_diameter_reshift import mark_by_plank
A_size, A_d= mark_by_self(z,size_cluster)
view_d = A_size*U.rad
#### section 2: cite the data and save fits figure
from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astroML.plotting import setup_text_plots
from astropy.table import Table
R_A = 0.5*view_d.to(U.arcsec) # angular radius in angular second unit
band = ['u','g','r','i','z']
for k in range(len(z)):
    pos = coords.SkyCoord('%fd %fd'%(ra[k],dec[k]), frame='icrs')
    try:
        xid = SDSS.query_region(pos, spectro = False, radius = R_A[k]/2, timeout = None)
        # for galaxy, don't take spectra into account
        name = xid.colnames
        aa = np.array([xid['ra'],xid['dec']])
        da = np.sqrt((aa[0,:]-ra[k])**2+(aa[1,:]-dec[k])**2)
        # select the goal region
        dl = da.tolist()
        pl = dl.index(np.min(da))
        # change the list as the astropy table type
        fd = Table(xid[pl])
        for p in range(len(band)):
            imdata = SDSS.get_images(matches = fd, band = band[p])
            hdu_tot = aft.HDUList(imdata[0])
            hdu_tot.writeto(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/data/img_ra%.3f_dec%.3f_tot_z%.3f_bnd%s.fits'
            %(ra[k],dec[k],z[k],band[p]),overwrite = True)
            # save the total information of the data
        data.clear_download_cache()
        # after saving, clear the cache
    except KeyError:
        continue
