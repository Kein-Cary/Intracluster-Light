import h5py
import numpy as np
import astropy.io.fits as fits

import mechanize
import pandas as pd 
from io import StringIO

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

#url = 'http://skyserver.sdss.org/dr12/en/tools/search/sql.aspx'
url = 'http://cas.sdss.org/dr7/en/tools/search/sql.asp'

with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sample_catalog.h5', 'r') as f:
    catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]

r_select = 0.16676 # centered at BCG, radius = 10 arcmin (1515.15 pixel)
N_tot = len(z)
sub_N = N_tot * 1

no_match = []
for kk in range(len(z)):
    ra_g = ra[kk]
    dec_g = dec[kk]
    z_g = z[kk]

    c_ra0 = str(ra_g - r_select)
    c_dec0 = str(dec_g - r_select)
    c_ra1 = str(ra_g + r_select)
    c_dec1 = str(dec_g + r_select)

    # query stars and saturated sources (may not be stars)
    data_set = """
    SELECT ALL
        p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.type,
        p.isoA_u, p.isoA_g, p.isoA_r, p.isoA_i, p.isoA_z,
        p.isoB_u, p.isoB_g, p.isoB_r, p.isoB_i, p.isoB_z,
        p.isoPhi_u, p.isoPhi_g, p.isoPhi_r, p.isoPhi_i, p.isoPhi_z,
        p.flags, dbo.fPhotoFlagsN(p.flags)
    FROM PhotoObj AS p
    WHERE
        p.ra BETWEEN %s AND %s AND p.dec BETWEEN %s AND %s
        AND (p.type = 6 OR (p.flags & dbo.fPhotoFlags('SATURATED')) > 0)
    ORDER by p.r
    """ % (c_ra0, c_ra1, c_dec0, c_dec1)

    print(data_set)
    br = mechanize.Browser()
    resp = br.open(url)
    resp.info()

    br.select_form(name = "sql")
    br['cmd'] = data_set
    br['format'] = ['csv']
    response = br.submit()
    s = str(response.get_data(), encoding = 'utf-8')

    doc = open(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/data/bright_star_dr7/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g), 'w')
    print(s, file = doc)
    doc.close()
    try:
        cat = pd.read_csv(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/data/bright_star_dr7/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g), skiprows = 1)
    except pd.errors.EmptyDataError:
        #doc = open('/mnt/ddnfs/data_users/cxkttwl/ICL/data/No_source_match_sample.txt', 'w')
        #print('%.3f,%.3f,%.3f' % (ra_g, dec_g, z_g), file = doc)
        #doc.close()
        no_match.append('%.3f,%.3f,%.3f' % (ra_g, dec_g, z_g) )
        sub_N -= 1
        print(sub_N)

doc = open('/mnt/ddnfs/data_users/cxkttwl/ICL/data/No_source_match_sample.txt', 'w')
for ll in range(len(no_match)):
    subx = no_match[ll].replace(',', ' ')
    print(subx, file = doc)
doc.close()
print(sub_N)
