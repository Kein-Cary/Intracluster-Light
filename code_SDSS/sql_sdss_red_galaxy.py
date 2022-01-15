import matplotlib as mpl
import matplotlib.pyplot as plt

import h5py
import numpy as np
import astropy.io.fits as fits

import mechanize
import pandas as pds
from io import StringIO

import scipy.stats as sts
import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

###################################
url = 'http://skyserver.sdss.org/dr12/en/tools/search/sql.aspx'

## read BCG effective radius
#  which measured by SDSS,
dat = pds.read_csv('/home/xkchen/mywork/ICL/code/SEX/result/test_1000-to-250_cat-match.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
rich, r_mag, g_mag = np.array(dat.rich), np.array(dat.r_Mag), np.array(dat.g_Mag)

load = '/media/xkchen/My Passport/data/SDSS/'
pixel = 0.396

Ns = len(ra)
R_eff = []

for kk in range( Ns ):
    ra_g, dec_g, z_g = ra[kk], dec[kk], z[kk]

    photo_txt = pds.read_csv(load + 'BCG_photometric/BCG_photo_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g), skiprows = 1,)
    r_eff = photo_txt['deVRad_r'][0]
    R_eff.append(r_eff)

R_eff = np.array(R_eff)

## color limitation
del_gr = g_mag - r_mag

D_l = Test_model.luminosity_distance(z).value
r_Mag = r_mag + 5 - 5 * np.log10(10**6 * D_l)
g_Mag = g_mag + 5 - 5 * np.log10(10**6 * D_l)
'''
## LRG catalog (Eisensten et al. 2001)
data_set = """
SELECT ALL
    p.ra, p.dec, p.g, p.r, p.i, p.type,

    p.deVRad_g, p.deVRad_r, p.deVRad_i, 
    p.deVAB_g, p.deVAB_r, p.deVAB_i, 
    p.deVPhi_g, p.deVPhi_r, p.deVPhi_i, 

    p.expRad_g, p.expRad_r, p.expRad_i, 
    p.expAB_g, p.expAB_r, p.expAB_i, 
    p.expPhi_g, p.expPhi_r, p.expPhi_i, 

    p.objID, p.modelMag_r, p.modelMag_g, p.modelMag_i, 
    photoz.absMagR, photoz.absMagG, photoz.absMagI,
    photoz.z, SpecObjAll.z 
FROM PhotoObjAll as p 
    JOIN SpecObjAll ON SpecObjAll.bestObjID = p.objID
    JOIN photoz ON photoz.objID = p.objID
WHERE
    p.type = 3
    AND p.mode = 1
    AND p.clean = 1

    AND ( ( flags & (dbo.fPhotoFlags('BINNED1')
    | dbo.fPhotoFlags('BINNED2') 
    | dbo.fPhotoFlags('BINNED4')) ) > 0 
    AND ( flags & (dbo.fPhotoFlags('BLENDED') 
    | dbo.fPhotoFlags('NODEBLEND') 
    | dbo.fPhotoFlags('CHILD')) ) != dbo.fPhotoFlags('BLENDED') 
    AND ( flags & (dbo.fPhotoFlags('EDGE') 
    | dbo.fPhotoFlags('SATURATED')) ) = 0)
    
    AND p.petroMag_i > 17.5 
    AND (p.petroMag_r > 15.5 OR p.petroR50_r > 2) 
    AND (p.petroMag_r > 0 AND p.g > 0 AND p.r > 0 AND p.i > 0)

    AND ( (p.petroMag_r - p.extinction_r) < 19.2
    AND ( p.psfMag_r - p.modelMag_r > 0.24 )
    AND ( ( (p.dered_r - p.dered_i) - (p.dered_g - p.dered_r)/4 - 0.177) BETWEEN -0.2 AND 0.2 )
    AND ( p.petroMag_r - p.extinction_r < 
        ( 13.116 + ( 0.7*(p.dered_g - p.dered_r) + 1.2 * (p.dered_r - p.dered_i - 0.177) ) / 0.3 ) )
    AND ( p.petroMag_r + 2.5 * LOG10(2 * 3.1415 * p.petroR50_r * p.petroR50_r) <= 24.2) )

    AND ( (photoz.z BETWEEN 0.2 AND 0.3 ) OR (SpecObjAll.z BETWEEN 0.2 AND 0.3) )
    AND ( p.ra BETWEEN 120 AND 270 )
    AND ( p.dec BETWEEN -10 AND 60 )
"""

br = mechanize.Browser()
resp = br.open(url)
resp.info()

br.select_form(name = "sql")
br['cmd'] = data_set
br['format'] = ['csv']
response = br.submit()
s = str(response.get_data(), encoding = 'utf-8')
doc = open('/home/xkchen/mywork/ICL/data/tmp_img/SDSS_red-G_A-250_match.txt', 'w')
print(s, file = doc)
doc.close()
'''

## A250-LRG match
red_cat = pds.read_csv('/home/xkchen/mywork/ICL/data/tmp_img/SDSS_red-G_A-250_match.txt', skiprows = 1)
cp_ra, cp_dec, cp_z = np.array(red_cat.ra), np.array(red_cat.dec), np.array(red_cat.z)
cp_R_eff = np.array(red_cat['deVRad_r'])
cp_rmag, cp_gmag = np.array(red_cat.r), np.array(red_cat.g)

id_zlim = (cp_z >= 0.2) & (cp_z <= 0.3)
id_Rlim = (cp_R_eff >= R_eff.min() ) & (cp_R_eff <= R_eff.max() )
id_lim = id_zlim & id_Rlim

lim_ra, lim_dec, lim_z = cp_ra[id_lim], cp_dec[id_lim], cp_z[id_lim]
lim_Reff = cp_R_eff[id_lim]

## select galaxies more details on the (z and R_eff) range
bins_z =  np.linspace(z.min(), z.max(), 21)
bins_R =  np.linspace(R_eff.min(), R_eff.max(), 21)

galx_cont, edg_R, edg_z = sts.binned_statistic_2d(R_eff, z, R_eff, statistic = 'count', bins = [bins_R, bins_z],)[:3]
galx_cont = galx_cont.astype(int)

targ_z, targ_ra, targ_dec, targ_Reff = np.array([0]), np.array([0]), np.array([0]), np.array([0])
targ_rmag, targ_gmag = np.array([0]), np.array([0])

for kk in range( len(edg_R) - 1 ):
    for ll in range( len(edg_z) - 1 ):
        if galx_cont[kk, ll] == 0:
            continue
        else:
            idy = (cp_R_eff >= edg_R[kk] ) * (cp_R_eff <= edg_R[kk+1] )
            idx = (cp_z >= edg_z[ll]) * (cp_z <= edg_z[ll+1] )
            idv = idx & idy

            sub_z, sub_ra, sub_dec, sub_Reff = cp_z[idv], cp_ra[idv], cp_dec[idv], cp_R_eff[idv]
            sub_rmag, sub_gmag = cp_rmag[idv], cp_gmag[idv]

            if len(sub_z) < galx_cont[kk,ll]:
                tt_order = np.random.choice( len(sub_z), size = len(sub_z), replace = False)
            else:
                tt_order = np.random.choice( len(sub_z), size = galx_cont[kk,ll], replace = False)

            targ_z = np.r_[ targ_z, sub_z[tt_order] ]
            targ_ra = np.r_[ targ_ra, sub_ra[tt_order] ]
            targ_dec = np.r_[ targ_dec, sub_dec[tt_order] ]
            targ_Reff = np.r_[ targ_Reff, sub_Reff[tt_order] ]
            targ_rmag = np.r_[ targ_rmag, sub_rmag[tt_order] ]
            targ_gmag = np.r_[ targ_gmag, sub_gmag[tt_order] ]

targ_z = targ_z[1:]
targ_ra = targ_ra[1:]
targ_dec = targ_dec[1:]
targ_Reff = targ_Reff[1:]
targ_rmag = targ_rmag[1:]
targ_gmag = targ_gmag[1:]

keys = ['ra', 'dec', 'z', 'R_eff', 'r_mag', 'g_mag']
values = [targ_ra, targ_dec, targ_z, targ_Reff, targ_rmag, targ_gmag]
fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
data.to_csv('A250_LRG_match_cat.csv')


plt.figure()
plt.hist(z, bins = 20, density = True, color = 'r', alpha = 0.5, label = 'A-250')
plt.hist(targ_z, bins = 20, density = True, color = 'b', alpha = 0.5, label = 'LRG')
plt.legend(loc = 1)
plt.xlabel('redshift')
plt.ylabel('PDF')
plt.savefig('z_compare.png', dpi = 100)
plt.close()

plt.figure()
plt.hist(R_eff, bins = 20, density = True, color = 'r', alpha = 0.5, label = 'A-250')
plt.axvline(np.mean(R_eff), ls = '--', color = 'r', alpha = 0.5, label = 'Mean')
plt.axvline(np.median(R_eff), ls = ':', color = 'r', alpha = 0.5, label = 'Median')

plt.hist(targ_Reff, bins = 20, density = True, color = 'b', alpha = 0.5, label = 'LRG')
plt.axvline(np.mean(targ_Reff), ls = '--', color = 'b', alpha = 0.5, )
plt.axvline(np.median(targ_Reff), ls = ':', color = 'b', alpha = 0.5, )

plt.legend(loc = 1)
plt.xlabel('effective radius [arcsec]')
plt.ylabel('PDF')
plt.savefig('R_eff_compare.png', dpi = 200)
plt.close()

plt.figure()
plt.hist(r_mag, bins = 20, density = True, color = 'r', alpha = 0.5, label = 'A-250')
plt.hist(targ_rmag, bins = 20, density = True, color = 'b', alpha = 0.5, label = 'LRG')
plt.legend(loc = 1)
plt.xlabel('r band mag')
plt.ylabel('PDF')
plt.savefig('r_mag_compare.png', dpi = 100)
plt.close()

