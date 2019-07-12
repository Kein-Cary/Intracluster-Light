# this file devide the data based on the S/S0
#combine with sextractor
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import h5py
import numpy as np
import scipy.stats as sts

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as aft

import astropy.wcs as awc
import astropy.io.ascii as asc
from dustmaps.sfd import SFDQuery
from extinction_redden import A_wave
from light_measure import flux_recal
from astropy.coordinates import SkyCoord

# extinction params
Rv = 3.1
sfd = SFDQuery()
band = ['u', 'g', 'r','i', 'z']
A_lambd = np.array([5.155, 3.793, 2.751, 2.086, 1.479])
l_wave = np.array([3551, 4686, 6166, 7480, 8932])

goal_data = aft.getdata(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
sub_data = aft.getdata(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')
# find the member of each BGC -cluster, by find the repeat ID
repeat = sts.find_repeats(sub_data.ID)
rept_ID = np.int0(repeat)
ID_array = np.int0(sub_data.ID)
sub_redshift = np.array(sub_data.Z_SPEC) #use to figure out how big the satellite
center_distance = sub_data.R # select the distance of satellite galaxies
member_pos = np.array([sub_data.RA,sub_data.DEC]) # record the position of satellite
# read the center galaxy position
RA = np.array(goal_data.RA)
DEC = np.array(goal_data.DEC)
redshift = np.array(goal_data.Z_SPEC)
richness = np.array(goal_data.LAMBDA)
host_ID = np.array(goal_data.ID)
# except the part with no spectra redshift
z_eff = redshift[redshift != -1]
ra_eff = RA[redshift != -1]
dec_eff = DEC[redshift != -1]
rich_eff = richness[redshift != -1]
ID_eff = host_ID[redshift != -1]
# select the nearly universe
z = z_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]
Ra = ra_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]
Dec = dec_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]
Rich = rich_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]
cg_ID = ID_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]

'''
vz = z_eff[z_eff <= 0.3]
####### read the S/S0, richness
with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/cluster_record.h5') as f:
    tot_sub = np.array(f['a'][0])
    inr_sub = np.array(f['a'][1])
    sub_ratio = np.array(f['a'][2])
    area_ratio = np.array(f['a'][3])
    reference_ratio = np.array(f['a'][4])
    rich = np.array(f['a'][5])
sampl_S_ratio = area_ratio[(vz<=0.3) & (vz>=0.2)]
sampl_refer = reference_ratio[(vz<=0.3) & (vz>=0.2)]
sampl_rich = rich[(vz<=0.3) & (vz>=0.2)]
sampl_ID = cg_ID
sampl_z = z
'''
## divide bins with richness
bins = 5
a0 = np.min(Rich)
b0 = np.max(Rich)
bins_S = np.linspace(a0, b0, bins+1)

x0 = np.linspace(0, 2047, 2048)
y0 = np.linspace(0, 1488, 1489)
img_grid = np.array(np.meshgrid(x0, y0))

def bins_divid():
    for k in range(bins):
        eta0 = bins_S[k]
        eta1 = bins_S[k+1]

        z_sub = z[(Rich >= eta0) & (Rich <= eta1)]
        pos_ra = Ra[(Rich >= eta0) & (Rich <= eta1)]
        pos_dec = Dec[(Rich >= eta0) & (Rich <= eta1)]

        for p in range(len(z_sub)):
            z_g = z_sub[p]
            ra_g = pos_ra[p]
            dec_g = pos_dec[p]
            #### r
            clust10 = aft.getdata(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            ('u',pos_ra[p],pos_dec[p],z_sub[p]),header = True)

            img0 = clust10[0]
            hwcs0 = awc.WCS(clust10[1])
            ra_img, dec_img = hwcs0.all_pix2world(img_grid[0,:], img_grid[1,:], 1)
            pos_g = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
            BEV = sfd(pos_g)
            Av = Rv * BEV * 0.86
            Al = A_wave(l_wave[0], Rv) * Av
            img0 = img0*10**(Al / 2.5)

            hdu10 = aft.PrimaryHDU()
            hdu10.data = img0
            hdu10.header = clust10[1]
            hdu10.writeto(
                    '/mnt/ddnfs/data_users/cxkttwl/ICL/data/richness/lvl_%d/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits'%\
            (np.int(k+1),'u',pos_ra[p],pos_dec[p],z_sub[p]),overwrite = True)

            #### u
            clust11 = aft.getdata(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            ('g',pos_ra[p],pos_dec[p],z_sub[p]),header = True)

            img1 = clust11[0]
            hwcs1 = awc.WCS(clust11[1])
            ra_img, dec_img = hwcs1.all_pix2world(img_grid[0,:], img_grid[1,:], 1)
            pos_g = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
            BEV = sfd(pos_g)
            Av = Rv * BEV * 0.86
            Al = A_wave(l_wave[1], Rv) * Av
            img1 = img1*10**(Al / 2.5)

            hdu11 = aft.PrimaryHDU()
            hdu11.data = img1
            hdu11.header = clust11[1]
            hdu11.writeto(
                    '/mnt/ddnfs/data_users/cxkttwl/ICL/data/richness/lvl_%d/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits'%\
            (np.int(k+1),'g',pos_ra[p],pos_dec[p],z_sub[p]),overwrite = True)

            #### g
            clust12 = aft.getdata(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            ('r',pos_ra[p],pos_dec[p],z_sub[p]),header = True)

            img2 = clust12[0]
            hwcs2 = awc.WCS(clust12[1])
            ra_img, dec_img = hwcs2.all_pix2world(img_grid[0,:], img_grid[1,:], 1)
            pos_g = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
            BEV = sfd(pos_g)
            Av = Rv * BEV * 0.86
            Al = A_wave(l_wave[2], Rv) * Av
            img2 = img2*10**(Al / 2.5)

            hdu12 = aft.PrimaryHDU()
            hdu12.data = img2
            hdu12.header = clust12[1]
            hdu12.writeto(
                    '/mnt/ddnfs/data_users/cxkttwl/ICL/data/richness/lvl_%d/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits'%\
            (np.int(k+1),'r',pos_ra[p],pos_dec[p],z_sub[p]),overwrite = True)

            #### i
            clust13 = aft.getdata(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            ('i',pos_ra[p],pos_dec[p],z_sub[p]),header = True)

            img3 = clust13[0]
            hwcs3 = awc.WCS(clust13[1])
            ra_img, dec_img = hwcs3.all_pix2world(img_grid[0,:], img_grid[1,:], 1)
            pos_g = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
            BEV = sfd(pos_g)
            Av = Rv * BEV * 0.86
            Al = A_wave(l_wave[3], Rv) * Av
            img3 = img3*10**(Al / 2.5)

            hdu13 = aft.PrimaryHDU()
            hdu13.data = img3
            hdu13.header = clust13[1]
            hdu13.writeto(
                    '/mnt/ddnfs/data_users/cxkttwl/ICL/data/richness/lvl_%d/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits'%\
            (np.int(k+1),'i',pos_ra[p],pos_dec[p],z_sub[p]),overwrite = True)

            #### z
            clust14 = aft.getdata(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            ('z',pos_ra[p],pos_dec[p],z_sub[p]),header = True)

            img4 = clust14[0]
            hwcs4 = awc.WCS(clust14[1])
            ra_img, dec_img = hwcs4.all_pix2world(img_grid[0,:], img_grid[1,:], 1)
            pos_g = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
            BEV = sfd(pos_g)
            Av = Rv * BEV * 0.86
            Al = A_wave(l_wave[4], Rv) * Av
            img4 = img4*10**(Al / 2.5)

            hdu14 = aft.PrimaryHDU()
            hdu14.data = img4
            hdu14.header = clust14[1]
            hdu14.writeto(
                    '/mnt/ddnfs/data_users/cxkttwl/ICL/data/richness/lvl_%d/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits'%\
            (np.int(k+1),'z',pos_ra[p],pos_dec[p],z_sub[p]),overwrite = True)
        print(k)
    return

def main():
    bins_divid()

if __name__ == "__main__":
    main()
