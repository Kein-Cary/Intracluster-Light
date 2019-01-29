# this file devide the data based on the S/S0
#combine with sextractor
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as C
import astropy.units as U
import astropy.io.fits as aft
import scipy.stats as sts
import h5py 
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
vz = z_eff[z_eff <= 0.3]
z = z_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]
Ra = ra_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]
Dec = dec_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]
Rich = rich_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]
cg_ID = ID_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]
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
## divide bins
bins = 5
a0 = np.min(sampl_S_ratio)
b0 = np.max(sampl_S_ratio)
bins_S = np.linspace(a0,b0,bins+1)
# select clusters and figure
for k in range(bins):
    eta0 = bins_S[k]
    eta1 = bins_S[k+1]
    lamda_sub = sampl_rich[(sampl_S_ratio>=eta0) & (sampl_S_ratio<=eta1)]
    id_sub = sampl_ID[(sampl_S_ratio>=eta0) & (sampl_S_ratio<=eta1)]
    z_sub = sampl_z[(sampl_S_ratio>=eta0) & (sampl_S_ratio<=eta1)] 
    pos_ra = Ra[(sampl_S_ratio>=eta0) & (sampl_S_ratio<=eta1)]
    pos_dec = Dec[(sampl_S_ratio>=eta0) & (sampl_S_ratio<=eta1)]
    for p in range(len(z_sub)):
        clust10 = aft.getdata(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
        ('r',pos_ra[p],pos_dec[p],z_sub[p]),header = True)
        hdu10 = aft.PrimaryHDU()
        hdu10.data = clust10[0]
        hdu10.header = clust10[1]
        hdu10.writeto(
                '/mnt/ddnfs/data_users/cxkttwl/ICL/data/S_S0_%.3f_%.3f/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
        (eta0,eta1,'r',pos_ra[p],pos_dec[p],z_sub[p]),overwrite = True)
        
        clust11 = aft.getdata(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
        ('u',pos_ra[p],pos_dec[p],z_sub[p]),header = True)
        hdu11 = aft.PrimaryHDU()
        hdu11.data = clust11[0]
        hdu11.header = clust11[1]
        hdu11.writeto(
                '/mnt/ddnfs/data_users/cxkttwl/ICL/data/S_S0_%.3f_%.3f/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
        (eta0,eta1,'u',pos_ra[p],pos_dec[p],z_sub[p]),overwrite = True) 
        
        clust12 = aft.getdata(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
        ('g',pos_ra[p],pos_dec[p],z_sub[p]),header = True)
        hdu12 = aft.PrimaryHDU()
        hdu12.data = clust12[0]
        hdu12.header = clust12[1]
        hdu12.writeto(
                '/mnt/ddnfs/data_users/cxkttwl/ICL/data/S_S0_%.3f_%.3f/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
        (eta0,eta1,'g',pos_ra[p],pos_dec[p],z_sub[p]),overwrite = True) 
        
        clust13 = aft.getdata(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
        ('i',pos_ra[p],pos_dec[p],z_sub[p]),header = True)
        hdu13 = aft.PrimaryHDU()
        hdu13.data = clust13[0]
        hdu13.header = clust13[1]
        hdu13.writeto(
                '/mnt/ddnfs/data_users/cxkttwl/ICL/data/S_S0_%.3f_%.3f/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
        (eta0,eta1,'i',pos_ra[p],pos_dec[p],z_sub[p]),overwrite = True) 
        
        clust14 = aft.getdata(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
        ('z',pos_ra[p],pos_dec[p],z_sub[p]),header = True)
        hdu14 = aft.PrimaryHDU()
        hdu14.data = clust14[0]
        hdu14.header = clust14[1]
        hdu14.writeto(
                '/mnt/ddnfs/data_users/cxkttwl/ICL/data/S_S0_%.3f_%.3f/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
        (eta0,eta1,'z',pos_ra[p],pos_dec[p],z_sub[p]),overwrite = True) 
