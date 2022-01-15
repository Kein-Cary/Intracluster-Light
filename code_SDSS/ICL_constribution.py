import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as C
import astropy.units as U
import astroquery.sdss as asds
import astropy.io.fits as aft
import scipy.stats as sts
import find
import h5py 
# setlect model
import handy.scatter as hsc
from astropy.wcs import *
import astropy.wcs as awc
from ICL_angular_diameter_reshift import mark_by_self
from ICL_angular_diameter_reshift import mark_by_plank
#### generally data get
goal_data = aft.getdata(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
sub_data = aft.getdata(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')
# find the member of each BGC -cluster, by find the repeat ID
repeat = sts.find_repeats(sub_data.ID)
rept_ID = np.int0(repeat)
ID_array = np.int0(sub_data.ID)
sub_redshift = np.array(sub_data.Z_SPEC)

RA = np.array(goal_data.RA)
DEC = np.array(goal_data.DEC)
redshift = np.array(goal_data.Z_SPEC)
# except the part with no spectra redshift
z_eff = redshift[redshift != -1]
ra_eff = RA[redshift != -1]
dec_eff = DEC[redshift != -1]
# select the nearly universe
z = z_eff[z_eff <= 0.3]
Ra = ra_eff[z_eff <= 0.3]
Dec = dec_eff[z_eff <= 0.3]
#### all of those z<=0.3
with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/cluster_record.h5') as f:
    tot_sub = np.array(f['a'][0])
    inr_sub = np.array(f['a'][1])
    sub_ratio = np.array(f['a'][2])
    area_ratio = np.array(f['a'][3])
    reference_ratio = np.array(f['a'][4])
    rich = np.array(f['a'][5])
plt.figure()
plt.hist(area_ratio,color = 'b',bins = 50, alpha = 0.5)
plt.xlabel(r'$S/S0$')
plt.ylabel(r'$Number$')
plt.title(r'$S/S0$')
plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/area_fraction_distribution.png',dpi = 600)
plt.close()

plt.figure()
plt.hist(tot_sub,color = 'green',bins = 50,alpha = 0.5,label = 'total_satellite')
plt.hist(inr_sub,color = 'red',bins = 50,alpha = 0.5,label = 'satellite_in_1Mpc/h')
plt.legend(loc = 1)
plt.xlabel(r'$richness$')
plt.ylabel(r'$Number$')
plt.title(r'$N_{satellite}$')
plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/satellite_distribution.png',dpi = 600)
plt.close()
#### sample: 0.2 <= z <= 0.3
sampl_tot = tot_sub[(z<=0.3) & (z>=0.2)]
sampl_inr = inr_sub[(z<=0.3) & (z>=0.2)]
sampl_S_ratio = area_ratio[(z<=0.3) & (z>=0.2)]
plt.figure()
plt.hist(sampl_S_ratio,color = 'b',bins = 50, alpha = 0.5)
plt.xlabel(r'$S/S0$')
plt.ylabel(r'$Number$')
plt.title(r'$[S/S0]_{0.2<=z<=0.3}$')
plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/Inr_area_fraction_distribution.png',dpi = 600)
plt.close()

plt.figure()
plt.hist(sampl_tot,color = 'green',bins = 50,alpha = 0.5,label = 'total_satellite')
plt.hist(sampl_inr,color = 'red',bins = 50,alpha = 0.5,label = 'satellite_in_1Mpc/h')
plt.legend(loc = 1)
plt.xlabel(r'$richness$')
plt.ylabel(r'$Number$')
plt.title(r'$N_{satellite}$')
plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/Inr_satellite_distribution.png',dpi = 600)
plt.close()