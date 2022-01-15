# this file use to ouput the data files
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as C
import astropy.units as U
import astroquery.sdss as asds
import astropy.io.fits as aft
import scipy.stats as sts
import h5py
import handy.scatter as hsc
import handy
##### section 1: read the redmapper data and the member data
goal_data = aft.getdata(
        '/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')

sub_data = aft.getdata(
        '/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')
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
with h5py.File('cluster_record.h5') as f:
    tot_sub = np.array(f['a'][0])
    inr_sub = np.array(f['a'][1])
    sub_ratio = np.array(f['a'][2])
    area_ratio = np.array(f['a'][3])
    reference_ratio = np.array(f['a'][4])
    rich = np.array(f['a'][5])
'''
sampl_tot = tot_sub[(z<=0.3) & (z>=0.2)]
sampl_sub = inr_sub[(z<=0.3) & (z>=0.2)]  # the two number just the galaxies number, the later is those in the 1Mpc/h circle
'''
sampl_S_ratio = area_ratio[(z<=0.3) & (z>=0.2)]
sampl_refer = reference_ratio[(z<=0.3) & (z>=0.2)]
sampl_z = z[(z<=0.3) & (z>=0.2)]
sampl_rich = rich[(z<=0.3) & (z>=0.2)]
###
plt.hist(rich,bins=50)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\sharp \ of \ cluster$')
plt.title(r'$distribution \ of \ \lambda$')
plt.yscale('log')
#plt.xscale('log')
plt.hist(sampl_rich,bins=50)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\sharp \ of \ cluster$')
plt.title(r'$distribution \ of \ \lambda$')
plt.yscale('log')
## devide the sampl_z into 5 bins: 0.20~0.22~0.24~0.26~0.28~0.30
plt.scatter(z,rich,s=10,alpha=0.25)
plt.ylabel(r'$\lambda$')
plt.xlabel(r'$z$')
plt.title(r'$z \ \lambda$')
## 0.2~z~0.3
plt.scatter(sampl_z,sampl_rich,s=10,alpha=0.25)
#handy.compare(sampl_z,sampl_rich)
plt.ylabel(r'$\lambda$')
plt.xlabel(r'$z$')
plt.title(r'$z \ \lambda_{0.2 \sim z \sim 0.3}$')
## z0.20~0.22
ta1 = sampl_S_ratio[(sampl_z>=0.20)&(sampl_z<0.22)]
tb1 = sampl_rich[(sampl_z>=0.20)&(sampl_z<0.22)]
plt.scatter(tb1,ta1,s=10,alpha=0.25)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$S/S0$')
plt.title(r'$[S/S0 \ \lambda]_{0.20\sim0.22}$')
## z0.22~0.24
ta2 = sampl_S_ratio[(sampl_z>=0.22)&(sampl_z<0.24)]
tb2 = sampl_rich[(sampl_z>=0.22)&(sampl_z<0.24)]
plt.scatter(tb2,ta2,s=10,alpha=0.25)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$S/S0$')
plt.title(r'$[S/S0 \ \lambda]_{0.22\sim0.24}$')
## z0.24~0.26
ta3 = sampl_S_ratio[(sampl_z>=0.24)&(sampl_z<0.26)]
tb3 = sampl_rich[(sampl_z>=0.24)&(sampl_z<0.26)]
plt.scatter(tb3,ta3,s=10,alpha=0.25)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$S/S0$')
plt.title(r'$[S/S0 \ \lambda]_{0.24\sim0.26}$')
## z0.26~0.28
ta4 = sampl_S_ratio[(sampl_z>=0.26)&(sampl_z<0.28)]
tb4 = sampl_rich[(sampl_z>=0.26)&(sampl_z<0.28)]
plt.scatter(tb4,ta4,s=10,alpha=0.25)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$S/S0$')
plt.title(r'$[S/S0 \ \lambda]_{0.26\sim0.28}$')
## z0.28~0.30
ta5 = sampl_S_ratio[(sampl_z>=0.28)&(sampl_z<=0.30)]
tb5 = sampl_rich[(sampl_z>=0.28)&(sampl_z<=30)]
plt.scatter(tb5,ta5,s=10,alpha=0.25)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$S/S0$')
plt.title(r'$[S/S0 \ \lambda]_{0.28\sim0.30}$')