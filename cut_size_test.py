# data count for cut region test
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as C
import astropy.units as U
import astropy.io.fits as aft
import scipy.stats as sts
# setlect model
import astropy.wcs as awc
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

RA = np.array(goal_data.RA)
DEC = np.array(goal_data.DEC)
redshift = np.array(goal_data.Z_SPEC)
richness = np.array(goal_data.LAMBDA)
# except the part with no spectra redshift
z_eff = redshift[redshift != -1]
ra_eff = RA[redshift != -1]
dec_eff = DEC[redshift != -1]
rich_eff = richness[redshift != -1]
# select the nearly universe
z = z_eff[(z_eff >= 0.2)&(z_eff <= 0.3)]
Ra = ra_eff[(z_eff >= 0.2)&(z_eff <= 0.3)]
Dec = dec_eff[(z_eff >= 0.2)&(z_eff <= 0.3)]
rich = rich_eff[(z_eff >= 0.2)&(z_eff <= 0.3)]
Ntotl = len(z)
# select region 
cut = np.array([250,300,350,400,450,500,550,600,650,700,740])
ratio = np.zeros(len(cut),dtype = np.float)
Num = np.zeros(len(cut), dtype = np.float)
for k in range(len(cut)):
    n = 0
    for q in range(len(z)):
        cir_data = aft.getdata(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            ('r',Ra[q],Dec[q],z[q]),header = True)
        wcs = awc.WCS(cir_data[1])
        cx,cy = wcs.all_world2pix(Ra[q]*U.deg,Dec[q]*U.deg,1)
        a0 = cx-cut[k]
        a1 = cx+cut[k]
        b0 = cy-cut[k]
        b1 = cy+cut[k]
        ref1 = cir_data[0].shape[1]-1
        ref2 = cir_data[0].shape[0]-1
        if ((a0 >= 0)&(a1 <= ref1))&((b0 >= 0)&(b1 <= ref2)):
            n = n+1
        else:
            n = n
    Num[k] = n
    ratio[k] = n/len(z)
plt.figure()
plt.plot(cut,Num)
plt.xlabel('# pixel')
plt.ylabel('# cluster')
plt.savefig('cut_region_test.png',dpi=600)
plt.close()
plt.figure()
plt.plot(cut,ratio)
plt.xlabel('# pixel')
plt.ylabel('$\eta$')
plt.savefig('cut_region_ratio.png',dpi=600)
plt.close()