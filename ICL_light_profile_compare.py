# light profile change for redshift change
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as C
import astropy.units as U
import astropy.io.fits as aft
import scipy.stats as sts
import find
import h5py 
import matplotlib.gridspec as grid
# setlect model
import astropy.wcs as awc
from astropy import cosmology as apcy
c00 = U.kpc.to(U.cm)
c01 = U.Mpc.to(U.pc)
c02 = U.Mpc.to(U.cm)
c03 = U.L_sun.to(U.erg/U.s)
c04 = U.rad.to(U.arcsec)
c05 = U.pc.to(U.cm)
Lsun = C.L_sun.value*10**7
# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
# read catalogue
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
z = z_eff[z_eff <= 0.3]
Ra = ra_eff[z_eff <= 0.3]
Dec = dec_eff[z_eff <= 0.3]
Rich = rich_eff[z_eff <= 0.3]
cg_ID = ID_eff[z_eff <= 0.3]
size_cluster = 2. # assumptiom: cluster size is 2.Mpc/h
from ICL_angular_diameter_reshift import mark_by_self
A_size, A_d= mark_by_self(z,size_cluster)
view_d = A_size*U.rad
R_A = 0.5*view_d.to(U.arcsec) # angular radius in angular second unit

use_z = redshift*1
use_rich = richness*1
use_ID = host_ID*1
pixel = 0.396
z_ref = 0.25
Nbins = 25
####### read the S/S0, richness
with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/cluster_record.h5') as f:
    tot_sub = np.array(f['a'][0])
    inr_sub = np.array(f['a'][1])
    sub_ratio = np.array(f['a'][2])
    area_ratio = np.array(f['a'][3])
    reference_ratio = np.array(f['a'][4])
    rich = np.array(f['a'][5])
sampl_S_ratio = area_ratio[(z<=0.3) & (z>=0.2)]
sampl_refer = reference_ratio[(z<=0.3) & (z>=0.2)]
sampl_z = z[(z<=0.3) & (z>=0.2)]
sampl_rich = rich[(z<=0.3) & (z>=0.2)]
sampl_R_A = R_A[(z<=0.3) & (z>=0.2)]
sampl_ID = cg_ID[(z<=0.3) & (z>=0.2)]
## divide bins
bins = 5
a0 = np.min(sampl_S_ratio)
b0 = np.max(sampl_S_ratio)
bins_S = np.linspace(a0,b0,bins+1)

def flux_scale(data,z,zref):
    obs = data 
    z0 = z
    z_stak = zref
    ref_data = obs*(1+z0)**4/(1+z_stak)**4 
    return ref_data

def flux_reunit(flux, area, z):
    F = flux
    A = area
    Z0 = z
    Da = Test_model.angular_diameter_distance(Z0).value
    S = Da**2*A*10**12/c04**2 # in unit pc^2
    L = 4*np.pi*(1+Z0)**4*Da**2*F*3.631*10**(-29)*c02**2*Lsun**(-1)
    LoA = L/S
    return LoA

def bin_light(data_in, Rp, N, cx, cy, z0, z_ref):
    R_pixel = Rp
    cx = cx
    cy = cy
    x0 = np.linspace(0,2047,2048)
    y0 = np.linspace(0,1488,1489)
    pix_id = np.array(np.meshgrid(x0,y0)) #data grid for original data  
    Nbins = N
    f_data = data_in
    r = np.logspace(1e-2, np.log10(R_pixel), Nbins)
    ia = r<= 2
    ib = np.array(np.where(ia == True))
    ic = ib.shape[1]
    R = (r/R_pixel)*10**3# in unit kpc/h
    R = R[np.max(ib):]
    dr = np.sqrt((pix_id[0]-cx)**2+(pix_id[1]-cy)**2)
    light = np.zeros((2,len(r)-ic+1), dtype = np.float)
    thero_l = np.zeros(len(r)-ic+1, dtype = np.float)
    for q in range(2):
        for k in range(1,len(r)):
            if q == 0:
                if r[k] <= 2:
                    ig = r <= 2
                    ih = np.array(np.where(ig == True))
                    im = np.max(ih)
                    ir = dr < r[im]
                    io = np.where(ir == True)
                    iy = io[0]
                    ix = io[1]
                    num = len(ix)
                    tot_flux = np.sum(f_data[iy,ix])/num
                    tot_area = pixel**2
                    light[q,0] = 22.5-2.5*np.log10(tot_flux)+2.5*np.log10(tot_area)
                    loa = flux_reunit(tot_flux, tot_area, z0)
                    thero_l[0] = 4.83-2.5*np.log10(loa)+10*np.log10(1+z0)-21.572 
                    k = im+1 
                else:
                    ir = (dr >= r[k-1]) & (dr < r[k])
                    io = np.where(ir == True)
                    iy = io[0]
                    ix = io[1]
                    num = len(ix)
                    tot_flux = np.sum(f_data[iy,ix])/num
                    tot_area = pixel**2
                    light[q,k-im] = 22.5-2.5*np.log10(tot_flux)+2.5*np.log10(tot_area) # mag/arcsec^2
                    loa = flux_reunit(tot_flux, tot_area, z0)
                    thero_l[k-im] = 4.83-2.5*np.log10(loa)+10*np.log10(1+z0)-21.572
            else:
                data = flux_scale(f_data, z0, z_ref)
                if r[k] <= 2:
                    ig = r <= 2
                    ih = np.array(np.where(ig == True))
                    im = np.max(ih)
                    ir = dr < r[im]
                    io = np.where(ir == True)
                    iy = io[0]
                    ix = io[1]
                    num = len(ix)
                    tot_flux = np.sum(data[iy,ix])/num
                    tot_area = pixel**2
                    light[q,0] = 22.5-2.5*np.log10(tot_flux)+2.5*np.log10(tot_area)
                    loa = flux_reunit(tot_flux, tot_area, z0)
                    thero_l[0] = 4.83-2.5*np.log10(loa)+10*np.log10(1+z0)-21.572
                    k = im+1
                else:
                    ir = (dr >= r[k-1]) & (dr < r[k])
                    io = np.where(ir == True)
                    iy = io[0]
                    ix = io[1]
                    num = len(ix)
                    tot_flux = np.sum(data[iy,ix])/num
                    tot_area = pixel**2
                    light[q,k-im] = 22.5-2.5*np.log10(tot_flux)+2.5*np.log10(tot_area) # mag/arcsec^2
                    loa = flux_reunit(tot_flux, tot_area, z0)
                    thero_l[k-im] = 4.83-2.5*np.log10(loa)+10*np.log10(1+z0)-21.572
    Da = Test_model.angular_diameter_distance(z0).value
    R_angu = ((R/1000)/Da)*c04
    return light, R, thero_l, R_angu

for k in range(bins):
    eta0 = bins_S[k]
    eta1 = bins_S[k+1]
    R_A_sub = sampl_R_A[(sampl_S_ratio>=eta0) & (sampl_S_ratio<=eta1)]
    lamda_sub = sampl_rich[(sampl_S_ratio>=eta0) & (sampl_S_ratio<=eta1)]
    id_sub = sampl_ID[(sampl_S_ratio>=eta0) & (sampl_S_ratio<=eta1)]
    z_sub = sampl_z[(sampl_S_ratio>=eta0) & (sampl_S_ratio<=eta1)] # the four lines use to get the selection sample 
    a1 = np.min(z_sub)
    b1 = np.max(z_sub)
    bins_z = np.linspace(a1,b1,bins+1)
    f = plt.figure(figsize = (30,30))
    f.suptitle(r'$S/S0_{%.3f \sim %.3f}$'%(eta0,eta1), fontsize = 30) # set the figure title
    spc = grid.GridSpec(ncols = 5,nrows = 5,figure = f)
    for p in range(bins):
        z0 = bins_z[p]
        z1 = bins_z[p+1]
        mean_z = (z0 + z1) /2
        try:
            R_A_array = R_A_sub[(z_sub>=z0) & (z_sub<=z1)]
            lamda_array = lamda_sub[(z_sub>=z0) & (z_sub<=z1)]
            id_array = id_sub[(z_sub>=z0) & (z_sub<=z1)]
            a2 = np.min(lamda_array)
            b2 = np.max(lamda_array)
            bins_L = np.logspace(np.log10(a2),np.log10(b2),bins+1)
            c1 = (bins_L[0] + bins_L[1]) /2
            c2 = (bins_L[1] + bins_L[2]) /2
            c3 = (bins_L[2] + bins_L[3]) /2
            c4 = (bins_L[3] + bins_L[4]) /2
            c5 = (bins_L[4] + bins_L[5]) /2
            k1 = find.find1d(np.abs(lamda_array -c1),np.min( np.abs(lamda_array -c1)))
            k2 = find.find1d(np.abs(lamda_array -c2),np.min( np.abs(lamda_array -c2)))
            k3 = find.find1d(np.abs(lamda_array -c3),np.min( np.abs(lamda_array -c3)))
            k4 = find.find1d(np.abs(lamda_array -c4),np.min( np.abs(lamda_array -c4)))
            k5 = find.find1d(np.abs(lamda_array -c5),np.min( np.abs(lamda_array -c5)))
            # check the goal clusters
            #### first
            di1 = find.find1d(use_ID,id_array[k1])
            ti1 = 0
            di1 = np.array([di1,di1])
            goal1 = di1[ti1] # find the goal cluster information
            ra1 = RA[goal1]
            dec1 = DEC[goal1]
            r1 = R_A_array[k1].value/0.396
            z_ref1 = use_z[goal1]
            # find the satellite and exclude center galaxy
            sum_IA1 = np.sum(rept_ID[1][:goal1])
            poa1 = member_pos[0][sum_IA1:sum_IA1+rept_ID[1][goal1]]
            pob1 = member_pos[1][sum_IA1:sum_IA1+rept_ID[1][goal1]]
            ak1 = find.find1d(poa1,ra1)
            posx1 = poa1[poa1!=poa1[ak1]]
            posy1 = pob1[pob1!=pob1[ak1]]
            clust1 = aft.getdata(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            ('r',ra1,dec1,z_ref1),header = True)
            wcs1 = awc.WCS(clust1[1])
            tx1, ty1 = wcs1.all_world2pix(posx1*U.deg, posy1*U.deg, 1)
            x01, y01 = wcs1.all_world2pix(ra1*U.deg, dec1*U.deg, 1)
            light1, R1, thero_L1, R_angu1 = bin_light(clust1[0], r1, Nbins, x01, y01, z_ref1, z_ref)
            ax1 = f.add_subplot(spc[p,0])
            # ax1.plot(R1, light1[0,:], c = 'b', label = 'SB_z0')
            # ax1.plot(R1, light1[1,:], c = 'r', label = 'SB_zref')
            ax1.plot(R_angu1, light1[0,:], c = 'b', label = 'SB_z0')
            ax1.plot(R_angu1, light1[1,:], c = 'r', label = 'SB_zref')
            ax1.plot(R_angu1, thero_L1, c = 'g', label = 'theory')
            
            ax1.set_xscale('log')
            ax1.set_title('SB ra%.3f dec%.3f z%.3f rich%.3f'%(ra1, dec1, z_ref1, lamda_array[k1]), fontsize = 10)
            # ax1.set_xlabel('R [kpc]')
            ax1.set_xlabel('R [arcsec]')
            ax1.set_ylabel('SB [mag/arcsec^2]')
            ax1.invert_yaxis()
            ax1.legend(loc = 1)
            
            #### second
            di2 = find.find1d(use_ID,id_array[k2])
            ti2 = 0
            di2 = np.array([di2,di2])
            goal2 = di2[ti2] # find the goal cluster
            ra2 = RA[goal2]
            dec2 = DEC[goal2]
            r2 = R_A_array[k2].value/0.396
            z_ref2 = use_z[goal2]
            # find the satellite and exclude center galaxy
            sum_IA2 = np.sum(rept_ID[1][:goal2])
            poa2 = member_pos[0][sum_IA2:sum_IA2+rept_ID[1][goal2]]
            pob2 = member_pos[1][sum_IA2:sum_IA2+rept_ID[1][goal2]]
            ak2 = find.find1d(poa2,ra2)
            posx2 = poa2[poa2!=poa2[ak2]]
            posy2 = pob2[pob2!=pob2[ak2]]
            clust2 = aft.getdata(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            ('r',ra2,dec2,z_ref2),header = True)
            wcs2 = awc.WCS(clust2[1])            
            tx2, ty2 = wcs2.all_world2pix(posx2*U.deg, posy2*U.deg, 1)
            x02, y02 = wcs2.all_world2pix(ra2*U.deg, dec2*U.deg, 1)
            light2, R2, thero_L2, R_angu2 = bin_light(clust2[0], r2, Nbins, x02, y02, z_ref2, z_ref)
            ax2 = f.add_subplot(spc[p,1])
            # ax2.plot(R2, light2[0,:], c = 'b', label = 'SB_z0')
            # ax2.plot(R2, light2[1,:], c = 'r', label = 'SB_zref')
            ax2.plot(R_angu2, light2[0,:], c = 'b', label = 'SB_z0')
            ax2.plot(R_angu2, light2[1,:], c = 'r', label = 'SB_zref')
            ax2.plot(R_angu2, thero_L2, c = 'g', label = 'theory')
            
            ax2.set_xscale('log')
            ax2.set_title('SB ra%.3f dec%.3f z%.3f rich%.3f'%(ra2, dec2, z_ref2, lamda_array[k2]), fontsize = 10)
            # ax2.set_xlabel('R [kpc]')
            ax2.set_xlabel('R [arcsec]')
            ax2.set_ylabel('SB [mag/arcsec^2]')
            ax2.invert_yaxis()
            ax2.legend(loc = 1)
            
            #### third
            di3 = find.find1d(use_ID,id_array[k3])
            ti3 = 0
            di3 = np.array([di3,di3])
            goal3 = di3[ti3] # find the goal cluster
            ra3 = RA[goal3]
            dec3 = DEC[goal3]
            r3 = R_A_array[k3].value/0.396
            z_ref3 = use_z[goal3]
            # find the satellite and exclude center galaxy
            sum_IA3 = np.sum(rept_ID[1][:goal3])
            poa3 = member_pos[0][sum_IA3:sum_IA3+rept_ID[1][goal3]]
            pob3 = member_pos[1][sum_IA3:sum_IA3+rept_ID[1][goal3]]
            ak3 = find.find1d(poa3,ra3)
            posx3 = poa3[poa3!=poa3[ak3]]
            posy3 = pob3[pob3!=pob3[ak3]]
            clust3 = aft.getdata(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            ('r',ra3,dec3,z_ref3),header = True)
            wcs3 = awc.WCS(clust3[1])
            tx3, ty3 = wcs3.all_world2pix(posx3*U.deg, posy3*U.deg, 1)
            x03, y03 = wcs3.all_world2pix(ra3*U.deg, dec3*U.deg, 1)
            light3, R3, thero_L3, R_angu3 = bin_light(clust3[0], r3, Nbins, x03, y03, z_ref3, z_ref)
            ax3 = f.add_subplot(spc[p,2])
            # ax3.plot(R3, light3[0,:], c = 'b', label = 'SB_z0')
            # ax3.plot(R3, light3[1,:], c = 'r', label = 'SB_zref')
            ax3.plot(R_angu3, light3[0,:], c = 'b', label = 'SB_z0')
            ax3.plot(R_angu3, light3[1,:], c = 'r', label = 'SB_zref') 
            ax3.plot(R_angu3, thero_L3, c = 'g', label = 'theory')
            
            ax3.set_xscale('log')
            ax3.set_title('SB ra%.3f dec%.3f z%.3f rich%.3f'%(ra3, dec3, z_ref3, lamda_array[k3]), fontsize = 10)
            #ax3.set_xlabel('R [kpc]')
            ax3.set_xlabel('R [arcsec]')
            ax3.set_ylabel('SB [mag/arcsec^2]')
            ax3.invert_yaxis()
            ax3.legend(loc = 1)
            
            #### fourth
            di4 = find.find1d(use_ID,id_array[k4])
            ti4 = 0
            di4 = np.array([di4,di4])
            goal4 = di4[ti4] # find the goal cluster
            ra4 = RA[goal4]
            dec4 = DEC[goal4]
            r4 = R_A_array[k4].value/0.396
            z_ref4 = use_z[goal4]
            # find the satellite and exclude center galaxy
            sum_IA4 = np.sum(rept_ID[1][:goal4])
            poa4 = member_pos[0][sum_IA4:sum_IA4+rept_ID[1][goal4]]
            pob4 = member_pos[1][sum_IA4:sum_IA4+rept_ID[1][goal4]]
            ak4 = find.find1d(poa4,ra4)
            posx4 = poa4[poa4!=poa4[ak4]]
            posy4 = pob4[pob4!=pob4[ak4]]
            clust4 = aft.getdata(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            ('r',ra4,dec4,z_ref4),header = True)
            wcs4 = awc.WCS(clust4[1])
            tx4, ty4 = wcs4.all_world2pix(posx4*U.deg, posy4*U.deg, 1)
            x04, y04 = wcs4.all_world2pix(ra4*U.deg, dec4*U.deg, 1)
            light4, R4, thero_L4, R_angu4 = bin_light(clust4[0], r4, Nbins, x04, y04, z_ref4, z_ref)
            ax4 = f.add_subplot(spc[p,3])
            # ax4.plot(R4, light4[0,:], c = 'b', label = 'SB_z0')
            # ax4.plot(R4, light4[1,:], c = 'r', label = 'SB_zref')
            ax4.plot(R_angu4, light4[0,:], c = 'b', label = 'SB_z0')
            ax4.plot(R_angu4, light4[1,:], c = 'r', label = 'SB_zref')
            ax4.plot(R_angu4, thero_L4, c = 'g', label = 'theory')
            
            ax4.set_xscale('log')
            ax4.set_title('SB ra%.3f dec%.3f z%.3f rich%.3f'%(ra4, dec4, z_ref4, lamda_array[k4]), fontsize = 10)
            # ax4.set_xlabel('R [kpc]')
            ax4.set_xlabel('R [arcsec]')
            ax4.set_ylabel('SB [mag/arcsec^2]')
            ax4.invert_yaxis()
            ax4.legend(loc = 1)
            
            #### fifth
            di5 = find.find1d(use_ID,id_array[k5])
            ti5 = 0
            di5 = np.array([di5,di5])
            goal5 = di5[ti5] # find the goal cluster
            ra5 = RA[goal5]
            dec5 = DEC[goal5]
            r5 = R_A_array[k5].value/0.396
            z_ref5 = use_z[goal5]
            # find the satellite and exclude center galaxy
            sum_IA5 = np.sum(rept_ID[1][:goal5])
            poa5 = member_pos[0][sum_IA5:sum_IA5+rept_ID[1][goal5]]
            pob5 = member_pos[1][sum_IA5:sum_IA5+rept_ID[1][goal5]]
            ak5 = find.find1d(poa5,ra5)
            posx5 = poa5[poa5!=poa5[ak5]]
            posy5 = pob5[pob5!=pob5[ak5]]
            clust5 = aft.getdata(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            ('r',ra5,dec5,z_ref5),header = True)
            wcs5 = awc.WCS(clust5[1])
            tx5, ty5 = wcs5.all_world2pix(posx5*U.deg, posy5*U.deg, 1)
            x05, y05 = wcs5.all_world2pix(ra5*U.deg, dec5*U.deg, 1)        
            light5, R5, thero_L5, R_angu5 = bin_light(clust5[0], r5, Nbins, x05, y05, z_ref5, z_ref)
            ax5 = f.add_subplot(spc[p,4])
            # ax5.plot(R5, light5[0,:], c = 'b', label = 'SB_z0')
            # ax5.plot(R5, light5[1,:], c = 'r', label = 'SB_zref')
            ax5.plot(R_angu5, light5[0,:], c = 'b', label = 'SB_z0')
            ax5.plot(R_angu5, light5[1,:], c = 'r', label = 'SB_zref')
            ax5.plot(R_angu5, thero_L5, c = 'g', label = 'theory')
            
            ax5.set_xscale('log')
            ax5.set_title('SB ra%.3f dec%.3f z%.3f rich%.3f'%(ra5, dec5, z_ref5, lamda_array[k5]), fontsize = 10)
            # ax5.set_xlabel('R [kpc]')
            ax5.set_xlabel('R [arcsec]')
            ax5.set_ylabel('SB [mag/arcsec^2]')
            ax5.invert_yaxis()
            ax5.legend(loc = 1)    
        except ValueError:
            continue
    plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/light_profile%.0f.pdf'%k,dpi=600)
    plt.close()