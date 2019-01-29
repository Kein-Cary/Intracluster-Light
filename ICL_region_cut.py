"""
this file use to cut square region from the frame files
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
import astropy.io.fits as fits
import astropy.wcs as awc
import h5py
from scipy.interpolate import interp2d as inter2
# parameter set
c0 = U.kpc.to(U.cm)
c1 = U.Mpc.to(U.pc)
c2 = U.Mpc.to(U.cm)
c3 = U.L_sun.to(U.erg/U.s)
c4 = U.rad.to(U.arcsec)
Lsun = C.L_sun.value*10**7
# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.250 
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2
select_a = 500 # in unit pixel
Da_ref = Test_model.angular_diameter_distance(z_ref).value
R_ref = Da_ref*select_a*pixel/c4
scal_ref = select_a/R_ref # 1Mpc = scal_ref pixel

def catalog_check(t):
    N = 0
    # read the catalog
    goal_data = fits.getdata(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
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
    cut = 500 # in unit pixel
    z_record = []
    ra_record = []
    dec_record = []
    rich_record = []
    for q in range(len(z)):
        cir_data = fits.getdata(
            '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
            ('r',Ra[q],Dec[q],z[q]),header = True)
        wcs = awc.WCS(cir_data[1])
        cx,cy = wcs.all_world2pix(Ra[q]*U.deg,Dec[q]*U.deg,1)
        a0 = cx-cut
        a1 = cx+cut
        b0 = cy-cut
        b1 = cy+cut
        ref1 = cir_data[0].shape[1]-1
        ref2 = cir_data[0].shape[0]-1
        if ((a0 >= 0)&(a1 <= ref1))&((b0 >= 0)&(b1 <= ref2)):
            z_record.append(z[q])
            ra_record.append(Ra[q])
            dec_record.append(Dec[q])
            rich_record.append(rich[q])
            N = N + 1
        else:
            N = N
    z_record = np.array(z_record)
    ra_record = np.array(ra_record)
    dec_record = np.array(dec_record)
    rich_record = np.array(rich_record)
    record = np.array([z_record, ra_record, dec_record, rich_record])
    with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/cut_record.h5','w') as f:
        f['a'] = record
    with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/cut_record.h5') as f:
        for k in range(len(record)):
            f['a'][k,:] = record[k,:]
    return N
###
def flux_scale(data,z,zref):
    obs = data+1000
    z0 = z
    z_stak = zref
    ref_data = obs*(1+z0)**4/(1+z_stak)**4 -1000*(1+z0)**4/(1+z_stak)**4
    return ref_data

def extractor(data,x0,x1,y0,y1):
    a0 = np.int0(x0)
    a1 = np.int0(x1)
    b0 = np.int0(y0)
    b1 = np.int0(y1)
    M = np.array(data)
    n0 = np.max([a0,0])
    n1 = np.min([a1,2047])
    m0 = np.max([b0,0])
    m1 = np.min([b1,1488])
    M_use = M[m0:m1+1,n0:n1+1]
    return M_use
###
#Ntot = catalog_check(t = True)
Ntot = np.int0(706)    
with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/cut_record.h5') as f:
    z = f['a'][0,:]
    ra = f['a'][1,:]
    dec = f['a'][2,:]
    rich = f['a'][3,:]
band = ['u','g','r','i','z']
pos_record = np.zeros((len(z),2), dtype = np.float)
R_stac = np.zeros((len(z),2), dtype = np.float)
R = np.zeros(len(z), dtype = np.float)
NMGY = np.zeros(len(z), dtype = np.float)
scal = np.zeros(len(z), dtype = np.float)
image = {}
for k in range(len(z)):
    for q in range(len(band)):
        cut_data = fits.getdata(
                '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
                (band[q],ra[k],dec[k],z[k]),header = True)
        wcs1 = awc.WCS(cut_data[1])
        transf = cut_data[1]['NMGY']
        NMGY[k] = transf
        z1 = z[k]
        ra1 = ra[k]
        dec1 = dec[k]
        Da1 = Test_model.angular_diameter_distance(z1).value
        R1 = Da1*select_a*pixel/c4
        R[k] = R1
        cx1, cy1 = wcs1.all_world2pix(ra1*U.deg, dec1*U.deg, 1)
        scal1 = select_a/R1
        scal[k] = scal1
        # cut region
        a0 = np.floor(cx1 - select_a)
        a1 = np.ceil(cx1 + select_a)
        b0 = np.floor(cy1 - select_a)
        b1 = np.ceil(cy1 + select_a)
        mirr0 = extractor(cut_data[0],a0,a1,b0,b1)
        oa = (cx1 - select_a) - np.floor(cx1 - select_a)
        ob = (cy1 - select_a) - np.floor(cy1 - select_a)
        rx, ry = np.array([select_a+oa,select_a+ob])
        # scale the select data to same metrix length
        xx = np.linspace(a0, a1, np.int0(a1 -a0 +1))
        yy = np.linspace(b0, b1, np.int0(b1 -b0 +1))
        fd = inter2(xx,yy,mirr0)
        ox = np.linspace(a0, a1, np.int0(select_a*2))
        oy = np.linspace(b0, b1, np.int0(select_a*2))
        mirr1 = fd(ox,oy)
        # flux re-calculate and resample
        interf = flux_scale(mirr1, z[k], z_ref)
        scal_data = interf/(scal_ref/scal1)
        image['%.0f'%k] = scal_data
        R_stac[k,0] = (mirr1.shape[1]/2)/scal_ref
        R_stac[k,1] = (mirr1.shape[0]/2)/scal_ref
        pos_record[k,0] = rx
        pos_record[k,1] = ry
        # save the cut region
        x0 = scal_data.shape[1]
        y0 = scal_data.shape[0]
        keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2',
                'NMGY','CRVAL1','CRVAL2','ORIGN_Z','Z_REF','SAMP_N']
        value = ['T', 32, 2, x0, y0, rx, ry, NMGY[k], ra1, dec1, z1, z_ref, Ntot]
        ff = dict(zip(keys,value))
        fil = fits.Header(ff)
        fits.writeto(
                '/mnt/ddnfs/data_users/cxkttwl/ICL/data/square_cut/stacking_image_%s_ra%.3f_dec%.3f.fits'%(
                band[q],ra1,dec1), scal_data, header = fil, overwrite=True)   
    print('finish-----%.3f'%(k/len(z)))
    print('R_stac=',R_stac[k])
### other data save
aray = np.array([pos_record[:,0].T, pos_record[:,1].T, 
                 R_stac[:,0].T, R_stac[:,1].T,
                 R, NMGY, scal])
with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/stacking_record.h5','w') as f:
    f['a'] = aray
with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/stacking_record.h5') as f:
    for t in range(len(aray)):
        f['a'][t,:] = aray[t,:]
        
shape_a,shape_b = image['0'].shape  
sum_f = np.zeros((len(z),shape_a,shape_b),dtype = np.float)
for p in range(len(image)):
    sum_f[p,:] = image['%.0f'%p]
with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/stacking_data.h5','w') as f:
    f['a'] = sum_f
with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/stacking_record.h5') as f:
    for l in range(len(image)):
        f['a'][l,:] = sum_f[l,:]
        