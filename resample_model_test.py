import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
from astropy import cosmology as apcy
# resample part
from resample_modelu import down_samp
from resample_modelu import sum_samp

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Lsun = C.L_sun.value*10**7

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

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

with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
    catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]

# resample process
x0 = np.linspace(0,2047,2048)
y0 = np.linspace(0,1488,1489)
pix_id = np.array(np.meshgrid(x0,y0))

def resamp_test():
    import time
    t0 = time.time()

    kd = 0
    zg = z[kd]
    rag = ra[kd]
    decg = dec[kd]
    load = '/home/xkchen/mywork/ICL/data/total_data/sample_02_03/'
    file = 'frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (rag, decg, zg)
    data = fits.open(load + file)
    img = data[0].data
    Head = data[0].header
    wcs = awc.WCS(Head)
    cx, cy = wcs.all_world2pix(rag*U.deg, decg*U.deg, 1)

    Da = Test_model.angular_diameter_distance(zg).value
    Da_ref = Test_model.angular_diameter_distance(z_ref).value

    #eta = Da_ref / Da
    eta = 1
    mu = 1 / eta
    if eta > 1:
        resam_data, cpos = sum_samp(eta, eta, img, cx, cy)
    else:
        resam_data, cpos = down_samp(eta, eta, img, cx, cy)
    cx1 = cpos[0]
    cy1 = cpos[1]

    t1 = time.time() - t0
    print(t1)

    plt.figure(figsize = (16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios = [1, 1])
    ax = plt.subplot(gs[0])
    bx = plt.subplot(gs[1])
    ax.imshow(img, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
    ax.scatter(cx, cy, s = 10, marker = 'o', facecolors = '', edgecolors = 'r', linewidth = 0.5, alpha = 0.5)
    bx.imshow(resam_data, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
    bx.scatter(cx1, cy1, s = 10, marker = 'o', facecolors = '', edgecolors = 'r', linewidth = 0.5, alpha = 0.5)

    plt.tight_layout()
    plt.savefig('/home/xkchen/mywork/ICL/code/resamp_test.png', dpi = 600)
    plt.show()
    plt.close()

    raise
    return

def main():
    resamp_test()
    
if __name__ == "__main__":
    main()