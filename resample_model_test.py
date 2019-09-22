import matplotlib as mpl
import matplotlib.pyplot as plt

import random
import numpy as np
import astropy.io.fits as fits

from resamp import gen
from resample_modelu import down_samp
from resample_modelu import sum_samp
from scipy.interpolate import interp2d as interp2

pixel = 0.396
def resamp_block():
    """
    mu : the pixel size ratio (mu = Pixel_new / Pixel_old)
    """
    data = fits.getdata('sub_img.fits', header = True)
    img = data[0]

    eta = np.linspace(0.844, 1.185, 11)
    for kk in range(len(eta)):
        #mu = eta[kk]
        mu = 0.84

        Nx = img.shape[1]
        Ny = img.shape[0]
        x_set = np.array( [random.randint(15, img.shape[1] - 15) for _ in range(2000)] )
        y_set = np.array( [random.randint(15, img.shape[0] - 15) for _ in range(2000)] )
        f_set = img[y_set, x_set] / pixel**2

        ## single resample
        '''
        cx, cy, cimg = gen(img, 1, mu, Nx / 2, Ny / 2 ) # previous code
        if mu > 1:
            cimg = cimg[1:, 1:]
        elif mu == 1:
            cimg = cimg[1:-1, 1:-1]
        else:
            cimg = cimg
        cx = np.int(cx)
        cy = np.int(cy)
        '''
        if mu > 1:
            cimg, cx, cy = sum_samp(mu, mu, img, Nx / 2, Ny / 2)
        else:
            cimg, cx, cy = down_samp(mu, mu, img, Nx / 2, Ny / 2)
        cx = np.int(cx)
        cy = np.int(cy)

        x_set1 = np.array( [np.int( ll / mu ) for ll in x_set] )
        y_set1 = np.array( [np.int( ll / mu ) for ll in y_set] )
        f_set1 = cimg[y_set1, x_set1] / (mu*pixel)**2

        fig = plt.figure(figsize = (16, 8))
        fig.suptitle('Pixel scale %.3f' % mu)
        ax0 = plt.subplot(121)
        ax1 = plt.subplot(122)
        ax0.set_title('original image')

        tf = ax0.imshow(img / pixel**2, cmap = 'Greys', vmin = 1e-5, vmax = 1e2, origin = 'lower', norm = mpl.colors.LogNorm())
        plt.colorbar(tf, ax = ax0, fraction = 0.047, pad = 0.01, label = '$Pixel \; SB \, [nmaggy/arcsec^2]$')
        ax1.set_title('original image')
        tf = ax1.imshow(cimg, cmap = 'Greys', vmin = 1e-5, vmax = 1e2, origin = 'lower', norm = mpl.colors.LogNorm())
        plt.colorbar(tf, ax = ax1, fraction = 0.047, pad = 0.01, label = '$Pixel \; SB \, [nmaggy/arcsec^2]$')

        plt.tight_layout()
        plt.savefig('re_correct_resample_%.3f_scale.png' % mu, dpi = 300)
        plt.close()
        #raise
        ## inverse resample
        kap = 1 / mu
        '''
        c2x, c2y, c2img = gen(cimg, 1, kap, np.int(cx), np.int(cy) ) # previous code
        if mu < 1:
            c2img = c2img[1:,1:]
        elif mu == 1:
            c2img = c2img[1:-1, 1:-1]
        else:
            c2img = c2img
        '''
        if kap > 1:
            c2img, c2x, c2y = sum_samp(kap, kap, cimg, cx, cy)
        else:
            c2img, c2x, c2y = down_samp(kap, kap, cimg, cx, cy)
        c2x = np.int(c2x)
        c2y = np.int(c2y)
        x_set2 = np.array( [np.int( ll / kap ) for ll in x_set1] )
        y_set2 = np.array( [np.int( ll / kap ) for ll in y_set1] )
        f_set2 = c2img[y_set2, x_set2] / pixel**2
        '''
        plt.figure()
        ax = plt.subplot(111)
        ax.set_title('Pixel scale %.2f' % mu)
        ax.scatter(f_set, f_set1, s = 10, marker = 'o', facecolors = 'b', edgecolors = 'b', label = 'single resample', alpha = 0.5)
        ax.scatter(f_set, f_set2, s = 10, marker = 'o', facecolors = 'g', edgecolors = 'g', label = 'twice resample', alpha = 0.5)
        ax.plot(f_set, f_set, 'r--', label = 'slope = 1', alpha = 0.5)
        ax.set_xlabel('Pixel SB [Before resampling]')
        ax.set_ylabel('Pixel SB [After resampling]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc = 2)
        plt.savefig('pix_SB_scale_%.2f.png' % mu, dpi = 300)
        plt.show()
        '''
        cl0 = (c2img.shape[0] - 3) / 2
        cl1 = (c2img.shape[1] - 3) / 2
        cut_0 = img[np.int(Ny/2 - cl0 - 1): np.int(Ny/2 + cl0), np.int(Nx/2 - cl1 - 1): np.int(Nx/2 + cl1)]
        cut_2 = c2img[np.int(c2y - cl0): np.int(c2y + cl0 + 1), np.int(c2x - cl1): np.int(c2x + cl1 + 1)]
        ddbs = (cut_2 - cut_0) / cut_0

        fig = plt.figure(figsize = (16, 8))
        fig.suptitle('pixel scale %.3f' % mu)
        ax0 = plt.subplot(221)
        ax1 = plt.subplot(222)
        ax2 = plt.subplot(223)
        ax3 = plt.subplot(224)

        ax0.set_title('original image')
        tf = ax0.imshow(img / pixel**2, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
        plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = '$Pixel \; SB \, [nmaggy/arcsec^2]$')

        ax1.set_title('single resample image')
        tf = ax1.imshow(cimg / (mu * pixel)**2, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
        plt.colorbar(tf, ax = ax1, fraction = 0.035, pad = 0.01, label = '$Pixel \; SB \, [nmaggy/arcsec^2]$')

        ax2.set_title('twice resample image')
        tf = ax2.imshow(c2img / pixel**2, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
        plt.colorbar(tf, ax = ax2, fraction = 0.035, pad = 0.01, label = '$Pixel \; SB \, [nmaggy/arcsec^2]$')

        ax3.set_title('twice resample - original / origin')
        from matplotlib import cm
        tf = ax3.imshow(ddbs, origin = 'lower', norm = mpl.colors.SymLogNorm(linthresh=0.01), cmap=cm.PRGn)
        plt.colorbar(tf, ax = ax3, fraction = 0.035, pad = 0.01)

        plt.tight_layout()
        plt.subplots_adjust(bottom = 0.1, right = 0.8, top = 0.9)
        plt.savefig('Twice-resample_scale_%.3f.png' % mu, dpi = 300)
        plt.show()

        raise
    return

def project():
    import h5py
    import astropy.wcs as awc
    from matplotlib import cm
    from reproject import reproject_exact

    with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
        catalogue = np.array(f['a'])
    z = catalogue[0]
    ra = catalogue[1]
    dec = catalogue[2]

    # get a example of "wcs" information
    load = '/home/xkchen/mywork/ICL/data/total_data/'
    file = load + 'frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (ra[0], dec[0], z[0])
    data = fits.open(file)
    img = data[0].data
    wcs = awc.WCS(data[0].header)
    '''
    # cut region for detail test
    nl = 50
    sub_img = img[758 - nl: 758 + nl, 1943 - nl: 1943 + nl]
    keys = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'CRPIX1', 'CRPIX2', 'CENTER_X', 'CENTER_Y', 'UNIT', 'P_SCALE']
    values = ['T', 32, 2, sub_img.shape[1], sub_img.shape[0], np.int(nl), np.int(nl), np.int(nl), np.int(nl), 'NMAGGY', 0.396]
    ff = dict(zip(keys, values))
    fil = fits.Header(ff)
    fits.writeto('sub_img_1.fits', sub_img, header = fil, overwrite=True)
    '''
    eta = 0.95
    #eta = 0.844# New_pix / Old_pix

    # cut a region for test
    nl = 50
    sub_img = img[939 - nl: 939 + nl, 1407 - nl: 1407 + nl]
    #sub_img = np.ones((nl, nl), dtype = np.float) # constant field

    # initial "wcs" for the cut region
    Nx_0 = np.int(sub_img.shape[1])
    Ny_0 = np.int(sub_img.shape[0])
    Cx_0 = np.int(nl)
    Cy_0 = np.int(nl)
    typ1_0 = data[0].header['CTYPE1']
    typ2_0 = data[0].header['CTYPE2']
    cra_0 = data[0].header['CRVAL1']
    cdec_0 = data[0].header['CRVAL2']
    CD11_0 = data[0].header['CD1_1']
    CD12_0 = data[0].header['CD1_2']
    CD21_0 = data[0].header['CD2_1']
    CD22_0 = data[0].header['CD2_2']

    keys = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'CRPIX1', 'CRPIX2', 'CTYPE1', 'CTYPE2', 
            'CRVAL1', 'CRVAL2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', ]
    values = ['T', 32, 2, Nx_0, Ny_0, Cx_0, Cy_0, typ1_0, typ2_0, cra_0, cdec_0, CD11_0, CD12_0, CD21_0, CD22_0]
    ff = dict(zip(keys, values))
    fil = fits.Header(ff)
    wcs0 = awc.WCS(fil)
    img_set = fits.ImageHDU(sub_img / pixel**2, header = fil)

    # set the wcs information for resample img
    Nx = np.int(Nx_0 / eta)
    Ny = np.int(Ny_0 / eta)
    Cx = np.int(nl / eta)
    Cy = np.int(nl / eta)
    typ1 = data[0].header['CTYPE1']
    typ2 = data[0].header['CTYPE2']
    cra = data[0].header['CRVAL1']
    cdec = data[0].header['CRVAL2']
    CD11 = data[0].header['CD1_1'] * eta
    CD12 = data[0].header['CD1_2'] * eta
    CD21 = data[0].header['CD2_1'] * eta
    CD22 = data[0].header['CD2_2'] * eta

    keys = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'CRPIX1', 'CRPIX2', 'CTYPE1', 'CTYPE2', 
            'CRVAL1', 'CRVAL2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', ]
    values = ['T', 32, 2, Nx, Ny, Cx, Cy, typ1, typ2, cra, cdec, CD11, CD12, CD21, CD22]
    ff = dict(zip(keys, values))
    fil_1 = fits.Header(ff)
    wcs1 = awc.WCS(fil_1)

    # resample 1-- single resample; 2-- twice resample (pixel scale is the initial value)
    cimg1, pix_foot1 = reproject_exact(img_set, output_projection = wcs1, shape_out = (Ny, Nx))

    img_set1 = fits.ImageHDU(cimg1, header = fil)
    cimg2, pix_foot2 = reproject_exact(img_set1, output_projection = wcs0, shape_out = (Ny_0, Nx_0))

    cl0 = (cimg2.shape[0] - 15) / 2
    cl1 = (cimg2.shape[1] - 15) / 2
    cut_0 = sub_img[np.int(Cy_0 - cl0): np.int(Cy_0 + cl0 + 1), np.int(Cx_0 - cl1): np.int(Cx_0 + cl1 + 1)]
    cut_2 = cimg2[np.int(Cy_0 - cl0 + 3): np.int(Cy_0 + cl0 + 4), np.int(Cx_0 - cl1 + 3): np.int(Cx_0 + cl1 + 4)] # eta = 0.95
    #cut_2 = cimg2[np.int(Cy_0 - cl0 + 7): np.int(Cy_0 + cl0 + 8), np.int(Cx_0 - cl1 + 7): np.int(Cx_0 + cl1 + 8)] # eta = 0.844
    ddbs = (cut_2 - cut_0) / cut_0

    fig = plt.figure(figsize = (16, 8))
    fig.suptitle('pixel scale %.2f' % eta)
    ax0 = plt.subplot(221)
    ax1 = plt.subplot(222)
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(224)

    ax0.set_title('original image')
    tf = ax0.imshow(sub_img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
    plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = '$Pixel \; SB \, [nmaggy/arcsec^2]$')

    ax1.set_title('single resample image')
    tf = ax1.imshow(cimg1, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
    plt.colorbar(tf, ax = ax1, fraction = 0.035, pad = 0.01, label = '$Pixel \; SB \, [nmaggy/arcsec^2]$')

    ax2.set_title('twice resample image')
    tf = ax2.imshow(cimg2, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
    plt.colorbar(tf, ax = ax2, fraction = 0.035, pad = 0.01, label = '$Pixel \; SB \, [nmaggy/arcsec^2]$')

    ax3.set_title('twice resample - original / origin')
    tf = ax3.imshow(ddbs, origin = 'lower', norm = mpl.colors.SymLogNorm(linthresh=0.01), cmap=cm.PRGn)
    plt.colorbar(tf, ax = ax3, fraction = 0.035, pad = 0.01)

    plt.tight_layout()
    plt.subplots_adjust(bottom = 0.1, right = 0.8, top = 0.9)
    plt.savefig('Reproject_scale_%.2f.png' % eta, dpi = 300)
    plt.show()

    raise

    return

def main():
    resamp_block()
    #project()

if __name__ == "__main__":
    main()
