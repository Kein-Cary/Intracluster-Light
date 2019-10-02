#!/usr/bin/env python3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import time

def sum_samp(m1, m2, data, cx0, cy0):
    mx = m1
    my = m2

    Mrr = np.zeros((data.shape[0] + 1, data.shape[1] + 1), dtype = np.float)
    Mrr[:-1, :-1] = data * 1

    #Mrr = data
    N0x = data.shape[1]
    N0y = data.shape[0]

    N2 = np.int(N0y * (1 / my))
    N1 = np.int(N0x * (1 / mx))

    cx = cx0 / mx
    cy = cy0 / my
    cpos = np.array([cx, cy])
    resam = np.zeros((N2 + 1, N1 + 1), dtype = np.float)

    eps = 1e-12
    for k in range(N0y):

        for l in range(N0x):

            in_mx = 1. / mx
            in_my = 1. / my

            idy = (2*k + 1) * in_my /2
            idx = (2*l + 1) * in_mx /2

            sor_y = np.int(idy)
            sor_x = np.int(idx)

            xa = (idx - in_mx/2) - sor_x
            it = eps < np.abs(xa)
            xa = xa * it

            xb = (idx + in_mx/2) - (sor_x + 1)
            it = eps < np.abs(xb)
            xb = xb * it

            ya = (idy - in_my/2) - sor_y
            it = eps < np.abs(ya)
            ya = ya * it

            yb = (idy + in_my/2) - (sor_y + 1)
            it = eps < np.abs(yb)
            yb = yb * it

            S_pix = in_my * in_mx
            abs_x0 = np.abs( xa )
            abs_x1 = np.abs( xb )
            abs_y0 = np.abs( ya )
            abs_y1 = np.abs( yb )   

            if (xa >= 0) & (xb <= 0) & (ya >= 0) & (yb <= 0):
                resam[sor_y, sor_x] += Mrr[k, l]

            elif (xa < 0) & (xb < 0) & (ya >= 0) & (yb <= 0):
                resam[sor_y, sor_x]     += in_my * (in_mx - abs_x0) * Mrr[k, l] / S_pix
                resam[sor_y, sor_x - 1] += abs_x0 * in_my * Mrr[k, l] / S_pix

            elif (xa >= 0) & (xb <= 0) & (ya > 0) & (yb > 0):
                resam[sor_y + 1, sor_x] += abs_y1 * in_mx * Mrr[k, l] / S_pix
                resam[sor_y, sor_x]     += (in_my - abs_y1) * in_mx * Mrr[k, l] / S_pix

            elif (xa > 0) & (xb > 0) & (ya >= 0) & (yb <= 0):
                resam[sor_y, sor_x + 1] += abs_x1 * in_my * Mrr[k, l] / S_pix
                resam[sor_y, sor_x]     += (in_mx - abs_x1) * in_my * Mrr[k, l] / S_pix

            elif (xa >= 0) & (xb <= 0) & (ya < 0) & (yb < 0):
                resam[sor_y, sor_x]     += in_mx * (in_my - abs_y0) * Mrr[k, l] / S_pix
                resam[sor_y - 1, sor_x] += abs_y0 * in_mx * Mrr[k, l] / S_pix

            elif (xa > 0) & (xb > 0) & (ya > 0) & (yb > 0):               
                resam[sor_y, sor_x]         += (in_mx - abs_x1) * (in_my - abs_y1) * Mrr[k, l]/ S_pix
                resam[sor_y, sor_x + 1]     += abs_x1 * (in_my - abs_y1) * Mrr[k, l] / S_pix
                resam[sor_y + 1, sor_x]     += abs_y1 * (in_mx - abs_x1) * Mrr[k, l] / S_pix
                resam[sor_y + 1, sor_x + 1] += abs_x1 * abs_y1 * Mrr[k, l] / S_pix

            elif (xa < 0) & (xb < 0) & (ya < 0) & (yb < 0):
                resam[sor_y, sor_x]        += (in_mx - abs_x0) * (in_my - abs_y0) * Mrr[k, l] / S_pix
                resam[sor_y, sor_x - 1]    += abs_x0 * (in_my - abs_y0) *Mrr[k, l] / S_pix
                resam[sor_y - 1, sor_x]    += abs_y0 * (in_mx - abs_x0) * Mrr[k, l] / S_pix
                resam[sor_y - 1, sor_x -1] += abs_x0 * abs_y0 * Mrr[k, l] / S_pix

            elif (xa > 0) & (xb > 0) & (ya < 0) & (yb < 0):
                resam[sor_y, sor_x]         += (in_mx - abs_x1) * (in_my - abs_y0) * Mrr[k, l] / S_pix
                resam[sor_y - 1, sor_x]     += (in_mx - abs_x1) * abs_y0 * Mrr[k, l] / S_pix
                resam[sor_y, sor_x + 1]     += abs_x1 * (in_my - abs_y0) * Mrr[k, l] / S_pix
                resam[sor_y - 1, sor_x + 1] += abs_x1 * abs_y0 * Mrr[k, l] / S_pix

            elif (xa < 0) & (xb < 0) & (ya > 0) & (yb > 0):
                resam[sor_y, sor_x]         += (in_mx - abs_x0) * (in_my - abs_y1) * Mrr[k, l] / S_pix
                resam[sor_y, sor_x - 1]     += abs_x0 * (in_my - abs_y1) * Mrr[k, l] / S_pix
                resam[sor_y + 1, sor_x]     += abs_y1 * (in_mx - abs_x0) * Mrr[k, l] / S_pix
                resam[sor_y + 1, sor_x - 1] += abs_x0 * abs_y1 * Mrr[k, l] / S_pix

            else:
                continue

    return resam, cx, cy

def down_samp(m3, m4, data, cx0, cy0):
    mx = m3
    my = m4

    Mrr = np.zeros((data.shape[0] + 1, data.shape[1] + 1), dtype = np.float)
    Mrr[:-1, :-1] = data * 1

    sNy = data.shape[0]
    sNx = data.shape[1]

    Nx = np.int(sNx * (1/mx))
    Ny = np.int(sNy * (1/my))

    cx = cx0 / mx
    cy = cy0 / my
    cpos = np.array([cx, cy])

    resam = np.zeros((Ny, Nx), dtype = np.float)

    eps = 1e-12
    for k in range(Ny):

        for l in range(Nx):

            idy = (2*k + 1) * my / 2
            idx = (2*l + 1) * mx / 2

            sor_y = np.int(idy)
            sor_x = np.int(idx)

            xa = (idx - mx/2) - sor_x
            it = eps < np.abs(xa)
            xa = xa * it

            xb = (idx + mx/2) - (sor_x + 1)
            it = eps < np.abs(xb)
            xb = xb * it

            ya = (idy - my/2) - sor_y
            it = eps < np.abs(ya)
            ya = ya * it

            yb = (idy + my/2) - (sor_y + 1)
            it = eps < np.abs(yb)
            yb = yb * it

            abs_x0 = np.abs( np.float('%.12f' % xa) )
            abs_x1 = np.abs( np.float('%.12f' % xb) )
            abs_y0 = np.abs( np.float('%.12f' % ya) )
            abs_y1 = np.abs( np.float('%.12f' % yb) )

            if ( (xa >= 0) & (xb <= 0) & (ya >= 0) & (yb <= 0) ):
                resam[k,l] = mx * my * Mrr[sor_y, sor_x]

            if (xa < 0) & (xb < 0) & (ya >= 0) & (yb <= 0):
                resam[k,l] = my * (mx - abs_x0) * Mrr[sor_y, sor_x] + abs_x0 * my * Mrr[sor_y, sor_x - 1]

            elif (xa >= 0) & (xb <= 0) & (ya > 0) & (yb > 0):
                resam[k,l] = abs_y1 * mx * Mrr[sor_y + 1, sor_x] + (my - abs_y1) * mx * Mrr[sor_y, sor_x]

            elif (xa > 0) & (xb > 0) & (ya >= 0) & (yb <= 0):
                resam[k,l] = abs_x1 * my * Mrr[sor_y, sor_x + 1] + (mx - abs_x1) * my * Mrr[sor_y, sor_x]

            elif (xa >= 0) & (xb <= 0) & (ya < 0) & (yb < 0):
                resam[k,l] = mx * (my - abs_y0) * Mrr[sor_y, sor_x] + abs_y0 * mx * Mrr[sor_y - 1, sor_x]

            elif (xa > 0) & (xb > 0) & (ya > 0) & (yb > 0):
                resam[k,l] = (mx - abs_x1) * (my - abs_y1) * Mrr[sor_y, sor_x] + abs_x1 * (my - abs_y1) * Mrr[sor_y, sor_x + 1] \
                            + abs_y1 * (mx - abs_x1) * Mrr[sor_y + 1, sor_x] + abs_x1 * abs_y1 * Mrr[sor_y + 1, sor_x + 1]

            elif (xa < 0) & (xb < 0) & (ya < 0) & (yb < 0):
                resam[k,l] = (mx - abs_x0) * (my - abs_y0) * Mrr[sor_y, sor_x] + abs_x0 * (my - abs_y0) *Mrr[sor_y, sor_x - 1] \
                            + abs_y0 * (mx - abs_x0) * Mrr[sor_y - 1, sor_x] + abs_x0 * abs_y0 * Mrr[sor_y - 1, sor_x -1]

            elif (xa > 0) & (xb > 0) & (ya < 0) & (yb < 0):
                resam[k,l] = (mx - abs_x1) * (my - abs_y0) * Mrr[sor_y, sor_x] + (mx - abs_x1) * abs_y0 * Mrr[sor_y - 1, sor_x] \
                            + abs_x1 * (my - abs_y0) * Mrr[sor_y, sor_x + 1] + abs_x1 * abs_y0 * Mrr[sor_y - 1, sor_x + 1]

            elif (xa < 0) & (xb < 0) & (ya > 0) & (yb > 0):
                resam[k,l] = (mx - abs_x0) * (my - abs_y1) * Mrr[sor_y, sor_x] + abs_x0 * (my - abs_y1) * Mrr[sor_y, sor_x - 1] \
                            + abs_y1 * (mx - abs_x0) * Mrr[sor_y + 1, sor_x] + abs_x0 * abs_y1 * Mrr[sor_y + 1, sor_x - 1]
            else:
                continue

    return resam, cx, cy

def main():
    cx0 = 2
    cy0 = 2
    import astropy.io.fits as fits

    A = np.random.random((100, 100))
    A = np.ones((100, 100), dtype = np.float)
    data = fits.open('sub_img.fits')
    A = data[0].data

    m1 = 1.05
    m2 = 0.95
    t0 = time.time()
    #resam = sum_samp(m1, m1, A, cx0, cy0)[0]
    resam = down_samp(m2, m2, A, cx0, cy0)[0]
    t1 = time.time() - t0
    print('t = ', t1)

    fig = plt.figure()
    ax = plt.subplot(121)
    bx = plt.subplot(122)
    ax.set_title('origin')
    bx.set_title('resample')

    ax.imshow(A, cmap = 'Greys', vmin = 1e-4, vmax = 1e2, origin = 'lower', norm = mpl.colors.LogNorm())
    bx.imshow(resam, cmap = 'Greys', vmin = 1e-4, vmax = 1e2, origin = 'lower', norm = mpl.colors.LogNorm())
    '''
    ax.imshow(A, origin = 'lower', norm = mpl.colors.SymLogNorm(linthresh=0.01))
    bx.imshow(resam[:-1,:-1], origin = 'lower', norm = mpl.colors.SymLogNorm(linthresh=0.01))
    '''
    plt.tight_layout()
    plt.savefig('test.png', dpi = 300)
    plt.show()

    raise

if __name__ == "__main__":
    main()
