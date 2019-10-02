import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time

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

            sor_y = np.int(k // (1/my))
            sor_x = np.int(l // (1/mx))

            xa = (idx - mx/2) - sor_x
            ## set "0" if the miuns resulte is small then eps
            # avoid failure of condition identifying caused by calculation accuracy
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

            if (xa >= 0) & (xb <= 0) & (ya >= 0) & (yb <= 0):
                resam[k, l] = mx * my * Mrr[sor_y, sor_x]

            elif (xa >= 0) & (xb <= 0) & (ya > 0) & (yb > 0):
                resam[k, l] = yb * mx * Mrr[sor_y + 1, sor_x] + (my - yb) * mx * Mrr[sor_y, sor_x]

            elif (xa > 0) & (xb > 0) & (ya >= 0) & (yb <= 0):
                resam[k, l] = xb * my * Mrr[sor_y, sor_x + 1] + (mx - xb) * my * Mrr[sor_y, sor_x]

            elif (xa > 0) & (xb > 0) & (ya > 0) & (yb > 0):
                resam[k, l] = (mx - xb) * (my - yb) * Mrr[sor_y, sor_x] + xb * (my - yb) * Mrr[sor_y, sor_x + 1] \
                            + yb * (mx - xb) * Mrr[sor_y + 1, sor_x] + xb * yb * Mrr[sor_y + 1, sor_x + 1]
            else:
                continue

    return resam, cx, cy

def sum_samp(m1, m2, data, cx0, cy0):
    mx = m1
    my = m2

    Mrr = np.zeros((data.shape[0] + 1, data.shape[1] + 1), dtype = np.float)
    Mrr[:-1, :-1] = data * 1

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

            sor_y = np.int(k // my)
            sor_x = np.int(l // mx)

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

            if (xa >= 0) & (xb <= 0) & (ya >= 0) & (yb <= 0):
                resam[sor_y, sor_x] += Mrr[k, l]

            elif (xa >= 0) & (xb <= 0) & (ya > 0) & (yb > 0):
                resam[sor_y, sor_x] += ( (in_my - yb) * in_mx / S_pix) * Mrr[k, l]
                resam[sor_y + 1, sor_x] += ( (yb * in_mx) / S_pix) * Mrr[k, l]

            elif (xa > 0) & (xb > 0) & (ya >= 0) & (yb <= 0):
                resam[sor_y, sor_x] += ( (in_mx - xb) * in_my / S_pix) * Mrr[k, l]
                resam[sor_y, sor_x + 1] += ( (xb * in_my) / S_pix) * Mrr[k, l]

            elif (xa > 0) & (xb > 0) & (ya > 0) & (yb > 0):
                resam[sor_y, sor_x] += ( (in_mx - xb) * (in_my - yb) / S_pix) * Mrr[k, l]
                resam[sor_y + 1, sor_x] += ( (in_mx - xb) * yb / S_pix) * Mrr[k, l]
                resam[sor_y, sor_x + 1] += ( xb * (in_my - yb) / S_pix) * Mrr[k, l]
                resam[sor_y + 1, sor_x + 1] += ( xb * yb / S_pix) * Mrr[k, l]

            else:
                continue

    return resam, cx, cy

def main():
    import skimage
    cx0 = 2
    cy0 = 2

    #A = np.random.random((100, 100))
    #A = np.ones((100, 100), dtype = np.float)

    x = np.linspace(0, 5, 101)
    y = np.linspace(0, 5, 101)
    grd = np.array(np.meshgrid(x,y))
    imgt = np.exp(-((grd[0,:] - 2.5)**2 + (grd[1,:] - 2.5)**2)/(2*0.5**2)) / (2*np.pi*0.5**2)
    dx = np.random.normal(0, 0.15, 101**2).reshape(101, 101)
    #A = imgt
    A = imgt + dx

    import astropy.io.fits as fits
    data = fits.open('sub_img.fits')
    A = data[0].data

    m = 0.95
    t0 = time.time()
    if m > 1:
        resam = sum_samp(m, m, A, cx0, cy0)[0]
    else:
        resam = down_samp(m, m, A, cx0, cy0)[0]
    t1 = time.time() - t0
    print('t = ', t1)

    fig = plt.figure()
    #fig.suptitle('scale %.3f [gaussian only]' % m )
    #fig.suptitle('scale %.3f [add noise]' % m )
    ax = plt.subplot(121)
    bx = plt.subplot(122)
    ax.set_title('origin')
    bx.set_title('resample')

    ax.imshow(A, cmap = 'Greys', vmin = 1e-3, vmax = 1e1, origin = 'lower', norm = mpl.colors.LogNorm())
    bx.imshow(resam, cmap = 'Greys', vmin = 1e-3, vmax = 1e1, origin = 'lower', norm = mpl.colors.LogNorm())

    plt.tight_layout()
    #plt.savefig('test_%.3f_ideal.png' % m, dpi = 300)
    plt.savefig('test_%.3f_noise.png' % m, dpi = 300)
    plt.show()

    '''
    t0 = time.time()
    resam = down_samp(m2, m2, A, cx0, cy0)[0]
    t1 = time.time() - t0

    x_arr = np.linspace(0,9,10)
    y_arr = np.ones(10, dtype = np.float) * 20
    flux_res0 = resam[20, :10]
    flux_res1 = resam[19, :10]
    flux_res2 = resam[21, :10]

    flux_set0 = np.array([-0.010080413818359375, 0.021439361572265624, -0.019343261718749998, 0.01169525146484375, -0.030339813232421874,
                        0.005679473876953124, 0.01018768310546875, -0.0018192291259765624, 0.0029412078857421887, 0.021461105346679686]) # center of line
    flux_set1 = np.array([0.008923645019531249, -0.045140075683593746, 0.011443023681640624, 0.010439910888671876, -0.013578262329101563,
                        -0.009075126647949218, -0.008577919006347657, 0.016951446533203123, 0.018959121704101564, -0.02608528137207031])
    flux_set2 = np.array([0.030467376708984372, -0.006287384033203126, -0.022661132812500002, -0.03761138916015625, -0.021346130371093747,
                        -0.023704833984374996, 0.03153961181640626, 0.003881301879882814, -0.022187271118164063, -0.02216011047363281])

    fig = plt.figure()
    ax = plt.subplot(121)
    bx = plt.subplot(122)
    ax.set_title('origin')
    ax.imshow(A, cmap = 'Greys', vmin = 1e-5, vmax = 1e2, origin = 'lower', norm = mpl.colors.LogNorm())
    bx.set_title('resample')
    bx.imshow(resam, cmap = 'Greys', vmin = 1e-5, vmax = 1e2, origin = 'lower', norm = mpl.colors.LogNorm())
    plt.tight_layout()
    plt.savefig('test.png', dpi = 300)
    plt.show()

    plt.figure(figsize = (16,8))
    gs = gridspec.GridSpec(1, 2)

    ax = plt.subplot(gs[0])
    gs0 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec = gs[1])
    bx0 = plt.subplot(gs0[:4])
    bx1 = plt.subplot(gs0[-1])

    ax.set_title('resample with scale %.3f' % m2)
    ax.imshow(resam, cmap = 'Greys', vmin = 1e-5, vmax = 1e2, origin = 'lower', norm = mpl.colors.LogNorm())
    ax.scatter(x_arr, y_arr, s = 5, marker = 'o', facecolors = 'r', edgecolors = 'r', label = 'pixel on the lattice', alpha = 0.5)
    ax.scatter(x_arr, y_arr - 1, s = 5, marker = 's', facecolors = 'g', edgecolors = 'g', label = 'pixel below the lattice', alpha = 0.5)
    ax.scatter(x_arr, y_arr + 1, s = 5, marker = '^', facecolors = 'b', edgecolors = 'b', label = 'pixel above the lattice', alpha = 0.5)
    ax.legend(loc = 1)

    bx0.set_title('Lattice pixel Flux check')
    bx0.plot(flux_res0, flux_set0, 'ro', label = 'pixel on the lattice', alpha = 0.5)
    bx0.plot(flux_res1, flux_set1, 'gs', label = 'pixel below the lattice', alpha = 0.5)
    bx0.plot(flux_res2, flux_set2, 'b^', label = 'pixel above the lattice', alpha = 0.5)
    xtik = bx0.get_xticks()
    bx0.plot(xtik[1:-1], xtik[1:-1], 'k--', label = 'slope = 1', alpha = 0.5)
    bx0.set_xlabel('$Flux \; on \; resample \; image \,[F_{c}]$')
    bx0.set_ylabel('$Flux \; from \; theory \, [F_{T}]$')
    bx0.legend(loc = 4)
    bx0.tick_params(axis = 'both', which = 'both', direction = 'in')

    bx1.plot(flux_res0, (flux_res0 - flux_set0) / flux_set0, 'ro', alpha = 0.5)
    bx1.plot(flux_res1, (flux_res1 - flux_set1) / flux_set1, 'gs', alpha = 0.5)
    bx1.plot(flux_res2, (flux_res2 - flux_set2) / flux_set2, 'b^', alpha = 0.5)
    bx1.axhline(y = 0., ls = '--', color = 'k', alpha = 0.5)
    bx1.set_xlabel('$Flux \; on \; resample \; image \,[F_{c}]$')
    bx1.set_ylabel('$F_{c} - F_{T} / F_{T}$')    
    bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

    plt.tight_layout()
    plt.savefig('points_check.png', dpi = 300)
    plt.close()
    '''
    raise

if __name__ == "__main__":
    main()
