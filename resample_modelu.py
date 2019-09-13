import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

a = np.ones((5,5), dtype = np.float)
a[1,2] = 0
a[3,3] = 0
 
cx0 = 2
cy0 = 2
m1 = 1.2
m2 = 1.2 # for up-resamp
m3 = 0.8
m4 = 0.8 # for down-resamp

def sum_samp(m1, m2, data, cx0, cy0):
    mx = m1
    my = m2
    a = data
    N0x = a.shape[1]
    N0y = a.shape[0]
    x0 = np.linspace(0, N0x-1, N0x)
    y0 = np.linspace(0, N0y-1, N0y)
    M = np.meshgrid(x0,y0)
    N2 = np.int(N0y * (1 / my))
    N1 = np.int(N0x * (1 / mx))

    cx = cx0 / mx
    cy = cy0 / my
    cpos = np.array([cx, cy])
    resamp = np.zeros((N2 + 1, N1 + 1), dtype = np.float)

    sig = N0y* N0x // 10
    for k in range(N0y):
        for l in range(N0x):

            if (k * N0x + l) % sig == 0:
                print( "%3.0f%%"%( (k *N0x + l) / (N0y * N0x) * 100 ) )

            in_mx = 1 / mx
            in_my = 1 / my

            idy = (2*k + 1) * in_my /2
            idx = (2*l + 1) * in_mx /2
            sor_y = np.int(k // my)
            sor_x = np.int(l // mx)

            xa = (idx - in_mx /2) - sor_x
            xb = (idx + in_mx /2) - (sor_x + 1)
            ya = (idy - in_my /2) - sor_y
            yb = (idy + in_my /2) - (sor_y + 1)

            S_pix = in_my * in_mx
            if (xa >= 0) & (xb <= 0) & (ya >= 0) & (yb <= 0):
                resamp[sor_y, sor_x] += a[k, l]

            elif (xa >= 0) & (xb <= 0) & (ya >= 0) & (yb > 0):
                resamp[sor_y, sor_x] += ( (in_my - yb) * in_mx / S_pix) * a[k, l]
                resamp[sor_y + 1, sor_x] += ( (yb * in_mx) / S_pix) * a[k, l]

            elif (xa >= 0) & (xb > 0) & (ya >= 0) & (yb <= 0):
                resamp[sor_y, sor_x] += ( (in_mx - xb) * in_my / S_pix) * a[k, l]
                resamp[sor_y, sor_x + 1] += ( (xb * in_my) / S_pix) * a[k, l]

            elif (xa >= 0) & (xb > 0) & (ya >= 0) & (yb > 0):
                resamp[sor_y, sor_x] += ( (in_mx - xb) * (in_my - yb) / S_pix) * a[k, l]
                resamp[sor_y + 1, sor_x] += ( (in_mx - xb) * yb / S_pix) * a[k, l]
                resamp[sor_y, sor_x + 1] += ( xb * (in_my - yb) / S_pix) * a[k, l]
                resamp[sor_y + 1, sor_x + 1] += ( xb * yb / S_pix) * a[k, l]

            else:
                continue

    return resamp, cx, cy

def down_samp(m3, m4, data, cx0, cy0):
    mx = m3
    my = m4
    a = data
    sNy = a.shape[0]
    sNx = a.shape[1]
    
    x0 = np.linspace(0,sNx-1,sNx)
    y0 = np.linspace(0,sNy-1,sNy)
    M0 = np.meshgrid(x0,y0)
    
    N1x = np.int(sNx*(1/mx))
    N1y = np.int(sNy*(1/my))
    cx = cx0/mx
    cy = cy0/my
    cpos = np.array([cx, cy])
    Ny = N1y
    Nx = N1x
    resamp = np.zeros((Ny, Nx), dtype = np.float)

    sig = Ny * Nx // 10
    for k in range(Ny):

        for l in range(Nx):

            if (k * Nx + l) % sig == 0:
                print( "%3.0f%%"%( (k *Nx + l) / (Ny * Nx) * 100 ) )

            idy = (2*k + 1) * my/2
            idx = (2*l + 1) * mx/2
            sor_y = np.int(k // (1/my))
            sor_x = np.int(l // (1/mx))

            xa = (idx - mx/2) - sor_x
            xb = (idx + mx/2) - (sor_x + 1)
            ya = (idy - my/2) - sor_y
            yb = (idy + my/2) - (sor_y + 1)

            if (xa >= 0) & (xb <= 0) & (ya >= 0) & (yb <= 0):
                resamp[k,l] = mx * my * a[sor_y, sor_x]

            elif (xa >= 0) & (xb <= 0) & (ya >= 0) & (yb > 0):
                resamp[k,l] = yb * mx * a[sor_y + 1, sor_x] + (my - yb) * mx * a[sor_y, sor_x]

            elif (xa >= 0) & (xb > 0) & (ya >= 0) & (yb <= 0):
                resamp[k,l] = xb * my * a[sor_y, sor_x + 1] + (mx - xb) * my * a[sor_y, sor_x]

            elif (xa >= 0) & (xb > 0) & (ya >= 0) & (yb > 0):
                resamp[k,l] = (mx - xb) * (my - yb) * a[sor_y, sor_x] + xb * (my - yb) * a[sor_y, sor_x + 1] \
                + yb * (mx - xb) * a[sor_y + 1, sor_x] + xb * yb * a[sor_y + 1, sor_x + 1]

            else:
                continue

    return resamp, cx, cy

def main():
    resam1 = sum_samp(m1, m2, a, cx0, cy0)[0]
    resam2 = down_samp(m3, m4, a, cx0, cy0)[0]
    print('before = ', a)
    print('*'*10)

    plt.figure()
    ax = plt.subplot(221)
    bx = plt.subplot(222)
    cx = plt.subplot(223)
    ax.imshow(a, origin = 'lower')
    bx.imshow(resam1, origin = 'lower')
    cx.imshow(resam2, origin = 'lower')
    plt.show()
    raise

if __name__ == "__main__":
    main()
