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
m3 = 0.5
m4 = 0.5 # for down-resamp

def sum_samp(m1, m2, data, cx0, cy0):
    mx = m1
    my = m2
    a = data
    N0x = a.shape[1]
    N0y = a.shape[0]
    x0 = np.linspace(0,N0x-1,N0x)
    y0 = np.linspace(0,N0y-1,N0y)
    M = np.meshgrid(x0,y0)
    N2 = np.int(N0y*(1/my))
    N1 = np.int(N0x*(1/mx))

    cx = cx0/mx
    cy = cy0/my
    cpos = np.array([cx, cy])
    res_cont = np.zeros((N2 + 1, N1 + 1), dtype = np.float)
    resamp = np.zeros((N2 + 1, N1 + 1), dtype = np.float)
    for k in range(N0y):
        for l in range(N0x):
            at = a[k, l] == 0
            idy = (2 * k + 1) * 1/2
            idx = (2 * l + 1) * 1/2

            x0 = idx - 0.5
            x1 = idx + 0.5
            y0 = idy - 0.5
            y1 = idy + 0.5

            a0 = np.int(x0 // mx)
            a1 = np.int(x1 // mx)
            b0 = np.int(y0 // my)
            b1 = np.int(y1 // my)
            px = np.int(idx // mx)
            py = np.int(idy // my)

            ddx0 = x0 % mx
            ddx1 = x1 % mx
            ddy0 = y0 % my
            ddy1 = y1 % my

            print('ddx0', ddx0)
            print('ddx1', ddx1)
            print('ddy0', ddy0)
            print('ddy1', ddy1)

            if (ddx1 == 0) | (ddx0 == 0):
                a0 = px
                a1 = px
            if (ddy1 == 0) | (ddy0 == 0):
                b0 = py
                b1 = py
            print('a0 = ', a0)
            print('a1 = ', a1)
            print('px = ', px)
            print('py = ', py)

            if (a0 == a1) & (b0 == b1):
                res_cont[b0, a0] = res_cont[b0, a0] + at
                if res_cont[b0, a0] == 0:
                    resamp[b0, a0] = resamp[b0, a0] + a[k, l]
                else:
                    resamp[b0, a0] = 0

            elif (a0 < px) & (a1 == px) & (b0 == b1):
                res_cont[b0, a0] = res_cont[b0, a0] + at
                res_cont[b0, a1] = res_cont[b0, a1] + at
                if res_cont[b0, a0] == 0:
                    resamp[b0 ,a0] = resamp[b0, a0] + np.abs(px * mx - x0) * a[k, l]
                else:
                    resamp[b0, a0] = 0

                if res_cont[b0, a1] == 0:
                    resamp[b0, a1] = resamp[b0, a1] + np.abs(x1 - px * mx) * a[k, l]
                else:
                    resamp[b0, a1] = 0 

            elif (a0 == px) & (a1 > px) & (b0 == b1):
                res_cont[b0, a0] = res_cont[b0, a0] + at
                res_cont[b0, a1] = res_cont[b0, a1] + at
                if res_cont[b0, a0] == 0:
                    resamp[b0, a0] = resamp[b0, a0] + np.abs(px * mx - x0) * a[k, l]
                else:
                    resamp[b0, a0] = 0

                if res_cont[b0, a1] == 0:
                    resamp[b0, a1] = resamp[b0, a1] + np.abs(x1 - px * mx) * a[k, l]
                else:
                    resamp[b0, a1] = 0

            elif (a0 == a1) & (b0 == py) & (b1 > py):
                res_cont[b0, a0] = res_cont[b0, a0] + at
                res_cont[b1, a0] = res_cont[b1, a0] + at
                if res_cont[b0, a0] == 0:
                    resamp[b0, a0] = resamp[b0, a0] + np.abs(py * my - y0) * a[k, l]
                else:
                    resamp[b0, a0] = 0

                if res_cont[b1, a0] == 0:
                    resamp[b1, a0] = resamp[b1, a0] + np.abs(y1 - py * my) * a[k, l]
                else:
                    resamp[b1, a0] = 0

            elif (a0 == a1) & (b0 < py) & (b1 == py):
                res_cont[b0, a0] = res_cont[b0, a0] + at
                res_cont[b1, a0] = res_cont[b1, a0] + at
                if res_cont[b0, a0] == 0:
                    resamp[b0, a0] = resamp[b0, a0] + np.abs(py * my - y0) * a[k, l]
                else:
                    resamp[b0, a0] = 0

                if res_cont[b1, a0] == 0:
                    resamp[b1, a0] = resamp[b1, a0] + np.abs(y1 - py * my) * a[k, l]
                else:
                    resamp[b1, a0] = 0

            elif (a0 < px) & (a1 == px) & (b0 < py) & (b1 == py):
                res_cont[b0, a0] = res_cont[b0, a0] + at
                res_cont[b0, a1] = res_cont[b0, a1] + at
                res_cont[b1, a0] = res_cont[b1, a0] + at
                res_cont[b1, a1] = res_cont[b1, a1] + at
                if res_cont[b0, a0] == 0:
                    resamp[b0, a0] = resamp[b0, a0] + np.abs(py * my - y0) * np.abs(px * mx - x0) * a[k, l]
                else:
                    resamp[b0, a0] = 0

                if res_cont[b0, a1] == 0:
                    resamp[b0, a1] = resamp[b0, a1] + np.abs(py * my - y0) * np.abs(x1 - px * mx) * a[k, l]
                else:
                    resamp[b0, a1] = 0

                if res_cont[b1, a0] == 0:
                    resamp[b1, a0] = resamp[b1, a0] + np.abs(y1 - py * my) * np.abs(px * mx - x0) * a[k, l]
                else:
                    resamp[b1, a0] = 0

                if res_cont[b1, a1] == 0:
                    resamp[b1, a1] = resamp[b1, a1] + np.abs(y1 - py * my) * np.abs(x1 - px * mx) * a[k, l]
                else:
                    resamp[b1, a1] = 0

            elif (a0 < px) & (a1 == px) & (b0 == py) & (b1 > py):
                res_cont[b0, a0] = res_cont[b0, a0] + at
                res_cont[b0, a1] = res_cont[b0, a1] + at
                res_cont[b1, a0] = res_cont[b1, a0] + at
                res_cont[b1, a1] = res_cont[b1, a1] + at
                if res_cont[b0, a0] == 0:
                    resamp[b0, a0] = resamp[b0, a0] + np.abs(py * my - y0) * np.abs(px * mx - x0) * a[k, l]
                else:
                    resamp[b0, a0] = 0

                if res_cont[b0, a1] == 0:
                    resamp[b0, a1] = resamp[b0, a1] + np.abs(py * my - y0) * np.abs(x1 - px * mx) * a[k, l]
                else:
                    resamp[b0, a1] = 0

                if res_cont[b1, a0] == 0:
                    resamp[b1, a0] = resamp[b1, a0] + np.abs(y1 - py * my) * np.abs(px * mx - x0) * a[k, l]
                else:
                    resamp[b1, a0] = 0

                if res_cont[b1, a1] == 0:
                    resamp[b1, a1] = resamp[b1, a1] + np.abs(y1 - py * my) * np.abs(x1 - px * mx) * a[k, l]
                else:
                    resamp[b1, a1] = 0

            elif (a0 == px) & (a1 > px) & (b0 < py) & (b1 == py):
                res_cont[b0, a0] = res_cont[b0, a0] + at
                res_cont[b0, a1] = res_cont[b0, a1] + at
                res_cont[b1, a0] = res_cont[b1, a0] + at
                res_cont[b1, a1] = res_cont[b1, a1] + at
                if res_cont[b0, a0] == 0:
                    resamp[b0, a0] = resamp[b0, a0] + np.abs(py * my - y0) * np.abs(px * mx - x0) * a[k, l]
                else:
                    resamp[b0, a0] = 0

                if res_cont[b0, a1] == 0:
                    resamp[b0, a1] = resamp[b0, a1] + np.abs(py * my - y0) * np.abs(x1 - px * mx) * a[k, l]
                else:
                    resamp[b0, a1] = 0

                if res_cont[b1, a0] == 0:
                    resamp[b1, a0] = resamp[b1, a0] + np.abs(y1 - py * my) * np.abs(px * mx - x0) * a[k, l]
                else:
                    resamp[b1, a0] = 0

                if res_cont[b1, a1] == 0:
                    resamp[b1, a1] = resamp[b1, a1] + np.abs(y1 - py * my) * np.abs(x1 - px * mx) * a[k, l]
                else:
                    resamp[b1, a1] = 0

            elif (a0 == px) & (a1 > px) & (b0 == py) & (b1 > py):
                res_cont[b0, a0] = res_cont[b0, a0] + at
                res_cont[b0, a1] = res_cont[b0, a1] + at
                res_cont[b1, a0] = res_cont[b1, a0] + at
                res_cont[b1, a1] = res_cont[b1, a1] + at
                if res_cont[b0, a0] == 0:
                    resamp[b0, a0] = resamp[b0, a0] + np.abs(py * my - y0) * np.abs(px * mx - x0) * a[k, l]
                else:
                    resamp[b0, a0] = 0

                if res_cont[b0, a1] == 0:
                    resamp[b0, a1] = resamp[b0, a1] + np.abs(py * my - y0) * np.abs(x1 - px * mx) * a[k, l]
                else:
                    resamp[b0, a1] = 0

                if res_cont[b1, a0] == 0:
                    resamp[b1, a0] = resamp[b1, a0] + np.abs(y1 - py * my) * np.abs(px * mx - x0) * a[k, l]
                else:
                    resamp[b1, a0] = 0

                if res_cont[b1, a1] == 0:
                    resamp[b1, a1] = resamp[b1, a1] + np.abs(y1 - py * my) * np.abs(x1 - px * mx) * a[k, l]
                else:
                    resamp[b1, a1] = 0

            else:
                continue
    print('cont = ', res_cont)
    raise
    return resamp, cpos

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
    for k in range(Ny):
        for l in range(Nx):

            idy = (2*k + 1) * my/2
            idx = (2*l + 1) * mx/2
            sor_y = np.int(k // (1/my))
            sor_x = np.int(l // (1/mx))

            xa = (idx - mx/2) - sor_x
            xb = (idx + mx/2) - (sor_x + 1)
            ya = (idy - my/2) - sor_y
            yb = (idy + my/2) - (sor_y + 1)

            if (xa >= 0) & (xb <= 0) & (ya >= 0) & (yb <= 0):
                at = a[sor_y, sor_x] != 0
                resamp[k,l] = at * (mx * my * a[sor_y, sor_x])

            elif (xa >= 0) & (xb <= 0) & (ya >= 0) & (yb > 0):
                at = (a[sor_y, sor_x] != 0) & (a[sor_y + 1, sor_x] != 0)
                resamp[k,l] = at * (yb * mx * a[sor_y + 1, sor_x] + (my - yb) * mx * a[sor_y, sor_x])

            elif (xa >= 0) & (xb > 0) & (ya >= 0) & (yb <= 0):
                at = (a[sor_y, sor_x] != 0) & (a[sor_y, sor_x + 1] != 0)
                resamp[k,l] = at * (xb * my * a[sor_y, sor_x + 1] + (mx - xb) * my * a[sor_y, sor_x])

            elif (xa >= 0) & (xb > 0) & (ya >= 0) & (yb > 0):
                at = (a[sor_y, sor_x] != 0) & (a[sor_y, sor_x + 1] != 0) & (a[sor_y + 1, sor_x] != 0) & (a[sor_y + 1, sor_x + 1] != 0)
                resamp[k,l] = at * ((mx - xb) * (my - yb) * a[sor_y, sor_x] + xb * (my - yb) * a[sor_y, sor_x + 1]
                                    + yb * (mx - xb) * a[sor_y + 1, sor_x] + xb * yb * a[sor_y + 1, sor_x + 1])
            else:
                continue

    return resamp, cpos

def main():
    resam1 = sum_samp(m1, m2, a, cx0, cy0)[0]
    #resam2 = down_samp(m3, m4, a, cx0, cy0)[0]
    print('before=', a)
    print('*'*10)

    print('bigger', resam1)
    #print('smaller', resam2)

    plt.figure()
    ax = plt.subplot(121)
    bx = plt.subplot(122)
    ax.imshow(a)
    bx.imshow(resam1)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
