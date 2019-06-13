import numpy as np


def newton_polynomial(raw):
    I = np.array(raw, dtype=np.double)

    (m, n) = I.shape

    R = np.zeros((m, n, 4))

    # Labeling different polatiration channels
    O = np.zeros((m, n), dtype=int)

    step = 1
    O[0::2, 0::2] = 0
    O[0::2, 1::2] = 1
    O[1::2, 1::2] = 2
    O[1::2, 0::2] = 3

    # Store intermediate results
    Y1 = np.array(raw, dtype=np.double)
    Y2 = np.array(raw, dtype=np.double)

    # for index in range(R.shape[2]):
    R[:, :, 0] = np.array(raw, dtype=np.double)
    R[:, :, 1] = np.array(raw, dtype=np.double)
    R[:, :, 2] = np.array(raw, dtype=np.double)
    R[:, :, 3] = np.array(raw, dtype=np.double)

    '''
    % Stage one interpolation: interpolate vertically for case Fig.6(b),
    % interpolate horizontally for case Fig.6(c), interpolate in diagonal
    % directions for case Fig.6(a). The Eqs.(14)-(17) are simplified in this
    % code.
    '''

    for i in range(3, m - 3):
        for j in range(3, n - 3):
            R[i, j, O[i, j]] = I[i, j]
            R[i, j, O[i, j + 1]] = 0.5 * I[i, j] + 0.0625 * I[i, j - 3] - 0.25 * I[i, j - 2] + \
                0.4375 * I[i, j - 1] + 0.4375 * I[i, j + 1] - \
                0.25 * I[i, j + 2] + 0.0625 * I[i, j + 3]
            R[i, j, O[i + 1, j]] = 0.5 * I[i, j] + 0.0625 * I[i - 3, j] - 0.25 * I[i - 2, j] + \
                0.4375*I[i-1, j] + 0.4375*I[i+1, j] - \
                0.25*I[i+2, j] + 0.0625*I[i+3, j]
            Y1[i, j] = 0.5*I[i, j] + 0.0625*I[i-3, j-3] - 0.25*I[i-2, j-2] + 0.4375 * \
                I[i-1, j-1] + 0.4375*I[i+1, j+1] - \
                0.25*I[i+2, j+2] + 0.0625*I[i+3, j+3]
            Y2[i, j] = 0.5*I[i, j] + 0.0625*I[i-3, j+3] - 0.25*I[i-2, j+2] + 0.4375 * \
                I[i-1, j+1] + 0.4375*I[i+1, j-1] - \
                0.25*I[i+2, j-2] + 0.0625*I[i+3, j-3]
    # One can adjust for better result
    thao = 2.8
    # Fusion of the estimations with edge classifier for case Fig.6(a).

    for i in range(3, m-3):
        for j in range(3, n-3):
            pha1 = 0.0
            pha2 = 0.0

            for k in range(-2, 3, 2):
                for l in range(-2, 3, 2):
                    pha1 = pha1 + abs(Y1[i+k, j+l] - I[i+k, j+l])
                    pha2 = pha2 + abs(Y2[i+k, j+l] - I[i+k, j+l])

            if (pha1 / pha2) > thao:
                R[i, j, O[i+1, j+1]] = Y2[i, j]
            elif (pha2/pha1) > thao:
                R[i, j, O[i+1, j+1]] = Y1[i, j]
            elif (((pha1/pha2) < thao) and ((pha2/pha1) < thao)):
                d1 = abs(I[i-1, j-1] - I[i+1, j+1]) + \
                    abs(2*I[i, j] - I[i-2, j-2] - I[i+2, j+2])
                d2 = abs(I[i+1, j-1] - I[i-1, j+1]) + \
                    abs(2*I[i, j] - I[i+2, j-2] - I[i-2, j+2])
                epsl = 0.000000000000001
                w1 = 1/(d1 + epsl)
                w2 = 1/(d2+epsl)
                R[i, j, O[i+1, j+1]] = (w1*Y1[i, j] + w2*Y2[i, j])/(w1 + w2)

    RR = np.array(R,dtype=np.double)

    XX1 = np.array(raw,dtype=np.double)
    XX2 = np.array(raw,dtype=np.double)
    YY1 = np.array(raw,dtype=np.double)
    YY2 = np.array(raw,dtype=np.double)

    # Stage two interpolation: interpolate horizontally for case Fig.6(b),
    # interpolate vertically for case Fig.6(c).

    for i in range(3, m-3):
        for j in range(3, n-3):
            XX1[i, j] = R[i, j, O[i, j+1]]
            XX2[i, j] = 0.5*I[i, j] + 0.0625 * \
                R[i-3, j, O[i, j+1]] - 0.25*I[i-2, j]
            XX2[i, j] = XX2[i, j] + 0.4375 * \
                R[i-1, j, O[i, j+1]] + 0.4375*R[i+1, j, O[i, j+1]]
            XX2[i, j] = XX2[i, j] - 0.25*I[i+2, j] + 0.0625*R[i+3, j, O[i, j+1]]
            YY1[i, j] = R[i, j, O[i+1, j]]
            YY2[i, j] = 0.5*I[i, j] + 0.0625 * \
                R[i, j-3, O[i+1, j]] - 0.25*I[i, j-2]
            YY2[i, j] = YY2[i, j] + 0.4375 * \
                R[i, j-1, O[i+1, j]] + 0.4375*R[i, j+1, O[i+1, j]]
            YY2[i, j] = YY2[i, j] - 0.25*I[i, j+2] + 0.0625*R[i, j+3, O[i+1, j]]

    # Fusion of the estimations with edge classifier for case Fig.6(b) and Fig.6(c).

    for i in range(3, m-4):
        for j in range(3, n-4):
            pha1 = 0.0
            pha2 = 0.0

            for k in range(-2, 3, 2):
                for l in range(-2, 3, 2):
                    pha1 = pha1 + abs(XX1[i+k, j+l] - I[i+k, j+l])
                    pha2 = pha2 + abs(XX2[i+k, j+l] - I[i+k, j+l])

            if (pha1 / pha2) > thao:
                R[i, j, O[i, j+1]] = XX2[i, j]
            elif (pha2/pha1) > thao:
                R[i, j, O[i, j+1]] = XX1[i, j]
            elif (((pha1/pha2) < thao) and ((pha2/pha1) < thao)):
                d1 = abs(I[i, j-1] - I[i, j+1]) + \
                    abs(2*I[i, j] - I[i, j-2] - I[i, j+2])
                d2 = abs(I[i+1, j] - I[i-1, j]) + \
                    abs(2 * I[i, j] - I[i + 2, j] - I[i - 2, j])
                epsl = 0.000000000000001
                w1 = 1 / (d1 + epsl)
                w2 = 1 / (d2 + epsl)
                R[i, j, O[i, j + 1]] = (w1 * XX1[i, j] + w2 * XX2[i, j]) / (w1 + w2)

            pha1 = 0.0
            pha2 = 0.0

            for k in range(-2, 3, 2):
                for l in range(-2, 3, 2):
                    pha1 = pha1 + abs(YY1[i + k, j + l] - I[i + k, j + l])
                    pha2 = pha2 + abs(YY2[i + k, j + l] - I[i + k, j + l])

            if (pha1 / pha2) > thao:
                R[i, j, O[i + 1, j]] = YY2[i, j]
            elif (pha2 / pha1) > thao:
                R[i, j, O[i + 1, j]] = YY1[i, j]
            elif (((pha1 / pha2) < thao) and ((pha2 / pha1) < thao)):
                d1 = abs(I[i, j - 1] - I[i, j + 1]) + \
                    abs(2 * I[i, j] - I[i, j - 2] - I[i, j + 2])
                d2 = abs(I[i + 1, j] - I[i - 1, j]) + \
                    abs(2 * I[i, j] - I[i + 2, j] - I[i - 2, j])
                epsl = 0.000000000000001
                w1 = 1 / (d1 + epsl)
                w2 = 1 / (d2 + epsl)
                R[i, j, O[i, j + 1]] = (w1 * YY1[i, j] + w2 * YY2[i, j]) / (w1 + w2)

    R = np.array(RR, dtype=np.double)
    I0 = np.array(R[:, :, 0], dtype=np.double)
    I45 = np.array(R[:, :, 1], dtype=np.double)
    I90 = np.array(R[:, :, 2], dtype=np.double)
    I135 = np.array(R[:, :, 3], dtype=np.double)

    return np.dstack((I0, I45, I90, I135))
