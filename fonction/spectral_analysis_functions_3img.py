#!/usr/bin/env python
# coding: utf-8

# # Méthode 3 images FFT2D3

# ## Packages

# In[3]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.fft import fftn
from scipy.fft import fftshift
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar


# In[4]:



def ULS(U, k, sig, imgtimes, FS):
#     """
#     ULS: least square estimate of current velocity from Fourier transform of images

#     Parameters:
#     - U : value of current to be tested
#     - k : wavenumber
#     - sig: radian frequency
#     - imgtimes: vector of times of the image acquisition
#     - FS : vector of observed complex amplitudes

#     Returns:
#     - Usum : value of the function that goes through 0 for U = true current
#     - Ac : complex amplitude of waves propagating in direction of k
#     - Bc : complex amplitude of waves propagating opposite to k
#     - eps2 : vector of squares for each image
    nt = len(imgtimes)
    imgtimes=np.array(imgtimes)

    RA1 = 1
    RB1 = 1
    RF1 = FS[0]
    RC1 = 1
    RF2 = FS[0]

    for i in range(1, nt):
        RA1 += np.exp(1j * (-2 * (sig - k * U) * imgtimes[i]))
        RB1 += np.exp(1j * (2 * (k * U) * imgtimes[i]))
        RF1 += FS[i] * np.exp(1j * (-(sig - k * U) * imgtimes[i]))
        RC1 += np.exp(1j * (2 * (sig + k * U) * imgtimes[i]))
        RF2 += FS[i] * np.exp(1j * ((sig + k * U) * imgtimes[i]))
        
        
#     print('taille imgtime',np.shape(imgtimes))
#     print('sig ',sig)
#     print(' k',k)
#     print(' U',U)
        
#     intermediate_result = -(sig - k * U) * imgtimes
#     print(f"Intermediate result: {intermediate_result}")

    RAs = np.exp(1j * (-(sig - k * U) * imgtimes))
    RBs = np.exp(1j * ((sig + k * U) * imgtimes))
    
    Bc = (RF2 - RF1 * RB1 / RA1) / (RC1 - RB1**2 / RA1)
    Ac = (RF1 - Bc * RB1) / RA1

    Usum = np.imag(np.sum(imgtimes[1:] * (RAs[1:] * Ac + RBs[1:] * Bc) * (RAs[1:] * Ac + RBs[1:] * Bc - FS[1:])))
    
    eps2 = np.sum(np.abs(Ac * RAs + Bc * RBs - FS)**2)
    
    return Usum, Ac, Bc, eps2


# def ULS(U, k, sig, imgtimes, FS):
# #     """
# #     ULS: least square estimate of current velocity from Fourier transform of images

# #     Parameters:
# #     - U : value of current to be tested
# #     - k : wavenumber
# #     - sig: radian frequency
# #     - imgtimes: vector of times of the image acquisition
# #     - FS : vector of observed complex amplitudes

# #     Returns:
# #     - eps2 : vector of squares for each image
# #     - Ac : complex amplitude of waves propagating in direction of k
# #     - Bc : complex amplitude of waves propagating opposite to k
# #     """
#     nt = len(imgtimes)

#     RA1 = 1
#     RB1 = 1
#     RF1 = FS[0]
#     RC1 = 1
#     RF2 = FS[0]

#     for i in range(1, nt):
#         RA1 += np.exp(1j * (-2 * (sig - k * U) * imgtimes[i]))
#         RB1 += np.exp(1j * (2 * (k * U) * imgtimes[i]))
#         RF1 += FS[i] * np.exp(1j * (-(sig - k * U) * imgtimes[i]))
#         RC1 += np.exp(1j * (2 * (sig + k * U) * imgtimes[i]))
#         RF2 += FS[i] * np.exp(1j * ((sig + k * U) * imgtimes[i]))

#     RAs = np.exp(1j * (-(sig - k * U) * imgtimes))
#     RBs = np.exp(1j * ((sig + k * U) * imgtimes))
    
#     Bc = (RF2 - RF1 * RB1 / RA1) / (RC1 - RB1**2 / RA1)
#     Ac = (RF1 - Bc * RB1) / RA1

#     eps2 = np.sum(np.abs(Ac * RAs + Bc * RBs - FS)**2)
    
#     return eps2, Ac, Bc

# def ULSmin(U, k, sig, imgtimes, FS):
#     nt = len(imgtimes)
#     RA1 = 1
#     RB1 = 1
#     RF1 = FS[0]
#     RC1 = 1
#     RF2 = FS[0]

#     for i in range(1, nt):
#         RA1 += np.exp(1j * (-2 * (sig - k * U) * imgtimes[i]))
#         RB1 += np.exp(1j * (2 * (k * U) * imgtimes[i]))
#         RF1 += FS[i] * np.exp(1j * (-(sig - k * U) * imgtimes[i]))
#         RC1 += np.exp(1j * (2 * (sig + k * U) * imgtimes[i]))
#         RF2 += FS[i] * np.exp(1j * ((sig + k * U) * imgtimes[i]))

#     RAs = np.exp(1j * (-(sig - k * U) * imgtimes))
#     RBs = np.exp(1j * ((sig + k * U) * imgtimes))
#     Bc = (RF2 - RF1 * RB1 / RA1) / (RC1 - RB1 ** 2 / RA1)
#     Ac = (RF1 - Bc * RB1) / RA1

#     eps2 = np.sum(np.abs(Ac * RAs + Bc * RBs - FS) ** 2)
#     return eps2, Ac, Bc

def ULSmin(U, k, sig, imgtimes, FS):
    nt = len(imgtimes)
    RA1 = 1
    RB1 = 1
    RF1 = FS[0]
    RC1 = 1
    RF2 = FS[0]

    for i in range(1, nt):
        RA1 += np.exp(1j * (-2 * (sig - k * U) * imgtimes[i]))
        RB1 += np.exp(1j * (2 * (k * U) * imgtimes[i]))
        RF1 += FS[i] * np.exp(1j * (-(sig - k * U) * imgtimes[i]))
        RC1 += np.exp(1j * (2 * (sig + k * U) * imgtimes[i]))
        RF2 += FS[i] * np.exp(1j * ((sig + k * U) * imgtimes[i]))

    # Vérification des types et des valeurs
#     print(f"sig: {sig}, type: {type(sig)}")
#     print(f"k: {k}, type: {type(k)}")
#     print(f"U: {U}, type: {type(U)}")
#     print('imgtimes',type(imgtimes))
    
#     assert np.isscalar(sig), "sig doit être un scalaire"
#     assert np.isscalar(k), "k doit être un scalaire"
#     assert np.isscalar(U), "U doit être un scalaire"
    imgtimes=np.array(imgtimes)

    # Vérification du calcul intermédiaire
    intermediate_result = -(sig - k * U) * imgtimes
#     print(f"Intermediate result: {intermediate_result}")
    
    RAs = np.exp(1j * (-(sig - k * U) * imgtimes))
    RBs = np.exp(1j * ((sig + k * U) * imgtimes))
    Bc = (RF2 - RF1 * RB1 / RA1) / (RC1 - RB1 ** 2 / RA1)
    Ac = (RF1 - Bc * RB1) / RA1

    eps2 = np.sum(np.abs(Ac * RAs + Bc * RBs - FS) ** 2)
    
#     print('eps2',type(eps2))
#     print('Ac',type(Ac))
#     print('Bc',type(Bc))
    
    return eps2, Ac, Bc



#-------------------------------------------------------------------------------------------------------------------------------#

def FFT2D3(array1, array2, array3, imgtimes, nxa, nya, dx, dy, n, Umin, Umax,Utol=2E-3):
    nx = int(np.floor(nxa / n))
    ny = int(np.floor(nya / n))

    print('nx',nxa,n,nx)
    print('ny',nya,n,ny)
    print('npixel',ny*nx)
    
    Umid = 0.5 * (Umin + Umax)
    
    # define windows
    # 1D windows
    
    hammingx = np.transpose(0.54 - 0.46 * np.cos(2 * np.pi * np.linspace(0, nx-1, nx) / (nx-1)))
    hanningx = np.transpose(0.5 * (1 - np.cos(2 * np.pi * np.linspace(0, nx-1, nx) / (nx-1))))
    hanningy = np.transpose(0.5 * (1 - np.cos(2 * np.pi * np.linspace(0, ny-1, ny) / (ny-1))))
    hanningxy = np.outer(hanningx, hanningy)
    
    # Calculate window correction factors
    wc2x = 1 / np.mean(hanningx**2)
    wc2y = 1 / np.mean(hanningy**2)

    dkx = 2 * np.pi / (dx * nx)
    dky = 2 * np.pi / (dy * ny)
    kx = np.linspace(0, (nx-1) * dkx, nx)
    ky = np.linspace(0, (ny-1) * dky, ny)
    
    shx = np.floor(nx / 2).astype(int)
    shy = np.floor(ny / 2).astype(int)
    kxs = np.roll(kx, shx)
    kys = np.roll(ky, shy)
    kxs[:shx] = kxs[:shx] - kx[nx-1] - dkx
    kys[:shy] = kys[:shy] - ky[ny-1] - dky
    
    E = np.zeros((nx, ny))

#     kx2 = np.tile(kxs, (1, ny))
#     ky2 = np.tile(kys, (nx, 1))

    kx2 = np.tile(kxs[:, np.newaxis], (1, ny))  # Répéter kxs en colonnes, forme (62, 62)
    ky2 = np.tile(kys, (nx, 1))  # Répéter kys en lignes, forme (62, 62)
    
#     print('kxs',np.shape(kxs))
#     print('kys',np.shape(kys))
#     print('ny',ny)
    
    nkx = len(kxs)
    nky = len(kys)
    kxs2 = np.tile(kxs.reshape(-1, 1), (1, nky)).T
    kys2 = np.tile(kys.reshape(-1, 1), (1, nkx))

    kn = np.sqrt(kxs2**2 + kys2**2)
    sig = np.sqrt(9.81 * kn)
    
#     print('kx2',np.shape(kx2))
#     print('ky2',np.shape(ky2))


    theta2 = np.arctan2(ky2, kx2)

    # 5. Initialize matrices
    E1 = E.copy()
    E2 = E.copy()
    E3 = E.copy()

    F1 = np.zeros((nx, ny), dtype=np.complex128)
    F2 = np.zeros((nx, ny), dtype=np.complex128)
    F3 = np.zeros((nx, ny), dtype=np.complex128)
 
    
    U = E.copy()
    Uvar = U.copy()
    EA = E.copy()
    EB = E.copy()
    Etb = E.copy()
    Unow = E.copy()
    eps2now = E.copy()

    phase12 = np.zeros((nx, ny), dtype=np.complex128)
    phase23 = np.zeros((nx, ny), dtype=np.complex128)
    phase31 = np.zeros((nx, ny), dtype=np.complex128)
    
    coh12 = E.copy()
    coh23 = E.copy()
    coh31 = E.copy()
    phases = np.zeros((nx, ny, n**2 + (n-1)**2),dtype=np.complex128)
    Uall = np.zeros((nx, ny, n**2 + (n-1)**2))
    eps2s = np.zeros((nx, ny, n**2 + (n-1)**2))

    nspec = 0
    nU = E.copy()
    UOK = E.copy()

    r1 = (wc2x * wc2y) / (dkx * dky)
    r2 = np.sqrt(r1)

    # Optimization options (Python equivalent using scipy.optimize.minimize)
    options = {'xatol': Utol, 'disp': False}

    nx1 = np.floor(nx / 2).astype(int)
    ny1 = np.floor(ny / 2).astype(int)
    
    mspec = n**2 + (n-1)**2

    # Main loop on samples
    for m in range(1, mspec + 1):
        if m <= n**2:
            i1 = (m-1) // n + 1
            i2 = m - (i1-1) * n

            # Extracting tiles
            tile1 = array1[nx*(i1-1):nx*i1, ny*(i2-1):ny*i2].astype(np.float64)
            tile2 = array2[nx*(i1-1):nx*i1, ny*(i2-1):ny*i2].astype(np.float64)
            tile3 = array3[nx*(i1-1):nx*i1, ny*(i2-1):ny*i2].astype(np.float64)
        else:
            # Shifted 50% like Welch
            i1 = (m - n**2 - 1) // (n - 1) + 1
            i2 = m - n**2 - (i1-1) * (n-1)

            # Extracting tiles
            tile1 = array1[nx*(i1-1)+nx1:nx*i1+nx1, ny*(i2-1)+ny1:ny*i2+ny1].astype(np.float64)
            tile2 = array2[nx*(i1-1)+nx1:nx*i1+nx1, ny*(i2-1)+ny1:ny*i2+ny1].astype(np.float64)
            tile3 = array3[nx*(i1-1)+nx1:nx*i1+nx1, ny*(i2-1)+ny1:ny*i2+ny1].astype(np.float64)

        # Processing tile1
        z2a = tile1 - np.mean(tile1)
        zb = z2a * hanningxy
        zc1 = fftshift(fftn(zb) / (nx * ny), axes=(0, 1))
        E1 += (np.abs(zc1) ** 2) * r1

        # Processing tile2
        z2a = tile2 - np.mean(tile2)
        zb = z2a * hanningxy
        zc2 = fftshift(fftn(zb) / (nx * ny), axes=(0, 1))
        E2 += (np.abs(zc2) ** 2) * r1

        # Processing tile3
        z2a = tile3 - np.mean(tile3)
        zb = z2a * hanningxy
        zc3 = fftshift(fftn(zb) / (nx * ny), axes=(0, 1))
        E3 += (np.abs(zc3) ** 2) * r1

        Z1a = np.angle(zc1)
        
#         print('Type de zc1:', type(zc1), 'Dtype de zc1:', zc1.dtype)
#         print('Type de Z1a:', type(Z1a), 'Dtype de Z1a:', Z1a.dtype)
#         print('Type de F1:', type(F1), 'Dtype de F1:', F1.dtype)

        F1 += zc1 * np.exp(-1j * Z1a)
        F2 += zc2 * np.exp(-1j * Z1a)
        F3 += zc3 * np.exp(-1j * Z1a)
              

        phase12 += zc2 * np.conj(zc1) * r1
        phase23 += zc3 * np.conj(zc2) * r1
        phase31 += zc3 * np.conj(zc1) * r1
        
        npixels = nx * ny

        UOK = np.zeros_like(E1)
        eps2now = np.zeros_like(E1)
        Unow = np.zeros_like(E1)+np.nan
        
#         print(f" Uall.shape: {Uall.shape}")
#         print(f" eps2s.shape: {eps2s.shape}")
#         print(f" eps2now.shape: {eps2now.shape}")
#         print(f" Unow.shape: {Unow.shape}")
#         print('npixel',npixels)

        numspec=Uall.shape[2]-1

        for jj in range(npixels):
#   kxs2.flat[jj] >= 0 : only half of the spectral plane, because it is symmetric. 
            if abs(zc1.flat[jj]) > 1E-8 and kn.flat[jj] > 1E-3 and kxs2.flat[jj] >= 0:  
                # Ufun utilise eps2 pour l'optimisation
                Ufun = lambda Uc: abs(ULSmin(Uc, kn.flat[jj], sig.flat[jj], imgtimes, [zc1.flat[jj], zc2.flat[jj], zc3.flat[jj]])[0])
                result = minimize_scalar(Ufun, bounds=(Umin-0.1, Umax+0.1), method='bounded', options=options)
                Uc = result.x
                eps2, A, B = ULSmin(Uc, kn.flat[jj], sig.flat[jj], imgtimes, [zc1.flat[jj], zc2.flat[jj], zc3.flat[jj]])

                if abs(Uc - Umid) < abs(Umax - Umid):
                    nU.flat[jj] += 1
                    UOK.flat[jj] = 1
                    eps2now.flat[jj] = np.sum(eps2) / (abs(zc1.flat[jj])**2 + abs(zc2.flat[jj])**2 + abs(zc3.flat[jj])**2)
                    Unow.flat[jj] = Uc
                    EA.flat[jj] += (abs(A)**2) * np.random.random()  # r1 not defined, using random value for example
                    EB.flat[jj] += (abs(B)**2) * np.random.random()  # r1 not defined, using random value for example
                    U += Unow
                else:
                    eps2now.flat[jj] = -1
        
#         print('nspec0', nspec)

        if nspec < numspec:  # Ensure nspec is within bounds
            Uall[:, :, nspec] = Unow
            eps2s[:, :, nspec] = eps2now
            phases[:, :, nspec] = zc3 * np.conj(zc1) / (abs(zc3) * abs(zc1))
            nspec += 1  # Only increment nspec if within bounds

    # Rotate phases around the mean phase to compute std
#     print('phases dtype',phases.dtype)
#     print('phase31 dtype',phase31.dtype)
#     print('E1 dtype',E1.dtype)
#     print('nspec ',type(nspec))
    E1 /= nspec
    E2 /= nspec
    E3 /= nspec
    U = np.nanmedian(Uall[:, :, :nspec], axis=2)  # More accurate than the average
    
    for m in range(nspec):
        phases[:, :, m] /= phase31
       

    EA /= nU
    EB /= nU
    Uvar =  np.nanstd(Uall[:, :, :nspec], axis=2)  
    coh12 = abs((phase12 / nspec) ** 2) / (E1 * E2)
    coh23 = abs((phase23 / nspec) ** 2) / (E2 * E3)
    coh31 = abs((phase31 / nspec) ** 2) / (E3 * E1)
    

#    UOK = np.zeros_like(E1)
#    for jj in range(npixels):
#        # Vérifier que jj est dans les limites des tableaux concernés
#        if jj < len(zc1.flat) and jj < len(kn.flat) and jj < len(sig.flat) and jj < len(F1.flat):
#            if abs(zc1.flat[jj]) > 1E-8 and kn.flat[jj] > 1E-3:
#                UOK.flat[jj] = 1
#
#                # Fonction d'erreur à minimiser, en vérifiant que les indices sont valides
#                def Ufun(Uc):
#                    return abs(ULS(Uc, kn.flat[jj], sig.flat[jj], imgtimes, [F1.flat[jj], F2.flat[jj], F3.flat[jj]])[0])
#
#                # Minimiser la fonction
#                result = minimize_scalar(Ufun, bounds=(Umin - 0.1, Umax + 0.1), method='bounded', options=options)
#                U2.flat[jj] = result.x
#
#                # Calculer les valeurs nécessaires avec ULS
#                Ufval, Ac, Bc, eps2 = ULS(U2.flat[jj], kn.flat[jj], sig.flat[jj], imgtimes, [F1.flat[jj], F2.flat[jj], F3.flat[jj]])
#        else:
#            # Lorsque jj dépasse la taille des tableaux, on peut soit sortir de la boucle soit simplement passer
#            continue

    ang12 = np.angle(phase12)
    ang23 = np.angle(phase23)
    ang31 = np.angle(phase31)

    ang = np.angle(phase31)
    angstd = np.std(np.angle(phases), axis=2)

    # uses cycles per meter for wavenumbers (i.e. "spatial frequency") 
    kxs /=  (2*np.pi)
    kys /=  (2*np.pi)
    facnorm=1/(2*np.pi)**2
    E1 /= facnorm
    E2 /= facnorm
    E3 /= facnorm
    
    return (E1, E2, E3, U, Uvar, Uall, EA, EB, nU, coh12, coh23, coh31, ang12, ang23, ang31, kxs, kys, angstd, phases, eps2s)

#----------------------------------------------------------------------------------------------------------------#

def FFT2D_three_arrays(arraya, arrayb, arrayc, dx, dy, n, isplot=0):
    # Welch-based 2D spectral analysis
    # nxa, nya : size of arraya
    # dx, dy : resolution of arraya
    # n : number of tiles in each direction
    # 
    # Eta is PSD of 1st image (arraya)
    # Etb is PSD of 2nd image (arrayb)
    # Etc is PSD of 3rd image (arrayc)
    
    [nxa, nya] = np.shape(arraya)

    mspec = n**2 + (n-1)**2
    nxtile = int(np.floor(nxa/n))
    nytile = int(np.floor(nya/n))

    dkxtile = 1/(dx*nxtile)
    dkytile = 1/(dy*nytile)

    shx = int(nxtile//2)   # OK if nxtile is even number
    shy = int(nytile//2)

    ### --- prepare wavenumber vectors -------------------------
    # wavenumbers starting at zero
    kx = np.fft.fftshift(np.fft.fftfreq(nxtile, dx)) # wavenumber in cycles / m
    ky = np.fft.fftshift(np.fft.fftfreq(nytile, dy)) # wavenumber in cycles / m
    kx2, ky2 = np.meshgrid(kx, ky, indexing='ij')

    if isplot:
        X = np.arange(0, nxa*dx, dx) # from 0 to (nx-1)*dx with a dx step
        Y = np.arange(0, nya*dy, dy)

    ### --- prepare Hanning windows for performing fft and associated normalization ------------------------

    hanningx = (0.5 * (1 - np.cos(2*np.pi*np.linspace(0, nxtile-1, nxtile) / (nxtile-1))))
    hanningy = (0.5 * (1 - np.cos(2*np.pi*np.linspace(0, nytile-1, nytile) / (nytile-1))))
    # 2D Hanning window
    hanningxy = np.atleast_2d(hanningy) * np.atleast_2d(hanningx).T 

    wc2x = 1 / np.mean(hanningx**2)  # window correction factor
    wc2y = 1 / np.mean(hanningy**2)  # window correction factor

    normalization = (wc2x * wc2y) / (dkxtile * dkytile)

    ### --- Initialize Eta = mean spectrum over tiles ---------------------

    Eta = np.zeros((nxtile, nytile))
    Etb = np.zeros((nxtile, nytile))
    Etc = np.zeros((nxtile, nytile))
    phase_ab = np.zeros((nxtile, nytile), dtype='complex_')
    phase_ac = np.zeros((nxtile, nytile), dtype='complex_')
    phase_bc = np.zeros((nxtile, nytile), dtype='complex_')
    
    phases_ab = np.zeros((nxtile, nytile, mspec), dtype='complex_')
    phases_ac = np.zeros((nxtile, nytile, mspec), dtype='complex_')
    phases_bc = np.zeros((nxtile, nytile, mspec), dtype='complex_')
    
    if isplot:
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.pcolormesh(X, Y, arraya)
        colors = plt.cm.seismic(np.linspace(0, 1, mspec))

    ### --- Calculate spectrum for each tiles ----------------------------
    nspec = 0
    for m in range(mspec):
        ### 1. Selection of tile ------------------------------
        if (m < n**2):
            i1 = int(np.floor(m/n) + 1)
            i2 = int(m + 1 - (i1 - 1) * n)

            ix1 = nxtile * (i1 - 1)
            ix2 = nxtile * i1 - 1
            iy1 = nytile * (i2 - 1)
            iy2 = nytile * i2 - 1

            array1 = np.double(arraya[ix1:ix2 + 1, iy1:iy2 + 1])
            array2 = np.double(arrayb[ix1:ix2 + 1, iy1:iy2 + 1])
            array3 = np.double(arrayc[ix1:ix2 + 1, iy1:iy2 + 1])
            if isplot:
                ax1.plot(X[[ix1, ix1, ix2, ix2, ix1]], Y[[iy1, iy2, iy2, iy1, iy1]], '-', color=colors[m], linewidth=2)
        else:
            # Select a 'tile' overlapping (50%) the main tiles
            i1 = int(np.floor((m - n**2) / (n - 1)) + 1)
            i2 = int(m + 1 - n**2 - (i1 - 1) * (n - 1))

            ix1 = nxtile * (i1 - 1) + shx 
            ix2 = nxtile * i1 + shx - 1
            iy1 = nytile * (i2 - 1) + shy
            iy2 = nytile * i2 + shy - 1

            array1 = np.double(arraya[ix1:ix2 + 1, iy1:iy2 + 1])
            array2 = np.double(arrayb[ix1:ix2 + 1, iy1:iy2 + 1])
            array3 = np.double(arrayc[ix1:ix2 + 1, iy1:iy2 + 1])
            if isplot:
                ax1.plot(X[[ix1, ix1, ix2, ix2, ix1]], Y[[iy1, iy2, iy2, iy1, iy1]], '-', color=colors[m], linewidth=2)

        ### 2. Work over 1 tile ------------------------------ 
        # For arraya
        tile_centered = array1 - np.mean(array1.flatten())
        tile_by_windows = (tile_centered) * hanningxy

        tileFFT1 = np.fft.fft2(tile_by_windows, norm="forward")
        tileFFT1_shift = np.fft.fftshift(tileFFT1)
        Eta[:, :] = Eta[:, :] + (abs(tileFFT1_shift) ** 2) * normalization

        # For arrayb
        tile_centered = array2 - np.mean(array2.flatten())
        tile_by_windows = (tile_centered) * hanningxy

        tileFFT2 = np.fft.fft2(tile_by_windows, norm="forward")
        tileFFT2_shift = np.fft.fftshift(tileFFT2)
        Etb[:, :] = Etb[:, :] + (abs(tileFFT2_shift) ** 2) * normalization

        # For arrayc
        tile_centered = array3 - np.mean(array3.flatten())
        tile_by_windows = (tile_centered) * hanningxy

        tileFFT3 = np.fft.fft2(tile_by_windows, norm="forward")
        tileFFT3_shift = np.fft.fftshift(tileFFT3)
        Etc[:, :] = Etc[:, :] + (abs(tileFFT3_shift) ** 2) * normalization

        phase_ab = phase_ab + (tileFFT2_shift * np.conj(tileFFT1_shift)) * normalization
        phase_ac = phase_ac + (tileFFT3_shift * np.conj(tileFFT1_shift)) * normalization
        phase_bc = phase_bc + (tileFFT3_shift * np.conj(tileFFT2_shift)) * normalization
        
        nspec = nspec + 1
        phases_ab[:, :, m] = tileFFT2_shift * np.conj(tileFFT1_shift) / (abs(tileFFT2_shift) * abs(tileFFT1_shift))
        phases_ac[:, :, m] = tileFFT3_shift * np.conj(tileFFT1_shift) / (abs(tileFFT3_shift) * abs(tileFFT1_shift))
        phases_bc[:, :, m] = tileFFT3_shift * np.conj(tileFFT2_shift) / (abs(tileFFT3_shift) * abs(tileFFT2_shift))

    # Rotates phases around the mean phase to be able to compute std
    for m in range(mspec):
        phases_ab[:, :, m] = phases_ab[:, :, m] / phase_ab
        phases_ac[:, :, m] = phases_ac[:, :, m] / phase_ac
        phases_bc[:, :, m] = phases_bc[:, :, m] / phase_bc

    # Now works with averaged spectra
    Eta = Eta / nspec
    Etb = Etb / nspec
    Etc = Etc / nspec
    
    coh_ab = abs((phase_ab / nspec) ** 2) / (Eta * Etb)  # spectral coherence between arraya and arrayb
    coh_ac = abs((phase_ac / nspec) ** 2) / (Eta * Etc)  # spectral coherence between arraya and arrayc
    coh_bc = abs((phase_bc / nspec) ** 2) / (Etb * Etc)  # spectral coherence between arrayb and arrayc
    
    ang_ab = np.angle(phase_ab, deg=False)
    ang_ac = np.angle(phase_ac, deg=False)
    ang_bc = np.angle(phase_bc, deg=False)
    
    crosr_ab = np.real(phase_ab) / mspec
    crosr_ac = np.real(phase_ac) / mspec
    crosr_bc = np.real(phase_bc) / mspec
    
    angstd_ab = np.std(np.angle(phases_ab, deg=False), axis=2)
    angstd_ac = np.std(np.angle(phases_ac, deg=False), axis=2)
    angstd_bc = np.std(np.angle(phases_bc, deg=False), axis=2)


    return (Eta, Etb, Etc, ang_ab, ang_ac, ang_bc, angstd_ab, angstd_ac, angstd_bc, coh_ab, coh_ac, coh_bc, crosr_ab, crosr_ac, crosr_bc, phases_ab, phases_ac, phases_bc, kx2, ky2, dkxtile, dkytile)




