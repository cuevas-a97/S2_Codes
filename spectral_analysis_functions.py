# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
import numpy as np

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal
import scipy.linalg

def FFT1D_two_arrays(arraya,arrayb,dx,isplot=0,fftaxis=0):
    [nxa,nya]=np.shape(arraya)
    array1=np.double(arraya)
    array2=np.double(arrayb)
    kx=np.fft.fftfreq(nxa, dx) # wavenumber in cycles / m
    dkxtile=1/(dx*nxa)  
    hanningx=(0.5 * (1-np.cos(2*np.pi*np.linspace(0,nxa-1,nxa)/(nxa-1))))
    hanningx2=np.tile(hanningx,(nya,1)).T
    fig,axs=plt.subplots(1,2,figsize=(11.2,7))#,sharey=True,sharex=True)
    
    wc2x=1/np.mean(hanningx**2);                              # window correction factor
    normalization = (wc2x)/(dkxtile)
    tile_centered=signal.detrend(array1,axis=fftaxis) 
    tile_by_windows = (tile_centered)*hanningx2
    tileFFT1 = np.fft.fft(tile_by_windows,norm="forward",axis=fftaxis)
    tile_centered=signal.detrend(array2,axis=fftaxis) 
    tile_by_windows = (tile_centered)*hanningx2
    tileFFT2 = np.fft.fft(tile_by_windows,norm="forward",axis=fftaxis)
    
    Eta=  np.mean( (abs(tileFFT1)**2),axis=1-fftaxis) *normalization #   
    Etb=  np.mean( (abs(tileFFT2)**2),axis=1-fftaxis) *normalization #   
    phase=np.mean(tileFFT2*np.conj(tileFFT1),axis=1-fftaxis) *normalization
    coh=abs((phase)**2)/(Eta*Etb)      # spectral coherence
    ang=np.angle(phase)
    return Eta[0:nxa//2],Etb[0:nxa//2],ang[0:nxa//2],coh[0:nxa//2],kx[0:nxa//2],dkxtile


def FFT2D_one_array(arraya,dx,dy,n,isplot=0):
# Welch-based 2D spectral analysis
# arraya: input array 
# dx,dy : resolution of arraya for dimensions 0,1
# n : number of tiles in each directions ... 
# 
# Eta is PSD of 1st image (arraya) 
    [nxa,nya]=np.shape(arraya)
    mspec=n**2+(n-1)**2
    nxtile=int(np.floor(nxa/n))
    nytile=int(np.floor(nya/n))

    dkxtile=1/(dx*nxtile)   
    dkytile=1/(dy*nytile)

    shx = int(nxtile//2)   # OK if nxtile is even number
    shy = int(nytile//2)

    ### --- prepare wavenumber vectors -------------------------
    kx=np.fft.fftshift(np.fft.fftfreq(nxtile, dx)) # wavenumber in cycles / m
    #print('NXt:',nxtile,1000/(dx*nxtile),1000/(dy*nytile))
    ky=np.fft.fftshift(np.fft.fftfreq(nytile, dy)) # wavenumber in cycles / m
    kx2,ky2 = np.meshgrid(kx,ky, indexing='ij')
    if isplot:
        X = np.arange(0,nxa*dx,dx) # from 0 to (nx-1)*dx with a dx step
        Y = np.arange(0,nya*dy,dy)

    ### --- prepare Hanning windows for performing fft and associated normalization ------------------------

    hanningx=(0.5 * (1-np.cos(2*np.pi*np.linspace(0,nxtile-1,nxtile)/(nxtile-1))))
    hanningy=(0.5 * (1-np.cos(2*np.pi*np.linspace(0,nytile-1,nytile)/(nytile-1))))
    # 2D Hanning window
    #hanningxy=np.atleast_2d(hanningx)*np.atleast_2d(hanningy).T 
    hanningxy=np.atleast_2d(hanningy)*np.atleast_2d(hanningx).T 

    wc2x=1/np.mean(hanningx**2);                              # window correction factor
    wc2y=1/np.mean(hanningy**2);                              # window correction factor

    normalization = (wc2x*wc2y)/(dkxtile*dkytile)

    ### --- Initialize Eta = mean spectrum over tiles ---------------------

    Eta=np.zeros((nxtile,nytile))
    Eta_all=np.zeros((nxtile,nytile,mspec))
    if isplot:
        fig1,ax1=plt.subplots(figsize=(12,6))
        ax1.pcolormesh(X,Y,arraya)
        colors = plt.cm.seismic(np.linspace(0,1,mspec))

    ### --- Calculate spectrum for each tiles ----------------------------
    for m in range(mspec):
        ### 1. Selection of tile ------------------------------
        if (m<n**2):
            i1=int(np.floor(m/n)+1)
            i2=int(m+1-(i1-1)*n)

            ix1 = nxtile*(i1-1)
            ix2 = nxtile*i1-1
            iy1 = nytile*(i2-1)
            iy2 = nytile*i2-1

            #                 array1=double(arraya(nx*(i1-1)+1:nx*i1,ny*(i2-1)+1:ny*i2));
            #        Select a 'tile' i.e. part of the surface : main loop ---------

            array1=np.double(arraya[ix1:ix2+1,iy1:iy2+1])
            if isplot:
                ax1.plot(X[[ix1,ix1,ix2,ix2,ix1]],Y[[iy1,iy2,iy2,iy1,iy1]],'-',color=colors[m],linewidth=2)
        else:
            #        # -- Select a 'tile' overlapping (50%) the main tiles ---
            #        %%%%%%%%%%%%%%% now shifted 50% , like Welch %%%%%%%%%%%%%%
            i1=int(np.floor((m-n**2)/(n-1))+1)
            i2=int(m+1-n**2-(i1-1)*(n-1))


            ix1 = nxtile*(i1-1)+shx 
            ix2 = nxtile*i1+shx-1
            iy1 = nytile*(i2-1)+shy
            iy2 = nytile*i2+shy-1

            array1=np.double(arraya[ix1:ix2+1,iy1:iy2+1])
            if isplot:
                ax1.plot(X[[ix1,ix1,ix2,ix2,ix1]],Y[[iy1,iy2,iy2,iy1,iy1]],'-',color=colors[m],linewidth=2)

        ### 2. Work over 1 tile ------------------------------ 
        tile_centered=array1-np.mean(array1.flatten())
        tile_by_windows = (tile_centered)*hanningxy

        # 
        tileFFT = np.fft.fft2(tile_by_windows,norm="forward")
        tileFFT_shift = np.fft.fftshift(tileFFT)
        Eta_all[:,:,m] = (abs(tileFFT_shift)**2) *normalization
        Eta[:,:] = Eta[:,:] + (abs(tileFFT_shift)**2) *normalization #          % sum of spectra for all tiles

    return Eta/mspec,Eta_all,kx2,ky2,dkxtile,dkytile


#####################################################################################
def FFT2D_two_arrays(arraya,arrayb,dx,dy,n,isplot=0):
# Welch-based 2D spectral analysis
# nxa, nya : size of arraya
# dx,dy : resolution of arraya
# n : number of tiles in each directions ... 
# 
# Eta is PSD of 1st image (arraya) 
# Etb is PSD of 2st image (arraya) 
    [nxa,nya]=np.shape(arraya)
    mspec=n**2+(n-1)**2
    nxtile=int(np.floor(nxa/n))
    nytile=int(np.floor(nya/n))

    dkxtile=1/(dx*nxtile)   
    dkytile=1/(dy*nytile)

    shx = int(nxtile//2)   # OK if nxtile is even number
    shy = int(nytile//2)

    ### --- prepare wavenumber vectors -------------------------
    # wavenumbers starting at zero
    kx=np.fft.fftshift(np.fft.fftfreq(nxtile, dx)) # wavenumber in cycles / m
    ky=np.fft.fftshift(np.fft.fftfreq(nytile, dy)) # wavenumber in cycles / m
    kx2,ky2 = np.meshgrid(kx,ky, indexing='ij')

    if isplot:
        X = np.arange(0,nxa*dx,dx) # from 0 to (nx-1)*dx with a dx step
        Y = np.arange(0,nya*dy,dy)

    ### --- prepare Hanning windows for performing fft and associated normalization ------------------------

    hanningx=(0.5 * (1-np.cos(2*np.pi*np.linspace(0,nxtile-1,nxtile)/(nxtile-1))))
    hanningy=(0.5 * (1-np.cos(2*np.pi*np.linspace(0,nytile-1,nytile)/(nytile-1))))
    # 2D Hanning window
    #hanningxy=np.atleast_2d(hanningx)*np.atleast_2d(hanningy).T 
    hanningxy=np.atleast_2d(hanningy)*np.atleast_2d(hanningx).T 

    wc2x=1/np.mean(hanningx**2);                              # window correction factor
    wc2y=1/np.mean(hanningy**2);                              # window correction factor

    normalization = (wc2x*wc2y)/(dkxtile*dkytile)

    ### --- Initialize Eta = mean spectrum over tiles ---------------------

    Eta=np.zeros((nxtile,nytile))
    Etb=np.zeros((nxtile,nytile))
    phase=np.zeros((nxtile,nytile),dtype = np.complex128)
    #Eta_all=np.zeros((nxtile,nytile,mspec))
    phases=np.zeros((nxtile,nytile,mspec),dtype = np.complex128)
    if isplot:
        fig1,ax1=plt.subplots(figsize=(12,6))
        ax1.pcolormesh(X,Y,arraya)
        colors = plt.cm.seismic(np.linspace(0,1,mspec))

    ### --- Calculate spectrum for each tiles ----------------------------
    nspec=0
    for m in range(mspec):
        ### 1. Selection of tile ------------------------------
        if (m<n**2):
            i1=int(np.floor(m/n)+1)
            i2=int(m+1-(i1-1)*n)

            ix1 = nxtile*(i1-1)
            ix2 = nxtile*i1-1
            iy1 = nytile*(i2-1)
            iy2 = nytile*i2-1

            #                 array1=double(arraya(nx*(i1-1)+1:nx*i1,ny*(i2-1)+1:ny*i2));
            #        Select a 'tile' i.e. part of the surface : main loop ---------

            array1=np.double(arraya[ix1:ix2+1,iy1:iy2+1])
            array2=np.double(arrayb[ix1:ix2+1,iy1:iy2+1])
            if isplot:
                ax1.plot(X[[ix1,ix1,ix2,ix2,ix1]],Y[[iy1,iy2,iy2,iy1,iy1]],'-',color=colors[m],linewidth=2)
        else:
            #        # -- Select a 'tile' overlapping (50%) the main tiles ---
            #        %%%%%%%%%%%%%%% now shifted 50% , like Welch %%%%%%%%%%%%%%
            i1=int(np.floor((m-n**2)/(n-1))+1)
            i2=int(m+1-n**2-(i1-1)*(n-1))


            ix1 = nxtile*(i1-1)+shx 
            ix2 = nxtile*i1+shx-1
            iy1 = nytile*(i2-1)+shy
            iy2 = nytile*i2+shy-1

            array1=np.double(arraya[ix1:ix2+1,iy1:iy2+1])
            array2=np.double(arrayb[ix1:ix2+1,iy1:iy2+1])
            if isplot:
                ax1.plot(X[[ix1,ix1,ix2,ix2,ix1]],Y[[iy1,iy2,iy2,iy1,iy1]],'-',color=colors[m],linewidth=2)

        ### 2. Work over 1 tile ------------------------------ 
        tile_centered=array1-np.mean(array1.flatten())
        tile_by_windows = (tile_centered)*hanningxy

        # 
        tileFFT1 = np.fft.fft2(tile_by_windows,norm="forward")
        tileFFT1_shift = np.fft.fftshift(tileFFT1)
        #Eta_all[:,:,m] = (abs(tileFFT1_shift)**2) *normalization
        Eta[:,:] = Eta[:,:] + (abs(tileFFT1_shift)**2) *normalization #          % sum of spectra for all tiles

        tile_centered=array2-np.mean(array2.flatten())
        tile_by_windows = (tile_centered)*hanningxy

        tileFFT2 = np.fft.fft2(tile_by_windows,norm="forward")
        tileFFT2_shift = np.fft.fftshift(tileFFT2)#
        #Etb_all[:,:,m] = (abs(tileFFT2_shift)**2) *normalization
        Etb[:,:] = Etb[:,:] + (abs(tileFFT2_shift)**2) *normalization #          % sum of spectra for all tiles

        phase=phase+(tileFFT2_shift*np.conj(tileFFT1_shift))*normalization
        nspec=nspec+1
        phases[:,:,m]=tileFFT2_shift*np.conj(tileFFT1_shift)/(abs(tileFFT2_shift)*abs(tileFFT1_shift)); 

# rotates phases around the mean phase to be able to compute std
    for m in range(mspec):
        phases[:,:,m]=phases[:,:,m]/phase;

# Now works with averaged spectra
    Eta=Eta/nspec
    Etb=Etb/nspec
    coh=abs((phase/nspec)**2)/(Eta*Etb)      # spectral coherence
    ang=np.angle(phase,deg=False)
    crosr=np.real(phase)/mspec
    angstd=np.std(np.angle(phases,deg=False),axis=2)
    
    return Eta,Etb,ang,angstd,coh,crosr,phases,kx2,ky2,dkxtile,dkytile

##############################################################################

def FFT2D_two_arrays_nonan(arraya, arrayb, dx, dy, n, isplot=0):
    # Welch-based 2D spectral analysis
    # nxa, nya : size of arraya
    # dx, dy : resolution of arraya
    # n : number of tiles in each direction

    [nxa, nya] = np.shape(arraya)
    mspec = n**2 + (n-1)**2
    nxtile = int(np.floor(nxa / n))
    nytile = int(np.floor(nya / n))

    dkxtile = 1 / (dx * nxtile)
    dkytile = 1 / (dy * nytile)

    shx = int(nxtile // 2)  # OK if nxtile is even number
    shy = int(nytile // 2)

    # --- Prepare wavenumber vectors ---
    kx = np.fft.fftshift(np.fft.fftfreq(nxtile, dx)) 
    ky = np.fft.fftshift(np.fft.fftfreq(nytile, dy)) 
    kx2, ky2 = np.meshgrid(kx, ky, indexing='ij')

    if isplot:
        X = np.arange(0, nxa * dx, dx)
        Y = np.arange(0, nya * dy, dy)

    # --- Prepare Hanning windows ---
    hanningx = (0.5 * (1 - np.cos(2 * np.pi * np.linspace(0, nxtile - 1, nxtile) / (nxtile - 1))))
    hanningy = (0.5 * (1 - np.cos(2 * np.pi * np.linspace(0, nytile - 1, nytile) / (nytile - 1))))
    hanningxy = np.atleast_2d(hanningy) * np.atleast_2d(hanningx).T 

    wc2x = 1 / np.mean(hanningx**2)
    wc2y = 1 / np.mean(hanningy**2)
    normalization = (wc2x * wc2y) / (dkxtile * dkytile)

    # --- Initialize ---
    Eta = np.zeros((nxtile, nytile))
    Etb = np.zeros((nxtile, nytile))
    phase = np.zeros((nxtile, nytile), dtype=np.complex128)
    phases = np.zeros((nxtile, nytile, mspec), dtype=np.complex128)

    if isplot:
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.pcolormesh(X, Y, arraya)
        colors = plt.cm.seismic(np.linspace(0, 1, mspec))

    # --- Calculate spectrum for each tile ---
    nspec = 0
    for m in range(mspec):
        if m < n**2:
            i1 = int(np.floor(m / n) + 1)
            i2 = int(m + 1 - (i1 - 1) * n)

            ix1, ix2 = nxtile * (i1 - 1), nxtile * i1 - 1
            iy1, iy2 = nytile * (i2 - 1), nytile * i2 - 1
        else:
            i1 = int(np.floor((m - n**2) / (n - 1)) + 1)
            i2 = int(m + 1 - n**2 - (i1 - 1) * (n - 1))

            ix1, ix2 = nxtile * (i1 - 1) + shx, nxtile * i1 + shx - 1
            iy1, iy2 = nytile * (i2 - 1) + shy, nytile * i2 + shy - 1

        array1 = np.array(arraya[ix1:ix2+1, iy1:iy2+1], dtype=np.float64)
        array2 = np.array(arrayb[ix1:ix2+1, iy1:iy2+1], dtype=np.float64)

        # **Gestion des NaN**
        mask1 = np.isnan(array1)
        mask2 = np.isnan(array2)

        # Remplacement des NaN par la moyenne locale
        if np.any(mask1):
            array1[mask1] = np.nanmean(array1)
        if np.any(mask2):
            array2[mask2] = np.nanmean(array2)

        # Ignorer les tuiles avec trop de NaN (ex: >50% de la tuile)
        if np.sum(mask1) > 0.5 * array1.size or np.sum(mask2) > 0.5 * array2.size:
            continue

        # **Application de la fenêtre de Hanning en tenant compte des NaN**
        valid_mask1 = (~mask1).astype(float) 
        valid_mask2 = (~mask2).astype(float)

        tile_centered = (array1 - np.nanmean(array1)) * valid_mask1
        tile_by_windows = tile_centered * hanningxy
        tileFFT1 = np.fft.fft2(tile_by_windows, norm="forward")
        tileFFT1_shift = np.fft.fftshift(tileFFT1)
        Eta += (abs(tileFFT1_shift) ** 2) * normalization  

        tile_centered = (array2 - np.nanmean(array2)) * valid_mask2
        tile_by_windows = tile_centered * hanningxy
        tileFFT2 = np.fft.fft2(tile_by_windows, norm="forward")
        tileFFT2_shift = np.fft.fftshift(tileFFT2)
        Etb += (abs(tileFFT2_shift) ** 2) * normalization  

        phase += tileFFT2_shift * np.conj(tileFFT1_shift) * normalization
        nspec += 1
        phases[:, :, m] = tileFFT2_shift * np.conj(tileFFT1_shift) / (abs(tileFFT2_shift) * abs(tileFFT1_shift))

    # **Rotation des phases autour de la moyenne**
    for m in range(mspec):
        phases[:, :, m] /= phase

    # **Normalisation finale**
    Eta /= nspec
    Etb /= nspec
    coh = abs((phase / nspec) ** 2) / (Eta * Etb)  # spectral coherence
    ang = np.angle(phase, deg=False)
    crosr = np.real(phase) / mspec
    angstd = np.std(np.angle(phases, deg=False), axis=2)

    return Eta, Etb, ang, angstd, coh, crosr, phases, kx2, ky2, dkxtile, dkytile
##############################################################################

def FFT2D_two_arrays_nm_detrend(arraya,arrayb,dx,dy,n,m,isplot=0,detrend='linear'):
# Welch-based 2D spectral analysis
# nxa, nya : size of arraya
# dx,dy : resolution of arraya
# n,m : number of tiles in each directions ... 
# 
# Eta is PSD of 1st image (arraya) 
# Etb is PSD of 2st image (arraya) 
    [nxa,nya]=np.shape(arraya)
    arrayad=np.zeros((nxa,nya))
    arraybd=np.zeros((nxa,nya))
    
    print('sizes:',nxa,nya,np.shape(arraya))
           
    mspec=n*m+(n-1)*(m-1)
    nxtile=nxa//n
    nytile=nya//m

    print('tile sizes:',nxtile,nytile)

    dkxtile=1/(dx*nxtile)   
    dkytile=1/(dy*nytile)

    shx = int(nxtile//2)   # OK if nxtile is even number
    shy = int(nytile//2)

    ### --- prepare wavenumber vectors -------------------------
    # wavenumbers starting at zero
    kx=np.fft.fftshift(np.fft.fftfreq(nxtile, dx)) # wavenumber in cycles / m
    ky=np.fft.fftshift(np.fft.fftfreq(nytile, dy)) # wavenumber in cycles / m
    kx2,ky2 = np.meshgrid(kx,ky, indexing='ij')

    if isplot:
        X = np.arange(0,nxa*dx,dx) # from 0 to (nx-1)*dx with a dx step
        Y = np.arange(0,nya*dy,dy)

    ### --- prepare Hanning windows for performing fft and associated normalization ------------------------

    hanningx=(0.5 * (1-np.cos(2*np.pi*np.linspace(0,nxtile-1,nxtile)/(nxtile-1))))
    hanningy=(0.5 * (1-np.cos(2*np.pi*np.linspace(0,nytile-1,nytile)/(nytile-1))))
    # 2D Hanning window
    #hanningxy=np.atleast_2d(hanningx)*np.atleast_2d(hanningy).T 
    hanningxy=np.atleast_2d(hanningy)*np.atleast_2d(hanningx).T 
  
    wc2x=1/np.mean(hanningx**2);                              # window correction factor
    wc2y=1/np.mean(hanningy**2);                              # window correction factor

    normalization = (wc2x*wc2y)/(dkxtile*dkytile)

    ### --- Initialize Eta = mean spectrum over tiles ---------------------

    Eta=np.zeros((nxtile,nytile))
    Etb=np.zeros((nxtile,nytile))
    phase=np.zeros((nxtile,nytile))
    #Eta_all=np.zeros((nxtile,nytile,mspec))
    phases=np.zeros((nxtile,nytile,mspec))
    if isplot:
        fig1,ax1=plt.subplots(figsize=(12,6))
        ax1.pcolormesh(X,Y,arraya)
        colors = plt.cm.seismic(np.linspace(0,1,mspec))

    ### --- Calculate spectrum for each tiles ----------------------------
    for im in range(mspec):
        ### 1. Selection of tile ------------------------------
        if (im<m*n):
            i1=int(np.floor(im/m)+1)
            i2=int(im+1-(i1-1)*m)

            ix1 = nxtile*(i1-1)
            ix2 = nxtile*i1-1
            iy1 = nytile*(i2-1)
            iy2 = nytile*i2-1

          
            #                 array1=double(arraya(nx*(i1-1)+1:nx*i1,ny*(i2-1)+1:ny*i2));
            #        Select a 'tile' i.e. part of the surface : main loop ---------

            array1=np.double(arraya[ix1:ix2+1,iy1:iy2+1])
            array2=np.double(arrayb[ix1:ix2+1,iy1:iy2+1])
            if isplot:
                ax1.plot(X[[ix1,ix1,ix2,ix2,ix1]],Y[[iy1,iy2,iy2,iy1,iy1]],'-',color=colors[m],linewidth=2)
        else:
            #        # -- Select a 'tile' overlapping (50%) the main tiles ---
            #        %%%%%%%%%%%%%%% now shifted 50% , like Welch %%%%%%%%%%%%%%
            i1=int(np.floor((im-m*n)/(m-1))+1)
            i2=int(im+1-n*m-(i1-1)*(m-1))


            ix1 = nxtile*(i1-1)+shx 
            ix2 = nxtile*i1+shx-1
            iy1 = nytile*(i2-1)+shy
            iy2 = nytile*i2+shy-1

            array1=np.double(arraya[ix1:ix2+1,iy1:iy2+1])
            array2=np.double(arrayb[ix1:ix2+1,iy1:iy2+1])
            if isplot:
                ax1.plot(X[[ix1,ix1,ix2,ix2,ix1]],Y[[iy1,iy2,iy2,iy1,iy1]],'-',color=colors[m],linewidth=2)

        if detrend =='linear':
            # Fits a plane using least squares
            nxy=nxtile*nytile
            X=np.arange(nxtile)
            Y=np.arange(nytile)
            X2,Y2 = np.meshgrid(X,Y,indexing='ij')
            # print('X2 size:',np.shape(arraya),np.shape(X2)) : these are same size with ij indexing
            XX = X2.T.flatten()
            YY = Y2.T.flatten()
            A = np.c_[XX,YY, np.ones(nxy)]
            ZZ = array1.T.flatten()
            print('sizes:',nxa,nya,np.shape(ZZ),nxy,np.shape(ZZ))
            C,_,_,_ = scipy.linalg.lstsq(A,ZZ)    # coefficients
            Z2 = C[0]*X2 + C[1]*Y2 + C[2]
            detrenda = array1- Z2
            ZZ = array2.T.flatten()
            C,_,_,_ = scipy.linalg.lstsq(A, ZZ)    # coefficients
            Z2 = C[0]*X2 + C[1]*Y2 + C[2]
            detrendb = array2-Z2
        else: 
            detrenda=array1
            detrendb=array2
        arrayad[ix1:ix2+1,iy1:iy2+1]=detrenda
        arraybd[ix1:ix2+1,iy1:iy2+1]=detrendb
       
        
        ### 2. Work over 1 tile ------------------------------ 
        tile_centered=detrenda-np.mean(detrenda.flatten())
        tile_by_windows = (tile_centered)*hanningxy

        # 
        tileFFT1 = np.fft.fft2(tile_by_windows,norm="forward")
        tileFFT1_shift = np.fft.fftshift(tileFFT1)
        #Eta_all[:,:,im] = (abs(tileFFT1_shift)**2) *normalization
        Eta[:,:] = Eta[:,:] + (abs(tileFFT1_shift)**2) *normalization #          % sum of spectra for all tiles

        tile_centered=detrendb-np.mean(detrendb.flatten())
        tile_by_windows = (tile_centered)*hanningxy

        tileFFT2 = np.fft.fft2(tile_by_windows,norm="forward")
        tileFFT2_shift = np.fft.fftshift(tileFFT2)#
        #Etb_all[:,:,im] = (abs(tileFFT2_shift)**2) *normalization
        Etb[:,:] = Etb[:,:] + (abs(tileFFT2_shift)**2) *normalization #          % sum of spectra for all tiles

        phase=phase+(tileFFT2_shift*np.conj(tileFFT1_shift))*normalization
        phases[:,:,im]=tileFFT2_shift*np.conj(tileFFT1_shift)/(abs(tileFFT2_shift)*abs(tileFFT1_shift)); 

# Now works with averaged spectra
   
    Eta=Eta/mspec
    Etb=Etb/mspec
    coh=abs((phase/mspec)**2)/(Eta*Etb)      # spectral coherence
    ang=np.angle(phase)
    crosr=np.real(phase)/mspec
    angstd=np.std(np.angle(phases),axis=2)

    return Eta,Etb,ang,angstd,coh,crosr,phases,kx2,ky2,dkxtile,dkytile,arrayad,arraybd


