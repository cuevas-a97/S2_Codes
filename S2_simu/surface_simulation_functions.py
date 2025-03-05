#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
from wave_physics_functions import *
import scipy.special as sps # function erf
import scipy.interpolate as spi # function griddata
import numpy as np
import xarray as xr    # only used for filters: this dependency should be removed
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from numpy import squeeze

def surface_1D_from_Z1kx(Z1,kX,i,nx=None,dx=None,dkx=None):
	if nx is None:
		nx = len(Z1)

	if (dx is None):
		if dkx is None:
			dx = 2*np.pi/((kX[1] - kX[0])*nx)
		else:
			dx = 2*np.pi/(dkx*nx)
			
	if (dkx is None):
		dkx = 2*np.pi/(dx*nx)

	rng = np.random.default_rng(i)
	rg = rng.uniform(low=0.0, high=1.0, size=(nx))
	zhats=np.fft.ifftshift(np.sqrt(2*Z1*dkx)*np.exp(1j*2*np.pi*rg))
	kx0=np.fft.ifftshift(kX)
    
	S1_r = np.real(np.fft.ifft(zhats,norm="forward"))
	S1_i = np.imag(np.fft.ifft(zhats,norm="forward"))

	X = np.arange(0,nx*dx,dx) # from 0 to (nx-1)*dx with a dx step
	
	return S1_r,S1_i,X,dkx
	
def surface_2D_from_Z1kxky(Z1,kX,kY,i,nx=None,ny=None,dx=None,dy=None, dkx=None, dky=None, phase_type='uniform', verbose=False):
	# /!\ Watch out : shape(S) = (ny,nx)
	# usually when doing X,Y=np.meshgrid(x,y) with size(x)=nx and size(y)=ny => size(X)=size(Y)= (ny,nx)
	kX0 = np.unique(kX)
	kY0 = np.unique(kY)
	if nx is None:
		nx = len(kX0)
	if ny is None:
		ny = len(kY0)
	shx = np.floor(nx/2-1)
	shy = np.floor(ny/2-1)
	if verbose: 
		print('from vec kX0, dkx = ',(kX0[1] - kX0[0]))
		print('from vec kY0, dky = ',(kY0[1] - kY0[0]))
	if (dx is None):
		if dkx is None:
			dx = 2*np.pi/((kX0[1] - kX0[0])*nx)
		else:
			dx = 2*np.pi/(dkx*nx)

	if (dy is None):
		if dky is None:
			dy = (2*np.pi/((kY0[1] - kY0[0])*ny))
		else:
			dy = 2*np.pi/(dky*ny)

	if (dkx is None):
		dkx = 2*np.pi/(dx*nx)
	if (dky is None):
		dky = 2*np.pi/(dy*ny)

	if verbose:    
		print("variables : ")
		print(' dx = ',dx,' ; dy = ',dy,' ; nx = ',nx,' ; ny = ',ny)
		print('dkx = ',dkx,' ; dky = ', dky)
	########################################################################################
	# SIDE NOTE :                                                                          #
	# obtain dkx, dky from dx and dy in order to account for eventual rounding of np.pi	   #
	# considering that dkx has been defined according to the surface requisite (nx,dx)     #
	# Eg:	                                                                           #
	# # initialisation to compute spectrum                                                 #
	# nx = 205										#
	# dx = 10										#
	# dkx = 2*np.pi/(dx*nx)								#
	# kX0 = dkx*np.arange(-nx//2+1,nx//2+1)						#
	# 											#
	# # Compute from kX0 and nx (found from kX0.shape)					#
	# dkx2 = kX0[1] - kX0[0]								#
	# dxbis = (2*np.pi/(dkx2*nx)                                                           #
	# dkx3 = 2*np.pi/(dxbis*nx)								#
	#											#
	# print('dkx = ',dkx)     		=> 0.0030649684425266273			#
	# print('dkx2 = ',dkx2) 		=> 0.0030649684425266277			#
	# print('dkx3 = ',dkx3) 		=> 0.0030649684425266273                           #
	#                                                                                      #
	########################################################################################
	rng = np.random.default_rng(i)
	if phase_type=='uniform':
		rg = rng.uniform(low=0.0, high=1.0, size=(ny,nx))
	else:
		rg = rng.normal(0,1,(ny,nx))
	zhats=np.fft.ifftshift(np.sqrt(2*Z1*dkx*dky)*np.exp(1j*2*np.pi*rg))
	ky2D=np.fft.ifftshift(kY) 
	kx2D=np.fft.ifftshift(kX) 

	#     real part
	S2_r = np.real(np.fft.ifft2(zhats,norm="forward"))
	#     also computes imaginary part (useful for envelope calculations) 
	S2_i = np.imag(np.fft.ifft2(zhats,norm="forward"))

	X = np.arange(0,nx*dx,dx) # from 0 to (nx-1)*dx with a dx step
	Y = np.arange(0,ny*dy,dy)

	return S2_r,S2_i,X,Y,kX0,kY0,i,dkx,dky	

def surface_from_Efth(Efth,f_vec,th_vec,seed=0,nx=2048,ny=2048,dx=10,dy=10,D=None):
	import xarray as xr
	g=9.81
	#
	# Here we start by turning the spectrum to cartesian coordinates in order 
	# to apply the Inverse Fourier to it and get the surface                   
	#
	# -- get cartesian spectrum ------
	Ekxky0, kx, ky = spectrum_from_fth_to_kxky(Efth,f_vec,th_vec)

	# -- GET THE INTERPOLATION GRID for the spectrum -----
	# -- get kx,ky values we want from the surface we want ---------
	dkx = 2*np.pi/(dx*nx)
	dky = 2*np.pi/(dy*ny)
	kX0 = np.fft.fftshift(np.fft.fftfreq(nx+1,d=dx))*2*np.pi
	kY0 = np.fft.fftshift(np.fft.fftfreq(ny+1,d=dy))*2*np.pi

	kX,kY = np.meshgrid(kX0,kY0)# , indexing='ij')

	# --- compute associated Kf, Phi(in deg) ---------
	kK = (np.sqrt(kX**2+kY**2))
	kF = f_from_k(kK,D=D)

	kPhi = np.arctan2(kY,kX)*180/np.pi
	kPhi[kPhi<0]=kPhi[kPhi<0]+360

	# create a dataArray with the new (i.e. wanted) values of F written in a cartesian array
	kF2 = xr.DataArray(kF, coords=[("ky", kY0), ("kx",kX0)])

	kPhi2 = xr.DataArray(kPhi, coords=[("ky", kY0), ("kx",kX0)])
	FPhi2s = xr.Dataset(
	{'kF': kF2,
	'kPhi': kPhi2}
	).stack(flattened=["ky", "kx"])

	Ekxky = xr.DataArray(Ekxky0, dims=("nf", "n_phi"), coords={"nf":f_vec , "n_phi":th_vec})
	B = Ekxky.interp(nf=FPhi2s.kF,n_phi=FPhi2s.kPhi,kwargs={"fill_value": 0})
	B.name='Ekxky_new'
	B0 = B.reset_coords(("nf","n_phi"))
	Ekxky_for_surf = B0.Ekxky_new.unstack(dim='flattened')

	S2_r,S2_i,X,Y,kX0,kY0,rg,dkx,dky = surface_2D_from_Z1kxky(Ekxky_for_surf,kX,kY,seed)

	return S2_r,S2_i,X,Y,rg,kX0,kY0,Ekxky_for_surf,dkx,dky

##################################################################################
def def_spectrumG_for_surface_1D(nx=2048,dx=10,T0=10,Hs=4,sk_k0=0.1,D=None,verbose=False):
	dkx = 2*np.pi/(dx*nx)

	kX = np.fft.fftshift(np.fft.fftfreq(nx,d=dx))*2*np.pi
	
	# --- only Gaussian -------------		
	Z1_Gaussian,kX,sk = Gaussian_1Dspectrum_kx(kX,T0,sk_k0,D=D)
	Z1 =(Hs/4)**2*Z1_Gaussian
	sumZ1=4*np.sqrt(sum(Z1.flatten()*dkx)) 
	if verbose:
		print('Hs for Gaussian : ',sumZ1)
	
	return Z1, kX,sk

def def_spectrumPM_for_surface_1D(nx=2048,dx=10,T0=10,D=None,verbose=False): #Hs=4,sk_k0=0.1
	dkx = 2*np.pi/(dx*nx)
	
	kX = np.fft.fftshift(np.fft.fftfreq(nx,d=dx))*2*np.pi
	
	# --- only PM -------------	
	Z1_PM = PM_spectrum_k(kX,1/T0,D=D)	
	# Z1_Gaussian,kX,sk = Gaussian_1Dspectrum_kx(kX,T0,sk_k0,D=D)
	# Z1 =(Hs/4)**2*Z1_Gaussian
	sumZ1=4*np.sqrt(sum(Z1_PM[np.isfinite(Z1_PM)].flatten()*dkx)) 
	if verbose:
		print('Hs for Gaussian : ',sumZ1)
	Z1_PM[np.isnan(Z1_PM)]=0
	return Z1_PM, kX
	
def def_spectrumJONSWAP_for_surface_1D(nx=2048,dx=10,T0=10,D=None,gammafac=3.3,sigA=0.07,sigB=0.09,verbose=False):
	dkx = 2*np.pi/(dx*nx)
	
	kX = np.fft.fftshift(np.fft.fftfreq(nx,d=dx))*2*np.pi
	fX = sig_from_k(kX,D=D)/(2*np.pi)
	Z1_PM = PM_spectrum_k(kX,1/T0,D=D)
	Z1_PM[np.isnan(Z1_PM)]=0
	fp = 1/T0
	sigAB = np.where(fX<fp,sigA,sigB)
	JSfactor = gammafac**np.exp((-(fX-fp)**2)/(2*sigAB**2*fp**2))
	Z1_JS = Z1_PM*JSfactor
	Z1_JS[np.isnan(Z1_JS)] = 0
	sumZ1=4*np.sqrt(sum(Z1_JS[np.isfinite(Z1_JS)].flatten()*dkx)) 
	if verbose:
		sumZ1=4*np.sqrt(sum(Z1_JS[np.isfinite(Z1_JS)].flatten()*dkx)) 
		print('Hs for Jonswap : ',sumZ1)
	return Z1_JS, kX

def def_spectrum_for_surface(nx=2048,ny=2048,dx=10,dy=10,theta_m=30,D=1000,T0=10,Hs=4,sk_theta=0.001,sk_k=0.001,\
nk=1001,nth=36,klims=(0.0002,0.2),n=4,typeSpec='Gaussian',verbose=False):
	dkx = 2*np.pi/(dx*nx)
	dky = 2*np.pi/(dy*ny)

	kX0 = np.fft.fftshift(np.fft.fftfreq(nx,d=dx))*2*np.pi
	kY0 = np.fft.fftshift(np.fft.fftfreq(ny,d=dy))*2*np.pi
	kX,kY = np.meshgrid(kX0, kY0)
	
		
	if typeSpec=='Gaussian':
		if verbose:
			print('Gaussian spectrum selected. Available options:\n - Hs, sk_theta, sk_k. \nWith (sk_k, sk_theta) the sigma values for the k-axis along the main direction and perpendicular to it respectively \n Other options (common to all spectrum types) are : nx, ny, dx, dy, T0, theta_m, D')
		
		Z1_Gaussian0,kX,kY = define_Gaussian_spectrum_kxky(kX,kY,T0,theta_m*np.pi/180,sk_theta,sk_k,D=D)
		
		Z1 =(Hs/4)**2*Z1_Gaussian0/np.sum(Z1_Gaussian0.flatten()*dkx*dky)
		sumZ1=4*np.sqrt(sum(Z1.flatten()*dkx*dky))
		if verbose: 
			print('Hs for Gaussian : ',sumZ1)
	
	elif typeSpec=='PM':
		if verbose:
			print('Pierson-Moskowitz* cos(theta)^(2*n) spectrum selected. Available options:\n - nk, nth, klims, n. \nWith n the exponent of the directional distribution: cos(theta)^(2*n)\n Other options (common to all spectrum types) are : nx, ny, dx, dy, T0, theta_m, D')
		k=np.linspace(klims[0],klims[1],nk)
		thetas=np.linspace(0,360*(nth-1)/nth,nth)*np.pi/180.

		Ekth,k,th = define_spectrum_PM_cos2n(k,thetas,T0,theta_m*np.pi/180.,D=D,n=n)
		Ekxky, kx, ky = spectrum_to_kxky(1,Ekth,k,thetas,D=D)

		Z1=spi.griddata((kx.flatten(),ky.flatten()),Ekxky.flatten(),(kX,kY),fill_value=0)
		sumZ1=4*np.sqrt(sum(Z1.flatten()*dkx*dky)) 
		if verbose:
			print('Hs for Pierson Moskowitz : ',sumZ1)

	return Z1, kX, kY,dkx,dky

def buoy_spectrum2d(a0,a1,a2,b1,b2, ndirs):
    # % This function calculates the Maximum Entropy Method estimate of
    # % the Directional Distribution of a wave field from buoy data (netcdf file)

    nfreq = np.size(a0)
    dr=np.pi/180
    dtheta=360/ndirs
    dirs=np.arange(0.,ndirs,1.)*dtheta
    #print(nfreq,ndirs,np.size(dirs),dirs)
    
    c1 = a1+1j*b1
    c2 = a2+1j*b2
    p1 = (c1-c2*np.conj(c1))/(1.-abs(c1)**2)
    p2 = c2-c1*p1
    
    # numerator(2D) : x
    x = 1.-p1*np.conj(c1)-p2*np.conj(c2)
    x=np.tile(np.real(x),(ndirs,1)).T
    # x = np.tile(np.real(x),(ndirs,1)).T
    
    # denominator(2D): y
    a = dirs*dr
    e1 = np.tile(np.cos(a)-1j*np.sin(a),(nfreq,1))
    e2 = np.tile(np.cos(2*a)-1j*np.sin(2*a),(nfreq,1))
    
    p1e1 = np.tile(p1,(ndirs,1)).T*e1
    p2e2 = np.tile(p2,(ndirs,1)).T*e2
    
    y = abs(1-p1e1-p2e2)**2
    
    D = x/(y)
    
    # normalizes the spreading function,
    # so that int D(theta,f) dtheta = 1 for each f  
    tot = np.tile(np.sum(D, axis=1),(ndirs,1)).T
    D = D/tot
    
    sp2d = np.tile(a0,(ndirs,1)).T*D/(dr*dtheta)
    
    return sp2d,D,dirs    
    
#############
# Images simulation from a buoy spectrum
#############

def wavespec_Efth_to_Ekxky(eft1s,fren,dfreq,dirn,dth,depth=3000.,dkx=0.0001,dky=0.0001,nkx=250,nky=250,doublesided=1,verbose=0,doplot=0,trackangle=0):
    '''
    Converts E(f,theta) spectrum from buoy or model to E(kx,ky) spectrum similar to image spectrum
    2023/11/14: preliminary version, assumes dfreq is symmetric (not eaxctly true with WW3 output and waverider data) 
    inputs :
            - etfs1 : spectrum
    output : 
            - Ekxky: spectrum
            - kx: wavenumber in cycles / m  
    '''
    [nf,nt]=np.shape(eft1s)
    tpi=2*np.pi
    grav=9.81

# makes a double sided spectrum
    if doublesided == 1:
        eftn=0.5*(eft1s+np.roll(eft1s,nt//2,axis=1))
    else: 
        eftn=eft1s
    Hs1 = 4*np.sqrt(np.sum(np.sum(eftn,axis=1)* dfreq)*dth)
# wraps around directions
    dlast=dirn[0]+360.
    dirm=np.concatenate([dirn,[dlast]])
    elast=eftn[:,0]
    eftm1=np.concatenate([eftn.T,[elast]]).T
# adds zero energy in a low frequency to avoid interpolation across k=0
    ffirst=fren[0]-0.9*(fren[1]-fren[0])
    frem=np.concatenate([[ffirst],fren])
    efirst=eftm1[0,:]*0
    eftm=np.concatenate([[efirst],eftm1])

#plt.pcolormesh(fren, dirm, np.log10(eftm).T)
    km=(2*np.pi*frem)**2/(grav*2*np.pi)   # cycles / meter
    for ii in range(nf):
       km[ii]=k_from_f(frem[ii],D=depth)/(2*np.pi)            # finite water depth
    km2=np.tile(km.reshape(nf+1,1),(1,nt+1))

# eftn*df*dth = Ek*k*dk*dth -> Ek = efth *df /(k * dk)  =  efth *Cg /k
    Cg2=np.sqrt(grav/(km2*tpi))*0.5
    Jac=Cg2/km2
    dirm2=np.tile(dirm.T,(nf+1,1))*np.pi/180.
    kxn=km2*np.cos(dirm2+trackangle)
    kyn=km2*np.sin(dirm2+trackangle)
    #plt.scatter(kxn,kyn,  marker='.', s = 20)
    kx=np.linspace(-nkx*dkx,(nkx-1)*dkx,nkx*2)
    ky=np.linspace(-nky*dky,(nky-1)*dky,nky*2)
    kx2, ky2 = np.meshgrid(kx,ky,indexing='ij')   #should we transpose kx2 and ky2 ???
    Ekxky = griddata((kxn.flatten(), kyn.flatten()), (eftm*Jac).flatten(), (kx2, ky2), method='nearest')
    Hs2=4*np.sqrt(np.sum(np.sum(Ekxky))*dkx*dky)
# make sure energy is exactly conserved (assuming kmax is consistent with fmax
    print('Hs1,Hs2:',Hs1,Hs2)
    Ekxky = Ekxky * (Hs1/Hs2)**2
    
    return Ekxky,kx,ky,kx2,ky2



def S2_simu(Efth,freq,df,dir2,dth,fac,facr,na,nt,ntime,dti,plotb,dx,nx,iseed,U10,Udir,Ux,Uy,betad,thetad):
    # S2_simu(Efth0,fb,dfb,dirs,dth ,fac,facr,na,nt,ntime,dt,plotb,dx,nx,iseed,U10,Udir,Ux,Uy,betad,thetad);
    
#   dx is pixel size       (NB: dy=dx)
#   nx is number of pixels (NB: nx=ny)
#   fac is the mean image value  
#   facr is the reflection coefficient used to impose reflections: if zero, full 2D spectrum is used
#   na: additive noise parameter
#   nt: multiplicative noise parameter
#   ntime: number of images 
#  dti: time step between images

    # Define constant
    d2r=np.pi/180
    nxp=(nx-1)/8+1
    nxp=int(nxp)

    print('nxp',nxp)
    
    dy=dx
    ny=nx

    # Defines wind and current vectors (parameters of the function in future version)
    Udir=Udir*d2r # direction to, trig. convention


    #Defines sun and sensor angles (should change with the different times)
    beta = betad * d2r
    thetav = thetad * d2r
    
    # Angles d'azimut trigonométriques: this should be a parameter passed to the function ... 
    phitrig = [148.1901, 148.8061, 149.1342, 149.4561]
    
    # Azimut de la pente spéculaire
    phip = phitrig[1] * d2r


    #-----------------------------------------------------
    # Computes maps of the surface elevation
    #-----------------------------------------------------

    # surface size
    nkx=nx*2   # Takes a bigger surface to avoid periodic funny boundary effects ... 
    nky=nx*2

    dkx = 2 * np.pi / (dx * nkx)
    dky = 2*np.pi/(dy*nky)

    kX0 = np.fft.fftshift(np.fft.fftfreq(nkx,d=dx))*2*np.pi
    kY0 = np.fft.fftshift(np.fft.fftfreq(nky,d=dy))*2*np.pi


    kX,kY = np.meshgrid(kX0,kY0)# , indexing='ij')
    kK = (np.sqrt(kX**2+kY**2))
    kPhi = np.arctan2(kY,kX)*180/np.pi
    kF = kK
    kPhi[kPhi<0]=kPhi[kPhi<0]+360

    # # create a dataArray with the new (i.e. wanted) values of F written in a cartesian array
    # kF2 = xr.DataArray(kF, coords=[("ky", kY0), ("kx",kX0)])

    # kPhi2 = xr.DataArray(kPhi, coords=[("ky", kY0), ("kx",kX0)])
    # FPhi2s = xr.Dataset(
    # {'kF': kF2,
    # 'kPhi': kPhi2}
    # ).stack(flattened=["ky", "kx"])

    print('nkx:',nkx,nky) 
    Ekxky,kx,ky,kx2T,ky2T = wavespec_Efth_to_Ekxky(Efth,freq,df,dir2,dth,dkx=dkx/(2*np.pi),dky=dky/(2*np.pi),nkx=nkx,nky=nky,doublesided=0) 

    kx2=kx2T.T
    ky2=ky2T.T 
    
    Ekxky_for_surf=Ekxky/(2*np.pi)**2
    Etot=np.sum(Ekxky_for_surf.flatten())*dkx*dky
    # compared to eq. 16 in De Carlo et al. (2023) the factor 0.5 corrects for single sided spec
    Qkk=np.sqrt(np.sum(Ekxky_for_surf.flatten()**2)*dkx*dky*0.5)/Etot

# computes Hs and Qkk
    Hskk=4*np.sqrt(Etot)
    print('Testing Hs:',Hskk,np.shape(Ekxky),nkx,nky) 

    plt.figure(1)
    plt.get_cmap('viridis')  # Utilisez 'gray' pour la carte de couleurs
    plt.title('Wave spectrum interpolated in kx,ky plane')
    plt.clf()  # Equivalent à 'clf'
    
    # Plotting convention is direction from
    pfac=1000
    plt.pcolor(kx2*pfac, ky2*pfac, 10 * np.log10(Ekxky_for_surf))   #, shading='flat')
    plt.colorbar()  # Ajouter une barre de couleur

    
    # Utiliser une alternative à `set_renderer` pour le rendu 'painters'
    fig = plt.gcf()  # Obtenez la figure actuelle
    fig.set_dpi(150)  # Réglage de la résolution
    fig.set_size_inches(3, 3)  # Ajuster la taille de la figure
    
    for i in range(1, 8):
        plt.plot(10 * i * np.cos(np.linspace(0, 2 * np.pi, 49)), 10 * i * np.sin(np.linspace(0, 2 * np.pi, 49)), 'k-', linewidth=1)
    
    # Ajouter les lignes de référence
    plt.plot([-60, 60], [0, 0], 'k-', linewidth=1)
    plt.plot([0, 0], [-60, 60], 'k-', linewidth=1)
        
    # Ajuster les axes
    plt.axis('equal')
    plt.axis([-50, 50, -50, 50])
    plt.xlabel('k_x / 2 \pi (counts per km)')
     # plt.caxis([20, 70])
        
    plt.show()
        
    Efths = np.roll(Efth, shift=36, axis=1)  # Equivalent à circshift avec [0 36]

    Hspec = 4 * Efth * Efths / ((Efth + Efths)**2)
        

    # -------------------------Random draw of phases....
    phases = np.random.rand(nky*2, nkx*2) * 2 * np.pi   # WARNING: is this ny,nx or nx,ny ? 
    i1=nkx//2
    i2=i1+nkx
  

    rng = np.random.default_rng(iseed)
    rg = rng.uniform(low=0.0, high=1.0, size=(nky*2,nkx*2))
    zhats=np.fft.ifftshift(np.sqrt(2*Ekxky_for_surf*dkx*dky)*np.exp(1j*2*np.pi*rg))
    # ky2D=np.fft.ifftshift(ky2) 
    # kx2D=np.fft.ifftshift(kx2) 

   # 28/01/2025 
    kx2D=np.fft.ifftshift(ky2) 
    ky2D=np.fft.ifftshift(kx2) 
    print('kx2D',kx2D)

             
    kN=np.sqrt(kx2D**2+ky2D**2)
    grav=9.81
    
    si2s=np.sqrt(grav*kN*2*np.pi) 
    
    # plt.figure
    
    # plt.plot(si2s[0,:],abs(zhats[0,:]),c='k')
    # #plt.plot(ky2D[:,0],abs(zhats[:,0]),c='r')
    # #plt.plot(ky2D[0,:],abs(zhats[0,:]),c='m')
    # #plt.plot(kx2D[:,0],abs(zhats[:,0]),c='b')
    # plt.xlim([-0.8,0.8])

    
    xp = np.linspace(0, dx * nxp, nxp)
    x2 = np.tile(xp, (nxp, 1))
    y2 = np.tile(xp.reshape(-1, 1), (1, nxp))
    yp = xp


    t = np.linspace(0, (ntime - 1) * dti, ntime)

    # Set colormap to gray
    plt.set_cmap("viridis")    
    # Create a figure
    nfig = 30
    plt.figure(nfig, figsize=(3 * nxp / 100, 3 * nxp / 100))
    plt.clf()
    
    
    # Initialize variables
    allB = np.zeros((nx, nx, ntime))
    allz = np.zeros((nx, nx, ntime))
    mssx = 0.001 + 0.00316 * U10
    mssy = 0.003 + 0.00185 * U10
    mss = mssx + mssy
    sx0 = -np.tan(beta) * np.sin(phip)  # slope without long wave effect
    sy0 = -np.tan(beta) * np.cos(phip)


    allind = (nx * 2) * 10 + 10 + np.arange(1, nx ** 2 + 1)
    for ii in range(1, nx - 1):
        allind[ii * nx:] += nx
    
    choppy = 0 


    for i in range(ntime):
        # Crée ou récupère l'image de la trame
        # Exemple : Créer une trame vide
        frame = np.zeros((ny, nx, 3), dtype=np.uint8)

    ######### Loop on the time #########
    tpi=2*np.pi   # this factor is needed because the kx2,ky2 are in cycles / m not in rad/m
    for ii in range(ntime):
    
        t1 = t[ii]
        #phasor = np.exp(-1j * (si2s - (kx2*tpi) * Ux - (ky2*tpi) * Uy) * t1)
        phasor = np.exp(-1j * (si2s -( kx2D*tpi*Ux + ky2D*tpi*Uy)) * t1)
        # Compute zeta1
        inds=np.where(np.abs(zhats[0,:]) > 0.01)[0]
        # print('ii:',ii,t1,inds,'##',si2s[0,inds]*t1,np.real(np.exp(-1j*si2s[0,inds]*t1)) )
        
        zeta1= np.real(np.fft.ifft2(zhats*phasor,norm="forward"))
        # print('phasor:',np.real(phasor[0,inds]),'##',np.real(zhats[0,inds]*phasor[0,inds]))
        
        # print('time=',t1,', checking Hs: 4*std of zeta1:',4*np.std(zeta1.flatten()))
# CHOPPY NOT WORKING NOW ... 
        if choppy == 1:
            Dx = np.real(ifft2(zhats * 1j * kx2s * phasor / kns)) * (nkx**2) / dx
            Dy = np.real(ifft2(zhats * 1j * ky2s * phasor / kns)) * (nkx**2) / dx
            
            iDx = np.floor(Dx).astype(int)
            iDy = np.floor(Dy).astype(int)
            wDx = Dx - iDx
            wDy = Dy - iDy
        
            zetachoppy = (
                (1 - wDx[allind]) * (
                    zeta1[allind + iDx[allind] * (nx * 2) + iDy[allind]] * (1 - wDy[allind]) +
                    zeta1[allind + iDx[allind] * (nx * 2) + iDy[allind] + 1] * wDy[allind]
                )
                + wDx[allind] * (
                    zeta1[allind + (iDx[allind] + 1) * (nx * 2) + iDy[allind]] * (1 - wDy[allind]) +
                    zeta1[allind + (iDx[allind] + 1) * (nx * 2) + iDy[allind] + 1] * wDy[allind]
                )
            )

            zeta = zetachoppy
        else:
            zeta = zeta1[:nkx:2, :nky:2]
        
        # Slope computation
        
        sx1 = np.real(np.fft.ifft2(zhats * (1j*tpi) * kx2D *phasor,norm="forward"))
        sy1 = np.real(np.fft.ifft2(zhats * (1j*tpi) * ky2D *phasor,norm="forward"))
        
        sx = sx1[:nky:2, :nkx:2]
        sy = sy1[:nky:2, :nkx:2]
        print('dx,dy:',dx,dy)
        dx2=dx/2
        dy2=dy/2
        x=np.linspace(0,dx*(nx-1),nx)
        y=np.linspace(0,dy*(ny-1),ny)

        
        # Adds slope of bistatic look direction + rotate in wind direction
        sxt = (sx + sx0) * np.cos(Udir) + (sy + sy0) * np.sin(Udir)
        syt = (sy + sy0) * np.cos(Udir) - (sx + sx0) * np.sin(Udir)
        sx0t = sx0 * np.cos(Udir) + sy0 * np.sin(Udir)
        sy0t = sy0 * np.cos(Udir) - sx0 * np.sin(Udir)
        
        # Backscatter coefficient computation
        norma = (2 * np.pi * np.sqrt(mssx * mssy))
        B = np.exp(-0.5 * (sxt**2 / mssx + syt**2 / mssy)) / (np.cos(beta)**4 * np.cos(thetav))
        B0 = np.exp(-0.5 * (sx0t**2 / mssx + sy0t**2 / mssy))


        # Store results
        allz[:, :, ii] = zeta
        allB[:, :, ii] = B / B0
        allB[:, :, ii] = zeta
    
        #something to check on
        if ii==0:
            np.mean(np.std(B))/np.mean(np.mean(B))
        
        if plotb == 1 and ii == 0:
            # Plot zeta
            # print('x', np.shape(x))
            # print('zeta[:,0]', np.shape(zeta[:, 0]))
            # print('zeta[10:nx+10,10]', np.shape(zeta1[10:nx+10, 10]))
            
            plt.figure(101)
            plt.plot(x, zeta[:, 0], 'k-', x, zeta1[10:nx+10, 10], 'r-')
            plt.plot(x,B[:, 0], 'g-')
            
            # Image zeta
            plt.figure(nfig)
            plt.clf()  # Nettoyer la figure précédente
            plt.gcf().set_size_inches((1 * nxp + 300) / 100, (1 * nxp + 200) / 100)  # Diviser par 100 pour convertir en pouces 
            # Afficher zeta
            img = plt.imshow(
                np.fliplr(zeta[:nxp, :nxp]).T, 
                extent=[0, 1000, 0, 1000], 
                vmin=-1, vmax=1, 
                cmap='viridis'
            )
            
            plt.colorbar(img)
            
            # Configurer les axes
            plt.axis([0, 1000, 0, 1000])
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            

        if plotb == 1:
            # Plots B
            plt.figure(nfig + 2)
            plt.clf()
            plt.imshow(np.fliplr(B[:nxp, :nxp] / B0).T, extent=[0, 1000, 0, 1000], cmap=plt.get_cmap('gray'))
            
            plt.colorbar()
            plt.axis('equal')
            plt.axis([0, 1000, 0, 1000])
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.gcf().set_size_inches((1 * nxp + 300) / 100, (1 * nxp + 200) / 100)  # Diviser par 100 pour convertir en pouces
            
    
    ind1, ind2, ind3, ind4 = 0, 2, 3, 4  
    # print('ind:',ind1,ind4)
    # Generate images based on backscatter coefficient and noise

    print('allB',np.shape(allB))
    img1 = np.round((nt * np.random.rand(nx, nx) + 1 - nt / 2) * fac * (squeeze(allB[:, :, ind1]) * (1 - na / 2) + na * np.random.rand(nx, nx)))
    img2 = np.round((nt * np.random.rand(nx, nx) + 1 - nt / 2) * fac * (squeeze(allB[:, :, ind2]) * (1 - na / 2) + na * np.random.rand(nx, nx)))
    img3 = np.round((nt * np.random.rand(nx, nx) + 1 - nt / 2) * fac * (squeeze(allB[:, :, ind3]) * (1 - na / 2) + na * np.random.rand(nx, nx)))
    img4 = np.round((nt * np.random.rand(nx, nx) + 1 - nt / 2) * fac * (squeeze(allB[:, :, ind4]) * (1 - na / 2) + na * np.random.rand(nx, nx)))
        
    # Image times
    imgtimes = t[[ind1, ind2, ind3, ind4]] - t[ind1]

    return img1, img2, img3, img4, imgtimes, phitrig, nx,ny,x,y,dx,dy


