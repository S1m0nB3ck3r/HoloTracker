# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
from numpy.fft import fft2 as np_fft2
from numpy.fft import ifft2 as np_ifft2
from numpy.fft import fftshift as np_fftshift
from numpy.fft import ifftshift as np_ifftshift

import cupy as cp
from cupy.fft import rfft2, fft2, ifft2, fftshift, ifftshift, fftn, ifftn
from cupyx import jit
from cupyx.scipy import ndimage as cp_ndimage
import typeHolo
from traitement_holo import *


@jit.rawkernel()
def d_calc_phase(d_plan_complex, d_phase, size_x, size_y):

    #### Fonction qui ne marche pas
    ### impossible d'extraire la partie imaginaire d'un complexe dans un kernel jit.rawkernel

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    sizeXY = size_x * size_y

    jj = index // size_x
    ii = index - jj * size_x

    if (ii < size_x and jj < size_y):
        cplx = cp.complex64(d_plan_complex[ii, jj])
        r = cp.real(cplx)
        if (r == 0.0):
            d_phase[ii, jj] = 0.0
        elif(cp.real(cplx) > 0.0):
            d_phase[ii, jj] = cp.arctan(cp.imag(cplx) / cp.real(cplx))
        else:
            d_phase[ii, jj] = cp.pi + cp.arctan(cp.imag(cplx) / cp.real(cplx))

#########################################################################################################################################
################################               traitements des KERNELS               ####################################################
#########################################################################################################################################


@jit.rawkernel()
def d_filter_FFT(d_plan_IN, d_plan_OUT, sizeX, sizeY, dMin, dMax):
    #### Filtre passe bande   dMin>F>dMax avec dMin et dMax en pixel
    ### d_plan_IN et d_plan_OUT des plans complex sur GPU de taille sizeX et sizeY 

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    sizeXY = sizeX * sizeY

    jj = cp.int32(index) // cp.int32(sizeX)
    ii = cp.int32(index) - cp.int32(jj * sizeY)

    if (ii < sizeX and jj < sizeY):
        #calc distance
        centreX = sizeX // 2
        centreY = sizeY // 2

        distanceCentre = cp.sqrt((centreX - ii)*(centreX - ii) + (centreY - jj)*(centreY - jj))

        if ((distanceCentre > dMin) and (distanceCentre < dMax )):
            d_plan_OUT[jj, ii] = d_plan_IN[jj, ii]
        else:
            d_plan_OUT[jj, ii] = 0.0 + 0.0j


@jit.rawkernel()
def d_spec_filter_FFT(d_plan_IN, d_plan_OUT, sizeX, sizeY, dMin, dMax):

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    sizeXY = sizeX * sizeY

    jj = cp.int32(index) // cp.int32(sizeX)
    ii = cp.int32(index) - cp.int32(jj * sizeY)

    if (ii < sizeX and jj < sizeY):
        #calc distance
        centreX = sizeX // 2
        centreY = sizeY // 2

        distanceCentre = cp.sqrt((centreX - ii)*(centreX - ii) + (centreY - jj)*(centreY - jj))

        if (dMin != 0 and distanceCentre < dMin) or (dMax != 0 and distanceCentre > dMax):
            d_plan_OUT[jj, ii] = 0.0 + 0.0j
        else:
            d_plan_OUT[jj, ii] = d_plan_IN[jj, ii]



        # if ((distanceCentre > dMin) and (distanceCentre < dMax )):
        #     #d_plan_OUT[ii, jj] = d_plan_IN[ii, jj] * cp.log(1 + (distanceCentre - dMin)/ (dMax - dMin))
        #     d_plan_OUT[jj, ii] = d_plan_IN[jj, ii]
        # else:
        #     d_plan_OUT[jj, ii] = 0.0 + 0.0j


#		spec_mask_filter_device[xy] = log(1 + (R - R_low));


#########################################################################################################################################
################################               calculs des KERNELS                  #####################################################
#########################################################################################################################################

@jit.rawkernel()
def d_calc_kernel_propag_Rayleigh_Sommerfeld(d_KERNEL, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance):
    
    ### calcul du noyau de propagation d_KERNEL selon la méthode Rayleigh_Sommerfeld
    ### H_propag = FFT-1 ( FFT(HOLO) * FFT(KERNEL_RAYLEIGH_SOMMERFELD)

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    sizeXY = nb_pix_X * nb_pix_Y

    jj = cp.int32(index) // cp.int32(nb_pix_X)
    ii = cp.int32(index) - cp.int32(jj * nb_pix_X)

    if (ii < nb_pix_X and jj < nb_pix_Y):

        K = cp.float32(2.0 * cp.pi / lambda_milieu)
        X = cp.float32( cp.int32(ii) - cp.int32(nb_pix_X // 2) )
        Y = cp.float32( cp.int32(jj) - cp.int32(nb_pix_Y // 2) )
        dpix = cp.float32(pixSize / magnification)
        mod = cp.float32(distance) / cp.float32(lambda_milieu * (distance * distance + X * X *dpix * dpix + Y * Y * dpix*dpix))
        phase = cp.float32(K * cp.sqrt(distance * distance + X * X * dpix * dpix + Y * Y * dpix * dpix))
        d_KERNEL[jj, ii] = cp.complex64(mod * cp.exp(2.0j*cp.pi*phase))

    
@jit.rawkernel()
def d_calc_kernel_angular_spectrum_jit(d_KERNEL, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance):

    ### calcul du plan de phase d_KERNEL du noyau de propagation du spectre angulaire
    ###  H_propag = FFT-1 ( FFT(HOLO) * KERNEL_ANGULAR_SPECTRUM ) 

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    sizeXY = nb_pix_X * nb_pix_Y

    jj = index // nb_pix_X
    ii = index - jj * nb_pix_X

    if (ii < nb_pix_X and jj < nb_pix_Y):
        du = magnification / (pixSize * cp.float32(nb_pix_X))
        dv = magnification / (pixSize * cp.float32(nb_pix_Y))

        offset_u = nb_pix_X//2
        offset_v = nb_pix_Y//2

        U = ( cp.int32(ii) - cp.int32(offset_u) )*du
        V = ( cp.int32(jj) - cp.int32(offset_v) )*dv
        
        arg = 1.0 - cp.square(lambda_milieu * U) - cp.square(lambda_milieu *V )
        
        if(arg>0):
            d_KERNEL[jj, ii] = cp.exp(2 * 1j * cp.pi * distance * cp.sqrt(arg) / lambda_milieu)
        else:
            d_KERNEL[jj, ii] = 0.0+0j


@jit.rawkernel()
def d_propag_fresnel_phase1_jit(d_HOLO_IN, d_HOLO_OUT, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance):
    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    jj = index // nb_pix_X
    ii = index - jj * nb_pix_X

    if (ii < nb_pix_X and jj < nb_pix_Y):
        offsetX = nb_pix_X//2
        offsetY = nb_pix_Y//2

        X = ii - offsetX
        Y = jj - offsetY

        dp_X = magnification * lambda_milieu * distance / (nb_pix_X * pixSize)
        dp_Y = magnification * lambda_milieu * distance / (nb_pix_Y * pixSize)

        arg = (cp.pi * 1.0j / (lambda_milieu * distance))  * (X*X*dp_X*dp_X + Y*Y*dp_Y*dp_Y)
        mod = 1.0j * cp.exp(2.0j*cp.pi*distance/lambda_milieu)/(lambda_milieu*distance)
        d_HOLO_OUT[jj, ii]=mod * cp.exp(arg) * d_HOLO_IN[jj, ii]

@jit.rawkernel()
def d_propag_fresnel_phase2_jit(d_HOLO_IN, d_HOLO_OUT, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance):
    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    jj = index // nb_pix_X
    ii = index - jj * nb_pix_X

    if (ii < nb_pix_X and jj < nb_pix_Y):
        offsetX = nb_pix_X//2
        offsetY = nb_pix_Y//2

        X = ii - offsetX
        Y = jj - offsetY

        dp_X = pixSize / magnification
        dp_Y = pixSize / magnification

        arg = (cp.pi * 1.0j / (lambda_milieu * distance)) * (X*X*dp_X*dp_X + Y*Y*dp_Y*dp_Y)
        d_HOLO_OUT[jj, ii] = cp.exp(arg) * d_HOLO_IN[jj, ii]


#########################################################################################################################################
################################               propagations                  ############################################################
#########################################################################################################################################

def propag_angular_spectrum(d_HOLO, d_FFT_HOLO, d_KERNEL, d_FFT_HOLO_PROPAG, d_HOLO_PROPAG,
lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance, f_pix_min, f_pix_max):
    
    ### propagation de type spectre angulaire (méthode à 2 FFTs): propagation à utiliser pour les distance faibles
    ### dx et dy ne dépendent pas de la distance de propagation
    
    nthread = 1024
    nBlock = math.ceil(nb_pix_X * nb_pix_Y // nthread)
    d_FFT_HOLO = fftshift(fft2(d_HOLO, norm = 'ortho'))

    if ((f_pix_min != 0) or (f_pix_max != 0)):
        d_filter_FFT(d_FFT_HOLO, d_FFT_HOLO, nb_pix_X, nb_pix_Y, f_pix_min, f_pix_max)

    d_calc_kernel_angular_spectrum_jit[nBlock, nthread](d_KERNEL, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance)
    d_FFT_HOLO_PROPAG = d_FFT_HOLO * d_KERNEL
    d_HOLO_PROPAG = fft2(fftshift(d_FFT_HOLO_PROPAG), norm = 'ortho')
    return(d_HOLO_PROPAG)


def propag_fresnell(d_HOLO, d_HOLO_2, d_FFT, d_HOLO_PROPAG,
lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance):
    
    ### propagation de type Fresnell (méthode à 1 FFT): propagation à utiliser pour les distances grandes 
    ### dx et dy dépendent de la distance de propagation
    ### méthode pas encore testée...un peu merdique

    nthread = 1024
    nBlock = math.ceil(nb_pix_X * nb_pix_Y // nthread)
    d_propag_fresnel_phase1_jit[nBlock, nthread](d_HOLO, d_HOLO_2, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance)
    d_FFT = fftshift(fft2(d_HOLO_2))
    d_propag_fresnel_phase2_jit[nBlock, nthread](d_FFT, d_HOLO_PROPAG, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance)

def propag_Rayleigh_Sommerfeld(d_HOLO, d_FFT_HOLO, d_KERNEL, d_FFT_KERNEL, d_FFT_HOLO_PROPAG, d_HOLO_PROPAG,
lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance):
    
    ### propagation de type Rayleigh Sommerfeld (méthode à 3 FFTs): propagation à utiliser pour les distances grandes 
    ### dx et dy ne dépendent pas de la distance de propagation

    nthread = 1024
    nBlock = math.ceil(nb_pix_X * nb_pix_Y // nthread)
    d_FFT_HOLO = fftshift(fft2(d_HOLO, norm = 'ortho'))
    d_calc_kernel_propag_Rayleigh_Sommerfeld[nBlock, nthread](d_KERNEL, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance)
    d_FFT_KERNEL = fftshift(fft2(d_KERNEL, norm = 'ortho'))
    d_FFT_HOLO_PROPAG = d_FFT_HOLO * d_KERNEL
    d_HOLO_PROPAG = fft2(fftshift(d_FFT_HOLO_PROPAG), norm = 'ortho')

#########################################################################################################################################
################################               Calculs volumes               ############################################################
#########################################################################################################################################

def volume_propag_angular_spectrum_complex(d_HOLO, d_FFT_HOLO, d_KERNEL, d_FFT_HOLO_PROPAG, d_HOLO_VOLUME_PROPAG,
lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distancePropagIni, pasPropag, nbPropag, f_pix_min, f_pix_max):

    nthread = 1024
    nBlock = math.ceil(nb_pix_X * nb_pix_Y // nthread)

    d_FFT_HOLO = fftshift(fft2(d_HOLO, norm = 'ortho'))

    #print('somme avant fft:', cp.asnumpy(intensite(d_HOLO)).sum(), 'somme après FFT', cp.asnumpy(intensite(d_FFT_HOLO)).sum())

    if ((f_pix_min != 0) and (f_pix_max != 0)):
        #d_spec_filter_FFT[nBlock, nthread](d_FFT_HOLO, d_FFT_HOLO, nb_pix_X, nb_pix_Y, f_pix_min, f_pix_max)
        d_spec_filter_FFT[nBlock, nthread](d_FFT_HOLO, d_FFT_HOLO, nb_pix_X, nb_pix_Y, f_pix_min, f_pix_max)

    for i in range(nbPropag):
        distance = distancePropagIni + i * pasPropag
        d_calc_kernel_angular_spectrum_jit[nBlock, nthread](d_KERNEL, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance)
        #analyse_array_cplx(d_KERNEL)

        d_FFT_HOLO_PROPAG = d_FFT_HOLO * d_KERNEL
        #analyse_array_cplx(d_FFT_HOLO_PROPAG)
        d_HOLO_VOLUME_PROPAG[:,:,i] = fft2(fftshift(d_FFT_HOLO_PROPAG), norm = 'ortho')
        #print("\n intensité distance ", distance, " :")
        #analyse_array_cplx(d_HOLO_VOLUME_PROPAG[:,:,i])


def volume_propag_angular_spectrum_to_module(d_HOLO, d_FFT_HOLO, d_KERNEL, d_FFT_HOLO_PROPAG, d_HOLO_VOLUME_PROPAG_MODULE,
lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distancePropagIni, pasPropag, nbPropag, f_pix_min, f_pix_max):

    nthread = 1024
    nBlock = math.ceil(nb_pix_X * nb_pix_Y // nthread)

    d_FFT_HOLO = fftshift(fft2(d_HOLO, norm = 'ortho'))

    d_HOLO_PROPAG = cp.zeros(shape = (nb_pix_Y, nb_pix_X), dtype = cp.complex64)

    #print('somme avant fft:', cp.asnumpy(intensite(d_HOLO)).sum(), 'somme après FFT', cp.asnumpy(intensite(d_FFT_HOLO)).sum())

    d_spec_filter_FFT[nBlock, nthread](d_FFT_HOLO, d_FFT_HOLO, nb_pix_X, nb_pix_Y, f_pix_min, f_pix_max)

    for i in range(nbPropag):
        distance = distancePropagIni + i * pasPropag
        d_calc_kernel_angular_spectrum_jit[nBlock, nthread](d_KERNEL, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance)
        #analyse_array_cplx(d_KERNEL)
        d_FFT_HOLO_PROPAG = d_FFT_HOLO * d_KERNEL
        d_HOLO_PROPAG = fft2(fftshift(d_FFT_HOLO_PROPAG), norm = 'ortho')
        #analyse_array_cplx(d_FFT_HOLO_PROPAG)
        #d_HOLO_VOLUME_PROPAG_MODULE[i,:,:] = cp.flip(cp.flip(cp.sqrt(cp.real(d_HOLO_PROPAG)**2 + cp.imag(d_HOLO_PROPAG)**2), axis=1), axis=0)
        d_HOLO_VOLUME_PROPAG_MODULE[i,:,:] = cp.flip(cp.flip(cp.sqrt(cp.real(d_HOLO_PROPAG)**2 + cp.imag(d_HOLO_PROPAG)**2), axis=1), axis=0)

        #print("\n intensité distance ", distance, " :")
        #analyse_array_cplx(d_HOLO_VOLUME_PROPAG[:,:,i])

def test_multiFFT(d_plan, nb_FFT):
    for i in range(nb_FFT):
        d_fft_plan = fftshift(fft2(d_plan, norm = 'ortho'))
        print('somme avant fft:', cp.asnumpy(intensite(d_plan)).sum(), 'somme après FFT', cp.asnumpy(intensite(d_fft_plan)).sum())
        d_plan = fft2(fftshift(d_fft_plan), norm = 'ortho')
        print('somme avant fft:', cp.asnumpy(intensite(d_plan)).sum(), 'somme après FFT', cp.asnumpy(intensite(d_fft_plan)).sum())

def volume_propag_Rayleigh_Sommerfeld(d_HOLO, d_FFT_HOLO, d_KERNEL, d_FFT_KERNEL, d_FFT_HOLO_PROPAG, d_HOLO_VOLUME_PROPAG,
lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, pasPropag, nbPropag):

    nthread = 1024
    nBlock = math.ceil(nb_pix_X * nb_pix_Y // nthread)

    d_FFT_HOLO = fftshift(fft2(d_HOLO, norm = 'ortho'))
    for i in range(nbPropag):
        distance = (i + 1)* pasPropag
        d_calc_kernel_propag_Rayleigh_Sommerfeld[nBlock, nthread](d_KERNEL, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance)
        #affichage(phase(d_KERNEL))
        d_FFT_KERNEL = fftshift(fft2(d_KERNEL, norm = 'ortho'))
        #affichage(phase(d_FFT_KERNEL))
        d_FFT_HOLO_PROPAG = d_FFT_HOLO * d_KERNEL
        d_HOLO_VOLUME_PROPAG[:,:,i] = fft2(fftshift(d_FFT_HOLO_PROPAG), norm = 'ortho')

def volume_propag_fresnell(d_HOLO, d_Holo_temp, d_FFT, d_HOLO_VOLUME_PROPAG, 
lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, pasPropag, nbPropag):

    nthread = 1024
    nBlock = math.ceil(nb_pix_X * nb_pix_Y // nthread)

    for i in range(nbPropag):
        distance = (i + 1)* pasPropag
        d_propag_fresnel_phase1_jit[nBlock, nthread](d_HOLO, d_Holo_temp, lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance)
        d_FFT = fftshift(fft2(d_Holo_temp, norm = 'ortho'))
        d_propag_fresnel_phase2_jit[nBlock, nthread](d_FFT, d_HOLO_VOLUME_PROPAG[:,:,i], lambda_milieu, magnification, pixSize, nb_pix_X, nb_pix_Y, distance)


@jit.rawkernel()
def clean_plan_cplx_device(d_plan_cplx, size_x, size_y, posX, posY, clean_radius_pix, replace_cplx_value):

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    sizeXY = size_x * size_y

    jj = cp.int32(index) // cp.int32(size_x)
    ii = cp.int32(index) - cp.int32(jj * size_x)

    if (ii < size_x and jj < size_y):

        #calcul distance
        distance = cp.sqrt((posX - ii)**2 + (posY - jj)**2)
        cplx = d_plan_cplx[ii, jj]
        r = cp.real(cplx)
        i = cp.imag(cplx)
        mod = cp.sqrt(r**2 + i**2)

        if (distance < clean_radius_pix):
            d_plan_cplx[ii, jj] = 0.0+0j
        else:
            d_plan_cplx[ii, jj] = mod + 0j


def clean_plan_cplx(d_plan_cplx, size_x, size_y, posX, posY, clean_radius_pix, replace_value):

    nthread = 1024
    nBlock = math.ceil(size_x * size_y // nthread)

    print(type(d_plan_cplx[0,0]))

    clean_plan_cplx_device[nBlock, nthread](d_plan_cplx, size_x, size_y, posX, posY, clean_radius_pix, replace_value)


