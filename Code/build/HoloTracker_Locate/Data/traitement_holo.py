# -*- coding: utf-8 -*-
import os
from PIL import Image
import numpy as np
import cupy as cp
import time
import math
import cc3d
from cupyx import jit
import cupy as cp
from cupy.fft import rfft2, fft2, ifft2, fftshift, ifftshift, fftn, ifftn


def test_var(var):
    return "VAR: " + str(type(var)) + str(var)


def read_image(path_image, sizeX = 0, sizeY = 0):
        
        h_holo = np.asarray(Image.open(path_image))

        if ((sizeX != 0) and (sizeY != 0)):

            sx = np.size(h_holo, axis = 1)
            sy = np.size(h_holo, axis = 0)

            offsetX = (sx - sizeX)//2
            offsetY = (sy - sizeY)//2

            h_holo = h_holo[offsetY:offsetY+sizeY:1, offsetX:offsetX+sizeX:1]
        
        h_holo = h_holo.astype(np.float32)
        return(h_holo)


def affichage(plan):

    if isinstance(plan, cp.ndarray):
        h_plan = cp.asnumpy(plan)
        min = h_plan.min()
        max = h_plan.max()
        img = Image.fromarray((h_plan - min) * 255 / (max - min))

    else:
        min = plan.min()
        max = plan.max()
        img = Image.fromarray((plan - min) * 255 / (max - min))
    
    img.show(title = "plan")
    img.close()


@cp.fuse()
def div_holo(A, B):
    if (B!=0.0):
        C = A/B
    else:
        C = 0.0
    return C

def module(planComplex):
    if isinstance(planComplex, cp.ndarray):
        return(cp.sqrt(cp.square(cp.real(planComplex)) + cp.square(cp.imag(planComplex))))
    else:
        return(np.sqrt(np.square(np.real(planComplex)) + np.square(np.imag(planComplex))))

def intensite(planComplex):
    if isinstance(planComplex, cp.ndarray):
        return(cp.square(cp.real(planComplex)) + cp.square(cp.imag(planComplex)))
    else:
        return(np.square(np.real(planComplex)) + np.square(np.imag(planComplex)))

def phase(planComplex):
    if isinstance(planComplex, cp.ndarray):
        return(cp.arctan(cp.imag(planComplex) /cp.real(planComplex)))
    else:
        return(np.arctan(np.imag(planComplex) /np.real(planComplex)))

def affiche_particule(x, y, z, boxSizeXY, boxSizeZ, d_volume):

    sizeX, sizeY, sizeZ = d_volume.shape
    planXY = np.zeros(shape=(boxSizeXY, boxSizeXY))
    planXZ = np.zeros(shape=(boxSizeXY, boxSizeZ))
    planYZ = np.zeros(shape=(boxSizeXY, boxSizeZ))


    #test des limites des coordonées xyz
    xMin = int(x - boxSizeXY//2)
    xMax = int(x + boxSizeXY//2)
    yMin = int(y - boxSizeXY//2)
    yMax = int(y + boxSizeXY//2)
    zMin = int(z - boxSizeZ//2)
    zMax = int(z + boxSizeZ//2)

    xMin = xMin if xMin > 0 else 0
    xMax = xMax if xMax < sizeX else sizeX 
    yMin = yMin if yMin > 0 else 0 
    yMax = yMax if yMax < sizeY else sizeY
    zMin = zMin if zMin > 0 else 0 
    zMax = zMax if zMax < sizeZ else sizeZ

    if isinstance(d_volume, cp.ndarray):
        if (d_volume.dtype == cp.complex64):
            planXY_t = cp.asnumpy(intensite(d_volume[xMin : xMax, yMin : yMax, z ]))
            planXY[0:boxSizeXY, 0:boxSizeXY] = planXY_t
            planXZ[0:boxSizeXY, 0:boxSizeZ] = cp.asnumpy(intensite(d_volume[xMin : xMax, y, zMin : zMax]))
            planYZ[0:boxSizeXY, 0:boxSizeZ] = cp.asnumpy(intensite(d_volume[x , yMin : yMax, zMin : zMax ]))
        else:
            planXY[0:boxSizeXY, 0:boxSizeXY]  = cp.asnumpy(d_volume[xMin : xMax, yMin : yMax, z ])
            planXZ[0:boxSizeXY, 0:boxSizeZ]  = cp.asnumpy(d_volume[xMin : xMax, y, zMin : zMax])
            planYZ[0:boxSizeXY, 0:boxSizeZ]  = cp.asnumpy(d_volume[x , yMin : yMax, zMin : zMax ])
    else:
        if (d_volume.dtype == np.complex64):
            planXY[0:boxSizeXY, 0:boxSizeXY]  = intensite(d_volume[xMin : xMax, yMin : yMax, z ])
            planXZ[0:boxSizeXY, 0:boxSizeZ]  = intensite(d_volume[xMin : xMax, y, zMin : zMax])
            planYZ[0:boxSizeXY, 0:boxSizeZ]  = intensite(d_volume[x , yMin : yMax, zMin : zMax ])
        else:
            planXY[0:boxSizeXY, 0:boxSizeXY]  = d_volume[xMin : xMax, yMin : yMax, z ]
            planXZ[0:boxSizeXY, 0:boxSizeZ]  = d_volume[xMin : xMax, y, zMin : zMax]
            planYZ[0:boxSizeXY, 0:boxSizeZ]  = d_volume[x , yMin : yMax, zMin : zMax ]

    min = planXY.min()
    max = planXY.max()
    planXY = (planXY - min) * 255 / (max - min)
            
    #planXZ = np.rot90(planXZ)
    min = planXZ.min()
    max = planXZ.max()
    planXZ = (planXZ - min) * 255 / (max - min)
    #planXZ = np.rot90(planXZ)

            
    #planYZ = np.rot90(planYZ)
    min = planYZ.min()
    max = planYZ.max()
    planYZ = (planYZ - min) * 255 / (max - min)
    #planYZ = planYZ.astype(np.uint8)
    #planYZ = np.rot90(planYZ)
    planYZ.reshape((boxSizeXY, boxSizeZ))


    planTot = np.concatenate((planXY, planXZ, planYZ), axis = 1)
    img = Image.fromarray(planTot)
    img.show(title = "objet 3 plans")


def get_sub_plane(x, y, z, boxSizeXY, boxSizeZ, d_volume):

    sizeX, sizeY, sizeZ = d_volume.shape
    planXY = np.zeros(shape=(boxSizeXY, boxSizeXY))
    planXZ = np.zeros(shape=(boxSizeXY, boxSizeZ))
    planYZ = np.zeros(shape=(boxSizeXY, boxSizeZ))


    #test des limites des coordonées xyz
    xMin = int(x - boxSizeXY//2)
    xMax = int(x + boxSizeXY//2)
    yMin = int(y - boxSizeXY//2)
    yMax = int(y + boxSizeXY//2)
    zMin = int(z - boxSizeZ//2)
    zMax = int(z + boxSizeZ//2)

    xMin = xMin if xMin > 0 else 0
    xMax = xMax if xMax < sizeX else sizeX 
    yMin = yMin if yMin > 0 else 0 
    yMax = yMax if yMax < sizeY else sizeY
    zMin = zMin if zMin > 0 else 0 
    zMax = zMax if zMax < sizeZ else sizeZ

    if isinstance(d_volume, cp.ndarray):
        if (d_volume.dtype == cp.complex64):
            planXY_t = cp.asnumpy(intensite(d_volume[xMin : xMax, yMin : yMax, z ]))
            planXY[0:boxSizeXY, 0:boxSizeXY] = planXY_t
            planXZ[0:boxSizeXY, 0:boxSizeZ] = cp.asnumpy(intensite(d_volume[xMin : xMax, y, zMin : zMax]))
            planYZ[0:boxSizeXY, 0:boxSizeZ] = cp.asnumpy(intensite(d_volume[x , yMin : yMax, zMin : zMax ]))
        else:
            planXY[0:boxSizeXY, 0:boxSizeXY]  = cp.asnumpy(d_volume[xMin : xMax, yMin : yMax, z ])
            planXZ[0:boxSizeXY, 0:boxSizeZ]  = cp.asnumpy(d_volume[xMin : xMax, y, zMin : zMax])
            planYZ[0:boxSizeXY, 0:boxSizeZ]  = cp.asnumpy(d_volume[x , yMin : yMax, zMin : zMax ])
    else:
        if (d_volume.dtype == np.complex64):
            planXY[0:boxSizeXY, 0:boxSizeXY]  = intensite(d_volume[xMin : xMax, yMin : yMax, z ])
            planXZ[0:boxSizeXY, 0:boxSizeZ]  = intensite(d_volume[xMin : xMax, y, zMin : zMax])
            planYZ[0:boxSizeXY, 0:boxSizeZ]  = intensite(d_volume[x , yMin : yMax, zMin : zMax ])
        else:
            planXY[0:boxSizeXY, 0:boxSizeXY]  = d_volume[xMin : xMax, yMin : yMax, z ]
            planXZ[0:boxSizeXY, 0:boxSizeZ]  = d_volume[xMin : xMax, y, zMin : zMax]
            planYZ[0:boxSizeXY, 0:boxSizeZ]  = d_volume[x , yMin : yMax, zMin : zMax ]

    min = planXY.min()
    max = planXY.max()
    planXY = (planXY - min) * 255 / (max - min)
            
    min = planXZ.min()
    max = planXZ.max()
    planXZ = (planXZ - min) * 255 / (max - min)

    min = planYZ.min()
    max = planYZ.max()
    planYZ = (planYZ - min) * 255 / (max - min)

    planYZ.reshape((boxSizeXY, boxSizeZ))

    return np.concatenate((planXY, planXZ, planYZ), axis = 1)


def calc_holo_moyen(dirPath, sizeX, sizeY, extension):

    mean_dir = os.path.join(dirPath, "mean")
    if not os.path.exists(mean_dir):
        os.mkdir(mean_dir)

    mean_file_path = os.path.join(mean_dir, "mean_holo.npy")

    if os.path.exists(mean_file_path):

        holo_m = np.load(mean_file_path, allow_pickle=True)
        return holo_m
    
    else:
        holo_m = np.empty((sizeY,sizeX), dtype = np.float32)
        nb_images_tot = len(os.listdir(dirPath))
        nb_images = 0
        for image in os.listdir(dirPath):
            if (image.split('.')[-1].lower() == extension.lower()):
                im_path = os.path.join(dirPath, image)
                nb_images +=1
                img= Image.open(im_path)
                holo = np.asarray(img)

                sx = np.size(holo, axis=1)
                sy = np.size(holo, axis=0)

                offsetX = (sx - sizeX)//2
                offsetY = (sy - sizeY)//2
                
                holo = holo[offsetY:offsetY+sizeY:1, offsetX:offsetX+sizeX:1]
                holo_m += holo
                img.close()
                print(round(100* nb_images/nb_images_tot, 1), "% Done")

        holo_m = holo_m / nb_images
        np.save(mean_file_path, arr=holo_m)
        return(holo_m)


def analyse_array_cplx(data):
    if isinstance(data, cp.ndarray):
        h_data = intensite(cp.asnumpy(data))
    else:
        h_data = intensite(data)
    
    min = h_data.min()
    max = h_data.max()
    sum = h_data.sum()
    mean = h_data.mean()
    std = h_data.std()
    print('min = ', min, 'max = ', max, 'sum =', sum, 'mean = ', mean, 'std =', std)
    return(min, max, mean, sum, std)

def analyse_array(data, titre = ""):
    if isinstance(data, cp.ndarray):
        h_data = cp.asnumpy(data)
    else:
        h_data = data
    
    min = h_data.min()
    max = h_data.max()
    sum = h_data.sum()
    mean = h_data.mean()
    std = h_data.std()
    print(titre, ' min = ', min, 'max = ', max, 'sum =', sum, 'mean = ', mean, 'std =', std)
    return(min, max, mean, sum, std)

def sum_plans(d_volum_focus):
    return(d_volum_focus.sum(axis = 0), d_volum_focus.sum(axis = 1), d_volum_focus.sum(axis = 2))


@jit.rawkernel()
def d_filter_FFT_3D(d_VOLUME_IN, d_VOLUME_OUT, sizeX, sizeY, sizeZ, dMinXY, dMaxXY, dMinZ, dMaxZ):

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x

    planSize = sizeX * sizeY
    kk = index // planSize
    jj = ( index - kk * planSize )// sizeX
    ii = index - jj * sizeX - kk * planSize

    if (ii < sizeX and jj < sizeY):
        #calc distance
        centreX = sizeX // 2
        centreY = sizeY // 2
        centreZ = sizeZ // 2

        distanceCentre = cp.sqrt((centreX - ii)*(centreX - ii) + (centreY - jj)*(centreY - jj))
        distanceZ = cp.abs(centreZ - kk)

        if ((distanceCentre > dMinXY) and (distanceCentre < dMaxXY ) and (distanceZ > dMinZ) and (distanceZ < dMaxZ )):
            d_VOLUME_OUT[ii, jj, kk] = d_VOLUME_IN[ii, jj, kk]
        else:
            d_VOLUME_OUT[ii, jj, kk] = 0.0 + 0.0j

def filtre_volume(d_FFT_volume_IN, d_FFT_volume_OUT, sizeX, sizeY, sizeZ, dMinXY, dMaxXY, dMinZ, dMaxZ):

    nthread = 1024
    nBlock = math.ceil(sizeX * sizeX * sizeZ // nthread)

    d_filter_FFT_3D[nBlock, nthread](d_FFT_volume_IN, d_FFT_volume_OUT, sizeX, sizeY, sizeZ, dMinXY, dMaxXY, dMinZ, dMaxZ)

def normalise_to_U8_volume(d_volume_IN):

    min = cp.min(d_volume_IN)
    max = cp.max(d_volume_IN)

    #d_volume_out = cp.zeros(dtype = cp.uint8, shape = d_volume_IN.shape)

    return(((d_volume_IN - min) * 255 / (max - min)).astype(cp.uint8))





    

