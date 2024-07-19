# -*- coding: utf-8 -*-

"""
Filename: CCL3D.py

Description:
Groups of functions needed for calling and adapting the cc3d library (https://pypi.org/project/connected-components-3d/) to hologram analysis purpose.
Author: Simon BECKER
Date: 2024-07-09

License:
GNU General Public License v3.0

Copyright (C) [2024] Simon BECKER

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import cupy as cp
from typeHolo import *
import cc3d
import numpy as np
from cupyx import jit
from cupyx.scipy import ndimage as cp_ndimage
import numpy as np
from numba import njit
import time
from traitement_holo import *
from enum import Enum

class type_threshold(Enum):
    NB_STD_DEV = 1
    THRESHOLD = 2
    NB_LOCAL_MEAN = 3


dobjet = np.dtype([
    ('i_image', np.uint64),
    ('nb_pix', np.uint32),
    ('baryX', np.float32),
    ('baryY', np.float32),
    ('baryZ', np.float32),
    ('pSum', np.float32),
    ('pxSumX', np.float32),
    ('pxSumY', np.float32),
    ('pxSumZ', np.float32),
    ('xMin', np.uint32),
    ('xMax', np.uint32),
    ('yMin', np.uint32),
    ('yMax', np.uint32),
    ('zMin', np.uint32),
    ('zMax', np.uint32),
])

cp_dobjet = cp.dtype([
    ('i_image', np.uint64),
    ('nb_pix', np.uint32),
    ('baryX', np.float32),
    ('baryY', np.float32),
    ('baryZ', np.float32),
    ('pSum', np.float32),
    ('pxSumX', np.float32),
    ('pxSumY', np.float32),
    ('pxSumZ', np.float32),
    ('xMin', np.uint32),
    ('xMax', np.uint32),
    ('yMin', np.uint32),
    ('yMax', np.uint32),
    ('zMin', np.uint32),
    ('zMax', np.uint32),
])

### Calcule du seuil global sur tout un volume
def calc_threshold(d_focus_volume, nbStdVar):
    mean = cp.mean(d_focus_volume)
    stdVar = cp.std(d_focus_volume)
    return(mean.item() + nbStdVar * stdVar.item())

#binarise un volume de taille sizeX, sizeY, sizeZ
@jit.rawkernel()
def cuda_Binaries_Focus_Volume(d_bin_volume, d_focus_volume, threshold, sizeX, sizeY, sizeZ):

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    planSize = sizeX * sizeY
    kk = index // planSize
    jj = ( index - kk * planSize )// sizeX
    ii = index - jj * sizeX - kk * planSize

    if (ii < sizeX and jj < sizeY and kk < sizeZ):

        if d_focus_volume[kk, jj, ii] > threshold:
            d_bin_volume[kk, jj, ii] = True
        else:
            d_bin_volume[kk, jj, ii] = False

    jit.syncthreads()

#binarise un plan de taille sizeX, sizeY
@jit.rawkernel()
def cuda_Binaries_Focus_Plan(d_bin_plan, d_focus_plan, threshold, sizeX, sizeY):

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    planSize = sizeX * sizeY
    jj = index // sizeX
    ii = index - jj * sizeX

    if (ii < sizeX and jj < sizeY):

        if d_focus_plan[jj, ii] > threshold:
            d_bin_plan[jj, ii] = True
        else:
            d_bin_plan[jj, ii] = False

    jit.syncthreads()


def binaries_Focus_Volume_local_contrast(d_bin_volume, d_focus_volume, threshold_nb_mean, sizeMeanXY, sizeMeanZ):

    sizeX = d_focus_volume.shape[2]
    sizeY = d_focus_volume.shape[1]
    sizeZ = d_focus_volume.shape[0]

    #si sumSize est pair rajouter 1
    sizeMeanXY = (sizeMeanXY //2 ) * 2 + 1
    sizeMeanZ = (sizeMeanZ //2 ) * 2 + 1

    #allocation volume de convolution (cube dont la somme est 1 de taille sizeSumX, sizeSumY, sizeSumZ)
    value = 1.0 / ( sizeMeanXY * sizeMeanXY * sizeMeanZ )
    convolve_volume = cp.full(fill_value=value, dtype=cp.float32, shape=(sizeMeanZ, sizeMeanXY, sizeMeanXY))

    #convolution du plan avec convolve_volume
    d_threashold_volume = cp_ndimage.convolve(input = d_focus_volume, weights = convolve_volume, mode = 'reflect') * threshold_nb_mean
    cp.greater(d_focus_volume, d_threashold_volume, out = d_bin_volume)


def div_by_mean_convolution(d_focus_volume, d_focus_volume_2, sizeMeanXY, sizeMeanZ):

    sizeX = d_focus_volume.shape[2]
    sizeY = d_focus_volume.shape[1]
    sizeZ = d_focus_volume.shape[0]

    #si sumSize est pair rajouter 1
    sizeMeanXY = (sizeMeanXY //2 ) * 2 + 1
    sizeMeanZ = (sizeMeanZ //2 ) * 2 + 1

    #allocation volume de convolution (cube dont la somme est 1 de taille sizeSumX, sizeSumY, sizeSumZ)
    value = 1.0 / ( sizeMeanXY * sizeMeanXY * sizeMeanZ )
    convolve_volume = cp.full(fill_value=value, dtype=cp.float32, shape=(sizeMeanZ, sizeMeanXY, sizeMeanXY))

    d_focus_volume_2 = cp.copy(d_focus_volume)

    #convolution du plan avec convolve_volume
    cp_ndimage.convolve(input = d_focus_volume_2, output = d_focus_volume, weights = convolve_volume, mode = 'reflect')


    


def CCL3D(d_bin_volume, d_focus_volume,  t_threshold, threshold, n_connectivity, filter_size = 0):

    sizeZ, sizeY, sizeX = d_focus_volume.shape
    n_threads = 1024
    n_blocks = (sizeX * sizeY * sizeZ)//1024 +1
    #calcul du seuil

    if t_threshold == type_threshold.NB_STD_DEV :
        threshold = calc_threshold(d_focus_volume, threshold)
        cuda_Binaries_Focus_Volume[n_blocks, n_threads](d_bin_volume, d_focus_volume, threshold, sizeX, sizeY, sizeZ)
    elif t_threshold == type_threshold.THRESHOLD :
        cuda_Binaries_Focus_Volume[n_blocks, n_threads](d_bin_volume, d_focus_volume, threshold, sizeX, sizeY, sizeZ)
    else:
        binaries_Focus_Volume_local_contrast(d_bin_volume, d_focus_volume, threshold, 20, 10)

    nbpix = np.sum(cp.asnumpy(d_bin_volume), keepdims = False)
    #print('nb TRUE = ', nbpix)
    #print('percent = ', 100.0 * nbpix / (sizeX * sizeY * sizeZ))
    
    h_bin_volum = cp.asnumpy(d_bin_volume)
    
    if filter_size != 0:
        cc3d.dust(h_bin_volum, threshold, n_connectivity, in_place=False)

    labels, number_of_labels = cc3d.connected_components(h_bin_volum, connectivity=n_connectivity, out_dtype=np.uint32, return_N=True)
    #stats = cc3d.statistics(labels)
    stats = 0
    return(labels, number_of_labels, stats)

    def label_is_in_list_objet(liste_objet, label):

        for obj in liste_objet:
            if obj.label == label:
                return True
        
        return False

@njit(nopython = True, fastmath=True)
def CCA(h_labels_volume, h_focus_volume, features, i_image, dx, dy, dz):

    #reinit
    for i in range(len(features)):
        features[i]['baryX'] = 0.0
        features[i]['baryY'] = 0.0
        features[i]['baryZ'] = 0.0
        features[i]['i_image'] = i_image
        features[i]['nb_pix'] = 0
        features[i]['xMin'] = 0.0
        features[i]['xMax'] = 0.0
        features[i]['yMin'] = 0.0
        features[i]['yMax'] = 0.0
        features[i]['zMin'] = 0.0
        features[i]['zMax'] = 0.0
        features[i]['pSum'] = 0.0
        features[i]['pxSumX'] = 0.0
        features[i]['pxSumY'] = 0.0
        features[i]['pxSumZ'] = 0.0

    sizeX, sizeY, sizeZ = h_labels_volume.shape
    #creation de la liste d'objets analysés
    #on parcours le volume labelisé
    for z in range(sizeZ):
        #print('z= ', z,'\n')
        for y in range(sizeY):
            for x in range(sizeX):
                label = h_labels_volume[x,y,z]
                if label!=0:
                    #print('xyz= ',x,' ',y, ' ', z, 'label :', label)
                    i = int(label-1)
                    #nb_pix
                    features[i]['nb_pix']+=1
                    #xMin
                    features[i]['xMin'] = min(features[i]['xMin'], x * dx)
                    #xMax
                    features[i]['xMax'] = max(features[i]['xMax'], x * dx)
                    features[i]['yMin'] = min(features[i]['yMin'], y * dy)
                    features[i]['yMax'] = max(features[i]['yMax'], y * dy)
                    features[i]['zMin'] = min(features[i]['zMin'], z * dz)
                    features[i]['zMax'] = max(features[i]['zMax'], z * dz)
                    features[i]['pSum'] += h_focus_volume[x,y,z]
                    features[i]['pxSumX'] += x * h_focus_volume[x,y,z]
                    features[i]['pxSumY'] += y * h_focus_volume[x,y,z] 
                    features[i]['pxSumZ'] += z * h_focus_volume[x,y,z] 

    #calcul des barycentres
    for i in range(len(features)):
        features[i]['baryX'] = dx *features[i]['pxSumX'] / features[i]['pSum']
        features[i]['baryY'] = dy * features[i]['pxSumY'] / features[i]['pSum']
        features[i]['baryZ'] = dz * features[i]['pxSumZ'] / features[i]['pSum']

#version test du CCA avec une liste de features de type np.ndarray2D
@njit(nopython = True)
def CCA2(h_labels_volume, h_focus_volume, features):

    sizeX, sizeY, sizeZ = h_labels_volume.shape
    #creation de la liste d'objets analysés
    #on parcours le volume labelisé
    for z in range(sizeZ):
        #print('z= ', z,'\n')
        for y in range(sizeY):
            for x in range(sizeX):
                label = h_labels_volume[x,y,z]
                if label!=0:
                    #print('xyz= ',x,' ',y, ' ', z, 'label :', label)
                    i = int(label-1)
                    #label
                    features[i,0] = label
                    #nb_pix
                    features[i, 1]+=1
                    #xMin
                    features[i, 5] = min(features[i, 2], x)
                    #xMax
                    features[i, 5] = max(features[i, 3], x)
                    #yMin
                    features[i, 6] = min(features[i, 4], y)
                    #yMax
                    features[i, 7] = max(features[i, 5], y)
                    #zMin
                    features[i, 8] = min(features[i, 6], z)
                    #zMax
                    features[i, 9] = max(features[i, 7], z)
                    #pxSumX
                    features[i, 10] += h_focus_volume[x,y,z]
                    #pxSumY
                    features[i, 11] += x * h_focus_volume[x,y,z]
                    #pxSumZ
                    features[i, 12] += y * h_focus_volume[x,y,z]
                    #pSum
                    features[i, 13] += z * h_focus_volume[x,y,z] 

    #calcul des barycentres
    for i in range(len(features)):
        #baryX = psumX / psum
        features[i, 2]= features[i, 10] / features[i, 13]
        #baryY = psumY / psum
        features[i, 3] = features[i, 11] / features[i, 13]
        #baryZ = psumZ / psum
        features[i, 4] = features[i, 12] / features[i, 13]


@jit.rawkernel()
def device_CCA(d_volume_label, d_volume_focus, d_features, sizeX, sizeY, sizeZ, dx, dy, dz):

    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    planSize = sizeX * sizeY
    kk = index // planSize
    jj = ( index - kk * planSize )// sizeX
    ii = index - jj * sizeX - kk * planSize

    if (ii < sizeX and jj < sizeY and kk < sizeZ):
        label = d_volume_label[kk, jj, ii]
        if (label != 0):
            #calcul des index, coordonnées et valeurs à ajouter aux features
            index = label-1
            posX = ii * dx
            posY = jj * dy
            posZ = kk * dz
            focus = d_volume_focus[kk, jj, ii]
            pxSumX = posX * focus
            pxSumY = posY * focus
            pxSumZ = posZ * focus
            """
            #index des données de type features
            # (nb_pix: 0,
            # pxSumX:1, pxSumY:2, pxSumZ:3, pSum:4
            #   )
            """
            #les valeurs actu
            #excuecution des fonctions atomics
            #nb_pix
            jit.atomic_add(d_features, (index, 0), 1)
            #pxSumX, pxSumY, pxSumZ, pSum
            jit.atomic_add(d_features, (index, 1), pxSumX)
            jit.atomic_add(d_features, (index, 2), pxSumY)
            jit.atomic_add(d_features, (index, 3), pxSumZ)
            jit.atomic_add(d_features, (index, 4), focus)
            
    jit.syncthreads()


@jit.rawkernel()
def device_CCA_plane(d_plane_label, d_plane_focus, d_features, sizeX, sizeY, dx, dy, dz, planeNumber):
    #version optimisé de device_CCA pour ne travailler que sur un plan et économiser de la méméoire GPU
    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    planSize = sizeX * sizeY
    jj = index // sizeX
    ii = index - jj * sizeX

    if (ii < sizeX and jj < sizeY):
        label = d_plane_label[jj,ii]
        if (label != 0):
            #calcul des index, coordonnées et valeurs à ajouter aux features
            index = label-1
            posX = ii * dx
            posY = jj * dy
            posZ = planeNumber * dz
            focus = d_plane_focus[jj,ii]
            pxSumX = posX * focus
            pxSumY = posY * focus
            pxSumZ = posZ * focus
            """
            #index des données de type features
            # (nb_pix: 0,
            # pxSumX:1, pxSumY:2, pxSumZ:3, pSum:4
            #   )
            """
            #les valeurs actu
            #excuecution des fonctions atomics
            #nb_pix
            jit.atomic_add(d_features, (index, 0), 1)
            #pxSumX, pxSumY, pxSumZ, pSum
            jit.atomic_add(d_features, (index, 1), pxSumX)
            jit.atomic_add(d_features, (index, 2), pxSumY)
            jit.atomic_add(d_features, (index, 3), pxSumZ)
            jit.atomic_add(d_features, (index, 4), focus)
            
    jit.syncthreads()


def CCA_CUDA(h_volume_label, d_volume_focus, number_of_labels, i_image, sizeX, sizeY, sizeZ, dx, dy, dz):

    d_features = cp.ndarray(shape = (number_of_labels, 5), dtype = cp.float32)

    for i in range(number_of_labels):
        d_features[i, 0] = 0.0  #nb_pix 
        d_features[i, 1] = 0.0  #psSumX
        d_features[i, 2] = 0.0  #psSumY
        d_features[i, 3] = 0.0  #psSumZ
        d_features[i, 4] = 0.0  #pSum

    sizeX, sizeY, sizeZ = h_volume_label.shape
    n_threads = 1024
    n_blocks = (sizeX * sizeY * sizeZ)//1024 +1
    d_volume_label = cp.asarray(h_volume_label)

    device_CCA[n_blocks, n_threads](d_volume_label, d_volume_focus, d_features, sizeX, sizeY, sizeZ, dx, dy, dz)
    
    h_features = cp.asnumpy(d_features)

    features = np.ndarray(shape = (number_of_labels,), dtype = dobjet)

    #copie h_features vers features
    for i in range(number_of_labels):
        features[i]['i_image'] = i_image
        features[i]['nb_pix'] = h_features[i, 0]
        features[i]['pxSumX'] = h_features[i, 1]
        features[i]['pxSumY'] = h_features[i, 2]
        features[i]['pxSumZ'] = h_features[i, 3]
        features[i]['pSum'] =   h_features[i, 4]

    #calcul des barycentres

    features['baryX'] = features['pxSumX'] / features['pSum']
    features['baryY'] = features['pxSumY'] / features['pSum']
    features['baryZ'] = features['pxSumZ'] / features['pSum']
    features['i_image'] = i_image

    return features


def CCA_CUDA_float(h_volume_label, d_volume_focus, number_of_labels, i_image, sizeX, sizeY, sizeZ, dx, dy, dz):

    d_CCA = cp.ndarray(shape = (number_of_labels, 5), dtype = cp.float32)

    for i in range(number_of_labels):
        d_CCA[i, 0] = 0.0  #nb_pix 
        d_CCA[i, 1] = 0.0  #psSumX
        d_CCA[i, 2] = 0.0  #psSumY
        d_CCA[i, 3] = 0.0  #psSumZ
        d_CCA[i, 4] = 0.0  #pSum

    sizeZ, sizeY, sizeX = h_volume_label.shape
    n_threads = 1024
    n_blocks = (sizeX * sizeY * sizeZ)//1024 +1
    d_volume_label = cp.asarray(h_volume_label)

    device_CCA[n_blocks, n_threads](d_volume_label, d_volume_focus, d_CCA, sizeX, sizeY, sizeZ, dx, dy, dz)

    h_CCA = cp.asnumpy(d_CCA)

    features = np.ndarray(shape = (number_of_labels, 5), dtype = np.float32)

    #calcul barycentres et copie données
    for i in range(number_of_labels):
        features[i, 0] = i_image
        features[i, 1] = h_CCA[i, 1] / h_CCA[i, 4] #calcul baryX = pxSumX / pSum
        features[i, 2] = h_CCA[i, 2] / h_CCA[i, 4] #calcul baryY = pxSumY / pSum
        features[i, 3] = h_CCA[i, 3] / h_CCA[i, 4] #calcul baryZ = pxSumZ / pSum
        features[i, 4] = h_CCA[i, 0] #nb_pix

    return features

def CCL_filter(features_in, nb_vox_min, nb_vox_max):

    features_out = np.ndarray(shape = (0, 5), dtype = cp.float32)
    feat_temp = np.ndarray(shape = (1, 5), dtype = cp.float32)

    for i in range(features_in.shape[0]):

        if (features_in[i, 4] > nb_vox_min or nb_vox_min == 1 or nb_vox_min == 0)  and  (features_in[i, 4] < nb_vox_max or nb_vox_max == 0) :
            feat_temp[0] = features_in[i]
            features_out = np.append(features_out, feat_temp, axis = 0)
    
    return features_out

