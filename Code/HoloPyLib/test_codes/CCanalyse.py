# -*- coding: utf-8 -*-
import numpy as np
from typeHolo import *
import cython
from numba import njit
import time

dobjet = np.dtype([
    ('label', np.uint64),
    ('baryX', np.float32),
    ('baryY', np.float32),
    ('baryZ', np.float32),
    ('nb_pix', np.uint32),
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

def label_is_in_list_objet(liste_objet, label):

    for obj in liste_objet:
        if obj.label == label:
            return True
    
    return False

@njit(nopython = True, fastmath=True)
def CCA(h_labels_volume, h_focus_volume, features):

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
                    features[i]['label'] = label
                    #nb_pix
                    features[i]['nb_pix']+=1
                    #xMin
                    features[i]['xMin'] = min(features[i]['xMin'], x)
                    #xMax
                    features[i]['xMax'] = max(features[i]['xMax'], x)
                    features[i]['yMin'] = min(features[i]['yMin'], y)
                    features[i]['yMax'] = max(features[i]['yMax'], y)
                    features[i]['zMin'] = min(features[i]['zMin'], z)
                    features[i]['zMax'] = max(features[i]['zMax'], z)
                    features[i]['pSum'] += h_focus_volume[x,y,z]
                    features[i]['pxSumX'] += x * h_focus_volume[x,y,z]
                    features[i]['pxSumY'] += y * h_focus_volume[x,y,z] 
                    features[i]['pxSumZ'] += z * h_focus_volume[x,y,z] 

    #calcul des barycentres
    for i in range(len(features)):
        features[i]['baryX'] = features[i]['pxSumX'] / features[i]['pSum']
        features[i]['baryY'] = features[i]['pxSumY'] / features[i]['pSum']
        features[i]['baryZ'] = features[i]['pxSumZ'] / features[i]['pSum']

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
                    features[i, 2] = min(features[i, 2], x)
                    #xMax
                    features[i, 3] = max(features[i, 3], x)
                    #yMin
                    features[i, 4] = min(features[i, 4], y)
                    #yMax
                    features[i, 5] = max(features[i, 5], y)
                    #zMin
                    features[i, 6] = min(features[i, 6], z)
                    #zMax
                    features[i, 7] = max(features[i, 7], z)
                    #pxSumX
                    features[i, 8] += h_focus_volume[x,y,z]
                    #pxSumY
                    features[i, 9] += x * h_focus_volume[x,y,z]
                    #pxSumZ
                    features[i, 10] += y * h_focus_volume[x,y,z]
                    #pSum
                    features[i, 11] += z * h_focus_volume[x,y,z] 

    #calcul des barycentres
    for i in range(len(features)):
        features[i, 12]= features[i, 8] / features[i, 11]
        features[i, 13] = features[i, 9] / features[i, 11]
        features[i, 14] = features[i, 10] / features[i, 11]












