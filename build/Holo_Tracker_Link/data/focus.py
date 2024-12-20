# -*- coding: utf-8 -*-

"""
Filename: focus.py

Description:
Groups of functions needed for treat 3d holograms with a focus criterion to make appear objects of interest.
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


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
from numpy.fft import fft2 as np_fft2
from numpy.fft import ifft2 as np_ifft2
from numpy.fft import fftshift as np_fftshift
from numpy.fft import ifftshift as np_ifftshift
import cupy as cp
from cupy.fft import fft2, ifft2, fftshift, ifftshift
from cupyx import jit
from cupyx.scipy import ndimage as cp_ndimage
import typeHolo
from traitement_holo import *

from enum import Enum
class Focus_type(Enum):
    ALL = 1
    SUM_OF_LAPLACIAN = 2
    SUM_OF_VARIANCE = 3
    TENEGRAD = 4
    SUM_OF_INTENSITY = 5


def focus_sum_square_of_laplacien(d_volume_IN, d_focus_OUT, sumSize):
    
    sizeZ, sizeY, sizeX = cp.shape(d_volume_IN)

    #allocation intensity plane
    module_plane = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)
    laplace_plane = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)

    #si sumSize est pair rajouter 1
    sumSize = (sumSize //2) * 2  + 1
    #allocation plan de convolution (carré de 1.0 de taille sumSize)
    convolve_plane = cp.full(fill_value= 1.0 / (sumSize*sumSize), dtype=cp.float32, shape=(sumSize, sumSize))

    if d_volume_IN.dtype == cp.complex64:
        for p in range(sizeZ):
            module_plane = module(d_volume_IN[p,:,:])
            laplace_plane = cp_ndimage.laplace(module_plane)
            laplace_plane = cp.square(laplace_plane)
            # laplace_plane = laplace_plane / module_plane
            cp_ndimage.convolve(laplace_plane, convolve_plane, output = d_focus_OUT[p,:,:], mode = 'reflect')

    else : # module float32
        for p in range(sizeZ):
            module_plane = d_volume_IN[p,:,:]
            laplace_plane = cp_ndimage.laplace(module_plane)
            laplace_plane = cp.square(laplace_plane)
            # laplace_plane = laplace_plane / module_plane
            cp_ndimage.convolve(laplace_plane, convolve_plane, output = d_focus_OUT[p,:,:], mode = 'reflect')
        

def focus_sum_of_variance(d_volume_IN, d_focus_OUT, sumSize):
    sizeZ, sizeY, sizeX = cp.shape(d_volume_IN)

    ### calcul du focus selon la méthode sumOfVariance
    ### REF Autofocusing in digital holographic phase contrast microscopy on pure phase objects for live cell imaging
    ### dans tout le volume, pour chaque voxel est calcule la variance locale dans un plan XY (de taille sumSize) autour de celui-ci
    ### pour chaque plan:
    ### 1. calcul du plan d'amplitude (qui marche mieux que plan d'intensité)
    ### 2. calcul d'un plan de moyenne locale de l'amplitude (taille sumSize) par une convolution d'un kernel de valeurs 1/sumSize
    ### 3. calcul d'un plan d'écart à la moyenne locale au carré
    ### 4. convolution du plan précédent ->kernel de taille sumSize de valeur 1/sumSize

    #allocation intensity plane
    module_plane = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)
    local_mean_plane = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)
    variance_plane = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)

    #si sumSize est pair rajouter 1
    sumSize = (sumSize //2 ) * 2 + 1
    #allocation plan de convolution (carré de 1.0 de taille sumSize)
    convolve_mean = cp.full(fill_value=1.0 / (sumSize*sumSize), dtype=cp.float32, shape=(sumSize, sumSize))

    for p in range(sizeZ):
        module_plane = module(d_volume_IN[p,:,:])
        cp_ndimage.convolve(module_plane, convolve_mean, output = local_mean_plane, mode = 'reflect')
        # variance_plane = cp.square( local_mean_plane - module_plane ) / local_mean_plane

        variance_plane = cp.square( local_mean_plane - module_plane )
        cp_ndimage.convolve(variance_plane, convolve_mean, output = d_focus_OUT[p,:,:], mode = 'reflect')
        

def focus_TENEGRAD(d_volume_IN, d_focus_OUT, sumSize):
    sizeZ, sizeY, sizeX = cp.shape(d_volume_IN)

    #allocation intensity plane
    plane_module = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)
    plane_ten1 = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)
    plane_ten2 = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)

    ten1 = cp.asarray(((-1.0, 0.0, 1.0),(-2.0, 0.0, 2.0),(-1.0, 0.0, 1.0)), dtype = np.float32)
    ten2 = cp.asarray(((-1.0, -2.0, -1.0),(0.0, 0.0, 0.0),(1.0, 2.0, 1.0)), dtype = np.float32)

    #si sumSize est pair rajouter 1
    sumSize = (sumSize //2 )*2 +1
    #allocation plan de convolution (carré de 1.0 de taille sumSize)
    convolve_plane = cp.full(fill_value=1.0 / (sumSize * sumSize), dtype=cp.float32, shape=(sumSize, sumSize))

    if d_volume_IN.dtype == cp.complex64:
        for p in range(sizeZ):
            plane_module = module(d_volume_IN[p,:,:])
            cp_ndimage.convolve(plane_module, ten1, output = plane_ten1, mode = 'reflect')
            cp_ndimage.convolve(plane_module, ten2, output = plane_ten2, mode = 'reflect')
            plane_tenegard = cp.sqrt(plane_ten1**2 + plane_ten2**2)
            # plane_tenegard = plane_ten1**2 + plane_ten2**2

            cp_ndimage.convolve(plane_tenegard, convolve_plane,  output = d_focus_OUT[p,:,:], mode = 'reflect')
    else : # module float32
        for p in range(sizeZ):
            cp_ndimage.convolve(d_volume_IN[p,:,:], ten1, output = plane_ten1, mode = 'reflect')
            cp_ndimage.convolve(d_volume_IN[p,:,:], ten2, output = plane_ten2, mode = 'reflect')
            plane_tenegard = cp.sqrt(plane_ten1**2 + plane_ten2**2)
            # plane_tenegard = plane_ten1**2 + plane_ten2**2
            cp_ndimage.convolve(plane_tenegard, convolve_plane,  output = d_focus_OUT[p,:,:], mode = 'reflect')


def focus_SUM_OF_INTENSITY(d_volume_IN, d_focus_OUT, sumSize):

    sizeZ, sizeY, sizeX  = cp.shape(d_volume_IN)

    #allocation intensity plane
    plane = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)

    #si sumSize est pair rajouter 1
    sumSize = (sumSize //2 ) * 2 + 1
    #allocation plan de convolution (carré de 1.0 de taille sumSize)
    convolve_plane = cp.full(fill_value=1.0 / (sumSize * sumSize), dtype=cp.float32, shape=(sumSize, sumSize))

    if d_volume_IN.dtype == cp.complex64:
        for p in range(sizeZ):
            plane = intensite(d_volume_IN[p,:,:])
            cp_ndimage.convolve(plane, convolve_plane, output = d_focus_OUT[p,:,:], mode = 'reflect')
    else : # module float32
        for p in range(sizeZ):
            plane = d_volume_IN[p,:,:]**2
            cp_ndimage.convolve(plane, convolve_plane, output = d_focus_OUT[p,:,:], mode = 'reflect')




def focus(d_volume_IN, d_focus_OUT, sumSize, type_of_focus):

    if type_of_focus == Focus_type.TENEGRAD:
        focus_TENEGRAD(d_volume_IN, d_focus_OUT, sumSize)
    elif type_of_focus == Focus_type.SUM_OF_VARIANCE:
        focus_sum_of_variance(d_volume_IN, d_focus_OUT, sumSize)
    elif type_of_focus == Focus_type.SUM_OF_LAPLACIAN:
        focus_sum_square_of_laplacien(d_volume_IN, d_focus_OUT, sumSize)
    elif type_of_focus == Focus_type.SUM_OF_INTENSITY:
        focus_SUM_OF_INTENSITY(d_volume_IN, d_focus_OUT, sumSize)
