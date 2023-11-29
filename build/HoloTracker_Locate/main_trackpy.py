# -*- coding: utf-8 -*-
import cupy as cp
import numpy as np
from cupyx import jit
import time
import os
from PIL import Image
from traitement_holo import *
import propagation as propag
import focus 
from focus import Focus_type
import typeHolo
import trackpy as tp

#from CCanalyse import *
#from cupyx.scipy.fft import fftn as cpxfftn
#from cupyx.scipy.fft import ifftn as icpxfftn


from cupy.fft import rfft2, fft2, ifft2, fftshift, ifftshift, fftn, ifftn

from typeHolo import *
import math
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
import trackpy.diag as dg

# repertoire courant
path = r'C:\\TRAVAIL\\developpement\\imagesHolo\\1000im_manip3\\'
result_filename = 'result_python_sum15_TENEGRAD_STD15_each.csv'
type_image = 'bmp'

nb_plan = 100

infoHolo = info_Holo()
infoHolo.lambdaMilieu = 660e-9 / 1.33
infoHolo.magnification = 40.0
infoHolo.nb_pix_X = 1024
infoHolo.nb_pix_Y = 1024
infoHolo.pixSize = 5e-6

sizeX = infoHolo.nb_pix_X
sizeY = infoHolo.nb_pix_Y
sumSize = 20
#pas en µm
dx = 1000000 * infoHolo.pixSize / infoHolo.magnification
dy = 1000000 * infoHolo.pixSize / infoHolo.magnification
dz = 1

#seuillage, binarisation, et CCL
nbStdVarThreshold = 15.0
n_connectivity = 26
filter_size = 0


#allocations
h_holo = np.zeros(shape = (sizeX, sizeY), dtype = np.float32)
d_holo = cp.zeros(shape = (sizeX, sizeY), dtype = cp.float32)
d_fft_holo = cp.zeros(shape = (sizeX, sizeY), dtype = cp.complex64)
d_fft_holo_propag = cp.zeros(shape = (sizeX, sizeY), dtype = cp.complex64)
d_holo_propag = cp.zeros(shape = (sizeX, sizeY), dtype = cp.float32)
d_KERNEL = cp.zeros(shape = (sizeX, sizeY), dtype = cp.complex64)
d_FFT_KERNEL = cp.zeros(shape = (sizeX, sizeY), dtype = cp.complex64)
d_volume = cp.zeros(shape = (sizeX, sizeY, nb_plan), dtype = cp.complex64)
d_FFT_volume = cp.zeros(shape = (sizeX, sizeY, nb_plan), dtype = cp.complex64)
d_volume_focus = cp.zeros(shape = (sizeX, sizeY, nb_plan), dtype = cp.float32)
d_volume_focus_2 = cp.zeros(shape = (sizeX, sizeY, nb_plan), dtype = cp.float32)
d_bin_volume_focus = cp.zeros(shape = (sizeX, sizeY, nb_plan), dtype = cp.dtype(bool))

#calcul holo moyen
h_mean_holo = calc_holo_moyen(sizeX, sizeY, path, 'bmp')
min = h_mean_holo.min()
max = h_mean_holo.max()
mean = h_mean_holo.mean()
std_var = h_mean_holo.std()

d_mean_holo = cp.asarray(h_mean_holo)
img_mean_holo = Image.fromarray(h_mean_holo)
#img_mean_holo.show()

i_image = np.uint64(0)
images = [image for image in os.listdir(path) if (image.split('.')[-1].lower() == type_image.lower())]
nb_images = len(images)

if os.path.exists(result_filename):
    os.remove(result_filename)

#pour chaque hologramme du répertoire
for image in os.listdir(path):
    if (image.split('.')[-1].lower() == type_image.lower()):

        ini_time = time.perf_counter()
        i_image += 1
        #print('analyse image numéro', int(i_image), ' sur ', nb_images )

        #read image
        img, h_holo = read_image(path + image, sizeX, sizeY)

        #div image moyenne
        h_holo = h_holo / h_mean_holo
        min = h_holo.min()
        max = h_holo.max()
        img = Image.fromarray((h_holo - min) * 255 / (max - min))
        #img.show()

        d_holo = cp.asarray(h_holo)

        t1 = time.perf_counter()
        #calcul du volume propagé SPECTRE ANGULAIRE
        propag.volume_propag_angular_spectrum(d_holo, d_fft_holo, d_KERNEL, d_fft_holo_propag, d_volume,
            infoHolo.lambdaMilieu, infoHolo.magnification, infoHolo.pixSize, infoHolo.nb_pix_X, infoHolo.nb_pix_Y, 0.0, dz * 1e-6, nb_plan, 15,125)
        
        t2 = time.perf_counter()

        #calcul du focus sur tout le volume
        focus.focus(d_volume, d_volume_focus, sumSize, Focus_type.TENEGRAD)
        
        h_volume_focus_U8 = cp.asnumpy(normalise_to_U8_volume(d_volume_focus))
        h_volume_focus = cp.asnumpy(d_volume_focus)

        t3 = time.perf_counter()
        features = tp.grey_dilation(h_volume_focus_U8, percentile=90, separation=(10, 10, 5))
        #features = tp.locate(raw_image=h_volume_focus, diameter=(7.0, 7.0,7.0), percentile=95, engine='numba')
        t4 = time.perf_counter()
        print("temps localise ", t4-t3)
        print(features)
        end_time = time.perf_counter()
        print("temps total: ", end_time - ini_time)



        










