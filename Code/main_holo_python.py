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
from CCL3D import *
import pyximport; pyximport.install()
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
path = 'C:\\TRAVAIL\\developpement\\simuHologramPython\\results\\2024_03_21_15_31_38\\holograms\\'
result_filename = 'result_python_sum15_TENEGRAD_STD15_each.csv'
type_image = 'bmp'

nb_plan = 200

infoHolo = info_Holo()
infoHolo.lambdaMilieu = 660e-9 / 1.33
infoHolo.magnification = 40.0
infoHolo.nb_pix_X = 1024
infoHolo.nb_pix_Y = 1024
infoHolo.pixSize = 7e-6

sizeX = infoHolo.nb_pix_X
sizeY = infoHolo.nb_pix_Y
sumSize = 15
#pas en µm
dx = 1000000 * infoHolo.pixSize / infoHolo.magnification
dy = 1000000 * infoHolo.pixSize / infoHolo.magnification
dz = 0.2

#seuillage, binarisation, et CCL
nbStdVarThreshold = 15
typeThreshold = type_threshold.THRESHOLD
n_connectivity = 26
filter_size = 0


#allocations
h_holo = np.zeros(shape = (sizeY, sizeX), dtype = np.float32)
d_holo = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)
d_fft_holo = cp.zeros(shape = (sizeY, sizeX), dtype = cp.complex64)
d_fft_holo_propag = cp.zeros(shape = (sizeY, sizeX), dtype = cp.complex64)
d_holo_propag = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)
d_KERNEL = cp.zeros(shape = (sizeY, sizeX), dtype = cp.complex64)
d_FFT_KERNEL = cp.zeros(shape = (sizeY, sizeX), dtype = cp.complex64)
d_volume_module = cp.zeros(shape = (nb_plan, sizeY, sizeX), dtype = cp.float32)
d_bin_volume_focus = cp.zeros(shape = (nb_plan, sizeY, sizeX), dtype = cp.dtype(bool))

#calcul holo moyen
h_mean_holo = calc_holo_moyen(path, sizeX, sizeY, 'bmp')
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

        #read image
        h_holo = read_image(path + image, sizeX, sizeY)

        # affichage(h_holo)

        #div image moyenne
        h_holo = h_holo / h_mean_holo
        min = h_holo.min()
        max = h_holo.max()
        img = Image.fromarray((h_holo - min) * 255 / (max - min))
        # img.show()

        #copie holo host vers gpu
        d_holo = cp.asarray(h_holo)

        t1 = time.perf_counter()

        #calcul du volume propagé SPECTRE ANGULAIRE
        propag.volume_propag_angular_spectrum_to_module(d_holo, d_fft_holo, d_KERNEL, d_fft_holo_propag, d_volume_module,
            infoHolo.lambdaMilieu, infoHolo.magnification, infoHolo.pixSize, infoHolo.nb_pix_X, infoHolo.nb_pix_Y, 0.0, dz * 1e-6, nb_plan, 15,125)
        
        t2 = time.perf_counter()

        #calcul du focus sur tout le volume
        focus.focus(d_volume_module, d_volume_module, sumSize, Focus_type.SUM_OF_LAPLACIAN)
        t3 = time.perf_counter()

        #affichage(d_volume_module[:,:, 100])
        #affichage(d_volume_module[512,:,:])
        #affichage(d_volume_module[:,512,:])

        """
        t_fft3d = time.perf_counter()
        d_FFT_volume = fftn(d_volume_focus)
        d_FFT_volume = fftshift(d_FFT_volume)

        
    
        #affichage(intensite(d_FFT_volume[:,:,50]))
        filtre_volume(d_FFT_volume, d_FFT_volume, sizeX, sizeY, nb_plan, 1, 10000, 1, 10000)
        #affichage(intensite(d_FFT_volume[:,:, 50]))

        d_FFT_volume = ifftshift(d_FFT_volume)
        d_volume_focus = cp.abs(ifftn(d_FFT_volume))

        t_fft3d_end = time.perf_counter()
        print('temps Filtre3D: ', t_fft3d_end - t_fft3d)
        """

        
        #affichage(intensite(d_volume_focus[:,:, 50]))
        #affichage(intensite(d_volume_focus[512,:,:]))
        #affichage(intensite(d_volume_focus[:,512,:]))


        #affichage(h_holo)

        #affichage(cp.rot90(d_volume_focus.sum(axis = 0)))
        #affichage(cp.rot90(d_volume_focus.sum(axis = 1)))
        #affichage(d_volume_focus.sum(axis = 2))


        #CCL3D
        if i_image == 1:
            threshold = calc_threshold(d_volume_module, nbStdVarThreshold)

        sizeMeanXY = 100

        sizeMeanZ = 5

        #t_div = time.perf_counter()
        #analyse_array(d_volume_focus)
        #div_by_mean_convolution(d_volume_focus, d_volume_focus_2, sizeMeanXY, sizeMeanZ)
        #print("temps local mean = ", time.perf_counter() - t_div)

        #analyse_array(d_volume_focus)


        #affichage(cp.rot90(d_volume_focus.sum(axis = 0)))
        #affichage(cp.rot90(d_volume_focus.sum(axis = 1)))
        #affichage(d_volume_focus.sum(axis = 2))

        h_labels, number_of_labels, statsCCL3D = CCL3D(d_bin_volume_focus, d_volume_module, typeThreshold, threshold, n_connectivity, filter_size)
        print(h_labels.dtype)
        h_bin = cp.asnumpy(d_bin_volume_focus)
        #print("nb pix: ", d_bin_volume_focus.sum())
        t4 = time.perf_counter()


        #analyse des labels
        features = np.ndarray(shape = (number_of_labels,), dtype = dobjet)
        print('nombre d\'objet trouvés: ', number_of_labels)
        #CCA(h_labels, h_focus_volume, features, i_image, dx, dy , dz)
        # features = CCA_CUDA(h_labels, d_volume_module, number_of_labels, i_image, sizeX, sizeY, nb_plan, dx, dy, dz)

        features = CCA_CUDA_float(h_labels, d_volume_module, number_of_labels, i_image, sizeX, sizeY, nb_plan, dx, dy, dz)

        features_filtered = CCL_filter(features, 1, 0)



        end_CCL_CCA_time = time.perf_counter()

        positions = pd.DataFrame({'frame' : features['i_image'],'x': features['baryX'],
        'y' : features['baryY'], 'z' : features['baryZ'],'nb_pix' : features['nb_pix']} )

        #positions.to_csv(result_filename, mode = 'a', index = False, header = (i_image == 1))
        positions.to_csv(result_filename, mode = 'a', index = False, header = False)

        t5 = time.perf_counter()
        
        final_time = time.perf_counter()
        t_propag = t2 - t1
        t_focus = t3 - t2
        t_ccl = t4 - t3
        t_cca = t5 - t4
        print('t propag : ', t_propag)
        print('t focus : ', t_focus)
        print('t ccl : ', t_ccl)
        print('t cca : ', t_cca, '\n')
        print('temps traitement: ', final_time - ini_time)


        #h_intensite = cp.asnumpy(d_volume_module**2).reshape((sizeX * sizeY * nb_plan, ))
        #plt.hist(h_intensite, bins = 1000)
        #plt.axis()
        #plt.yscale('log')
        #plt.show()






        """
        for i in range(number_of_labels):

            x = features[i]['baryX']
            y = features[i]['baryY']
            z = features[i]['baryZ']

            affiche_particule(x, y, z, 20, 10, d_volume)
        """

        """
        #affichage 3D
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        Z = features['baryZ']
        Y = features['baryY']
        X = features['baryX']
        ax.scatter3D(X, Y, Z)

        plt.show()
        """


        










