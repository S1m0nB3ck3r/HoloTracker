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
from cupyx.scipy.fft import *
from typeHolo import *
import math
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
import trackpy.diag as dg


part = particule(1,2,3,0.0)
print(part)

# repertoire courant
path = r'C:\\TRAVAIL\\developpement\\imagesHolo\\1000im_manip3\\'
result_filename = 'result_python_cleaning_20percent.csv'
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
sumSize = 15
sizeClean = 20
#pas en µm
dx = 1000000 * infoHolo.pixSize / infoHolo.magnification
dy = 1000000 * infoHolo.pixSize / infoHolo.magnification
dz = 1

#seuillage, binarisation, et CCL
nbStdVarThreshold = 12.0
typeThreshold = type_threshold.THRESHOLD
n_connectivity = 26
filter_size = 0

#allocations
h_holo = np.zeros(shape = (sizeX, sizeY), dtype = cp.float32)
d_holo = cp.zeros(shape = (sizeX, sizeY), dtype = cp.float32)
d_fft_holo = cp.zeros(shape = (sizeX, sizeY), dtype = cp.complex64)
d_fft_holo_propag = cp.zeros(shape = (sizeX, sizeY), dtype = cp.complex64)
d_holo_propag = cp.zeros(shape = (sizeX, sizeY), dtype = cp.float32)
d_KERNEL = cp.zeros(shape = (sizeX, sizeY), dtype = cp.complex64)
d_FFT_KERNEL = cp.zeros(shape = (sizeX, sizeY), dtype = cp.complex64)
d_volume = cp.zeros(shape = (sizeX, sizeY, nb_plan), dtype = cp.complex64)
d_volume_focus = cp.zeros(shape = (sizeX, sizeY, nb_plan), dtype = cp.float32)
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

t_holo = time.perf_counter()
t_1_clean = time.perf_counter()

#pour chaque hologramme du répertoire
for image in os.listdir(path):
    print("temps clean Holo : ", time.perf_counter() - t_holo)
    t_holo = time.perf_counter()

    if (image.split('.')[-1].lower() == type_image.lower()):

        print("temps 1passe cleaning : ", time.perf_counter() - t_1_clean)
        t_1_clean = time.perf_counter()

        i_image += 1
        print('analyse image numéro', int(i_image), ' sur ', nb_images )

        #read image
        img, h_holo = read_image(path + image, sizeX, sizeY)

        #div image moyenne
        h_holo = h_holo / h_mean_holo
        #img = Image.fromarray((h_holo - min) * 255 / (max - min))

        d_holo = cp.asarray(h_holo)

        liste_particule = []
        iteration = 0
        percent = 100.0
        FL = 15
        FH = 125

        while (percent > 20.0):

            if (iteration == 0.0):
                FL = 15
                FH = 125
            else:
                FL = 0.0
                FH = 0.0

            #calcul du volume propagé SPECTRE ANGULAIRE
            propag.volume_propag_angular_spectrum(d_holo, d_fft_holo, d_KERNEL, d_fft_holo_propag, d_volume,
                infoHolo.lambdaMilieu, infoHolo.magnification, infoHolo.pixSize, infoHolo.nb_pix_X, infoHolo.nb_pix_Y, 0.0, dz * 1e-6, nb_plan, 15,125)

            #calcul du focus sur tout le volume
            focus.focus(d_volume, d_volume_focus, sumSize, Focus_type.TENEGRAD)

            if (iteration == 0):
                sumIni =  intensite(d_volume).sum()
                sum = sumIni
            else:
                FL = 0.0
                FH = 0.0
                sum = intensite(d_volume).sum()
                percent = 100.0 * sum / sumIni

            iteration+=1
            print("percent : ", percent)

            #recherche max focus
            index_max_X, index_max_Y , index_max_Z = np.unravel_index(cp.asnumpy(d_volume_focus.argmax()), (1024, 1024, 100))

            #nouvelle_particule = particule(index_max_X, index_max_Y, index_max_Z, 0.0)
            liste_particule.append([i_image, index_max_X, index_max_Y, index_max_Z, 0.0])

            replace_value = d_volume_focus[:,:, index_max_Z].mean()
            
            pos_ini = 0.0
            distance_back_propag = - (pos_ini + index_max_Z * dz * 1e-6)
            #clean hologramme
            print(cp.real(d_volume[0,0, index_max_Z]))
            print(cp.imag(d_volume[0,0, index_max_Z]))

            print( cp.sqrt(cp.real( d_volume[512,512, index_max_Z] )**2 + cp.imag( d_volume[512,512, index_max_Z])**2 ))

            propag.clean_plan_cplx(d_volume[:,:, index_max_Z], infoHolo.nb_pix_X, infoHolo.nb_pix_Y, index_max_X, index_max_Y, 50, replace_value)

            #affichage(intensite(d_volume[:,:, index_max_Z]))

            d_holo = propag.propag_angular_spectrum(d_volume[:,:, index_max_Z], d_fft_holo, d_KERNEL, d_fft_holo, d_holo, 
                                    infoHolo.lambdaMilieu, infoHolo.magnification, infoHolo.pixSize, infoHolo.nb_pix_X, infoHolo.nb_pix_Y,
                                    distance_back_propag, 0, 0)
            
            print("image ", i_image, " particule N° ", iteration,  "XYZ", index_max_X," ", index_max_Y ," ", index_max_Z)

df_particules = pd.DataFrame(liste_particule, columns=['frame', 'X', 'Y', 'Z', 'nb_pix'])
df_particules.to_csv(result_filename, mode = 'a', index = False, header = False)
                        


        






