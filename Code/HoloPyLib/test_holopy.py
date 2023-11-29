import holopy as hp
from holopy.scattering import calc_holo, Sphere, Spheres, Cylinder
from holopy.core.metadata import get_spacing
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import time as ti
import sys

matplotlib.use('TkAgg',force=True)

matplotlib.rcParams['backend'] = 'TkAgg' 


#infos générales
indice_milieu = 1.33
indice_particules = 1.59
longueur_onde = 0.660
taille_pix = 5.5
grossissement = 4
taille_pix_holo = taille_pix / grossissement
rayon_particules = 2
pas_propagation = 0.5
pola = (1, 0)


# Générer les positions et rayons des sphères aléatoirement
np.random.seed(0)  # Pour reproductibilité
nb_pix_detector = 100
size_hologram = nb_pix_detector * taille_pix_holo


sph1 = Sphere(n = indice_particules, r = 2, center = (10, 10, 1))
sph2 = Sphere(n = indice_particules, r = 2, center = (20, 10, 1))
sph3 = Sphere(n = indice_particules, r = 2, center = (30, 10, 1))
sph4 = Sphere(n = indice_particules, r = 2, center = (40, 10, 1))

spheres = Spheres([sph1, sph2, sph3, sph4])

detecteur = hp.detector_grid(shape = nb_pix_detector, spacing = taille_pix_holo)

print(get_spacing(detecteur))

print("début calcul hologramme")

holo = calc_holo(detector = detecteur, scatterer  = spheres, medium_index = indice_milieu,
                 illum_wavelen = longueur_onde, illum_polarization = pola, theory='auto')

sys.stdout = sys.__stdout__

hp.save_image('test_simu_holopy.bmp', holo)
hp.show(holo)

# data_2d=holo.squeeze(dim='z', drop='True')
# plt.imshow(holo,cmap='gray')
# plt.axis('off')
# plt.show()
a=1