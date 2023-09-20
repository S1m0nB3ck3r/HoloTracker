import holopy as hp
from holopy.scattering import calc_holo, Sphere, Spheres, Cylinder
import matplotlib.pyplot as plt
import numpy as np
import time as ti


#infos générales
indice_milieu = 1.33
indice_particules = 1.59
longueur_onde = 0.660
taille_pix = 5.5
grossissement = 40
taille_pix_holo = taille_pix / grossissement
rayon_particules = 2
pas_propagation = 0.5
illum_polarization = (1, 0)


# Générer les positions et rayons des sphères aléatoirement
np.random.seed(0)  # Pour reproductibilité
nb_pix_detector = 1024
size_hologram = nb_pix_detector * taille_pix_holo
distance_plane = 100

nombre_spheres = 20

sph_pos_X = np.random.randint(low= rayon_particules +1, high= size_hologram - rayon_particules -1, size = nombre_spheres)  # Choisir les dimensions du volume ici
sph_pos_Y = np.random.randint(low= rayon_particules +1, high= size_hologram - rayon_particules -1, size = nombre_spheres)  # Choisir les dimensions du volume ici
sph_pos_Z = np.random.randint(low= rayon_particules +1, high= distance_plane - rayon_particules -1, size = nombre_spheres)  # Choisir les dimensions du volume ici
liste_positions = []
list_spheres = []

for i in range(nombre_spheres):
    list_spheres.append(Sphere(n = indice_particules, r = rayon_particules, center=(sph_pos_X[i], sph_pos_Y[i], sph_pos_Z[i])))
    liste_positions.append([sph_pos_X[i], sph_pos_Y[i], sph_pos_Z[i]])

print(list_spheres)

with open('test_simu_holopy.xls', 'w') as fichier:
    # Parcourir chaque ligne dans la liste
    for ligne in liste_positions:
        # Convertir chaque élément de la ligne en chaîne et joindre avec des tabulations
        ligne_formatee = '\t'.join(str(element) for element in ligne)
        # Écrire la ligne formatée dans le fichier
        fichier.write(ligne_formatee + '\n')


spheres = Spheres(list_spheres)

detector = hp.detector_grid(shape = nb_pix_detector, spacing = taille_pix_holo)

print("début calcul hologramme")

holo = calc_holo(detector = detector, scatterer  = spheres, medium_index = indice_milieu,
                 illum_wavelen = longueur_onde, illum_polarization = illum_polarization, theory='auto')


hp.save_image('test_simu_holopy.bmp', holo)



#hp.show(holo)
data_2d=holo.squeeze(dim='z', drop='True')
plt.imshow(data_2d,cmap='gray')
plt.axis('off')
plt.show()
a=1