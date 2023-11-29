#25/08/2023
#Simon Becker
#script de simulation de propagation d'hologram à travers un volume remplis d'objets (bactéries ou sphères)

# -*- coding: utf-8 -*-

import cupy as cp
import numpy as np


#création d'un volume 3D
#taille du volume (pix)
size_x = 1024
size_y = 1024
size_z = 500
#pitch du pixel et pas de propagation
dxy = 1e-6
dz = 1e-6

lambda_illum = 660e-6
indice_milieu = 1.33

#différence dephasage dans le milieu par rapport à l'eau (par pas de propagation)
dphi = (1.332-1.33) * dz / (lambda_illum / indice_milieu)

#nombre d'objet
n_objet = 1

#une sphère est définie par ses coordonnées xyz, son rayon, son indice relatif (au milieu environnant)
sphere_dtype = np.dtype([
    ('pos_x', np.uint16),
    ('pos_y', np.uint16),
    ('pos_z', np.uint16),
    ('r', np.float32),
    ('attenuation', np.float32),
    ('dephasage', np.float32)
])

liste_sphere = np.array([(512, 512, 500, 1.0, np.pi/10.0)],dtype = sphere_dtype)
d_liste_sphere = cp.asnumpy(liste_sphere)

list_objet = np.empty(dtype = sphere_dtype)

#creation de n_objets alleatoires
for i in range(n_objet):

    ok = False

    while (not ok):
        ok = True
        #generation d'un nouvel objet
        x = np.random.random_integers(0, size_x)
        y = np.random.random_integers(0, size_y)
        y = np.random.random_integers(0, size_z)

        #test si objet est connexe à un précént objet
        for o in range(len(list_objet)):
            #test
            obj = list_objet[o]
            obj['pos_x']



#création d'un volume d'attenuation et de déphasage
milieu_dtype = cp.dtype([
    ('attenuation', cp.float32),
    ('dephasage', cp.float32)
])

att = cp.pi / 10.0

#penser à calculer la valeur du dephasage
h_volume_milieu = np.ndarray(shape = (size_x, size_y, size_z), dtype=milieu_dtype)

print("toto")
