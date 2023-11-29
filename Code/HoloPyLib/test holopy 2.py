#doit être excécuté avec Python 3.7 + toutes les dépendances nécéssaires

import numpy as np
import holopy as hp
from holopy.core.io import get_example_data_path, load_average
from holopy.core.process import bg_correct
import matplotlib.pyplot as plt
import time as ti

imagepath = get_example_data_path('image01.jpg')
raw_holo = hp.load_image(imagepath, spacing = 0.0851, medium_index = 1.33, illum_wavelen = 0.66)
bgpath = get_example_data_path(['bg01.jpg','bg02.jpg','bg03.jpg'])
bg = load_average(bgpath, refimg = raw_holo)
holo = bg_correct(raw_holo, bg)

zstack = np.linspace(0, 20, 400)
t1 = ti.time()
rec_vol = hp.propagate(holo, zstack)
t2 = ti.time()
print("temps = ",t2 -t1, "\n")

hp.show(rec_vol)
plt.show()
a= 1


import numpy as np
import matplotlib.pyplot as plt
from holopy.core import Experiment, Source, PlaneWave, FreederickszTransformation, Optics

# Générer les positions et rayons des sphères aléatoirement
np.random.seed(0)  # Pour reproductibilité
num_spheres = 10
sphere_positions = np.random.rand(num_spheres, 3) * [100, 100, 50]  # Choisir les dimensions du volume ici
sphere_radii = np.random.rand(num_spheres) * 5  # Rayon maximal de 5 unités

# Créer l'objet de l'expérience
exp = Experiment()

# Ajouter la source (onde plane)
source = Source(wavelength=0.632, polarization=(1, 0))
exp.append(source)

# Ajouter la transformation de Freedericksz (pour la propagation)
propagation = FreederickszTransformation()
exp.append(propagation)

# Ajouter les sphères
for i in range(num_spheres):
    sphere = exp.make(Sphere, position=sphere_positions[i], radius=sphere_radii[i])
    exp.append(sphere)

# Ajouter les paramètres de l'optique
optics = Optics(magnification=20)
exp.append(optics)

# Propagation de l'onde
exp.propagate()

# Calcul de l'hologramme
hologram = exp.hologram()

# Afficher l'hologramme
plt.imshow(np.abs(hologram), cmap='gray')
plt.title("Hologramme simulé")
plt.show()