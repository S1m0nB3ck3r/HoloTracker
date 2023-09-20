import holopy as hp
from holopy.scattering import Scatterer, Sphere, calc_holo
import numpy as np
s1 = Sphere(r = .5, center = (0, -.4, 0))
s2 = Sphere(r = .5, center = (0, .4, 0))
detector = hp.detector_grid(100, .1)
dumbbell = Scatterer(lambda point: np.logical_or(s1.contains(point), s2.contains(point)),
                     1.59, (5, 5, 5))
holo = calc_holo(detector, dumbbell, medium_index=1.33, illum_wavelen=.66, illum_polarization=(1, 0))