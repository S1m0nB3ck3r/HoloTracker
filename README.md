# HoloTracker
version 1.0

HoloTracker is an in-line hologram analysis software (developed in Python/Labview) designed to locate micro-objects under a microscope in three dimensions (such as microspheres or bacteria) within a sequence of images to determine their 3D trajectories.

HoloTracker features a user-friendly human-machine interface for simplified usage.

HoloTracker requires an Nvidia GPU and utilizes the Cupy library for hologram backpropagation. For tracking objects once they are localized, HoloTracker utilizes the TrackPy library (soft-matter.github.io/trackpy/).

For testing HoloTracker_Locate without GUI, a script is located at \Code\HoloPyLib\test_HoloTracker_locate.py

Here are the python library/package needed to use HoloTracker.

python                  3.10
cupy-cuda11x            12.2.0  
Cython                  0.29.33  
matplotlib              3.8.0  
numba                   0.56.4  
numpy                   1.23.5  
pandas                  1.5.3  
Pillow                  9.4.0  
PIMS                    0.6.1  
scipy                   1.10.0  
trackpy                 0.6.1  


How to install this sofwtare:

https://github.com/S1m0nB3ck3r/HoloTracker/assets/130139208/3f2520b3-3bbd-4ec6-936a-41fe0caa5623



