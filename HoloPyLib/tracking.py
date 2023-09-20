import numpy as np
import trackpy as tp
import trackpy.diag as dg
import numba
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
def link(path, srch_rg, mem):
    
    features = pd.read_csv(path)

    pos_columns=['xum', 'yum', 'zum']

    trajets = tp.link(features, search_range = srch_rg,pos_columns=['x', 'y', 'z'],
             t_column='frame', memory = mem, predictor=None,
             adaptive_stop=None, adaptive_step=0.95,
             neighbor_strategy=None, link_strategy='auto',
             dist_func=None, to_eucl=None)
    
    return(trajets)
"""
    

def filter_stubs(min, trajets):

    return(tp.filtering.filter_stubs(trajets, threshold=min))
    
def plot(trajets):
    ax = tp.plot_traj3d(traj = trajets)

if __name__ == "__main__":
    path_file = r'C:\\TRAVAIL\\developpement\\imagesHolo\\1000im_manip3\\HOLO_2023_06_28_11_56_10.csv'
    file, ext = os.path.splitext(path_file)
    result_file_path = file + '_linked_' + ext


    trajets = 0
    trajetsFiltre = 0
    srch_rg = (2.0,2.0,2.0)
    mem = 1
    trajet_min = 0
    objets = pd.read_csv(file)
    objets.columns= ["frame","x","y","z", "nb_pix" ]
    trajets = tp.link(f = objets, search_range = (1e-6, 1e-6,1e-6), memory  = 1, t_column = 'frame', pos_columns = ['x', 'y', 'z'])
    trajets_filtre = tp.filter_stubs(tracks = trajets, threshold = 3)
    ax = tp.plot_traj3d(trajets_filtre)