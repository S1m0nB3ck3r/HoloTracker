import numpy as np
import trackpy as tp
import trackpy.diag as dg
import numba
import pandas as pd
import matplotlib.pyplot as plt
import os
from analyse_data import *

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
    path = r'C:\\TRAVAIL\\developpement\\imagesHolo\\20um_bon_40X\\'
    result_filename = 'result_python_sum15_TENEGRAD_STD15_each.csv'
    file = path + result_filename

    objets = pd.read_csv(file)
    pix_min = 200
    objets.columns= ["frame","x","y","z", "nb_pix" ]
    objets_filtres = objets[(objets.nb_pix > pix_min)]
    print(objets.size)
    print(objets_filtres.size)
    trajets = tp.link(f = objets, search_range = (5.0,5.0,5.0), memory  = 5, t_column = 'frame', pos_columns = ['x', 'y', 'z'])

    #ne garde que les trajets plus long qu'un seuil
    # trajets_filtre_l = tp.filter_stubs(tracks = trajets, threshold = 100)
    ax = tp.plot_traj3d(trajets)

    # color_particules = {}
    
    #ne garde que les trajectoires des particules motiles
    # index_trajets_filtre = get_good_index(trajets_filtre_l)

    # #filtre trajets
    # trajets_motiles = [t for t in trajets_filtre_l if t[5] in index_trajets_filtre]

    # data_traj = pd.DataFrame(data = trajets_motiles)
    # data_traj.columns= ["frame","x","y","z", "nb_pix", "particle" ]

    # ax = tp.plot_traj3d(trajets)