import numpy as np
import trackpy as tp
import trackpy.diag as dg
import numba
import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = ""
features = pd.DataFrame()
filtered_features = pd.DataFrame()
trajectories = pd.DataFrame()
filtered_trajectories = pd.DataFrame()
search_r = (0.0,0.0,0.0)
memo = 0

def read_features_positions_file(path):
    global features
    global file_path

    file_path = path
    features = pd.read_csv(file_path, sep = ",", decimal = ".")
    features.columns= ["frame","x","y","z", "nb_pix" ]
    return features.values.tolist()

def get_features():
    global features
    return features.values.tolist()

def get_filtered_features_by_frame(frame_number):
    global features
    filitered = features[features['frame']==frame_number]
    return filitered.values.tolist()

def filtered_features_by_size(voxel_size):
    global features
    global filtered_features
    filtered_features = features[features['nb_pix']>voxel_size]
    return filtered_features.values.tolist()

def set_features(features_np_array):
    global features
    features = pd.DataFrame(features_np_array, columns = ["frame","x","y","z", "nb_pix" ])
    return features_np_array.tolist()

def link(srch_range, mem):
    global features
    global trajectories
    global filtered_trajectories
    global search_r
    global memo
    search_r = srch_range
    memo = mem
    try:
        trajectories = tp.link(f = features, search_range = search_r, memory  = memo, t_column = 'frame', pos_columns = ['x', 'y', 'z'])
    except:
        return
    else:
        filtered_trajectories = trajectories
        filtered_trajectories.columns= ["frame","x","y","z", "nb_pix","particle" ]
        filtered_trajectories = filtered_trajectories.sort_values(by=['particle', 'frame'])

        return filtered_trajectories.values.tolist()

def filter_trajectories_by_lenght(min):
    global trajectories
    global filtered_trajectories
    global search_r 
    global memo

    filtered_trajectories = tp.filtering.filter_stubs(trajectories, threshold=min)
    filtered_trajectories = tp.link(f = filtered_trajectories, search_range = search_r, memory  = memo, t_column = 'frame', pos_columns = ['x', 'y', 'z'])
    filtered_trajectories.columns= ["frame","x","y","z", "nb_pix","particle" ]
    filtered_trajectories = filtered_trajectories.sort_values(by=['particle', 'frame'])
    
    return filtered_trajectories.values.tolist()

def get_one_trajectorie(feature_number):
    global filtered_trajectories
    filtered_trajectories.columns= ["frame","x","y","z", "nb_pix", "particle" ]
    trajectorie = filtered_trajectories[filtered_trajectories['particle']==feature_number]
    return trajectorie.values.tolist()

def get_trajectories_portion(frame_start, frame_stop):
    global filtered_trajectories
    filtered_trajectories.columns= ["frame","x","y","z", "nb_pix", "particle" ]
    trajectorie = filtered_trajectories[(filtered_trajectories['frame'] >= frame_start) & (filtered_trajectories['frame'] <= frame_stop)]
    traj_sorted = trajectorie.sort_values(by=['particle'])
    return traj_sorted.values.tolist()

def get_all_trajectories():
    global filtered_trajectories
    # df.sort_values(by=['col1'])
    traj_sorted = filtered_trajectories.sort_values(by=['particle', 'frame'])
    return traj_sorted.values.tolist()

def save_trajectories(path_traj):
    file, ext = os.path.splitext(path_traj)
    result_file_path = file + '_linked_' + ext
    filtered_trajectories.to_csv(result_file_path, mode = 'a')

def get_frame_max():
    global filtered_trajectories
    return filtered_trajectories['frame'].idxmax()

def test_func():
    global filtered_trajectories




        


if __name__ == "__main__":

    path_file = r'C:\\TRAVAIL\developpement\\imagesHolo\\1000im_manip3\\HOLO_2023_06_28_11_56_10.csv'
    file, ext = os.path.splitext(path_file)
    result_file_path = file + '_linked_' + ext

    test_numpy = read_features_positions_file(path_file)
    trajets = 0
    trajetsFiltre = 0
    srch_rg = (2.0,2.0,2.0)
    mem = 1
    trajet_min = 0
    objets = pd.read_csv(path_file)
    objets.columns= ["frame","x","y","z", "nb_pix" ]
    trajets = tp.link(f = objets, search_range = (1e-6, 1e-6,1e-6), memory  = 1, t_column = 'frame', pos_columns = ['x', 'y', 'z'])
    trajets_filtre = tp.filter_stubs(tracks = trajets, threshold = 3)
    trajets = tp.link(f = trajets_filtre, search_range = (1e-6, 1e-6,1e-6), memory  = 1, t_column = 'frame', pos_columns = ['x', 'y', 'z'])
    trajets.columns= ["frame","x","y","z", "nb_pix","particle" ]
    traj_sorted = trajets.sort_values(by=['particle', 'frame'])
    test = trajets['particle'].unique()

    print(test)
    ax = tp.plot_traj3d(trajets_filtre)