import numpy as np
from scipy.sparse.linalg import svds
import sklearn
from sklearn.decomposition import TruncatedSVD
from PIL import Image
import os
from tifffile import imread, imsave
from traitement_holo import *


# repertoire courant
image_path = 'C:\\TRAVAIL\\developpement\\HoloTracker\\Images test\\'

# image_path = 'C:\\TRAVAIL\\developpement\\imagesHolo\\20um_bon_40X\\'


svd_min = 5
svd_max = 32
result_directory = image_path + "SVD_min_" + str(svd_min).zfill(2) +"_max_" + str(svd_max).zfill(2) + "\\"

if not os.path.exists(result_directory):
            os.makedirs(result_directory)

result_filename = result_directory + "images_SVD_cleaned_"


type_image = 'bmp'

size_x = 1024
size_y = 1024
size_xy = size_x*size_y
nb_im_svd = 32

# i = 0
# for image in os.listdir(image_path):
#     if (image.split('.')[-1].lower() == type_image.lower()):

#         i+=1

# nb_im_svd = i
i = 0
i_im = 0 #
data = np.full(shape=(size_xy, nb_im_svd), dtype=np.float32, fill_value=0.0)

for image in os.listdir(image_path):
    if (image.split('.')[-1].lower() == type_image.lower()):

        i+=1
        data_1D = read_image(image_path + image, size_x, size_y).flatten()
        data[:,i-1] = data_1D
        print(i)

        if i == nb_im_svd:
        
            # Utiliser Scikit-learn
            svd = TruncatedSVD(n_components=(nb_im_svd-1))
            u_sklearn = svd.fit_transform(data)
            s_sklearn = svd.singular_values_
            vt_sklearn = svd.components_


            # print("Valeurs singulières (Scikit-learn):", s_sklearn)

            # Mettre à zéro les quatre premières (plus grandes) valeurs singulières
            for i in range(s_sklearn.size):

                if i > svd_max or i < svd_min :
                    s_sklearn[i] = 0

            # print("Valeurs singulières filtrée (Scikit-learn):", s_sklearn)  

            # Reconstruire la matrice avec les valeurs singulières modifiées
            reconstructed_matrix = np.dot(u_sklearn * s_sklearn, vt_sklearn)  # Multipliez chaque ligne de U par les valeurs singulières

            # Convertir la matrice reconstruite en float32
            reconstructed_matrix_float32 = reconstructed_matrix.astype(np.float32)

            reconstructed_matrix_float32 = reconstructed_matrix_float32.reshape(size_x, size_y, nb_im_svd)

            for j in range(nb_im_svd):
                image_to_save = Image.fromarray(normalise_to_U8_volume(reconstructed_matrix_float32[:,:,j]))
                image_to_save.save(result_filename + str(i_im).zfill(5) + '.bmp')
                i_im+=1
                print("save image " + str(i_im))

            data = np.full(shape=(size_xy, nb_im_svd), dtype=np.float32, fill_value=0.0)
            i = 0





# Enregistrer la matrice reconstruite dans un fichier RAW
# with open('data_filtered_sclearn.raw', 'wb') as f:
#     f.write(reconstructed_matrix_float32.tobytes())

# # Charger les données depuis le fichier TIFF
# data = imread('im_00000.tif').astype(float)
# size_t= data.shape[0]
# size_y= data.shape[1]
# size_x= data.shape[2]
# size_xy= size_x*size_y
# matrix = data.reshape(size_t, size_xy)

# # Utiliser Scikit-learn
# svd = TruncatedSVD(n_components=(size_t-1))
# u_sklearn = svd.fit_transform(matrix)
# s_sklearn = svd.singular_values_
# vt_sklearn = svd.components_

# print("Valeurs singulières (Scikit-learn):", s_sklearn)

# # Mettre à zéro les quatre premières (plus grandes) valeurs singulières
# s_sklearn[0] = 0
# s_sklearn[1] = 0
# s_sklearn[2] = 0
# s_sklearn[3] = 0

# print("Valeurs singulières filtrée (Scikit-learn):", s_sklearn)  

# # Reconstruire la matrice avec les valeurs singulières modifiées
# reconstructed_matrix = np.dot(u_sklearn * s_sklearn, vt_sklearn)  # Multipliez chaque ligne de U par les valeurs singulières

# # Convertir la matrice reconstruite en float32
# reconstructed_matrix_float32 = reconstructed_matrix.astype(np.float32)

# # Enregistrer la matrice reconstruite dans un fichier RAW
# with open('data_filtered_sclearn.raw', 'wb') as f:
#     f.write(reconstructed_matrix_float32.tobytes())

# #CALCUL AVEC SCIPY

# # Utiliser Scipy pour obtenir les size_t-1 plus  grandes valeurs singulières et vecteurs associés
# u_scipy, s_scipy, vt_scipy = svds(matrix, k=(size_t-1), which='LM')  
# # 'LM' signifie les plus grandes valeurs singulières

# print("Valeurs singulières (Scipy):", s_scipy)
# # Mettre à zéro les quatre premières (plus grandes) valeurs singulières
# # Mettre à zéro les deux dernières (plus grandes) valeurs singulières
# s_scipy[-1] = 0
# s_scipy[-2] = 0
# s_scipy[-3] = 0
# s_scipy[-4] = 0
# print("Valeurs singulières filtrée (Scipy):", s_scipy)  

# # Reconstruire la matrice avec les valeurs singulières modifiées
# reconstructed_matrix = np.dot(u_scipy, np.dot(np.diag(s_scipy[::-1]), vt_scipy))

# # Convertir la matrice reconstruite en float32
# reconstructed_matrix_float32 = reconstructed_matrix.astype(np.float32)

# # Enregistrer la matrice reconstruite dans un fichier RAW
# with open('data_filtered_scipy.raw', 'wb') as f:
#     f.write(reconstructed_matrix_float32.tobytes())

    
# a=1
