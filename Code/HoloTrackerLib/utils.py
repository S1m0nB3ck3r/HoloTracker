import json

def save_parameters(params, filename='parameters.json'):
    """
    Enregistre les paramètres dans un fichier JSON.

    :param params: Dictionnaire contenant les paramètres à enregistrer.
    :param filename: Nom du fichier où enregistrer les paramètres.
    """
    with open(filename, 'w') as file:
        json.dump(params, file, indent=4)


def load_parameters(filename='parameters.json'):
    """
    Charge les paramètres depuis un fichier JSON.

    :param filename: Nom du fichier depuis lequel charger les paramètres.
    :return: Dictionnaire contenant les paramètres chargés.
    """
    try:
        with open(filename, 'r') as file:
            params = json.load(file)
        return params
    except FileNotFoundError:
        print(f"Aucun fichier de paramètres trouvé: {filename}")
        return None

if __name__ == "__main__":


    # Exemple d'utilisation
    params = {
        'display_images': False,
        'holo_directory': 'Images_test',
        'result_filename': 'result_.csv',
        'image_type': 'bmp',
        "wavelength": 660e-9,  # in m,
        "medium_index": 1.33,  # index of refraction of the medium
        'cam_magnification': 40.0,
        'cam_nb_pix_X': 1024,
        'cam_nb_pix_Y': 1024,
        'nb_plane': 200,
        'cam_pix_size': 5.5e-6,
        'plane_step': 0.5e-6,
        'focus_smooth_size': 15,
        'threshold_value': 10,
        'type_Threshold': 'NB_STDVAR', #or VALUE
        'n_connectivity': 26,
        'particle_filter_size': 0
    }

    save_parameters(params)
    loaded_params = load_parameters()
    print(loaded_params)        
