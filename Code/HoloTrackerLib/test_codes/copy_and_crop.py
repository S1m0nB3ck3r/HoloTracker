import os
from PIL import Image
from pathlib import Path

def crop_center(image, crop_width, crop_height):
    """Recadre l'image au centre selon les dimensions spécifiées."""
    img_width, img_height = image.size
    left = (img_width - crop_width) // 2
    top = (img_height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    return image.crop((left, top, right, bottom))

def process_images(source_dir, dest_dir, crop_width, crop_height):
    """
    Copie toutes les images du répertoire source vers le répertoire de destination
    après les avoir recadrées.
    
    :param source_dir: Répertoire source contenant les images
    :param dest_dir: Répertoire de destination
    :param crop_width: Largeur du recadrage
    :param crop_height: Hauteur du recadrage
    """
    # Vérifier et créer le répertoire de destination s'il n'existe pas
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    for file_name in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file_name)

        # Vérifier si le fichier est une image
        if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            try:
                with Image.open(file_path) as img:
                    # Recadrer l'image
                    cropped_img = crop_center(img, crop_width, crop_height)

                    # Sauvegarder l'image recadrée dans le répertoire de destination
                    dest_path = os.path.join(dest_dir, file_name)
                    cropped_img.save(dest_path)

                    print(f"Image {file_name} traitée et copiée.")
            except Exception as e:
                print(f"Erreur lors du traitement de l'image {file_name}: {e}")

if __name__ == "__main__":
    # Spécifiez les chemins des répertoires source et destination
    source_directory = "C:\TRAVAIL\RepositoriesGithub\HoloTracker\Images test"
    destination_directory = "C:\TRAVAIL\RepositoriesGithub\HoloTracker\Images test 2"

    # Définir les dimensions du recadrage
    crop_width = 1024  # Largeur du recadrage
    crop_height = 512 # Hauteur du recadrage

    # Exécuter la fonction principale
    process_images(source_directory, destination_directory, crop_width, crop_height)
