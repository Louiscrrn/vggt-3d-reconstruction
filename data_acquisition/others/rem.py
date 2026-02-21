import os
import shutil

def clean_and_sync_eth3d(root_dir):
    """
    1. Supprime les dossiers parasites et remonte les fichiers.
    2. Supprime les images sans masques et les masques sans images.
    """
    print(f"--- Début du traitement dans : {root_dir} ---")
    
    # --- ÉTAPE 1 : Nettoyage de la structure ---
    targets = {
        "dslr_images_undistorted": "images",
        "dslr_images": "masks_for_images"
    }

    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in dirs:
            if name in targets:
                source_path = os.path.join(root, name)
                parent_path = os.path.dirname(source_path)
                
                if os.path.basename(parent_path) == targets[name]:
                    for file_name in os.listdir(source_path):
                        shutil.move(os.path.join(source_path, file_name), 
                                    os.path.join(parent_path, file_name))
                    os.rmdir(source_path)
                    print(f"[Structure] Dossier parasite supprimé : {source_path}")

    # --- ÉTAPE 2 : Vérification de correspondance (Sync) ---
    print("\n--- Vérification des correspondances Images/Masques ---")
    
    scenes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for scene in scenes:
        scene_path = os.path.join(root_dir, scene)
        img_dir = os.path.join(scene_path, "images")
        mask_dir = os.path.join(scene_path, "masks_for_images")

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            print(f"[Attention] Dossiers manquants dans la scène {scene}. Passage.")
            continue

        # On récupère les sets de noms de fichiers (sans extension si elles diffèrent)
        # Note : On suppose que l'image '001.jpg' correspond au masque '001.png' ou '001.jpg'
        images = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))}
        masks = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))}

        # 1. Images sans masques
        for img_name_no_ext, full_name in images.items():
            if img_name_no_ext not in masks:
                file_to_del = os.path.join(img_dir, full_name)
                os.remove(file_to_del)
                print(f"[{scene}] Image supprimée (pas de masque) : {full_name}")

        # 2. Masques sans images
        for mask_name_no_ext, full_name in masks.items():
            if mask_name_no_ext not in images:
                file_to_del = os.path.join(mask_dir, full_name)
                os.remove(file_to_del)
                print(f"[{scene}] Masque supprimé (pas d'image) : {full_name}")

    print("\n--- Tout est propre ! Prêt pour l'upload. ---")

if __name__ == "__main__":
    # Chemin vers ton dossier local
    path_to_eth3d = "../data/eth3d_clean" 
    
    if os.path.exists(path_to_eth3d):
        clean_and_sync_eth3d(path_to_eth3d)
    else:
        print(f"Erreur : Le dossier {path_to_eth3d} n'existe pas.")