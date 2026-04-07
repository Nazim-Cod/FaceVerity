"""
dataset_loader.py
-----------------
Chargement de CelebA (fine-tuning) et LFW (évaluation) avec TensorFlow/Keras.
"""

import os
import csv
import urllib.request
import tarfile
import zipfile
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from face_utils import detect_and_align


# ─── Constantes ──────────────────────────────────────────────────────────────────

IMG_SIZE   = (160, 160)
AUTOTUNE   = tf.data.AUTOTUNE


# ─── Téléchargement LFW ──────────────────────────────────────────────────────────

LFW_URL        = "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"
LFW_PAIRS_URL  = "http://vis-www.cs.umass.edu/lfw/pairs.txt"


def download_lfw(root='./data/lfw'):
    """Télécharge LFW (images + paires officielles)."""
    os.makedirs(root, exist_ok=True)
    
    # Noms de fichiers possibles
    possible_archives = ['lfw-funneled.tgz', 'lfw-deepfunneled.tgz', 'lfw.tgz']
    possible_pairs = ['pairs.txt', 'pairs.csv']
    
    # Chercher l'archive existante
    archive = None
    for arch_name in possible_archives:
        arch_path = os.path.join(root, arch_name)
        if os.path.exists(arch_path):
            archive = arch_path
            break
    
    # Chercher le fichier pairs existant
    pairs_file = None
    for pairs_name in possible_pairs:
        pairs_path = os.path.join(root, pairs_name)
        if os.path.exists(pairs_path):
            pairs_file = pairs_path
            break

    # Essayer plusieurs URLs alternatives si rien n'existe
    urls_to_try = [
        "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz",
        "https://dl.fbaipublicfiles.com/lfw/lfw-funneled.tgz",
    ]

    # Chercher d'abord un dossier d'images déjà extrait
    img_dir = None
    possible_img_dirs = ['lfw_funneled', 'lfw-deepfunneled', 'lfw']
    for dirname in possible_img_dirs:
        test_dir = os.path.join(root, dirname)
        if os.path.exists(test_dir) and os.listdir(test_dir):
            # Vérifier que c'est bien un dossier d'images avec des sous-dossiers
            subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
            if subdirs:
                img_dir = test_dir
                print(f"[LFW] Dossier d'images trouvé : {dirname}")
                break
    
    # Si pas trouvé, essayer de télécharger et extraire
    if img_dir is None:
        print("[LFW] Recherche des images...")
        download_success = False
        
        if archive is None:
            # Télécharger si aucun fichier n'existe
            for url in urls_to_try:
                try:
                    print(f"[LFW] Tentative de téléchargement depuis {url}...")
                    urllib.request.urlretrieve(url, os.path.join(root, 'lfw-funneled.tgz'))
                    archive = os.path.join(root, 'lfw-funneled.tgz')
                    print("[LFW] Téléchargement réussi!")
                    download_success = True
                    break
                except Exception as e:
                    print(f"[LFW] Échec avec {url}: {e}")
                    continue
        else:
            download_success = True

        if not download_success:
            print("[LFW] ⚠️  Impossible de télécharger automatiquement.")
            print("       → Téléchargez manuellement depuis:")
            print("         https://vis-www.cs.umass.edu/lfw/")
            print("       → Placez lfw-funneled.tgz ou lfw-deepfunneled.tgz dans ./data/lfw/")
            print("       → Et pairs.txt ou pairs.csv dans ./data/lfw/")
            return None, None

        print("[LFW] Extraction...")
        try:
            with tarfile.open(archive) as f:
                f.extractall(root)
            # Après extraction, chercher le dossier d'images
            for dirname in possible_img_dirs:
                test_dir = os.path.join(root, dirname)
                if os.path.exists(test_dir) and os.listdir(test_dir):
                    subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
                    if subdirs:
                        img_dir = test_dir
                        break
        except Exception as e:
            print(f"[LFW] Erreur d'extraction: {e}")
            print("       → Vérifiez que le fichier téléchargé est valide")
            return None, None

        print("[LFW] Extraction...")
        try:
            with tarfile.open(archive) as f:
                f.extractall(root)
        except Exception as e:
            print(f"[LFW] Erreur d'extraction: {e}")
            print("       → Vérifiez que le fichier téléchargé est valide")
            return None, None

    if pairs_file is None:
        pairs_urls = [
            "http://vis-www.cs.umass.edu/lfw/pairs.txt",
            "https://dl.fbaipublicfiles.com/lfw/pairs.txt"
        ]
        for url in pairs_urls:
            try:
                print(f"[LFW] Téléchargement des paires depuis {url}...")
                urllib.request.urlretrieve(url, os.path.join(root, 'pairs.txt'))
                pairs_file = os.path.join(root, 'pairs.txt')
                break
            except Exception as e:
                print(f"[LFW] Échec avec {url}: {e}")
                continue
        else:
            print("[LFW] ⚠️  Impossible de télécharger les paires.")
            print("       → Téléchargez pairs.txt manuellement")
            return img_dir, None

    print(f"[LFW] Prêt dans {root}")
    return img_dir, pairs_file


def parse_lfw_pairs(pairs_file, img_dir):
    """
    Parse le fichier pairs.txt ou pairs.csv de LFW.

    Returns:
        list de (path1, path2, label) où label=1 même personne, 0 sinon
    """
    pairs = []
    
    # Déterminer le séparateur selon l'extension
    separator = '\t' if pairs_file.endswith('.txt') else ','
    
    with open(pairs_file, 'r') as f:
        lines = f.readlines()

    # Détecter le format du fichier
    first_line = lines[0].strip().split(separator)
    
    # Si la première ligne contient des nombres, c'est le format traditionnel
    try:
        n_pairs = int(first_line[1])
        start_line = 1
        print(f"[LFW] Format traditionnel détecté : {n_pairs} paires attendues")
    except (ValueError, IndexError):
        # Sinon, c'est un format avec header
        start_line = 1  # Sauter la ligne d'en-tête
        print("[LFW] Format CSV avec header détecté")

    for line in lines[start_line:]:
        if not line.strip():  # Ignorer les lignes vides
            continue
            
        parts = line.strip().split(separator)
        
        if len(parts) == 3:
            # Même personne : name, n1, n2
            try:
                name, n1, n2 = parts
                p1 = os.path.join(img_dir, name, f"{name}_{int(n1):04d}.jpg")
                p2 = os.path.join(img_dir, name, f"{name}_{int(n2):04d}.jpg")
                if os.path.exists(p1) and os.path.exists(p2):
                    pairs.append((p1, p2, 1))
            except (ValueError, IndexError) as e:
                print(f"[LFW] Erreur ligne même personne : {line.strip()} - {e}")
                continue
                
        elif len(parts) == 4:
            # Personnes différentes : name1, n1, name2, n2
            try:
                name1, n1, name2, n2 = parts
                p1 = os.path.join(img_dir, name1, f"{name1}_{int(n1):04d}.jpg")
                p2 = os.path.join(img_dir, name2, f"{name2}_{int(n2):04d}.jpg")
                if os.path.exists(p1) and os.path.exists(p2):
                    pairs.append((p1, p2, 0))
            except (ValueError, IndexError) as e:
                print(f"[LFW] Erreur ligne personnes différentes : {line.strip()} - {e}")
                continue

    print(f"[LFW] {len(pairs)} paires valides chargées depuis {os.path.basename(pairs_file)}")
    return pairs


def load_and_preprocess_image(path):
    """Charge et prétraite une image LFW avec MTCNN."""
    import cv2
    from PIL import Image as PILImage
    img = PILImage.open(path).convert('RGB')
    face = detect_and_align(np.array(img))
    if face is None:
        # Fallback : redimensionner sans détection
        img_resized = img.resize((160, 160))
        face = (np.array(img_resized).astype(np.float32) - 127.5) / 128.0
    return face


def get_lfw_pairs_dataset(root='./data/lfw', batch_size=32):
    """
    Prépare les paires LFW sous forme de numpy arrays.

    Returns:
        pairs_data : liste de (face1, face2, label) — numpy arrays
    """
    img_dir, pairs_file = download_lfw(root)

    if img_dir is None or pairs_file is None:
        print("[LFW] ⚠️  LFW non disponible.")
        print("       → Téléchargez manuellement depuis:")
        print("         https://vis-www.cs.umass.edu/lfw/")
        print("       → Ou utilisez un autre dataset de test")
        print("       → Le script continuera sans évaluation LFW")
        return None, None, None

    raw_pairs = parse_lfw_pairs(pairs_file, img_dir)

    faces1, faces2, labels = [], [], []
    print("[LFW] Prétraitement des paires...")
    for p1, p2, label in tqdm(raw_pairs):
        f1 = load_and_preprocess_image(p1)
        f2 = load_and_preprocess_image(p2)
        if f1 is not None and f2 is not None:
            faces1.append(f1)
            faces2.append(f2)
            labels.append(label)

    return (
        np.array(faces1, dtype=np.float32),
        np.array(faces2, dtype=np.float32),
        np.array(labels, dtype=np.int32)
    )


# ─── CelebA (Fine-tuning) ────────────────────────────────────────────────────────

CELEBA_URL = "https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM"


def download_celeba_instructions():
    """Affiche les instructions pour télécharger CelebA manuellement."""
    print("""
╔══════════════════════════════════════════════════════╗
║         TÉLÉCHARGEMENT CELEBA (manuel requis)        ║
╠══════════════════════════════════════════════════════╣
║ CelebA nécessite une authentification Google Drive.  ║
║                                                      ║
║ 1. Aller sur :                                       ║
║    https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html║
║ 2. Télécharger : img_align_celeba.zip               ║
║                  identity_CelebA.txt                 ║
║ 3. Placer dans : ./data/celeba/                      ║
║ 4. Relancer : python train.py                        ║
╚══════════════════════════════════════════════════════╝
    """)


def get_celeba_tf_dataset(celeba_dir='./data/celeba',
                          batch_size=32,
                          max_samples=50000,
                          img_size=IMG_SIZE):
    """
    Crée un tf.data.Dataset depuis CelebA pour le fine-tuning.

    Args:
        celeba_dir  : dossier contenant img_align_celeba/ et identity_CelebA.txt
        batch_size  : taille des batches
        max_samples : limite d'images (pour rester rapide)
        img_size    : taille de redimensionnement

    Returns:
        tf.data.Dataset ou None si CelebA non disponible
    """
    # Chercher le dossier d'images avec flexibilité
    img_dir = None
    possible_dirs = [
        os.path.join(celeba_dir, 'img_align_celeba'),
        os.path.join(celeba_dir, 'img_align_celeba', 'img_align_celeba'),
        celeba_dir
    ]
    
    for d in possible_dirs:
        if os.path.exists(d) and len(os.listdir(d)) > 0:
            # Vérifier si ce dossier contient des images
            files = os.listdir(d)
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                img_dir = d
                break
    
    # Chercher le fichier identity
    identity_file = None
    possible_files = [
        os.path.join(celeba_dir, 'identity_CelebA.txt'),
        os.path.join(celeba_dir, 'identity.txt'),
    ]
    
    for f in possible_files:
        if os.path.exists(f):
            identity_file = f
            break

    if img_dir is None or identity_file is None:
        print(f"\n⚠️  ERREUR CELEBA :")
        print(f"  - Dossier images trouvé : {img_dir is not None}")
        print(f"  - Fichier identity trouvé : {identity_file is not None}")
        print(f"  - Chemin cherché : {celeba_dir}")
        download_celeba_instructions()
        return None, 0

    # Lecture des labels d'identité
    image_paths, raw_labels = [], []
    with open(identity_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                fname, identity = parts
                path = os.path.join(img_dir, fname)
                if os.path.exists(path):
                    image_paths.append(path)
                    raw_labels.append(int(identity))

    # Créer un mapping des identités vers des indices consécutifs
    unique_identities = sorted(set(raw_labels))
    identity_to_index = {identity: idx for idx, identity in enumerate(unique_identities)}
    labels = [identity_to_index[identity] for identity in raw_labels]

    # Limiter le nombre d'échantillons
    if max_samples and max_samples < len(image_paths):
        indices = np.random.choice(len(image_paths), max_samples, replace=False)
        image_paths = [image_paths[i] for i in indices]
        labels = [labels[i] for i in indices]

    num_classes = len(unique_identities)
    print(f"[CelebA] {len(image_paths)} images — {num_classes} identités (mapping créé)")

    # tf.data pipeline
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32)
        img = (img - 127.5) / 128.0   # normalisation [-1, 1]
        return img, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        return img, label

    path_ds = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )
    dataset = (path_ds
               .map(load_image, num_parallel_calls=AUTOTUNE)
               .map(augment, num_parallel_calls=AUTOTUNE)
               .shuffle(buffer_size=min(10000, len(image_paths)))
               .batch(batch_size)
               .prefetch(AUTOTUNE))

    return dataset, num_classes
