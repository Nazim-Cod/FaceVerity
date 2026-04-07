"""
face_utils.py
-------------
Détection et alignement des visages avec MTCNN (compatible TensorFlow 2.13).
La librairie `mtcnn` est native TensorFlow, pas PyTorch.
"""

import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN

# ─── Initialisation MTCNN ────────────────────────────────────────────────────────

detector = MTCNN()   # utilise TensorFlow en backend


# ─── Taille cible des visages ────────────────────────────────────────────────────

TARGET_SIZE = (160, 160)


# ─── Fonctions utilitaires ───────────────────────────────────────────────────────

def detect_and_align(image, target_size=TARGET_SIZE):
    """
    Détecte et aligne le visage principal dans une image.

    Args:
        image       : np.ndarray (RGB) ou PIL.Image
        target_size : (hauteur, largeur) du visage recadré

    Returns:
        face_array : np.ndarray [H, W, 3] float32 normalisé [-1, 1], ou None
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = image[:, :, :3]

    results = detector.detect_faces(image)

    if not results:
        return None

    # Prendre le visage avec la plus grande boîte
    best = max(results, key=lambda r: r['box'][2] * r['box'][3])
    x, y, w, h = best['box']

    # Sécurisation des coordonnées
    x, y = max(0, x), max(0, y)
    x2 = min(image.shape[1], x + w)
    y2 = min(image.shape[0], y + h)

    face = image[y:y2, x:x2]
    if face.size == 0:
        return None

    face_resized = cv2.resize(face, (target_size[1], target_size[0]))

    # Normalisation [-1, 1] (standard FaceNet/ArcFace)
    face_norm = (face_resized.astype(np.float32) - 127.5) / 128.0
    return face_norm


def preprocess_from_path(image_path):
    """
    Charge une image depuis un chemin et extrait le visage.

    Returns:
        face_array np.ndarray ou None
    """
    try:
        img = Image.open(image_path).convert('RGB')
        return detect_and_align(np.array(img))
    except Exception as e:
        print(f"[ERREUR] {image_path} : {e}")
        return None


def preprocess_from_bytes(file_bytes):
    """
    Traite une image uploadée (bytes) depuis Streamlit.

    Returns:
        face_array np.ndarray ou None, image PIL originale
    """
    import io
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    face = detect_and_align(np.array(img))
    return face, img


def preprocess_from_frame(frame_bgr):
    """
    Traite une frame BGR depuis OpenCV (webcam).

    Returns:
        face_array np.ndarray ou None
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return detect_and_align(frame_rgb)


def draw_face_boxes(frame_bgr):
    """
    Dessine les boîtes de détection sur une frame webcam.

    Returns:
        frame annotée np.ndarray
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(frame_rgb)

    for face in results:
        x, y, w, h = face['box']
        conf = face['confidence']
        if conf > 0.9:
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 100), 2)
            cv2.putText(frame_bgr, f'{conf:.2f}',
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 100), 2)
    return frame_bgr, results


def compute_cosine_similarity(emb1, emb2):
    """
    Similarité cosinus entre deux embeddings numpy.

    Returns:
        float entre -1 et 1
    """
    emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
    return float(np.dot(emb1, emb2))
