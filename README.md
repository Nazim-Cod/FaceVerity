# 🎯 Reconnaissance Faciale — Vérification d'Identité (TensorFlow)

Ce projet implémente un système de vérification d'identité par reconnaissance faciale en comparant deux approches : **ArcFace** et **FaceNet**. Il repose sur **TensorFlow 2.13** et propose :

- entraînement par fine-tuning sur le dataset **CelebA**
- évaluation de modèle sur le dataset **LFW**
- interface utilisateur avec **Streamlit**

---

## 📌 Présentation

L'objectif est de comparer deux modèles de reconnaissance faciale sur une même base de données :

- **ArcFace** : ResNet-50 + loss angulaire
- **FaceNet** : InceptionResNetV2 + triplet loss

Le projet fournit des scripts pour :

1. charger les datasets
2. entraîner les modèles
3. évaluer les performances
4. tester en mode interactif

---

## 📦 Structure du projet

```
.
├── app.py                # Interface Streamlit
├── arcface_model.py      # Modèle ArcFace + fine-tuning
├── dataset_loader.py     # Chargement CelebA et LFW
├── evaluate.py           # Évaluation automatique LFW
├── facenet_model.py      # Modèle FaceNet + fine-tuning
├── face_utils.py         # Détection/alignment MTCNN
├── metrics.py            # Calcul FAR/FRR/ROC/AUC
├── README.md
├── requirements.txt
├── train.py              # Script de fine-tuning
└── face_verify_tf/       # Version structurée du même projet
```

---

## ✅ Prérequis

- Python 3.8 ou 3.10 recommandé
- TensorFlow 2.13.1
- `pip` à jour
- Dataset **CelebA** téléchargé manuellement
- Dataset **LFW** pour l'évaluation automatique

> Le projet fonctionne aussi sans GPU, mais l'entraînement est plus lent.

---

## 🛠️ Installation

### 1. Créez l'environnement virtuel

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Mettez pip à jour

```bash
pip install --upgrade pip
```

### 3. Installez les dépendances

```bash
pip install -r requirements.txt
```

### 4. Vérifiez l'installation

```bash
python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"
python --version 
```

---

## 🗂️ Datasets

### CelebA

Ce dataset doit être téléchargé manuellement car il requiert une authentification Google Drive.

- Télécharger `img_align_celeba.zip`
- Télécharger `identity_CelebA.txt`
- Extraire `img_align_celeba.zip` dans `./data/celeba/`
- Placer `identity_CelebA.txt` dans `./data/celeba/`

### LFW

L'évaluation automatique nécessite le dataset LFW. Le script accepte :

- `lfw-deepfunneled` extrait
- `pairs.txt` ou `pairs.csv`

Placez les fichiers dans `./data/lfw/`.

---

## 🚀 Utilisation

### 1. Fine-tuning des modèles

```bash
python train.py --model both --epochs 5
```

Options :

- `--model arcface`
- `--model facenet`
- `--model both`
- `--epochs N`
- `--batch_size N`
- `--max_samples N`
- `--celeba_dir ./data/celeba`

### 2. Évaluation automatique LFW

```bash
python evaluate.py
```

### 3. Interface Streamlit

```bash
streamlit run app.py
```

---

## 📈 Résultats attendus

- **Fine-tuning** des embeddings ArcFace et FaceNet
- **Sauvegarde** des poids dans `./models/`
- **Évaluation** des performances sur LFW
- **Interface** pour tester des images ou une webcam

---

## � Captures d'écran

### Interface principale
![Interface principale](screenshots/interface_principale.png)

### Mode upload d'images
![Upload d'images](screenshots/mode_upload.png)

### Mode webcam temps réel
![Webcam temps réel](screenshots /mode_webcam.png)

### Résultats de comparaison
![Résultats](screenshots/resultats.png)

---

## �📊 Métriques

| Métrique | Description |
|----------|-------------|
| **FAR** | False Acceptance Rate — imposteur accepté incorrectement |
| **FRR** | False Rejection Rate — sujet légitime rejeté incorrectement |
| **ROC** | Receiver Operating Characteristic |
| **AUC** | Aire sous la courbe ROC |
| **EER** | Equal Error Rate — point où FAR = FRR |

---

## 💡 Bonnes pratiques

- Utilisez `--max_samples` pendant le développement pour accélérer les tests
- Confirmez que `./data/celeba/` contient `img_align_celeba/` et `identity_CelebA.txt`
- Confirmez que `./data/lfw/` contient `lfw-deepfunneled/` et `pairs.csv` ou `pairs.txt`

---

## 📘 Remarques

- Si LFW n'est pas disponible, le script `evaluate.py` s'arrête proprement et affiche les instructions de téléchargement.
- Si CelebA n'est pas disponible, `train.py` indique à l'utilisateur de télécharger le dataset manuellement.

---
