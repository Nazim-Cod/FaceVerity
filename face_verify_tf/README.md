# 🎯 Reconnaissance Faciale — Vérification d'Identité (TensorFlow)

Système de vérification d'identité comparant **ArcFace** et **FaceNet** implémentés avec **TensorFlow 2.13**.

---

## ⚠️ Compatibilité Python / TensorFlow

| Python | TensorFlow max supporté | Statut |
|--------|--------------------------|--------|
| 3.8    | **2.13.1** ✅            | Fonctionne (ce projet) |
| 3.9    | 2.13.1                   | ✅ OK |
| **3.10** | 2.13 → 2.17            | ✅ **Recommandé** |
| 3.11   | 2.13 → 2.17              | ✅ OK |
| 3.12+  | 2.16+                    | ⚠️ Partiel |

> **Si tu as Python 3.8** : ce projet fonctionne directement avec `tensorflow==2.13.1`
> **Sinon** : télécharge Python 3.10.11 sur https://www.python.org/downloads/release/python-31011/

---

## 📁 Structure du Projet

```
face_verify_tf/
├── models/
│   ├── arcface_model.py      # ArcFace avec ArcFace Loss (TF/Keras)
│   └── facenet_model.py      # FaceNet InceptionResnetV1 (TF/Keras)
├── data/
│   └── dataset_loader.py     # Chargement CelebA + LFW
├── evaluation/
│   └── metrics.py            # FAR, FRR, ROC, AUC + graphiques
├── gui/
│   └── app.py                # Interface Streamlit
├── utils/
│   └── face_utils.py         # Détection MTCNN + alignement
├── train.py                  # Script fine-tuning
├── evaluate.py               # Script évaluation LFW
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation complète (étape par étape)

### Étape 1 — Vérifier ta version Python
```bash
python --version
# doit afficher Python 3.8.x ou 3.10.x
```

### Étape 2 — Créer un environnement virtuel
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python3 -m venv venv
source venv/bin/activate
```

### Étape 3 — Mettre à jour pip
```bash
pip install --upgrade pip
```

### Étape 4 — Installer les dépendances
```bash
pip install -r requirements.txt
```

### Étape 5 — Vérifier TensorFlow
```bash
python -c "import tensorflow as tf; print('TF version:', tf.__version__)"
# Doit afficher : TF version: 2.13.1
```

### Étape 6 (optionnel) — Vérifier le GPU
```bash
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

---

## 🚀 Utilisation

```bash
# 1. Télécharger les datasets et fine-tuner les modèles
python train.py --model both --epochs 5

# 2. Évaluer sur LFW (FAR / FRR / ROC)
python evaluate.py

# 3. Lancer l'interface graphique
streamlit run gui/app.py
```

---

## 📊 Métriques évaluées

| Métrique | Description |
|----------|-------------|
| **FAR** | False Acceptance Rate — un imposteur est accepté |
| **FRR** | False Rejection Rate — une vraie personne est rejetée |
| **ROC** | Courbe Receiver Operating Characteristic |
| **AUC** | Aire sous la courbe ROC (1.0 = parfait) |
| **EER** | Equal Error Rate (point où FAR = FRR) |

---

## 🔧 Configuration matérielle requise

| Composant | Minimum | Recommandé |
|-----------|---------|------------|
| RAM | 4 GB | 8 GB |
| Stockage | 5 GB | 10 GB |
| GPU | Non requis | NVIDIA CUDA ≥ 3.5 |
| OS | Windows 10 / Ubuntu 18+ / macOS 10.14+ | Ubuntu 20.04 |
