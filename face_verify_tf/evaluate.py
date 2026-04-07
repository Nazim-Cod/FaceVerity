"""
evaluate.py
-----------
Évalue ArcFace et FaceNet sur les paires LFW.
Génère FAR, FRR, ROC et AUC comparatifs.

Usage :
    python evaluate.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from data.dataset_loader import get_lfw_pairs_dataset
from models.arcface_model import load_arcface_embedding, get_embedding as arcface_emb
from models.facenet_model import load_facenet_embedding, get_embedding as facenet_emb
from evaluation.metrics import evaluate_model, plot_all_metrics


def main():
    print("\n" + "="*55)
    print("  ÉVALUATION — ArcFace vs FaceNet sur LFW")
    print("="*55)
    print(f"  TensorFlow : {tf.__version__}")
    print(f"  GPU dispo  : {tf.config.list_physical_devices('GPU')}")

    # ── Chargement des paires LFW ──
    print("\n📂 Chargement des paires LFW...")
    faces1, faces2, labels = get_lfw_pairs_dataset(root='./data/lfw')
    print(f"  → {len(labels)} paires | {labels.sum()} genuine | {(labels==0).sum()} impostor")

    # ── Chargement des modèles ──
    print("\n🔧 Chargement des modèles...")
    arcface = load_arcface_embedding('./models/arcface_embedding.h5')
    facenet = load_facenet_embedding('./models/facenet_embedding.h5')

    # ── Évaluation ──
    res_arc = evaluate_model(
        model=arcface,
        get_embedding_fn=arcface_emb,
        faces1=faces1,
        faces2=faces2,
        labels=labels,
        model_name='ArcFace'
    )

    res_fn = evaluate_model(
        model=facenet,
        get_embedding_fn=facenet_emb,
        faces1=faces1,
        faces2=faces2,
        labels=labels,
        model_name='FaceNet'
    )

    # ── Tableau comparatif ──
    print("\n" + "="*55)
    print("  COMPARAISON FINALE — ArcFace vs FaceNet (LFW)")
    print("="*55)
    print(f"  {'Métrique':<20} {'ArcFace':>12} {'FaceNet':>12}")
    print("  " + "-"*46)
    for key, label, mult in [
        ('auc', 'AUC',     1),
        ('eer', 'EER (%)', 100),
    ]:
        va = res_arc[key] * mult
        vf = res_fn[key]  * mult
        better = '← ✅' if va > vf else ''
        print(f"  {label:<20} {va:>12.4f} {vf:>12.4f}  {better}")
    print("="*55)

    # ── Graphiques ──
    print("\n📊 Génération des graphiques comparatifs...")
    plot_all_metrics(
        [res_arc, res_fn],
        save_path='./evaluation/results_comparison.png'
    )

    print("\n✅ Évaluation terminée !")
    print("   → Résultats : ./evaluation/results_comparison.png")
    print("   → GUI       : streamlit run gui/app.py")


if __name__ == '__main__':
    main()
