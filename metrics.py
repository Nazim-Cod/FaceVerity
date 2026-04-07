"""
metrics.py
----------
Calcul FAR, FRR, courbes ROC, AUC sur les paires LFW.
Compatible TensorFlow 2.13 / NumPy / scikit-learn.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm


# ─── Calcul des scores sur LFW ───────────────────────────────────────────────────

def compute_lfw_scores(model, get_embedding_fn, faces1, faces2, labels, batch_size=64):
    """
    Calcule les scores de similarité cosinus sur toutes les paires LFW.

    Args:
        model            : modèle Keras
        get_embedding_fn : fonction(model, face_array) → np.ndarray
        faces1           : np.ndarray [N, H, W, 3]
        faces2           : np.ndarray [N, H, W, 3]
        labels           : np.ndarray [N] (1=même, 0=différent)

    Returns:
        scores : np.ndarray [N]
        labels : np.ndarray [N]
    """
    all_scores = []
    n = len(faces1)

    print(f"  Calcul embeddings sur {n} paires...")
    for i in tqdm(range(0, n, batch_size)):
        batch1 = faces1[i:i+batch_size]
        batch2 = faces2[i:i+batch_size]

        emb1 = model.predict(batch1, verbose=0)
        emb2 = model.predict(batch2, verbose=0)

        # Normalisation L2
        emb1 = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
        emb2 = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)

        # Similarité cosinus par paire
        sims = np.sum(emb1 * emb2, axis=1)
        all_scores.extend(sims.tolist())

    return np.array(all_scores), labels


# ─── FAR / FRR ───────────────────────────────────────────────────────────────────

def compute_far_frr(scores, labels, n_thresholds=200):
    """
    Calcule FAR et FRR pour une gamme de seuils.

    FAR : taux de fausse acceptation (imposteurs acceptés)
    FRR : taux de faux rejet (légitimes rejetés)

    Returns:
        thresholds, FAR_array, FRR_array
    """
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    genuine  = scores[labels == 1]
    impostor = scores[labels == 0]

    FAR = np.array([np.mean(impostor >= t) for t in thresholds])
    FRR = np.array([np.mean(genuine  <  t) for t in thresholds])

    return thresholds, FAR, FRR


def find_eer(FAR, FRR, thresholds):
    """
    Trouve l'Equal Error Rate (point où FAR ≈ FRR).

    Returns:
        eer_threshold, eer_value
    """
    idx = np.argmin(np.abs(FAR - FRR))
    eer = (FAR[idx] + FRR[idx]) / 2.0
    return thresholds[idx], float(eer)


# ─── ROC / AUC ───────────────────────────────────────────────────────────────────

def compute_roc_auc(scores, labels):
    """Calcule la courbe ROC et l'AUC."""
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, float(roc_auc)


# ─── Évaluation complète ─────────────────────────────────────────────────────────

def evaluate_model(model, get_embedding_fn, faces1, faces2, labels,
                   model_name='Modèle'):
    """
    Évaluation complète sur LFW.

    Returns:
        dict avec toutes les métriques
    """
    print(f"\n{'='*55}")
    print(f"  Évaluation : {model_name}")
    print(f"{'='*55}")

    scores, labels = compute_lfw_scores(
        model, get_embedding_fn, faces1, faces2, labels
    )
    thresholds, FAR, FRR = compute_far_frr(scores, labels)
    eer_threshold, eer   = find_eer(FAR, FRR, thresholds)
    fpr, tpr, roc_auc    = compute_roc_auc(scores, labels)

    idx = np.argmin(np.abs(thresholds - eer_threshold))
    print(f"  AUC            : {roc_auc:.4f}")
    print(f"  EER            : {eer*100:.2f}%")
    print(f"  Seuil optimal  : {eer_threshold:.4f}")
    print(f"  FAR @ EER      : {FAR[idx]*100:.2f}%")
    print(f"  FRR @ EER      : {FRR[idx]*100:.2f}%")

    return {
        'name': model_name,
        'scores': scores,
        'labels': labels,
        'thresholds': thresholds,
        'FAR': FAR,
        'FRR': FRR,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc,
        'eer': eer,
        'eer_threshold': eer_threshold,
    }


# ─── Graphiques comparatifs ───────────────────────────────────────────────────────

def plot_all_metrics(results_list, save_path='./evaluation/results_comparison.png'):
    """
    Génère un tableau de bord comparatif :
    - Courbes ROC
    - Courbes FAR/FRR
    - Distribution des scores de similarité
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    colors = ['#2563EB', '#DC2626']

    fig = plt.figure(figsize=(18, 5))
    fig.patch.set_facecolor('#0F172A')
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    for ax in axes:
        ax.set_facecolor('#1E293B')
        ax.tick_params(colors='#94A3B8')
        ax.xaxis.label.set_color('#94A3B8')
        ax.yaxis.label.set_color('#94A3B8')
        ax.title.set_color('#F1F5F9')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')

    # ── 1. Courbes ROC ──
    ax = axes[0]
    ax.set_title('Courbes ROC', fontsize=13, pad=10)
    ax.plot([0, 1], [0, 1], 'w--', alpha=0.3, lw=1)
    for res, color in zip(results_list, colors):
        ax.plot(res['fpr'], res['tpr'], color=color, lw=2.5,
                label=f"{res['name']} (AUC={res['auc']:.4f})")
    ax.set_xlabel('FAR (False Positive Rate)')
    ax.set_ylabel('1 - FRR (True Positive Rate)')
    ax.legend(facecolor='#1E293B', edgecolor='#334155',
              labelcolor='#F1F5F9', fontsize=9)
    ax.grid(alpha=0.15, color='#475569')

    # ── 2. FAR / FRR vs seuil ──
    ax = axes[1]
    ax.set_title('FAR / FRR vs Seuil', fontsize=13, pad=10)
    for res, color in zip(results_list, colors):
        ax.plot(res['thresholds'], res['FAR'] * 100, color=color,
                lw=2, linestyle='-',  label=f"FAR {res['name']}")
        ax.plot(res['thresholds'], res['FRR'] * 100, color=color,
                lw=2, linestyle='--', label=f"FRR {res['name']}", alpha=0.75)
        ax.axvline(res['eer_threshold'], color=color, lw=1,
                   linestyle=':', alpha=0.6)
        ax.annotate(f"EER={res['eer']*100:.1f}%",
                    (res['eer_threshold'], res['eer'] * 100),
                    xytext=(5, 5), textcoords='offset points',
                    color=color, fontsize=8)
    ax.set_xlabel('Seuil de décision')
    ax.set_ylabel('Taux (%)')
    ax.legend(facecolor='#1E293B', edgecolor='#334155',
              labelcolor='#F1F5F9', fontsize=8)
    ax.grid(alpha=0.15, color='#475569')

    # ── 3. Distribution des scores ──
    ax = axes[2]
    ax.set_title('Distribution des Scores', fontsize=13, pad=10)
    for res, color in zip(results_list, colors):
        genuine  = res['scores'][res['labels'] == 1]
        impostor = res['scores'][res['labels'] == 0]
        ax.hist(genuine,  bins=50, alpha=0.5, color=color,
                label=f"Genuine {res['name']}", density=True)
        ax.hist(impostor, bins=50, alpha=0.25, color=color,
                label=f"Impostor {res['name']}", density=True,
                histtype='step', lw=2, linestyle='--')
    ax.set_xlabel('Score de similarité cosinus')
    ax.set_ylabel('Densité')
    ax.legend(facecolor='#1E293B', edgecolor='#334155',
              labelcolor='#F1F5F9', fontsize=8)
    ax.grid(alpha=0.15, color='#475569')

    plt.suptitle('ArcFace vs FaceNet — Benchmark LFW (TensorFlow 2.13)',
                 color='#F1F5F9', fontsize=14, y=1.02)

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\n[Métriques] Graphiques sauvegardés → {save_path}")
    plt.close()
