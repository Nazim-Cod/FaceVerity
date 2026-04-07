"""
train.py
--------
Fine-tuning ArcFace et FaceNet sur CelebA avec TensorFlow 2.13.

Usage :
    python train.py --model arcface --epochs 5
    python train.py --model facenet --epochs 5
    python train.py --model both    --epochs 5
"""

import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Réduire les logs TF

import tensorflow as tf
from data.dataset_loader import get_celeba_tf_dataset
from models.arcface_model import fine_tune_arcface
from models.facenet_model import fine_tune_facenet


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tuning reconnaissance faciale (TensorFlow)'
    )
    parser.add_argument('--model', choices=['arcface', 'facenet', 'both'],
                        default='both')
    parser.add_argument('--epochs',     type=int,   default=5)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--max_samples',type=int,   default=50000)
    parser.add_argument('--celeba_dir', type=str,   default='./data/celeba')
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  FINE-TUNING — Reconnaissance Faciale (TensorFlow 2.13)")
    print("="*60)
    print(f"  TensorFlow : {tf.__version__}")
    print(f"  GPU dispo  : {tf.config.list_physical_devices('GPU')}")
    print(f"  Modèle(s)  : {args.model}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Max samples: {args.max_samples}")
    print("="*60)

    # ── Chargement CelebA ──
    print("\n📂 Chargement de CelebA...")
    celeba_dataset, num_classes = get_celeba_tf_dataset(
        celeba_dir=args.celeba_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )

    if celeba_dataset is None:
        print("\n⚠️  CelebA non disponible.")
        print("   → Télécharge CelebA et relance le script.")
        print("   → Les modèles utiliseront les poids ImageNet seuls.")
        return

    # ── Fine-tuning ArcFace ──
    if args.model in ('arcface', 'both'):
        fine_tune_arcface(
            celeba_dataset=celeba_dataset,
            num_classes=num_classes,
            epochs=args.epochs,
            save_path='./models/arcface_embedding.h5'
        )

    # ── Fine-tuning FaceNet ──
    if args.model in ('facenet', 'both'):
        fine_tune_facenet(
            celeba_dataset=celeba_dataset,
            epochs=args.epochs,
            save_path='./models/facenet_embedding.h5'
        )

    print("\n✅ Fine-tuning terminé !")
    print("   → Lance maintenant : python evaluate.py")


if __name__ == '__main__':
    main()
