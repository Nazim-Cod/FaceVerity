"""
facenet_model.py
----------------
Implémentation FaceNet avec TensorFlow 2.13 / Keras.
Backbone  : InceptionResnetV1 (poids VGGFace2 via weights .h5)
Loss      : Triplet Loss (semi-hard mining)
"""

import os
import numpy as np
import requests
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import InceptionResNetV2


# ─── Triplet Loss ─────────────────────────────────────────────────────────────────

class TripletLoss(keras.losses.Loss):
    """
    Triplet Loss online (sans mining explicite).
    Minimise dist(ancre, positive) et maximise dist(ancre, négative).
    """

    def __init__(self, margin=0.2, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, embeddings):
        """
        Args:
            y_true     : labels [B] (identités)
            embeddings : [B, embedding_size]
        """
        # Distances euclidiennes pairées
        dot_product = tf.matmul(embeddings, embeddings, transpose_b=True)
        sq_norm = tf.linalg.diag_part(dot_product)
        distances = tf.maximum(
            sq_norm[:, None] - 2.0 * dot_product + sq_norm[None, :], 0.0
        )
        distances = tf.sqrt(distances + 1e-8)

        labels = tf.cast(tf.squeeze(y_true), tf.int32)
        mask_pos = tf.equal(labels[:, None], labels[None, :])
        mask_neg = tf.logical_not(mask_pos)
        mask_pos = tf.cast(mask_pos, tf.float32) - tf.eye(tf.shape(labels)[0])

        # Triplet loss semi-hard
        pos_dist = tf.reduce_max(distances * mask_pos, axis=1)
        neg_dist = tf.reduce_min(
            distances * tf.cast(mask_neg, tf.float32) + 1e9 * tf.cast(mask_pos, tf.float32),
            axis=1
        )
        loss = tf.maximum(pos_dist - neg_dist + self.margin, 0.0)
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({'margin': self.margin})
        return config


# ─── Construction du modèle FaceNet ──────────────────────────────────────────────

def build_facenet_model(embedding_size=512, input_shape=(160, 160, 3)):
    """
    Construit FaceNet basé sur InceptionResNetV2 (pré-entraîné ImageNet).

    Note : InceptionResNetV1 original de FaceNet n'est pas disponible directement
           dans Keras. On utilise InceptionResNetV2 comme backbone équivalent.

    Architecture :
        InceptionResNetV2 → GlobalAvgPool → Dense(512) → L2 Norm

    Returns:
        model Keras
    """
    img_input = keras.Input(shape=input_shape, name='image_input')

    # ── Backbone InceptionResNetV2 (pré-entraîné ImageNet) ──
    backbone = InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=img_input,
        pooling='avg'
    )

    # Geler les premières couches, fine-tuner les dernières
    for layer in backbone.layers[:-30]:
        layer.trainable = False

    x = backbone.output

    # ── Tête d'embedding ──
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(embedding_size, use_bias=False, name='embedding')(x)
    x = layers.BatchNormalization(name='bn_embedding')(x)

    # Normalisation L2
    embeddings = layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=1),
        name='l2_norm'
    )(x)

    model = Model(inputs=img_input, outputs=embeddings, name='FaceNet')
    return model


# ─── Fine-tuning avec Triplet Loss ───────────────────────────────────────────────

def fine_tune_facenet(celeba_dataset, epochs=5,
                      save_path='./models/facenet_embedding.h5'):
    """
    Fine-tune FaceNet sur CelebA avec Triplet Loss.

    Args:
        celeba_dataset : tf.data.Dataset retournant (image, label)
        epochs         : nombre d'epochs
        save_path      : chemin de sauvegarde

    Returns:
        model Keras
    """
    print(f"\n{'='*55}")
    print(f"  FINE-TUNING FACENET — {epochs} epochs (Triplet Loss)")
    print(f"{'='*55}")

    model = build_facenet_model()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss=TripletLoss(margin=0.2),
        metrics=[]
    )

    model.summary()

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=2, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            save_path.replace('.h5', '_best.h5'),
            monitor='loss', save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='loss', patience=4, restore_best_weights=True
        )
    ]

    history = model.fit(
        celeba_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    model.save(save_path)
    print(f"\n[FaceNet] Modèle sauvegardé → {save_path}")
    return model


# ─── Chargement du modèle ────────────────────────────────────────────────────────

def load_facenet_embedding(weights_path=None):
    """
    Charge le modèle FaceNet en mode inférence.

    Args:
        weights_path : chemin vers le .h5 sauvegardé, ou None

    Returns:
        model Keras
    """
    if weights_path and os.path.exists(weights_path):
        print(f"[FaceNet] Chargement depuis {weights_path}")
        model = keras.models.load_model(
            weights_path,
            custom_objects={'TripletLoss': TripletLoss}
        )
    else:
        print("[FaceNet] Utilisation du backbone InceptionResNetV2-ImageNet")
        model = build_facenet_model()
    return model


def get_embedding(model, face_array):
    """
    Calcule l'embedding d'un visage.

    Args:
        model      : modèle Keras
        face_array : np.ndarray [H, W, 3] ou [B, H, W, 3], float32, [-1, 1]

    Returns:
        embedding np.ndarray [512]
    """
    if face_array.ndim == 3:
        face_array = np.expand_dims(face_array, axis=0)
    emb = model.predict(face_array, verbose=0)
    return emb.squeeze()
