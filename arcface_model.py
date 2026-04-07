"""
arcface_model.py
----------------
Implémentation ArcFace avec TensorFlow 2.13 / Keras.
Backbone : ResNet-50 pré-entraîné ImageNet.
Loss     : Additive Angular Margin Loss.
"""

import os
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50


# ─── ArcFace Loss Layer ───────────────────────────────────────────────────────────

class ArcFaceLayer(layers.Layer):
    """
    Couche ArcFace : calcule l'Additive Angular Margin Loss.
    Doit être utilisée comme dernière couche du modèle pendant l'entraînement.
    """

    def __init__(self, num_classes, s=64.0, m=0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.s = s      # scale
        self.m = m      # angular margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def build(self, input_shape):
        embedding_size = input_shape[0][-1]
        self.W = self.add_weight(
            name='arcface_weights',
            shape=(embedding_size, self.num_classes),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs, training=None):
        embeddings, labels = inputs

        # Normalisation L2
        emb_norm = tf.nn.l2_normalize(embeddings, axis=1)
        w_norm   = tf.nn.l2_normalize(self.W, axis=0)

        # cos(θ)
        cosine = tf.matmul(emb_norm, w_norm)

        if not training:
            return cosine * self.s

        # sin(θ) et cos(θ + m)
        sine = tf.sqrt(tf.maximum(1.0 - tf.square(cosine), 1e-8))
        phi  = cosine * self.cos_m - sine * self.sin_m

        # Condition : si cos(θ) > cos(π - m), utiliser cos(θ + m), sinon fallback
        phi = tf.where(cosine > self.th, phi, cosine - self.mm)

        # Masque one-hot
        one_hot = tf.one_hot(tf.cast(labels, tf.int32), self.num_classes)

        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes, 's': self.s, 'm': self.m})
        return config


# ─── Construction du modèle ArcFace ──────────────────────────────────────────────

def build_arcface_model(num_classes, embedding_size=512, input_shape=(160, 160, 3)):
    """
    Construit le modèle ArcFace complet (pour entraînement).

    Architecture :
        ResNet50 (ImageNet) → GlobalAvgPool → BN → Dense(512) → BN → ArcFaceLayer

    Returns:
        training_model  : modèle complet avec ArcFaceLayer (pour fine-tuning)
        embedding_model : sous-modèle qui retourne uniquement les embeddings
    """
    # ── Input ──
    img_input   = keras.Input(shape=input_shape, name='image_input')
    label_input = keras.Input(shape=(), name='label_input')

    # ── Backbone ResNet50 (pré-entraîné ImageNet) ──
    backbone = ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=img_input,
        pooling='avg'   # GlobalAveragePooling2D
    )

    # Geler toutes les couches sauf les 20 dernières (fine-tuning partiel)
    for layer in backbone.layers[:-20]:
        layer.trainable = False

    x = backbone.output

    # ── Tête d'embedding ──
    x = layers.BatchNormalization(name='bn_after_pool')(x)
    x = layers.Dense(embedding_size, use_bias=False, name='embedding_dense')(x)
    x = layers.BatchNormalization(name='bn_embedding')(x)

    # Normalisation L2 de l'embedding
    embeddings = layers.Lambda(
        lambda t: tf.nn.l2_normalize(t, axis=1),
        name='l2_norm'
    )(x)

    # ── ArcFace Loss Layer ──
    logits = ArcFaceLayer(num_classes, s=64.0, m=0.5, name='arcface')(
        [embeddings, label_input]
    )

    # ── Modèles ──
    training_model = Model(
        inputs=[img_input, label_input],
        outputs=logits,
        name='ArcFace_Training'
    )

    embedding_model = Model(
        inputs=img_input,
        outputs=embeddings,
        name='ArcFace_Embedding'
    )

    return training_model, embedding_model


# ─── Fine-tuning ─────────────────────────────────────────────────────────────────

def fine_tune_arcface(celeba_dataset, num_classes, epochs=5,
                      save_path='./models/arcface_embedding.h5'):
    """
    Fine-tune ArcFace sur CelebA.

    Args:
        celeba_dataset : tf.data.Dataset retournant (image, label)
        num_classes    : nombre d'identités CelebA
        epochs         : nombre d'epochs
        save_path      : chemin de sauvegarde du modèle d'embedding

    Returns:
        embedding_model : modèle Keras prêt pour l'inférence
    """
    print(f"\n{'='*55}")
    print(f"  FINE-TUNING ARCFACE — {epochs} epochs — {num_classes} classes")
    print(f"{'='*55}")

    training_model, embedding_model = build_arcface_model(num_classes)

    training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    training_model.summary()

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=2, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            save_path.replace('.h5', '_best.h5'),
            monitor='loss', save_best_only=True, verbose=1
        )
    ]

    # Adapter le dataset pour avoir (image, label) → ((image, label), label)
    def reformat(image, label):
        return (image, label), label

    training_dataset = celeba_dataset.map(reformat)

    history = training_model.fit(
        training_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Sauvegarder uniquement le modèle d'embedding (inférence)
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    embedding_model.save(save_path)
    print(f"\n[ArcFace] Modèle d'embedding sauvegardé → {save_path}")

    return embedding_model


# ─── Chargement du modèle d'embedding ────────────────────────────────────────────

def load_arcface_embedding(weights_path=None):
    """
    Charge le modèle ArcFace en mode inférence.

    Args:
        weights_path : chemin vers le .h5 sauvegardé, ou None (poids ImageNet seuls)

    Returns:
        embedding_model Keras
    """
    if weights_path and os.path.exists(weights_path):
        print(f"[ArcFace] Chargement depuis {weights_path}")
        model = keras.models.load_model(
            weights_path,
            custom_objects={'ArcFaceLayer': ArcFaceLayer}
        )
    else:
        print("[ArcFace] Utilisation du backbone ResNet50-ImageNet (sans fine-tuning)")
        _, model = build_arcface_model(num_classes=100)  # num_classes factice
    return model


def get_embedding(model, face_array):
    """
    Calcule l'embedding d'un visage.

    Args:
        model      : modèle Keras d'embedding
        face_array : np.ndarray [H, W, 3] ou [B, H, W, 3], float32, [-1, 1]

    Returns:
        embedding np.ndarray [512]
    """
    if face_array.ndim == 3:
        face_array = np.expand_dims(face_array, axis=0)
    emb = model.predict(face_array, verbose=0)
    return emb.squeeze()
