"""
app.py
------
Interface Streamlit pour la vérification d'identité faciale.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import numpy as np
import cv2
import time
from PIL import Image

from arcface_model import load_arcface_embedding, get_embedding as arcface_emb
from facenet_model import load_facenet_embedding, get_embedding as facenet_emb
from face_utils import (
    detect_and_align, preprocess_from_bytes,
    draw_face_boxes, compute_cosine_similarity
)


# ─── Config page ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FaceVerify TF",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080D1A;
    color: #CBD5E1;
}
.main { background-color: #080D1A; }

.hero {
    background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 100%);
    border: 1px solid #312E81;
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
}
.hero h1 {
    font-size: 4rem;
    font-weight: 700;
    line-height: 1.05;
    color: #FFFFFF;
    background: none;
    -webkit-background-clip: unset;
    -webkit-text-fill-color: #FFFFFF;
    margin: 0;
}
.hero p { color: #64748B; margin: 0.5rem 0 0; font-size: 1.1rem; }

.card {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 14px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    text-align: center;
}
.score-num {
    font-size: 2.8rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
}
.verdict { font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem; }
.model-tag {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.stButton > button {
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85; }
hr { border-color: #1E293B; }
</style>
""", unsafe_allow_html=True)


# ─── Chargement modèles (cache) ───────────────────────────────────────────────────

@st.cache_resource
def load_models():
    arcface = load_arcface_embedding('./models/arcface_embedding.h5')
    facenet = load_facenet_embedding('./models/facenet_embedding.h5')
    return arcface, facenet


# ─── Helpers ─────────────────────────────────────────────────────────────────────

def sim_to_pct(sim):
    return round((sim + 1) / 2 * 100, 1)


def verdict_html(sim, threshold, model_name, color):
    pct = sim_to_pct(sim)
    accept = sim >= threshold
    v_color = '#10B981' if accept else '#F43F5E'
    v_label = '✅ MÊME PERSONNE' if accept else '❌ DIFFÉRENTES'
    return f"""
    <div class="card">
        <div class="model-tag" style="color:{color};">{model_name}</div>
        <div class="score-num" style="color:{v_color};">{pct}%</div>
        <div style="color:#64748B;font-size:0.8rem;">Score de similarité</div>
        <div class="verdict" style="color:{v_color};">{v_label}</div>
    </div>
    """


# ─── Interface principale ─────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="hero">
        <h1>🕵🏼 Reconnaissance Faciale</h1>
        <p>Vérification d'Identité par Reconnaissance Faciale · ArcFace vs FaceNet · </p>
    </div>
    """, unsafe_allow_html=True)

    # Chargement modèles
    with st.spinner("⏳ Chargement des modèles TensorFlow..."):
        try:
            arcface_model, facenet_model = load_models()
            st.success("✅ Modèles chargés")
        except Exception as e:
            st.warning(f"⚠️ Modèles chargés avec poids ImageNet (pas de fine-tuning trouvé) : {e}")
            arcface_model, facenet_model = load_models()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### ⚙️ Paramètres")
        st.markdown("---")
        mode = st.radio("Mode", ["📁 Upload d'images", "📷 Webcam temps réel"])
        st.markdown("---")
        threshold = st.slider("Seuil de décision", 0.0, 1.0, 0.5, 0.05,
                              help="Au-dessus = même personne")
        st.markdown("---")
        st.markdown("### 📦 Environnement")
        import tensorflow as tf
        st.code(f"TF  : {tf.__version__}\nPy  : {sys.version[:6]}", language='text')
        st.markdown("---")
        st.markdown("### 📊 Modèles")
        st.markdown("🔵 **ArcFace** — ResNet50 + Angular Margin Loss")
        st.markdown("🔴 **FaceNet** — InceptionResNetV2 + Triplet Loss")

    # ════════════════════════════════════════════════
    # MODE 1 — Upload
    # ════════════════════════════════════════════════
    if "Upload" in mode:
        st.markdown("## 📁 Vérification par images")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🖼️ Image de référence")
            file1 = st.file_uploader("", type=['jpg','jpeg','png'], key='img1')
            if file1:
                st.image(Image.open(file1), use_column_width=True)

        with col2:
            st.markdown("#### 🖼️ Image à vérifier")
            file2 = st.file_uploader("", type=['jpg','jpeg','png'], key='img2')
            if file2:
                st.image(Image.open(file2), use_column_width=True)

        st.markdown("---")

        if file1 and file2:
            if st.button("🔍 Lancer la vérification"):
                with st.spinner("Détection des visages et calcul des embeddings..."):
                    file1.seek(0); file2.seek(0)
                    face1, _ = preprocess_from_bytes(file1.read())
                    file2.seek(0)
                    face2, _ = preprocess_from_bytes(file2.read())

                if face1 is None:
                    st.error("❌ Aucun visage détecté dans l'image 1")
                elif face2 is None:
                    st.error("❌ Aucun visage détecté dans l'image 2")
                else:
                    # Embeddings ArcFace
                    emb1_arc = arcface_emb(arcface_model, face1)
                    emb2_arc = arcface_emb(arcface_model, face2)
                    sim_arc  = compute_cosine_similarity(emb1_arc, emb2_arc)

                    # Embeddings FaceNet
                    emb1_fn = facenet_emb(facenet_model, face1)
                    emb2_fn = facenet_emb(facenet_model, face2)
                    sim_fn  = compute_cosine_similarity(emb1_fn, emb2_fn)

                    # Résultats
                    st.markdown("## 📊 Résultats")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(
                            verdict_html(sim_arc, threshold, "🔵 ArcFace", "#818CF8"),
                            unsafe_allow_html=True
                        )
                    with c2:
                        st.markdown(
                            verdict_html(sim_fn, threshold, "🔴 FaceNet", "#F472B6"),
                            unsafe_allow_html=True
                        )

                    # Consensus
                    agree = (sim_arc >= threshold) == (sim_fn >= threshold)
                    if agree:
                        label = "Accepté ✅" if sim_arc >= threshold else "Rejeté ❌"
                        st.success(f"Les deux modèles sont d'accord : **{label}**")
                    else:
                        st.warning("⚠️ Les modèles divergent — essayez d'ajuster le seuil.")

    # ════════════════════════════════════════════════
    # MODE 2 — Webcam
    # ════════════════════════════════════════════════
    else:
        st.markdown("## 📷 Vérification en temps réel")
        col_ref, col_cam = st.columns([1, 2])

        with col_ref:
            st.markdown("#### 📸 Image de référence")
            ref_file = st.file_uploader("", type=['jpg','jpeg','png'], key='ref')

            if ref_file:
                st.image(Image.open(ref_file), use_column_width=True)
                ref_file.seek(0)
                ref_face, _ = preprocess_from_bytes(ref_file.read())

                if ref_face is not None:
                    st.success("✅ Visage de référence détecté")
                    ref_emb_arc = arcface_emb(arcface_model, ref_face)
                    ref_emb_fn  = facenet_emb(facenet_model, ref_face)
                    st.session_state['ref_arc'] = ref_emb_arc
                    st.session_state['ref_fn']  = ref_emb_fn
                else:
                    st.error("❌ Aucun visage détecté")

        with col_cam:
            st.markdown("#### 📡 Flux webcam")
            run = st.checkbox("🎥 Activer la webcam")

            if run and 'ref_arc' in st.session_state:
                stframe = st.empty()
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                stop = st.button("⏹ Arrêter")

                while cap.isOpened() and not stop:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    annotated, results = draw_face_boxes(frame.copy())

                    if results:
                        face = detect_and_align(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        )
                        if face is not None:
                            emb = arcface_emb(arcface_model, face)
                            sim = compute_cosine_similarity(
                                emb, st.session_state['ref_arc']
                            )
                            pct     = sim_to_pct(sim)
                            accept  = sim >= threshold
                            color   = (0, 220, 120) if accept else (0, 80, 255)
                            verdict = "ACCEPTE" if accept else "REJETE"
                            cv2.putText(annotated,
                                        f"ArcFace: {pct}% — {verdict}",
                                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.85, color, 2)

                    stframe.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        channels="RGB", use_column_width=True
                    )
                    time.sleep(0.03)

                cap.release()

            elif run and 'ref_arc' not in st.session_state:
                st.warning("⚠️ Uploadez d'abord une image de référence.")


if __name__ == '__main__':
    main()
