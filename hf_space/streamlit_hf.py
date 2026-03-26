import streamlit as st
import numpy as np
import requests
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

API_URL = "https://lepiso-fashion-mnist-api.hf.space"

st.set_page_config(
    page_title="Fashion MNIST MLOps",
    page_icon="👗",
    layout="wide"
)

FASHION_LABELS = {
    0: 'T-shirt/top', 1: 'Trouser',  2: 'Pullover',
    3: 'Dress',       4: 'Coat',     5: 'Sandal',
    6: 'Shirt',       7: 'Sneaker',  8: 'Bag',
    9: 'Ankle boot'
}

# ── Session state — évite le re-render ────────────────────────
if 'result'     not in st.session_state: st.session_state.result     = None
if 'image'      not in st.session_state: st.session_state.image      = None
if 'pixels'     not in st.session_state: st.session_state.pixels     = None
if 'last_file'  not in st.session_state: st.session_state.last_file  = None

def preprocess_image(image):
    if image.mode == 'RGBA':
        bg = Image.new('RGB', image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    img = image.convert('L').resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype(np.float32)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = arr / 255.0
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    return arr.flatten().tolist()

def predict(pixels):
    try:
        requests.get(f"{API_URL}/health", timeout=15)
        r = requests.post(
            f"{API_URL}/predict",
            json={"features": pixels},
            timeout=60
        )
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        return None

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def plot_probabilities(probabilities):
    fig, ax = plt.subplots(figsize=(10, 4))
    labels  = [FASHION_LABELS[i] for i in range(10)]
    colors  = ['#4CAF50' if i == int(np.argmax(probabilities))
        else '#2196F3' for i in range(10)]
    bars = ax.barh(labels, probabilities, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probabilité', fontsize=12)
    ax.set_title('Probabilités par classe', fontsize=13, fontweight='bold')
    for bar, prob in zip(bars, probabilities):
        if prob > 0.01:
            ax.text(bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'{prob:.1%}', va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig

def main():
    st.title("👗  Fashion MNIST MLOps Pipeline")
    st.markdown("**Classification de vêtements — Interface complète**")
    st.markdown("---")

    with st.sidebar:
        st.header("🔌  Statut API")
        health = check_api()
        if health:
            st.success("✅ API en ligne")
            st.markdown(f"**Statut :** {health.get('status','')}")
        else:
            st.error("❌ API hors ligne")
        st.markdown("---")
        st.header("ℹ️  Informations")
        st.markdown("**Dataset :** Fashion MNIST")
        st.markdown("**Modèle :** MLP Neural Network")
        st.markdown("**Accuracy :** ~85.75%")
        st.markdown("**Classes :** 10 catégories")
        st.markdown("---")
        st.header("🔗  Liens")
        st.markdown(f"[📖 API Docs]({API_URL}/docs)")
        st.markdown(f"[🏥 Health]({API_URL}/health)")

    tab1, tab2, tab3 = st.tabs([
        "🎯 Prédiction",
        "📊 Monitoring",
        "ℹ️ À propos"
    ])

    with tab1:
        st.subheader("📁  Uploader une image de vêtement")
        st.info("JPG/PNG — redimensionnée en 28×28 pixels automatiquement.")

        uploaded = st.file_uploader(
            "Choisir une image...",
            type=["jpg","jpeg","png","bmp"],
            key="uploader"
        )

        # Détecter nouveau fichier — appeler l'API UNE SEULE FOIS
        if uploaded is not None:
            file_id = uploaded.name + str(uploaded.size)
            if file_id != st.session_state.last_file:
                st.session_state.last_file = file_id
                image  = Image.open(uploaded)
                pixels = preprocess_image(image)
                st.session_state.image  = image
                st.session_state.pixels = pixels
                with st.spinner("Analyse via API..."):
                    st.session_state.result = predict(pixels)

        # Afficher résultats depuis session_state
        if st.session_state.result and st.session_state.image:
            image  = st.session_state.image
            result = st.session_state.result

            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                st.subheader("🖼️  Originale")
                st.image(image, width=200)
                st.caption(f"{image.size[0]}×{image.size[1]} px")

            with col2:
                st.subheader("🔄  Traitée")
                img_gray = image.convert('L').resize((28, 28))
                st.image(img_gray, width=200, clamp=True)
                st.caption("28×28 niveaux de gris")

            with col3:
                st.subheader("🎯  Prédiction")
                label      = result["label"]
                confidence = result["confidence"]
                top3       = result["top3"]

                st.markdown(f"""
                <div style='background:#DCFFE4;padding:20px;border-radius:12px;
                            border:2px solid #4AC26B;text-align:center;'>
                    <h2 style='color:#116329;margin:0;'>{label}</h2>
                    <p style='color:#2DA44E;font-size:1.2rem;margin:8px 0 0 0;'>
                        Confiance : {confidence:.1%}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**🏆  Top 3 :**")
                for i, item in enumerate(top3):
                    emoji = ["🥇","🥈","🥉"][i]
                    st.markdown(
                        f"{emoji} **{item['label']}** — {item['probability']:.1%}"
                    )

            st.markdown("---")
            st.subheader("📊  Probabilités par classe")
            fig = plot_probabilities(result["probabilities"])
            st.pyplot(fig)
            plt.close()

            with st.expander("🔢  Pixels normalisés"):
                st.write(f"784 valeurs — aperçu des 20 premières :")
                st.write(st.session_state.pixels[:20])

        elif uploaded is None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                ### 🚀  Comment utiliser ?
                1. **Uploadez** une image de vêtement
                2. L'API **prédit** automatiquement
                3. Consultez les **probabilités**
                """)
            with col2:
                st.markdown("""
                ### 🎯  Classes détectées
                | # | Vêtement |
                |---|----------|
                | 0 | T-shirt/top |
                | 1 | Trouser |
                | 2 | Pullover |
                | 3 | Dress |
                | 4 | Coat |
                | 5 | Sandal |
                | 6 | Shirt |
                | 7 | Sneaker |
                | 8 | Bag |
                | 9 | Ankle boot |
                """)

    with tab2:
        st.subheader("📈  Monitoring Evidently AI")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Drift", "NON détecté", "✅ Stable")
        with col2:
            st.metric("Colonnes driftées", "20%", "4/20 colonnes")
        with col3:
            st.metric("Accuracy", "85.75%", "MLP NeuralNet")
        st.markdown("---")
        st.markdown("""
        | Métrique | Description | Seuil d'alerte |
        |---|---|---|
        | Dataset Drift | % colonnes différentes | > 30% |
        | Missing Values | Valeurs manquantes | > 0% |
        | Data Quality | Types et plages | Tout changement |
        """)

    with tab3:
        st.subheader("🏗️  Architecture MLOps")
        st.markdown("""
```
        Fashion MNIST (70 000 images, 10 classes)
                ↓
        generate_data.py → Normalisation CSV
                ↓
        train.py → RandomForest + MLP + CNN + MLflow
                ↓
        models/model.pkl → MLP 85.75%
                ↓
        FastAPI → /predict /health /explain
                ↓
        Docker → Container autonome
                ↓
        HuggingFace → Déploiement public
                ↓
        Evidently AI → Monitoring dérive
```
        """)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🛠️  Stack technique
            - Python 3.12 + scikit-learn
            - TensorFlow/Keras (CNN)
            - MLflow (tracking)
            - FastAPI + Docker
            - Evidently AI + SHAP
            - Streamlit + HuggingFace
            """)
        with col2:
            st.markdown("""
            ### 📊  Résultats
            | Modèle | Accuracy |
            |---|---|
            | Random Forest | 85.45% |
            | **MLP** | **85.75%** |
            | CNN | 85.55% |
            """)
            st.markdown(f"[🌐 API FastAPI]({API_URL}/docs)")

result_container = st.container()

if __name__ == '__main__':
    main()