import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration page ─────────────────────────────────────────
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👗",
    layout="wide"
)

FASHION_LABELS = {
    0: 'T-shirt/top', 1: 'Trouser',  2: 'Pullover',
    3: 'Dress',       4: 'Coat',     5: 'Sandal',
    6: 'Shirt',       7: 'Sneaker',  8: 'Bag',
    9: 'Ankle boot'
}

# ── Chargement du modèle ───────────────────────────────────────
@st.cache_resource
def load_model():
    model         = joblib.load('models/model.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, feature_names

# ── Prétraitement image ────────────────────────────────────────
def preprocess_image(image):
    import numpy as np
    from PIL import Image

    
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    img = image.convert('L')
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype(np.float32)

    arr_min = arr.min()
    arr_max = arr.max()
    
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = arr / 255.0
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    arr = np.clip(arr, 0.0, 1.0)

    return arr.flatten().tolist()

# ── Graphe probabilités ────────────────────────────────────────
def plot_probabilities(probabilities):
    fig, ax = plt.subplots(figsize=(10, 4))
    labels  = [FASHION_LABELS[i] for i in range(10)]
    colors  = ['#2196F3' if i != np.argmax(probabilities)
    else '#4CAF50' for i in range(10)]
    bars = ax.barh(labels, probabilities, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probabilité', fontsize=12)
    ax.set_title('Probabilités par classe', fontsize=13, fontweight='bold')
    for bar, prob in zip(bars, probabilities):
        if prob > 0.01:
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{prob:.1%}', va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig

# ── SHAP heatmap ───────────────────────────────────────────────
def plot_shap_heatmap(pixels, model, feature_names, pred):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    try:
        df    = pd.read_csv('data/raw/dataset.csv')
        if 'label_name' in df.columns:
            df = df.drop(columns=['label_name'])
        X_sample  = df.drop(columns=['target']).sample(1000, random_state=42)
        y_sample  = df['target'][X_sample.index]
        scaler    = StandardScaler()
        X_scaled  = scaler.fit_transform(X_sample)
        rf        = RandomForestClassifier(n_estimators=30, max_depth=8,
        n_jobs=-1, random_state=42)
        rf.fit(X_scaled, y_sample)
        pixels_scaled = scaler.transform([pixels])
        explainer     = shap.TreeExplainer(rf)
        shap_vals     = explainer.shap_values(pixels_scaled)
        sv            = shap_vals[pred][0]
        img_orig      = np.array(pixels).reshape(28, 28)
        shap_img      = np.abs(sv).reshape(28, 28)
        fig, axes     = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(img_orig, cmap='gray')
        axes[0].set_title('Image originale', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        axes[1].imshow(img_orig, cmap='gray',  alpha=0.4)
        axes[1].imshow(shap_img, cmap='hot',   alpha=0.8)
        axes[1].set_title('SHAP — Zones importantes', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.suptitle(f'Explication SHAP — {FASHION_LABELS[pred]}',
        fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

# ── Interface principale ───────────────────────────────────────
def main():
    # Header
    st.title("👗  Fashion MNIST Classifier")
    st.markdown("**Classifiez des vêtements avec XAI (SHAP) — MLOps Pipeline**")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️  Informations")
        st.markdown("**Modèle :** MLP Neural Network")
        st.markdown("**Dataset :** Fashion MNIST")
        st.markdown("**Classes :** 10 catégories")
        st.markdown("**Accuracy :** ~85.75%")
        st.markdown("---")
        st.markdown("**Classes disponibles :**")
        for i, label in FASHION_LABELS.items():
            st.markdown(f"  {i} — {label}")
        st.markdown("---")
        show_shap = st.checkbox("🔍 Afficher explication SHAP", value=True)

    # Chargement modèle
    try:
        model, feature_names = load_model()
        st.success("✅ Modèle chargé avec succès")
    except Exception as e:
        st.error(f"❌ Erreur chargement modèle : {e}")
        return

    # Upload image
    st.subheader("📁  Uploader une image")
    st.info("Uploadez une image de vêtement (JPG, PNG, BMP). "
            "L'image sera automatiquement redimensionnée en 28×28 pixels et convertie en niveaux de gris.")

    uploaded = st.file_uploader(
        "Choisir une image...",
        type=["jpg", "jpeg", "png", "bmp", "gif"],
        help="Formats acceptés : JPG, PNG, BMP, GIF"
    )

    if uploaded:
        # Affichage en 3 colonnes
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            st.subheader("🖼️  Image uploadée")
            image = Image.open(uploaded)
            st.image(image, caption="Image originale", use_column_width=True)
            st.caption(f"Taille originale : {image.size[0]}×{image.size[1]} px")

        with col2:
            st.subheader("🔄  Image traitée")
            img_gray = image.convert('L').resize((28, 28))
            st.image(img_gray, caption="28×28 niveaux de gris",
                use_column_width=True, clamp=True)
            st.caption("Redimensionnée et convertie pour le modèle")

        with col3:
            st.subheader("🎯  Prédiction")
            with st.spinner("Analyse en cours..."):
                pixels = preprocess_image(image)
                X      = pd.DataFrame([pixels], columns=feature_names)
                pred   = int(model.predict(X)[0])
                proba  = model.predict_proba(X)[0]

            # Résultat principal
            confidence = float(np.max(proba))
            st.markdown(f"""
            <div style='background:#DCFFE4;padding:20px;border-radius:12px;
                        border:2px solid #4AC26B;text-align:center;'>
                <h2 style='color:#116329;margin:0;'>
                    {FASHION_LABELS[pred]}
                </h2>
                <p style='color:#2DA44E;font-size:1.2rem;margin:8px 0 0 0;'>
                    Confiance : {confidence:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**🏆  Top 3 prédictions :**")
            top3 = sorted(range(10), key=lambda i: proba[i], reverse=True)[:3]
            for rank, idx in enumerate(top3):
                emoji  = ["🥇", "🥈", "🥉"][rank]
                bar_w  = int(proba[idx] * 100)
                st.markdown(
                    f"{emoji} **{FASHION_LABELS[idx]}** — {proba[idx]:.1%}  "
                    f"`{'█' * (bar_w // 5)}{'░' * (20 - bar_w // 5)}`"
                )

        # Graphe probabilités
        st.markdown("---")
        st.subheader("📊  Probabilités pour toutes les classes")
        fig_proba = plot_probabilities(proba)
        st.pyplot(fig_proba)
        plt.close()

        # SHAP
        if show_shap:
            st.markdown("---")
            st.subheader("🔍  Explication SHAP — Pourquoi cette prédiction ?")
            st.info("SHAP montre quels pixels ont le plus influencé la décision. "
                    "Les zones rouges/jaunes = pixels importants.")
            with st.spinner("Calcul SHAP en cours (~30 secondes)..."):
                fig_shap = plot_shap_heatmap(pixels, model, feature_names, pred)
            if fig_shap:
                st.pyplot(fig_shap)
                plt.close()
            else:
                st.warning("SHAP non disponible pour cette prédiction.")

        # Données brutes
        with st.expander("🔢  Voir les données brutes (pixels normalisés)"):
            st.write(f"**784 pixels normalisés [0.0 - 1.0] :**")
            pixel_df = pd.DataFrame(
                [pixels],
                columns=[f"pixel_{i}" for i in range(784)]
            )
            st.dataframe(pixel_df.T.rename(columns={0: "valeur"}))

    else:
        # Message d'accueil
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🚀  Comment utiliser cette application ?
            1. **Uploadez** une image de vêtement (JPG/PNG)
            2. Le modèle **prédit** automatiquement la catégorie
            3. Consultez les **probabilités** par classe
            4. Activez **SHAP** pour comprendre pourquoi
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

if __name__ == '__main__':
    main()