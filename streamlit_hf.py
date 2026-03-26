import streamlit as st
import numpy as np
import requests
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
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

# --- SESSION STATE ---
if 'result'         not in st.session_state: st.session_state.result = None
if 'image'          not in st.session_state: st.session_state.image = None
if 'pixels'         not in st.session_state: st.session_state.pixels = None
if 'last_run_id'    not in st.session_state: st.session_state.last_run_id = None

# --- FONCTIONS UTILES ---
def preprocess_image(image):
    """Prépare l'image pour le modèle (28x28, niveaux de gris, normalisé)"""
    if image.mode == 'RGBA':
        bg = Image.new('RGB', image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    img = image.convert('L').resize((28, 28), Image.LANCZOS)
    arr = np.array(img).astype(np.float32)
    
    # Normalisation 0-1
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = arr / 255.0
        
    # Inversion si le fond est clair (Fashion MNIST est blanc sur noir)
    if arr.mean() > 0.5:
        arr = 1.0 - arr
        
    return arr.flatten().tolist()

def predict(pixels, model_type):
    """Appel à l'API FastAPI avec choix du modèle"""
    try:
        r = requests.post(
            f"{API_URL}/predict",
            json={
                "features": pixels,
                "model_type": model_type
            },
            timeout=30
        )
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"Erreur API : {e}")
        return None

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def plot_probabilities(probabilities):
    fig, ax = plt.subplots(figsize=(10, 4))
    labels  = [FASHION_LABELS[i] for i in range(10)]
    # Mise en évidence de la classe prédite
    colors  = ['#4CAF50' if i == int(np.argmax(probabilities)) else '#2196F3' for i in range(10)]
    
    bars = ax.barh(labels, probabilities, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probabilité')
    ax.set_title('Confiance du modèle par classe')
    
    for bar, prob in zip(bars, probabilities):
        if prob > 0.01:
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{prob:.1%}', va='center')
    plt.tight_layout()
    return fig

# --- INTERFACE PRINCIPALE ---
def main():
    st.title("👗 Fashion MNIST MLOps Pipeline")
    st.markdown("Interface de classification multi-modèles")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🔌 Statut API")
        health = check_api()
        if health:
            st.success("✅ API en ligne")
        else:
            st.error("❌ API hors ligne")

        st.markdown("---")
        st.header("🧠 Configuration ML")
        
        # SÉLECTEUR DE MODÈLE
        model_display_name = st.selectbox(
            "Choisir le modèle :",
            ["Random Forest", "MLP Neural Network"],
            help="RF est plus stable, MLP est plus complexe."
        )
        model_key = "rf" if model_display_name == "Random Forest" else "mlp"
        
        st.markdown("---")
        st.info(f"**Modèle actif :** {model_display_name}")

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["🎯 Prédiction", "📊 Monitoring", "ℹ️ Architecture"])

    with tab1:
        st.subheader("📁 Uploader une image")
        uploaded = st.file_uploader("Image de vêtement (PNG, JPG)...", type=["jpg","jpeg","png"])

        if uploaded is not None:
            # Création d'un ID unique (Nom de fichier + Modèle choisi)
            current_run_id = f"{uploaded.name}_{uploaded.size}_{model_key}"
            
            if current_run_id != st.session_state.last_run_id:
                st.session_state.last_run_id = current_run_id
                image = Image.open(uploaded)
                pixels = preprocess_image(image)
                
                st.session_state.image = image
                st.session_state.pixels = pixels
                
                with st.spinner(f"Calcul via {model_display_name}..."):
                    st.session_state.result = predict(pixels, model_key)

        # Affichage des résultats
        if st.session_state.result and st.session_state.image:
            res = st.session_state.result
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                st.image(st.session_state.image, caption="Originale", use_container_width=True)
            
            with col2:
                # Simulation visuelle du 28x28 pour l'utilisateur
                img_gray = st.session_state.image.convert('L').resize((28, 28))
                st.image(img_gray, caption="Vue modèle (28x28)", use_container_width=True)

            with col3:
                # Utilisation des clés renvoyées par ton API
                # Note: assure-hui que ton API renvoie bien ces clés (label, confidence, top3)
                st.metric("Prédiction", res.get("label", "Inconnu"))
                st.metric("Confiance", f"{res.get('confidence', 0):.2%}")
                
                if "top3" in res:
                    st.write("**Top 3 :**")
                    for item in res["top3"]:
                        st.write(f"- {item['label']} ({item['probability']:.1%})")

            if "probabilities" in res:
                st.pyplot(plot_probabilities(res["probabilities"]))

    with tab2:
        st.subheader("📈 Monitoring (Simulé via Evidently AI)")
        c1, c2 = st.columns(2)
        c1.metric("Data Drift", "0.12", "-0.02")
        c2.metric("Model Precision", "85.2%", "+1.1%")
        st.write("Le monitoring permet de détecter si les images envoyées par les utilisateurs deviennent trop différentes de celles de l'entraînement.")

    with tab3:
        st.markdown("""
        ### 🏗️ Stack Technique
        - **Backend :** FastAPI hébergé sur Hugging Face
        - **Frontend :** Streamlit
        - **Modèles :** Sklearn (Random Forest & MLP)
        - **CI/CD :** GitHub Actions
        """)

if __name__ == '__main__':
    main()