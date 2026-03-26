import gradio as gr
import numpy as np
import requests
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

API_URL = "https://lepiso-fashion-mnist-api.hf.space"

FASHION_LABELS = {
    0: 'T-shirt/top', 1: 'Trouser',  2: 'Pullover',
    3: 'Dress',       4: 'Coat',     5: 'Sandal',
    6: 'Shirt',       7: 'Sneaker',  8: 'Bag',
    9: 'Ankle boot'
}

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

def plot_probabilities(probabilities):
    fig, ax = plt.subplots(figsize=(10, 5))
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

def predict_image(image):
    if image is None:
        return "Aucune image", 0.0, "", None

    try:
        # Prétraitement
        pil_image = Image.fromarray(image)
        pixels    = preprocess_image(pil_image)

        # Appel API
        r = requests.post(
            f"{API_URL}/predict",
            json={"features": pixels},
            timeout=30
        )

        if r.status_code != 200:
            return "Erreur API", 0.0, "", None

        result     = r.json()
        label      = result["label"]
        confidence = result["confidence"]
        top3       = result["top3"]
        proba      = result["probabilities"]

        # Top 3 texte
        top3_text = "\n".join([
            f"{'🥇🥈🥉'[i]}  {item['label']} — {item['probability']:.1%}"
            for i, item in enumerate(top3)
        ])

        # Graphe probabilités
        fig = plot_probabilities(proba)

        return label, confidence, top3_text, fig

    except Exception as e:
        return f"Erreur : {e}", 0.0, "", None


# ── Interface Gradio ───────────────────────────────────────────
with gr.Blocks(
    title="Fashion MNIST MLOps",
    theme=gr.themes.Soft(),
    css=".gradio-container {max-width: 1200px; margin: auto;}"
) as demo:

    gr.Markdown("""
    # 👗 Fashion MNIST MLOps Pipeline
    **Classification de vêtements — 10 catégories**
    Uploadez une image de vêtement et le modèle prédit sa catégorie.
    """)

    with gr.Row():
        # Colonne gauche — Input
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📁 Uploader une image",
                type="numpy",
                height=300
            )
            predict_btn = gr.Button(
                "🎯 Prédire",
                variant="primary",
                size="lg"
            )
            gr.Markdown("""
            **Classes disponibles :**
            `T-shirt` `Trouser` `Pullover` `Dress` `Coat`
            `Sandal` `Shirt` `Sneaker` `Bag` `Ankle boot`
            """)

        # Colonne droite — Output
        with gr.Column(scale=1):
            label_output = gr.Textbox(
                label="🎯 Classe prédite",
                interactive=False,
                text_align="center"
            )
            confidence_output = gr.Number(
                label="📊 Confiance",
                interactive=False
            )
            top3_output = gr.Textbox(
                label="🏆 Top 3 prédictions",
                interactive=False,
                lines=3
            )

    # Graphe probabilités
    with gr.Row():
        proba_plot = gr.Plot(label="📊 Probabilités par classe")

    # Exemples
    gr.Markdown("---")
    gr.Markdown("### ℹ️ Architecture du Pipeline MLOps")
    gr.Markdown("""
```
    Fashion MNIST → generate_data.py → train.py (RF+MLP+CNN+MLflow)
        → FastAPI (/predict /health) → Docker → HuggingFace
        → Evidently AI (monitoring) → SHAP (XAI)
```

    | Modèle | Accuracy | F1 |
    |---|---|---|
    | Random Forest | 85.45% | 0.851 |
    | **MLP NeuralNet** | **85.75%** | **0.857** |
    | CNN Keras | 85.55% | 0.854 |
    """)

    # Action
    predict_btn.click(
        fn=predict_image,
        inputs=[image_input],
        outputs=[label_output, confidence_output, top3_output, proba_plot]
    )

    # Aussi au changement d'image
    image_input.change(
        fn=predict_image,
        inputs=[image_input],
        outputs=[label_output, confidence_output, top3_output, proba_plot]
    )

if __name__ == "__main__":
    demo.launch()