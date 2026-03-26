import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/dataset.csv")

print(df.head())
print(df.info())
print(df.describe())

def show_images(df, n_images=10):
    pixel_cols = [f"pixel_{i}" for i in range(784)]

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for i in range(n_images):
        row = df.iloc[i]
        image = row[pixel_cols].values.astype(float).reshape(28, 28)  # ← corrigé
        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(f"{int(row['target'])} - {row['label_name']}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("sample_images.png")
    plt.show()
    print("✅ Image sauvegardée : sample_images.png")

show_images(df)