import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

FASHION_LABELS = {
    0: 'T-shirt/top', 1: 'Trouser',  2: 'Pullover',
    3: 'Dress',       4: 'Coat',     5: 'Sandal',
    6: 'Shirt',       7: 'Sneaker',  8: 'Bag',
    9: 'Ankle boot'
}

def load_data():
    df = pd.read_csv('data/raw/dataset.csv')
    if 'label_name' in df.columns:
        df = df.drop(columns=['label_name'])
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y

def generate_shap_summary():
    print('Chargement des donnees...')
    X, y = load_data()
    feature_names = list(X.columns)

    # RF leger entrainé sur 2000 images — TreeExplainer est fait pour RF
    print('Entrainement RF leger pour SHAP (30 secondes)...')
    X_sample  = X.sample(2000, random_state=42)
    y_sample  = y[X_sample.index]
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X_sample)
    rf        = RandomForestClassifier(
        n_estimators=50, max_depth=10,
        n_jobs=-1, random_state=42
    )
    rf.fit(X_scaled, y_sample)

    # Données à expliquer
    explain_idx  = X_sample.sample(30, random_state=42).index
    X_explain    = scaler.transform(X.loc[explain_idx])
    y_explain    = y.loc[explain_idx]

    print('Calcul SHAP (TreeExplainer)...')
    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_explain)
    os.makedirs('reports', exist_ok=True)

    # Graphe 1 : Summary plot
    print('Generation summary plot...')
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_explain,
        feature_names=feature_names,
        class_names=[FASHION_LABELS[i] for i in range(10)],
        max_display=20, show=False
    )
    plt.title('SHAP — Importance globale des pixels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('   Sauvegarde : reports/shap_summary.png')

    # Graphe 2 : Heatmaps par classe
    print('Generation heatmaps par classe...')
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for cls in range(10):
        sv        = shap_values[cls]
        mean_shap = np.abs(sv).mean(axis=0)
        heatmap   = mean_shap.reshape(28, 28)
        im = axes[cls].imshow(heatmap, cmap='hot', interpolation='nearest')
        axes[cls].set_title(FASHION_LABELS[cls], fontsize=11, fontweight='bold')
        axes[cls].axis('off')
        plt.colorbar(im, ax=axes[cls], fraction=0.046)
    plt.suptitle('SHAP — Zones importantes par classe (rouge = tres important)',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/shap_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('   Sauvegarde : reports/shap_heatmaps.png')

    # Graphe 3 : Explication individuelle
    print('Generation explication individuelle...')
    idx        = 0
    img_orig   = X.loc[explain_idx[idx]].values.reshape(28, 28)
    true_label = y_explain.iloc[idx]
    pred       = int(rf.predict(X_explain[[idx]])[0])
    sv_sample  = shap_values[pred][idx]
    fig, axes  = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img_orig, cmap='gray')
    axes[0].set_title(
        f'Vrai: {FASHION_LABELS[true_label]} | Predit: {FASHION_LABELS[pred]}',
        fontsize=12, fontweight='bold'
    )
    axes[0].axis('off')
    shap_img = np.abs(sv_sample).reshape(28, 28)
    axes[1].imshow(img_orig, cmap='gray',  alpha=0.5)
    axes[1].imshow(shap_img, cmap='hot',   alpha=0.7)
    axes[1].set_title('SHAP Heatmap — Zones importantes', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.suptitle('SHAP — Explication individuelle', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/shap_individual.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('   Sauvegarde : reports/shap_individual.png')
    print('\nGraphes SHAP generes avec succes !')
    print('   reports/shap_summary.png')
    print('   reports/shap_heatmaps.png')
    print('   reports/shap_individual.png')

if __name__ == '__main__':
    generate_shap_summary()