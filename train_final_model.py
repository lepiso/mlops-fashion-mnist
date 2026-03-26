import joblib
import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Création du dossier models s'il n'existe pas
if not os.path.exists('models'):
    os.makedirs('models')

print("📥 Chargement des données Fashion-MNIST...")
mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, train_size=20000, random_state=42)

# --- GÉNÉRATION DES FEATURE NAMES ---
# Important pour que FastAPI puisse reconstruire le DataFrame avec les bons noms de colonnes
print("📝 Génération des noms de colonnes...")
feature_names = [f"pixel{i}" for i in range(784)]
joblib.dump(feature_names, 'models/feature_names.pkl')

# --- 1. RANDOM FOREST ---
print("🌲 Entraînement Random Forest...")
rf = Pipeline([
    ('scaler', StandardScaler()), 
    ('model', RandomForestClassifier(n_estimators=100, n_jobs=-1))
])
rf.fit(X_train, y_train)
joblib.dump(rf, 'models/rf_model.pkl')

# --- 2. MLP (Neural Network) ---
print("🧠 Entraînement MLP...")
mlp = Pipeline([
    ('scaler', StandardScaler()), 
    ('model', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, random_state=42))
])
mlp.fit(X_train, y_train)
joblib.dump(mlp, 'models/mlp_model.pkl')

print(f"✅ Terminé ! Fichiers générés dans /models :")
print(f"   - rf_model.pkl")
print(f"   - mlp_model.pkl")
print(f"   - feature_names.pkl")