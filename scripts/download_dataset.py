import os

# Ruta destino
DATA_PATH = "data"
os.makedirs(DATA_PATH, exist_ok=True)

# Descargar dataset con Kaggle API
# ⚠️ Debes tener instalado kaggle y configurado kaggle.json en ~/.kaggle/
os.system(f"kaggle datasets download -d crowdflower/twitter-airline-sentiment -p {DATA_PATH} --unzip")

print("✅ Dataset descargado en", DATA_PATH)