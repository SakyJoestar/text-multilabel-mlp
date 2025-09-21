import torch
import torch.nn as nn
import joblib
import re
import string

# --- 1. Cargar artefactos guardados ---

# Definir la misma arquitectura del modelo MLP
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Parámetros del modelo (deben ser los mismos que en el entrenamiento)
INPUT_DIM = 10770  # max_features del TfidfVectorizer
HIDDEN_DIM = 128
OUTPUT_DIM = 3

# Cargar el estado del modelo entrenado
model = MLPClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
model.load_state_dict(torch.load('results/mlp_model.pth'))
model.eval()  # Poner el modelo en modo de evaluación

# Cargar el vectorizador
vectorizer = joblib.load('results/tfidf_vectorizer.pkl')

# Mapeo inverso de etiquetas
label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

# --- 2. Función para predecir un solo texto ---

# Reutilizar la misma función de limpieza del script de entrenamiento
def clean_text(text):
    text = text.lower()
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join(text.split())
    return text

def predict_sentiment(text):
    """
    Realiza una predicción de sentimiento para un texto dado.
    """
    # 1. Limpiar el texto
    cleaned_text = clean_text(text)
    
    # 2. Vectorizar usando el vectorizador cargado
    # Se pasa como una lista para que el vectorizador lo trate como un documento
    vectorized_text = vectorizer.transform([cleaned_text]).toarray()
    
    # 3. Convertir a tensor de PyTorch
    text_tensor = torch.tensor(vectorized_text, dtype=torch.float32)
    
    # 4. Realizar la predicción
    with torch.no_grad():
        output = model(text_tensor)
        _, predicted_idx = torch.max(output, 1)
        
    # 5. Mapear el índice a la etiqueta
    predicted_label = label_map[predicted_idx.item()]
    
    return predicted_label

# --- 3. Ejemplo de uso ---
if __name__ == '__main__':
    # Textos de ejemplo para probar la predicción
    tweet1 = "@AmericanAir your customer service is the worst. I've been on hold for 3 hours."
    tweet2 = "@united Thanks for the great flight and friendly staff! Really enjoyed it."
    tweet3 = "@SouthwestAir My flight was delayed by 15 minutes but it's okay."

    print(f"Tweet: '{tweet1}'")
    print(f"Predicción: {predict_sentiment(tweet1)}\n")

    print(f"Tweet: '{tweet2}'")
    print(f"Predicción: {predict_sentiment(tweet2)}\n")

    print(f"Tweet: '{tweet3}'")
    print(f"Predicción: {predict_sentiment(tweet3)}\n")