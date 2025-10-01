import torch
import torch.nn as nn
import joblib
import re
import string

# --- 1. Cargar artefactos guardados ---

# Cargar el vectorizador primero para obtener INPUT_DIM dinámicamente
vectorizer = joblib.load('results/tfidf_vectorizer.pkl')
INPUT_DIM = len(vectorizer.get_feature_names_out())
HIDDEN_DIM = 128
OUTPUT_DIM = 3
DROPOUT_P = 0.5

# Definir la misma arquitectura del modelo MLP
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, p_drop=0.3):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out

# Cargar el estado del modelo entrenado
model = MLPClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, p_drop=DROPOUT_P)
model.load_state_dict(torch.load('results/mlp_model.pth', map_location='cpu'))
model.eval()  # Poner el modelo en modo de evaluación

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
    # 15 textos de ejemplo para probar la predicción
    tweets = [
        "@AmericanAir your customer service is the worst. I've been on hold for 3 hours.",
        "@united Thanks for the great flight and friendly staff! Really enjoyed it.",
        "@SouthwestAir My flight was delayed by 15 minutes but it's okay.",
        "Boarding was smooth and quick today. Appreciate the efficiency.",
        "The seats were so cramped, my back hurts now.",
        "Flight arrived right on time, no issues at all.",
        "Why did you cancel my flight without notice? Totally unacceptable.",
        "Check-in was fine, nothing special to report.",
        "Absolutely loved the in-flight entertainment and the snacks!",
        "Lost my baggage again. I am beyond frustrated.",
        "Gate agents were helpful and polite, thank you.",
        "wifi was spotty but overall the trip was fine",
        "Crew seemed tired and uninterested today.",
        "A little turbulence, but the pilot handled it well.",
        "Seriously impressed with how clean the cabin was this time.",
        # +20 nuevos tweets (incluye neutrales)
        "Flight departed at 9:05 as scheduled.",
        "Boarding group C was called after group B.",
        "I have seat 18A by the window.",
        "Layover in Denver for 45 minutes.",
        "Gate changed from B12 to B18.",
        "Cabin temperature felt comfortable throughout the flight.",
        "Snack was pretzels and water, pretty standard.",
        "Security line took about 20 minutes today.",
        "I'm traveling to Boston tomorrow morning.",
        "Checked in via the app without any problems.",
        "Crew went above and beyond, really appreciated the kindness!",
        "Still no update on my luggage—this is getting ridiculous.",
        "Seats had decent legroom, better than I expected.",
        "Another delay added to an already long day. Not happy.",
        "Boarding started at 10am and finished quickly.",
        "Plane was clean and the restroom was stocked.",
        "The coffee was lukewarm and tasted off.",
        "Smooth landing and no issues at baggage claim.",
        "Website kept erroring out during check-in.",
        "Average experience overall—nothing to complain about, nothing standout."
    ]

    for i, tw in enumerate(tweets, start=1):
        print(f"Tweet {i}: '{tw}'")
        print(f"Predicción: {predict_sentiment(tw)}\n")