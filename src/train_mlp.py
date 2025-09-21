import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- 1. Carga y Preprocesamiento de Datos ---
print("Cargando datos...")
# El dataset corresponde al "Twitter US Airline Sentiment Dataset" 
df = pd.read_csv('data/Tweets.csv')

# Selección de columnas relevantes y limpieza inicial
df = df[['airline_sentiment', 'text']]
df = df.rename(columns={'airline_sentiment': 'sentiment', 'text': 'text'})
df.dropna(inplace=True)

# Mapeo de etiquetas a valores numéricos
sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['sentiment_label'] = df['sentiment'].map(sentiment_map)

# Función de limpieza de texto
def clean_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)  # Eliminar menciones
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)  # Eliminar URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Eliminar caracteres no alfabéticos
    text = text.translate(str.maketrans('', '', string.punctuation)) # Eliminar puntuación
    text = " ".join(text.split()) # Eliminar espacios extra
    return text

print("Limpiando y preprocesando texto...")
df['cleaned_text'] = df['text'].apply(clean_text)

# --- 2. Preparación para el Modelo ---
X = df['cleaned_text']
y = df['sentiment_label'].values

# División en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorización TF-IDF
print("Vectorizando texto con TF-IDF...")
vectorizer = TfidfVectorizer(max_features=100000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Conversión a tensores de PyTorch
X_train_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_tfidf, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# --- 3. Definición del Modelo MLP en PyTorch ---
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

# --- 4. Entrenamiento del Modelo ---
INPUT_DIM = X_train_tfidf.shape[1]
HIDDEN_DIM = 128
OUTPUT_DIM = len(sentiment_map)
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 64

model = MLPClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Iniciando entrenamiento del Perceptrón Multicapa...")
for epoch in range(EPOCHS):
    for i in range(0, len(X_train_tensor), BATCH_SIZE):
        inputs = X_train_tensor[i:i+BATCH_SIZE]
        labels = y_train_tensor[i:i+BATCH_SIZE]
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

# --- 5. Evaluación del Modelo ---
print("\nEvaluando modelo...")
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    y_pred = predicted.numpy()

# Métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=sentiment_map.keys()))

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sentiment_map.keys(), yticklabels=sentiment_map.keys())
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.savefig('results/confusion_matrix_mlp.png')
print("\nMatriz de confusión guardada en 'results/confusion_matrix_mlp.png'")

# Guardar el modelo entrenado
torch.save(model.state_dict(), 'results/mlp_model.pth')
print("Modelo guardado en 'results/mlp_model.pth'")

# Guardar el vectorizador TF-IDF
joblib.dump(vectorizer, 'results/tfidf_vectorizer.pkl')
print("Vectorizador guardado en 'results/tfidf_vectorizer.pkl'")