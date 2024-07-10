from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as rt
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = FastAPI()

# Cargar el modelo ONNX
model_path = 'modelPML.onnx'
session = rt.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Cargar el vectorizador y escalador
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
scaler = StandardScaler(with_mean=False)

# Simular el ajuste del vectorizador y escalador
# Esto debe coincidir con el preprocesamiento original
data = pd.read_csv('bbc_data.csv')
data['data'] = data['data'].apply(lambda x: re.sub(r'[^\w\s]', '', x).lower())

# Dividir el conjunto de datos en entrenamiento y prueba
X = vectorizer.fit_transform(data['data'])
X_train, X_test, y_train, y_test = train_test_split(X, data['labels'], test_size=0.2, random_state=42)

# Ajustar el escalador solo con los datos de entrenamiento
scaler.fit(X_train)
X_train_vect_dense = scaler.transform(X_train).toarray()
X_test_vect_dense = scaler.transform(X_test).toarray()

# Convertir las etiquetas a números
y_train, category_labels = pd.factorize(y_train)
y_test = pd.Categorical(y_test, categories=category_labels).codes

# Definir las etiquetas de las categorías
category_labels = ['sport', 'business', 'politics', 'tech', 'entertainment']

class TextData(BaseModel):
    text: str

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    vectorized_text = vectorizer.transform([text])
    scaled_text = scaler.transform(vectorized_text)
    return scaled_text.toarray()

def calculate_accuracy():
    correct_predictions = 0
    for i, input_data in enumerate(X_test_vect_dense):
        input_data = input_data.astype(np.float32).reshape(1, -1)
        result = session.run([output_name], {input_name: input_data})
        predicted_label = np.argmax(result[0])
        if predicted_label == y_test[i]:
            correct_predictions += 1
    accuracy = correct_predictions / len(y_test)
    return accuracy

# Calcular la accuracy antes de iniciar el servidor
accuracy = calculate_accuracy()

@app.post("/predict")
async def predict(data: TextData):
    processed_text = preprocess_text(data.text)
    input_data = processed_text.astype(np.float32).reshape(1, -1)
    result = session.run([output_name], {input_name: input_data})
    predicted_label_index = np.argmax(result[0])
    predicted_label = category_labels[predicted_label_index]

    return {
        "text": data.text,
        "predicted_label": predicted_label,
        "accuracy": accuracy
    }

@app.get("/")
async def read_root():
    return {"message": "Welcome to the text classification API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
