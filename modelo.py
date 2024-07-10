import onnxruntime as rt
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

# Cargar el modelo ONNX
model_path = 'modelPML1.onnx'
session = rt.InferenceSession(model_path)

# Obtener información sobre las entradas y salidas del modelo
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Función para preprocesar los datos de texto
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

# Cargar el conjunto de datos
data = pd.read_csv('bbc_data.csv')
print(data.head())

# Preparación de datos: Limpieza del texto
data['data'] = data['data'].apply(clean_text)

# Tokenización y Vectorización
X = data['data']
y = data['labels']

# Convertir las etiquetas a números
y, category_labels = pd.factorize(y)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Contar la cantidad de datos por etiqueta en el conjunto de entrenamiento
train_label_counts = pd.Series(y_train).value_counts()
print("Cantidad de datos por etiqueta en el conjunto de entrenamiento:")
for idx, count in train_label_counts.items():
    print(f"Etiqueta '{category_labels[idx]}': {count} datos")

# Vectorización y Normalización
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

scaler = StandardScaler(with_mean=False)
X_train_vect = scaler.fit_transform(X_train_vect)
X_test_vect = scaler.transform(X_test_vect)

# Convertir las matrices dispersas a matrices densas
X_train_vect_dense = X_train_vect.toarray()
X_test_vect_dense = X_test_vect.toarray()

# Preprocesar los textos de prueba
test_inputs = np.array([vectorizer.transform([text]).toarray()[0] for text in X_test])

# Realizar las predicciones y calcular la precisión
correct_predictions = 0

for i, input_data in enumerate(test_inputs):
    input_data = input_data.astype(np.float32).reshape(1, -1)  # Asegurarse de que la forma sea correcta
    result = session.run([output_name], {input_name: input_data})
    predicted_label = np.argmax(result[0])
    if predicted_label == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(y_test)
print(f'Accuracy: {accuracy:.2f}')
