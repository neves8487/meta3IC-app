import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pickle

# Configuração da Página
st.set_page_config(page_title="Classificador de Sinais", layout="centered")
st.title("Classificador de Sinais de Trânsito 🚦")
st.write("Projeto de Inteligência Computacional - Fase III")

# Carregar o Label Encoder
try:
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
        classes = label_encoder.classes_
except FileNotFoundError:
    st.error("Erro: 'label_encoder.pkl' não encontrado. Executa o treino primeiro.")
    classes = []

# Carregar o Modelo
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('modelo_sinais_simples.h5')
    except:
        return None

model = load_model()

# Interface de Upload
file = st.file_uploader("Carrega uma imagem do sinal", type=["jpg", "png", "jpeg"])

def import_and_predict(image_data, model):
    # Redimensionar para 224x224 (MobileNetV2)
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img = img.astype('float32') / 255.0
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is not None:
    image = Image.open(file)
    st.image(image, caption='Imagem carregada', use_column_width=True)
    
    if model is None:
        st.error("Erro: Modelo não encontrado.")
    else:
        if st.button("Classificar"):
            predictions = import_and_predict(image, model)
            
            score = tf.nn.softmax(predictions[0])
            class_idx = np.argmax(predictions, axis=1)[0]
            class_name = classes[class_idx]
            confidence = np.max(predictions)
            
            st.success(f"Resultado: **{class_name}**")
            st.info(f"Confiança: {confidence*100:.2f}%")