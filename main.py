import streamlit as st
import tensorflow as tf 
import numpy as np 
import gdown
import os

def download_model():
    if not os.path.exists("trained_best_model2.h5"):
        url = "https://drive.google.com/file/d/1AH2jsefRTUGeM1h8hr1VdjbfKCe-2uqW"
        output = "trained_best_model2.h5"
        gdown.download(url, output, quiet=False)
    else:
        print("El modelo ya está descargado")

def model_prediction(test_image):
    download_model()
    model = tf.keras.models.load_model("trained_best_model2.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

#sidebar
st.sidebar.title("Panel")
app_mode=st.sidebar.selectbox("Seleccionar", ["Inicio", "Información general", "Identificación del producto"])

#main page
if (app_mode=="Inicio"):
    st.markdown(
    "<h2 style='text-align: center;'>Identificación del estado de productos de exportación perecederos para mejorar el proceso de control de calidad: Bananos, Manzanas y Naranjas</h2>",
    unsafe_allow_html=True)
    image_path = "img_01.jpg"
    st.image(image_path)

#About project
elif(app_mode=="Información general"):
    st.header("Información general")
    st.subheader("Objetivo General")

    st.text("Este proyecto busca ayudar a mejorar el proceso de control de calidad\nde ciertos productos perecederos, en específico, frutas por medio de un modelo\nque identifique si el producto está en buen o mal estado. Este tipo de productos\ntienen que contar con una rigurosa inspección antes de ser exportados, por lo\ncual este modelo ayudaría a separar y filtrar este tipo de productos y garantizar\nla efectividad del control de calidad. En otras palabras, el producto debe estar\nen óptimas condiciones para el consumo y para prevenir contaminar al resto del\nproducto durante todo su proceso de exportación")
    st.subheader("Stakeholders")
    st.text("En lo que se lleva del 2025 las exportaciones del grupo de productos Agropecuarios,\nalimentos y bebidas ha tenido una contribución de 11 Millones de\ndólares habiendo crecido un 36.9% versus el inicio del año según cifras del DANE\ny su tendencia es positiva. Dado que la demanda de este tipo de productos crece,\nes crucial un buen control de calidad y contar con herramientas tecnológicas que permitan una mayor efectividad y precisión a la hora de escoger estos productos.\nEste rubro será vital para la economía colombiana y para todas las empresas que\nestán en busca de nuevos clientes internacionales y genera confianza en la calidad de sus productos.")
    st.subheader("Expectativas")
    st.text("Mejorar la efectividad del proceso de selección de frutas, minimizando el riesgo de\nreportar frutas en mal estado en pedidos de exportación y evitar\npérdidas monetarias y de reputación a la compañía.")
    image_path = "img_02.jpg"
    st.image(image_path, caption="Exportaciones 2025", use_column_width=True)
#Identificación del producto
elif(app_mode=="Identificación del producto"):
    st.header("Identificación del producto")
    test_image = st.file_uploader("Subir una imagen:")
    if(st.button("Visualizar la imagen")):
        st.image(test_image, width=4,use_column_width=True)
    #Button
    if(st.button("Predecir")):
        st.write("Nuestra predicción")
        result_index = model_prediction(test_image)
        #Reading labels
        with open("labels.txt") as f:
            content = f.readlines()
        label = []
        
        for i in content:
            label.append(i[:-1])
        st.success("El modelo predice que la futa es {}".format(label[result_index]))
       
       
