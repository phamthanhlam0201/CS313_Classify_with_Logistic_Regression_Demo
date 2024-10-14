import streamlit as st
import cv2
import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from joblib import load
import matplotlib.pyplot as plt

classes = ['Lung Adenocarcinoma', 'Lung Benign Tissue <3', 'Lung Squamous Cell Carcinoma']

def compute_glcm_features(image):
    # Chuyển đổi ảnh sang grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = []
    distances = [1]  # You can experiment with different distances
    #angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # You can experiment with different angles
    glcm = graycomatrix(image, distances=distances, angles=[0], levels=256,
                        symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').ravel()
    energy = graycoprops(glcm, 'energy').ravel()
    homogeneity = graycoprops(glcm, 'homogeneity').ravel()
    correlation = graycoprops(glcm, 'correlation').ravel()
    asm = graycoprops(glcm, 'ASM').ravel()  # Angular Second Moment
    entropy = -np.sum(glcm * np.log(glcm + 1e-10))  # Compute entropy manually
    glcm_features = np.concatenate((contrast, energy, homogeneity, correlation, asm, [entropy]))
    features.append(glcm_features)
    return np.array(features)

def predict_image(image,model):
    X = compute_glcm_features(image)
    y_pred = model.predict(X)
    print(X)
    return y_pred

lr_loaded = load('logistic_regression_model.joblib')


st.markdown(f"<h2 style='text-align: center;'>LOGISTIC REGRESSION DEMO</h2>", unsafe_allow_html=True)

uploaded_file_image = st.file_uploader("Upload an image...", type=['png', 'jpg', 'jpeg'])

image_placeholder = st.empty()

if uploaded_file_image is not None:
    image_path = uploaded_file_image.name
    image_name, _ = os.path.splitext(image_path)
    image = np.array(bytearray(uploaded_file_image.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Convert the image to RGB for correct display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Sử dụng container để tổ chức caption và hình ảnh
    with st.container():
        # Hiển thị caption trước khi render hình ảnh
        st.markdown(f"<h4 style='text-align: center;'>TRUE LABEL: {image_name.upper()}</h4>", unsafe_allow_html=True)

        # Render hình ảnh
        st.image(image, caption='', use_column_width=True)
    
    # Container for button and classification result
    y_predict = -1
    with st.container():
        # Create an empty container for the button
        _,_, col3, _, _ = st.columns(5)
        # Create the button
        with col3:
            if st.button("Click me to classify!"):
                y_pred = predict_image(image, lr_loaded)
                y_predict = y_pred[0]
                # Display the predicted label
    with st.container():
        column1, column2, column3 = st.columns(3)
        with column1:
            # Giả sử y_pred[0] có giá trị, nếu là 0 thì đổi nền thành đỏ
            button_color = "white"  # Mặc định là trắng
            text_color = 'black' # Màu chữ 
            if y_predict == 0:  # Nếu y_pred[0] là 0 thì đổi màu nền thành đỏ
                button_color = "red"
                text_color = "white"
            st.markdown(
                f"""
                <div style='text-align: center;'>
                    <button style="background-color: {button_color}; color: {text_color}"
                    window.location.reload();">{classes[0].upper()}</button>
                </div>
                <style>
                    .button {{
                        border: none; /* Không có đường viền */
                        padding: 15px 32px; /* Padding */
                        text-align: center; /* Căn giữa */
                        text-decoration: none; /* Không có gạch chân */
                        display: inline-block; /* Hiển thị như một khối nội tuyến */
                        font-size: 16px; /* Kích thước chữ */
                        margin: 4px 2px; /* Margin */
                        border-radius: 4px; /* Bo góc */
                    }}
                </style>
                """,
                unsafe_allow_html=True
            )
        with column2:
            # Giả sử y_pred[0] có giá trị, nếu là 0 thì đổi nền thành đỏ
            button_color = "white"  # Mặc định là trắng
            if y_predict == 2:  # Nếu y_pred[0] là 0 thì đổi màu nền thành đỏ
                button_color = "red"
            st.markdown(
                f"""
                <div style='text-align: center;'>
                    <button style="background-color: {button_color}; color: {text_color}"
                    window.location.reload();">{classes[2].upper()}</button>
                </div>
                <style>
                    .button {{
                        border: none; /* Không có đường viền */
                        padding: 15px 32px; /* Padding */
                        text-align: center; /* Căn giữa */
                        text-decoration: none; /* Không có gạch chân */
                        display: inline-block; /* Hiển thị như một khối nội tuyến */
                        font-size: 16px; /* Kích thước chữ */
                        margin: 4px 2px; /* Margin */
                        border-radius: 4px; /* Bo góc */
                    }}
                </style>
                """,
                unsafe_allow_html=True
            )
        with column3:
            # Giả sử y_pred[0] có giá trị, nếu là 0 thì đổi nền thành đỏ
            button_color = "white"  # Mặc định là trắng
            if y_predict == 1:  # Nếu y_pred[0] là 0 thì đổi màu nền thành đỏ
                button_color = "red"
            st.markdown(
                f"""
                <div style='text-align: center;'>
                    <button style="background-color: {button_color}; color: {text_color}"
                    window.location.reload();">{classes[1].upper()}</button>
                </div>
                <style>
                    .button {{
                        border: none; /* Không có đường viền */
                        padding: 15px 32px; /* Padding */
                        text-align: center; /* Căn giữa */
                        text-decoration: none; /* Không có gạch chân */
                        display: inline-block; /* Hiển thị như một khối nội tuyến */
                        font-size: 16px; /* Kích thước chữ */
                        margin: 4px 2px; /* Margin */
                        border-radius: 4px; /* Bo góc */
                    }}
                </style>
                """,
                unsafe_allow_html=True
            )
    # if y_predict != -1:
    #     st.markdown(f"<h4 style='text-align: center;'>PREDICTED LABEL: {classes[y_predict].upper()}</h4>", unsafe_allow_html=True)

else:
    st.warning("Please upload an image file to proceed.")
