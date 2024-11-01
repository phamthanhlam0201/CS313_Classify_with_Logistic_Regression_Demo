import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from joblib import load
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

classes = ['Lung Adenocarcinoma', 'Lung Benign Tissue <3', 'Lung Squamous Cell Carcinoma']
n = 16 # Số đặc trưng sau khi giảm chiều

def compute_glcm_features(image):

    # Chuyển đổi ảnh sang grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Tính toán với 8 hướng xung quanh (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]

    # Tính GLCM và các đặc trưng trực tiếp
    glcm = graycomatrix(image, distances=[1], angles=angles, levels=256, symmetric=True, normed=True)
    
    # Tính các đặc trưng và kết hợp chúng trong một dòng
    features = np.concatenate([
        graycoprops(glcm, 'contrast').ravel(),
        # Tính toán đặc trưng contrast (độ tương phản) của ma trận đồng xuất GLCM. Độ tương phản đo lường mức độ thay đổi trong cường độ pixel, cho biết sự khác biệt giữa các pixel trong ảnh.
        
        graycoprops(glcm, 'energy').ravel(), 
        # ASM Tính toán đặc trưng energy (năng lượng) của GLCM, thường được gọi là Angular Second Moment (ASM). Đặc trưng này cho biết mức độ đồng nhất trong cường độ pixel.
        
        graycoprops(glcm, 'homogeneity').ravel(),
        # Tính toán đặc trưng homogeneity (đồng nhất) của GLCM. Đặc trưng này đo lường mức độ tương đồng giữa các pixel.
        
        graycoprops(glcm, 'correlation').ravel(), 
        # Tính toán đặc trưng correlation (tương quan) của GLCM. Đặc trưng này đo lường mức độ liên quan giữa các pixel.
        
        [-np.sum(glcm * np.log(glcm + 1e-10))]  
        # Tính toán entropy (độ hỗn loạn) của GLCM. Entropy đo lường mức độ không chắc chắn trong phân phối cường độ pixel. Để tránh lỗi số học khi tính log, một hằng số nhỏ (1e-10) được cộng vào GLCM.
    ])
    
    return features

def predict_image(image,scaler, pca, model):
    X = compute_glcm_features(image)
    
    X = scaler.transform([X])
    X = pca.transform(X)
    
    y_pred = model.predict(X)
    
    # Tạo tên cột cho DataFrame
    columns = []
    # Mỗi tập hợp đặc trưng GLCM gồm 4 đặc trưng cho mỗi góc (tổng cộng 8 góc)
    # (Contrast, Energy, Homogeneity, Correlation)
    for i in range(n):
        columns.extend([
            f"Feature_{i}", 
        ])
    df_X = pd.DataFrame(X, columns=columns)
    
    # In DataFrame ra màn hình
    print(df_X)
    return y_pred

scaler_loaded = load('scaler.joblib')
pca_loaded = load('pca_transformer.joblib')
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
                y_pred = predict_image(image, scaler_loaded, pca_loaded, lr_loaded)
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
            text_color = 'black' # Màu chữ 
            if y_predict == 2:  # Nếu y_pred[0] là 0 thì đổi màu nền thành đỏ
                button_color = "red"
                text_color = "white"
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
            text_color = 'black' # Màu chữ 
            if y_predict == 1:  # Nếu y_pred[0] là 0 thì đổi màu nền thành đỏ
                button_color = "red"
                text_color = "white"
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

else:
    st.warning("Please upload an image file to proceed.")
