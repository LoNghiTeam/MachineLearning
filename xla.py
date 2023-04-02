from keras.models import load_model
import cv2
import csv
import os
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from PIL import Image, ImageEnhance
import numpy as np
import streamlit as st
import warnings
import pandas as pd
def xla():
    img_file = st.file_uploader("Upload File", type=['png', 'jpg', 'jpeg'])
    if img_file is not None:
        up_img = Image.open(img_file)
        st.image(up_img)
    if st.button("Predict Now"):
        try:
            img = np.array(up_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # Phần tử cấu trúc
            kernel = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
            # Thực hiện erosion và dilation
            erosion = cv2.erode(img, kernel, iterations=1)
            dilation = cv2.dilate(img, kernel, iterations=1)
            
            # Hiển thị hai hình ảnh cùng trên một hàng
            percent = 50
            new_width = int(img.shape[1] * percent / 100)
            new_height = int(img.shape[0] * percent / 100)
            erosion = cv2.resize(erosion,(new_width, new_height),interpolation=cv2.INTER_AREA)
            dilation = cv2.resize(dilation,(new_width, new_height),interpolation=cv2.INTER_AREA)
            col1, col2 = st.columns(2)
            with col1:
                st.title("Erosion")
                st.image(erosion)
            with col2:
                st.title("Dilation")
                st.image(dilation)
        except Exception as e:
            st.error("Connection Error")