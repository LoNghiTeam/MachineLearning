from keras.models import load_model
import cv2
import csv
import os
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from PIL import Image, ImageEnhance
import numpy as np
import streamlit as st
import warnings
import pandas as pd

def process():
    X = None
    y = None
    
    if st.button("Random mảng"):
        
        np.random.seed(None)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # Huấn luyện mô hình hồi quy logistic
        clf = LogisticRegression(random_state=0).fit(X, y)

        # Hiển thị giá trị của mảng X và Y trên cùng một hàng
        col1, col2 = st.columns(2)
        # Nút Xem mảng X
        if X is not None:
            with col1:
                with st.expander("Giá trị của mảng X:"):
                    st.write(X)

        # Nút Xem mảng Y
        if y is not None:
            with col2:
                with st.expander("Giá trị của mảng Y:"):
                    st.write(y)

        # Vẽ đồ thị
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        st.pyplot(fig)
        
        