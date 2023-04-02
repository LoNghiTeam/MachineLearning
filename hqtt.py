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

def rendFile():
    # Tạo dữ liệu giá nhà với 1 biến đầu vào X và 1 biến đầu ra y
        # Tạo textbox để nhập độ lớn mảng
        n = st.number_input("Nhập độ lớn mảng:", min_value=10, max_value=1000, value=100)
        
        # Khởi tạo biến X và biến y
        X = []
        y = []
        dem = 0
        # Nút Random mảng
        if st.button("Random mảng"):
            dem += 1
            X = [random.uniform(50, 200) for i in range(int(n))]
            y = [x * 1000 + random.normalvariate(0, 10000) for x in X]
            # Lưu dữ liệu vào file csv
            with open('data.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['X', 'y'])
                for i in range(len(X)):
                    writer.writerow([X[i], y[i]])
            st.write("Đã sinh mảng ngẫu nhiên")
            
            # Hiển thị giá trị của mảng X và Y trên cùng một hàng
            col1, col2 = st.columns(2)
            #Xem mảng X
            if (X):
                with col1:
                    with st.expander("Giá trị của mảng X:"):
                        st.write(X)
            #Xem mảng Y
            if (y):
                with col2:
                    with st.expander("Giá trị của mảng Y:"):
                        st.write(y)
def HQTT():
    
    # Đọc dữ liệu từ file csv bằng pandas
        data = pd.read_csv('data.csv')

        # Chọn biến đầu vào và đầu ra
        x_col = st.selectbox("Chọn biến đầu vào", list(data.columns))
        y_col = st.selectbox("Chọn biến đầu ra", list(data.columns))

        # Tách dữ liệu thành 2 tập train set và test set
        X_train, X_test, y_train, y_test = train_test_split(data[x_col], data[y_col], test_size=0.2)

        # Khởi tạo mô hình Linear Regression
        model = LinearRegression()

        # Huấn luyện mô hình trên tập train set
        model.fit(X_train.values.reshape(-1,1), y_train.values.reshape(-1,1))

        # Dự đoán kết quả trên tập test set
        y_pred = model.predict(X_test.values.reshape(-1,1))

        # Tính toán mean squared error giữa kết quả dự đoán và kết quả thực tế trên tập test set
        mse = mean_squared_error(y_test.values.reshape(-1,1), y_pred)

        # Vẽ biểu đồ hồi quy tuyến tính
        fig, ax = plt.subplots()
        ax.scatter(X_test, y_test, color='blue')
        ax.plot(X_test, y_pred, color='red')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title("Hồi quy tuyến tính giữa {} và {}".format(x_col, y_col))
        st.pyplot(fig)

        # In ra kết quả mean squared error
        st.write('Mean Squared Error:', mse)
        