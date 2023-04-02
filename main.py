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
import hqtt
import xla
import hoi_quy_logistic as hql
warnings.filterwarnings('ignore')

def preprocessing(img):
    try:
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = img/255
        return img
    except Exception as e:
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = img/255
        return img


def main():
    st.title("Dự án học máy")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    activities = {
        1: "Hình thái học",
        2: "Hồi quy tuyến tính",
        3: "Hồi quy logistic",
    }
    choices = st.sidebar.selectbox("Select Option", list(activities.items()), format_func=lambda x: x[1])
    #Xử lý choice 1:
    if choices[0] == 1:
        st.subheader(choices[1])
        xla.xla()
    #Xử lý choice 2:
    elif choices[0] == 2:
        st.subheader(choices[1])
        hqtt.rendFile()
        hqtt.HQTT()
    #Xử lý choice 3:
    elif choices[0] == 3:
        st.subheader(choices[1])
        hql.process()
if __name__ == '__main__':
    main()
