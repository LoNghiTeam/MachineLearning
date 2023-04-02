# from win32com.client import Dispatch
from keras.models import load_model
import cv2
import os
from PIL import Image, ImageEnhance
import numpy as np
import streamlit as st
import warnings
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
    st.title("Handwritten Digit Classification Web App")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    activities = ["Program"]
    choices = st.sidebar.selectbox("Select Option", activities)

    if choices == "Program":
        st.subheader("Kindly upload file below")
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
                st.title("erosion")
                st.image(erosion)
                st.title("dilation")
                st.image(dilation)
            except Exception as e:
                st.error("Connection Error")

    # elif choices == 'Credits':
    #     st.write(
    #         "Application Developed by Abhishek Tripathi, Aman Verma, Manvendra Pratap Singh.")


if __name__ == '__main__':
    main()
