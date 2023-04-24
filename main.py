import cv2
import os
import pytesseract
from PIL import Image
import numpy as np
import streamlit as st
from pytesseract import image_to_string

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

counter = 0
def main():
    st.title("Dự án xử lý ảnh số")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    activities = {
        1: "Xử lý chữ viết tay bằng OCR",
        2: "Xử lý chữ viết tay bằng hình thái học"
    }
    choices = st.sidebar.selectbox("Select Option", list(activities.items()), format_func=lambda x: x[1])
    #Xử lý choice 1:
    if choices[0] == 1:
        st.subheader(choices[1])
        img_file = st.file_uploader("Upload File", type=['png', 'jpg', 'jpeg'])
        if img_file is not None:
            up_img = Image.open(img_file)
            st.image(up_img)
        if st.button("Xử lý ngay"):
            try:
                img = np.array(up_img)
                text = pytesseract.image_to_string(img)
                st.write(text)
            except Exception as e:
                st.error(e)
    elif choices[0] == 2:
        st.subheader(choices[1])
        img_file = st.file_uploader("Upload File", type=['png', 'jpg', 'jpeg'])
        if img_file is not None:
            up_img = Image.open(img_file)
            st.image(up_img)
        col1, col2, col3 = st.columns(3)
        
        # Đặt button trong cột 1
        button1 = col1.button("Xử lý bằng dilation và erosion")

        # Đặt button trong cột 2
        button2 = col2.button("Nhận dạng từng chữ cái")
        
        # Đặt button trong cột 3
        button3 = col3.button("Xử lý bằng Dilation")
        
        if button1:
            try:
                # Call the extract_text function to extract the text from the image
                st.write("Kết quả:")
                process_image(up_img)
            except Exception as e:
                st.error("Unknown error: " + str(e))
        if button2:
            try:
                # Call the extract_text function to extract the text from the image
                st.write("Kết quả:")
                process_handwritten_text(up_img)
            except Exception as e:
                st.error("Unknown error: " + str(e))
        if button3:
            try:
                # Call the extract_text function to extract the text from the image
                st.write("Kết quả:")
                process_handwritten_dilation(up_img)
            except Exception as e:
                st.error("Unknown error: " + str(e))
def process_handwritten_dilation(image_path):
    img = np.array(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # (thresh, binary_img) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # Định nghĩa kernel 3x3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Thực hiện phép toán dilation với kernel 3x3
    dilated_img = cv2.dilate(gray, kernel, iterations=1)
    
    # # Thực hiện phép toán erosion với kernel 3x3
    # eroded_img = cv2.erode(dilated_img, kernel, iterations=1)
    
    # Hiển thị ảnh đã được dilation bằng Streamlit
    st.image(dilated_img, channels='GRAY')

def process_handwritten_text(image_path):
    # Đọc ảnh và chuyển sang ảnh xám
    img = np.array(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Áp dụng bộ lọc trung bình để loại bỏ các thông tin cao tần
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Áp dụng bộ lọc cạnh để tìm ra các đường viền
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # Tìm khu vực chứa chữ viết tay bằng giải thuật Otsu
    ret, thresh = cv2.threshold(blur_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Đưa ra kết quả cuối cùng bằng cách đánh dấu các khu vực chứa chữ viết tay trên ảnh gốc
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    with st.container():
        st.image(img)
# Hàm xử lý ảnh và trích xuất chữ
def process_image(image):
     # Chuyển đổi ảnh sang định dạng numpy array
    img = np.array(image.convert('RGB'))
    threshval = 127; n = 255
    retval, imB = cv2.threshold(img, threshval, n, cv2.THRESH_BINARY)
    # Thực hiện phép toán Erosion và Dilationz
    kernel = np.ones((1,1), np.uint8)
    img_dil = cv2.dilate(imB, kernel, iterations=1)
    img_dil = cv2.erode(img_dil, kernel, iterations=1)
    with st.container():
        st.image(img_dil)
    
if __name__ == '__main__':
    main()
    
