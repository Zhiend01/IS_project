import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import gdown

# โหลดโมเดล CNN ที่เทรนไว้
cnn_model = tf.keras.models.load_model("cnn_model.h5")

# Google Drive ID ของ svm_model.pkl
svm_model_id = "1YyCwd6HRKVm5fXlClO93PMT0of1606I1"
svm_model_path = "svm_model.pkl"

# ดาวน์โหลดไฟล์จาก Google Drive ถ้าไม่มีอยู่ในเครื่อง
try:
    with open(svm_model_path, "rb") as f:
        print("✅ Found existing SVM model file, no need to download")
except FileNotFoundError:
    print("📥 Downloading SVM model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={svm_model_id}", svm_model_path, quiet=False)

# โหลดโมเดล ML (SVM และ Random Forest)
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")
 
# ฟังก์ชัน Preprocess รูปภาพ
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize
    img_array = np.array(image) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

def extract_hog_feature(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    resized_gray = cv2.resize(gray, (128, 128))  # ปรับขนาดภาพให้เท่ากับที่ใช้ฝึกโมเดล
    feature = hog(resized_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return feature.reshape(1, -1)


# UI หน้าแรก
st.title("โปรแกรมจำแนกภาพแมวและสุนัข")
st.sidebar.title("เลือกหน้า")
page = st.sidebar.radio("ไปยัง", ["ทดสอบ ML", "ทดสอบ CNN", "ทฤษฎี ML", "ทฤษฎี CNN"])

if page == "ทดสอบ ML":
    st.header("ทดสอบการจำแนกภาพโดยใช้ Machine Learning (SVM & Random Forest)")
    st.write("ในหน้านี้ คุณสามารถอัปโหลดรูปภาพของแมวหรือสุนัข และให้โมเดล Machine Learning ทำการทำนายว่าภาพนั้นเป็นแมวหรือสุนัข โดยใช้โมเดล SVM และ Random Forest")
    st.write("ตัวอย่างการทำงาน Machine Learning")
    st.image("Img.png", caption="กระบวนการพัฒนาโมเดล Machine Learning", use_container_width=True)


    uploaded_file = st.file_uploader("อัปโหลดรูปภาพ (JPG หรือ PNG)", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="รูปที่อัปโหลด", use_container_width=True)
        feature = extract_hog_feature(image)
        prediction_svm = svm_model.predict(feature)[0]
        prediction_rf = rf_model.predict(feature)[0]
        labels = ["แมว", "สุนัข"]
        st.write(f"ผลลัพธ์จาก SVM: {labels[prediction_svm]}")
        st.write(f"ผลลัพธ์จาก Random Forest: {labels[prediction_rf]}")

elif page == "ทดสอบ CNN":
    st.header("ทดสอบการจำแนกภาพโดยใช้ Deep Learning (CNN)")
    st.write("ในหน้านี้ คุณสามารถอัปโหลดรูปภาพของแมวหรือสุนัข และให้โมเดล CNN ทำการทำนายผล")
    st.write("ตัวอย่างการทำงาน Neural Network")
    st.image("Img2.png", caption="กระบวนการพัฒนาโมเดล Machine Learning", use_container_width=True)


    uploaded_file = st.file_uploader("อัปโหลดรูปภาพ (JPG หรือ PNG)", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="รูปที่อัปโหลด", use_container_width=True)
        processed_image = preprocess_image(image)
        prediction_cnn = cnn_model.predict(processed_image)
        predicted_label = np.argmax(prediction_cnn)
        labels = ["แมว", "สุนัข"]
        st.write(f"ผลลัพธ์จาก CNN: {labels[predicted_label]}")

elif page == "ทฤษฎี ML":
    st.header("หลักการทำงานของ Machine Learning")
    st.write("### 1. ที่มาของ Dataset")
    st.write("Dataset ที่ใช้มาจาก **TensorFlow Cats vs Dogs Dataset** ซึ่งมีรูปภาพของแมวและสุนัขจำนวน 25,000 รูป")
    
    st.write("### 2. อธิบายคุณลักษณะของ Dataset")
    st.write("Dataset มีรูปภาพของแมวและสุนัขในหลากหลายมุมและสภาพแวดล้อม ขนาดของภาพไม่สม่ำเสมอ ต้องมีการปรับแต่งก่อนนำเข้าโมเดล")
    
    st.write("### 3. แนวทางการพัฒนาโมเดล")
    st.write("ใช้วิธี Machine Learning โดยนำคุณลักษณะของภาพมาแปลงเป็นข้อมูลที่สามารถวิเคราะห์ได้ โดยใช้ HOG Feature Extraction")
    
    st.write("### 4. การเตรียมข้อมูล")
    st.write("1. แปลงรูปภาพเป็น Grayscale")
    st.write("2. ใช้ HOG Feature Extraction เพื่อดึงคุณลักษณะ")
    st.write("3. แบ่งข้อมูลออกเป็นชุดฝึกและชุดทดสอบ")
    
    st.write("### 5. ทฤษฎีของอัลกอริทึมที่พัฒนา")
    st.write("- **SVM (Support Vector Machine):** ใช้เส้นแบ่งเขตแดนเพื่อจำแนกข้อมูล")
    st.write("- **Random Forest:** ใช้หลาย Decision Tree เพื่อสร้างโมเดลที่แม่นยำ")
    
    st.write("### 6. ขั้นตอนการพัฒนาโมเดล")
    st.write("1. ดึงข้อมูลจาก Dataset")
    st.write("2. ทำ Preprocessing โดยใช้ HOG Feature Extraction")
    st.write("3. ฝึกโมเดล SVM และ Random Forest บนข้อมูลที่เตรียมไว้")
    st.write("4. ทดสอบโมเดลกับชุดข้อมูลทดสอบ")
    st.write("5. วัดผลลัพธ์และปรับแต่งโมเดล")

elif page == "ทฤษฎี CNN":
    st.header("หลักการทำงานของ Convolutional Neural Network (CNN)")
    st.write("### 1. ที่มาของ Dataset")
    st.write("Dataset ที่ใช้มาจาก **TensorFlow Cats vs Dogs Dataset** ซึ่งมีรูปภาพของแมวและสุนัขจำนวน 25,000 รูป")
    
    st.write("### 2. อธิบายคุณลักษณะของ Dataset")
    st.write("Dataset มีขนาดภาพไม่สม่ำเสมอ ต้องมีการทำ Preprocessing เช่น Resizing และ Normalization ก่อนนำเข้าโมเดล CNN")
    
    st.write("### 3. แนวทางการพัฒนาโมเดล")
    st.write("ใช้ Convolutional Neural Network (CNN) ซึ่งเป็นโครงข่ายประสาทเทียมที่สามารถดึงลักษณะเด่นจากรูปภาพและใช้ในการจำแนกประเภท")
    
    st.write("### 4. การเตรียมข้อมูล")
    st.write("1. ปรับขนาดภาพให้เป็น 128x128 pixels")
    st.write("2. ทำ Normalization เพื่อลดค่าความต่างของพิกเซล")
    st.write("3. แบ่งข้อมูลออกเป็นชุดฝึกและชุดทดสอบ")
    
    st.write("### 5. ทฤษฎีของอัลกอริทึมที่พัฒนา")
    st.write("- **Conv2D:** ใช้ตัวกรองเพื่อดึงคุณลักษณะของภาพ")
    st.write("- **MaxPooling2D:** ลดขนาดของภาพเพื่อให้คำนวณได้เร็วขึ้น")
    st.write("- **Fully Connected Layers:** เชื่อมโยงคุณลักษณะเพื่อทำการจำแนกประเภท")
    
    st.write("### 6. ขั้นตอนการพัฒนาโมเดล")
    st.write("1. โหลดข้อมูลและทำการ Normalize")
    st.write("2. สร้างโมเดล CNN โดยใช้ Conv2D, MaxPooling2D และ Fully Connected Layers")
    st.write("3. ใช้ Optimizer Adam และ Loss Function Categorical Cross-Entropy")
    st.write("4. ฝึกโมเดลด้วยข้อมูลฝึกและทดสอบผลลัพธ์กับข้อมูลทดสอบ")