import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the custom model
model_custom = load_model('model_custom.h5')  # Ensure you have the appropriate model loading function
mask_label = {0: 'MASK INCORRECT', 1: 'MASK', 2: 'NO MASK'}
color_label = {0: (0, 255, 255), 1: (0, 255, 0), 2: (255, 0, 0)}

# Function to perform face recognition
def recognize_faces(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    faces = extract_faces(img)  # Extract faces from the image
    for face in faces:
        resized_face = cv2.resize(face, (35, 35))
        reshaped_face = np.reshape(resized_face, [1, 35, 35, 3]) / 255.0
        face_result = model_custom.predict(reshaped_face)
        # Modify this part to draw bounding boxes and labels on the image
        # You can refer to the original code for drawing bounding boxes and labels

        # Draw bounding boxes and labels on the image
        # cv2.putText(), cv2.rectangle() can be used here

    return img

# Function to extract faces using OpenCV's pre-trained face detector
def extract_faces(image):
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    face_images = []
    for (x, y, w, h) in faces:
        face_images.append(image[y:y+h, x:x+w])
    return face_images

# Streamlit app
def main():
    st.title("Face Recognition App")
    st.write("Upload an image to recognize faces.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform face recognition and display the result
        result_image = recognize_faces(image)
        st.image(result_image, caption='Result', use_column_width=True)

if __name__ == '__main__':
    main()


