import streamlit as st
from fastai.vision.all import *
import io

# Load the model
model = load_learner('models/new_model.pkl')

st.title('Cat Classifier: Appa vs Momo')

# Option to choose input type
input_type = st.radio("Choose input type:", ('Upload an image', 'Take a picture'))

if input_type == 'Upload an image':
    uploaded_file = st.file_uploader("Choose an image of Appa or Momo", type=['jpg', 'png', 'jpeg'])
elif input_type == 'Take a picture':
    uploaded_file = st.camera_input("Take a picture of Appa or Momo")

if uploaded_file is not None:
    # Display the uploaded/taken image
    image_data = uploaded_file.getvalue()
    st.image(image_data, caption='Uploaded/Taken Image.', use_column_width=True)
    img = PILImage.create(io.BytesIO(image_data))

    # Predict button
    if st.button('Identify'):
        # Prediction
        pred, pred_idx, probs = model.predict(img)
        st.write(f'Prediction: {pred}')
        st.write(f'Probability: {probs[pred_idx]:.04f}')
