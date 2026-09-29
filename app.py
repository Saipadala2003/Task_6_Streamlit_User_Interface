import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="centered"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("deep_learning_model.h5")
    return model


model = load_model()

# --------------------------------------------------
# Application Title
# --------------------------------------------------

st.title("🔢 MNIST Handwritten Digit Classifier")

st.write(
    "Upload an image of a handwritten digit and the trained "
    "deep learning model will predict the digit."
)

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("About the Application")

st.sidebar.write(
    """
    This application uses a trained Convolutional Neural
    Network (CNN) model for MNIST handwritten digit
    classification.

    Input:
    - Handwritten digit image
    - Grayscale preprocessing
    - 28 × 28 image

    Output:
    - Predicted digit
    - Prediction confidence
    """
)

# --------------------------------------------------
# File Upload
# --------------------------------------------------

st.subheader("Upload Digit Image")

uploaded_file = st.file_uploader(
    "Choose a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Input Handwritten Digit",
        width=250
    )

    st.divider()

    if st.button("🔮 Predict Digit", use_container_width=True):

        image = image.convert("L")
        image = image.resize((28, 28))
        image_array = np.array(image)
        image_array = image_array.astype("float32") / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        # Make prediction
        prediction = model.predict(image_array, verbose=0)

        predicted_digit = int(np.argmax(prediction))

        confidence = float(np.max(prediction)) * 100

        # --------------------------------------------------
        # Display Results
        # --------------------------------------------------

        st.success("Prediction completed successfully!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Digit",
                predicted_digit
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        st.subheader("Prediction Probabilities")

        probabilities = prediction[0] * 100

        probability_data = {
            str(i): float(probabilities[i])
            for i in range(10)
        }

        st.bar_chart(probability_data)

else:

    st.info(
        "Please upload a handwritten digit image to begin prediction."
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Task 6 – Creating a Streamlit User Interface | MSc AI"
)