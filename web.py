import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import time


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AgroScan - Plant Disease Detection",
    page_icon="🌱",
    layout="wide"
)


# -----------------------------
# Custom UI Styling
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #F4F4F4;
}

.stButton button {
    background-color: #4CAF50;
    color: white;
    font-size: 16px;
    border-radius: 8px;
}

.header {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    color: #2E8B57;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():

    with st.spinner("Loading AI Model... Please wait ⏳"):
        time.sleep(1)

        model = tf.keras.models.load_model(
            "trained_plant_disease_model.keras",
            compile=False
        )

    return model


model = load_model()
st.success("Model loaded successfully ✅")


# -----------------------------
# Class Labels
# -----------------------------
class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___Healthy"
]


# -----------------------------
# Prediction Function
# -----------------------------
def model_prediction(image):

    image = image.convert("RGB")
    image = image.resize((128,128))

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    confidence_score = np.max(prediction)

    return predicted_class, confidence_score, prediction[0]


# -----------------------------
# Prediction Logging
# -----------------------------
prediction_file = "predictions.csv"

if not os.path.exists(prediction_file):
    pd.DataFrame(
        columns=["Timestamp","Prediction","Confidence"]
    ).to_csv(prediction_file,index=False)


def log_prediction(prediction,confidence):

    df = pd.read_csv(prediction_file)

    new_row = pd.DataFrame({
        "Timestamp":[pd.Timestamp.now()],
        "Prediction":[prediction],
        "Confidence":[confidence]
    })

    df = pd.concat([df,new_row],ignore_index=True)

    df.to_csv(prediction_file,index=False)


# -----------------------------
# Disease Information
# -----------------------------
plant_diseases = {

"Potato___Early_blight":{

"solution":"Use fungicides containing chlorothalonil or copper sprays. Remove infected leaves and maintain air circulation.",

"fact":"Early blight is caused by Alternaria solani and spreads quickly in warm humid weather.",

"impact":"Can reduce potato yield by up to 30% if untreated."

},

"Potato___Late_blight":{

"solution":"Apply fungicides like metalaxyl or mancozeb. Remove infected plants immediately.",

"fact":"Late blight caused the Irish Potato Famine in the 1840s.",

"impact":"One of the most destructive crop diseases globally."

},

"Potato___Healthy":{

"solution":"No disease detected. Continue proper watering and nutrient supply.",

"fact":"Healthy potato plants produce around 10–20 potatoes per plant.",

"impact":"Healthy crops support global food security."

}

}


# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("🌿 AgroScan Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Home","Disease Detection","Prediction History"]
)


# -----------------------------
# HOME PAGE
# -----------------------------
if page == "Home":

    st.markdown(
        '<p class="header">AgroScan - AI Plant Disease Detection 🌱</p>',
        unsafe_allow_html=True
    )

    if os.path.exists("Disease.png"):
        st.image("Disease.png", width="stretch")


# -----------------------------
# DISEASE DETECTION PAGE
# -----------------------------
elif page == "Disease Detection":

    st.header("🔍 Upload Potato Leaf Image")

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_image:

        image = Image.open(uploaded_image)

        st.image(
            image,
            caption="Uploaded Image",
            width="stretch"
        )

        if st.button("Analyze Disease"):

            with st.spinner("Analyzing image..."):

                result_index,confidence_score,predictions = model_prediction(image)

                disease_name = class_names[result_index]

                confidence_percent = round(confidence_score*100,2)

                log_prediction(disease_name,confidence_percent)

                st.success(f"Prediction: {disease_name}")

                st.info(f"Confidence Score: {confidence_percent}%")

                if confidence_percent < 60:

                    st.warning(
                        "Low confidence prediction. Try uploading a clearer leaf image."
                    )

                st.subheader("Treatment Recommendation")

                st.write(
                    plant_diseases[disease_name]["solution"]
                )

                st.subheader("Interesting Fact")

                st.write(
                    plant_diseases[disease_name]["fact"]
                )

                st.subheader("Global Impact")

                st.write(
                    plant_diseases[disease_name]["impact"]
                )

                # -----------------------------
                # Probability Chart
                # -----------------------------
                st.subheader("Prediction Confidence for Each Disease")

                prob_df = pd.DataFrame({
                    "Disease": class_names,
                    "Probability": predictions
                })

                prob_df["Probability"] = prob_df["Probability"] * 100

                st.bar_chart(
                    prob_df.set_index("Disease")
                )


# -----------------------------
# PREDICTION HISTORY
# -----------------------------
elif page == "Prediction History":

    st.header("📊 Prediction History")

    if os.path.exists(prediction_file):

        df = pd.read_csv(prediction_file)

        st.dataframe(df, width="stretch")

    else:

        st.info("No predictions recorded yet.")


# -----------------------------
# Footer
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.write("Developed with ❤️ for Smart Agriculture")
