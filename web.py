import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import time
import os

# Set page config at the very start
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌱", layout="wide")

# Custom CSS for UI enhancement
st.markdown(
    """
    <style>
        .main { background-color: #F4F4F4; }
        .stButton button { background-color: #4CAF50; color: white; font-size: 16px; border-radius: 8px; }
        .stSidebar { background-color: #DFF0D8; }
        .header { text-align: center; font-size: 28px; font-weight: bold; color: #4CAF50; }
    </style>
    """,
    unsafe_allow_html=True
)

# Display loading indicator for model
@st.cache_resource
def load_model():
    with st.spinner('Loading model... Please wait ⏳'):
        time.sleep(2)  # Simulate loading delay
        return tf.keras.models.load_model('trained_plant_disease_model.keras')

model = load_model()
st.success("Model loaded successfully! ✅")

def model_prediction(image):
    image = image.resize((128, 128))  # Resize image to match model input size
    input_arr = np.array(image) / 255.0  # Normalize pixel values
    input_arr = np.expand_dims(input_arr, axis=0)  # Expand dimensions
    predictions = model.predict(input_arr)
    predicted_class = np.argmax(predictions)
    confidence_score = np.max(predictions)  # Get confidence score
    return predicted_class, confidence_score

# Store predictions in a CSV file
prediction_file = "predictions.csv"
if not os.path.exists(prediction_file):
    pd.DataFrame(columns=["Timestamp", "Prediction", "Confidence Score"]).to_csv(prediction_file, index=False)

def log_prediction(prediction, confidence):
    df = pd.read_csv(prediction_file)
    new_entry = pd.DataFrame({"Timestamp": [pd.Timestamp.now()], "Prediction": [prediction], "Confidence Score": [confidence]})
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(prediction_file, index=False)

# Disease information
plant_diseases = {
    "Potato___Early_blight": {
        "solution": "Use fungicides with chlorothalonil or copper-based sprays. Remove infected leaves. Ensure good air circulation.",
        "fact": "Early blight is caused by Alternaria solani, and it spreads in humid conditions.",
        "global_impact": "Early blight is a common issue affecting potato crops worldwide, causing up to 30% yield loss in severe cases."
    },
    "Potato___Late_blight": {
        "solution": "Apply fungicides like metalaxyl or mancozeb. Remove infected plants immediately. Avoid overhead watering.",
        "fact": "Late blight was responsible for the Irish Potato Famine in the 1840s!",
        "global_impact": "Late blight is one of the most destructive potato diseases worldwide, leading to billions of dollars in crop losses annually."
    },
    "Potato___Healthy": {
        "solution": "Great! No disease detected. Maintain proper watering and nutrient supply.",
        "fact": "Healthy potato plants produce about 10-20 potatoes per season!",
        "global_impact": "Healthy potato crops are essential for food security, as potatoes are the world's fourth-largest food crop."
    }
}

# Sidebar Navigation
st.sidebar.title("🌿 Plant Disease System")
page = st.sidebar.radio("Navigation", ["Home", "Disease Recognition", "Prediction History"])

if page == "Home":
    st.markdown("<p class='header'>Welcome to the Plant Disease Detection System 🌱</p>", unsafe_allow_html=True)
    st.write("This system helps farmers identify potato leaf diseases and suggests treatment solutions.")
    st.image("Disease.png", use_container_width=True)

elif page == "Disease Recognition":
    st.header("🔍 Disease Recognition")
    uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, use_container_width=True, caption='Uploaded Image')
        
        if st.button("Predict Disease"):
            with st.spinner("Analyzing Image... 🤔"):
                result_index, confidence_score = model_prediction(image)
                class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___Healthy']
                disease_name = class_names[result_index]
                
                log_prediction(disease_name, round(confidence_score * 100, 2))
                
                st.success(f"**Prediction:** {disease_name}")
                st.info(f"**Confidence Score:** {round(confidence_score * 100, 2)}%")
                st.info(f"**Solution:** {plant_diseases[disease_name]['solution']}")
                st.warning(f"**Fun Fact:** {plant_diseases[disease_name]['fact']}")
                
                # Display Global Impact
                st.markdown("### 🌍 Global Impact")
                st.write(f"{plant_diseases[disease_name]['global_impact']}")

elif page == "Prediction History":
    st.header("📜 Prediction History")
    df = pd.read_csv(prediction_file)
    st.dataframe(df, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.write("Developed with ❤️ for Sustainable Agriculture")
