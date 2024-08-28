import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import os

# Set the page configuration (this must be the first Streamlit command)
st.set_page_config(page_title="Motor Failure Prediction", page_icon="⚙️", layout="wide")

# Define the correct paths for the model and scaler
model_path = r'C:\Users\dhyne\OneDrive\Desktop\motar_failure\my_model (1).keras'
scaler_path = r'C:\Users\dhyne\OneDrive\Desktop\motar_failure\scaler (1).pkl'

# Load the model using TensorFlow
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
else:
    st.error(f"Model file not found at {model_path}. Please check the file path.")
    st.stop()

# Load the scaler using joblib
if os.path.exists(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        st.success("Scaler loaded successfully.")
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        st.stop()
else:
    st.error(f"Scaler file not found at {scaler_path}. Please check the file path.")
    st.stop()

# Streamlit UI
st.title('Motor Failure Prediction System ⚙️')
st.write("""
    Welcome to the Motor Failure Prediction System! 🚀
    
    This application predicts the likelihood of motor failure based on accelerometer data inputs. Adjust the **X**, **Y**, and **Z** values from the accelerometer to get insights into the motor’s performance.
    
    📊 **How it Works:**
    1. Adjust the accelerometer values using the sliders.
    2. Click on **Predict Motor Failure**.
    3. The system will show you the predicted RPM percentage and whether the motor is likely to face failure or not.
    
    Let's get started!
""")

st.subheader('Adjust Accelerometer Data:')
x_input = st.slider("**X Value**", min_value=-1.0, max_value=1.0, value=0.0, step=0.01, format="%.5f", help="Adjust the X-axis value from the accelerometer.")
y_input = st.slider("**Y Value**", min_value=-1.0, max_value=1.0, value=0.0, step=0.01, format="%.5f", help="Adjust the Y-axis value from the accelerometer.")
z_input = st.slider("**Z Value**", min_value=-1.0, max_value=1.0, value=0.0, step=0.01, format="%.5f", help="Adjust the Z-axis value from the accelerometer.")

if st.button("🔍 Predict Motor Failure"):
    st.write(f"Received inputs - X: {x_input}, Y: {y_input}, Z: {z_input}")

    input_data = np.array([[x_input, y_input, z_input]])
    
    # Scale the input data
    try:
        input_data_scaled = scaler.transform(input_data)
        st.write(f"Scaled input data: {input_data_scaled}")
    except Exception as e:
        st.error(f"Error scaling input data: {e}")
        st.stop()

    # Ensure input data has the correct sequence length
    sequence_length = 20
    if input_data_scaled.shape[0] < sequence_length:
        repeated_data = np.tile(input_data_scaled, (sequence_length, 1))
    else:
        repeated_data = input_data_scaled[-sequence_length:]

    input_data_scaled_reshaped = np.reshape(repeated_data, (1, sequence_length, 3))
    st.write(f"Reshaped input data: {input_data_scaled_reshaped.shape}")

    # Predict using the model
    try:
        rpm_prediction, failure_prediction = model.predict(input_data_scaled_reshaped)
        st.write(f"RPM Prediction Raw Output: {rpm_prediction}")
        st.write(f"Failure Prediction Raw Output: {failure_prediction}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    # Interpret failure prediction
    failure_prediction_binary = 1 if failure_prediction[0][0] >= 0.5 else 0

    st.write("### Prediction Results:")
    st.write(f"**Predicted RPM Percentage:** {rpm_prediction[0][0]:.2f}%")
    
    if failure_prediction_binary == 1:
        st.markdown("<h2 style='color: red;'>**Warning:** The motor is predicted to experience failure.</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: green;'>**The motor is predicted to operate normally.**</h2>", unsafe_allow_html=True)
