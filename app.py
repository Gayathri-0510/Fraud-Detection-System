import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd

# --- Custom Layer for Positional Encoding (must be defined to load model) ---
class PositionalEncoding(tf.keras.layers.Layer):
    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_length, delta=1)
        positions = tf.cast(positions, tf.float32)
        positions = tf.reshape(positions, (1, seq_length, 1))
        return inputs + positions

# --- Load Model and Components ---
@st.cache_resource
def load_model_and_components():
    try:
        # Load the Keras model with custom objects
        model = tf.keras.models.load_model(
            "fraud_detection_model.keras",
            custom_objects={'PositionalEncoding': PositionalEncoding}
        )
        # Load scalers
        amount_scaler = joblib.load("amount_scaler.pkl")
        time_scaler = joblib.load("time_scaler.pkl")
        # Load model info (SEQ_LEN, num_features, threshold)
        model_info = joblib.load("model_info.pkl")
        return model, amount_scaler, time_scaler, model_info
    except Exception as e:
        st.error(f"Error loading model or components: {e}")
        st.stop()

model, amount_scaler, time_scaler, model_info = load_model_and_components()

SEQ_LEN = model_info['seq_len']
NUM_FEATURES = model_info['num_features'] # This is 30 in your case
PREDICTION_THRESHOLD = model_info['threshold']

# --- Streamlit App Layout ---
st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("💳 Deep Learning Fraud Detection System")
st.markdown("Predict fraudulent transactions from sequential customer activity.")

st.header("Transaction Input")
st.write("Enter details for a single transaction to predict if it's fraudulent.")

# Create input fields for V1-V28, Time, and Amount
with st.form("transaction_form"):
    col1, col2 = st.columns(2)
    with col1:
        time_val = st.number_input("Time (seconds since first transaction)", value=0.0,step=100.0, format="%.2f")
        amount_val = st.number_input("Amount", min_value=1.0, value=500.0, step=100.0, format="%.2f")
    # Dynamically create V-features inputs
    st.subheader("Anonymized Features (V1-V28)")
    v_features = {}
    cols_per_row = 4 # Adjust as needed
    for i in range(1, 29):
        col_idx = (i - 1) % cols_per_row
        if col_idx == 0:
            current_cols = st.columns(cols_per_row)
        with current_cols[col_idx]:
            v_features[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.6f", key=f"V{i}")

    submitted = st.form_submit_button("Predict Fraud")

# --- Prediction Logic ---
if submitted:
    # Create a DataFrame for the single input transaction
    input_data = {"Time": time_val, "Amount": amount_val}
    input_data.update(v_features)
    input_df = pd.DataFrame([input_data])

    # Ensure columns are in the correct order (matching training data)
    # Assuming your original training features were df.drop('Class', axis=1).columns
    feature_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    input_df = input_df[feature_cols]

    # Scale Time and Amount (only these two were scaled in the notebook)
    input_df['Time'] = time_scaler.transform(input_df[['Time']])
    input_df['Amount'] = amount_scaler.transform(input_df[['Amount']])

    # Create a dummy sequence for a single prediction
    # For a real-time system, you'd need the last SEQ_LEN transactions for the customer.
    # For this demo, we'll just repeat the single transaction SEQ_LEN times.
    # This is a simplification; a production system would manage actual sequences.
    input_sequence = np.array([input_df.values.flatten() for _ in range(SEQ_LEN)])
    input_sequence = np.expand_dims(input_sequence, axis=0) # Add batch dimension

    if input_sequence.shape[2] != NUM_FEATURES:
        st.error(f"Input features mismatch! Expected {NUM_FEATURES}, got {input_sequence.shape[2]}")
    else:
        # Make prediction
        prediction = model.predict(input_sequence)[0][0]
        fraud_probability = prediction * 100

        st.subheader("Prediction Results:")
        st.metric("Fraud Probability", f"{fraud_probability:.2f}%")

        if prediction >= PREDICTION_THRESHOLD:
            st.error(f"**High Risk of Fraud!** (Probability: {fraud_probability:.2f}%) - Above threshold {PREDICTION_THRESHOLD:.2f}")
        else:
            st.success(f"**Low Risk of Fraud.** (Probability: {fraud_probability:.2f}%) - Below threshold {PREDICTION_THRESHOLD:.2f}")

        st.info("Note: In a true sequential system, the model would consider the last N transactions for a user.")
        st.info(f"The prediction threshold for 'High Risk' is set to {PREDICTION_THRESHOLD}. You can modify this in `model_info.pkl`.")
